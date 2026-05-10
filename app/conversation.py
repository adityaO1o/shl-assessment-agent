"""Conversation analysis utilities for SHL recommendation assistant.

Provides `ConversationAnalyzer` which inspects conversation history and the
latest user message to detect intent, whether clarification is needed,
comparison/refinement requests, and off-topic/legal queries. Results are
returned as Pydantic models for easy downstream consumption.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class Message(BaseModel):
    """Represents a single conversation message."""

    role: str = Field(..., description="Sender role, e.g., 'user' or 'assistant'")
    content: str = Field(..., description="Message text content")
    meta: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")

    @validator("role", "content", pre=True, always=True)
    def _strip_strings(cls, v):
        return str(v).strip() if v is not None else v


class ComparisonDetail(BaseModel):
    """Details extracted from a comparison request."""

    left: Optional[str] = None
    right: Optional[str] = None
    raw: str = ""


class RefinementDetail(BaseModel):
    """Details extracted from a refinement request."""

    action: Optional[str] = None  # e.g., 'add', 'remove', 'include', 'exclude'
    targets: List[str] = Field(default_factory=list)
    raw: str = ""


class AnalysisResult(BaseModel):
    """Structured analysis of a conversation state and latest message."""

    latest_message: Message
    intent: Optional[str] = None
    clarification_needed: bool = False
    comparison: Optional[ComparisonDetail] = None
    refinement: Optional[RefinementDetail] = None
    off_topic: bool = False
    reasons: List[str] = Field(default_factory=list)


class ConversationAnalyzer:
    """Analyze conversation history to detect user intent and needs.

    The implementation uses lightweight, deterministic heuristics suitable
    for production usage as a first-pass intent detector. For higher
    accuracy, this can be replaced with an ML-based classifier later.
    """

    # Patterns for various detections
    _vague_patterns = [
        r"\bneed an assessment\b",
        r"\brecommend (me )?something\b",
        r"\bhelp me (find|choose)\b",
        r"\bwhat should I (use|do)\b",
        r"\bi need (an )?assessment\b",
    ]

    _comparison_patterns = [
        r"\bdifference between (?P<a>.+?) and (?P<b>.+)\b",
        r"\bcompare (?P<a>.+?) (and|to|vs\.?|v\.) (?P<b>.+)\b",
        r"\b(?P<a>.+?) vs\.? (?P<b>.+)\b",
    ]

    _refinement_actions = [r"\badd\b", r"\bremove\b", r"\binclude\b", r"\bexclude\b", r"\border\b", r"\bonly\b"]
    _refinement_targets = [
        r"personality",
        r"cognitive",
        r"skills",
        r"language",
        r"job level",
        r"job_levels",
        r"test",
        r"assessment",
    ]

    _offtopic_legal_keywords = [
        r"\blegal\b",
        r"\bcompliance\b",
        r"\bgdpr\b",
        r"\bprivacy\b",
        r"\blaw\b",
        r"\bsue\b",
        r"\bcontract\b",
    ]

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.DEBUG)
        logger.debug("ConversationAnalyzer initialized; verbose=%s", verbose)

    def _normalize_messages(self, messages: Sequence[Any]) -> List[Message]:
        """Normalize input sequence of dicts or Message instances into `Message` objects."""
        out: List[Message] = []
        for m in messages:
            if isinstance(m, Message):
                out.append(m)
            elif isinstance(m, dict):
                role = m.get("role") or m.get("sender") or "user"
                content = m.get("content") or m.get("text") or ""
                out.append(Message(role=role, content=content, meta=m.get("meta")))
            else:
                # Fallback convert any object with string representation
                out.append(Message(role="user", content=str(m)))
        return out

    def analyze(self, messages: Sequence[Any]) -> AnalysisResult:
        """Analyze a conversation and return an `AnalysisResult`.

        The analyzer focuses on the latest user message and uses the prior
        history for context if needed.
        """
        msgs = self._normalize_messages(messages)
        if not msgs:
            raise ValueError("messages must be a non-empty sequence")

        # Find the latest user message; fall back to last message
        latest_user = None
        for m in reversed(msgs):
            if m.role.lower() == "user":
                latest_user = m
                break
        latest = latest_user or msgs[-1]

        logger.debug("Analyzing latest message: %s", latest.content)

        result = AnalysisResult(latest_message=latest)

        text = latest.content.lower().strip()

        # Off-topic / legal detection
        if self.is_off_topic(text):
            result.off_topic = True
            result.reasons.append("Detected legal/compliance or off-topic keywords")
            result.intent = "off_topic"
            logger.debug("Off-topic/legal detected for message: %s", latest.content)
            return result

        # Comparison detection
        comp = self._detect_comparison(text)
        if comp:
            result.comparison = comp
            result.intent = "comparison"
            result.reasons.append("Comparison request detected")

        # Refinement detection
        ref = self._detect_refinement(text)
        if ref:
            result.refinement = ref
            result.intent = result.intent or "refinement"
            result.reasons.append("Refinement request detected")

        # Clarification detection (vague queries)
        needs_clarify, clarify_reasons = self.needs_clarification(text, msgs, explain=True)
        if needs_clarify:
            result.clarification_needed = True
            result.reasons.extend(clarify_reasons or ["Vague or underspecified request; clarification needed"])
            result.intent = result.intent or "clarify"

        # If no specific flags, set a default intent
        if not result.intent:
            # Heuristic: if message is a question -> intent 'query'
            result.intent = "query" if text.endswith("?") or "what" in text.split()[:1] else "unknown"

        logger.debug("AnalysisResult: %s", result.json())
        return result

    def needs_clarification(
        self,
        latest_text: str,
        history: Optional[Sequence[Any]] = None,
        explain: bool = False,
    ) -> Tuple[bool, List[str]]:
        """Determine whether clarification is needed for the latest text.

        Returns a tuple `(needs_clarification, reasons)` when `explain=True`.
        When `explain=False` the function still returns `(bool, [])` for
        backward-compatible usage in callers that ignore reasons.

                Heuristics applied:
                - Give a strong clarification signal only for generic requests like
                    "need assessments" or "recommend tests".
                - Reduce false positives by treating role/job-title, domain, and
                    seniority keywords as strong evidence that the request is already
                    specific enough.
                - Use a simple score instead of requiring every metadata field.
        """
        text = latest_text.lower().strip()
        candidate_reasons: List[str] = []

        # Build recent context from the latest message plus a small slice of the
        # history. We intentionally keep the window small so one good clarifying
        # turn does not get overwhelmed by stale context.
        context_parts: List[str] = [text]
        if history:
            msgs = self._normalize_messages(history)
            context_parts.extend(m.content.lower() for m in msgs[-5:])
        context_text = " ".join(context_parts)

        # Keyword heuristics
        role_keywords = [
            "manager",
            "recruiter",
            "hiring manager",
            "hr",
            "candidate",
            "applicant",
            "interviewer",
            "engineer",
            "developer",
            "analyst",
            "analysts",
            "operator",
            "operators",
            "technician",
            "scientist",
            "specialist",
            "consultant",
        ]

        domain_keywords = [
            "java",
            "python",
            "backend",
            "frontend",
            "data",
            "sales",
            "marketing",
            "finance",
            "financial",
            "devops",
            "product",
            "engineering",
            "aws",
            "spring",
            "chemical",
            "backend",
        ]

        seniority_keywords = [
            "senior",
            "junior",
            "mid",
            "mid-level",
            "lead",
            "principal",
            "manager",
            "director",
            "entry",
            "associate",
            "graduate",
        ]

        category_keywords = [
            "personality",
            "cognitive",
            "skills",
            "aptitude",
            "technical",
            "behavioral",
            "competency",
            "language",
            "situational",
            "opq",
            "dsi",
        ]

        def _found_any(keywords: List[str]) -> bool:
            for k in keywords:
                if re.search(r"\b" + re.escape(k) + r"\b", context_text):
                    return True
            return False

        has_role = _found_any(role_keywords)
        has_domain = _found_any(domain_keywords)
        has_seniority = _found_any(seniority_keywords)
        has_category = _found_any(category_keywords)

        # Simple heuristic scoring:
        # - Generic assessment wording increases the likelihood of ambiguity.
        # - Real context signals (role, domain, seniority) quickly reduce it.
        # - We only ask for clarification when the score says the message is
        #   genuinely vague; this prevents false positives on specific hiring
        #   requests like "senior backend Java engineer".
        score = 0

        generic_patterns = [
            r"\bneed assessments?\b",
            r"\brecommend (something|tests?|assessments?)\b",
            r"\bsuggest (tests?|assessments?)\b",
            r"\bneed (a )?test(s)?\b",
            r"\bneed something\b",
            r"\brecommend something\b",
        ]
        if any(re.search(pat, text) for pat in generic_patterns):
            score += 3
            candidate_reasons.append("Generic assessment request detected; more context is needed.")

        # Very short messages with assessment-related nouns are often ambiguous.
        if len(text) <= 24 and re.search(r"\b(assessment|assessments|test|tests|recommend|suggest)\b", text):
            score += 1

        # Strong contextual signals reduce the ambiguity score.
        if has_role:
            score -= 2
        if has_domain:
            score -= 2
        if has_seniority:
            score -= 2
        if has_category:
            score -= 1

        # If the user is explicitly asking for hiring assessments but has
        # already specified role/seniority/domain, do not ask for clarification.
        if re.search(r"\b(hiring|hire|job|candidate|assessment|test)\b", text) and (has_role or has_domain or has_seniority):
            score -= 1

        if score >= 2:
            reasons: List[str] = list(candidate_reasons)
            if not reasons:
                reasons.append("Request is still too generic; please add role, domain, or seniority.")
            if not has_role and not has_domain and not has_seniority and not has_category:
                reasons.append(
                    "Missing context: role/job title, technical/domain keywords, seniority, or assessment category."
                )
        else:
            reasons = []

        needs = bool(reasons)
        logger.debug("needs_clarification: needs=%s reasons=%s", needs, reasons)
        if explain:
            return needs, reasons
        return needs, []

    def is_comparison_request(self, latest_text: str) -> bool:
        """Public API to check if a text is a comparison request."""
        return bool(self._detect_comparison(latest_text))

    def _detect_comparison(self, text: str) -> Optional[ComparisonDetail]:
        text = text.lower().strip()
        for pat in self._comparison_patterns:
            m = re.search(pat, text)
            if m:
                left = m.groupdict().get("a")
                right = m.groupdict().get("b")
                logger.debug("_detect_comparison: matched left=%s right=%s", left, right)
                return ComparisonDetail(left=left.strip() if left else None, right=right.strip() if right else None, raw=text)
        return None

    def is_refinement_request(self, latest_text: str) -> bool:
        """Public API to check if a text is a refinement request."""
        return bool(self._detect_refinement(latest_text))

    def _detect_refinement(self, text: str) -> Optional[RefinementDetail]:
        low = text.lower().strip()
        # Look for an action verb followed by target nouns
        action = None
        for a_pat in self._refinement_actions:
            if re.search(a_pat, low):
                action = re.search(a_pat, low).group(0)
                break

        targets: List[str] = []
        for t in self._refinement_targets:
            if re.search(r"\b" + re.escape(t) + r"\b", low):
                targets.append(t)

        if action or targets:
            logger.debug("_detect_refinement: action=%s targets=%s", action, targets)
            return RefinementDetail(action=action, targets=targets, raw=text)

        return None

    def is_off_topic(self, text: str) -> bool:
        """Detect off-topic or legal/compliance questions.

        Returns True if the message contains legal/compliance keywords, which
        should be escalated to a human or a separate handler.
        """
        low = text.lower()
        for pat in self._offtopic_legal_keywords:
            if re.search(pat, low):
                logger.debug("is_off_topic: matched pattern '%s'", pat)
                return True
        return False


__all__ = ["ConversationAnalyzer", "AnalysisResult", "Message"]
