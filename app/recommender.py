"""Recommendation orchestration for an SHL conversational assistant.

`SHLRecommender` combines conversation analysis, semantic retrieval, and
Gemini-generated response wording while keeping recommendations grounded in
the catalog. The model must never invent assessments: only retrieved catalog
items are returned to the caller.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from app.conversation import AnalysisResult, ConversationAnalyzer, Message
from app.catalog_loader import CatalogItem
from app.catalog_loader import load_catalog
from app.ranking import AssessmentRanker
from dotenv import load_dotenv

load_dotenv()

from google import genai

if TYPE_CHECKING:
    from app.retriever import SHLRetriever

logger = logging.getLogger(__name__)


class SHLRecommender:
    """Generate grounded SHL recommendations from conversation messages.

    The recommender uses:
    - `ConversationAnalyzer` for intent detection and clarification handling.
    - `SHLRetriever` for semantic retrieval over the catalog.
    - `google-genai` SDK for response wording only, never for inventing items.

    The final response structure is:
        {
          "reply": str,
          "recommendations": list,
          "end_of_conversation": bool
        }
    """

    def __init__(
        self,
        analyzer: Optional[ConversationAnalyzer] = None,
        retriever: Optional["SHLRetriever"] = None,
        model_name: str = "gemini-3-flash-preview",
        api_key_env: str = "GEMINI_API_KEY",
    ) -> None:
        self.analyzer = analyzer or ConversationAnalyzer(verbose=False)
        self.retriever = retriever or self._create_retriever()
        self.model_name = model_name
        self.api_key_env = api_key_env
        self._gemini = None

        self._configure_gemini()
        logger.info("SHLRecommender initialized with model=%s", model_name)

    def _create_retriever(self) -> Any:
        """Create the semantic retriever lazily to avoid import-time failures."""
        try:
            from app.retriever import SHLRetriever

            return SHLRetriever()
        except Exception as exc:  # pragma: no cover - dependency/runtime guard
            logger.exception("Failed to create SHLRetriever; grounding will be unavailable: %s", exc)
            return None

    def _configure_gemini(self) -> None:
        """Configure Gemini client from the environment if available.

        Uses the new official google-genai SDK. The module remains usable
        without an API key; in that case the recommender falls back to
        deterministic response text.
        """
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            logger.warning(
                "%s is not set; Gemini wording will fall back to deterministic replies",
                self.api_key_env,
            )
            return

        try:
            # Create client from the new official SDK. The API key is sourced
            # from GEMINI_API_KEY via dotenv-loaded environment variables.
            self._gemini = genai.Client()
            logger.info("Configured Gemini client with model %s", self.model_name)
        except Exception as exc:  # pragma: no cover - dependency/runtime guard
            logger.exception("Failed to configure Gemini client; continuing without it: %s", exc)
            self._gemini = None

    def build_recommendation_prompt(
        self,
        messages: Sequence[Any],
        analysis: AnalysisResult,
        recommendations: Sequence[Dict[str, Any]],
    ) -> str:
        """Build a strict, grounding-first prompt for Gemini.

        Gemini is only allowed to rephrase the response. It must not invent
        or modify catalog items. All recommendation candidates are provided
        explicitly in the prompt.
        """
        normalized_messages = self._normalize_messages(messages)
        message_block = "\n".join(f"{m.role}: {m.content}" for m in normalized_messages)
        rec_block = json.dumps(list(recommendations), ensure_ascii=False, indent=2)

        prompt = (
            "You are an SHL assessment recommendation assistant.\n"
            "Rules:\n"
            "- Never invent assessment names, URLs, or categories.\n"
            "- Only mention the catalog items listed in the provided JSON.\n"
            "- If clarification is needed, ask one concise clarifying question.\n"
            "- If the request is off-topic or legal/compliance related, refuse briefly and do not provide legal advice.\n"
            "- If comparison is requested, compare only the provided items.\n"
            "- If refinement is requested, keep the recommendations grounded in the retrieved items.\n\n"
            f"Conversation analysis:\n{analysis.model_dump_json(indent=2)}\n\n"
            f"Conversation history:\n{message_block}\n\n"
            f"Retrieved catalog items (authoritative):\n{rec_block}\n\n"
            "Write a short helpful reply. Do not output JSON unless explicitly asked."
        )
        return prompt

    def _infer_test_type_abbrev(self, item: CatalogItem) -> str:
        """Infer SHL-style test type abbreviation from catalog item keys.

        Mapping:
        - Knowledge & Skills -> K
        - Ability & Aptitude -> A
        - Personality & Behavior -> P
        - Biodata & Situational Judgment -> B
        - Simulations -> S
        - Competencies -> C
        - Development & 360 -> D

        If multiple categories, combines with comma (e.g., "P,C").
        If no known categories found, returns "Other".
        """
        mapping = {
            "knowledge & skills": "K",
            "ability & aptitude": "A",
            "personality & behavior": "P",
            "biodata & situational judgment": "B",
            "simulations": "S",
            "competencies": "C",
            "development & 360": "D",
        }

        abbrevs: List[str] = []

        # Check the item's keys field (preserved from catalog)
        for key in (item.keys or []):
            key_lower = (key or "").lower()
            for map_key, abbrev in mapping.items():
                if map_key in key_lower or key_lower == map_key:
                    if abbrev not in abbrevs:
                        abbrevs.append(abbrev)
                    break

        # Fallback: check test_types if keys didn't yield anything
        if not abbrevs and item.test_types:
            for tt in item.test_types:
                tt_lower = (tt or "").lower()
                for map_key, abbrev in mapping.items():
                    if map_key in tt_lower or tt_lower == map_key:
                        if abbrev not in abbrevs:
                            abbrevs.append(abbrev)
                        break

        # Fallback: parse searchable_text for category hints
        if not abbrevs and item.searchable_text:
            text_lower = item.searchable_text.lower()
            for map_key, abbrev in mapping.items():
                if map_key in text_lower and abbrev not in abbrevs:
                    abbrevs.append(abbrev)
                    if len(abbrevs) >= 3:  # limit to avoid too many
                        break

        # Return sorted combined abbreviations or "Other"
        result = ",".join(sorted(set(abbrevs))) if abbrevs else "Other"
        return result

    def format_recommendations(
        self,
        items: Sequence[CatalogItem],
        scores: Sequence[float],
    ) -> List[Dict[str, Any]]:
        """Return the public recommendation payload.

        The payload intentionally includes only fields that are safe to expose:
        - name
        - url
        - test_type (inferred from catalog keys and item metadata)
        """
        formatted: List[Dict[str, Any]] = []
        for item, _score in zip(items, scores):
            test_type = self._infer_test_type_abbrev(item)
            # Only expose safe fields: name, url, test_type
            formatted.append({"name": item.name, "url": item.url, "test_type": test_type})
        return formatted

    def generate_reply(self, messages: Sequence[Any]) -> Dict[str, Any]:
        """Analyze conversation, retrieve candidates, and generate a grounded reply.

        Returns:
            {
              "reply": str,
              "recommendations": list,
              "end_of_conversation": bool
            }
        """
        normalized_messages = self._normalize_messages(messages)
        if not normalized_messages:
            raise ValueError("messages must be a non-empty sequence")

        analysis = self.analyzer.analyze(normalized_messages)
        latest_user = self._latest_user_message(normalized_messages)
        query_text = latest_user.content if latest_user else normalized_messages[-1].content

        # Off-topic or legal/compliance questions are refused.
        if analysis.off_topic:
            reply = (
                "I can help with assessment recommendations, but I can’t provide legal or compliance advice. "
                "If you want assessment suggestions, share the role, domain, seniority, and assessment type."
            )
            return {"reply": reply, "recommendations": [], "end_of_conversation": True}

        # Ask for clarification when the analyzer indicates the request is too vague.
        if analysis.clarification_needed:
            reply = self._build_clarification_reply(analysis)
            return {"reply": reply, "recommendations": [], "end_of_conversation": False}

        # Retrieve candidates from the catalog using the latest user request.
        if self.retriever is None or not hasattr(self.retriever, "search"):
            logger.warning("Retriever is unavailable; returning clarification-style fallback")
            return {
                "reply": "I’m unable to access the catalog right now. Please try again later or share more context so I can narrow down the assessment type.",
                "recommendations": [],
                "end_of_conversation": False,
            }

        try:
            retrieved = self.retriever.search(query_text, top_k=10)
        except Exception as exc:  # pragma: no cover - runtime dependency guard
            logger.exception("Retrieval failed: %s", exc)
            return {
                "reply": "I’m having trouble retrieving assessments right now. Please try again in a moment.",
                "recommendations": [],
                "end_of_conversation": False,
            }

        # Attempt deterministic reranking using the AssessmentRanker. If
        # ranking fails for any reason, fall back to the original retrieval
        # order to keep behavior safe and auditable.
        try:
            ranker = AssessmentRanker()
            # `rerank(..., explain=True)` returns (item, final_score, breakdown)
            reranked = ranker.rerank(retrieved, query_text, explain=True)
            # Convert to (item, score) pairs expected by the rest of the flow
            retrieved = [(it, score) for (it, score, _) in reranked]
        except Exception as e:  # pragma: no cover - ranking guard
            logger.exception("Ranking failed, using original retrieval order: %s", e)

        items = [item for item, _ in retrieved]
        scores = [score for _, score in retrieved]

        # Compose a realistic multi-assessment battery by combining
        # retrieved results with rule-based augmentation using the
        # authoritative catalog. This never invents items — only adds
        # candidates that exist in the catalog.
        try:
            items, scores = self._compose_battery(items, scores, query_text, analysis)
        except Exception as e:  # pragma: no cover - safety
            logger.exception("Battery composition failed, using retrieved set: %s", e)
        recommendations = self.format_recommendations(items, scores)

        # If nothing was retrieved, ask for more context instead of inventing items.
        if not recommendations:
            reply = (
                "I couldn’t find a grounded recommendation from the catalog. "
                "Please add the role, domain, seniority, and preferred assessment type."
            )
            return {"reply": reply, "recommendations": [], "end_of_conversation": False}

        prompt = self.build_recommendation_prompt(normalized_messages, analysis, recommendations)
        reply = self._generate_grounded_reply(prompt, analysis, recommendations)

        return {
            "reply": reply,
            "recommendations": recommendations,
            "end_of_conversation": False,
        }

    def _generate_grounded_reply(
        self,
        prompt: str,
        analysis: AnalysisResult,
        recommendations: Sequence[Dict[str, Any]],
    ) -> str:
        """Generate the final reply, preferring Gemini but falling back safely.

        The fallback path is deterministic and grounded in the retrieved items.
        Uses the new official google-genai SDK.
        """
        if self._gemini is None:
            logger.debug("Using deterministic fallback")
            return self._deterministic_reply(analysis, recommendations)

        try:
            response = self._gemini.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            if getattr(response, "text", None):
                logger.debug("Using Gemini response")
                return response.text.strip()
        except Exception as exc:  # pragma: no cover - API/runtime guard
            logger.warning("Gemini generation failed: %s", str(exc))

        logger.debug("Using deterministic fallback")
        return self._deterministic_reply(analysis, recommendations)

    def _deterministic_reply(
        self,
        analysis: AnalysisResult,
        recommendations: Sequence[Dict[str, Any]],
    ) -> str:
        """
        Build a concise, professional fallback reply grounded in the
        retrieved recommendations.
        """

        latest = ""
        if analysis.latest_message and analysis.latest_message.content:
            latest = analysis.latest_message.content.lower()

        # Detect hiring context
        technical = any(
            kw in latest
            for kw in [
                "java",
                "python",
                "backend",
                "frontend",
                "spring",
                "sql",
                "aws",
                "docker",
                "developer",
                "engineer",
                "microservices",
            ]
        )

        leadership = any(
            kw in latest
            for kw in [
                "leadership",
                "manager",
                "director",
                "executive",
                "cxo",
            ]
        )

        graduate = any(
            kw in latest
            for kw in [
                "graduate",
                "entry-level",
                "trainee",
                "campus",
            ]
        )

        safety = any(
            kw in latest
            for kw in [
                "safety",
                "chemical",
                "plant",
                "operator",
                "compliance",
            ]
        )

        customer_service = any(
            kw in latest
            for kw in [
                "customer service",
                "contact center",
                "call center",
                "support",
            ]
        )

        # Professional contextual replies
        if technical:
            return (
                "For a senior backend engineering role, these assessments evaluate "
                "Java and backend framework expertise, cloud deployment skills, "
                "cognitive reasoning ability, and workplace behavior fit."
            )

        if leadership:
            return (
                "These assessments evaluate leadership capability, workplace behavior, "
                "decision-making style, and executive-level competencies."
            )

        if graduate:
            return (
                "These assessments evaluate graduate-level reasoning ability, "
                "situational judgment, and workplace behavioral fit."
            )

        if safety:
            return (
                "These assessments evaluate safety awareness, dependability, "
                "workplace compliance behavior, and operational knowledge."
            )

        if customer_service:
            return (
                "These assessments evaluate customer interaction skills, "
                "service orientation, communication ability, and workplace behavior."
            )

        # Comparison response
        if analysis.comparison:
            return (
                "These assessments differ in focus area, measurement approach, "
                "and hiring use case. I can help compare them further if needed."
            )

        # Generic fallback
        return (
            "These assessments were selected because they align with the role requirements, "
            "skills, and hiring context provided."
        )

    def _build_clarification_reply(self, analysis: AnalysisResult) -> str:
        """Translate clarification reasons into a concise user-facing question."""
        if analysis.reasons:
            primary = analysis.reasons[0]
        else:
            primary = "Please share more context so I can recommend the right assessments."
        return f"I need a bit more detail before recommending assessments. {primary}"

    def _normalize_messages(self, messages: Sequence[Any]) -> List[Message]:
        """Normalize dict-like conversation messages into `Message` objects."""
        normalized: List[Message] = []
        for message in messages:
            if isinstance(message, Message):
                normalized.append(message)
            elif isinstance(message, dict):
                normalized.append(
                    Message(
                        role=message.get("role") or message.get("sender") or "user",
                        content=message.get("content") or message.get("text") or "",
                        meta=message.get("meta"),
                    )
                )
            else:
                normalized.append(Message(role="user", content=str(message)))
        return normalized

    def _latest_user_message(self, messages: Sequence[Message]) -> Optional[Message]:
        """Return the latest user-authored message when available."""
        for message in reversed(messages):
            if message.role.lower() == "user":
                return message
        return messages[-1] if messages else None

    # ---- Battery composition utilities ----
    def _detect_hiring_categories(self, query: str, analysis: AnalysisResult) -> List[str]:
        """Detect hiring intent categories from the query and analysis.

        Returns a list of categories in priority order.
        """
        q = (query or "").lower()
        tokens = set(re.split(r"\W+", q))
        cats: List[str] = []

        # Simple keyword-based detectors
        if any(t in tokens for t in {"graduate", "grad", "graduate"}):
            cats.append("graduate")
        if any(t in tokens for t in {"lead", "leadership", "manager", "management"}):
            cats.append("leadership")
        if any(t in tokens for t in {"safety", "safety-critical", "safetycritical", "safety critical", "critical"}):
            cats.append("safety")
        if any(t in tokens for t in {"technical", "developer", "engineer", "java", "python", "backend", "frontend"}):
            cats.append("technical")
        if any(t in tokens for t in {"customer", "service", "customer service", "client"}):
            cats.append("customer_service")

        # Ensure deterministic ordering and uniqueness
        seen = set()
        out: List[str] = []
        for c in cats:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def _find_items_by_name_substrings(self, substrings: Sequence[str], catalog: Sequence[CatalogItem]) -> List[CatalogItem]:
        """Return catalog items whose name or searchable_text contains any substring (case-insensitive)."""
        found: List[CatalogItem] = []
        lower_subs = [s.lower() for s in substrings if s]
        for item in catalog:
            text = " ".join(filter(None, [item.name, item.searchable_text or ""]))
            tl = text.lower()
            for sub in lower_subs:
                if sub in tl and item not in found:
                    found.append(item)
                    break
        return found

    def _compose_battery(
        self,
        items: Sequence[CatalogItem],
        scores: Sequence[float],
        query: str,
        analysis: AnalysisResult,
    ) -> Tuple[List[CatalogItem], List[float]]:
        """Compose a 1-10 item battery by augmenting retrieved items with
        rule-based candidates from the catalog. Deterministic and
        grounded: only items present in `data/shl_catalog.json` are used.
        """
        # Load the authoritative catalog once
        catalog = load_catalog()

        # Map of category -> candidate substrings to look for in catalog
        category_candidates = {
            "technical": ["verify interactive g+", "verify g+", "opq32r", "verify"],
            "graduate": ["graduate scenarios", "verify g+", "verify"],
            "leadership": ["opq32r", "opq leadership", "leadership"],
            "safety": ["dsi", "safety"],
            "customer_service": ["customer", "service"],
        }

        # Start with the retrieved (already-ranked) items, preserving order
        final: List[CatalogItem] = list(items)
        final_ids = {it.entity_id for it in final}

        # Detect categories and try to augment in priority order
        categories = self._detect_hiring_categories(query, analysis)

        # Always ensure we keep recommendations between 1 and 10 items
        MAX_ITEMS = 10
        MIN_ITEMS = 1

        # Primary augmentation pass: add category-specific candidates
        for cat in categories:
            logger.info("Detected hiring category: %s", cat)
            substrs = category_candidates.get(cat, [])
            if not substrs:
                continue
            candidates = self._find_items_by_name_substrings(substrs, catalog)
            for c in candidates:
                if c.entity_id in final_ids:
                    continue
                if len(final) < MAX_ITEMS:
                    final.append(c)
                    final_ids.add(c.entity_id)
                    logger.info("Augmented battery with %s (category=%s)", c.name, cat)
                else:
                    # battery full: replace lowest-ranked non-required item
                    logger.info("Battery full; attempting to replace a low-priority item with %s", c.name)
                    # Find index of lowest-scoring original retrieved item, prefer those with no original score
                    id_to_score = {it.entity_id: s for it, s in zip(items, scores)}
                    lowest_idx = None
                    lowest_score = float("inf")
                    for idx, it in enumerate(final):
                        sc = id_to_score.get(it.entity_id)
                        if sc is None:
                            # treat missing as low priority
                            lowest_idx = idx
                            break
                        if sc < lowest_score:
                            lowest_score = sc
                            lowest_idx = idx
                    if lowest_idx is not None:
                        replaced = final[lowest_idx]
                        if replaced.entity_id not in final_ids:
                            # should not happen, but guard
                            pass
                        final[lowest_idx] = c
                        final_ids.discard(replaced.entity_id)
                        final_ids.add(c.entity_id)
                        logger.info("Replaced %s with %s to include category candidate", replaced.name, c.name)

        # Ensure inclusion of technical required assessments even if battery was full
        # For technical hiring, force-insert Verify G+ and OPQ32r when present in catalog
        if "technical" in categories:
            # broadened search substrings to match catalog variants
            required = [
                "shl verify interactive g+",
                "verify interactive",
                "verify g+",
                "verify g",
                "occupational personality questionnaire opq32r",
                "opq32r",
                "opq",
            ]
            for req in required:
                matches = self._find_items_by_name_substrings([req], catalog)
                for m in matches:
                    if m.entity_id in final_ids:
                        continue
                    # Insert or replace low-priority item deterministically
                    if len(final) < MAX_ITEMS:
                        final.append(m)
                        final_ids.add(m.entity_id)
                        logger.info("Added required technical assessment %s", m.name)
                    else:
                        # replace lowest scoring item
                        id_to_score = {it.entity_id: s for it, s in zip(items, scores)}
                        lowest_idx = None
                        lowest_score = float("inf")
                        for idx, it in enumerate(final):
                            sc = id_to_score.get(it.entity_id)
                            if sc is None:
                                lowest_idx = idx
                                break
                            if sc < lowest_score:
                                lowest_score = sc
                                lowest_idx = idx
                        if lowest_idx is not None:
                            replaced = final[lowest_idx]
                            final[lowest_idx] = m
                            final_ids.discard(replaced.entity_id)
                            final_ids.add(m.entity_id)
                            logger.info("Replaced %s with required technical assessment %s", replaced.name, m.name)

        # If still small, try to add diverse items from the top of catalog
        # to ensure realistic batteries (e.g., mix personality/cognitive/simulation)
        if len(final) < MIN_ITEMS:
            for c in catalog:
                if c.entity_id in final_ids:
                    continue
                final.append(c)
                final_ids.add(c.entity_id)
                if len(final) >= MIN_ITEMS:
                    break

        # As a final deterministic step, ensure required technical assessments
        # are present when the query is clearly technical.
        def _match_catalog_by_name_variants(variants: Sequence[str]) -> List[CatalogItem]:
            vs = [v.lower() for v in variants if v]
            found_items: List[CatalogItem] = []
            for c in catalog:
                name_l = (c.name or "").lower()
                # prefer exact-equality with some variants, then substring
                for v in vs:
                    if name_l == v:
                        found_items.append(c)
                        break
                else:
                    for v in vs:
                        if v in name_l:
                            found_items.append(c)
                            break
            return found_items

        if "technical" in categories:
            tech_required_variants = [
                ["shl verify interactive g+", "verify interactive", "verify g+", "verify g"],
                ["occupational personality questionnaire opq32r", "opq32r", "opq"],
            ]
            required_matches: List[CatalogItem] = []
            def _select_best_match(matches: List[CatalogItem], variants: Sequence[str]) -> Optional[CatalogItem]:
                vlist = [v.lower() for v in variants if v]
                # prefer exact name matches
                for v in vlist:
                    for m in matches:
                        if (m.name or "").lower() == v:
                            return m
                # then prefer first variant that appears in the name
                for v in vlist:
                    for m in matches:
                        if v in (m.name or "").lower():
                            return m
                return matches[0] if matches else None

            for variants in tech_required_variants:
                matches = _match_catalog_by_name_variants(variants)
                if matches:
                    m = _select_best_match(matches, variants)
                    if m and m.entity_id not in final_ids and m not in required_matches:
                        required_matches.append(m)

            if required_matches:
                logger.info("Technical required matches found: %s", [m.name for m in required_matches])
                space_left = MAX_ITEMS - len(final)
                if space_left >= len(required_matches):
                    for m in required_matches:
                        final.append(m)
                        final_ids.add(m.entity_id)
                        logger.info("Added required technical assessment (final step): %s", m.name)
                else:
                    # Replace the last N items deterministically with required matches
                    n = len(required_matches)
                    for i, m in enumerate(required_matches):
                        replace_idx = len(final) - n + i
                        if replace_idx < 0:
                            replace_idx = 0
                        replaced = final[replace_idx]
                        final[replace_idx] = m
                        final_ids.discard(replaced.entity_id)
                        final_ids.add(m.entity_id)
                        logger.info("Replaced tail item %s with required assessment %s", replaced.name, m.name)

        # Truncate to MAX_ITEMS (deterministic: preserve order then trim)
        final = final[:MAX_ITEMS]

        # Align scores: prefer original retrieval scores when available,
        # otherwise assign a conservative mid-score based on position.
        id_to_score = {it.entity_id: s for it, s in zip(items, scores)}
        final_scores: List[float] = []
        for idx, it in enumerate(final):
            if it.entity_id in id_to_score:
                final_scores.append(float(id_to_score[it.entity_id]))
            else:
                # decay score based on appended position to keep determinism
                # map idx -> [0,1] via simple decay
                final_scores.append(max(0.0, 1.0 - 0.08 * float(idx)))

        return final, final_scores


__all__ = ["SHLRecommender"]
