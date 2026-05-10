"""Assessment reranking utility for SHL recommendations.

Provides a deterministic, explainable `AssessmentRanker` that improves
precision by applying heuristic boosts for technical keywords, roles,
seniority, and assessment categories.

The ranker is intentionally simple and auditable: scores are a weighted
sum of embedding similarity and several interpretable signals.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from app.catalog_loader import CatalogItem

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Simple deterministic tokenizer: lower-case, split on non-word chars."""
    if not text:
        return []
    return [t for t in re.split(r"\W+", text.lower()) if t]


@dataclass
class RankExplanation:
    base_similarity: float
    keyword_score: float
    category_score: float
    seniority_score: float
    penalty: float
    final_score: float


class AssessmentRanker:
    """Deterministic reranker for retrieved assessments.

    Usage:
        ranker = AssessmentRanker()
        reranked = ranker.rerank(retrieved, query)

    `retrieved` should be a sequence of `(CatalogItem, similarity_score)`
    returned by the semantic retriever. `similarity_score` is expected to be
    in the range [-1, 1] (cosine-like). The ranker returns the items sorted
    by descending `final_score` along with an optional explanation map.
    """

    # Keywords to boost (lowercase)
    BOOST_KEYWORDS = {
        "java",
        "spring",
        "sql",
        "aws",
        "docker",
        "leadership",
        "graduate",
        "safety",
        "personality",
        "cognitive",
        "simulations",
    }

    # Technical stack keywords (strong boost for technical queries)
    TECHNICAL_STACK_KEYWORDS = {
        "java",
        "spring",
        "sql",
        "rest",
        "aws",
        "docker",
        "backend",
        "microservices",
        "python",
        "dotnet",
        ".net",
        "javascript",
        "typescript",
        "react",
        "angular",
        "kubernetes",
        "ci/cd",
        "devops",
    }

    # Irrelevant keywords: penalize when query is technical but item matches these
    IRRELEVANT_KEYWORDS = {
        "interviewing",
        "interview",
        "hiring",
        "recruitment",
        "cms",
        "content management",
        "wordpress",
        "sharepoint",
        "unrelated",
    }

    # Seniority indicators
    SENIORITY_KEYWORDS = {
        "senior": 1.0,
        "lead": 0.9,
        "principal": 1.0,
        "manager": 0.8,
        "junior": -0.5,
        "entry": -0.5,
        "graduate": 0.7,
        "intern": -0.6,
    }

    # Category keywords mapping (simple heuristics)
    CATEGORY_KEYWORDS = {
        "leadership": {"lead", "manager", "leadership"},
        "safety": {"safety", "compliance", "critical"},
        "personality": {"personality", "behavioral"},
        "cognitive": {"cognitive", "ability", "aptitude"},
        "simulations": {"simulation", "scenario", "exercise"},
        "technical": {"technical", "skill", "coding", "programming"},
    }

    def __init__(
        self,
        weight_similarity: float = 0.6,
        weight_keyword: float = 0.2,
        weight_category: float = 0.15,
        weight_seniority: float = 0.05,
        min_keyword_overlap_for_boost: float = 0.05,
        penalty_for_weak_match: float = 0.15,
    ) -> None:
        """Configure weights and thresholds.

        All weights should sum to 1.0 for normalized interpretability,
        but the class will normalize them if they do not.
        """
        self.weight_similarity = weight_similarity
        self.weight_keyword = weight_keyword
        self.weight_category = weight_category
        self.weight_seniority = weight_seniority

        # Normalize weights
        s = weight_similarity + weight_keyword + weight_category + weight_seniority
        if s <= 0:
            raise ValueError("Sum of weights must be positive")
        self.weight_similarity /= s
        self.weight_keyword /= s
        self.weight_category /= s
        self.weight_seniority /= s

        self.min_keyword_overlap_for_boost = float(min_keyword_overlap_for_boost)
        self.penalty_for_weak_match = float(penalty_for_weak_match)

        logger.info(
            "AssessmentRanker initialized (weights: sim=%.2f, kw=%.2f, cat=%.2f, sen=%.2f)",
            self.weight_similarity,
            self.weight_keyword,
            self.weight_category,
            self.weight_seniority,
        )

    def _detect_query_domain(self, query_tokens: List[str]) -> str:
        """Detect the primary domain/intent from query tokens.

        Returns: 'technical', 'leadership', 'graduate', 'safety', or 'general'.
        """
        query_set = set(query_tokens)
        if any(t in query_set for t in {"technical", "engineer", "developer", "backend", "frontend", "java", "python", "aws", "docker"}):
            return "technical"
        if any(t in query_set for t in {"leadership", "lead", "manager", "management"}):
            return "leadership"
        if any(t in query_set for t in {"graduate", "grad", "entry", "entry-level"}):
            return "graduate"
        if any(t in query_set for t in {"safety", "critical", "safetycritical"}):
            return "safety"
        return "general"

    def _technical_stack_score(self, query_tokens: List[str], item: CatalogItem) -> float:
        """Compute technical stack alignment bonus (0 to 1).

        High score if query is technical and item mentions relevant tech keywords.
        """
        query_set = set(query_tokens)
        item_text = " ".join(
            filter(None, [getattr(item, "name", ""), getattr(item, "description", "")])
        ).lower()
        item_tokens = set(_tokenize(item_text))

        # Count matches between query/item tech keywords
        matches = 0
        for tk in self.TECHNICAL_STACK_KEYWORDS:
            if tk in query_set and tk in item_tokens:
                matches += 1

        # Bonus: strong boost for tech keyword alignment
        if matches > 0:
            return min(1.0, matches * 0.3)
        return 0.0

    def _irrelevant_penalty(self, query_domain: str, item: CatalogItem) -> float:
        """Compute penalty for irrelevant item in specific domain (0 to 1).

        High penalty if query is technical but item is unrelated/generic.
        """
        if query_domain != "technical":
            return 0.0  # only penalize in technical domain

        item_text = " ".join(
            filter(None, [getattr(item, "name", ""), getattr(item, "description", "")])
        ).lower()

        # Check for irrelevant keywords
        for irr in self.IRRELEVANT_KEYWORDS:
            if irr in item_text:
                logger.info("Item %s penalized: contains irrelevant keyword '%s'", item.name, irr)
                return 0.4  # strong penalty

        # Generic/unrelated items also penalized
        # If item name is very generic and doesn't match technical stack, penalize
        item_name_lower = (item.name or "").lower()
        generic_terms = {"report", "solution", "module", "tool", "system"}
        if any(gt in item_name_lower for gt in generic_terms):
            # Check if it has any technical relevance
            tech_match = any(tk in item_name_lower for tk in self.TECHNICAL_STACK_KEYWORDS)
            if not tech_match:
                logger.info("Item %s penalized: generic product without technical relevance", item.name)
                return 0.2  # moderate penalty for generic items

        return 0.0

    def _keyword_overlap_score(self, query_tokens: List[str], item: CatalogItem) -> float:
        """Compute a normalized keyword overlap score between query and item.

        Returns a value in [0, 1] representing fraction of boost keywords
        found in the item and query intersection, weighted by presence in
        the canonical BOOST_KEYWORDS set.
        """
        item_text = " ".join(
            filter(None, [getattr(item, "name", ""), getattr(item, "description", "")])
        ).lower()
        item_tokens = set(_tokenize(item_text))
        query_set = set(query_tokens)

        # Count how many boost keywords appear in both query and item
        matches = 0
        for kw in self.BOOST_KEYWORDS:
            if kw in query_set and kw in item_tokens:
                matches += 1

        # Fractional score normalized by number of boost keywords present in the query
        query_boosts = sum(1 for kw in self.BOOST_KEYWORDS if kw in query_set)
        if query_boosts == 0:
            return 0.0
        return matches / float(query_boosts)

    def _category_score(self, query_tokens: List[str], item: CatalogItem) -> float:
        """Assess category relevance by matching query tokens to category keywords
        and checking item test types / description for category signals.
        """
        item_text = " ".join(
            filter(None, [getattr(item, "name", ""), getattr(item, "description", ""), " ".join(getattr(item, "test_types", []) or [])])
        ).lower()
        item_tokens = set(_tokenize(item_text))
        score = 0.0
        for cat, keys in self.CATEGORY_KEYWORDS.items():
            if any(k in query_tokens for k in keys):
                # boost if item signals the category
                if any(k in item_tokens for k in keys):
                    score = max(score, 1.0)
                else:
                    # partial credit if item mentions related terms
                    overlap = len(item_tokens.intersection(keys)) / max(1, len(keys))
                    score = max(score, 0.5 * overlap)
        return float(score)

    def _seniority_score(self, query_tokens: List[str], item: CatalogItem) -> float:
        """Score seniority alignment between query and item.

        Positive values indicate better seniority match (e.g., senior roles),
        negative values penalize mismatches (e.g., senior query vs junior assessment).
        """
        score = 0.0
        # Query seniority hints
        for token in query_tokens:
            if token in self.SENIORITY_KEYWORDS:
                score += float(self.SENIORITY_KEYWORDS[token])

        # Normalize by number of indicators to keep scale reasonable
        if score == 0.0:
            return 0.0

        # If the item exposes job_levels, try to match
        item_levels = [s.lower() for s in (getattr(item, "job_levels", []) or [])]
        if item_levels:
            # simple heuristic: if any item level token appears in the query, boost
            for lvl in item_levels:
                lvl_tokens = _tokenize(lvl)
                if any(t in query_tokens for t in lvl_tokens):
                    # amplify match
                    score *= 1.0
                    break
            else:
                # if no match, gently reduce the score
                score *= 0.7

        # Clamp to a reasonable range
        return float(max(min(score, 1.0), -1.0))

    def rerank(
        self,
        retrieved: Sequence[Tuple[CatalogItem, float]],
        query: str,
        explain: bool = False,
    ) -> Sequence[Tuple[CatalogItem, float, Dict[str, float]]]:
        """Rerank retrieved items.

        Args:
            retrieved: sequence of (CatalogItem, similarity_score) where
                similarity_score is expected in [-1, 1].
            query: user query string.
            explain: if True, return per-item score breakdowns.

        Returns:
            If explain=False: list of (CatalogItem, final_score) sorted desc.
            If explain=True: list of (CatalogItem, final_score, breakdown_dict).
        """
        query_tokens = _tokenize(query)

        reranked: List[Tuple[CatalogItem, float, Dict[str, float]]] = []

        for item, sim in retrieved:
            base_sim = float(sim)
            # Normalize similarity from [-1,1] to [0,1]
            sim_norm = (base_sim + 1.0) / 2.0

            kw_score = self._keyword_overlap_score(query_tokens, item)
            cat_score = self._category_score(query_tokens, item)
            sen_score = self._seniority_score(query_tokens, item)
            query_domain = self._detect_query_domain(query_tokens)
            tech_boost = self._technical_stack_score(query_tokens, item)
            irr_penalty = self._irrelevant_penalty(query_domain, item)


            # Log domain detection once (avoid repetition)
            if item == retrieved[0][0]:
                logger.info("Detected query domain: %s", query_domain)

            # Log boosts
            if tech_boost > 0.1:
                logger.info("Item %s boosted for technical stack match (+%.2f)", item.name, tech_boost)

            # Penalty for weak or overly-general matches
            penalty = irr_penalty
            if kw_score < self.min_keyword_overlap_for_boost and cat_score < 0.1:
                penalty = max(penalty, self.penalty_for_weak_match)

            # Weighted sum with tech boost
            final = (
                self.weight_similarity * sim_norm
                + self.weight_keyword * kw_score
                + self.weight_category * cat_score
                + self.weight_seniority * max(0.0, sen_score)
                + tech_boost  # additional boost for technical relevance
            )

            # Apply penalty multiplicatively to keep deterministic scale
            final = float(max(0.0, final * (1.0 - penalty)))

            breakdown = {
                "base_similarity": sim_norm,
                "keyword_score": kw_score,
                "category_score": cat_score,
                "seniority_score": sen_score,
                "tech_boost": tech_boost,
                "penalty": penalty,
                "final_score": final,
            }

            logger.debug(
                "Ranked item=%s sim=%.4f kw=%.3f cat=%.3f sen=%.3f tech=%.3f pen=%.2f final=%.4f",
                getattr(item, "name", "<unknown>"),
                base_sim,
                kw_score,
                cat_score,
                sen_score,
                tech_boost,
                penalty,
                final,
            )

            reranked.append((item, final, breakdown))

        # Sort by final score descending, stable sort to keep determinism
        reranked.sort(key=lambda x: (-x[1], getattr(x[0], "name", "")))

        if explain:
            return reranked

        # Return without breakdown
        return [(it, score, {}) for (it, score, _) in reranked]


__all__ = ["AssessmentRanker", "RankExplanation"]
