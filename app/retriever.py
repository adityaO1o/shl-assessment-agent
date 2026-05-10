"""Semantic retriever for SHL assessment catalog using SentenceTransformers and FAISS.

Provides the `SHLRetriever` class which builds a FAISS index from
`CatalogItem.searchable_text` embeddings and performs semantic search.

Requirements:
- sentence-transformers
- faiss (faiss-cpu)
- numpy

The retriever normalizes embeddings to use cosine similarity via an
inner-product FAISS index (IndexFlatIP).
"""

from __future__ import annotations

import logging
from typing import Any, List, Tuple, Optional

from app.catalog_loader import CatalogItem, load_catalog

logger = logging.getLogger(__name__)


class SHLRetriever:
    """Semantic retriever that builds a FAISS index for fast similarity search.

    Usage:
        retriever = SHLRetriever()  # loads default model and catalog
        retriever.build_index()
        results = retriever.search("communication skills", top_k=5)

    The class stores the catalog items in `self.items` and the FAISS index in
    `self.index`. Returned scores are cosine similarities in [-1, 1].
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        catalog_path: Optional[str] = None,
        auto_initialize: bool = True,
    ) -> None:
        """Initialize the retriever.

        Args:
            model_name: name of the SentenceTransformers model to load.
            catalog_path: optional path to the catalog JSON; if None `load_catalog`
                          will use its default.
        """
        self.model_name = model_name
        self.catalog_path = catalog_path
        self.model: Optional[Any] = None
        self.faiss: Optional[Any] = None
        self.index: Optional[Any] = None
        self.emb_dim: Optional[int] = None
        self.items: List[CatalogItem] = []
        self._dependencies_attempted = False
        logger.info("Initialized SHLRetriever with model=%s", model_name)
        if auto_initialize:
            self._initialize()

    def _load_dependencies(self) -> None:
        """Load optional heavy dependencies lazily.

        Keeping imports here helps the module remain importable in constrained
        environments while still allowing a fully initialized retriever when
        the dependencies are present.
        """
        # Load the sentence-transformers model lazily. It may still work even if
        # NumPy/FAISS are unavailable; some environments only need the model for
        # embedding-based fallbacks.
        self._dependencies_attempted = True
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(self.model_name)
                logger.info("Loaded SentenceTransformer model %s", self.model_name)
            except Exception as exc:  # pragma: no cover - runtime dependency
                logger.warning("Could not load SentenceTransformer model: %s", exc)
                self.model = None

        # Attempt to import faiss; if unavailable, enable a lightweight
        # in-memory fallback search that does not require FAISS or NumPy.
        if self.faiss is None:
            try:  # faiss package name varies by installation
                import faiss  # type: ignore

                self.faiss = faiss
            except Exception as exc:  # pragma: no cover - runtime dependency
                logger.warning(
                    "FAISS unavailable; falling back to simple in-memory search: %s",
                    exc,
                )
                self.faiss = None
                # Mark that we will use a lightweight fallback search
                setattr(self, "_use_fallback_search", True)

    def _initialize(self) -> None:
        """Load catalog, create embeddings, and build the FAISS index automatically."""
        try:
            self.build_index()
        except Exception as exc:  # pragma: no cover - runtime dependency/IO guard
            logger.exception("Automatic retriever initialization failed: %s", exc)
            # Leave a partially initialized instance available; search() will
            # re-check readiness and return safely if dependencies are missing.
            self.items = self.items or []
            self.index = self.index if self.index is not None else None

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts to a normalized float32 embedding matrix.

        The returned array has shape (n_texts, dim) and rows are L2-normalized
        so inner product in FAISS yields cosine similarity.
        """
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover - platform/compiler guard
            # NumPy import failed (common on misconfigured Windows/Python); fall
            # back to a Python-list based embedding pipeline when possible.
            logger.warning("NumPy import failed; using Python-list fallback: %s", exc)
            np = None

        if not texts:
            if np is None:
                return []
            return np.zeros((0, 0), dtype=np.float32)

        if self.model is None:
            self._load_dependencies()

        # Try to get numpy-backed embeddings; if NumPy isn't available, ask the
        # model for Python-list output and normalize in pure Python.
        try:
            if np is not None:
                embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)  # type: ignore[union-attr]
                embs = np.asarray(embs, dtype=np.float32)
                # L2 normalize each vector (row)
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0
                embs = embs / norms
                return embs
            else:
                # Obtain list-of-lists output and normalize with pure Python
                raw = self.model.encode(texts, convert_to_numpy=False, show_progress_bar=False)  # type: ignore[union-attr]
                normalized = []
                for vec in raw:
                    norm = sum(x * x for x in vec) ** 0.5
                    if norm == 0:
                        norm = 1.0
                    normalized.append([float(x) / norm for x in vec])
                return normalized
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.exception("Failed to create embeddings: %s", exc)
            if np is None:
                return []
            return np.zeros((0, 0), dtype=np.float32)

    def build_index(self) -> None:
        """Load catalog, compute embeddings, and build a FAISS index.

        This method populates `self.items`, `self.index`, and `self.emb_dim`.
        It is safe to call multiple times to rebuild the index.
        """
        if not self._dependencies_attempted:
            self._load_dependencies()

        if self.catalog_path:
            items = load_catalog(self.catalog_path)
        else:
            items = load_catalog()
        self.items = items
        logger.info("Loaded %d catalog items", len(items))

        if not items:
            raise ValueError("Catalog is empty; cannot build retriever index")

        if self.model is None:
            self._use_fallback_search = True
            self._fallback_texts = [((getattr(it, "searchable_text", "") or "").lower(), idx) for idx, it in enumerate(self.items)]
            logger.info("Built fallback in-memory search over %d items", len(self._fallback_texts))
            return

        texts = [getattr(it, "searchable_text", "") or "" for it in items]
        embs = self._embed_texts(texts)

        if getattr(self, "_use_fallback_search", False) is not True and self.faiss is not None:
            try:
                n, d = embs.shape
                self.emb_dim = d
                logger.info("Created embeddings for %d catalog items", n)
                self.index = self.faiss.IndexFlatIP(d)  # type: ignore[union-attr]
                self.index.add(embs)
                logger.info("Built FAISS index with %d vectors (dim=%d)", self.index.ntotal, d)
                return
            except Exception as exc:  # pragma: no cover - runtime guard
                logger.warning("FAISS index build failed; switching to fallback search: %s", exc)
                setattr(self, "_use_fallback_search", True)

        self._use_fallback_search = True
        self._fallback_texts = [((getattr(it, "searchable_text", "") or "").lower(), idx) for idx, it in enumerate(self.items)]
        logger.info("Built fallback in-memory search over %d items", len(self._fallback_texts))

    def _ensure_ready(self) -> bool:
        """Ensure the retriever has loaded items and a FAISS index."""
        if self.items and (self.index is not None or getattr(self, "_use_fallback_search", False)):
            return True
        try:
            self.build_index()
        except Exception as exc:  # pragma: no cover - runtime dependency/IO guard
            logger.warning("Retriever is not ready: %s", exc)
            return False
        return bool(self.items and self.index is not None)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[CatalogItem, float]]:
        """Search the FAISS index for the most similar catalog items to `query`.

        Args:
            query: user query string.
            top_k: number of top results to return.

        Returns:
            List of tuples `(CatalogItem, score)` ordered by descending score.
            Score is the cosine similarity in [-1, 1]. If the index is empty or
            no items are available, an empty list is returned.
        """
        if not self._ensure_ready():
            logger.warning("Attempted search before retriever was ready")
            return []

        if getattr(self, "_use_fallback_search", False):
            q_tokens = set((query or "").lower().split())
            scored = []
            for idx, it in enumerate(self.items):
                text = (getattr(it, "searchable_text", "") or "").lower()
                t_tokens = set(text.split())
                if not t_tokens:
                    score = 0.0
                else:
                    inter = q_tokens.intersection(t_tokens)
                    union = q_tokens.union(t_tokens) or {""}
                    score = len(inter) / len(union)
                scored.append((idx, float(score)))
            scored.sort(key=lambda x: x[1], reverse=True)
            results: List[Tuple[CatalogItem, float]] = []
            for idx, score in scored[:top_k]:
                results.append((self.items[idx], score))
            return results

        q_emb = self._embed_texts([query])
        try:
            if hasattr(q_emb, "size") and q_emb.size == 0:
                return []
        except Exception:
            if not q_emb:
                return []

        scores, idxs = self.index.search(q_emb, top_k)
        scores = scores[0]
        idxs = idxs[0]

        results: List[Tuple[CatalogItem, float]] = []
        for score, idx in zip(scores, idxs):
            if idx < 0 or idx >= len(self.items):
                continue
            item = self.items[int(idx)]
            results.append((item, float(score)))

        return results


__all__ = ["SHLRetriever"]
