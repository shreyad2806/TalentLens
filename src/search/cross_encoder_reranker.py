"""Cross-encoder reranker for TalentLens search.

Provides a lightweight wrapper around sentence-transformers CrossEncoder
models (cross-encoder/ms-marco-MiniLM-L-6-v2 by default).
"""

import logging
import os

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Rerank (query, passage) pairs using a cross-encoder model."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, max_length=512)
                logger.info("Loaded reranker %s", self.model_name)
            except Exception as exc:
                logger.warning(
                    "Could not load CrossEncoder %s: %s. Reranking will fall back to 0.",
                    self.model_name,
                    exc,
                )
                self._model = None
        return self._model

    def rerank(self, query: str, texts: list[str], top_k: int | None = None) -> list[float]:
        """Return a normalized relevance score for each (query, text) pair."""
        if not texts:
            return []

        model = self._load()
        if model is None:
            return [0.0] * len(texts)

        pairs = [[query, t] for t in texts]
        try:
            scores = model.predict(pairs, show_progress_bar=False, batch_size=8)
        except Exception as exc:
            logger.warning("Reranker scoring failed: %s", exc)
            return [0.0] * len(texts)

        try:
            import numpy as np

            scores = np.asarray(scores, dtype=float)
            scores = 1.0 / (1.0 + np.exp(-scores))  # sigmoid to [0, 1]
            return [float(s) for s in scores.tolist()]
        except Exception:
            # If numpy is missing or model returns already-normalized scores, return as-is.
            return [float(s) for s in scores]
