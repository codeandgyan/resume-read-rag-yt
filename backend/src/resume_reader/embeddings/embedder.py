"""Load one CPU embedding model and encode text to unit-length vectors."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any, Protocol

from resume_reader.settings import Settings

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Turns a batch of texts into one vector each, in the same order."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text."""


class LocalEmbedder:
    """``all-MiniLM-L6-v2`` loaded once, encoding on CPU.

    ``encode`` is called with ``normalize_embeddings=True``, so each vector
    has length 384 and unit L2 norm.
    """

    def __init__(self, model_name: str, dimension: int) -> None:
        from sentence_transformers import SentenceTransformer

        self._dimension = dimension
        self._model_name = model_name
        # Model download chatter is not part of the ingestion log.
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logger.info("loading embedding model=%s device=cpu", model_name)
        self._model = SentenceTransformer(model_name, device="cpu")
        logger.info("embedding model ready model=%s dim=%s", model_name, dimension)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode ``texts`` on CPU. An empty batch does not call the model."""
        if not texts:
            return []
        started = time.perf_counter()
        encoded = self._model.encode(
            list(texts),
            device="cpu",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = _as_rows(encoded)
        if len(vectors) != len(texts):
            raise ValueError("embedding count does not match the input count")
        for vector in vectors:
            if len(vector) != self._dimension:
                raise ValueError(f"embedding dimension must be {self._dimension}")
        logger.info(
            "embedded count=%s dim=%s model=%s latency_ms=%.1f",
            len(vectors),
            self._dimension,
            self._model_name,
            (time.perf_counter() - started) * 1000,
        )
        return vectors


def load_embedder(settings: Settings) -> LocalEmbedder:
    """Load the configured model once. Callers keep the returned instance."""
    return LocalEmbedder(settings.embedding_model, settings.embedding_dim)


def _as_rows(encoded: Any) -> list[list[float]]:
    raw = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
    if not raw:
        return []
    if isinstance(raw[0], (int, float)):
        raw = [raw]
    return [[float(value) for value in row] for row in raw]
