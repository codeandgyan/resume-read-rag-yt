"""CPU embeddings: the encode call, and one real MiniLM vector."""

from __future__ import annotations

import logging
import math
import sys
import types

import pytest

from resume_reader.embeddings.embedder import LocalEmbedder, load_embedder
from resume_reader.settings import Settings

_ENV_KEYS = (
    "GROQ_API_KEY",
    "DATABASE_URL",
    "EMBEDDING_DIM",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K",
    "MAX_UPLOAD_BYTES",
    "EMBEDDING_MODEL",
    "GROQ_MODEL",
    "CORS_ORIGIN",
)


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")


def test_embed_normalizes_on_cpu(isolated_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self, name: str, device: str) -> None:
            captured["name"] = name
            captured["device"] = device

        def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            captured["texts"] = list(texts)
            captured["kwargs"] = kwargs
            return [[1.0, *([0.0] * 383)]]

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    embedder = LocalEmbedder("all-MiniLM-L6-v2", 384)
    assert embedder.embed([]) == []
    vectors = embedder.embed(["hello resume"])

    assert captured["name"] == "all-MiniLM-L6-v2"
    assert captured["device"] == "cpu"
    assert captured["texts"] == ["hello resume"]
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["device"] == "cpu"
    assert kwargs["show_progress_bar"] is False
    assert len(vectors) == 1
    assert len(vectors[0]) == 384


def test_minilm_returns_384_dimensional_unit_vectors(
    isolated_env: None, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    embedder = load_embedder(Settings(_env_file=None))
    secret = "SECRET_SENTENCE_quartz_marker_9f3a"
    vectors = embedder.embed([secret, "A different sentence about bridges and rivers."])
    empty = embedder.embed([])

    assert empty == []
    assert len(vectors) == 2
    assert vectors[0] != vectors[1]
    for vector in vectors:
        assert len(vector) == 384
        norm = math.sqrt(sum(value * value for value in vector))
        assert math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5)
    assert secret not in caplog.text
