"""Settings validation. These tests do not open a database connection."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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


def test_defaults_do_not_need_a_database(isolated_env: None) -> None:
    settings = Settings(_env_file=None)
    assert settings.groq_api_key == ""
    assert settings.database_url == "postgresql://postgres:postgres@localhost:5432/resume_reader"
    assert settings.embedding_dim == 384
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 120
    assert settings.top_k == 5
    assert settings.max_upload_bytes == 10 * 1024 * 1024
    assert settings.embedding_model == "all-MiniLM-L6-v2"
    assert settings.groq_model == "llama-3.3-70b-versatile"
    assert settings.cors_origin == "http://localhost:5173"


def test_reads_limits_from_the_environment(
    isolated_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "40")
    monkeypatch.setenv("TOP_K", "3")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "2048")
    settings = Settings(_env_file=None)
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 40
    assert settings.top_k == 3
    assert settings.max_upload_bytes == 2048


def test_allows_boundary_values(isolated_env: None) -> None:
    settings = Settings(
        _env_file=None,
        chunk_size=1,
        chunk_overlap=0,
        top_k=1,
        max_upload_bytes=1,
        embedding_dim=384,
    )
    assert settings.chunk_size == 1
    assert settings.chunk_overlap == 0
    assert settings.top_k == 1
    assert settings.max_upload_bytes == 1


@pytest.mark.parametrize(
    ("overrides", "field_name"),
    [
        ({"embedding_dim": 0}, "embedding_dim"),
        ({"embedding_dim": 385}, "embedding_dim"),
        ({"embedding_dim": 768}, "embedding_dim"),
        ({"chunk_size": 0}, "chunk_size"),
        ({"chunk_size": -1}, "chunk_size"),
        ({"max_upload_bytes": 0}, "max_upload_bytes"),
        ({"max_upload_bytes": -5}, "max_upload_bytes"),
        ({"chunk_overlap": -1}, "chunk_overlap"),
        ({"chunk_overlap": 800}, "chunk_overlap"),
        ({"chunk_overlap": 801}, "chunk_overlap"),
        ({"top_k": 0}, "top_k"),
        ({"top_k": -2}, "top_k"),
    ],
)
def test_rejects_impossible_values(
    isolated_env: None, overrides: dict[str, int], field_name: str
) -> None:
    with pytest.raises(ValidationError, match=field_name):
        Settings(_env_file=None, **overrides)


def test_rejects_overlap_that_is_not_smaller_than_a_custom_chunk_size(isolated_env: None) -> None:
    with pytest.raises(ValidationError, match="chunk_overlap"):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)
