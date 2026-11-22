"""Environment-backed settings. Impossible values are rejected at construction."""

from __future__ import annotations

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[3]
_POSITIVE_LIMITS = ("chunk_size", "max_upload_bytes")


def _env_files() -> tuple[str, ...]:
    candidates = (_REPO_ROOT / ".env", Path.cwd() / ".env")
    seen: set[Path] = set()
    files: list[str] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        files.append(str(resolved))
    return tuple(files)


class Settings(BaseSettings):
    """Runtime configuration loaded from the environment and an optional .env file."""

    model_config = SettingsConfigDict(
        env_file=_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = ""
    database_url: str = "postgresql://postgres:postgres@localhost:5432/resume_reader"
    embedding_dim: int = 384
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    max_upload_bytes: int = 10 * 1024 * 1024
    embedding_model: str = "all-MiniLM-L6-v2"
    groq_model: str = "llama-3.3-70b-versatile"
    cors_origin: str = "http://localhost:5173"

    @model_validator(mode="after")
    def reject_impossible_values(self) -> Settings:
        problems: list[str] = []
        if self.embedding_dim != 384:
            problems.append("embedding_dim must be 384")
        for name in _POSITIVE_LIMITS:
            if getattr(self, name) < 1:
                problems.append(f"{name} must be at least 1")
        if self.top_k < 1:
            problems.append("top_k must be at least 1")
        if self.chunk_overlap < 0:
            problems.append("chunk_overlap must be at least 0")
        elif self.chunk_overlap >= self.chunk_size:
            problems.append("chunk_overlap must be smaller than chunk_size")
        if problems:
            raise ValueError("; ".join(problems))
        return self


def load_settings() -> Settings:
    """Load settings, failing fast when a value cannot be used."""
    return Settings()
