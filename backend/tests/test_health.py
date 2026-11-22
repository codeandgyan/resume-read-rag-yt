"""Health check against Postgres, plus a database-failure response."""

from __future__ import annotations

import psycopg
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from resume_reader.main import create_app
from resume_reader.settings import Settings


class _FailingPool:
    def connection(self) -> None:
        raise ConnectionError("database unavailable")


class _UnusedEmbedder:
    def embed(self, texts: object) -> list[list[float]]:
        raise AssertionError("the health check does not embed")


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_health_ok_and_schema_has_no_approximate_index() -> None:
    settings = Settings(_env_file=None)
    app = create_app(settings)
    app.state.embedder = _UnusedEmbedder()
    async with app.router.lifespan_context(app):
        async with _client(app) as client:
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}

            cors = await client.get("/health", headers={"Origin": "http://localhost:5173"})
            assert cors.headers["access-control-allow-origin"] == "http://localhost:5173"

        with psycopg.connect(settings.database_url, autocommit=True) as conn:
            embedding = conn.execute(
                """
                SELECT format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute AS a
                JOIN pg_class AS c ON c.oid = a.attrelid
                JOIN pg_namespace AS n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public'
                  AND c.relname = 'chunks'
                  AND a.attname = 'embedding'
                  AND a.attnum > 0
                  AND NOT a.attisdropped
                """
            ).fetchone()
            assert embedding == ("vector(384)",)

            section = conn.execute(
                """
                SELECT is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'chunks'
                  AND column_name = 'section'
                """
            ).fetchone()
            assert section == ("YES",)

            indexes = conn.execute(
                """
                SELECT indexdef
                FROM pg_indexes
                WHERE schemaname = 'public' AND tablename = 'chunks'
                """
            ).fetchall()
            for (indexdef,) in indexes:
                lowered = indexdef.lower()
                assert "hnsw" not in lowered
                assert "ivfflat" not in lowered


async def test_health_is_unavailable_when_the_database_fails() -> None:
    app = create_app(Settings(_env_file=None))
    app.state.pool = _FailingPool()
    app.state.embedder = _UnusedEmbedder()
    async with app.router.lifespan_context(app):
        async with _client(app) as client:
            response = await client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}
