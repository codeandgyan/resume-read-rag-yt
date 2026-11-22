"""Async Postgres pool and startup schema."""

from __future__ import annotations

import logging
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def split_statements(script: str) -> list[str]:
    """Split a SQL script into statements. The schema has no semicolons inside literals."""
    statements: list[str] = []
    for raw in script.split(";"):
        statement = raw.strip()
        if statement:
            statements.append(statement)
    return statements


async def apply_schema(conn: psycopg.AsyncConnection) -> None:
    script = SCHEMA_PATH.read_text(encoding="utf-8")
    for statement in split_statements(script):
        await conn.execute(statement)
    logger.info("schema applied")


async def _register_vector(conn: psycopg.AsyncConnection) -> None:
    await register_vector_async(conn)


async def create_pool(database_url: str) -> AsyncConnectionPool:
    """Apply the schema, then open a pool whose connections understand vector columns."""
    async with await psycopg.AsyncConnection.connect(
        database_url,
        autocommit=True,
        connect_timeout=3,
    ) as conn:
        await apply_schema(conn)

    pool = AsyncConnectionPool(
        conninfo=database_url,
        open=False,
        min_size=1,
        max_size=10,
        timeout=3,
        reconnect_timeout=30,
        kwargs={"autocommit": True, "connect_timeout": 3},
        configure=_register_vector,
        check=AsyncConnectionPool.check_connection,
    )
    await pool.open()
    logger.info("database pool opened")
    return pool
