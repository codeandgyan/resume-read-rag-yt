"""HTTP application: schema on startup, CORS, and a database health check."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from resume_reader.db import create_pool
from resume_reader.runtime import install_compatible_event_loop
from resume_reader.settings import Settings, load_settings

install_compatible_event_loop()

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings if settings is not None else load_settings()
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = resolved
        owns_pool = not hasattr(app.state, "pool")
        if owns_pool:
            app.state.pool = await create_pool(resolved.database_url)
            logger.info(
                "resume reader ready embedding_dim=%s chunk_size=%s top_k=%s",
                resolved.embedding_dim,
                resolved.chunk_size,
                resolved.top_k,
            )
        try:
            yield
        finally:
            if owns_pool:
                await app.state.pool.close()
                logger.info("database pool closed")

    app = FastAPI(title="Resume Reader", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[resolved.cors_origin],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        try:
            async with request.app.state.pool.connection() as conn:
                cursor = await conn.execute("SELECT 1")
                row = await cursor.fetchone()
            if row != (1,):
                raise RuntimeError("unexpected health check result")
        except Exception as exc:
            logger.warning("health check failed: %s", type(exc).__name__)
            body = HealthResponse(status="unavailable")
            return JSONResponse(status_code=503, content=body.model_dump())
        body = HealthResponse(status="ok")
        return JSONResponse(status_code=200, content=body.model_dump())

    return app


app = create_app()
