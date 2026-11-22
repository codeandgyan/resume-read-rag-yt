"""HTTP application: schema on startup, CORS, document upload, and health."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal
from uuid import UUID

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from resume_reader.db import create_pool
from resume_reader.embeddings.embedder import load_embedder
from resume_reader.ingest.extract import UnreadablePdf
from resume_reader.ingest.pipeline import NoExtractableText, index_pdf
from resume_reader.runtime import install_compatible_event_loop
from resume_reader.settings import Settings, load_settings
from resume_reader.uploads import UploadTooLarge, read_upload

install_compatible_event_loop()

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]


class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    page_count: int
    chunk_count: int
    status: Literal["indexed"]


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
        owns_embedder = not hasattr(app.state, "embedder")
        if owns_pool:
            app.state.pool = await create_pool(resolved.database_url)
        try:
            if owns_embedder:
                app.state.embedder = load_embedder(resolved)
            if owns_pool:
                logger.info(
                    "resume reader ready embedding_dim=%s chunk_size=%s top_k=%s",
                    resolved.embedding_dim,
                    resolved.chunk_size,
                    resolved.top_k,
                )
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

    @app.post("/documents")
    async def create_document(
        request: Request,
        file: Annotated[UploadFile, File()],
    ) -> DocumentResponse:
        settings: Settings = request.app.state.settings
        filename = _upload_filename(file)
        try:
            data = await read_upload(file, settings.max_upload_bytes)
        except UploadTooLarge:
            logger.info("rejected upload filename=%s reason=too_large", filename)
            raise HTTPException(
                status_code=413,
                detail="The file exceeds the upload limit.",
            ) from None
        try:
            indexed = await index_pdf(
                data,
                filename,
                settings,
                request.app.state.embedder,
                request.app.state.pool,
            )
        except UnreadablePdf:
            logger.info("rejected upload filename=%s reason=unreadable", filename)
            raise HTTPException(
                status_code=400,
                detail="The file is not a readable PDF.",
            ) from None
        except NoExtractableText:
            logger.info("rejected upload filename=%s reason=no_text", filename)
            raise HTTPException(
                status_code=422,
                detail="The PDF has no extractable text.",
            ) from None
        return DocumentResponse(
            id=indexed.id,
            filename=indexed.filename,
            page_count=indexed.page_count,
            chunk_count=indexed.chunk_count,
            status="indexed",
        )

    return app


def _upload_filename(file: UploadFile) -> str:
    name = (file.filename or "").replace("\x00", "").strip()
    return name or "resume.pdf"


app = create_app()
