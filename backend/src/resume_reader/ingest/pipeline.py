"""Extract, chunk, embed, and store one uploaded PDF."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from uuid import UUID

from psycopg_pool import AsyncConnectionPool

from resume_reader.embeddings.embedder import Embedder
from resume_reader.ingest.chunk import chunk_pages
from resume_reader.ingest.extract import UnreadablePdf, extract_pdf
from resume_reader.settings import Settings
from resume_reader.storage.documents import insert_document

logger = logging.getLogger(__name__)


class NoExtractableText(Exception):
    """The PDF opened, but it produced no body text to index."""


@dataclass(frozen=True)
class IndexedDocument:
    """A resume that was stored under a new id."""

    id: UUID
    filename: str
    page_count: int
    chunk_count: int


async def index_pdf(
    data: bytes,
    filename: str,
    settings: Settings,
    embedder: Embedder,
    pool: AsyncConnectionPool,
) -> IndexedDocument:
    """Index ``data`` as a new document.

    PDF parsing and embedding run off the event loop. The log line records
    the document id, filename, page count, chunk count, and latency. It does
    not include the resume text.
    """
    started = time.perf_counter()
    extracted = await asyncio.to_thread(extract_pdf, data)
    if not extracted.pages:
        raise NoExtractableText("PDF has no extractable text")

    chunks = await asyncio.to_thread(chunk_pages, extracted.pages, settings)
    if not chunks:
        raise NoExtractableText("PDF has no extractable text")

    texts = [chunk.text for chunk in chunks]
    vectors = await asyncio.to_thread(embedder.embed, texts)
    if len(vectors) != len(chunks):
        raise RuntimeError("embedding count does not match chunk count")
    for vector in vectors:
        if len(vector) != settings.embedding_dim:
            raise RuntimeError("embedding dimension does not match settings")

    document_id = await insert_document(
        pool,
        filename=filename,
        page_count=extracted.page_count,
        chunks=chunks,
        embeddings=vectors,
    )
    logger.info(
        "indexed document id=%s filename=%s page_count=%s chunk_count=%s latency_ms=%.1f",
        document_id,
        filename,
        extracted.page_count,
        len(chunks),
        (time.perf_counter() - started) * 1000,
    )
    return IndexedDocument(
        id=document_id,
        filename=filename,
        page_count=extracted.page_count,
        chunk_count=len(chunks),
    )
