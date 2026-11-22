"""Insert one new document and its chunks. Uploads are never deduplicated."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID, uuid4

from pgvector import Vector
from psycopg_pool import AsyncConnectionPool

from resume_reader.ingest.chunk import Chunk


async def insert_document(
    pool: AsyncConnectionPool,
    *,
    filename: str,
    page_count: int,
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
) -> UUID:
    """Insert a new document id and one row per chunk.

    A second call with the same filename and text creates another document.
    There is no upsert and no lookup of an existing resume.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("each chunk needs one embedding")
    if not chunks:
        raise ValueError("a document needs at least one chunk")

    document_id = uuid4()
    async with pool.connection() as conn:
        async with conn.transaction():
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO documents (id, filename, page_count)
                    VALUES (%s, %s, %s)
                    """,
                    (document_id, filename, page_count),
                )
                for chunk, embedding in zip(chunks, embeddings, strict=True):
                    await cursor.execute(
                        """
                        INSERT INTO chunks (
                            id, document_id, chunk_index, page, section, text, embedding
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            uuid4(),
                            document_id,
                            chunk.index,
                            chunk.page,
                            chunk.section,
                            chunk.text,
                            Vector([float(value) for value in embedding]),
                        ),
                    )
    return document_id
