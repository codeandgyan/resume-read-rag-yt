"""Upload a PDF, embed its chunks, and store a new document."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from uuid import UUID

import psycopg
import pymupdf
import pytest
from httpx import ASGITransport, AsyncClient, Response

from resume_reader.ingest.extract import extract_pdf
from resume_reader.main import create_app
from resume_reader.settings import Settings
from resume_reader.uploads import UPLOAD_PIECE_BYTES, UploadTooLarge, read_upload

_CORPUS = Path(__file__).resolve().parents[2] / "corpus"
_MARKER = "UNIQUE_CHUNK_TEXT_quartz_9f3a"
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


class DeterministicEmbedder:
    """One unit vector for every text, so tests never download model weights."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.threads: list[int] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        self.threads.append(threading.get_ident())
        vector = [0.0] * 384
        vector[0] = 1.0
        return [vector[:] for _ in texts]


class RecordingUpload:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.offset = 0
        self.sizes: list[int] = []
        self.closed = False

    async def read(self, size: int = -1) -> bytes:
        self.sizes.append(size)
        if self.offset >= len(self._payload):
            return b""
        if size < 0:
            size = len(self._payload) - self.offset
        end = min(len(self._payload), self.offset + size)
        piece = self._payload[self.offset : end]
        self.offset = end
        return piece

    async def close(self) -> None:
        self.closed = True


class Api:
    def __init__(
        self,
        client: AsyncClient,
        embedder: DeterministicEmbedder,
        settings: Settings,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.settings = settings
        self.created: list[str] = []

    def keep(self, response: Response) -> dict[str, object]:
        body = response.json()
        if response.status_code == 200 and isinstance(body, dict) and "id" in body:
            self.created.append(str(body["id"]))
        return body


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return Settings(_env_file=None)


@pytest.fixture
async def api(settings: Settings) -> Api:
    app = create_app(settings)
    embedder = DeterministicEmbedder()
    app.state.embedder = embedder
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=60)
    async with app.router.lifespan_context(app):
        async with client:
            handle = Api(client, embedder, settings)
            yield handle
    if not handle.created:
        return
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        for document_id in handle.created:
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (UUID(document_id),))
            conn.execute("DELETE FROM documents WHERE id = %s", (UUID(document_id),))


def _pdf_bytes(document: pymupdf.Document) -> bytes:
    data = document.tobytes()
    document.close()
    return data


def _marker_pdf() -> bytes:
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    for index, line in enumerate(
        (
            "Alex Rivera",
            "Experience",
            f"Staff engineer at Northwind {_MARKER}",
        )
    ):
        page.insert_text((72, 80 + index * 22), line, fontsize=12)
    document.new_page(width=612, height=792)
    return _pdf_bytes(document)


def _blank_pdf() -> bytes:
    document = pymupdf.open()
    document.new_page(width=612, height=792)
    return _pdf_bytes(document)


def _heading_only_pdf() -> bytes:
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    page.insert_text((72, 80), "Skills", fontsize=12)
    return _pdf_bytes(document)


def _document_count(database_url: str) -> int:
    with psycopg.connect(database_url, autocommit=True) as conn:
        row = conn.execute("SELECT count(*) FROM documents").fetchone()
        assert row is not None
        return int(row[0])


def _chunk_rows(database_url: str, document_id: str) -> list[tuple[object, ...]]:
    with psycopg.connect(database_url, autocommit=True) as conn:
        return conn.execute(
            """
            SELECT chunk_index, page, section, text, vector_dims(embedding),
                   embedding <=> %s::vector
            FROM chunks
            WHERE document_id = %s
            ORDER BY chunk_index
            """,
            ("[" + ",".join(["1", *["0"] * 383]) + "]", UUID(document_id)),
        ).fetchall()


async def test_read_upload_uses_64kib_pieces_and_stops_past_the_limit() -> None:
    assert UPLOAD_PIECE_BYTES == 64 * 1024
    exact = RecordingUpload(b"abc")
    assert await read_upload(exact, 3) == b"abc"
    assert exact.closed
    assert exact.sizes == [64 * 1024, 64 * 1024]

    payload = b"x" * (11 * 1024 * 1024)
    upload = RecordingUpload(payload)
    with pytest.raises(UploadTooLarge):
        await read_upload(upload, 10 * 1024 * 1024)
    assert upload.closed
    assert upload.sizes == [64 * 1024] * (10 * 1024 * 1024 // (64 * 1024) + 1)
    assert upload.offset == 10 * 1024 * 1024 + 64 * 1024
    assert upload.offset < len(payload)


async def test_lifespan_loads_one_embedder_unless_tests_replace_it(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: list[str] = []

    def fake_load(resolved: Settings) -> DeterministicEmbedder:
        loaded.append(resolved.embedding_model)
        return DeterministicEmbedder()

    monkeypatch.setattr("resume_reader.main.load_embedder", fake_load)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        assert loaded == ["all-MiniLM-L6-v2"]
        assert isinstance(app.state.embedder, DeterministicEmbedder)

    loaded.clear()
    replaced = DeterministicEmbedder()
    skipped = create_app(settings)
    skipped.state.embedder = replaced
    async with skipped.router.lifespan_context(skipped):
        assert loaded == []
        assert skipped.state.embedder is replaced


async def test_upload_indexes_a_new_document_and_omits_resume_text_from_logs(
    api: Api, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    threads: list[int] = []
    real_extract = extract_pdf

    def spy(data: bytes):
        threads.append(threading.get_ident())
        return real_extract(data)

    monkeypatch.setattr("resume_reader.ingest.pipeline.extract_pdf", spy)
    caplog.set_level(logging.INFO)
    pdf = _marker_pdf()
    before = _document_count(api.settings.database_url)

    response = await api.client.post(
        "/documents",
        files={"file": ("resume-unique.pdf", pdf, "application/octet-stream")},
    )
    body = api.keep(response)

    assert response.status_code == 200
    assert body["status"] == "indexed"
    assert body["filename"] == "resume-unique.pdf"
    assert body["page_count"] == 2
    assert body["chunk_count"] == 2
    assert _document_count(api.settings.database_url) == before + 1
    assert threads
    assert threads[0] != threading.get_ident()
    assert api.embedder.threads
    assert api.embedder.threads[0] != threading.get_ident()
    assert api.embedder.calls
    assert _MARKER in api.embedder.calls[0][1]
    assert _MARKER not in caplog.text
    assert "Alex Rivera" not in caplog.text

    document_id = str(body["id"])
    indexed = [record.getMessage() for record in caplog.records if "indexed document" in record.getMessage()]
    assert len(indexed) == 1
    assert document_id in indexed[0]
    assert "filename=resume-unique.pdf" in indexed[0]
    assert "page_count=2" in indexed[0]
    assert "chunk_count=2" in indexed[0]
    assert "latency_ms=" in indexed[0]
    for record in caplog.records:
        assert _MARKER not in record.getMessage()
        assert _MARKER not in str(record.args)

    rows = _chunk_rows(api.settings.database_url, document_id)
    assert [(row[0], row[1], row[2], row[3]) for row in rows] == [
        (0, 1, None, "Alex Rivera"),
        (1, 1, "Experience", f"Staff engineer at Northwind {_MARKER}"),
    ]
    for row in rows:
        assert row[4] == 384
        assert row[5] == pytest.approx(0.0, abs=1e-5)


async def test_a_second_upload_of_the_same_file_gets_a_new_id(api: Api) -> None:
    pdf = (_CORPUS / "04_sneha_iyer_compact_one_page.pdf").read_bytes()
    first = await api.client.post("/documents", files={"file": ("04_sneha_iyer_compact_one_page.pdf", pdf, "application/pdf")})
    second = await api.client.post("/documents", files={"file": ("04_sneha_iyer_compact_one_page.pdf", pdf, "application/pdf")})
    first_body = api.keep(first)
    second_body = api.keep(second)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first_body["status"] == "indexed"
    assert second_body["status"] == "indexed"
    assert int(first_body["chunk_count"]) > 0
    assert first_body["chunk_count"] == second_body["chunk_count"]
    assert first_body["id"] != second_body["id"]
    for document_id in (str(first_body["id"]), str(second_body["id"])):
        rows = _chunk_rows(api.settings.database_url, document_id)
        assert len(rows) == first_body["chunk_count"]


@pytest.mark.parametrize(
    ("filename", "payload"),
    [
        ("notes.txt", b"NOT_A_PDF_MARKER_z9"),
        ("truncated.pdf", b"%PDF-1.4\n%truncated"),
        ("empty.pdf", b""),
        ("broken.pdf", b"%PDF"),
    ],
)
async def test_unreadable_uploads_are_400(
    api: Api, caplog: pytest.LogCaptureFixture, filename: str, payload: bytes
) -> None:
    caplog.set_level(logging.INFO)
    before = _document_count(api.settings.database_url)
    response = await api.client.post(
        "/documents",
        files={"file": (filename, payload, "application/pdf")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "The file is not a readable PDF."
    assert api.embedder.calls == []
    assert _document_count(api.settings.database_url) == before
    assert b"NOT_A_PDF_MARKER_z9".decode() not in caplog.text


@pytest.mark.parametrize("payload", [_blank_pdf(), _heading_only_pdf()])
async def test_pdf_without_body_text_is_422(api: Api, payload: bytes) -> None:
    before = _document_count(api.settings.database_url)
    response = await api.client.post(
        "/documents",
        files={"file": ("blank.pdf", payload, "application/pdf")},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "The PDF has no extractable text."
    assert api.embedder.calls == []
    assert _document_count(api.settings.database_url) == before


async def test_eleven_megabyte_upload_is_rejected(api: Api) -> None:
    before = _document_count(api.settings.database_url)
    payload = b"%PDF-" + (b"0" * (11 * 1024 * 1024))
    response = await api.client.post(
        "/documents",
        files={"file": ("big.pdf", payload, "application/pdf")},
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "The file exceeds the upload limit."
    assert api.embedder.calls == []
    assert _document_count(api.settings.database_url) == before

