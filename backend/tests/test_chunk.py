"""Section headings and page-bounded chunks."""

from __future__ import annotations

import logging

import pytest

from resume_reader.ingest.chunk import Chunk, chunk_pages
from resume_reader.ingest.extract import ExtractedPage
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
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return Settings(_env_file=None)


def _page(number: int, *lines: str) -> ExtractedPage:
    return ExtractedPage(number=number, lines=lines)


def test_prefixed_heading_with_a_colon_is_stored_without_the_colon(settings: Settings) -> None:
    chunks = chunk_pages(
        (
            _page(
                1,
                "Professional Experience:",
                "Built payment APIs.",
                "KEY SKILLS:",
                "Python",
            ),
        ),
        settings,
    )

    assert chunks == (
        Chunk(0, 1, "Professional Experience", "Built payment APIs."),
        Chunk(1, 1, "KEY SKILLS", "Python"),
    )


@pytest.mark.parametrize(
    ("line", "stored"),
    [
        ("summary", "summary"),
        ("Profile", "Profile"),
        ("OBJECTIVE", "OBJECTIVE"),
        ("employment", "employment"),
        ("Education:", "Education"),
        ("strengths :", "strengths"),
        ("projects", "projects"),
        ("Certifications:", "Certifications"),
        ("contributions", "contributions"),
        ("Core Skills:", "Core Skills"),
        ("Technical skills", "Technical skills"),
        ("Selected Projects:", "Selected Projects"),
        ("Relevant experience:", "Relevant experience"),
        ("key strengths", "key strengths"),
        ("professional summary:", "professional summary"),
    ],
)
def test_heading_vocabulary(settings: Settings, line: str, stored: str) -> None:
    chunks = chunk_pages((_page(1, line, "Kept body."),), settings)
    assert chunks == (Chunk(0, 1, stored, "Kept body."),)


@pytest.mark.parametrize(
    "line",
    [
        "Experience with Python",
        "Professional",
        "Skills and tools",
        "My education",
        "Selected",
        "- Skills",
        "Skills: Python, Go",
    ],
)
def test_a_line_that_only_contains_a_heading_word_is_not_a_heading(
    settings: Settings, line: str
) -> None:
    chunks = chunk_pages((_page(1, line, "Skills", "Go"),), settings)
    assert chunks[0] == Chunk(0, 1, None, line)
    assert chunks[1] == Chunk(1, 1, "Skills", "Go")


def test_preamble_before_the_first_heading_has_no_section(settings: Settings) -> None:
    chunks = chunk_pages(
        (_page(1, "Aarav Mehta", "Staff Engineer", "Summary:", "Builds reliable services."),),
        settings,
    )

    assert chunks == (
        Chunk(0, 1, None, "Aarav Mehta\nStaff Engineer"),
        Chunk(1, 1, "Summary", "Builds reliable services."),
    )


def test_section_continues_onto_the_next_page_without_merging_pages(settings: Settings) -> None:
    chunks = chunk_pages(
        (
            _page(1, "Experience", "Built payment APIs."),
            _page(2, "Shipped the ledger.", "Education", "B.E. Computer Engineering"),
        ),
        settings,
    )

    assert chunks == (
        Chunk(0, 1, "Experience", "Built payment APIs."),
        Chunk(1, 2, "Experience", "Shipped the ledger."),
        Chunk(2, 2, "Education", "B.E. Computer Engineering"),
    )


def test_short_section_stays_one_chunk(settings: Settings) -> None:
    chunks = chunk_pages((_page(1, "Skills", "Python", "Go", "Postgres"),), settings)

    assert chunks == (Chunk(0, 1, "Skills", "Python\nGo\nPostgres"),)


def test_whitespace_only_sections_are_dropped(settings: Settings) -> None:
    chunks = chunk_pages(
        (_page(1, "Aarav Mehta", "Skills", "   ", "\t", "Education", "B.E. 2018"),),
        settings,
    )

    assert chunks == (
        Chunk(0, 1, None, "Aarav Mehta"),
        Chunk(1, 1, "Education", "B.E. 2018"),
    )


def test_eight_hundred_character_boundary(settings: Settings) -> None:
    exact = chunk_pages((_page(1, "Summary", "c" * 800),), settings)
    assert exact == (Chunk(0, 1, "Summary", "c" * 800),)

    lines = chunk_pages((_page(1, "Summary", "a" * 399, "b" * 400),), settings)
    assert lines == (Chunk(0, 1, "Summary", "a" * 399 + "\n" + "b" * 400),)
    assert len(lines[0].text) == 800

    over = chunk_pages((_page(1, "Summary", "a" * 400, "b" * 400),), settings)
    assert over == (
        Chunk(0, 1, "Summary", "a" * 400),
        Chunk(1, 1, "Summary", "b" * 400),
    )
    assert all(len(chunk.text) <= settings.chunk_size for chunk in over)


def test_overlap_on_a_long_paragraph(settings: Settings) -> None:
    words = [f"{index:010d}" for index in range(200)]
    paragraph = " ".join(words)
    chunks = chunk_pages((_page(1, "Experience", paragraph),), settings)

    assert len(chunks) > 1
    assert all(chunk.section == "Experience" and chunk.page == 1 for chunk in chunks)
    assert all(len(chunk.text) <= settings.chunk_size for chunk in chunks)
    rebuilt = chunks[0].text
    for previous, chunk in zip(chunks, chunks[1:], strict=False):
        assert chunk.text.startswith(previous.text[-settings.chunk_overlap :])
        rebuilt += chunk.text[settings.chunk_overlap :]
    assert rebuilt == paragraph
    assert chunks[0].text.endswith("0000000071")
    assert chunks[1].text.startswith("0000000061 ")


def test_single_word_longer_than_chunk_size_splits_on_characters(settings: Settings) -> None:
    word = "z" * 950
    chunks = chunk_pages((_page(1, "Projects", word),), settings)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= settings.chunk_size for chunk in chunks)
    rebuilt = chunks[0].text
    for previous, chunk in zip(chunks, chunks[1:], strict=False):
        assert chunk.text.startswith(previous.text[-settings.chunk_overlap :])
        rebuilt += chunk.text[settings.chunk_overlap :]
    assert rebuilt == word
    assert chunks[0].text == "z" * 800


def test_indexes_keep_increasing_across_pages(settings: Settings) -> None:
    chunks = chunk_pages(
        (
            _page(1, "Experience", "z" * 801),
            _page(2, "Still on the second page.", "Skills", "Python"),
        ),
        settings,
    )

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.page for chunk in chunks] == [1, 1, 2, 2]
    assert chunks[-1].index > max(chunk.index for chunk in chunks if chunk.page == 1)


def test_chunk_size_and_overlap_come_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    configured = Settings(_env_file=None, chunk_size=20, chunk_overlap=5)
    words = [f"w{index:02d}" for index in range(12)]
    paragraph = " ".join(words)
    chunks = chunk_pages((_page(1, "Summary", paragraph),), configured)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= configured.chunk_size for chunk in chunks)
    assert chunks[0].text.endswith("w04")
    assert chunks[1].text.startswith("w04 ")
    assert paragraph.startswith(chunks[0].text)


def test_blank_lines_split_before_single_line_breaks(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    configured = Settings(_env_file=None, chunk_size=30, chunk_overlap=0)
    chunks = chunk_pages(
        (_page(1, "Summary", "alpha bravo", "", "charlie delta echo"),),
        configured,
    )

    assert [chunk.text for chunk in chunks] == ["alpha bravo", "charlie delta echo"]


def test_sentences_split_before_spaces(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    configured = Settings(_env_file=None, chunk_size=40, chunk_overlap=0)
    chunks = chunk_pages(
        (_page(1, "Summary", "Hello there friend. This is longer than the limit."),),
        configured,
    )

    assert [chunk.text for chunk in chunks] == [
        "Hello there friend.",
        "This is longer than the limit.",
    ]


def test_chunk_log_records_counts_and_omits_resume_text(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    body = "Built payment APIs for the ledger team."
    with caplog.at_level(logging.INFO, logger="resume_reader.ingest.chunk"):
        chunk_pages((_page(1, "Experience", body),), settings)

    assert body not in caplog.text
    assert "chunks=1" in caplog.text
    assert "pages=1" in caplog.text
