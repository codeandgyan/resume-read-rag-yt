"""PDF extraction: two-column reading order, the corpus, and corrupt input."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from resume_reader.ingest.extract import UnreadablePdf, extract_pdf

_CORPUS = Path(__file__).resolve().parents[2] / "corpus"

_TITLE = "Jordan Lee, Staff Engineer"
_LEFT_TOP = ("Platform experience", "Built payment APIs")
_RIGHT_TOP = ("Core skills", "Python and Postgres")
_BRIDGE = "Selected work across payments and platforms"
_LEFT_BOTTOM = ("B.E. Computer Engineering",)
_RIGHT_BOTTOM = ("AWS certified developer",)


def _pdf_bytes(document: pymupdf.Document) -> bytes:
    data = document.tobytes()
    document.close()
    return data


def _two_column_pdf() -> bytes:
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    page.insert_textbox(
        pymupdf.Rect(48, 36, 564, 78),
        _TITLE,
        fontsize=16,
        align=pymupdf.TEXT_ALIGN_CENTER,
    )
    for baseline, line in enumerate(_LEFT_TOP):
        page.insert_text((72, 130 + baseline * 22), line, fontsize=12)
    for baseline, line in enumerate(_RIGHT_TOP):
        page.insert_text((360, 130 + baseline * 22), line, fontsize=12)
    page.insert_textbox(
        pymupdf.Rect(48, 200, 564, 240),
        _BRIDGE,
        fontsize=12,
        align=pymupdf.TEXT_ALIGN_CENTER,
    )
    page.insert_text((72, 280), _LEFT_BOTTOM[0], fontsize=12)
    page.insert_text((360, 280), _RIGHT_BOTTOM[0], fontsize=12)
    document.new_page(width=612, height=792)
    return _pdf_bytes(document)


def _single_column_pdf() -> bytes:
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    for baseline, line in enumerate(
        (
            "Professional summary",
            "Builds reliable services for finance teams.",
            "Education",
            "B.E. Computer Engineering, 2018",
        )
    ):
        page.insert_text((72, 80 + baseline * 22), line, fontsize=12)
    return _pdf_bytes(document)


def _pdf_with_a_blank_middle_page() -> bytes:
    document = pymupdf.open()
    first = document.new_page(width=612, height=792)
    first.insert_text((72, 80), "First page line", fontsize=12)
    document.new_page(width=612, height=792)
    third = document.new_page(width=612, height=792)
    third.insert_text((72, 80), "Third page line", fontsize=12)
    return _pdf_bytes(document)


def test_two_column_page_reads_full_width_lines_around_each_column() -> None:
    extracted = extract_pdf(_two_column_pdf())
    assert extracted.page_count == 2
    assert [page.number for page in extracted.pages] == [1]
    assert extracted.pages[0].lines == (
        _TITLE,
        *_LEFT_TOP,
        *_RIGHT_TOP,
        _BRIDGE,
        *_LEFT_BOTTOM,
        *_RIGHT_BOTTOM,
    )


def test_single_column_page_stays_in_vertical_order() -> None:
    extracted = extract_pdf(_single_column_pdf())
    assert extracted.page_count == 1
    assert extracted.pages[0].number == 1
    assert extracted.pages[0].lines == (
        "Professional summary",
        "Builds reliable services for finance teams.",
        "Education",
        "B.E. Computer Engineering, 2018",
    )


def test_blank_page_is_dropped_without_renumbering() -> None:
    extracted = extract_pdf(_pdf_with_a_blank_middle_page())
    assert extracted.page_count == 3
    assert [(page.number, page.lines) for page in extracted.pages] == [
        (1, ("First page line",)),
        (3, ("Third page line",)),
    ]


def test_corpus_pages_keep_one_based_numbers_and_no_blank_text() -> None:
    pdfs = sorted(_CORPUS.glob("*.pdf"))
    assert pdfs
    for path in pdfs:
        data = path.read_bytes()
        extracted = extract_pdf(data)
        with pymupdf.open(stream=data, filetype="pdf") as source:
            assert extracted.page_count == source.page_count
        assert extracted.page_count >= 1
        assert extracted.pages, path.name
        numbers = [page.number for page in extracted.pages]
        assert numbers == sorted(set(numbers))
        for page in extracted.pages:
            assert 1 <= page.number <= extracted.page_count
            assert page.lines
            assert all(line.strip() for line in page.lines)


def test_two_column_corpus_reads_the_left_column_before_the_right() -> None:
    aarav = "\n".join(
        extract_pdf((_CORPUS / "01_aarav_mehta_modern_two_column.pdf").read_bytes()).pages[0].lines
    )
    assert aarav.index("PROFILE") < aarav.index("EXPERIENCE")
    assert aarav.index("EDUCATION") < aarav.index("Northstar Payments")
    assert "Python • FastAPI • Django • PostgreSQL •" in aarav
    assert "Redis • Kafka • AWS • Docker •" in aarav
    assert "EXPERIENCE Senior" not in aarav

    karthik = "\n".join(
        extract_pdf(
            (_CORPUS / "05_karthik_rao_academic_project_heavy.pdf").read_bytes()
        ).pages[0].lines
    )
    assert karthik.index("PROFILE") < karthik.index("EXPERIENCE")
    assert karthik.index("KARTHIK") < karthik.index("SilverFern")


def test_chronological_corpus_stays_in_vertical_order() -> None:
    priya = "\n".join(
        extract_pdf((_CORPUS / "02_priya_nair_classic_chronological.pdf").read_bytes()).pages[0].lines
    )
    assert priya.index("PROFESSIONAL SUMMARY") < priya.index("PROFESSIONAL EXPERIENCE")
    assert priya.index("BlueKite") < priya.index("MangoTree")
    assert priya.index("MangoTree") < priya.index("TECHNICAL SKILLS")


@pytest.mark.parametrize("payload", [b"%PDF", b"%PDF-1.4\n%truncated", b""])
def test_truncated_pdf_is_unreadable(payload: bytes) -> None:
    with pytest.raises(UnreadablePdf):
        extract_pdf(payload)
