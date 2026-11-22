"""Extract resume text in reading order from PDF word boxes."""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass

import pymupdf

logger = logging.getLogger(__name__)

_BIN_PT = 2.0
_MIN_VALLEY_PT = 10.0
_SIDE_BALANCE = 0.2
_OPEN_FRACTION = 0.35


class UnreadablePdf(Exception):
    """The bytes are missing, truncated, or not a PDF that can be opened."""


@dataclass(frozen=True)
class ExtractedPage:
    """One page that contained text. ``number`` is the PDF's 1-based index."""

    number: int
    lines: tuple[str, ...]


@dataclass(frozen=True)
class ExtractedPdf:
    """Kept pages, plus the file's page count including dropped blank pages."""

    pages: tuple[ExtractedPage, ...]
    page_count: int


@dataclass(frozen=True)
class _Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass(frozen=True)
class _Line:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass(frozen=True)
class _Gutter:
    """Vertical channel between columns. A point gutter is the page center."""

    x: float
    x0: float
    x1: float


def extract_pdf(data: bytes) -> ExtractedPdf:
    """Read ``data`` as a PDF and return non-blank lines in reading order.

    Blank pages are omitted and the pages that remain keep their original
    1-based indexes. ``page_count`` includes those dropped pages. A missing or
    corrupt PDF raises ``UnreadablePdf``.
    """
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise UnreadablePdf("unreadable PDF") from exc
    try:
        page_count = document.page_count
        pages: list[ExtractedPage] = []
        for index in range(page_count):
            lines = _page_lines(document[index])
            if not lines:
                continue
            pages.append(ExtractedPage(number=index + 1, lines=tuple(lines)))
    except UnreadablePdf:
        raise
    except Exception as exc:
        raise UnreadablePdf("unreadable PDF") from exc
    finally:
        document.close()

    logger.info(
        "extracted pdf page_count=%s pages_with_text=%s lines=%s",
        page_count,
        len(pages),
        sum(len(page.lines) for page in pages),
    )
    return ExtractedPdf(pages=tuple(pages), page_count=page_count)


def _page_lines(page: pymupdf.Page) -> list[str]:
    words = _words(page)
    if not words:
        return []
    gutter = _find_gutter(words, float(page.rect.x0), float(page.rect.x1))
    return _reading_order(_lines(words, gutter), gutter)


def _words(page: pymupdf.Page) -> list[_Word]:
    words: list[_Word] = []
    for item in page.get_text("words"):
        text = str(item[4]).strip()
        if not text:
            continue
        x0, y0, x1, y1 = (float(item[0]), float(item[1]), float(item[2]), float(item[3]))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        words.append(_Word(x0, y0, x1, y1, text))
    return words


def _find_gutter(words: list[_Word], origin: float, edge: float) -> _Gutter:
    """Find the blank vertical channel that has text on both sides.

    The page center is used when no such channel exists, so a single column
    stays a single column.
    """
    width = edge - origin
    center = origin + width / 2
    fallback = _Gutter(center, center, center)
    if width <= 0 or not words:
        return fallback

    bins = max(1, int(width / _BIN_PT) + 1)
    coverage = [0] * bins
    for word in words:
        start = min(bins - 1, max(0, int((word.x0 - origin) / _BIN_PT)))
        stop = min(bins, max(start + 1, int((word.x1 - origin) / _BIN_PT) + 1))
        for index in range(start, stop):
            coverage[index] += 1

    sample = coverage[int(bins * 0.2) : int(bins * 0.8)] or coverage
    open_level = max(2, int(statistics.median(sample) * _OPEN_FRACTION))
    best: tuple[float, int, int] | None = None
    index = 0
    while index < bins:
        if coverage[index] > open_level:
            index += 1
            continue
        stop = index + 1
        while stop < bins and coverage[stop] <= open_level:
            stop += 1
        valley = (stop - index) * _BIN_PT
        midpoint = origin + (index + stop) * _BIN_PT / 2
        left_mass = sum(coverage[:index])
        right_mass = sum(coverage[stop:])
        total = left_mass + right_mass
        balanced = total > 0 and min(left_mass, right_mass) / total >= _SIDE_BALANCE
        central = origin + width * 0.15 < midpoint < origin + width * 0.85
        if valley >= _MIN_VALLEY_PT and balanced and central:
            if best is None or valley > best[0]:
                best = (valley, index, stop)
        index = stop

    if best is None:
        return fallback
    _, start, stop = best
    x0 = origin + start * _BIN_PT
    x1 = origin + stop * _BIN_PT
    return _Gutter((x0 + x1) / 2, x0, x1)


def _lines(words: list[_Word], gutter: _Gutter) -> list[_Line]:
    lines: list[_Line] = []
    for row in _rows(words):
        ordered = sorted(row, key=lambda word: word.x0)
        group = [ordered[0]]
        for word in ordered[1:]:
            if _gap_splits_columns(group[-1], word, gutter):
                lines.append(_line(group))
                group = [word]
            else:
                group.append(word)
        lines.append(_line(group))
    return [line for line in lines if line.text.strip()]


def _rows(words: list[_Word]) -> list[list[_Word]]:
    """Group words that share a baseline.

    The tolerance stays under the gap between stacked lines. A tall glyph can
    overlap the next line's box, so the boxes themselves are not the signal.
    Words from the other column may share that baseline; the gutter split
    separates them afterward.
    """
    tolerance = _baseline_tolerance(words)
    rows: list[list[_Word]] = []
    anchors: list[float] = []
    for word in sorted(words, key=lambda item: (item.y0, item.x0)):
        match: int | None = None
        closest = tolerance
        for index, anchor in enumerate(anchors):
            delta = abs(word.y0 - anchor)
            if delta <= closest:
                match = index
                closest = delta
        if match is None:
            rows.append([word])
            anchors.append(word.y0)
        else:
            rows[match].append(word)
    return rows


def _baseline_tolerance(words: list[_Word]) -> float:
    heights = sorted(word.y1 - word.y0 for word in words)
    median_height = heights[len(heights) // 2]
    return max(1.0, median_height * 0.45)


def _gap_splits_columns(previous: _Word, word: _Word, gutter: _Gutter) -> bool:
    gap_start = previous.x1
    gap_end = word.x0
    if gap_end - gap_start < 8:
        return False
    if not (gap_start < gutter.x < gap_end):
        return False
    valley = gutter.x1 - gutter.x0
    return gap_end - gap_start >= max(8.0, valley * 0.5)


def _line(words: list[_Word]) -> _Line:
    ordered = sorted(words, key=lambda word: word.x0)
    return _Line(
        x0=min(word.x0 for word in ordered),
        y0=min(word.y0 for word in ordered),
        x1=max(word.x1 for word in ordered),
        y1=max(word.y1 for word in ordered),
        text=" ".join(word.text for word in ordered),
    )


def _reading_order(lines: list[_Line], gutter: _Gutter) -> list[str]:
    ordered = sorted(lines, key=lambda line: (line.y0, line.x0))
    if not _is_two_column(ordered, gutter):
        return [line.text for line in ordered]

    result: list[str] = []
    left: list[_Line] = []
    right: list[_Line] = []

    def flush() -> None:
        left.sort(key=lambda line: (line.y0, line.x0))
        right.sort(key=lambda line: (line.y0, line.x0))
        result.extend(line.text for line in left)
        result.extend(line.text for line in right)
        left.clear()
        right.clear()

    for line in ordered:
        side = _side(line, gutter)
        if side == "full":
            flush()
            result.append(line.text)
        elif side == "left":
            left.append(line)
        else:
            right.append(line)
    flush()
    return result


def _is_two_column(lines: list[_Line], gutter: _Gutter) -> bool:
    if len(lines) < 4:
        return False
    sided = sum(1 for line in lines if _side(line, gutter) != "full")
    return sided / len(lines) >= 0.6


def _side(line: _Line, gutter: _Gutter) -> str:
    if line.x0 < gutter.x < line.x1:
        return "full"
    if line.x1 <= gutter.x:
        return "left"
    return "right"
