"""Detect resume headings and chunk each section within a page."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from resume_reader.ingest.extract import ExtractedPage
from resume_reader.settings import Settings

logger = logging.getLogger(__name__)

# A heading is only a known section name, optionally with one allowed prefix
# and a trailing colon. Anything else is body text.
_HEADING = re.compile(
    r"^(?:(?:professional|core|technical|selected|key|relevant)\s+)?"
    r"(?:certifications|contributions|experience|employment|education|strengths|"
    r"objective|projects|summary|profile|skills)"
    r"\s*:?\s*$",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """One piece of a section that stays on a single page.

    ``index`` counts from 0 across the whole document. ``section`` is ``None``
    for text that appears before the first heading.
    """

    index: int
    page: int
    section: str | None
    text: str


@dataclass(frozen=True)
class _Separator:
    """A split point, and the string used when the pieces are joined again."""

    joiner: str
    pattern: str = ""
    regex: re.Pattern[str] | None = None


# First match wins. Later separators are used only on a piece that is still
# longer than the chunk size.
_SEPARATORS = (
    _Separator("\n\n", "\n\n"),
    _Separator("\n", "\n"),
    _Separator(" ", regex=_SENTENCE),
    _Separator(" ", " "),
    _Separator(""),
)


@dataclass
class _Group:
    page: int
    section: str | None
    lines: list[str]


def chunk_pages(pages: Sequence[ExtractedPage], settings: Settings) -> tuple[Chunk, ...]:
    """Chunk body lines by page and section.

    Heading lines label the following text and are not copied into a chunk.
    The label stays in force on later pages until the next heading. A group
    of ``settings.chunk_size`` characters or fewer is one chunk. Longer text
    is split on blank lines, line breaks, sentence boundaries, spaces, and
    then characters, keeping ``settings.chunk_overlap`` characters of overlap.
    Whitespace-only pieces are dropped. Chunks are numbered from 0.
    """
    chunks: list[Chunk] = []
    for group in _groups(pages):
        for piece in _split(
            "\n".join(group.lines),
            _SEPARATORS,
            settings.chunk_size,
            settings.chunk_overlap,
        ):
            chunks.append(
                Chunk(
                    index=len(chunks),
                    page=group.page,
                    section=group.section,
                    text=piece,
                )
            )

    logger.info(
        "chunked pages=%s chunks=%s chunk_size=%s chunk_overlap=%s",
        len(pages),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return tuple(chunks)


def _section_heading(line: str) -> str | None:
    """Return the heading as written, without a trailing colon."""
    stripped = line.strip()
    if _HEADING.fullmatch(stripped) is None:
        return None
    return re.sub(r"\s*:\s*$", "", stripped)


def _groups(pages: Sequence[ExtractedPage]) -> list[_Group]:
    groups: list[_Group] = []
    section: str | None = None
    current: _Group | None = None
    for page in pages:
        for line in page.lines:
            heading = _section_heading(line)
            if heading is not None:
                section = heading
                current = None
                continue
            if current is None or current.page != page.number or current.section != section:
                current = _Group(page.number, section, [])
                groups.append(current)
            current.lines.append(line)
    return groups


def _split(
    text: str,
    separators: Sequence[_Separator],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    separator, rest = _choose(cleaned, separators)
    parts = _pieces(cleaned, separator)
    chunks: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        if not pending:
            return
        chunks.extend(_merge(pending, separator.joiner, chunk_size, chunk_overlap))
        pending.clear()

    for part in parts:
        if len(part) <= chunk_size:
            pending.append(part)
            continue
        flush()
        if rest:
            chunks.extend(_split(part, rest, chunk_size, chunk_overlap))
        else:
            chunks.extend(_merge(list(part), "", chunk_size, chunk_overlap))
    flush()
    return [chunk for chunk in chunks if chunk.strip()]


def _choose(text: str, separators: Sequence[_Separator]) -> tuple[_Separator, tuple[_Separator, ...]]:
    for index, separator in enumerate(separators):
        if _present(text, separator):
            return separator, tuple(separators[index + 1 :])
    return _Separator(""), ()


def _present(text: str, separator: _Separator) -> bool:
    if separator.regex is not None:
        return separator.regex.search(text) is not None
    if separator.pattern == "":
        return True
    return separator.pattern in text


def _pieces(text: str, separator: _Separator) -> list[str]:
    if separator.regex is not None:
        parts = separator.regex.split(text)
    elif separator.pattern == "":
        parts = list(text)
    else:
        parts = text.split(separator.pattern)
    return [part for part in parts if part]


def _merge(parts: list[str], joiner: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Pack pieces up to ``chunk_size``, repeating a suffix within ``chunk_overlap``."""
    chunks: list[str] = []
    window: list[str] = []
    joiner_len = len(joiner)

    for part in parts:
        if window and _joined_len(window, joiner_len) + joiner_len + len(part) > chunk_size:
            text = joiner.join(window).strip()
            if text:
                chunks.append(text)
            while window and not _suffix_fits(window, part, joiner_len, chunk_size, chunk_overlap):
                window.pop(0)
        window.append(part)

    if window:
        text = joiner.join(window).strip()
        if text:
            chunks.append(text)
    return chunks


def _suffix_fits(
    window: list[str],
    part: str,
    joiner_len: int,
    chunk_size: int,
    chunk_overlap: int,
) -> bool:
    size = _joined_len(window, joiner_len)
    if size > chunk_overlap:
        return False
    extra = joiner_len if window else 0
    return size + extra + len(part) <= chunk_size


def _joined_len(parts: list[str], joiner_len: int) -> int:
    if not parts:
        return 0
    return sum(len(part) for part in parts) + joiner_len * (len(parts) - 1)
