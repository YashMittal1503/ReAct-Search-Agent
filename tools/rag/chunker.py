"""Markdown-aware chunking of scraped pages.

Pages arrive as clean Markdown from trafilatura. Chunks are built from whole paragraphs where
possible, so passages stay readable and quotable. Oversize paragraphs are split at sentence
boundaries, and only a sentence longer than the chunk size is hard-split by words. Each chunk
remembers the nearest preceding heading, which is included when scoring (headings carry a lot of
topical signal) and when rendering. Consecutive chunks overlap by a few trailing sentences so a fact
split across a boundary is still retrievable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass(frozen=True)
class Chunk:
    """A contiguous passage of one source."""

    url: str
    index: int          # position within the source (0-based), used for neighbours and ordering
    text: str
    heading: str = ""

    @property
    def scoring_text(self) -> str:
        """Text used for retrieval: heading + body."""
        return f"{self.heading}\n{self.text}" if self.heading else self.text


def _sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(paragraph) if s.strip()]


def _hard_split(text: str, max_chars: int) -> list[str]:
    words, parts, current = text.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    return parts + ([current] if current else [])


def _units(paragraph: str, max_chars: int) -> list[str]:
    """Split a paragraph into pieces no longer than ``max_chars`` (sentences, then words)."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    units: list[str] = []
    for sentence in _sentences(paragraph):
        units.extend([sentence] if len(sentence) <= max_chars else _hard_split(sentence, max_chars))
    return units


def chunk_markdown(url: str, text: str, *, max_chars: int = 900, overlap: int = 150) -> list[Chunk]:
    """Split a Markdown document into overlapping, heading-tagged chunks.

    >>> doc = "# Intro\\nShort para.\\n\\n## Costs\\n" + "Cost sentence number one is here. " * 40
    >>> chunks = chunk_markdown("u", doc, max_chars=300, overlap=60)
    >>> chunks[0].heading, chunks[1].heading, all(len(c.text) <= 300 for c in chunks)
    ('Intro', 'Costs', True)
    """
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    chunks: list[Chunk] = []
    heading = ""
    current: list[str] = []   # units of the chunk being built (may start with carried overlap)
    fresh = False             # True once the current chunk holds at least one NEW unit
    current_heading = ""

    def flush(carry_overlap: bool = True) -> None:
        nonlocal current, fresh
        body = " ".join(current).strip()
        if body and fresh:  # never emit a chunk that only repeats the previous chunk's overlap
            chunks.append(Chunk(url=url, index=len(chunks), text=body, heading=current_heading))
        carry: list[str] = []
        if carry_overlap and overlap and body:
            size = 0
            for sentence in reversed(_sentences(body)):
                if size + len(sentence) > overlap:
                    break
                carry.insert(0, sentence)
                size += len(sentence) + 1
        current, fresh = carry, False

    for block in re.split(r"\n\s*\n", text):
        lines = [ln for ln in block.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        if m := _HEADING_RE.match(lines[0]):
            flush(carry_overlap=False)  # sections don't share overlap
            heading = m.group(1).strip().strip("#").strip()
            lines = lines[1:]
            if not lines:
                continue
        if not fresh:
            current_heading = heading
        paragraph = " ".join(ln.strip() for ln in lines)
        for unit in _units(paragraph, max_chars):
            if fresh and len(" ".join(current)) + 1 + len(unit) > max_chars:
                flush()
                current_heading = heading
            elif current and not fresh and len(" ".join(current)) + 1 + len(unit) > max_chars:
                current = []  # carried overlap + this unit would overflow: drop the overlap
            current.append(unit)
            fresh = True
    flush(carry_overlap=False)
    return chunks
