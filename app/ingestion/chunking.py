"""Recursive, structure-aware chunking with overlap.

Text is split on the coarsest separator that works (paragraph, line, sentence,
word) and the pieces are greedily packed back into chunks of at most
`chunk_size` characters. Consecutive chunks share up to `overlap` characters
so an answer that straddles a boundary is still retrievable.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from app.ingestion.loaders import Section

SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


@dataclass
class Chunk:
    id: str
    doc_id: str
    filename: str
    index: int
    text: str
    page: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def search_text(self) -> str:
        # The filename carries useful signal ("pricing_2024.pdf") for both
        # dense and lexical retrieval, so it is prepended at index time.
        return f"{self.filename}\n{self.text}"


def _split(text: str, separators: list[str], size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    separator = next((s for s in separators if s and s in text), "")
    if not separator:
        return [text[i : i + size] for i in range(0, len(text), size)]
    remaining = separators[separators.index(separator) + 1 :]
    parts = text.split(separator)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        piece = part + (separator if i < len(parts) - 1 else "")
        if len(piece) <= size:
            pieces.append(piece)
        else:
            pieces.extend(_split(piece, remaining, size))
    return pieces


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    window: list[str] = []
    length = 0
    for piece in pieces:
        if window and length + len(piece) > size:
            chunks.append("".join(window))
            # Drop from the front until only the overlap tail remains and the
            # next piece fits.
            while window and (length > overlap or length + len(piece) > size):
                length -= len(window.pop(0))
        window.append(piece)
        length += len(piece)
    if window:
        chunks.append("".join(window))
    return [c.strip() for c in chunks if c.strip()]


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    return _merge(_split(text, SEPARATORS, chunk_size), chunk_size, overlap)


def chunk_sections(
    sections: list[Section], doc_id: str, filename: str, chunk_size: int, overlap: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        for text in split_text(section.text, chunk_size, overlap):
            index = len(chunks)
            chunk_id = hashlib.sha1(f"{doc_id}:{index}".encode()).hexdigest()[:16]
            chunks.append(Chunk(chunk_id, doc_id, filename, index, text, section.page))
    return chunks
