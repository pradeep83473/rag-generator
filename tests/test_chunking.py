import pytest

from app.ingestion.chunking import chunk_sections, split_text
from app.ingestion.loaders import Section


def test_short_text_is_single_chunk():
    assert split_text("hello world", 100, 10) == ["hello world"]


def test_chunks_respect_size_and_cover_text():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(20))
    chunks = split_text(text, 300, 50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    for i in range(20):
        assert any(f"Paragraph {i}." in c for c in chunks)


def test_consecutive_chunks_overlap():
    text = " ".join(f"w{i}" for i in range(400))
    chunks = split_text(text, 200, 60)
    for a, b in zip(chunks, chunks[1:]):
        assert a.split()[-1] in b.split()


def test_unbreakable_text_is_hard_split():
    chunks = split_text("x" * 1000, 256, 32)
    assert all(len(c) <= 256 for c in chunks)
    assert sum(len(c) for c in chunks) >= 1000


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text("abc", 10, 10)


def test_chunk_sections_keeps_page_numbers():
    chunks = chunk_sections([Section("page one text", 1), Section("page two text", 2)], "d1", "f.pdf", 100, 10)
    assert [c.page for c in chunks] == [1, 2]
    assert [c.index for c in chunks] == [0, 1]
    assert len({c.id for c in chunks}) == 2
