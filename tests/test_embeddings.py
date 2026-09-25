"""Tests for the embedding + FAISS storage pipeline — AGENTS-37.

embed_texts() is the one seam that needs real network access (it downloads
BAAI/bge-small-en-v1.5 from Hugging Face Hub on first use), so every test here
replaces it with a small deterministic fake instead of the real model -- the
same reasoning test_chunking.py already applies to chunk_all_pages() needing a
database: the pure logic is tested directly, the one I/O-heavy dependency is
faked. build_index(), save_index() and load_index() still run against the
REAL faiss library, so the storage mechanics are genuinely exercised, not
mocked away.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from app import embeddings
from app.chunking import Chunk


def _fake_embed_texts(texts: list[str]) -> NDArray[np.float32]:
    """Stand-in for the real model: the exact same text always gets the exact
    same vector, and different texts get different (uncorrelated) vectors --
    enough to test storage and retrieval without downloading anything.
    """
    if not texts:
        return np.empty((0, embeddings.EMBEDDING_DIM), dtype="float32")
    vectors = []
    for text in texts:
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=embeddings.EMBEDDING_DIM).astype("float32")
        vector /= np.linalg.norm(vector)  # normalized, like the real model's output
        vectors.append(vector)
    return np.array(vectors, dtype="float32")


@pytest.fixture(autouse=True)
def _no_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Applies to every test in this file: use the fake embedder, never the real one."""
    monkeypatch.setattr(embeddings, "embed_texts", _fake_embed_texts)


def _chunk(**overrides: object) -> Chunk:
    """A Chunk with sane defaults, so each test only states what it cares about."""
    defaults: dict[str, object] = {
        "text": "Some chunk text.",
        "doc_id": "confluence:1",
        "doc_type": "raid_log",
        "space_key": "ConProK",
        "page_id": "1",
        "version": 1,
        "ticket_ids": [],
        "author_id": "author-1",
        "updated_at": "2026-09-25T00:00:00",
    }
    defaults.update(overrides)
    return Chunk(**defaults)  # type: ignore[arg-type]


def test_build_index_returns_one_vector_per_chunk() -> None:
    chunks = [_chunk(text="First chunk."), _chunk(text="Second chunk.")]
    index, metadata = embeddings.build_index(chunks)
    assert index.ntotal == 2
    assert len(metadata) == 2


def test_build_index_handles_no_chunks() -> None:
    """An empty Confluence space, or one where every page got skipped -- not an error."""
    index, metadata = embeddings.build_index([])
    assert index.ntotal == 0
    assert metadata == []


def test_metadata_carries_every_field_a_future_query_needs() -> None:
    chunk = _chunk(text="Risk R2 has no owner.", ticket_ids=["AGENTS-37"])
    _, metadata = embeddings.build_index([chunk])
    (row,) = metadata
    assert row["text"] == "Risk R2 has no owner."
    assert row["doc_id"] == "confluence:1"
    assert row["doc_type"] == "raid_log"
    assert row["ticket_ids"] == ["AGENTS-37"]


def test_save_and_load_index_round_trips(tmp_path: Path) -> None:
    chunks = [_chunk(text="First chunk."), _chunk(text="Second chunk.")]
    index, metadata = embeddings.build_index(chunks)
    embeddings.save_index(index, metadata, directory=tmp_path)

    loaded_index, loaded_metadata = embeddings.load_index(directory=tmp_path)
    assert loaded_index.ntotal == index.ntotal
    assert loaded_metadata == metadata


def test_a_chunk_is_findable_by_its_own_text() -> None:
    """Proves the stored vectors are actually searchable, not just stored.

    Embedding chunk 2's exact text again and searching the index it's part
    of should return chunk 2 first -- the fake embedder is deterministic per
    exact text, so this is the same guarantee the real model gives: the same
    text always lands in the same place in vector space.
    """
    chunks = [_chunk(text="Unrelated first chunk."), _chunk(text="Risk R2 has no owner.")]
    index, metadata = embeddings.build_index(chunks)

    query_vector = embeddings.embed_texts(["Risk R2 has no owner."])
    _, result_positions = index.search(query_vector, 1)

    assert metadata[result_positions[0][0]]["text"] == "Risk R2 has no owner."