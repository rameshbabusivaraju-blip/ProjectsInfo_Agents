"""Embed chunks and store them in FAISS — AGENTS-37, charter section 2b (steps 3-4).

Takes the Chunks that AGENTS-36's chunk_all_pages() produces and turns each
one into a vector a similarity search can query, then persists those vectors
to disk (in faiss_index/, alongside the metadata needed to turn a matching
vector back into a citeable chunk) so a future query-time ticket can search
without re-embedding on every question.

Settles decision-log item on the embedding model and charter section 2b step
3: BAAI/bge-small-en-v1.5 via sentence-transformers, 384 numbers per chunk,
running locally on CPU, no API cost. That choice is not remade here -- it is
the charter's own spec.

One rule that cannot be broken (charter section 2b): the same embedding model
must index documents and embed queries, because different models produce
vectors in different spaces and mixing them returns nonsense, not an error.
embed_texts() is the one function both this module and any future query-time
code must call, so that rule holds by construction rather than by convention.

Deliberately out of scope here (charter section 8's "each is its own
ticket" rule): searching the index. This module only builds and persists it;
a later ticket embeds a question with embed_texts() and calls
faiss.Index.search() against what load_index() returns.

Run it with:  python -m app.embeddings
"""

from __future__ import annotations

import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from app.chunking import Chunk, chunk_all_pages

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # this model's fixed output size; asserted in build_index below

# Relative to the current working directory, same convention excel_export.py uses
# for exports/ -- and matching the faiss_index/ entry already in .gitignore.
FAISS_INDEX_DIR = Path("faiss_index")


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the embedding model once per process and reuse it.

    Loading (and, on a machine's first run, downloading the model from
    Hugging Face Hub) is the slow part -- seconds to minutes -- while running
    an already-loaded model is fast. lru_cache on a zero-argument function is
    a simple way to compute something once and remember it for next time.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]) -> NDArray[np.float32]:
    """Embed a list of texts with the one model this whole project shares.

    Both chunk indexing (this module) and question embedding (a future
    query-time ticket) must call this exact function rather than each
    constructing their own model -- that is what makes the charter's
    "never mix embedding models" rule impossible to violate by accident.

    normalize_embeddings=True scales every vector to length 1, so plain dot
    product between two vectors equals cosine similarity -- which is what
    build_index's IndexFlatIP below assumes.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype="float32")
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return np.asarray(vectors, dtype="float32")


def build_index(chunks: list[Chunk]) -> tuple[faiss.Index, list[dict[str, Any]]]:
    """Embed every chunk and build a FAISS index over the resulting vectors.

    Returns the index paired with a metadata list in the same order: FAISS
    itself stores only vectors and an integer position, nothing else, so
    metadata[i] is what position i in the index actually is -- the chunk's
    text plus everything AGENTS-38's filtering needs (doc_id, doc_type,
    page_id, version, ticket_ids, author_id, updated_at). asdict() is used
    rather than listing fields by hand so a new Chunk field is never
    silently missing from what gets stored.

    IndexFlatIP does exact (not approximate) search by inner product --
    correct here because embed_texts() already normalizes every vector to
    length 1, making inner product equal cosine similarity. "Flat" and exact
    is the right choice at ProjectPulse's scale (dozens of pages, not
    millions) -- an approximate index would trade correctness for speed this
    project does not need (ADR-009: learning-grade, not production-grade).
    """
    vectors = embed_texts([chunk.text for chunk in chunks])
    assert vectors.shape[1] == EMBEDDING_DIM if len(vectors) else True

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    if len(vectors):
        index.add(vectors)

    metadata = [asdict(chunk) for chunk in chunks]
    return index, metadata


def save_index(
    index: faiss.Index, metadata: list[dict[str, Any]], directory: Path = FAISS_INDEX_DIR
) -> None:
    """Persist an index and its metadata to disk, so it survives past one run.

    Two files, not one: FAISS's own write_index only knows how to write
    vectors, so the metadata list travels alongside it as plain JSON --
    readable in a text editor, and consistent with doc_type_rules.json's use
    of JSON for data this project hand-inspects.
    """
    directory.mkdir(exist_ok=True)
    faiss.write_index(index, str(directory / "index.faiss"))
    with (directory / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f)


def load_index(directory: Path = FAISS_INDEX_DIR) -> tuple[faiss.Index, list[dict[str, Any]]]:
    """Read back what save_index() wrote. What a future query-time ticket calls."""
    index = faiss.read_index(str(directory / "index.faiss"))
    with (directory / "metadata.json").open(encoding="utf-8") as f:
        metadata: list[dict[str, Any]] = json.load(f)
    return index, metadata


def embed_all_chunks() -> tuple[faiss.Index, list[dict[str, Any]]]:
    """Full pipeline: chunk every stored page, embed the chunks, save the index.

    AGENTS-37's actual entry point -- chunk_all_pages() is AGENTS-36's, kept
    separate so this module can be tested without needing a populated
    database, the same reasoning chunking.py separates chunk_page() from
    chunk_all_pages().
    """
    chunks = chunk_all_pages()
    index, metadata = build_index(chunks)
    save_index(index, metadata)
    return index, metadata


if __name__ == "__main__":
    result_index, result_metadata = embed_all_chunks()
    by_type: dict[str, int] = {}
    for row in result_metadata:
        by_type[row["doc_type"]] = by_type.get(row["doc_type"], 0) + 1

    print(f"Embedded {result_index.ntotal} chunk(s) into {FAISS_INDEX_DIR}/:")
    for doc_type, count in sorted(by_type.items()):
        print(f"  {doc_type}: {count}")