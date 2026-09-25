"""Embed chunks and store them in FAISS — AGENTS-37, charter section 2b (steps 3-4).

Takes the Chunks that AGENTS-36's chunk_all_pages() produces and turns each
one into a vector a similarity search can query, then persists those vectors
to disk so a future query-time ticket can search without re-embedding on
every question.

One index per document type, not one combined index -- this is AGENTS-37's
own ticket spec and the first half of ADR-003's metadata-filtering
workaround: a narrative or hybrid question always searches within one type (a
RAID-log question never needs to rank against decision-log chunks), so
keeping separate indexes IS the filtering mechanism, not just an
optimization -- no candidate from the wrong type can ever occupy one of a
search's k results. AGENTS-38 builds the second half (ticket/date filtering
within a type) on top of this.

Settles decision-log item on the embedding model and charter section 2b step
3: BAAI/bge-small-en-v1.5 via sentence-transformers, 384 numbers per chunk,
running locally on CPU, no API cost. That choice is not remade here -- it is
the charter's own spec (ADR-005).

One rule that cannot be broken (ADR-005, charter section 2b): the same
embedding model must index documents and embed queries, because different
models produce vectors in different spaces and mixing them returns nonsense,
not an error. embed_texts() is the one function both this module and
AGENTS-38's query-time search must call, so that rule holds by construction
rather than by convention.

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
EMBEDDING_DIM = 384  # this model's fixed output size

# Relative to the current working directory, same convention excel_export.py uses for
# exports/ -- and matching the faiss_index/ entry already in .gitignore. Each doc_type
# gets its own subdirectory: faiss_index/<doc_type>/index.faiss + metadata.json.
FAISS_INDEX_DIR = Path("faiss_index")

IndexAndMetadata = tuple[faiss.Index, list[dict[str, Any]]]


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

    Both chunk indexing (this module) and question embedding (AGENTS-38) must
    call this exact function rather than each constructing their own model --
    that is what makes the charter's "never mix embedding models" rule
    impossible to violate by accident.

    normalize_embeddings=True scales every vector to length 1, so plain dot
    product between two vectors equals cosine similarity -- which is what
    _build_one_index's IndexFlatIP below, and AGENTS-38's search, assume.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype="float32")
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return np.asarray(vectors, dtype="float32")


def _build_one_index(chunks: list[Chunk]) -> IndexAndMetadata:
    """Embed one doc_type's chunks and build a single FAISS index over them.

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

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    if len(vectors):
        index.add(vectors)

    metadata = [asdict(chunk) for chunk in chunks]
    return index, metadata


def build_indexes(chunks: list[Chunk]) -> dict[str, IndexAndMetadata]:
    """Group chunks by doc_type and build one FAISS index per group.

    Returns a dict keyed by doc_type, e.g. {"raid_log": (index, metadata),
    "decision_log": (index, metadata), ...} -- exactly what save_indexes()
    below expects. A doc_type with zero chunks never appears as a key, rather
    than as an entry pointing at an empty index.
    """
    by_type: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_type.setdefault(chunk.doc_type, []).append(chunk)
    return {doc_type: _build_one_index(group) for doc_type, group in by_type.items()}


def save_indexes(indexes: dict[str, IndexAndMetadata], directory: Path = FAISS_INDEX_DIR) -> None:
    """Persist every doc_type's index and metadata to its own subdirectory.

    faiss_index/<doc_type>/index.faiss + faiss_index/<doc_type>/metadata.json.
    Two files per type, not one: FAISS's own write_index only knows how to
    write vectors, so the metadata list travels alongside it as plain JSON --
    readable in a text editor, and consistent with doc_type_rules.json's use
    of JSON for data this project hand-inspects.
    """
    for doc_type, (index, metadata) in indexes.items():
        type_dir = directory / doc_type
        type_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(type_dir / "index.faiss"))
        with (type_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f)


def load_index(doc_type: str, directory: Path = FAISS_INDEX_DIR) -> IndexAndMetadata:
    """Read back one doc_type's index -- what AGENTS-38's search calls.

    Scoped to one type rather than loading everything: a narrative or hybrid
    question already knows which type it wants (the same categories
    doc_type_rules.json classifies pages into), so there is never a reason to
    load more than one index for a single search.
    """
    type_dir = directory / doc_type
    index = faiss.read_index(str(type_dir / "index.faiss"))
    with (type_dir / "metadata.json").open(encoding="utf-8") as f:
        metadata: list[dict[str, Any]] = json.load(f)
    return index, metadata


def available_doc_types(directory: Path = FAISS_INDEX_DIR) -> list[str]:
    """Which doc_types actually have a built index on disk.

    AGENTS-38's search calls this to tell "nothing indexed for this type yet"
    apart from a real error, rather than letting a missing-file exception
    from faiss.read_index surface as the signal. Also this module's own
    smoke-test listing below.
    """
    if not directory.exists():
        return []
    return sorted(p.name for p in directory.iterdir() if p.is_dir())


def embed_all_chunks() -> dict[str, IndexAndMetadata]:
    """Full pipeline: chunk every stored page, embed the chunks, save one index per doc_type.

    AGENTS-37's actual entry point -- chunk_all_pages() is AGENTS-36's, kept
    separate so this module can be tested without needing a populated
    database, the same reasoning chunking.py separates chunk_page() from
    chunk_all_pages().
    """
    chunks = chunk_all_pages()
    indexes = build_indexes(chunks)
    save_indexes(indexes)
    return indexes


if __name__ == "__main__":
    result = embed_all_chunks()
    total = sum(index.ntotal for index, _ in result.values())
    print(f"Embedded {total} chunk(s) across {len(result)} doc types, into {FAISS_INDEX_DIR}:")
    for type_name, (index, _) in sorted(result.items()):
        print(f"  {type_name}: {index.ntotal}")
