"""Search one doc_type's FAISS index with metadata filtering — AGENTS-38.

The second half of ADR-003's workaround for FAISS having no native metadata
filtering. AGENTS-37 already solved the doc_type dimension: each type has its
own index, so searching "within the RAID log" means loading the raid_log
index and nothing else -- no filtering code is needed for that dimension at
all, by construction.

What's left is filtering WITHIN a type -- by ticket, or by a date range (a
sprint's own start/end dates, for a "this sprint" question) -- and FAISS
still can't do that natively. The workaround, per ADR-003 and this ticket's
own spec: retrieve a wider set than needed (k=15), filter the survivors in
Python against the metadata every chunk already carries, then keep the top
few (3). A combined filter (ticket AND date) can end up returning fewer than
top_n, or nothing -- that is accepted, not a bug to chase (ADR-003: "accept
lower recall"). The alternative -- re-querying with a larger k until enough
survive -- trades a simple, bounded operation for an unbounded one, to buy
back recall this project's scale does not need.

Run it with:  python -m app.retrieval "<query>" <doc_type>
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.embeddings import FAISS_INDEX_DIR, available_doc_types, embed_texts, load_index

DEFAULT_K = 15  # how many candidates FAISS returns before Python-side filtering (ADR-003)
DEFAULT_TOP_N = 3  # how many survive filtering and get returned (ADR-003)


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk, ranked by how well it matched the query.

    score is cosine similarity: embed_texts() normalizes every vector to
    length 1, so FAISS's inner product on IndexFlatIP already equals cosine
    similarity. 1.0 is an exact match, 0.0 unrelated, negative opposed.
    """

    score: float
    text: str
    doc_id: str
    doc_type: str
    space_key: str
    page_id: str
    version: int
    ticket_ids: list[str]
    author_id: str | None
    updated_at: str


def _passes_filters(
    row: dict[str, Any],
    ticket_id: str | None,
    updated_after: str | None,
    updated_before: str | None,
) -> bool:
    """One chunk's stored metadata against this search's filters. All optional, all AND together.

    updated_at is stored as an ISO-8601 string (chunking.py: page.updated.isoformat()),
    and ISO-8601 strings sort the same way their dates do, so plain string
    comparison works without parsing either side back into a datetime.
    updated_after is inclusive, updated_before is exclusive -- a half-open
    interval [after, before), the same convention Python slicing uses.
    """
    if ticket_id is not None and ticket_id not in row["ticket_ids"]:
        return False
    if updated_after is not None and row["updated_at"] < updated_after:
        return False
    if updated_before is not None and row["updated_at"] >= updated_before:
        return False
    return True


def search(
    query: str,
    doc_type: str,
    *,
    ticket_id: str | None = None,
    updated_after: str | None = None,
    updated_before: str | None = None,
    k: int = DEFAULT_K,
    top_n: int = DEFAULT_TOP_N,
    directory: Path = FAISS_INDEX_DIR,
) -> list[SearchResult]:
    """Embed query, search doc_type's index for the k nearest chunks, filter, keep top_n.

    doc_type picks which of AGENTS-37's per-type indexes to search. A type
    with no index built yet (nothing of that kind indexed so far) returns an
    empty list, the same "nothing here yet, not an error" treatment
    chunk_page() gives an empty body.

    ticket_id/updated_after/updated_before are applied here, in Python, to
    the k candidates FAISS returns, because FAISS cannot express them as part
    of the search itself. They can only narrow the k candidates down, never
    widen them -- see the module docstring on why a short result is accepted
    rather than compensated for.
    """
    if doc_type not in available_doc_types(directory):
        return []

    index, metadata = load_index(doc_type, directory=directory)
    if index.ntotal == 0:
        return []

    query_vector = embed_texts([query])
    scores, positions = index.search(query_vector, min(k, index.ntotal))

    results = []
    for score, position in zip(scores[0], positions[0], strict=True):
        row = metadata[position]
        if _passes_filters(row, ticket_id, updated_after, updated_before):
            results.append(
                SearchResult(
                    score=float(score),
                    text=row["text"],
                    doc_id=row["doc_id"],
                    doc_type=row["doc_type"],
                    space_key=row["space_key"],
                    page_id=row["page_id"],
                    version=row["version"],
                    ticket_ids=row["ticket_ids"],
                    author_id=row["author_id"],
                    updated_at=row["updated_at"],
                )
            )
    return results[:top_n]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: python -m app.retrieval "<query>" <doc_type>')
        raise SystemExit(1)
    cli_query, cli_doc_type = sys.argv[1], sys.argv[2]

    cli_results = search(cli_query, cli_doc_type)
    if not cli_results:
        print(f"No results in {cli_doc_type!r} (no index, empty index, or filtered out).")
    for cli_result in cli_results:
        print(f"{cli_result.score:.3f}  {cli_result.doc_id}  {cli_result.text[:100]!r}")