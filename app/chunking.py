"""Split Confluence pages into embedding-sized chunks — AGENTS-36, charter section 2b.

Turns each stored ConfluencePage.body_text (AGENTS-34) into a list of Chunks,
each small enough to embed usefully (AGENTS-37) and each carrying the
metadata the question catalogue's schema specifies, so a chunk retrieved
later can be traced back to its page and filtered on sprint, ticket or date
(AGENTS-38).

Settles decision-log item O1: fixed-size chunks, 1,000 characters with 200
characters of overlap, via LangChain's RecursiveCharacterTextSplitter — that
is charter section 2b's own spec, not a new choice made here. The
alternative, one chunk per heading section, was not built: ProjectPulse's
own pages are short enough that heading-aware splitting would rarely change
the result, and fixed-size chunking is the simpler mechanism for it.

Inclusion policy: every page in the Confluence space is chunked and indexed,
with no code change required when a new page is added. doc_type_rules.json
only refines which specific type a page is tagged with for filtering
(AGENTS-38) -- it is not a gate on whether a page is searchable at all. See
_DEFAULT_DOC_TYPE below for the one case that changed this from an earlier
version of this file, where an unmatched title was excluded outright.

Run it with:  python -m app.chunking
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlmodel import Session, select

from app.db import engine, init_db
from app.github_connector import TICKET_PATTERN
from app.models import ConfluencePage

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Title rules -> doc_type live in doc_type_rules.json (same folder as this file), not in a
# Python literal here -- see _load_doc_type_rules() below. First match wins. These rules
# only refine WHICH type a page gets; they do not gate whether it's indexed at all -- every
# page in the space is indexed regardless (see _DEFAULT_DOC_TYPE just below). Giving a new
# kind of page its own specific type is still a JSON-only edit: add one row, no change to
# this module. If the space grows large enough that per-row rules get tedious even as a
# nice-to-have, the next lever is a Confluence label (e.g. "rag-type: xyz") fetched by the
# connector -- not built here.
_DOC_TYPE_RULES_PATH = Path(__file__).parent / "doc_type_rules.json"

# What a page gets when no row above matches its title. Not an error and not "excluded" --
# every page added to Confluence must end up searchable with no code change required, so
# there is no longer an "out of scope for RAG" outcome, only "no specific label for this one
# yet." A specific doc_type is a bonus for AGENTS-38's filtering, never a precondition for
# being in RAG.
_DEFAULT_DOC_TYPE = "general"


def _load_doc_type_rules() -> list[tuple[str, str]]:
    """Read the title -> doc_type allow-list from doc_type_rules.json.

    A JSON file rather than a Python literal, so a rule can be added or removed
    by editing that file alone -- nothing in this module has to change. Order is
    preserved from the file (JSON arrays are ordered), which matters because
    _infer_doc_type() below stops at the first match.
    """
    with _DOC_TYPE_RULES_PATH.open(encoding="utf-8") as f:
        rows = json.load(f)  # [{"match": "...", "doc_type": "..."}, ...], in file order
    return [(row["match"], row["doc_type"]) for row in rows]


_DOC_TYPE_RULES: list[tuple[str, str]] = _load_doc_type_rules()  # read once, at import time


def _infer_doc_type(title: str) -> str:
    """Classify a page by its title. Always returns something -- never None.

    Tries every row loaded from doc_type_rules.json in order and returns the
    first match. A title matching nothing gets _DEFAULT_DOC_TYPE rather than
    being treated as out of scope: there is no "exclude this page" outcome
    any more, only "no specific label for it yet." See the comment above
    _DEFAULT_DOC_TYPE.
    """
    lowered = title.lower()  # case-fold once, e.g. "01 Project Charter" -> "01 project charter"
    for needle, doc_type in _DOC_TYPE_RULES:  # walk rules top to bottom, in order -- first hit wins
        if needle in lowered:  # substring test, e.g. "project charter" in "01 project charter"
            return doc_type  # matched -> this page's specific type
    return _DEFAULT_DOC_TYPE  # matched nothing -> still indexed, just without a specific label


@dataclass(frozen=True)
class Chunk:
    """One embeddable piece of a Confluence page, carrying what AGENTS-38 filters on."""

    text: str  # the chunk's own text -- this is what gets embedded
    doc_id: str  # e.g. "confluence:456" -- stable page id, shared by every chunk of that page
    doc_type: str  # e.g. "decision_log" -- same for every chunk of a page (see chunk_page)
    space_key: str  # Confluence space the page lives in, e.g. "ConProK"
    page_id: str  # the page's own Confluence id, e.g. "456"
    version: int  # the page's Confluence version number at chunking time
    ticket_ids: list[str]  # tickets mentioned in THIS chunk specifically -- not the whole page
    author_id: str | None  # who last edited this page version; None if unresolved
    updated_at: str  # ISO-8601 timestamp string, e.g. "2026-09-20T08:30:00"


def chunk_page(page: ConfluencePage) -> list[Chunk]:
    """Split one page's body_text into Chunks, each carrying the page's metadata.

    Every page with real text is chunked and indexed -- the only reason a
    page produces no chunks is an empty body (fetch failed, or the page is
    genuinely empty so far), which is an ordinary, growing state of a live
    Confluence space, not an error. A page whose title matches no row in
    doc_type_rules.json is still fully indexed, just under _DEFAULT_DOC_TYPE
    instead of a specific type.

    ticket_ids is computed per chunk, not inherited whole from the page: a
    page can mention AGENTS-14 in its intro and AGENTS-22 three sections
    later, and tagging every chunk with both would make AGENTS-38's
    ticket-scoped filtering wrong for the chunks that aren't actually about
    either one.
    """
    doc_type = _infer_doc_type(page.title)  # always succeeds -- see _infer_doc_type

    # The one remaining gate: does the page actually have text? Covers None, "" and
    # whitespace-only bodies -- there is nothing to embed either way.
    if not page.body_text or not page.body_text.strip():
        return []

    # LangChain's splitter cuts body_text into <=1000-char pieces (CHUNK_SIZE), each overlapping
    # the previous piece by 200 chars (CHUNK_OVERLAP) so a sentence split at a boundary isn't lost.
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    chunks = []
    for piece in splitter.split_text(page.body_text):  # one loop iteration per chunk of THIS page
        # Ticket refs found in just this piece, not the whole page -- see the docstring above.
        # TICKET_PATTERN (shared with github_connector) matches "AGENTS-14" and "Agents 22" alike;
        # the set{} dedupes repeats within one chunk, sorted() makes the output order deterministic.
        ticket_ids = sorted(
            {f"AGENTS-{n}" for n in TICKET_PATTERN.findall(piece)},
            key=lambda key: int(key.split("-")[1]),  # numeric sort: AGENTS-2 before AGENTS-14
        )
        chunks.append(
            Chunk(
                text=piece,
                doc_id=f"confluence:{page.id}",  # identical for every chunk of this page
                doc_type=doc_type,  # identical for every chunk of this page -- computed once, above
                space_key=page.space_key,
                page_id=page.id,
                version=page.version,
                ticket_ids=ticket_ids,  # differs per chunk -- see the comment above
                author_id=page.version_author_id,
                updated_at=page.updated.isoformat(),  # datetime -> ISO-8601 string
            )
        )
    return chunks


def _all_pages() -> list[ConfluencePage]:
    """Every stored Confluence page, unfiltered — chunk_all_pages() and __main__ share this.

    A plain full-table read, so this uses SQLModel's select() rather than the
    hand-written reviewed SQL every other read in this app uses — that
    convention is for computing a metric's answer deterministically; this
    isn't computing anything, only fetching rows there is nothing to review.
    """
    init_db()  # creates tables on a fresh DB file; a no-op if they already exist
    with Session(engine) as session:  # opens a DB session, closed automatically on exit
        return list(session.exec(select(ConfluencePage)).all())  # SELECT * FROM confluencepage


def chunk_all_pages() -> list[Chunk]:
    """Read every stored Confluence page and chunk it — the input AGENTS-37 embeds."""
    # For every stored page, run chunk_page() and flatten all the per-page lists into one.
    # A skipped page just contributes an empty list here -- no special-casing needed.
    return [chunk for page in _all_pages() for chunk in chunk_page(page)]


# Manual smoke-test entry point: `python -m app.chunking` prints a summary against whatever
# is currently in the DB. Not part of the pipeline itself, and not exercised by the tests.
if __name__ == "__main__":
    all_pages = _all_pages()  # every stored page -- all of them get chunked below
    result = [chunk for page in all_pages for chunk in chunk_page(page)]
    empty = [
        page.title for page in all_pages if not page.body_text or not page.body_text.strip()
    ]  # the only pages chunk_page() actually skips
    unlabeled = [
        page.title for page in all_pages if _infer_doc_type(page.title) == _DEFAULT_DOC_TYPE
    ]  # indexed fine, but no row in doc_type_rules.json matched -- candidates for one

    by_type: dict[str, int] = {}
    for c in result:
        by_type[c.doc_type] = by_type.get(c.doc_type, 0) + 1  # tally chunk count per doc_type

    page_count = len({c.page_id for c in result})  # distinct pages (set{} dedupes page ids)
    print(f"{len(result)} chunk(s) from {page_count} page(s):")
    for doc_type, count in sorted(by_type.items()):
        print(f"  {doc_type}: {count}")

    if empty:
        print(f"\n{len(empty)} page(s) with no body text, skipped:")
        for title in empty:
            print(f"  {title!r}")  # !r -> quoted repr, so a blank title is still visible

    if unlabeled:
        print(f"\n{len(unlabeled)} page(s) indexed under '{_DEFAULT_DOC_TYPE}' (no rule matched):")
        for title in unlabeled:
            print(f"  {title!r}")