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

Run it with:  python -m app.chunking
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlmodel import Session, select

from app.db import engine, init_db
from app.github_connector import TICKET_PATTERN
from app.models import ConfluencePage

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Ordered title rules -> doc_type. First match wins. Checked against every
# page actually in the project's Confluence space at the time this was
# written: the first five rows below account for "01 Project Charter" through
# "05 PM Notes -- Phase 1"; "retro" and "sprint " cover the two sprint-ceremony
# pages. Nothing here is the catalogue's original seven-type list verbatim --
# real page titles are "Plan of Action" and "PM Notes", not the catalogue's
# hypothetical ones, so the list follows what is actually there.
_DOC_TYPE_RULES: list[tuple[str, str]] = [
    ("project charter", "charter"),
    ("question catalogue", "question_catalogue"),
    ("decision log", "decision_log"),
    ("plan of action", "plan_of_action"),
    ("pm notes", "pm_notes"),
    ("retro", "retro"),
    ("sprint ", "sprint_summary"),
]


def _infer_doc_type(title: str) -> str:
    """Classify a page by its title — ADR-003's first workaround needs one type per page.

    A small, explicit table rather than anything cleverer: the project's own
    Confluence space holds under a dozen pages. "other" is not a failure
    case, it is the honest answer for a page this table was never told
    about — a template page or the space's own landing page, say — and the
    space will keep growing pages this list does not yet know.
    """
    lowered = title.lower()
    for needle, doc_type in _DOC_TYPE_RULES:
        if needle in lowered:
            return doc_type
    return "other"


@dataclass(frozen=True)
class Chunk:
    """One embeddable piece of a Confluence page, carrying what AGENTS-38 filters on."""

    text: str
    doc_id: str
    doc_type: str
    space_key: str
    page_id: str
    version: int
    ticket_ids: list[str]
    author_id: str | None
    updated_at: str


def chunk_page(page: ConfluencePage) -> list[Chunk]:
    """Split one page's body_text into Chunks, each carrying the page's metadata.

    ticket_ids is computed per chunk, not inherited whole from the page: a
    page can mention AGENTS-14 in its intro and AGENTS-22 three sections
    later, and tagging every chunk with both would make AGENTS-38's
    ticket-scoped filtering wrong for the chunks that aren't actually about
    either one.

    A page with no body_text yet (fetch failed, or the page is genuinely
    empty) produces no chunks rather than an error — there is nothing to
    embed, and a growing Confluence space will always have a page or two
    like this.
    """
    if not page.body_text or not page.body_text.strip():
        return []

    doc_type = _infer_doc_type(page.title)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    chunks = []
    for piece in splitter.split_text(page.body_text):
        ticket_ids = sorted(
            {f"AGENTS-{n}" for n in TICKET_PATTERN.findall(piece)},
            key=lambda key: int(key.split("-")[1]),
        )
        chunks.append(
            Chunk(
                text=piece,
                doc_id=f"confluence:{page.id}",
                doc_type=doc_type,
                space_key=page.space_key,
                page_id=page.id,
                version=page.version,
                ticket_ids=ticket_ids,
                author_id=page.version_author_id,
                updated_at=page.updated.isoformat(),
            )
        )
    return chunks


def chunk_all_pages() -> list[Chunk]:
    """Read every stored Confluence page and chunk it — the input AGENTS-37 embeds.

    A plain full-table read, so this uses SQLModel's select() rather than the
    hand-written reviewed SQL every other read in this app uses — that
    convention is for computing a metric's answer deterministically; this
    isn't computing anything, only fetching rows there is nothing to review.
    """
    init_db()
    with Session(engine) as session:
        pages = session.exec(select(ConfluencePage)).all()
    return [chunk for page in pages for chunk in chunk_page(page)]


if __name__ == "__main__":
    result = chunk_all_pages()
    by_type: dict[str, int] = {}
    for c in result:
        by_type[c.doc_type] = by_type.get(c.doc_type, 0) + 1

    page_count = len({c.page_id for c in result})
    print(f"{len(result)} chunk(s) from {page_count} page(s):")
    for doc_type, count in sorted(by_type.items()):
        print(f"  {doc_type}: {count}")