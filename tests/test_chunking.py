"""Tests for the Confluence chunking pipeline — AGENTS-36.

Covers chunk_page() and _infer_doc_type() only — both are pure functions over
a ConfluencePage instance built in memory, no database involved. chunk_all_pages()
just reads every row and calls chunk_page() on each; it isn't exercised here,
same reasoning the project already applies to confluence_connector.load() and
github_connector.load(): a thin DB-reading wrapper around logic that is tested
directly, not worth re-testing through a database.
"""

from datetime import datetime

from app.chunking import (
    _DEFAULT_DOC_TYPE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    Chunk,
    _infer_doc_type,
    chunk_page,
)
from app.models import ConfluencePage


def _page(**overrides: object) -> ConfluencePage:
    """A ConfluencePage with sane defaults, so each test only states what it cares about."""
    defaults: dict[str, object] = {
        "id": "123",
        "title": "01 Project Charter",
        "space_key": "ConProK",
        "url": "https://example.atlassian.net/wiki/spaces/ConProK/pages/123",
        "version": 3,
        "version_author_id": "abc-author",
        "updated": datetime(2026, 9, 10, 12, 0, 0),
        "body_text": "Some page content.",
    }
    defaults.update(overrides)
    return ConfluencePage(**defaults)


# ---------------------------------------------------------------------------
# _infer_doc_type — title -> doc_type, always something, never None
# ---------------------------------------------------------------------------


def test_infer_doc_type_matches_every_known_project_page() -> None:
    """Checked against the titles actually in the project's Confluence space."""
    assert _infer_doc_type("01 Project Charter") == "charter"
    assert _infer_doc_type("02 Question Catalogue") == "question_catalogue"
    assert _infer_doc_type("03 Decision Log") == "decision_log"
    assert _infer_doc_type("04 Plan of Action") == "plan_of_action"
    assert _infer_doc_type("05 PM Notes — Phase 1") == "pm_notes"
    assert _infer_doc_type("Sprint 2") == "sprint_summary"


def test_infer_doc_type_matches_newly_added_raid_log_rule() -> None:
    """Added to doc_type_rules.json without touching chunking.py -- the scenario
    this JSON file exists for: a new Confluence page type given its own specific
    label by editing the JSON file alone (see the comment over _DEFAULT_DOC_TYPE).
    """
    assert _infer_doc_type("RAID Log — ProjectPulse") == "raid_log"


def test_infer_doc_type_prefers_retro_over_sprint() -> None:
    """A retro page's title also contains the word 'Sprint' — retro must win, not sprint_summary."""
    assert _infer_doc_type("Retro — SCRUM Sprint 1") == "retro"


def test_infer_doc_type_falls_back_to_general_for_an_unmatched_title() -> None:
    """A template page, the space's own landing page, or anything not yet given its
    own rule -- still classified, still indexed, just under _DEFAULT_DOC_TYPE.
    """
    assert _infer_doc_type("Template - Project plan") == _DEFAULT_DOC_TYPE
    assert _infer_doc_type("ProjectsInfo_Agents") == _DEFAULT_DOC_TYPE


# ---------------------------------------------------------------------------
# chunk_page — splitting, metadata propagation, ticket_ids, empty bodies
# ---------------------------------------------------------------------------


def test_chunk_page_returns_empty_list_for_blank_body() -> None:
    """No text, no chunks — not an error. Covers None, empty string and whitespace-only."""
    assert chunk_page(_page(body_text=None)) == []
    assert chunk_page(_page(body_text="")) == []
    assert chunk_page(_page(body_text="   \n  ")) == []


def test_chunk_page_still_chunks_a_page_with_no_matching_rule() -> None:
    """A title matching no row in doc_type_rules.json is still indexed -- every
    page added to Confluence must be searchable automatically, with no code
    change required. It just gets _DEFAULT_DOC_TYPE instead of a specific type.
    See the comment above _DEFAULT_DOC_TYPE in chunking.py.
    """
    page = _page(title="Some Unrelated Page Nobody Added Yet", body_text="Plenty of real content.")
    (chunk,) = chunk_page(page)
    assert chunk.doc_type == _DEFAULT_DOC_TYPE
    assert chunk.text == "Plenty of real content."


def test_chunk_page_splits_long_text_into_multiple_chunks() -> None:
    """Text well past CHUNK_SIZE must come back as more than one chunk."""
    long_text = "This is one sentence of the page. " * 100  # well over 1,000 chars
    chunks = chunk_page(_page(body_text=long_text))

    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(len(c.text) <= CHUNK_SIZE + CHUNK_OVERLAP for c in chunks)


def test_chunk_page_keeps_short_text_as_a_single_chunk() -> None:
    """Well under CHUNK_SIZE — nothing to split."""
    chunks = chunk_page(_page(body_text="A short page with one paragraph."))
    assert len(chunks) == 1


def test_chunk_page_carries_page_metadata_onto_every_chunk() -> None:
    """doc_id, space_key, page_id, version, author_id, updated_at all trace back to the page."""
    page = _page(
        id="456",
        space_key="ConProK",
        version=7,
        version_author_id="author-xyz",
        updated=datetime(2026, 9, 20, 8, 30),
        body_text="Short enough to be one chunk.",
    )
    (chunk,) = chunk_page(page)

    assert chunk.doc_id == "confluence:456"
    assert chunk.page_id == "456"
    assert chunk.space_key == "ConProK"
    assert chunk.version == 7
    assert chunk.author_id == "author-xyz"
    assert chunk.updated_at == "2026-09-20T08:30:00"


def test_chunk_page_handles_missing_author() -> None:
    """A brand-new page can have no resolved author yet — must not raise."""
    (chunk,) = chunk_page(_page(version_author_id=None, body_text="Short page."))
    assert chunk.author_id is None


def test_chunk_page_extracts_ticket_ids_mentioned_in_the_chunk() -> None:
    """Tolerates the same AGENTS-nn / Agents nn variations github_connector does."""
    text = "Discussed the delay on AGENTS-14 and on Agents 22 during the retro."
    (chunk,) = chunk_page(_page(body_text=text))
    assert chunk.ticket_ids == ["AGENTS-14", "AGENTS-22"]


def test_chunk_page_ticket_ids_empty_when_none_mentioned() -> None:
    """Most chunks won't mention a ticket at all — that's the common case, not an edge case."""
    (chunk,) = chunk_page(_page(body_text="General background with no ticket reference."))
    assert chunk.ticket_ids == []


def test_chunk_page_infers_doc_type_once_for_the_whole_page() -> None:
    """doc_type comes from the page title, so every chunk of one page must agree."""
    long_text = "Decision context and consequences. " * 100
    chunks = chunk_page(_page(title="03 Decision Log", body_text=long_text))
    assert len(chunks) > 1
    assert all(c.doc_type == "decision_log" for c in chunks)