"""Tests for confluence_connector.load() (AGENTS-91). Fully offline: the API calls are replaced."""

from pathlib import Path
from typing import Any

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app import confluence_connector
from app.models import ConfluencePage, SyncStatus


def _page() -> dict[str, Any]:
    """One page as the Confluence v2 list endpoint returns it."""
    return {
        "id": "p1",
        "title": "RAID Log",
        "version": {"number": 2, "createdAt": "2026-10-09T08:00:00Z", "authorId": "acc-1"},
        "_links": {"webui": "/spaces/TEST/pages/p1"},
    }


def test_load_stores_the_pages_and_records_when_confluence_was_synced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The sync time is saved so the agent can tell how old the stored pages are."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(confluence_connector, "engine", engine)
    monkeypatch.setattr(confluence_connector, "init_db", lambda: None)
    monkeypatch.setattr(confluence_connector, "_space_id", lambda key: "1")
    monkeypatch.setattr(confluence_connector, "fetch_pages", lambda space_id: [_page()])
    monkeypatch.setattr(confluence_connector, "fetch_body_markdown", lambda page_id: "R2 unowned")

    confluence_connector.load()

    with Session(engine) as session:
        pages = session.exec(select(ConfluencePage)).all()
        assert [(page.title, page.body_text) for page in pages] == [("RAID Log", "R2 unowned")]
        assert [row.source for row in session.exec(select(SyncStatus)).all()] == ["confluence"]