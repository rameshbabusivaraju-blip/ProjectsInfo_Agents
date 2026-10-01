"""Tests for jira_connector.load() — AGENTS-48.

load() used to drop every table in the database, which silently deleted the
GitHub and Confluence rows loaded before it. These tests confirm it now clears
only the four Jira tables. Fully offline: a temp SQLite file replaces
projectpulse.db and the three fetch functions return canned data.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine, select

from app import jira_connector
from app.models import Commit, ConfluencePage, Person, Sprint, Ticket, WorkLog

_USER = {"accountId": "acc-1", "displayName": "Ramesh"}


def _issue(key: str) -> dict[str, Any]:
    """A minimal Jira issue payload with just the fields load() reads."""
    return {
        "key": key,
        "fields": {
            "summary": "Test issue",
            "issuetype": {"name": "Task", "hierarchyLevel": 0},
            "status": {"name": "Done"},
            "statusCategory": {"key": "done"},
            "created": "2026-09-01T10:00:00.000+0530",
            "updated": "2026-09-02T10:00:00.000+0530",
            "assignee": _USER,
            "customfield_10020": [{"id": 1}],
        },
    }


@pytest.fixture()
def engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Engine:
    """A temp database seeded with GitHub, Confluence and old Jira rows."""
    test_engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    SQLModel.metadata.create_all(test_engine)

    with Session(test_engine) as session:
        session.add(Commit(sha="abc", message="AGENTS-1 x", authored_at=datetime(2026, 9, 1)))
        session.add(Person(account_id="acc-1", display_name="acc-1"))
        session.add(
            ConfluencePage(
                id="p1",
                title="Page",
                space_key="ConProK",
                url="http://x",
                version=1,
                version_author_id="acc-1",
                updated=datetime(2026, 9, 1),
            )
        )
        session.add(Sprint(id=99, name="Old sprint", state="closed"))
        session.commit()

    monkeypatch.setattr(jira_connector, "engine", test_engine)
    monkeypatch.setattr(
        jira_connector,
        "fetch_sprints",
        lambda: [{"id": 1, "name": "SCRUM Sprint 1", "state": "active"}],
    )
    monkeypatch.setattr(jira_connector, "fetch_issues", lambda: [_issue("AGENTS-2")])
    monkeypatch.setattr(
        jira_connector,
        "fetch_worklogs",
        lambda key: [
            {
                "id": "w1",
                "author": _USER,
                "started": "2026-09-02T09:00:00.000+0530",
                "timeSpentSeconds": 3600,
            }
        ],
    )
    return test_engine


def test_load_keeps_github_and_confluence_rows(engine: Engine) -> None:
    """Commits and Confluence pages must survive a Jira reload."""
    jira_connector.load()

    with Session(engine) as session:
        assert len(session.exec(select(Commit)).all()) == 1
        assert len(session.exec(select(ConfluencePage)).all()) == 1


def test_load_replaces_jira_rows_and_updates_people(engine: Engine) -> None:
    """Old Jira rows go, new ones arrive, and the shared person is updated in place."""
    jira_connector.load()

    with Session(engine) as session:
        assert [s.name for s in session.exec(select(Sprint)).all()] == ["SCRUM Sprint 1"]
        assert [t.key for t in session.exec(select(Ticket)).all()] == ["AGENTS-2"]
        assert len(session.exec(select(WorkLog)).all()) == 1

        # The Confluence placeholder name is replaced by the real Jira name.
        person = session.get(Person, "acc-1")
        assert person is not None
        assert person.display_name == "Ramesh"