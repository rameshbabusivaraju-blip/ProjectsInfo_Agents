"""Tests for app/freshness.py: when each source was last copied, and whether that is too old."""

from datetime import datetime
from pathlib import Path

import pytest
from sqlmodel import Session, SQLModel, create_engine

from app import agent, freshness, tools
from app.models import Sprint, Ticket, TicketSprint

NOW = datetime(2026, 10, 9, 12, 0, 0)


def _new_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    """An empty temp database that the agent and the clock helper both use."""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(agent, "DB_PATH", str(db_path))
    monkeypatch.setattr(freshness, "_now", lambda: NOW)
    return Session(engine)


def test_every_metric_key_has_a_source() -> None:
    """A new query cannot be added without saying which source it reads."""
    assert set(freshness.SOURCE_FOR_METRIC) == set(agent._QUERY_MAP)


def test_a_source_that_was_never_synced_is_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _new_db(tmp_path, monkeypatch):
        pass

    result = freshness.get_freshness(str(tmp_path / "test.db"), "jira")

    assert result == {"source": "jira", "last_synced_at": None, "age_hours": None, "stale": True}


def test_a_database_without_the_table_counts_as_never_synced(tmp_path: Path) -> None:
    result = freshness.get_freshness(str(tmp_path / "old.db"), "github")

    assert result["last_synced_at"] is None
    assert result["stale"] is True


def test_a_recent_sync_is_fresh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with _new_db(tmp_path, monkeypatch) as session:
        freshness.record_sync(session, "jira")
        session.commit()
    monkeypatch.setattr(freshness, "_now", lambda: datetime(2026, 10, 9, 18, 30, 0))

    result = freshness.get_freshness(str(tmp_path / "test.db"), "jira")

    assert result == {
        "source": "jira",
        "last_synced_at": "2026-10-09T12:00:00",
        "age_hours": 6.5,
        "stale": False,
    }


def test_a_sync_older_than_the_limit_is_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _new_db(tmp_path, monkeypatch) as session:
        freshness.record_sync(session, "jira")
        session.commit()
    monkeypatch.setattr(freshness, "_now", lambda: datetime(2026, 10, 10, 13, 0, 0))

    result = freshness.get_freshness(str(tmp_path / "test.db"), "jira")

    assert result["age_hours"] == 25.0
    assert result["stale"] is True


def test_exactly_at_the_limit_is_not_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _new_db(tmp_path, monkeypatch) as session:
        freshness.record_sync(session, "jira")
        session.commit()
    monkeypatch.setattr(freshness, "_now", lambda: datetime(2026, 10, 10, 12, 0, 0))

    assert freshness.get_freshness(str(tmp_path / "test.db"), "jira")["stale"] is False


def test_recording_again_replaces_the_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _new_db(tmp_path, monkeypatch) as session:
        freshness.record_sync(session, "github")
        session.commit()
        monkeypatch.setattr(freshness, "_now", lambda: datetime(2026, 10, 9, 15, 0, 0))
        freshness.record_sync(session, "github")
        session.commit()

    result = freshness.get_freshness(str(tmp_path / "test.db"), "github")

    assert result["last_synced_at"] == "2026-10-09T15:00:00"


def test_get_metric_reports_the_age_of_its_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """velocity reads Jira data, so the result carries the Jira sync age."""
    with _new_db(tmp_path, monkeypatch) as session:
        session.add(Sprint(id=1, name="Sprint 1", state="closed", start_date=datetime(2026, 1, 1)))
        session.add(
            Ticket(
                key="AGENTS-1", summary="t", issue_type="Task", hierarchy_level=0,
                status="Done", status_category="done", story_points=5.0,
                created=datetime(2026, 1, 1), updated=datetime(2026, 1, 2),
            )
        )
        session.add(TicketSprint(ticket_key="AGENTS-1", sprint_id=1, position=0))
        freshness.record_sync(session, "jira")
        session.commit()

    result = tools.get_metric("velocity")

    assert result["freshness"]["source"] == "jira"
    assert result["freshness"]["age_hours"] == 0.0
    assert result["freshness"]["stale"] is False


def test_get_metric_without_a_sync_record_says_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _new_db(tmp_path, monkeypatch):
        pass

    result = tools.get_metric("unreviewed_prs")

    assert result["freshness"]["source"] == "github"
    assert result["freshness"]["stale"] is True
    assert result["rows"] == []