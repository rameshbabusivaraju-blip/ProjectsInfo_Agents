"""Tests for the LangGraph agent's metric routing — AGENTS-30.

Confirms _QUERY_MAP points each known metric_key at its own reviewed SQL
constant, and that metric_path() actually runs that SQL and returns the rows
it produces. Fully offline: a temp SQLite file stands in for projectpulse.db,
monkeypatched onto agent.DB_PATH for the duration of each test — no live Jira
or GitHub call, same spirit as test_provider.py.
"""

from datetime import datetime
from pathlib import Path

import pytest
from openpyxl import load_workbook
from sqlmodel import Session, SQLModel, create_engine

from app import agent, excel_export
from app.github_connector import REVIEW_TURNAROUND_SQL, UNREVIEWED_PRS_SQL
from app.jira_connector import VELOCITY_SQL
from app.models import PrReview, PullRequest, Sprint, Ticket, TicketSprint

# ---------------------------------------------------------------------------
# _QUERY_MAP — each metric_key must point at its own reviewed SQL constant
# ---------------------------------------------------------------------------


def test_query_map_velocity_uses_velocity_sql() -> None:
    """velocity must run jira_connector's own VELOCITY_SQL, not a copy of it."""
    assert agent._QUERY_MAP["velocity"] is VELOCITY_SQL


def test_query_map_review_turnaround_uses_review_turnaround_sql() -> None:
    """review_turnaround must run github_connector's own REVIEW_TURNAROUND_SQL."""
    assert agent._QUERY_MAP["review_turnaround"] is REVIEW_TURNAROUND_SQL


def test_query_map_unreviewed_prs_uses_unreviewed_prs_sql() -> None:
    """unreviewed_prs must run github_connector's own UNREVIEWED_PRS_SQL."""
    assert agent._QUERY_MAP["unreviewed_prs"] is UNREVIEWED_PRS_SQL


# ---------------------------------------------------------------------------
# metric_path() — must run the mapped SQL against the database and return rows
# ---------------------------------------------------------------------------


def test_metric_path_runs_velocity_sql(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A ticket done in its most recent sprint must count toward that sprint's velocity."""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        session.add(
            Sprint(id=1, name="Sprint 1", state="closed", start_date=datetime(2026, 1, 1))
        )
        session.add(
            Ticket(
                key="AGENTS-1",
                summary="Test ticket",
                issue_type="Task",
                hierarchy_level=0,
                status="Done",
                status_category="done",
                story_points=5.0,
                created=datetime(2026, 1, 1),
                updated=datetime(2026, 1, 2),
            )
        )
        session.add(TicketSprint(ticket_key="AGENTS-1", sprint_id=1, position=0))
        session.commit()

    monkeypatch.setattr(agent, "DB_PATH", str(db_path))

    state = agent.metric_path({"metric_key": "velocity"})

    assert state["rows"] == [{"sprint": "Sprint 1", "delivered": 5.0}]


def test_metric_path_runs_review_turnaround_sql(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The gap between a PR's creation and its first review must come back in hours."""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        session.add(
            PullRequest(
                number=1,
                title="Add thing",
                state="closed",
                created_at=datetime(2026, 1, 1, 9, 0),
                merged_at=datetime(2026, 1, 1, 15, 0),
                head_branch="AGENTS-1-add-thing",
            )
        )
        session.add(
            PrReview(
                id=1,
                pr_number=1,
                state="APPROVED",
                submitted_at=datetime(2026, 1, 1, 12, 0),
            )
        )
        session.commit()

    monkeypatch.setattr(agent, "DB_PATH", str(db_path))

    state = agent.metric_path({"metric_key": "review_turnaround"})

    assert state["rows"] == [{"avg_hours": 3.0}]


def test_metric_path_runs_unreviewed_prs_sql(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A PR merged with no review row at all must show up as unreviewed."""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        session.add(
            PullRequest(
                number=2,
                title="Unreviewed change",
                state="closed",
                created_at=datetime(2026, 1, 2, 9, 0),
                merged_at=datetime(2026, 1, 2, 10, 0),
                head_branch="AGENTS-2-unreviewed-change",
            )
        )
        session.commit()

    monkeypatch.setattr(agent, "DB_PATH", str(db_path))

    state = agent.metric_path({"metric_key": "unreviewed_prs"})

    assert state["rows"] == [{"number": 2, "title": "Unreviewed change"}]


# ---------------------------------------------------------------------------
# export_path() — must pick the most recent sprint and write it to Excel
# ---------------------------------------------------------------------------


def test_export_path_writes_last_sprint_to_excel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """export_path must pick the most recent sprint by start_date, not any of them."""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        session.add(
            Sprint(id=1, name="Sprint 1", state="closed", start_date=datetime(2026, 1, 1))
        )
        session.add(
            Sprint(id=2, name="Sprint 2", state="closed", start_date=datetime(2026, 1, 15))
        )
        session.add(
            Ticket(
                key="AGENTS-1",
                summary="First ticket",
                issue_type="Task",
                hierarchy_level=0,
                status="Done",
                status_category="done",
                story_points=5.0,
                created=datetime(2026, 1, 1),
                updated=datetime(2026, 1, 2),
            )
        )
        session.add(
            Ticket(
                key="AGENTS-2",
                summary="Second ticket",
                issue_type="Task",
                hierarchy_level=0,
                status="Done",
                status_category="done",
                story_points=8.0,
                created=datetime(2026, 1, 15),
                updated=datetime(2026, 1, 16),
            )
        )
        session.add(TicketSprint(ticket_key="AGENTS-1", sprint_id=1, position=0))
        session.add(TicketSprint(ticket_key="AGENTS-2", sprint_id=2, position=0))
        session.commit()

    monkeypatch.setattr(agent, "DB_PATH", str(db_path))
    monkeypatch.setattr(excel_export, "EXPORTS_DIR", tmp_path)

    state = agent.export_path({"metric_key": "export_excel"})

    assert state["rows"] == [{"sprint": "Sprint 2", "delivered": 8.0}]
    assert state["answer"] == f"Wrote 1 row(s) to {tmp_path / 'last_sprint_velocity.xlsx'}."

    workbook = load_workbook(tmp_path / "last_sprint_velocity.xlsx")
    worksheet = workbook.active
    assert worksheet is not None
    assert [cell.value for cell in worksheet[2]] == ["Sprint 2", 8.0]