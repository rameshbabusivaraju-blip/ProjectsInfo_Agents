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

from app import agent, excel_export, retrieval
from app.github_connector import (
    LONGEST_REVIEW_WAIT_SQL,
    REVIEW_TURNAROUND_SQL,
    UNREVIEWED_PRS_SQL,
)
from app.jira_connector import COMMITTED_VS_DELIVERED_SQL, VELOCITY_SQL
from app.models import PrReview, PullRequest, Sprint, Ticket, TicketSprint

# ---------------------------------------------------------------------------
# _QUERY_MAP — each metric_key must point at its own reviewed SQL constant
# ---------------------------------------------------------------------------


def test_query_map_velocity_uses_velocity_sql() -> None:
    """velocity must run jira_connector's own VELOCITY_SQL, not a copy of it."""
    assert agent._QUERY_MAP["velocity"] is VELOCITY_SQL

def test_query_map_committed_vs_delivered_uses_committed_vs_delivered_sql() -> None:
    """committed_vs_delivered must run jira_connector's own COMMITTED_VS_DELIVERED_SQL."""
    assert agent._QUERY_MAP["committed_vs_delivered"] is COMMITTED_VS_DELIVERED_SQL

def test_query_map_review_turnaround_uses_review_turnaround_sql() -> None:
    """review_turnaround must run github_connector's own REVIEW_TURNAROUND_SQL."""
    assert agent._QUERY_MAP["review_turnaround"] is REVIEW_TURNAROUND_SQL


def test_query_map_unreviewed_prs_uses_unreviewed_prs_sql() -> None:
    """unreviewed_prs must run github_connector's own UNREVIEWED_PRS_SQL."""
    assert agent._QUERY_MAP["unreviewed_prs"] is UNREVIEWED_PRS_SQL

def test_query_map_longest_review_wait_uses_longest_review_wait_sql() -> None:
    """longest_review_wait must run github_connector's own LONGEST_REVIEW_WAIT_SQL."""
    assert agent._QUERY_MAP["longest_review_wait"] is LONGEST_REVIEW_WAIT_SQL

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

def test_metric_path_runs_committed_vs_delivered_sql(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Committed counts tickets that existed at sprint start; delivered follows velocity."""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)

    def ticket(key: str, points: float, created: datetime, category: str) -> Ticket:
        return Ticket(
            key=key,
            summary=key,
            issue_type="Task",
            hierarchy_level=0,
            status="Done" if category == "done" else "To Do",
            status_category=category,
            story_points=points,
            created=created,
            updated=created,
        )

    with Session(engine) as session:
        session.add(Sprint(id=1, name="Sprint 1", state="closed", start_date=datetime(2026, 1, 10)))
        session.add(Sprint(id=2, name="Sprint 2", state="active", start_date=datetime(2026, 1, 20)))
        # A: existed at start, done in Sprint 1.   B: added mid-sprint, done in Sprint 1.
        # C: existed at start, not done, so it carries over from Sprint 1 into Sprint 2.
        session.add(ticket("AGENTS-1", 5.0, datetime(2026, 1, 1), "done"))
        session.add(ticket("AGENTS-2", 3.0, datetime(2026, 1, 12), "done"))
        session.add(ticket("AGENTS-3", 2.0, datetime(2026, 1, 1), "new"))
        session.add(TicketSprint(ticket_key="AGENTS-1", sprint_id=1, position=0))
        session.add(TicketSprint(ticket_key="AGENTS-2", sprint_id=1, position=0))
        session.add(TicketSprint(ticket_key="AGENTS-3", sprint_id=1, position=0))
        session.add(TicketSprint(ticket_key="AGENTS-3", sprint_id=2, position=1))
        session.commit()

    monkeypatch.setattr(agent, "DB_PATH", str(db_path))

    state = agent.metric_path({"metric_key": "committed_vs_delivered"})

    assert state["rows"] == [
        {"sprint": "Sprint 1", "committed": 7.0, "delivered": 8.0},
        {"sprint": "Sprint 2", "committed": 2.0, "delivered": 0.0},
    ]

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

def test_metric_path_runs_longest_review_wait_sql(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only the PR with the longest wait for its first review must come back."""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # PR 1 waited 3 hours; PR 2 waited 10 hours for its first review. PR 2's later
        # review must not matter, only the first one counts.
        for number, title in ((1, "Quick PR"), (2, "Slow PR")):
            session.add(
                PullRequest(
                    number=number,
                    title=title,
                    state="closed",
                    created_at=datetime(2026, 1, 1, 9, 0),
                    head_branch=f"AGENTS-{number}-x",
                )
            )
        session.add(
            PrReview(id=1, pr_number=1, state="APPROVED", submitted_at=datetime(2026, 1, 1, 12, 0))
        )
        session.add(
            PrReview(id=2, pr_number=2, state="COMMENTED", submitted_at=datetime(2026, 1, 1, 19, 0))
        )
        session.add(
            PrReview(id=3, pr_number=2, state="APPROVED", submitted_at=datetime(2026, 1, 2, 9, 0))
        )
        session.commit()

    monkeypatch.setattr(agent, "DB_PATH", str(db_path))

    state = agent.metric_path({"metric_key": "longest_review_wait"})

    assert state["rows"] == [{"number": 2, "title": "Slow PR", "wait_hours": 10.0}]

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


# ---------------------------------------------------------------------------
# route() — the narrative branch added by AGENTS-39
# ---------------------------------------------------------------------------


def test_route_sends_narrative_to_narrative_path() -> None:
    """A narrative classification must reach narrative_path, not fall through to metric."""
    assert agent.route({"metric_key": "narrative"}) == "narrative"


# ---------------------------------------------------------------------------
# narrative_path() — AGENTS-39: retrieval instead of SQL, same rows shape.
# search() itself is already covered by test_retrieval.py, so it is faked
# here rather than built against a real FAISS index -- this only has to
# prove narrative_path's own logic: turning SearchResults into rows, and
# answering honestly when nothing comes back.
# ---------------------------------------------------------------------------


def test_narrative_path_turns_search_results_into_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrieved chunks must land in state["rows"] in the same shape metric_path uses."""
    fake_result = retrieval.SearchResult(
        score=0.692,
        text="R2 | ... | Unassigned | Open",
        doc_id="confluence:10092545",
        doc_type="raid_log",
        space_key="ConProK",
        page_id="10092545",
        version=1,
        ticket_ids=[],
        author_id=None,
        updated_at="2026-09-25T00:00:00",
    )
    monkeypatch.setattr(agent, "search", lambda question, doc_type: [fake_result])

    state = agent.narrative_path({"question": "which risks have no owner", "doc_type": "raid_log"})

    assert state["rows"] == [
        {"text": fake_result.text, "doc_id": "confluence:10092545", "score": 0.692}
    ]
    assert "answer" not in state  # compose_answer still has to run on these rows


def test_narrative_path_answers_honestly_when_nothing_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty search results must not reach compose_answer with nothing to summarise."""
    monkeypatch.setattr(agent, "search", lambda question, doc_type: [])

    state = agent.narrative_path({"question": "anything", "doc_type": "retro"})

    assert state["rows"] == []
    assert state["answer"] == "No matching content found in the project's retro documents."

def test_compose_prompt_tells_model_to_admit_missing_data() -> None:
    """The compose prompt must forbid guessing when the rows do not answer the question."""
    messages = agent._COMPOSE_PROMPT.format_messages(question="Who won the cup?", rows=[])
    system_text = str(messages[0].content)

    assert "does not answer the question, say so plainly" in system_text
    assert "Never use general knowledge or guess" in system_text
    # The original rule about invented numbers must still be there.
    assert "Do not add any number that is not in the data" in system_text