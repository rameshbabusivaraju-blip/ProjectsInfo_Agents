"""Tests for the catalogue queries added in AGENTS-55 to AGENTS-60.

Each query is checked two ways: _QUERY_MAP must point at the connector's own SQL
constant, and metric_path() must run that SQL against a temp SQLite file and
return the right rows. Fully offline: no Jira, GitHub or model call.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlmodel import Session, SQLModel, create_engine

from app import agent
from app.github_connector import LONG_OPEN_PRS_SQL
from app.models import PullRequest


def _new_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    """Create an empty temp database, point the agent at it, and return an open session."""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(agent, "DB_PATH", str(db_path))
    return Session(engine)


# ---------------------------------------------------------------------------
# AGENTS-55 — C2: pull requests open for more than three days
# ---------------------------------------------------------------------------


def test_query_map_long_open_prs_uses_long_open_prs_sql() -> None:
    """long_open_prs must run github_connector's own LONG_OPEN_PRS_SQL, not a copy of it."""
    assert agent._QUERY_MAP["long_open_prs"] is LONG_OPEN_PRS_SQL


def test_metric_path_runs_long_open_prs_sql(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A PR counts when it was open for more than three days, still open or already merged."""
    now = datetime.now(UTC).replace(tzinfo=None)

    def pr(
        number: int, title: str, state: str, days_open: float, merged: bool = False
    ) -> PullRequest:
        """A PR that has been open for days_open days; merged ones were merged at that point."""
        created = now - timedelta(days=days_open)
        # A merged PR stays "open" only until its merge time, so end it exactly days_open later.
        end = created + timedelta(days=days_open) if merged else None
        return PullRequest(
            number=number,
            title=title,
            state=state,
            created_at=created,
            merged_at=end,
            closed_at=end,
            head_branch=f"AGENTS-{number}-x",
        )

    with _new_db(tmp_path, monkeypatch) as session:
        session.add(pr(1, "Still open, five days", "open", 5))
        session.add(pr(2, "Still open, one day", "open", 1))
        session.add(pr(3, "Merged after four days", "closed", 4, merged=True))
        session.add(pr(4, "Merged after two days", "closed", 2, merged=True))
        session.commit()

    state = agent.metric_path({"metric_key": "long_open_prs"})

    # Only the 5-day open PR and the PR merged after 4 days pass; the longest comes first.
    assert state["rows"] == [
        {"number": 1, "title": "Still open, five days", "state": "open", "days_open": 5.0},
        {"number": 3, "title": "Merged after four days", "state": "closed", "days_open": 4.0},
    ]


def test_refuse_path_names_prs_open_for_more_than_three_days() -> None:
    """The refusal text must list the new question so a user knows it can be asked."""
    state = agent.refuse_path({"question": "what is the weather"})

    assert "PRs open for more than three days" in state["answer"]
