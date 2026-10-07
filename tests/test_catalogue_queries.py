"""Tests for the catalogue queries added in AGENTS-55 to AGENTS-60.

Each query is checked two ways: _QUERY_MAP must point at the connector's own SQL
constant, and metric_path() must run that SQL against a temp SQLite file and
return the right rows. Fully offline: no Jira, GitHub or model call.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel
from sqlmodel import Session, SQLModel, create_engine

from app import agent
from app.github_connector import (
    COMMITS_FOR_TICKET_SQL,
    COMMITS_WITHOUT_TICKET_SQL,
    LONG_OPEN_PRS_SQL,
    NON_CONVENTION_BRANCHES_SQL,
)
from app.models import Commit, PullRequest


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


# ---------------------------------------------------------------------------
# AGENTS-56 — C3: commits for one ticket (the first query that takes a value)
# ---------------------------------------------------------------------------


class _FakeModel:
    """Stands in for a chat model: with_structured_output gives back one fixed classification."""

    def __init__(self, result: agent.Classification) -> None:
        """Remember the classification this fake model will always return."""
        self._result = result

    def with_structured_output(
        self, schema: type[BaseModel]
    ) -> RunnableLambda[Any, agent.Classification]:
        """Return a runnable that ignores its input and answers with the fixed classification."""

        def fixed(_: Any) -> agent.Classification:
            return self._result

        return RunnableLambda(fixed)


def _commit(letter: str, message: str, ticket_key: str | None, day: int) -> Commit:
    """A commit whose 40-character id is one repeated letter, made on the given January day."""
    return Commit(
        sha=letter * 40,
        message=message,
        author_login="dev",
        authored_at=datetime(2026, 1, day),
        ticket_key=ticket_key,
    )


def test_query_map_commits_for_ticket_uses_commits_for_ticket_sql() -> None:
    """commits_for_ticket must run github_connector's own COMMITS_FOR_TICKET_SQL."""
    assert agent._QUERY_MAP["commits_for_ticket"] is COMMITS_FOR_TICKET_SQL


def test_metric_path_runs_commits_for_ticket_sql(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only the named ticket's commits come back, oldest first, with one line of message."""
    with _new_db(tmp_path, monkeypatch) as session:
        session.add(_commit("a", "AGENTS-14 add thing\n\nCo-Authored-By: someone", "AGENTS-14", 2))
        session.add(_commit("b", "AGENTS-14 fix thing", "AGENTS-14", 1))
        session.add(_commit("c", "AGENTS-15 other work", "AGENTS-15", 3))
        session.commit()

    state = agent.metric_path({"metric_key": "commits_for_ticket", "ticket_key": "AGENTS-14"})

    assert state["rows"] == [
        {
            "sha": "bbbbbbb",
            "subject": "AGENTS-14 fix thing",
            "author_login": "dev",
            "authored_at": "2026-01-01 00:00:00.000000",
        },
        {
            "sha": "aaaaaaa",
            "subject": "AGENTS-14 add thing",
            "author_login": "dev",
            "authored_at": "2026-01-02 00:00:00.000000",
        },
    ]


def test_metric_path_treats_the_ticket_key_as_a_value_not_as_sql(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A ticket key shaped like SQL must match nothing, not change the query."""
    with _new_db(tmp_path, monkeypatch) as session:
        session.add(_commit("a", "AGENTS-14 add thing", "AGENTS-14", 1))
        session.commit()

    state = agent.metric_path(
        {"metric_key": "commits_for_ticket", "ticket_key": "AGENTS-14' OR '1'='1"}
    )

    assert state["rows"] == []


def test_metric_path_returns_no_rows_when_no_ticket_key_was_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A commits_for_ticket question with no ticket in it must return nothing, not every commit."""
    with _new_db(tmp_path, monkeypatch) as session:
        session.add(_commit("a", "AGENTS-14 add thing", "AGENTS-14", 1))
        session.commit()

    state = agent.metric_path({"metric_key": "commits_for_ticket"})

    assert state["rows"] == []


def test_metric_path_still_runs_queries_that_have_no_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Passing a ticket key to a query that does not use it must not break that query."""
    with _new_db(tmp_path, monkeypatch):
        pass

    state = agent.metric_path({"metric_key": "long_open_prs", "ticket_key": "AGENTS-14"})

    assert state["rows"] == []


def test_clean_ticket_key_gives_the_stored_form() -> None:
    """Spelling variations all become AGENTS-14; text with no ticket in it becomes None."""
    assert agent._clean_ticket_key("AGENTS-14") == "AGENTS-14"
    assert agent._clean_ticket_key("agents 14") == "AGENTS-14"
    assert agent._clean_ticket_key("AGENTS14") == "AGENTS-14"
    assert agent._clean_ticket_key("fourteen") is None
    assert agent._clean_ticket_key("") is None
    assert agent._clean_ticket_key(None) is None


def test_classify_intent_stores_a_cleaned_ticket_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model's ticket reference must reach the state in the stored form."""
    result = agent.Classification(metric_key="commits_for_ticket", ticket_key="AGENTS 14")
    monkeypatch.setattr(agent, "get_llm", lambda tier: _FakeModel(result))

    state = agent.classify_intent({"question": "Which commits relate to AGENTS 14?"})

    assert state["metric_key"] == "commits_for_ticket"
    assert state["ticket_key"] == "AGENTS-14"


def test_classify_intent_leaves_the_ticket_key_out_when_there_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A question that names no ticket must not put a ticket key in the state."""
    result = agent.Classification(metric_key="velocity")
    monkeypatch.setattr(agent, "get_llm", lambda tier: _FakeModel(result))

    state = agent.classify_intent({"question": "What was the velocity of each sprint?"})

    assert "ticket_key" not in state


def test_hybrid_path_passes_the_ticket_key_to_the_metric_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hybrid answer built on commits_for_ticket must hand over the ticket key it found."""
    seen: list[agent.AgentState] = []

    def fake_metric_path(state: agent.AgentState) -> agent.AgentState:
        seen.append(state)
        return {"rows": []}

    monkeypatch.setattr(agent, "metric_path", fake_metric_path)
    monkeypatch.setattr(agent, "search", lambda question, doc_type: [])

    agent.hybrid_path(
        {
            "question": "why so many commits on AGENTS-14",
            "doc_type": "retro",
            "hybrid_metric_key": "commits_for_ticket",
            "ticket_key": "AGENTS-14",
        }
    )

    assert seen == [{"metric_key": "commits_for_ticket", "ticket_key": "AGENTS-14"}]


def test_refuse_path_names_the_commits_for_one_ticket_question() -> None:
    """The refusal text must list the new question so a user knows it can be asked."""
    state = agent.refuse_path({"question": "what is the weather"})

    assert "the commits for one ticket" in state["answer"]


# ---------------------------------------------------------------------------
# AGENTS-57 — C5: how many commits have no ticket ID
# ---------------------------------------------------------------------------


def test_query_map_commits_without_ticket_uses_commits_without_ticket_sql() -> None:
    """commits_without_ticket must run github_connector's own COMMITS_WITHOUT_TICKET_SQL."""
    assert agent._QUERY_MAP["commits_without_ticket"] is COMMITS_WITHOUT_TICKET_SQL


def test_metric_path_runs_commits_without_ticket_sql(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Commits with no ticket key are counted, next to the total number of commits."""
    with _new_db(tmp_path, monkeypatch) as session:
        session.add(_commit("a", "AGENTS-14 add thing", "AGENTS-14", 1))
        session.add(_commit("b", "quick fix, no id", None, 2))
        session.add(_commit("c", "another one with no id", None, 3))
        session.commit()

    state = agent.metric_path({"metric_key": "commits_without_ticket"})

    assert state["rows"] == [{"commits_without_ticket": 2, "total_commits": 3}]


def test_metric_path_counts_zero_commits_without_ticket_on_an_empty_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no commits at all the answer is still one row of zeros, not an empty list."""
    with _new_db(tmp_path, monkeypatch):
        pass

    state = agent.metric_path({"metric_key": "commits_without_ticket"})

    assert state["rows"] == [{"commits_without_ticket": 0, "total_commits": 0}]


def test_refuse_path_names_the_commits_without_ticket_question() -> None:
    """The refusal text must list the new question so a user knows it can be asked."""
    state = agent.refuse_path({"question": "what is the weather"})

    assert "how many commits have no ticket ID" in state["answer"]


# ---------------------------------------------------------------------------
# AGENTS-58 — J3: branches that do not follow AGENTS-<n>-description
# ---------------------------------------------------------------------------


def test_query_map_non_convention_branches_uses_non_convention_branches_sql() -> None:
    """non_convention_branches must run github_connector's own NON_CONVENTION_BRANCHES_SQL."""
    assert agent._QUERY_MAP["non_convention_branches"] is NON_CONVENTION_BRANCHES_SQL


def test_metric_path_runs_non_convention_branches_sql(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only branches that break AGENTS-<n>-description come back, in PR order."""
    branches = {
        1: "AGENTS-58-fix-thing",  # follows the convention
        2: "fix-typo",  # no ticket prefix
        3: "agents-6-lowercase",  # wrong case
        4: "AGENTS-7",  # no description
        5: "AGENTS-abc-x",  # not a number
        6: "AGENTS-55-60-catalogue-queries",  # several ticket ids still follow it
    }
    with _new_db(tmp_path, monkeypatch) as session:
        for number, branch in branches.items():
            session.add(
                PullRequest(
                    number=number,
                    title=f"PR {number}",
                    state="closed",
                    created_at=datetime(2026, 1, 1),
                    head_branch=branch,
                )
            )
        session.commit()

    state = agent.metric_path({"metric_key": "non_convention_branches"})

    assert state["rows"] == [
        {"number": 2, "title": "PR 2", "head_branch": "fix-typo"},
        {"number": 3, "title": "PR 3", "head_branch": "agents-6-lowercase"},
        {"number": 4, "title": "PR 4", "head_branch": "AGENTS-7"},
        {"number": 5, "title": "PR 5", "head_branch": "AGENTS-abc-x"},
    ]


def test_refuse_path_names_the_naming_convention_question() -> None:
    """The refusal text must list the new question so a user knows it can be asked."""
    state = agent.refuse_path({"question": "what is the weather"})

    assert "branches that break the naming convention" in state["answer"]
