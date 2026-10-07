"""Tests for the GitHub Actions connector and the DORA queries (AGENTS-54).

Fully offline: the GitHub API is replaced by fake responses, and every query runs
against a temp SQLite file.
"""

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app import actions_connector, agent
from app.actions_connector import DORA_PER_SPRINT_SQL, FAILED_RUNS_BY_STAGE_SQL
from app.models import Commit, PipelineJob, PipelineRun, Sprint


def _new_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    """Create an empty temp database, point the agent and the connector at it, and open it."""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(agent, "DB_PATH", str(db_path))
    monkeypatch.setattr(actions_connector, "engine", engine)
    # The tables already exist, and init_db would otherwise create a real projectpulse.db.
    monkeypatch.setattr(actions_connector, "init_db", lambda: None)
    return Session(engine)


def _at(day: int, hour: int = 0) -> datetime:
    """A time in October 2026, as the naive UTC the store uses."""
    return datetime(2026, 10, day, hour)


def _run(
    run_id: int,
    sha: str,
    end: datetime,
    conclusion: str = "success",
    event: str = "push",
    branch: str = "main",
) -> PipelineRun:
    """A finished CI run that ended at the given time."""
    return PipelineRun(
        id=run_id,
        name="CI",
        event=event,
        head_branch=branch,
        head_sha=sha,
        status="completed",
        conclusion=conclusion,
        created_at=end - timedelta(minutes=3),
        updated_at=end,
    )


def _job(job_id: int, run_id: int, name: str, conclusion: str, end: datetime) -> PipelineJob:
    """A finished job inside a run."""
    return PipelineJob(
        id=job_id, run_id=run_id, name=name, conclusion=conclusion, completed_at=end
    )


def _commit(sha: str, authored_at: datetime) -> Commit:
    """A commit on main."""
    return Commit(sha=sha, message="AGENTS-54: x", authored_at=authored_at, ticket_key="AGENTS-54")


# ---------------------------------------------------------------------------
# Fetching and loading
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Stands in for an httpx response. Holds one page of the Actions API."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        """A good response never raises."""

    def json(self) -> dict[str, Any]:
        """The wrapped list, the way the Actions API returns it."""
        return self._payload


def test_get_pages_follows_pagination_and_unwraps_the_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full page of 100 means ask for the next page; the list sits under the given key."""
    pages_asked: list[int] = []

    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        """Return 100 items on page 1 and 3 on page 2."""
        page = kwargs["params"]["page"]
        pages_asked.append(page)
        count = 100 if page == 1 else 3
        runs = [{"id": i} for i in range(count)]
        return _FakeResponse({"total_count": 103, "workflow_runs": runs})

    monkeypatch.setattr(httpx, "get", fake_get)

    items = actions_connector._get_pages("/actions/runs", "workflow_runs")

    assert len(items) == 103
    assert pages_asked == [1, 2]


def test_get_pages_stops_after_a_short_first_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """A page with fewer than 100 items is the last one, so no second call is made."""
    pages_asked: list[int] = []

    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        """Return 2 jobs."""
        pages_asked.append(kwargs["params"]["page"])
        return _FakeResponse({"jobs": [{"id": 1}, {"id": 2}]})

    monkeypatch.setattr(httpx, "get", fake_get)

    assert len(actions_connector._get_pages("/actions/runs/9/jobs", "jobs")) == 2
    assert pages_asked == [1]


def test_get_pages_retries_after_a_dropped_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """One dropped connection is retried, and the second try returns the data."""
    calls: list[int] = []

    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        """Drop the connection on the first call, then answer normally."""
        calls.append(1)
        if len(calls) == 1:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return _FakeResponse({"jobs": [{"id": 1}]})

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    assert actions_connector._get_pages("/actions/runs/9/jobs", "jobs") == [{"id": 1}]
    assert len(calls) == 2


def test_get_pages_gives_up_after_three_failed_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the connection keeps dropping, the real error is raised after three tries."""
    calls: list[int] = []

    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        """Always drop the connection."""
        calls.append(1)
        raise httpx.RemoteProtocolError("Server disconnected without sending a response.")

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    with pytest.raises(httpx.RemoteProtocolError):
        actions_connector._get_pages("/actions/runs/9/jobs", "jobs")
    assert len(calls) == 3


def test_to_run_and_to_job_convert_github_timestamps_to_naive_utc() -> None:
    """GitHub sends ISO times ending in Z. The store keeps naive UTC, like the other tables."""
    run = actions_connector._to_run(
        {
            "id": 7, "name": "CI", "event": "push", "head_branch": "main", "head_sha": "abc",
            "status": "completed", "conclusion": "success",
            "created_at": "2026-10-02T10:00:00Z", "updated_at": "2026-10-02T10:03:30Z",
        }
    )
    job = actions_connector._to_job(
        {"id": 70, "name": "deploy", "conclusion": None, "completed_at": None}, run_id=7
    )

    assert run.updated_at == datetime(2026, 10, 2, 10, 3, 30)
    assert run.updated_at.tzinfo is None
    assert (job.run_id, job.conclusion, job.completed_at) == (7, None, None)


def test_load_replaces_pipeline_rows_and_leaves_other_tables_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """load() clears only pipeline_runs and pipeline_jobs, then stores what the API returned."""
    with _new_db(tmp_path, monkeypatch) as session:
        session.add(_commit("keep-me", _at(1)))
        session.add(_run(1, "old", _at(1)))
        session.add(_job(10, 1, "test", "success", _at(1)))
        session.commit()

    monkeypatch.setattr(
        actions_connector,
        "fetch_runs",
        lambda: [
            {
                "id": 2, "name": "CI", "event": "push", "head_branch": "main",
                "head_sha": "new", "status": "completed", "conclusion": "failure",
                "created_at": "2026-10-03T10:00:00Z", "updated_at": "2026-10-03T10:02:00Z",
            }
        ],
    )
    monkeypatch.setattr(
        actions_connector,
        "fetch_jobs",
        lambda run_id: [
            {"id": 20, "name": "test", "conclusion": "failure",
             "completed_at": "2026-10-03T10:02:00Z"}
        ],
    )

    actions_connector.load()

    with Session(create_engine(f"sqlite:///{tmp_path / 'test.db'}")) as session:
        assert [r.id for r in session.exec(select(PipelineRun)).all()] == [2]
        assert [j.id for j in session.exec(select(PipelineJob)).all()] == [20]
        assert [c.sha for c in session.exec(select(Commit)).all()] == ["keep-me"]


# ---------------------------------------------------------------------------
# The two queries
# ---------------------------------------------------------------------------


def test_query_map_uses_the_actions_connector_queries() -> None:
    """The agent must run the connector's own SQL constants, not copies of them."""
    assert agent._QUERY_MAP["dora_per_sprint"] is DORA_PER_SPRINT_SQL
    assert agent._QUERY_MAP["failed_runs_by_stage"] is FAILED_RUNS_BY_STAGE_SQL


def test_dora_per_sprint_computes_all_four_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two sprints with known runs: every metric is worked out by hand in the comments."""
    with _new_db(tmp_path, monkeypatch) as session:
        session.add(Sprint(id=1, name="Sprint A", state="closed", start_date=_at(1)))
        session.add(Sprint(id=2, name="Sprint B", state="active", start_date=_at(8)))

        session.add(_commit("c1", _at(2, 10)))
        session.add(_commit("c2", _at(3, 10)))
        session.add(_commit("c2fix", _at(3, 12)))
        session.add(_commit("c3", _at(9, 10)))

        # Sprint A: run 1 deploys 120 minutes after its commit; run 2 fails in test; run 3 fixes it.
        session.add(_run(1, "c1", _at(2, 12)))
        session.add(_job(11, 1, "test", "success", _at(2, 11)))
        session.add(_job(12, 1, "deploy", "success", _at(2, 12)))
        session.add(_run(2, "c2", _at(3, 11), conclusion="failure"))
        session.add(_job(21, 2, "test", "failure", _at(3, 11)))
        session.add(_job(22, 2, "deploy", "skipped", _at(3, 11)))
        session.add(_run(3, "c2fix", _at(3, 13)))
        session.add(_job(31, 3, "test", "success", _at(3, 12)))
        session.add(_job(32, 3, "deploy", "success", _at(3, 13)))
        # A failed run on a pull request is not a change that reached main. DORA ignores it.
        session.add(_run(5, "pr", _at(2, 9), conclusion="failure", event="pull_request",
                         branch="AGENTS-54-x"))
        session.add(_job(51, 5, "test", "failure", _at(2, 9)))

        # Sprint B: one clean deploy, 60 minutes after its commit.
        session.add(_run(4, "c3", _at(9, 11)))
        session.add(_job(41, 4, "test", "success", _at(9, 10)))
        session.add(_job(42, 4, "deploy", "success", _at(9, 11)))
        session.commit()

    rows = agent.metric_path({"metric_key": "dora_per_sprint"})["rows"]

    assert rows == [
        # Sprint A: 2 deploys (runs 1, 3); lead time (120 + 60) / 2 minutes; 1 failure in 3
        # finished runs; run 2 failed at 11:00 and run 3 succeeded at 13:00, so MTTR is 120.
        {"sprint": "Sprint A", "deployments": 2, "avg_lead_time_minutes": 90.0,
         "change_failure_rate_pct": 33.3, "mttr_minutes": 120.0},
        # Sprint B: 1 deploy, lead time 60 minutes, no failures, so nothing to recover from.
        {"sprint": "Sprint B", "deployments": 1, "avg_lead_time_minutes": 60.0,
         "change_failure_rate_pct": 0.0, "mttr_minutes": None},
    ]


def test_dora_per_sprint_with_no_pipeline_data_gives_zero_and_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A sprint with no runs shows 0 deployments and no value for the rest, not made-up zeros."""
    with _new_db(tmp_path, monkeypatch) as session:
        session.add(Sprint(id=1, name="Sprint A", state="active", start_date=_at(1)))
        session.commit()

    rows = agent.metric_path({"metric_key": "dora_per_sprint"})["rows"]

    assert rows == [
        {"sprint": "Sprint A", "deployments": 0, "avg_lead_time_minutes": None,
         "change_failure_rate_pct": None, "mttr_minutes": None}
    ]


def test_failed_runs_by_stage_counts_this_sprint_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only jobs since the latest sprint start count, and a stage with no failures shows 0."""
    now = datetime.now(UTC).replace(tzinfo=None)

    with _new_db(tmp_path, monkeypatch) as session:
        session.add(Sprint(id=1, name="Old", state="closed", start_date=now - timedelta(days=20)))
        session.add(Sprint(id=2, name="Now", state="active", start_date=now - timedelta(days=3)))
        session.add(_run(1, "a", now))

        two_days_ago = now - timedelta(days=2)
        session.add(_job(1, 1, "test", "failure", two_days_ago))
        session.add(_job(2, 1, "test", "failure", two_days_ago))
        session.add(_job(3, 1, "test", "success", two_days_ago))
        session.add(_job(4, 1, "deploy", "success", two_days_ago))
        # A skipped job never ran, so it is not a finished run of that stage.
        session.add(_job(5, 1, "deploy", "skipped", two_days_ago))
        # A failure from the previous sprint does not count for this one.
        session.add(_job(6, 1, "test", "failure", now - timedelta(days=10)))
        session.commit()

    rows = agent.metric_path({"metric_key": "failed_runs_by_stage"})["rows"]

    assert rows == [
        {"stage": "test", "failed_runs": 2, "finished_runs": 3},
        {"stage": "deploy", "failed_runs": 0, "finished_runs": 1},
    ]


def test_failed_runs_by_stage_is_empty_when_no_sprint_has_started(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no sprint to measure from, the query returns no rows instead of guessing."""
    with _new_db(tmp_path, monkeypatch):
        pass

    assert agent.metric_path({"metric_key": "failed_runs_by_stage"})["rows"] == []


# ---------------------------------------------------------------------------
# Agent wiring
# ---------------------------------------------------------------------------


def test_classification_accepts_the_new_metric_keys() -> None:
    """The classifier's allowed answers include both new keys, also for a hybrid question."""
    for key in ("dora_per_sprint", "failed_runs_by_stage"):
        assert agent.Classification(metric_key=key).metric_key == key
        hybrid = agent.Classification(metric_key="hybrid", hybrid_metric_key=key)
        assert hybrid.hybrid_metric_key == key


def test_refuse_path_names_the_dora_questions() -> None:
    """The refusal text must list the new questions so a user knows they can be asked."""
    answer = agent.refuse_path({"question": "what is the weather"})["answer"]

    assert "the four DORA metrics per sprint" in answer
    assert "failed pipeline runs by stage" in answer
