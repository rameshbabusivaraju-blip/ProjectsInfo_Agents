"""Pull the GitHub Actions pipeline history into the local SQLite store.

Copies workflow runs and their jobs so that the four DORA metrics (catalogue
section D, ADR-019) can be answered with SQL. It uses the same token and the same
repository as github_connector.

Run it with:  python -m app.actions_connector
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from sqlalchemy import delete, text
from sqlmodel import Session

from app.db import engine, init_db
from app.github_connector import HEADERS, REPO_URL, _dt, _required_dt
from app.models import PipelineJob, PipelineRun

_ATTEMPTS = 3


def _get_with_retry(url: str, params: dict[str, Any]) -> httpx.Response:
    """GET one URL, trying again if the connection drops.

    The jobs endpoint is called once per run, so a full load makes many calls in a row.
    A dropped connection ("Server disconnected without sending a response") is usually
    a one-off, so waiting a moment and trying again is enough. An HTTP error such as
    403 is not retried: it is a real answer, and it reaches raise_for_status.
    """
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            return httpx.get(url, headers=HEADERS, params=params, timeout=30)
        except httpx.TransportError:
            # On the last attempt, let the real error through so a true outage is visible.
            if attempt == _ATTEMPTS:
                raise
            time.sleep(attempt)
    raise AssertionError("unreachable")


def _get_pages(path: str, key: str, **params: Any) -> list[dict[str, Any]]:
    """Call an Actions endpoint and follow pagination until a short page comes back.

    The commits and pulls endpoints return a plain list. The Actions endpoints wrap
    the list in an object, such as {"total_count": 3, "workflow_runs": [...]}, so
    key says which field holds the list.
    """
    items: list[dict[str, Any]] = []
    page = 1

    while True:
        response = _get_with_retry(
            f"{REPO_URL}{path}", {**params, "per_page": 100, "page": page}
        )
        response.raise_for_status()
        batch: list[dict[str, Any]] = response.json()[key]

        items.extend(batch)
        # GitHub returns at most 100 per page, so a shorter page is the last one.
        if len(batch) < 100:
            return items
        page += 1


def fetch_runs() -> list[dict[str, Any]]:
    """Every workflow run in the repository, newest first."""
    return _get_pages("/actions/runs", "workflow_runs")


def fetch_jobs(run_id: int) -> list[dict[str, Any]]:
    """The jobs inside one run. One extra call per run."""
    return _get_pages(f"/actions/runs/{run_id}/jobs", "jobs")


def _to_run(entry: dict[str, Any]) -> PipelineRun:
    """Turn one workflow run from the API into a table row."""
    return PipelineRun(
        id=entry["id"],
        name=entry["name"],
        event=entry["event"],
        head_branch=entry.get("head_branch"),
        head_sha=entry["head_sha"],
        status=entry["status"],
        conclusion=entry.get("conclusion"),
        created_at=_required_dt(entry["created_at"]),
        updated_at=_required_dt(entry["updated_at"]),
    )


def _to_job(entry: dict[str, Any], run_id: int) -> PipelineJob:
    """Turn one job from the API into a table row that points at its run."""
    return PipelineJob(
        id=entry["id"],
        run_id=run_id,
        name=entry["name"],
        conclusion=entry.get("conclusion"),
        completed_at=_dt(entry.get("completed_at")),
    )


def load() -> None:
    """Refresh the two pipeline tables from the API.

    Only these two tables are cleared, so running this does not disturb the Jira,
    GitHub or Confluence data already in the same database.
    """
    runs = fetch_runs()

    init_db()

    with Session(engine) as session:
        # Clear pipeline rows only. Jobs go first because they reference runs.
        session.execute(delete(PipelineJob))
        session.execute(delete(PipelineRun))

        job_rows: list[PipelineJob] = []
        for entry in runs:
            session.add(_to_run(entry))
            job_rows.extend(_to_job(job, entry["id"]) for job in fetch_jobs(entry["id"]))

        for row in job_rows:
            session.add(row)

        session.commit()

    print(f"Loaded {len(runs)} pipeline runs, {len(job_rows)} jobs")


# ---------------------------------------------------------------------------
# Catalogue questions D1 to D4: the four DORA metrics per sprint (ADR-019)
# ---------------------------------------------------------------------------

# One row per sprint. A sprint "owns" the time from its own start to the start of the
# next sprint, so two sprints never share a moment, even when one was closed late.
# Only push runs on main count: that is what reaches users. Runs on pull requests are
# tests of unmerged work and are left out.
#
# deployments          = deploy jobs that succeeded (deployment frequency).
# avg_lead_time_minutes = commit time to the end of its deploy job, averaged (lead time).
#                        You squash-merge, so the commit on main is made at merge time
#                        and this is short, a few minutes. That is expected.
# change_failure_rate_pct = failed runs on main as a percent of finished runs on main.
#                        Cancelled and skipped runs are not counted either way.
# mttr_minutes         = for each failed run on main, minutes until the next successful
#                        run on main, averaged. A failure with no later success yet is
#                        left out. NULL (no value) means there was nothing to measure.
#
# julianday turns a timestamp into a number of days, so a difference times 1440 is minutes.
DORA_PER_SPRINT_SQL = """
WITH windows AS (
    SELECT s.name,
           s.start_date AS win_start,
           COALESCE(
               (SELECT MIN(n.start_date) FROM sprints n WHERE n.start_date > s.start_date),
               '9999-12-31'
           ) AS win_end
    FROM sprints s
    WHERE s.start_date IS NOT NULL
)
SELECT w.name AS sprint,
       (SELECT COUNT(*)
        FROM pipeline_jobs j
        JOIN pipeline_runs r ON r.id = j.run_id
        WHERE r.event = 'push' AND r.head_branch = 'main'
          AND j.name = 'deploy' AND j.conclusion = 'success'
          AND j.completed_at >= w.win_start AND j.completed_at < w.win_end
       ) AS deployments,
       (SELECT ROUND(AVG((julianday(j.completed_at) - julianday(c.authored_at)) * 1440), 1)
        FROM pipeline_jobs j
        JOIN pipeline_runs r ON r.id = j.run_id
        JOIN commits c ON c.sha = r.head_sha
        WHERE r.event = 'push' AND r.head_branch = 'main'
          AND j.name = 'deploy' AND j.conclusion = 'success'
          AND j.completed_at >= w.win_start AND j.completed_at < w.win_end
       ) AS avg_lead_time_minutes,
       (SELECT ROUND(100.0 * SUM(r.conclusion = 'failure') / COUNT(*), 1)
        FROM pipeline_runs r
        WHERE r.event = 'push' AND r.head_branch = 'main'
          AND r.conclusion IN ('success', 'failure')
          AND r.updated_at >= w.win_start AND r.updated_at < w.win_end
       ) AS change_failure_rate_pct,
       (SELECT ROUND(AVG((julianday(
                   (SELECT MIN(ok.updated_at)
                    FROM pipeline_runs ok
                    WHERE ok.event = 'push' AND ok.head_branch = 'main'
                      AND ok.conclusion = 'success' AND ok.updated_at > f.updated_at)
               ) - julianday(f.updated_at)) * 1440), 1)
        FROM pipeline_runs f
        WHERE f.event = 'push' AND f.head_branch = 'main' AND f.conclusion = 'failure'
          AND f.updated_at >= w.win_start AND f.updated_at < w.win_end
       ) AS mttr_minutes
FROM windows w
ORDER BY w.win_start
"""

# Question D6: how many pipeline runs failed this sprint, and at which stage.
#
# A stage is a job: test or deploy. "This sprint" starts at the latest sprint start that
# has already happened. Pull request runs are included, because a test that fails on a
# branch is still a failed pipeline run. Every stage that ran appears, with 0 if none
# failed, so "nothing failed" shows as a 0 and not as an empty answer.
FAILED_RUNS_BY_STAGE_SQL = """
SELECT j.name AS stage,
       SUM(j.conclusion = 'failure') AS failed_runs,
       COUNT(*) AS finished_runs
FROM pipeline_jobs j
WHERE j.conclusion IN ('success', 'failure')
  AND j.completed_at >= (
      SELECT MAX(start_date) FROM sprints WHERE start_date <= datetime('now')
  )
GROUP BY j.name
ORDER BY failed_runs DESC, stage
"""


def report() -> None:
    """Print the DORA numbers per sprint, so a fresh load can be checked by eye."""
    with Session(engine) as session:
        for row in session.execute(text(DORA_PER_SPRINT_SQL)).all():
            sprint, deployments, lead, failure_rate, mttr = row
            print(
                f"{sprint}: {deployments} deployments, lead time {lead} min, "
                f"change failure rate {failure_rate} %, MTTR {mttr} min"
            )


if __name__ == "__main__":
    load()
    print()
    report()
