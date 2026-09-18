"""Pull the Jira project into the local SQLite store.

Jira is the source of truth. This file copies what it says into SQLite so that
questions about velocity, spillover and effort can be answered with one SQL
query instead of many API calls.

Run it with:  python -m app.jira_connector
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx
from dotenv import load_dotenv
from sqlalchemy import text
from sqlmodel import Session, SQLModel

from app.db import engine
from app.models import Person, Sprint, Ticket, TicketSprint, WorkLog

# Read .env into the process environment. Secrets never live in this file.
load_dotenv()

ATLASSIAN_SITE = os.environ["ATLASSIAN_SITE"]
ATLASSIAN_EMAIL = os.environ["ATLASSIAN_EMAIL"]
ATLASSIAN_API_TOKEN = os.environ["ATLASSIAN_API_TOKEN"]
JIRA_PROJECT_KEY = os.environ["JIRA_PROJECT_KEY"]

# Sprints belong to a board, not to a project, so the board id is needed too.
JIRA_BOARD_ID = os.environ.get("JIRA_BOARD_ID", "1")

# Jira custom field ids differ per site. These two were found by the API spike:
# 10016 holds story points, 10020 holds the list of sprints an issue has been in.
POINTS_FIELD = os.environ.get("JIRA_POINTS_FIELD", "customfield_10016")
SPRINT_FIELD = os.environ.get("JIRA_SPRINT_FIELD", "customfield_10020")

BASE_URL = f"https://{ATLASSIAN_SITE}"

# Atlassian uses HTTP Basic auth: email as the username, API token as the password.
AUTH = (ATLASSIAN_EMAIL, ATLASSIAN_API_TOKEN)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(path: str, **params: Any) -> dict[str, Any]:
    """Call a Jira endpoint and return the parsed JSON.

    raise_for_status turns an HTTP error into an exception, so a bad token or
    a wrong URL fails loudly here rather than producing empty tables later.
    """
    response = httpx.get(f"{BASE_URL}{path}", auth=AUTH, params=params, timeout=30)
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def _dt(value: str | None) -> datetime | None:
    """Convert a Jira timestamp to a naive UTC datetime, or None if absent.

    Jira returns times with mixed offsets (+0530, Z). Storing them as-is would
    make date comparisons in SQL unreliable, so everything is normalised to UTC
    and the timezone marker is dropped.
    """
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _required_dt(value: str) -> datetime:
    """Same as _dt, for fields Jira always provides (created, updated, started)."""
    parsed = _dt(value)
    assert parsed is not None
    return parsed


def _adf_text(node: Any) -> str:
    """Flatten Atlassian Document Format into plain text.

    Jira stores descriptions and comments as a nested JSON tree, not as text.
    Only nodes of type "text" carry words; everything else is structure. This
    walks the tree and joins the words it finds.
    """
    if isinstance(node, dict):
        if node.get("type") == "text":
            return str(node.get("text", ""))
        return " ".join(_adf_text(child) for child in node.get("content", []))
    if isinstance(node, list):
        return " ".join(_adf_text(child) for child in node)
    return ""


# ---------------------------------------------------------------------------
# Fetching — one function per Jira endpoint, each returning raw JSON
# ---------------------------------------------------------------------------


def fetch_sprints() -> list[dict[str, Any]]:
    """Every sprint on the board, past, present and future.

    Note this is the Agile API (/rest/agile/1.0/), which is a different API from
    the one that serves issues. Sprints do not exist in the core API at all.
    """
    values: list[dict[str, Any]] = _get(f"/rest/agile/1.0/board/{JIRA_BOARD_ID}/sprint")["values"]
    return values


def fetch_issues() -> list[dict[str, Any]]:
    """Every issue in the project, following pagination until Jira stops.

    Jira returns at most 100 issues per call and hands back a nextPageToken when
    more remain. Ignoring that token is the usual way a connector silently loses
    data once a project grows past one page.
    """
    issues: list[dict[str, Any]] = []
    next_token: str | None = None

    while True:
        params: dict[str, Any] = {
            "jql": f"project = {JIRA_PROJECT_KEY}",
            "maxResults": 100,
            "fields": "*all",
        }
        if next_token:
            params["nextPageToken"] = next_token

        page = _get("/rest/api/3/search/jql", **params)
        issues.extend(page.get("issues", []))

        next_token = page.get("nextPageToken")
        if not next_token:
            return issues


def fetch_worklogs(issue_key: str) -> list[dict[str, Any]]:
    """Logged hours for one issue.

    Work logs are not included in the issue search response, so this is one
    extra call per issue. Fine at this scale; would need batching at thousands.
    """
    logs: list[dict[str, Any]] = _get(f"/rest/api/3/issue/{issue_key}/worklog")["worklogs"]
    return logs


# ---------------------------------------------------------------------------
# Loading — transform the raw JSON into rows and write them
# ---------------------------------------------------------------------------


def load() -> None:
    """Rebuild the local database from what Jira currently says.

    The database is a mirror of Jira, not a second source of truth, so it is
    dropped and recreated rather than updated in place. Working out what changed
    would be more code and more bugs for no benefit at this size.
    """
    # Fetch everything first. If a call fails, the existing database is untouched.
    sprints = fetch_sprints()
    issues = fetch_issues()

    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)

    # People appear on issues and on work logs. Collect them once, write them once.
    people: dict[str, Person] = {}

    def remember(user: dict[str, Any] | None) -> str | None:
        """Record a Jira user if not seen before, and return their account id."""
        if not user:
            return None
        account_id: str = user["accountId"]
        if account_id not in people:
            people[account_id] = Person(
                account_id=account_id,
                display_name=user["displayName"],
            )
        return account_id

    with Session(engine) as session:
        # --- Sprints -------------------------------------------------------
        for sprint in sprints:
            session.add(
                Sprint(
                    id=sprint["id"],
                    name=sprint["name"],
                    state=sprint["state"],
                    goal=sprint.get("goal"),
                    start_date=_dt(sprint.get("startDate")),
                    end_date=_dt(sprint.get("endDate")),
                    complete_date=_dt(sprint.get("completeDate")),
                )
            )

        # Work logs are collected here and added after people, since they
        # reference a person by account id.
        work_logs: list[WorkLog] = []

        # --- Issues --------------------------------------------------------
        for issue in issues:
            fields = issue["fields"]
            parent = fields.get("parent")

            # hierarchy_level distinguishes epics (1) from stories and tasks (0)
            # without matching on the issue type name, which is configurable.
            session.add(
                Ticket(
                    key=issue["key"],
                    summary=fields["summary"],
                    issue_type=fields["issuetype"]["name"],
                    hierarchy_level=fields["issuetype"]["hierarchyLevel"],
                    status=fields["status"]["name"],
                    status_category=fields["statusCategory"]["key"],
                    story_points=fields.get(POINTS_FIELD),
                    parent_key=parent["key"] if parent else None,
                    assignee_id=remember(fields.get("assignee")),
                    created=_required_dt(fields["created"]),
                    updated=_required_dt(fields["updated"]),
                    resolved=_dt(fields.get("resolutiondate")),
                    original_estimate_seconds=fields.get("timeoriginalestimate"),
                    time_spent_seconds=fields.get("timespent"),
                    description=_adf_text(fields.get("description")) or None,
                )
            )

            # The sprint field holds every sprint the issue has ever been in,
            # in order. Storing the order is what makes spillover measurable:
            # more than one row means the issue moved between sprints.
            for position, issue_sprint in enumerate(fields.get(SPRINT_FIELD) or []):
                session.add(
                    TicketSprint(
                        ticket_key=issue["key"],
                        sprint_id=issue_sprint["id"],
                        position=position,
                    )
                )

            # --- Work logs for this issue ----------------------------------
            for entry in fetch_worklogs(issue["key"]):
                author_id = remember(entry["author"])
                assert author_id is not None
                work_logs.append(
                    WorkLog(
                        id=entry["id"],
                        ticket_key=issue["key"],
                        author_id=author_id,
                        started=_required_dt(entry["started"]),
                        # Jira reports time in seconds. Stored as-is; converted
                        # to hours at query time.
                        seconds=entry["timeSpentSeconds"],
                        comment=_adf_text(entry.get("comment")) or None,
                    )
                )

        # --- People and work logs ------------------------------------------
        for person in people.values():
            session.add(person)
        for work_log in work_logs:
            session.add(work_log)

        # One commit for the whole load: either the mirror is complete or the
        # transaction rolls back and nothing is written.
        session.commit()

    print(f"Loaded {len(sprints)} sprints, {len(issues)} issues, {len(work_logs)} work logs")


# ---------------------------------------------------------------------------
# First question from the catalogue: A1, velocity per sprint
# ---------------------------------------------------------------------------

# A ticket can appear in several sprints. It counts towards the sprint it was in
# last, which is the row with the highest position. status_category is used
# rather than status because "Done" is a configurable name but the category is not.
VELOCITY_SQL = """
SELECT s.name AS sprint, COALESCE(SUM(t.story_points), 0) AS delivered
FROM tickets t
JOIN ticket_sprints ts ON ts.ticket_key = t.key
JOIN sprints s ON s.id = ts.sprint_id
WHERE t.status_category = 'done'
  AND ts.position = (
      SELECT MAX(position) FROM ticket_sprints WHERE ticket_key = t.key
  )
GROUP BY s.id
ORDER BY s.start_date
"""


def velocity() -> None:
    """Print points delivered per sprint — catalogue question A1."""
    with Session(engine) as session:
        for sprint_name, delivered in session.execute(text(VELOCITY_SQL)).all():
            print(f"{sprint_name}: {delivered} points")


if __name__ == "__main__":
    load()
    print("\n=== A1 — velocity per sprint ===")
    velocity()