"""Live Jira tools: read one ticket, or search tickets with a few safe filters (ADR-023).

Both tools read from Jira itself, only when the stored copy cannot answer. They return the same
fields the Jira connector stores, so an answer reads the same whichever source it came from.
Nothing is written to the database, and Jira receives GET requests only (see live_http).

The model never writes JQL (Jira's query language). It gives a few named filters, and this file
builds the query, so a filter value cannot add its own clauses.
"""

import re
from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from app import agent
from app.jira_connector import AUTH, BASE_URL, JIRA_PROJECT_KEY, POINTS_FIELD, SPRINT_FIELD, _dt
from app.live_http import LiveDataError, LiveNotFound, fetched_at, live_get

MAX_RESULTS = 25
DEFAULT_RESULTS = 10

# Fields asked from Jira. They are the ones the connector stores for a ticket.
_FIELDS = ",".join(
    [
        "summary",
        "issuetype",
        "status",
        "statusCategory",
        "parent",
        "assignee",
        "created",
        "updated",
        "resolutiondate",
        POINTS_FIELD,
        SPRINT_FIELD,
    ]
)

# Letters, digits, spaces, dots, dashes and underscores only. A quote or a JQL keyword
# character cannot get through, so a filter value cannot change the query.
_SAFE_TEXT = re.compile(r"^[A-Za-z0-9 _.\-]{1,60}$")


def _time(value: str | None) -> str | None:
    """A Jira timestamp as UTC text, like the stored ones, or None."""
    parsed = _dt(value)
    return parsed.isoformat(timespec="seconds") if parsed else None


def _ticket(issue: dict[str, Any]) -> dict[str, Any]:
    """One Jira issue as the fields the connector stores, plus readable names."""
    fields = issue["fields"]
    parent = fields.get("parent")
    assignee = fields.get("assignee")
    return {
        "key": issue["key"],
        "summary": fields.get("summary"),
        "issue_type": (fields.get("issuetype") or {}).get("name"),
        "status": (fields.get("status") or {}).get("name"),
        "status_category": (fields.get("statusCategory") or {}).get("key"),
        "story_points": fields.get(POINTS_FIELD),
        "parent_key": parent["key"] if parent else None,
        "assignee": assignee["displayName"] if assignee else None,
        "created": _time(fields.get("created")),
        "updated": _time(fields.get("updated")),
        "resolved": _time(fields.get("resolutiondate")),
        "sprints": [sprint.get("name") for sprint in fields.get(SPRINT_FIELD) or []],
    }


def _envelope(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The common top of every live Jira result: where it came from, when, and the rows."""
    return {"source": "Jira", "live": True, "fetched_at": fetched_at(), "rows": rows}


def live_jira_ticket(ticket_key: str) -> dict[str, Any]:
    """Fetch one ticket from Jira now."""
    key = agent._clean_ticket_key(ticket_key)
    if key is None:
        return {"error": "live_jira_ticket needs a ticket key such as AGENTS-14."}
    try:
        issue = live_get(
            "Jira", f"{BASE_URL}/rest/api/3/issue/{key}", auth=AUTH, params={"fields": _FIELDS}
        )
    except LiveNotFound:
        result = _envelope([])
        result["message"] = f"Jira has no ticket {key}."
        return result
    except LiveDataError as exc:
        return {"error": str(exc)}
    return _envelope([_ticket(issue)])


def _clause(field: str, value: str) -> str:
    """One JQL clause such as status = "In Progress". The value is checked first."""
    if not _SAFE_TEXT.match(value):
        raise ValueError(
            f"The value for {field} may only use letters, digits, spaces, dots, dashes "
            "and underscores, up to 60 characters."
        )
    operator = "~" if field == "text" else "="
    return f'{field} {operator} "{value}"'


def _date_clause(field: str, value: str) -> str:
    """One JQL date clause such as created >= "2026-10-09". The value must be YYYY-MM-DD."""
    try:
        day = date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{field} must be a date written YYYY-MM-DD.") from None
    return f'{field} >= "{day.isoformat()}"'


def build_jql(
    status: str | None,
    issue_type: str | None,
    created_since: str | None,
    updated_since: str | None,
    text: str | None,
) -> str:
    """Build the whole JQL string from the named filters. Raises ValueError for a bad value."""
    clauses = [f"project = {JIRA_PROJECT_KEY}"]
    if status:
        clauses.append(_clause("status", status))
    if issue_type:
        clauses.append(_clause("issuetype", issue_type))
    if created_since:
        clauses.append(_date_clause("created", created_since))
    if updated_since:
        clauses.append(_date_clause("updated", updated_since))
    if text:
        clauses.append(_clause("text", text))
    return " AND ".join(clauses) + " ORDER BY created DESC"


def live_jira_search(
    status: str | None = None,
    issue_type: str | None = None,
    created_since: str | None = None,
    updated_since: str | None = None,
    text: str | None = None,
    max_results: int = DEFAULT_RESULTS,
) -> dict[str, Any]:
    """Search the project's tickets in Jira now, newest first."""
    try:
        jql = build_jql(status, issue_type, created_since, updated_since, text)
    except ValueError as exc:
        return {"error": str(exc)}
    limit = max(1, min(max_results, MAX_RESULTS))
    try:
        page = live_get(
            "Jira",
            f"{BASE_URL}/rest/api/3/search/jql",
            auth=AUTH,
            params={"jql": jql, "maxResults": limit, "fields": _FIELDS},
        )
    except LiveDataError as exc:
        return {"error": str(exc)}
    rows = [_ticket(issue) for issue in page.get("issues", [])]
    result = _envelope(rows)
    result["more_available"] = bool(page.get("nextPageToken"))
    if not rows:
        result["message"] = "Jira has no tickets that match these filters."
    return result


class LiveJiraTicketArgs(BaseModel):
    """Arguments the model may give live_jira_ticket."""

    ticket_key: str = Field(description="A ticket key such as AGENTS-14.")


class LiveJiraSearchArgs(BaseModel):
    """Arguments the model may give live_jira_search."""

    status: str | None = Field(default=None, description='A status name such as "In Progress".')
    issue_type: str | None = Field(default=None, description='A type such as "Task" or "Epic".')
    created_since: str | None = Field(
        default=None, description="Only tickets created on or after this date, YYYY-MM-DD."
    )
    updated_since: str | None = Field(
        default=None, description="Only tickets updated on or after this date, YYYY-MM-DD."
    )
    text: str | None = Field(default=None, description="A word or phrase in the ticket text.")
    max_results: int = Field(default=DEFAULT_RESULTS, description=f"At most {MAX_RESULTS}.")


TICKET_DESCRIPTION = (
    "Fetch one ticket from Jira itself, right now. Use it only when get_metric has no row for "
    "a ticket the user names, or its freshness says the stored data is stale. Returns the "
    "ticket's key, summary, type, status, story points, parent, assignee, dates and sprints."
)

SEARCH_DESCRIPTION = (
    "Search the project's tickets in Jira itself, right now, newest first. Use it only when "
    "the stored data has no match, or its freshness says it is stale, for example to find "
    "tickets created today. Give any of the filters; they are combined with AND. Returns at "
    "most 25 tickets with the same fields as live_jira_ticket."
)