"""Live GitHub tools: recent pull requests and recent commits (ADR-023).

Both tools read from GitHub itself, only when the stored copy cannot answer, for example when
a pull request was opened after the last sync. They return the same fields the GitHub connector
stores. Nothing is written to the database, and GitHub receives GET requests only (see
live_http). Each tool makes one call for one page, at most 25 items, so the rate limit is
barely touched. Reviews are not included, because they need one extra call per pull request.
"""

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.github_connector import HEADERS, REPO_URL, _dt, _ticket_key
from app.live_http import LiveDataError, fetched_at, live_get

MAX_RESULTS = 25
DEFAULT_RESULTS = 10


def _time(value: str | None) -> str | None:
    """A GitHub timestamp as UTC text, like the stored ones, or None."""
    parsed = _dt(value)
    return parsed.isoformat(timespec="seconds") if parsed else None


def _pull_request(entry: dict[str, Any]) -> dict[str, Any]:
    """One pull request as the fields the connector stores."""
    user = entry.get("user")
    branch: str = entry["head"]["ref"]
    return {
        "number": entry["number"],
        "title": entry["title"],
        "author_login": user["login"] if user else None,
        "state": entry["state"],
        "created_at": _time(entry["created_at"]),
        "merged_at": _time(entry.get("merged_at")),
        "closed_at": _time(entry.get("closed_at")),
        "head_branch": branch,
        "ticket_key": _ticket_key(branch) or _ticket_key(entry["title"]),
    }


def _commit(entry: dict[str, Any]) -> dict[str, Any]:
    """One commit as the fields the connector stores. The message is its first line only."""
    author = entry.get("author")
    message: str = entry["commit"]["message"]
    return {
        "sha": entry["sha"][:7],
        "message": message.splitlines()[0] if message else "",
        "author_login": author["login"] if author else None,
        "authored_at": _time(entry["commit"]["author"]["date"]),
        "ticket_key": _ticket_key(message),
    }


def _limit(max_results: int) -> int:
    """Keep the requested number between 1 and MAX_RESULTS."""
    return max(1, min(max_results, MAX_RESULTS))


def _result(rows: list[dict[str, Any]], limit: int, empty_message: str) -> dict[str, Any]:
    """The common shape of a live GitHub result: source, label, rows and a note if empty."""
    result: dict[str, Any] = {
        "source": "GitHub",
        "live": True,
        "fetched_at": fetched_at(),
        "rows": rows,
        # One page only. A full page means there may be older items that were not returned.
        "may_be_more": len(rows) == limit,
    }
    if not rows:
        result["message"] = empty_message
    return result


def live_github_pull_requests(
    state: Literal["open", "closed", "all"] = "all", max_results: int = DEFAULT_RESULTS
) -> dict[str, Any]:
    """The most recently opened pull requests, read from GitHub now."""
    if state not in ("open", "closed", "all"):
        return {"error": "state must be open, closed or all."}
    limit = _limit(max_results)
    try:
        entries = live_get(
            "GitHub",
            f"{REPO_URL}/pulls",
            headers=HEADERS,
            params={"state": state, "sort": "created", "direction": "desc", "per_page": limit},
        )
    except LiveDataError as exc:
        return {"error": str(exc)}
    rows = [_pull_request(entry) for entry in entries]
    return _result(rows, limit, "GitHub has no pull requests that match.")


def live_github_commits(
    since: str | None = None, max_results: int = DEFAULT_RESULTS
) -> dict[str, Any]:
    """The most recent commits on the default branch, read from GitHub now."""
    params: dict[str, Any] = {"per_page": _limit(max_results)}
    if since:
        try:
            params["since"] = f"{date.fromisoformat(since).isoformat()}T00:00:00Z"
        except ValueError:
            return {"error": "since must be a date written YYYY-MM-DD."}
    limit = params["per_page"]
    try:
        entries = live_get("GitHub", f"{REPO_URL}/commits", headers=HEADERS, params=params)
    except LiveDataError as exc:
        return {"error": str(exc)}
    rows = [_commit(entry) for entry in entries]
    return _result(rows, limit, "GitHub has no commits that match.")


class LiveGithubPullRequestsArgs(BaseModel):
    """Arguments the model may give live_github_pull_requests."""

    state: Literal["open", "closed", "all"] = Field(
        default="all", description="open, closed or all. Merged pull requests count as closed."
    )
    max_results: int = Field(default=DEFAULT_RESULTS, description=f"At most {MAX_RESULTS}.")


class LiveGithubCommitsArgs(BaseModel):
    """Arguments the model may give live_github_commits."""

    since: str | None = Field(
        default=None, description="Only commits made on or after this date, YYYY-MM-DD."
    )
    max_results: int = Field(default=DEFAULT_RESULTS, description=f"At most {MAX_RESULTS}.")


PULL_REQUESTS_DESCRIPTION = (
    "List the most recently opened pull requests straight from GitHub, right now, newest "
    "first. Use it only when the stored data has no match, or its freshness says it is "
    "stale, for example for a pull request opened after the last sync. Returns number, title, "
    "author, state, dates, branch and ticket key. It does not return reviews."
)

COMMITS_DESCRIPTION = (
    "List the most recent commits on the default branch straight from GitHub, right now, "
    "newest first. Use it only when the stored data has no match, or its freshness says it "
    "is stale. Returns a short commit id, the first line of the message, author, time and "
    "ticket key."
)