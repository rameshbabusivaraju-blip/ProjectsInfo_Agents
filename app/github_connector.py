"""Pull the GitHub repository into the local SQLite store.

Copies commits, pull requests and reviews so that source-control questions
(catalogue section C) can be answered with SQL.

Run it with:  python -m app.github_connector
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from typing import Any

import httpx
from dotenv import load_dotenv
from sqlalchemy import delete, text
from sqlmodel import Session

from app.db import engine, init_db
from app.models import Commit, PrReview, PullRequest

load_dotenv()

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_OWNER = os.environ["GITHUB_OWNER"]
GITHUB_REPO = os.environ["GITHUB_REPO"]

REPO_URL = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"

# GitHub wants a bearer token and an explicit API version in the Accept header.
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}

# ADR-016 requires every branch and commit message to carry the ticket id.
TICKET_PATTERN = re.compile(r"AGENTS-\d+")


def _get_pages(path: str, **params: Any) -> list[dict[str, Any]]:
    """Call a GitHub endpoint and follow pagination until a short page comes back.

    GitHub returns at most 100 items per page. Asking for page N+1 after a full
    page is the only reliable way to know you have everything.
    """
    items: list[dict[str, Any]] = []
    page = 1

    while True:
        response = httpx.get(
            f"{REPO_URL}{path}",
            headers=HEADERS,
            params={**params, "per_page": 100, "page": page},
            timeout=30,
        )
        response.raise_for_status()
        batch: list[dict[str, Any]] = response.json()

        items.extend(batch)
        if len(batch) < 100:
            return items
        page += 1


def _dt(value: str | None) -> datetime | None:
    """GitHub timestamps are ISO with a Z suffix. Store them as naive UTC."""
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(UTC).replace(tzinfo=None)


def _required_dt(value: str) -> datetime:
    parsed = _dt(value)
    assert parsed is not None
    return parsed


def _ticket_key(textual: str | None) -> str | None:
    """Pull the first AGENTS-nn out of a commit message or branch name."""
    if not textual:
        return None
    match = TICKET_PATTERN.search(textual)
    return match.group(0) if match else None


def fetch_commits() -> list[dict[str, Any]]:
    """Every commit on the default branch."""
    return _get_pages("/commits")


def fetch_pull_requests() -> list[dict[str, Any]]:
    """Every pull request — open, closed and merged."""
    return _get_pages("/pulls", state="all")


def fetch_reviews(pr_number: int) -> list[dict[str, Any]]:
    """Reviews left on one pull request. One extra call per PR."""
    return _get_pages(f"/pulls/{pr_number}/reviews")


def load() -> None:
    """Refresh the three GitHub tables from the API.

    Only these three tables are cleared, so running this does not disturb the
    Jira data already in the same database.
    """
    commits = fetch_commits()
    pull_requests = fetch_pull_requests()

    init_db()

    with Session(engine) as session:
        # Clear GitHub rows only. Reviews go first because they reference PRs.
        session.execute(delete(PrReview))
        session.execute(delete(PullRequest))
        session.execute(delete(Commit))

        # --- Commits -------------------------------------------------------
        for entry in commits:
            message: str = entry["commit"]["message"]
            author = entry.get("author")
            session.add(
                Commit(
                    sha=entry["sha"],
                    message=message,
                    # "author" is the GitHub account; it is null when the commit
                    # email does not match any account.
                    author_login=author["login"] if author else None,
                    authored_at=_required_dt(entry["commit"]["author"]["date"]),
                    ticket_key=_ticket_key(message),
                )
            )

        # --- Pull requests and their reviews --------------------------------
        review_rows: list[PrReview] = []

        for entry in pull_requests:
            number: int = entry["number"]
            branch: str = entry["head"]["ref"]
            user = entry.get("user")

            session.add(
                PullRequest(
                    number=number,
                    title=entry["title"],
                    author_login=user["login"] if user else None,
                    state=entry["state"],
                    created_at=_required_dt(entry["created_at"]),
                    merged_at=_dt(entry.get("merged_at")),
                    closed_at=_dt(entry.get("closed_at")),
                    head_branch=branch,
                    # Prefer the branch name, fall back to the PR title.
                    ticket_key=_ticket_key(branch) or _ticket_key(entry["title"]),
                )
            )

            for review in fetch_reviews(number):
                reviewer = review.get("user")
                review_rows.append(
                    PrReview(
                        id=review["id"],
                        pr_number=number,
                        reviewer_login=reviewer["login"] if reviewer else None,
                        state=review["state"],
                        submitted_at=_dt(review.get("submitted_at")),
                        body=review.get("body") or None,
                    )
                )

        for row in review_rows:
            session.add(row)

        session.commit()

    print(
        f"Loaded {len(commits)} commits, {len(pull_requests)} pull requests, "
        f"{len(review_rows)} reviews"
    )


# ---------------------------------------------------------------------------
# Catalogue question C1: average pull request review turnaround
# ---------------------------------------------------------------------------

# Turnaround is the gap between raising a PR and the first review on it.
# julianday returns a day number, so the difference is multiplied by 24 for hours.
REVIEW_TURNAROUND_SQL = """
SELECT ROUND(AVG((julianday(first_review) - julianday(created_at)) * 24), 1) AS avg_hours
FROM (
    SELECT p.number, p.created_at, MIN(r.submitted_at) AS first_review
    FROM pull_requests p
    JOIN pr_reviews r ON r.pr_number = p.number
    GROUP BY p.number
)
"""

# Question J1: pull requests merged with no review at all.
UNREVIEWED_PRS_SQL = """
SELECT p.number, p.title
FROM pull_requests p
LEFT JOIN pr_reviews r ON r.pr_number = p.number
WHERE p.merged_at IS NOT NULL AND r.id IS NULL
"""


def report() -> None:
    """Print the two source-control answers this data now supports."""
    with Session(engine) as session:
        (hours,) = session.execute(text(REVIEW_TURNAROUND_SQL)).one()
        print(f"C1 — average review turnaround: {hours} hours")

        unreviewed = session.execute(text(UNREVIEWED_PRS_SQL)).all()
        print(f"J1 — merged without a review: {len(unreviewed)}")
        for number, title in unreviewed:
            print(f"  #{number} {title}")


if __name__ == "__main__":
    load()
    print()
    report()