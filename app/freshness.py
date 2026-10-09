"""How old the stored data is (ADR-023).

Each connector calls record_sync() when it finishes a load, so the database knows when each
source was last copied. get_freshness() reads that back and says whether the copy is older than
MAX_STORED_AGE_HOURS. The agent loop adds the result to every stored-data tool result, so the
model can state the age of its answer and knows when a live read is allowed.

This file only reads and writes the sync_status table. It does not call Jira, GitHub or
Confluence.
"""

import os
import sqlite3
from datetime import UTC, datetime
from typing import Any

from sqlmodel import Session

from app.models import SyncStatus

# The one place the age limit lives. A stored copy older than this counts as stale.
MAX_STORED_AGE_HOURS = float(os.environ.get("MAX_STORED_AGE_HOURS", "24"))

# Which source each metric_key reads from. A test checks that every metric key is listed.
SOURCE_FOR_METRIC: dict[str, str] = {
    "velocity": "jira",
    "committed_vs_delivered": "jira",
    "logged_vs_planned_hours": "jira",
    "re_estimated_stories": "jira",
    "open_connectors_stories": "jira",
    "spilled_over_tickets": "jira",
    "review_turnaround": "github",
    "longest_review_wait": "github",
    "unreviewed_prs": "github",
    "long_open_prs": "github",
    "commits_for_ticket": "github",
    "commits_without_ticket": "github",
    "non_convention_branches": "github",
    "dora_per_sprint": "actions",
    "failed_runs_by_stage": "actions",
}


def _now() -> datetime:
    """The current time as naive UTC, like every timestamp in the store. Tests replace this."""
    return datetime.now(UTC).replace(tzinfo=None)


def record_sync(session: Session, source: str) -> None:
    """Note that `source` was copied just now. The caller commits the session."""
    session.merge(SyncStatus(source=source, last_synced_at=_now()))


def get_freshness(db_path: str, source: str) -> dict[str, Any]:
    """The age of the stored copy of one source.

    Returns source, last_synced_at (UTC text, or None if the connector has never run),
    age_hours (one decimal, or None) and stale (True when never synced or older than
    MAX_STORED_AGE_HOURS).
    """
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT last_synced_at FROM sync_status WHERE source = ?", (source,)
            ).fetchone()
    except sqlite3.Error:  # no table yet: no connector has run since this feature was added
        row = None

    if row is None:
        return {"source": source, "last_synced_at": None, "age_hours": None, "stale": True}

    synced = datetime.fromisoformat(row[0])
    age_hours = round((_now() - synced).total_seconds() / 3600, 1)
    return {
        "source": source,
        "last_synced_at": synced.isoformat(timespec="seconds"),
        "age_hours": age_hours,
        "stale": age_hours > MAX_STORED_AGE_HOURS,
    }