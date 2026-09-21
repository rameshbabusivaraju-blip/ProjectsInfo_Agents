"""Check the local store for broken links and for gaps in process coverage.

Two separate ideas live here and they are treated differently.

Referential integrity means every reference points at something that exists —
a work log naming a ticket that is in the database, a review naming a pull
request that is in the database. A break here is a bug in the connectors, so
it fails the script.

Process coverage means the development process left the trail it should have —
a finished ticket having a commit, a pull request and logged hours against it.
A gap here is not a bug. It is a finding about how the project was run, and it
is reported rather than failed.

Run it with:  python -m app.verify
"""

from __future__ import annotations

import sys

from sqlalchemy import text
from sqlmodel import Session

from app.db import engine

# ---------------------------------------------------------------------------
# Referential integrity — a result here means the data is broken
# ---------------------------------------------------------------------------

INTEGRITY_CHECKS: dict[str, str] = {
    "ticket_sprints rows naming a ticket that does not exist": """
        SELECT ticket_key FROM ticket_sprints
        WHERE ticket_key NOT IN (SELECT key FROM tickets)
    """,
    "ticket_sprints rows naming a sprint that does not exist": """
        SELECT sprint_id FROM ticket_sprints
        WHERE sprint_id NOT IN (SELECT id FROM sprints)
    """,
    "work_logs naming a ticket that does not exist": """
        SELECT id FROM work_logs
        WHERE ticket_key NOT IN (SELECT key FROM tickets)
    """,
    "work_logs naming a person that does not exist": """
        SELECT id FROM work_logs
        WHERE author_id NOT IN (SELECT account_id FROM people)
    """,
    "pr_reviews naming a pull request that does not exist": """
        SELECT id FROM pr_reviews
        WHERE pr_number NOT IN (SELECT number FROM pull_requests)
    """,
    "tickets whose parent does not exist": """
        SELECT key FROM tickets
        WHERE parent_key IS NOT NULL
          AND parent_key NOT IN (SELECT key FROM tickets)
    """,
}

# ---------------------------------------------------------------------------
# Process coverage — a result here is a finding about how the work was done
# ---------------------------------------------------------------------------

# Only level-0 issues are checked. Epics are containers, so an epic with no
# commit against it is normal rather than a gap.
COVERAGE_CHECKS: dict[str, str] = {
    "completed tickets with no commit": """
        SELECT t.key, t.summary FROM tickets t
        WHERE t.hierarchy_level = 0
          AND t.status_category = 'done'
          AND NOT EXISTS (SELECT 1 FROM commits c WHERE c.ticket_key = t.key)
    """,
    "completed tickets with no pull request": """
        SELECT t.key, t.summary FROM tickets t
        WHERE t.hierarchy_level = 0
          AND t.status_category = 'done'
          AND NOT EXISTS (
              SELECT 1 FROM pull_requests p WHERE p.ticket_key = t.key
          )
    """,
    "completed tickets with no hours logged": """
        SELECT t.key, t.summary FROM tickets t
        WHERE t.hierarchy_level = 0
          AND t.status_category = 'done'
          AND NOT EXISTS (SELECT 1 FROM work_logs w WHERE w.ticket_key = t.key)
    """,
    "commits with no ticket id in the message": """
        SELECT sha, message FROM commits WHERE ticket_key IS NULL
    """,
    "pull requests with no ticket id in the branch or title": """
        SELECT number, title FROM pull_requests WHERE ticket_key IS NULL
    """,
}


def _run(sql: str) -> list[tuple[object, ...]]:
    """Run one check and return its rows, empty if the check found nothing."""
    with Session(engine) as session:
        return [tuple(row) for row in session.execute(text(sql)).all()]


def check_integrity() -> int:
    """Report broken references. Returns the number of problems found."""
    print("=== Referential integrity ===")
    problems = 0

    for description, sql in INTEGRITY_CHECKS.items():
        rows = _run(sql)
        if rows:
            problems += len(rows)
            print(f"  FAIL  {description}: {len(rows)}")
            for row in rows[:5]:
                print(f"          {row}")
        else:
            print(f"  ok    {description}")

    return problems


def check_coverage() -> None:
    """Report gaps in the process trail. Never fails the script."""
    print("\n=== Process coverage ===")

    for description, sql in COVERAGE_CHECKS.items():
        rows = _run(sql)
        if rows:
            print(f"  {len(rows)} {description}")
            for row in rows:
                print(f"      {row[0]} — {str(row[1])[:70]}")
        else:
            print(f"  0 {description}")


if __name__ == "__main__":
    failures = check_integrity()
    check_coverage()

    if failures:
        print(f"\n{failures} integrity problems found.")
        sys.exit(1)

    print("\nIntegrity checks passed.")