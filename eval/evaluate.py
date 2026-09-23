"""evaluate.py — run the golden set against the real agent. AGENTS-33, ADR-008.

For each golden question, the real compiled agent answers it — a real LLM
call, not a mock, closing the gap AGENTS-30/31 left open: routing through
classify_intent was never exercised end to end without one. The rows the
agent retrieved are then compared against the reference query's own result
from the same live database. Answer quality becomes a number, not a feeling.

CI wiring (failing the build on a regression) waits for the GitHub Actions
connector — out of scope for this ticket. Run it locally:

    python -m eval.evaluate
"""

from __future__ import annotations

import sqlite3
import sys
from typing import Any

from app.agent import DB_PATH, agent
from eval.golden_set import GOLDEN_SET, GoldenQuestion


def _run_reference_query(sql: str, db_path: str) -> list[dict[str, Any]]:
    """Run a golden question's reference SQL against the given database."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def _values_match(actual: Any, expected: Any, tolerance: float = 0.01) -> bool:
    """Exact match, except two floats compare within a small tolerance.

    The catalogue's own scoring rule: "matches the reference query result, or
    within a stated tolerance for averages." Only C1 (an AVG) actually needs
    this — A1 and J1 compare exactly.
    """
    if isinstance(actual, float) and isinstance(expected, float):
        return abs(actual - expected) <= tolerance
    return bool(actual == expected)


def _rows_match(actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> bool:
    """Same number of rows, same columns, same values in each."""
    if len(actual) != len(expected):
        return False
    return all(
        set(a) == set(e) and all(_values_match(a[key], e[key]) for key in e)
        for a, e in zip(actual, expected, strict=True)
    )


def evaluate(golden_set: list[GoldenQuestion] = GOLDEN_SET) -> bool:
    """Run every golden question through the real agent; print PASS/FAIL and a pass rate.

    Returns True only if every question passed.
    """
    results = []
    for question in golden_set:
        state = agent.invoke({"question": question.question})
        expected_rows = _run_reference_query(question.reference_sql, DB_PATH)
        actual_rows = state.get("rows", [])
        passed = _rows_match(actual_rows, expected_rows)
        results.append(passed)

        print(f"[{'PASS' if passed else 'FAIL'}] {question.id}: {question.question}")
        if not passed:
            print(f"         classified as: {state.get('metric_key')!r}")
            print(f"         agent rows:    {actual_rows}")
            print(f"         expected rows: {expected_rows}")

    print(f"\n{sum(results)}/{len(results)} passed")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if evaluate() else 1)