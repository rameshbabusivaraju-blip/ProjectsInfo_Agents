"""Tests for evaluate.py's comparison logic — AGENTS-33.

Covers _values_match, _rows_match and _run_reference_query only. evaluate()
itself calls the real compiled agent, which makes a real LLM call — same
reasoning AGENTS-30 already established for not mocking graph routing: doing
it properly would need a live provider, which doesn't belong in an offline
test run. evaluate() is meant to be run locally (python -m eval.evaluate),
not exercised here.
"""

import sqlite3
from pathlib import Path

from eval.evaluate import _rows_match, _run_reference_query, _values_match

# ---------------------------------------------------------------------------
# _values_match — exact for everything except a float/float comparison
# ---------------------------------------------------------------------------


def test_values_match_exact_for_non_floats() -> None:
    """Strings and ints must match exactly — no tolerance applies to them."""
    assert _values_match("Sprint 1", "Sprint 1") is True
    assert _values_match("Sprint 1", "Sprint 2") is False
    assert _values_match(5, 5) is True
    assert _values_match(5, 6) is False


def test_values_match_uses_tolerance_for_floats() -> None:
    """C1's average is a float — small floating-point drift must still pass."""
    assert _values_match(3.001, 3.0) is True
    assert _values_match(3.5, 3.0) is False


# ---------------------------------------------------------------------------
# _rows_match — same row count, same columns, same values
# ---------------------------------------------------------------------------


def test_rows_match_true_for_identical_rows() -> None:
    """The exact case a passing golden question hits: agent rows equal expected rows."""
    rows = [{"sprint": "Sprint 1", "delivered": 5.0}]
    assert _rows_match(rows, rows) is True


def test_rows_match_false_for_different_row_count() -> None:
    """A dropped or extra row is always a fail, regardless of what's in it."""
    actual = [{"sprint": "Sprint 1", "delivered": 5.0}]
    expected = [
        {"sprint": "Sprint 1", "delivered": 5.0},
        {"sprint": "Sprint 2", "delivered": 8.0},
    ]
    assert _rows_match(actual, expected) is False


def test_rows_match_false_for_different_values() -> None:
    """A misclassified question would run the wrong query and return the wrong numbers."""
    actual = [{"sprint": "Sprint 1", "delivered": 3.0}]
    expected = [{"sprint": "Sprint 1", "delivered": 5.0}]
    assert _rows_match(actual, expected) is False


# ---------------------------------------------------------------------------
# _run_reference_query — runs real SQL against a real (temp) SQLite file
# ---------------------------------------------------------------------------


def test_run_reference_query_returns_rows_from_db(tmp_path: Path) -> None:
    """No monkeypatched globals needed — db_path is a plain argument, not a hidden import."""
    db_path = tmp_path / "test.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE t (name TEXT, points REAL)")
        conn.execute("INSERT INTO t VALUES ('Sprint 1', 5.0)")
        conn.commit()

    rows = _run_reference_query("SELECT name, points FROM t", str(db_path))

    assert rows == [{"name": "Sprint 1", "points": 5.0}]