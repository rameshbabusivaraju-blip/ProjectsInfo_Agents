"""Tests for evaluate.py's comparison logic — AGENTS-33.

Covers the comparison helpers, and (AGENTS-84) the checks, the usage meter, the
summaries and the results table. The real agents make real LLM calls, which
don't belong in an offline test run, so _answer is replaced by a fake that
returns prepared answers. The real run is done locally (python -m eval.evaluate).
"""

import re
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from eval import evaluate as ev
from eval.evaluate import _rows_match, _run_reference_query, _values_match
from eval.golden_set import GoldenQuestion

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

def test_run_reference_query_fills_a_placeholder_from_params(tmp_path: Path) -> None:
    """A reference query with a :name placeholder must get its value from params (C3)."""
    db_path = tmp_path / "test.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE t (ticket TEXT, n INTEGER)")
        conn.execute("INSERT INTO t VALUES ('AGENTS-14', 1), ('AGENTS-15', 2)")
        conn.commit()

    rows = _run_reference_query(
        "SELECT n FROM t WHERE ticket = :ticket_key", str(db_path), {"ticket_key": "AGENTS-14"}
    )

    assert rows == [{"n": 1}]



# ---------------------------------------------------------------------------
# _check — the rows and the expected source pages (AGENTS-84)
# ---------------------------------------------------------------------------


def _db_with_velocity(tmp_path: Path) -> str:
    db_path = tmp_path / "check.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE v (sprint TEXT, delivered REAL)")
        conn.execute("INSERT INTO v VALUES ('Sprint 1', 5.0)")
        conn.commit()
    return str(db_path)


VELOCITY_QUESTION = GoldenQuestion(
    id="T1",
    question="velocity?",
    reference_sql="SELECT sprint, delivered FROM v",
    expected_sources=("retro",),
)
GOOD_ANSWER = {
    "rows": [{"sprint": "Sprint 1", "delivered": 5.0}],
    "sources": [{"title": "Sprint 3 Retrospective"}],
}


def test_check_passes_when_rows_and_sources_both_match(tmp_path: Path) -> None:
    assert ev._check(VELOCITY_QUESTION, GOOD_ANSWER, _db_with_velocity(tmp_path)) == ""


def test_check_names_each_reason_for_failing(tmp_path: Path) -> None:
    answer: dict[str, Any] = {"rows": [], "sources": [{"title": "03 Decision Log"}]}

    failure = ev._check(VELOCITY_QUESTION, answer, _db_with_velocity(tmp_path))

    assert "rows differ (got 0 rows, expected 1)" in failure
    assert "no source page titled like 'retro'" in failure


def test_source_titles_are_matched_ignoring_case() -> None:
    assert ev._has_source([{"title": "03 Decision Log"}], "decision log") is True
    assert ev._has_source([{"title": "RAID Log"}], "decision log") is False
    assert ev._has_source([], "decision log") is False


def test_a_question_without_sql_is_judged_on_sources_only(tmp_path: Path) -> None:
    question = GoldenQuestion(id="T2", question="how?", expected_sources=("decision log",))

    assert ev._check(question, {"sources": [{"title": "03 Decision Log"}]}, "unused.db") == ""
    assert ev._check(question, {"rows": [{"a": 1}]}, "unused.db") != ""


# ---------------------------------------------------------------------------
# The usage meter and the cost
# ---------------------------------------------------------------------------


def _llm_result(model: str, tokens_in: int, tokens_out: int) -> LLMResult:
    message = AIMessage(
        content="x",
        usage_metadata={
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "total_tokens": tokens_in + tokens_out,
        },
        response_metadata={"model_name": model},
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_meter_counts_calls_and_adds_tokens_per_model() -> None:
    meter = ev._Meter()
    meter.on_llm_end(_llm_result("claude-haiku-4-5-20251001", 1000, 200))
    meter.on_llm_end(_llm_result("claude-sonnet-5", 3000, 500))
    meter.on_llm_end(_llm_result("claude-sonnet-5", 1000, 100))

    tokens, cost = ev._tokens_and_cost(meter)

    assert meter.calls == 3
    assert tokens == 1000 + 200 + 3000 + 500 + 1000 + 100
    # haiku: 1000 * $1 + 200 * $5 ; sonnet: 4000 * $2 + 600 * $10 ; per million
    assert cost == pytest.approx((1000 * 1 + 200 * 5 + 4000 * 2 + 600 * 10) / 1e6)


def test_cost_is_none_when_a_model_has_no_price() -> None:
    meter = ev._Meter()
    meter.on_llm_end(_llm_result("claude-haiku-4-5-20251001", 1000, 100))
    meter.on_llm_end(_llm_result("some-other-model", 1000, 100))

    tokens, cost = ev._tokens_and_cost(meter)

    assert tokens == 2200
    assert cost is None


def test_the_longest_matching_price_prefix_wins() -> None:
    assert ev._price("claude-sonnet-5-5-20261001") == (2.0, 10.0)
    assert ev._price("claude-haiku-4-5-20251001") == (1.0, 5.0)
    assert ev._price("gpt-4o") is None


# ---------------------------------------------------------------------------
# run_once, evaluate and the results table (with a fake agent)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_answers(monkeypatch: pytest.MonkeyPatch) -> list[tuple[dict[str, Any], int]]:
    """Replace _answer: each call returns the next prepared answer and records one model call."""
    prepared: list[tuple[dict[str, Any], int]] = []

    def fake_answer(question: str, mode: str) -> tuple[dict[str, Any], int]:
        meter = ev._METER.get()
        assert meter is not None  # run_once must set the meter before answering
        meter.on_llm_end(_llm_result("claude-haiku-4-5-20251001", 1000, 100))
        return prepared.pop(0)

    monkeypatch.setattr(ev, "_answer", fake_answer)
    return prepared


def test_run_once_measures_the_answer_and_clears_the_meter(
    fake_answers: list[tuple[dict[str, Any], int]], tmp_path: Path
) -> None:
    fake_answers.append((GOOD_ANSWER, 2))

    run = ev.run_once(VELOCITY_QUESTION, "loop", _db_with_velocity(tmp_path))

    assert run.passed
    assert (run.calls, run.tokens, run.live_calls) == (1, 1100, 2)
    assert run.cost == pytest.approx((1000 * 1 + 100 * 5) / 1e6)
    assert ev._METER.get() is None


def test_run_once_turns_a_provider_error_into_a_failed_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def broken(question: str, mode: str) -> tuple[dict[str, Any], int]:
        raise RuntimeError("rate limit")

    monkeypatch.setattr(ev, "_answer", broken)

    run = ev.run_once(VELOCITY_QUESTION, "fixed", _db_with_velocity(tmp_path))

    assert not run.passed
    assert run.failure == "error: RuntimeError: rate limit"
    assert ev._METER.get() is None


def _summaries(
    fake_answers: list[tuple[dict[str, Any], int]], tmp_path: Path
) -> list[ev.Summary]:
    """Two modes, 2 runs: fixed passes once, loop passes twice."""
    bad: dict[str, Any] = {"rows": [], "sources": []}
    fake_answers.extend([(GOOD_ANSWER, 0), (bad, 0), (GOOD_ANSWER, 1), (GOOD_ANSWER, 3)])
    return ev.evaluate([VELOCITY_QUESTION], ev.MODES, 2, _db_with_velocity(tmp_path))


def test_evaluate_runs_every_mode_the_given_number_of_times(
    fake_answers: list[tuple[dict[str, Any], int]], tmp_path: Path
) -> None:
    summaries = _summaries(fake_answers, tmp_path)

    assert [(s.mode, s.passes, len(s.runs)) for s in summaries] == [("fixed", 1, 2), ("loop", 2, 2)]
    assert summaries[1].avg_live_calls == 2.0
    assert summaries[0].avg_calls == 1.0
    assert summaries[0].avg_tokens == 1100.0
    assert ev.pass_rate(summaries, "fixed") == 0.5
    assert ev.pass_rate(summaries, "loop") == 1.0


def test_the_table_has_a_row_per_question_and_mode_and_lists_failures(
    fake_answers: list[tuple[dict[str, Any], int]], tmp_path: Path
) -> None:
    summaries = _summaries(fake_answers, tmp_path)

    table = ev.format_table(summaries, 2, date(2026, 10, 10))

    assert table.startswith("# Golden set results, 2026-10-10")
    assert "| T1 | fixed | 1/2 | 1.0 | 1,100 | 0.0 | $0.0030 |" in table
    assert "| T1 | loop | 2/2 | 1.0 | 1,100 | 2.0 | $0.0030 |" in table
    assert "| fixed | 1/2 | 50% | $0.0030 |" in table
    assert "| loop | 2/2 | 100% | $0.0030 |" in table
    assert "- T1 (fixed): rows differ (got 0 rows, expected 1); no source page titled" in table


def test_an_unpriced_cost_shows_as_n_a() -> None:
    run = ev.RunResult(failure="", calls=1, tokens=10, cost=None, live_calls=0)
    table = ev.format_table([ev.Summary(VELOCITY_QUESTION, "fixed", [run])], 1, date(2026, 10, 10))

    assert "| T1 | fixed | 1/1 | 1.0 | 10 | 0.0 | n/a |" in table


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_main_saves_the_table_and_returns_the_exit_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = ev.RunResult(failure="", calls=1, tokens=10, cost=0.0, live_calls=0)
    seen: dict[str, Any] = {}

    def fake_evaluate(questions: list[GoldenQuestion], modes: Any, runs: int) -> list[ev.Summary]:
        seen.update(ids=[q.id for q in questions], modes=tuple(modes), runs=runs)
        return [ev.Summary(questions[0], "loop", [run])]

    monkeypatch.setattr(ev, "evaluate", fake_evaluate)

    code = ev.main(["--runs", "1", "--mode", "loop", "--ids", "A1", "--out", str(tmp_path)])

    assert code == 0
    assert seen == {"ids": ["A1"], "modes": ("loop",), "runs": 1}
    saved = list(tmp_path.glob("results_*.md"))
    assert len(saved) == 1
    assert re.fullmatch(r"results_\d{4}-\d{2}-\d{2}_\d{4}\.md", saved[0].name)
    assert "| A1 | loop | 1/1 |" in saved[0].read_text(encoding="utf-8")


def test_main_fails_when_a_pass_rate_is_below_the_minimum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bad = ev.RunResult(failure="rows differ", calls=1, tokens=10, cost=0.0, live_calls=0)
    monkeypatch.setattr(
        ev, "evaluate", lambda questions, modes, runs: [ev.Summary(questions[0], "loop", [bad])]
    )

    arguments = ["--mode", "loop", "--ids", "A1", "--min-pass-rate", "0.8", "--out", str(tmp_path)]
    code = ev.main(arguments)

    assert code == 1


def test_main_rejects_an_unknown_question_id(capsys: pytest.CaptureFixture[str]) -> None:
    assert ev.main(["--ids", "ZZ9"]) == 2
    assert "Unknown question ids: ZZ9" in capsys.readouterr().err
