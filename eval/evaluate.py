"""evaluate.py — run the golden set against the real agent. AGENTS-33, AGENTS-84, ADR-008.

Each golden question is answered by the real agent, a real LLM call and not a mock. A run
passes when every check the question has passes: the rows equal the reference query's own
result from the same database, and each expected page appears among the sources.

AGENTS-84 adds three things:
  - both agents: the fixed agent (app/agent.py) and the tool-calling loop (app/agent_loop.py)
  - repeated runs, because a model can answer differently each time
  - measurements per run: model calls, tokens and an estimated cost

The result is printed as a markdown table and saved under eval/results/. Run it locally:

    python -m eval.evaluate                     # every question, both agents, 3 runs each
    python -m eval.evaluate --runs 1 --mode loop --ids A1 H1

CI wiring waits for a cheaper way to run it; it makes real model calls.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.tracers.context import register_configure_hook

from app.agent import DB_PATH, agent
from app.agent_loop import loop_result, run_loop
from eval.golden_set import GOLDEN_SET, GoldenQuestion

MODES = ("fixed", "loop")
RESULTS_DIR = Path("eval/results")

# Dollars per million tokens (input, output), from Anthropic's pricing page on 9 Oct 2026.
# A model name is matched by prefix, longest first, so dated names such as
# claude-haiku-4-5-20251001 work. A model that is not listed has no cost: the table shows
# "n/a" and the tokens are still counted. Update this when prices or models change.
PRICES_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
}


# ---------------------------------------------------------------------------
# Comparing rows (AGENTS-33)
# ---------------------------------------------------------------------------


def _run_reference_query(
    sql: str, db_path: str, params: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """Run a golden question's reference SQL against the given database.

    params gives the value for each :name placeholder in the SQL, if it has any.
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(sql, params or {})
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


def _has_source(sources: list[dict[str, Any]], needle: str) -> bool:
    """True when some source page has a title containing the needle, ignoring case."""
    return any(needle.lower() in str(source.get("title", "")).lower() for source in sources)


def _check(question: GoldenQuestion, result: dict[str, Any], db_path: str) -> str:
    """Judge one answer. Returns "" when every check passes, else the reasons, joined."""
    failures: list[str] = []
    if question.reference_sql is not None:
        expected = _run_reference_query(question.reference_sql, db_path, question.params)
        actual = result.get("rows", [])
        if not _rows_match(actual, expected):
            failures.append(f"rows differ (got {len(actual)} rows, expected {len(expected)})")
    sources = result.get("sources", [])
    for needle in question.expected_sources:
        if not _has_source(sources, needle):
            failures.append(f"no source page titled like '{needle}'")
    return "; ".join(failures)


# ---------------------------------------------------------------------------
# Measuring model calls, tokens and cost (AGENTS-84)
# ---------------------------------------------------------------------------


class _Meter(UsageMetadataCallbackHandler):
    """Counts model calls and adds up tokens per model, for everything run while it is set."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.calls += 1
        super().on_llm_end(response, **kwargs)


# LangChain hands this handler to every model call made while it is set, however deep inside
# the graph the call is. The hook is registered once, when this module loads.
_METER: ContextVar[_Meter | None] = ContextVar("projectpulse_eval_meter", default=None)
register_configure_hook(_METER, inheritable=True)


def _price(model: str) -> tuple[float, float] | None:
    """The (input, output) price per million tokens for a model name, or None if unlisted."""
    for prefix in sorted(PRICES_PER_MILLION_TOKENS, key=len, reverse=True):
        if model.startswith(prefix):
            return PRICES_PER_MILLION_TOKENS[prefix]
    return None


def _tokens_and_cost(meter: _Meter) -> tuple[int, float | None]:
    """Total tokens, and the cost in dollars. The cost is None if any model has no price."""
    tokens = 0
    cost = 0.0
    priced = True
    for model, usage in meter.usage_metadata.items():
        tokens += usage["input_tokens"] + usage["output_tokens"]
        price = _price(model)
        if price is None:
            priced = False
        else:
            cost += (usage["input_tokens"] * price[0] + usage["output_tokens"] * price[1]) / 1e6
    return tokens, (cost if priced else None)


# ---------------------------------------------------------------------------
# Running questions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    """One answer to one question. failure is "" when the run passed."""

    failure: str
    calls: int
    tokens: int
    cost: float | None
    live_calls: int

    @property
    def passed(self) -> bool:
        return not self.failure


@dataclass(frozen=True)
class Summary:
    """All runs of one question in one mode."""

    question: GoldenQuestion
    mode: str
    runs: list[RunResult]

    @property
    def passes(self) -> int:
        return sum(run.passed for run in self.runs)

    @property
    def avg_calls(self) -> float:
        return sum(run.calls for run in self.runs) / len(self.runs)

    @property
    def avg_tokens(self) -> float:
        return sum(run.tokens for run in self.runs) / len(self.runs)

    @property
    def avg_live_calls(self) -> float:
        return sum(run.live_calls for run in self.runs) / len(self.runs)

    @property
    def total_cost(self) -> float | None:
        """The cost of all runs together, or None if any run could not be priced."""
        costs = [run.cost for run in self.runs]
        if any(cost is None for cost in costs):
            return None
        return sum(cost for cost in costs if cost is not None)


def _answer(question: str, mode: str) -> tuple[dict[str, Any], int]:
    """Answer one question with the chosen agent. Returns the /ask fields and the live calls."""
    if mode == "fixed":
        return dict(agent.invoke({"question": question})), 0
    state = run_loop(question)
    return loop_result(state), state.get("live_calls", 0)


def run_once(question: GoldenQuestion, mode: str, db_path: str = DB_PATH) -> RunResult:
    """Answer the question once and measure it. A provider error counts as a failed run."""
    meter = _Meter()
    token = _METER.set(meter)
    live_calls = 0
    try:
        result, live_calls = _answer(question.question, mode)
        failure = _check(question, result, db_path)
    except Exception as error:  # a rate limit or outage must not lose the other results
        failure = f"error: {type(error).__name__}: {error}"
    finally:
        _METER.reset(token)
    tokens, cost = _tokens_and_cost(meter)
    return RunResult(failure, meter.calls, tokens, cost, live_calls)


def evaluate(
    golden_set: Sequence[GoldenQuestion] = GOLDEN_SET,
    modes: Sequence[str] = MODES,
    runs: int = 3,
    db_path: str = DB_PATH,
) -> list[Summary]:
    """Run every question `runs` times in every mode and print one line per run."""
    summaries: list[Summary] = []
    for question in golden_set:
        for mode in modes:
            results: list[RunResult] = []
            for number in range(1, runs + 1):
                run = run_once(question, mode, db_path)
                results.append(run)
                label = "PASS" if run.passed else "FAIL"
                print(
                    f"[{label}] {question.id} {mode} run {number}/{runs}: "
                    f"{run.calls} calls, {run.tokens:,} tokens"
                    + (f" -- {run.failure}" if run.failure else "")
                )
            summaries.append(Summary(question, mode, results))
    return summaries


# ---------------------------------------------------------------------------
# The results table
# ---------------------------------------------------------------------------


def _money(cost: float | None) -> str:
    return "n/a" if cost is None else f"${cost:.4f}"


def format_table(summaries: Sequence[Summary], runs: int, day: date) -> str:
    """The results as markdown: one row per question and mode, then totals per mode."""
    lines = [
        f"# Golden set results, {day.isoformat()}",
        "",
        f"Each question was answered {runs} time(s) per agent. A run passes when every check "
        "of the question passes. Calls, tokens and live calls are averages per run; cost is "
        "the total for all runs of that row, estimated from "
        "`PRICES_PER_MILLION_TOKENS` in `eval/evaluate.py`.",
        "",
        "| Question | Agent | Passed | Model calls | Tokens | Live calls | Cost |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in summaries:
        lines.append(
            f"| {item.question.id} | {item.mode} | {item.passes}/{len(item.runs)} "
            f"| {item.avg_calls:.1f} | {item.avg_tokens:,.0f} | {item.avg_live_calls:.1f} "
            f"| {_money(item.total_cost)} |"
        )

    lines += ["", "## Totals", "", "| Agent | Passed | Pass rate | Cost |", "|---|---|---|---|"]
    for mode in dict.fromkeys(item.mode for item in summaries):
        rows = [item for item in summaries if item.mode == mode]
        passed = sum(item.passes for item in rows)
        total = sum(len(item.runs) for item in rows)
        costs = [item.total_cost for item in rows]
        cost = None if any(c is None for c in costs) else sum(c for c in costs if c is not None)
        lines.append(f"| {mode} | {passed}/{total} | {passed / total:.0%} | {_money(cost)} |")

    failed = [(item, run.failure) for item in summaries for run in item.runs if run.failure]
    if failed:
        lines += ["", "## Failures", ""]
        seen: set[tuple[str, str, str]] = set()
        for item, failure in failed:
            key = (item.question.id, item.mode, failure)
            if key not in seen:
                seen.add(key)
                lines.append(f"- {item.question.id} ({item.mode}): {failure}")
    return "\n".join(lines) + "\n"


def pass_rate(summaries: Sequence[Summary], mode: str) -> float:
    """The share of runs that passed in one mode, from 0 to 1."""
    rows = [item for item in summaries if item.mode == mode]
    total = sum(len(item.runs) for item in rows)
    return sum(item.passes for item in rows) / total if total else 0.0


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the golden set against the real agent.")
    parser.add_argument("--runs", type=int, default=3, help="runs per question and agent")
    parser.add_argument("--mode", choices=[*MODES, "both"], default="both")
    parser.add_argument("--ids", nargs="+", help="only these question ids, such as A1 H1")
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.0,
        help="exit with an error if any agent's pass rate (0 to 1) is below this",
    )
    parser.add_argument("--out", type=Path, default=RESULTS_DIR, help="folder for the results")
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluation, print and save the table. Returns the process exit code."""
    args = _parse_args(argv)
    questions = [q for q in GOLDEN_SET if args.ids is None or q.id in args.ids]
    unknown = set(args.ids or []) - {q.id for q in GOLDEN_SET}
    if unknown:
        print(f"Unknown question ids: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    modes = MODES if args.mode == "both" else (args.mode,)
    print(f"{len(questions)} questions x {len(modes)} agent(s) x {args.runs} run(s)\n")

    summaries = evaluate(questions, modes, args.runs)

    table = format_table(summaries, args.runs, date.today())
    args.out.mkdir(parents=True, exist_ok=True)
    # The time is in the name so a second run on the same day does not overwrite the first.
    path = args.out / f"results_{datetime.now():%Y-%m-%d_%H%M}.md"
    path.write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"Saved to {path}")

    low = [mode for mode in modes if pass_rate(summaries, mode) < args.min_pass_rate]
    return 1 if low else 0


if __name__ == "__main__":
    sys.exit(main())
