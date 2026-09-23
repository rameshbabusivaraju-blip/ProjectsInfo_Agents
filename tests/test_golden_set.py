"""Tests for the golden question set — AGENTS-32, ADR-008/ADR-015.

Guards against the two ways this data could quietly rot: a golden entry
pointing at a metric_key the agent no longer knows about, and a golden
entry's reference_sql drifting away from the SQL the agent actually runs.
"""

from app import agent
from eval.golden_set import GOLDEN_SET


def test_golden_set_covers_exactly_the_starred_metric_questions() -> None:
    """AGENTS-32 is metric-only: A1, C1, J1 — no more, no fewer, until AGENTS-30's queries grow."""
    assert {q.id for q in GOLDEN_SET} == {"A1", "C1", "J1"}


def test_every_metric_key_is_known_to_the_agent() -> None:
    """A golden entry must route through a metric_key the agent can actually classify into."""
    for question in GOLDEN_SET:
        assert question.metric_key in agent._QUERY_MAP


def test_reference_sql_is_the_same_query_the_agent_runs() -> None:
    """No second, hand-duplicated copy of the SQL — drift here would make the eval lie."""
    for question in GOLDEN_SET:
        assert question.reference_sql is agent._QUERY_MAP[question.metric_key]


def test_questions_are_non_empty_and_ids_are_unique() -> None:
    """Basic well-formedness — an empty question or a duplicate id is always a mistake."""
    ids = [q.id for q in GOLDEN_SET]
    assert len(ids) == len(set(ids))
    assert all(q.question.strip() for q in GOLDEN_SET)