"""Golden question set — ADR-008/ADR-015, AGENTS-32.

Metric-only for now: A1, C1, J1, the three catalogue questions the agent can
already answer (AGENTS-29/30). Each entry reuses the exact SQL constant the
agent itself runs for that metric_key, imported from the connector module
that defines it — the same constant app.agent._QUERY_MAP points at, not a
copy of its text.

That is a deliberate choice, not a shortcut. These three metrics each have
exactly one correct SQL shape — there is no second, differently-worded query
that would check the first independently, so writing one would only
duplicate the reviewed query, not verify it. What this golden set checks is
whether the agent classifies a question correctly, calls the right query,
and phrases its result honestly — not whether the query itself is right.
That review happened once, carefully, when the query was written, per
ADR-015's rule that a reference query is reviewed as carefully as production
code.

Narrative and hybrid golden questions wait for the Confluence connector and
use ADR-015's other mechanism, a frozen snapshot under eval/snapshot_<date>/
— out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.github_connector import REVIEW_TURNAROUND_SQL, UNREVIEWED_PRS_SQL
from app.jira_connector import VELOCITY_SQL


@dataclass(frozen=True)
class GoldenQuestion:
    """One golden-set entry: a catalogue question paired with its reference query.

    metric_key must match a key in app.agent._QUERY_MAP. That is what lets
    evaluate.py (AGENTS-33) send `question` through the real agent and check
    its answer against reference_sql's result, without hardcoding SQL twice.
    """

    id: str
    question: str
    metric_key: str
    reference_sql: str


GOLDEN_SET: list[GoldenQuestion] = [
    GoldenQuestion(
        id="A1",
        question="What was the velocity of each sprint?",
        metric_key="velocity",
        reference_sql=VELOCITY_SQL,
    ),
    GoldenQuestion(
        id="C1",
        question="What is the average pull request review turnaround time?",
        metric_key="review_turnaround",
        reference_sql=REVIEW_TURNAROUND_SQL,
    ),
    GoldenQuestion(
        id="J1",
        question="Which pull requests were merged without a review comment?",
        metric_key="unreviewed_prs",
        reference_sql=UNREVIEWED_PRS_SQL,
    ),
]