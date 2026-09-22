"""LangGraph agent skeleton — classify, route, compose.

First ticket toward Phase 4 (handoff doc, section 8). Answers the
three questions with a working query today — velocity, PR review
turnaround, unreviewed PRs — and refuses anything else honestly.
"""

import sqlite3
import sys
from typing import Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.github_connector import REVIEW_TURNAROUND_SQL, UNREVIEWED_PRS_SQL
from app.jira_connector import VELOCITY_SQL
from app.llm.provider import get_llm

DB_PATH = "projectpulse.db"


class AgentState(TypedDict, total=False):
    """State passed between nodes. Grows as more paths are added."""

    question: str
    metric_key: str
    rows: list[dict]
    answer: str


class Classification(BaseModel):
    """Structured output for classify_intent: one known query, or 'other'."""

    metric_key: Literal["velocity", "review_turnaround", "unreviewed_prs", "other"] = Field(
        description="Which known query answers the question, or 'other' if none does"
    )


_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Classify the question into exactly one known query, or 'other' if none fits.\n"
     "velocity = sprint velocity, points completed per sprint\n"
     "review_turnaround = average pull request review turnaround time\n"
     "unreviewed_prs = pull requests merged without a review"),
    ("human", "{question}"),
])


def classify_intent(state: AgentState) -> AgentState:
    """Decide which of the three known SQL queries answers the question, or 'other'."""
    chain = _CLASSIFY_PROMPT | get_llm("fast").with_structured_output(Classification)
    result = chain.invoke({"question": state["question"]})
    state["metric_key"] = result.metric_key
    return state


def route(state: AgentState) -> str:
    """Conditional edge: refuse anything the three known queries can't answer."""
    return "refuse" if state["metric_key"] == "other" else "metric"


_QUERY_MAP = {
    "velocity": VELOCITY_SQL,
    "review_turnaround": REVIEW_TURNAROUND_SQL,
    "unreviewed_prs": UNREVIEWED_PRS_SQL,
}


def metric_path(state: AgentState) -> AgentState:
    """Run the SQL for the classified metric and store the rows. No model call."""
    sql = _QUERY_MAP[state["metric_key"]]
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description]
        state["rows"] = [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
    return state


def refuse_path(state: AgentState) -> AgentState:
    """Answer honestly when the question needs document search, not built yet."""
    state["rows"] = []
    state["answer"] = (
        "That needs document search over Confluence, which isn't built yet. "
        "Right now I can answer sprint velocity, PR review turnaround, and "
        "PRs merged without review."
    )
    return state


_COMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the question in one short sentence using only the data given. "
     "Do not add any number that is not in the data."),
    ("human", "Question: {question}\nData: {rows}"),
])


def compose_answer(state: AgentState) -> AgentState:
    """Phrase the SQL rows as a sentence. Skipped if refuse_path already answered."""
    if state.get("answer"):
        return state
    chain = _COMPOSE_PROMPT | get_llm("strong") | StrOutputParser()
    state["answer"] = chain.invoke({"question": state["question"], "rows": state["rows"]})
    return state


graph = StateGraph(AgentState)
graph.add_node("classify_intent", classify_intent)
graph.add_node("metric_path", metric_path)
graph.add_node("refuse_path", refuse_path)
graph.add_node("compose_answer", compose_answer)

graph.add_edge(START, "classify_intent")
graph.add_conditional_edges(
    "classify_intent", route, {"metric": "metric_path", "refuse": "refuse_path"}
)
graph.add_edge("metric_path", "compose_answer")
graph.add_edge("refuse_path", "compose_answer")
graph.add_edge("compose_answer", END)

agent = graph.compile()


if __name__ == "__main__":
    question = sys.argv[1]
    result = agent.invoke({"question": question})
    print(result["answer"])