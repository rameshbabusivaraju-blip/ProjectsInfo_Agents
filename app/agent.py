"""LangGraph agent skeleton — classify, route, compose.

First ticket toward Phase 4 (handoff doc, section 8). Answers the
three questions with a working query today — velocity, PR review
turnaround, unreviewed PRs — and refuses anything else honestly.

Answers the three known metric questions with a working query — velocity,
PR review turnaround, unreviewed PRs — refuses anything else honestly, and
can write the last sprint's velocity to a formatted Excel file (AGENTS-31,
catalogue question I4, ADR-010).
"""

import sqlite3
import sys
from typing import Any, Literal, TypedDict
 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
 
from app.excel_export import export_to_excel
from app.github_connector import REVIEW_TURNAROUND_SQL, UNREVIEWED_PRS_SQL
from app.jira_connector import VELOCITY_SQL
from app.llm.provider import get_llm
 
DB_PATH = "projectpulse.db"
 
 
class AgentState(TypedDict, total=False):
    """State passed between nodes. Grows as more paths are added."""
 
    question: str
    metric_key: str
    rows: list[dict[str, Any]]
    answer: str
 
class Classification(BaseModel):
    """Structured output for classify_intent: one known query, one action, or 'other'."""
 
    metric_key: Literal[
        "velocity", "review_turnaround", "unreviewed_prs", "export_excel", "other"
    ] = Field(
        description="Which known query answers the question, 'export_excel' to write "
        "the last sprint's velocity to a file, or 'other' if none fits"
    )
 
_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Classify the question into exactly one known query or action, or 'other' if none fits.\n"
     "velocity = sprint velocity, points completed per sprint\n"
     "review_turnaround = average pull request review turnaround time\n"
     "unreviewed_prs = pull requests merged without a review\n"
     "export_excel = write the last sprint's velocity to an Excel file"),
    ("human", "{question}"),
])
 
 
def classify_intent(state: AgentState) -> AgentState:
    """Decide which of the three known SQL queries answers the question, or 'other'."""
    chain = _CLASSIFY_PROMPT | get_llm("fast").with_structured_output(Classification)
    result = chain.invoke({"question": state["question"]})
    state["metric_key"] = result.metric_key
    return state
 
 
def route(state: AgentState) -> str:
    """Conditional edge: send the export action and refusals down their own paths."""
    if state["metric_key"] == "other":
        return "refuse"
    if state["metric_key"] == "export_excel":
        return "export"
    return "metric"
 
 
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
 
 
def export_path(state: AgentState) -> AgentState:
    """Write the last sprint's velocity to a formatted .xlsx. No model call.
 
    Reuses VELOCITY_SQL as-is rather than writing a new query — it already
    orders by start_date, so the last row is the most recent sprint.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(VELOCITY_SQL)
        cols = [d[0] for d in cur.description]
        all_sprints = [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
 
    last_sprint = all_sprints[-1:]
    state["rows"] = last_sprint
    path = export_to_excel(last_sprint, "last_sprint_velocity.xlsx")
    state["answer"] = f"Wrote {len(last_sprint)} row(s) to {path}."
    return state
 
 
def refuse_path(state: AgentState) -> AgentState:
    """Answer honestly when the question needs document search, not built yet."""
    state["rows"] = []
    state["answer"] = (
        "That needs document search over Confluence, which isn't built yet. "
        "Right now I can answer sprint velocity, PR review turnaround, PRs "
        "merged without review, and export the last sprint's velocity to Excel."
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
graph.add_node("export_path", export_path)
graph.add_node("refuse_path", refuse_path)
graph.add_node("compose_answer", compose_answer)
 
graph.add_edge(START, "classify_intent")
graph.add_conditional_edges(
    "classify_intent",
    route,
    {"metric": "metric_path", "export": "export_path", "refuse": "refuse_path"},
)
graph.add_edge("metric_path", "compose_answer")
graph.add_edge("export_path", "compose_answer")
graph.add_edge("refuse_path", "compose_answer")
graph.add_edge("compose_answer", END)
 
agent = graph.compile()
 
 
if __name__ == "__main__":
    question = sys.argv[1]
    result = agent.invoke({"question": question})
    print(result["answer"])