"""LangGraph agent skeleton — classify, route, compose.

First ticket toward Phase 4 (handoff doc, section 8). Answers the
three questions with a working query today — velocity, PR review
turnaround, unreviewed PRs — and refuses anything else honestly.

Answers the three known metric questions with a working query — velocity,
PR review turnaround, unreviewed PRs — and can write the last sprint's
velocity to a formatted Excel file (AGENTS-31, catalogue question I4,
ADR-010).

AGENTS-39 adds the fourth path: narrative questions, answered by AGENTS-38's
filtered retrieval over the Confluence documents rather than SQL. Only a
genuine "other" now falls through to refuse_path.

AGENTS-41 adds the fifth path: hybrid questions, which need a database number
and a document search in the same answer. hybrid_path runs metric_path and
narrative_path exactly as they already are and merges their rows -- no new
SQL or retrieval code, just combining what the other two paths already do.
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
from app.retrieval import search

DB_PATH = "projectpulse.db"


class AgentState(TypedDict, total=False):
    """State passed between nodes. Grows as more paths are added."""

    question: str
    metric_key: str
    doc_type: str
    hybrid_metric_key: str
    rows: list[dict[str, Any]]
    answer: str

class Classification(BaseModel):
    """Structured output for classify_intent.

    One known query, a narrative lookup, a hybrid of both, one action to
    take, or 'other'.
    """

    metric_key: Literal[
        "velocity", "review_turnaround", "unreviewed_prs", "export_excel",
        "narrative", "hybrid", "other",
    ] = Field(
        description="Which known query answers the question, 'narrative' if it is answered by "
        "searching the project's own documents rather than a database query, 'hybrid' if it "
        "needs both a database number and a document search in the same answer, 'export_excel' "
        "to write the last sprint's velocity to a file, or 'other' if none fits"
    )
    doc_type: Literal[
        "charter", "decision_log", "general", "plan_of_action", "pm_notes",
        "question_catalogue", "raid_log", "retro", "sprint_summary",
    ] | None = Field(
        default=None,
        description="Which document type to search. Set when metric_key is 'narrative' or "
        "'hybrid'. Matches AGENTS-37's per-doc-type FAISS indexes, hand-maintained same as "
        "doc_type_rules.json (RAID log A2: the document set stays small enough for that).",
    )
    hybrid_metric_key: Literal["velocity", "review_turnaround", "unreviewed_prs"] | None = Field(
        default=None,
        description="Which metric query supplies the number half of a hybrid answer. Only set "
        "when metric_key is 'hybrid'.",
    )

_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Classify the question into exactly one known query or action, or 'other' if none fits.\n"
     "velocity = sprint velocity, points completed per sprint\n"
     "review_turnaround = average pull request review turnaround time\n"
     "unreviewed_prs = pull requests merged without a review\n"
     "export_excel = write the last sprint's velocity to an Excel file\n"
     "narrative = answered by searching the project's own documents, not a database query -- "
     "when you pick this, also set doc_type to whichever of charter, decision_log, general, "
     "plan_of_action, pm_notes, question_catalogue, raid_log, retro, sprint_summary the "
     "question is actually about\n"
     "hybrid = needs both a database number and a document search in the same answer -- when "
     "you pick this, set doc_type as above AND hybrid_metric_key to whichever of velocity, "
     "review_turnaround, unreviewed_prs supplies the number"),
    ("human", "{question}"),
])


def classify_intent(state: AgentState) -> AgentState:
    """Decide which known query, action or document search answers the question, or 'other'."""
    chain = _CLASSIFY_PROMPT | get_llm("fast").with_structured_output(Classification)
    result = chain.invoke({"question": state["question"]})
    state["metric_key"] = result.metric_key
    if result.doc_type:
        state["doc_type"] = result.doc_type
    if result.hybrid_metric_key:
        state["hybrid_metric_key"] = result.hybrid_metric_key
    return state


def route(state: AgentState) -> str:
    """Conditional edge: routes export, narrative, hybrid and refusal cases to their own paths."""
    if state["metric_key"] == "other":
        return "refuse"
    if state["metric_key"] == "export_excel":
        return "export"
    if state["metric_key"] == "narrative":
        return "narrative"
    if state["metric_key"] == "hybrid":
        return "hybrid"
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
 
 
def narrative_path(state: AgentState) -> AgentState:
    """Retrieve the top matching chunks for a narrative question (AGENTS-39).

    Mirrors metric_path: it only gathers data into state["rows"], in the same
    shape compose_answer already expects, so compose_answer needs no change to
    handle retrieval instead of SQL. Uses AGENTS-38's search(), which confines
    the search to state["doc_type"] and applies whatever ticket/date filters
    were passed (none, here -- this ticket only wires the plain lookup in).

    A combined filter or a genuinely unmatched question can make search()
    return nothing (ADR-003: accept lower recall). That is answered honestly
    here rather than handed to compose_answer, the same way refuse_path
    answers honestly instead of guessing -- an empty prompt would tempt the
    strong-tier model to invent an answer from outside the retrieved data.
    """
    doc_type = state["doc_type"]
    results = search(state["question"], doc_type)
    if not results:
        state["rows"] = []
        state["answer"] = f"No matching content found in the project's {doc_type} documents."
        return state
    state["rows"] = [{"text": r.text, "doc_id": r.doc_id, "score": r.score} for r in results]
    return state


def hybrid_path(state: AgentState) -> AgentState:
    """Run the SQL metric and the document search, and merge their rows (AGENTS-41).

    Reuses metric_path and narrative_path exactly as they already are -- this
    node's only job is to call both and combine what they put in
    state["rows"], so compose_answer (which already just reads state["rows"])
    needs no change to summarise a hybrid answer instead of a single-source one.
    """
    question, doc_type = state["question"], state["doc_type"]
    metric_rows = metric_path({"metric_key": state["hybrid_metric_key"]})["rows"]
    narrative_rows = narrative_path({"question": question, "doc_type": doc_type})["rows"]
    state["rows"] = metric_rows + narrative_rows
    return state


def refuse_path(state: AgentState) -> AgentState:
    """Answer honestly when the question matches none of the known paths."""
    state["rows"] = []
    state["answer"] = (
        "I can answer sprint velocity, PR review turnaround, PRs merged "
        "without review, export the last sprint's velocity to Excel, "
        "questions answered by the project's own documents, or questions "
        "that combine a number with the project's own documents. This "
        "question doesn't match any of those."
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
graph.add_node("narrative_path", narrative_path)
graph.add_node("hybrid_path", hybrid_path)
graph.add_node("refuse_path", refuse_path)
graph.add_node("compose_answer", compose_answer)

graph.add_edge(START, "classify_intent")
graph.add_conditional_edges(
    "classify_intent",
    route,
    {
        "metric": "metric_path",
        "export": "export_path",
        "narrative": "narrative_path",
        "hybrid": "hybrid_path",
        "refuse": "refuse_path",
    },
)
graph.add_edge("metric_path", "compose_answer")
graph.add_edge("export_path", "compose_answer")
graph.add_edge("narrative_path", "compose_answer")
graph.add_edge("hybrid_path", "compose_answer")
graph.add_edge("refuse_path", "compose_answer")
graph.add_edge("compose_answer", END)
 
agent = graph.compile()
 
 
if __name__ == "__main__":
    question = sys.argv[1]
    result = agent.invoke({"question": question})
    print(result["answer"])