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
from app.github_connector import (
    COMMITS_FOR_TICKET_SQL,
    COMMITS_WITHOUT_TICKET_SQL,
    LONG_OPEN_PRS_SQL,
    LONGEST_REVIEW_WAIT_SQL,
    NON_CONVENTION_BRANCHES_SQL,
    REVIEW_TURNAROUND_SQL,
    TICKET_PATTERN,
    UNREVIEWED_PRS_SQL,
)
from app.jira_connector import (
    COMMITTED_VS_DELIVERED_SQL,
    LOGGED_VS_PLANNED_HOURS_SQL,
    OPEN_CONNECTORS_STORIES_SQL,
    RE_ESTIMATED_STORIES_SQL,
    VELOCITY_SQL,
)
from app.llm.provider import get_llm
from app.retrieval import SearchResult, search

DB_PATH = "projectpulse.db"


class AgentState(TypedDict, total=False):
    """State passed between nodes. Grows as more paths are added."""

    question: str
    metric_key: str
    doc_type: str
    hybrid_metric_key: str
    ticket_key: str
    search_query: str
    rows: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    answer: str
    file_path: str

class Classification(BaseModel):
    """Structured output for classify_intent.

    One known query, a narrative lookup, a hybrid of both, one action to
    take, or 'other'.
    """

    metric_key: Literal[
        "velocity", "committed_vs_delivered", "logged_vs_planned_hours", "re_estimated_stories",
        "review_turnaround", "longest_review_wait", "unreviewed_prs", "long_open_prs",
        "commits_for_ticket", "commits_without_ticket", "non_convention_branches",
        "open_connectors_stories", "export_excel", "narrative", "hybrid", "other",
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
    search_query: str | None = Field(
        default=None,
        description="Set when metric_key is 'narrative' or 'hybrid'. The question rewritten as "
        "a search phrase for the project documents: drop filler words and use the wording the "
        "documents would use (for example 'pull request review and approval' for 'how are code "
        "reviews done'). A short generic question searches badly as written.",
    )
    hybrid_metric_key: Literal[
        "velocity", "committed_vs_delivered", "logged_vs_planned_hours", "re_estimated_stories",
        "review_turnaround", "longest_review_wait", "unreviewed_prs", "long_open_prs",
        "commits_for_ticket", "commits_without_ticket", "non_convention_branches",
        "open_connectors_stories",
    ] | None = Field(
        default=None,
        description="Which metric query supplies the number half of a hybrid answer. Only set "
        "when metric_key is 'hybrid'. Any metric query can be used.",
    )
    ticket_key: str | None = Field(
        default=None,
        description="Set when the question names one ticket, such as 'AGENTS-14', 'AGENTS 14' or "
        "'ticket 14'. Always write it as AGENTS-14. Needed when metric_key is "
        "'commits_for_ticket' (or hybrid_metric_key is).",
    )

_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Classify the question into exactly one known query or action, or 'other' if none fits.\n"
     "velocity = sprint velocity, points completed per sprint\n"
     "committed_vs_delivered = story points committed versus delivered per sprint\n"
     "logged_vs_planned_hours = hours logged versus planned (estimated) hours per sprint\n"
     "re_estimated_stories = stories whose story points were changed after the sprint started\n"
     "review_turnaround = average pull request review turnaround time\n"
     "longest_review_wait = the single longest time a pull request waited for its first review\n"
     "unreviewed_prs = pull requests merged without a review\n"
     "long_open_prs = pull requests that were open for more than three days, whether they are "
     "still open or were merged or closed after that long\n"
     "commits_for_ticket = the commits that belong to one named ticket -- when you pick this, "
     "also set ticket_key, written like AGENTS-14\n"
     "commits_without_ticket = how many commits went in without a ticket ID in the message\n"
     "non_convention_branches = which pull request branches do not follow the "
     "AGENTS-<n>-description naming convention\n"
     "open_connectors_stories = how many stories (tickets) are still open under the "
     "Connectors epic\n"
     "export_excel = write the last sprint's velocity to an Excel file\n"
     "narrative = answered by searching the project's own documents, not a database query -- "
     "when you pick this, also set doc_type to whichever of charter, decision_log, general, "
     "plan_of_action, pm_notes, question_catalogue, raid_log, retro, sprint_summary the "
     "question is actually about\n"
     "What each doc_type holds:\n"
     "decision_log = architecture decisions (ADRs) and how the team works: development process, "
     "code review, branching, tooling, tech choices and why they were made\n"
     "charter = project goals, scope, stakeholders, success criteria\n"
     "plan_of_action = phases, milestones, the build plan\n"
     "pm_notes = project manager's working notes\n"
     "question_catalogue = the list of questions the agent is meant to answer\n"
     "raid_log = risks, assumptions, issues, dependencies\n"
     "retro = sprint retrospectives: what went well, what did not\n"
     "sprint_summary = per-sprint summaries\n"
     "general = only for pages that fit none of the above; never pick it for a question about "
     "how the team works or why a decision was made\n"
     "For narrative or hybrid, also set search_query: the question rewritten as a short phrase "
     "using the words the documents would use, e.g. 'how are code reviews done' -> 'pull request "
     "review and approval process'\n"
     "hybrid = needs both a database number and a document search in the same answer -- when "
     "you pick this, set doc_type as above AND hybrid_metric_key to whichever of velocity, "
     "committed_vs_delivered, logged_vs_planned_hours, re_estimated_stories, "
     "review_turnaround, longest_review_wait, unreviewed_prs, long_open_prs, "
     "commits_for_ticket, commits_without_ticket, non_convention_branches, "
     "open_connectors_stories supplies the number"),
    ("human", "{question}"),
])


def _clean_ticket_key(value: str | None) -> str | None:
    """Turn the model's ticket reference into the stored form, AGENTS-14, or None if it has none.

    Reuses the connector's own pattern, so "AGENTS 14" and "agents-14" both give AGENTS-14.
    """
    match = TICKET_PATTERN.search(value or "")
    return f"AGENTS-{match.group(1)}" if match else None


def classify_intent(state: AgentState) -> AgentState:
    """Decide which known query, action or document search answers the question, or 'other'."""
    chain = _CLASSIFY_PROMPT | get_llm("fast").with_structured_output(Classification)
    result = chain.invoke({"question": state["question"]})
    state["metric_key"] = result.metric_key
    ticket_key = _clean_ticket_key(result.ticket_key)
    if ticket_key:
        state["ticket_key"] = ticket_key
    if result.doc_type:
        state["doc_type"] = result.doc_type
    if result.search_query:
        state["search_query"] = result.search_query
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
    "committed_vs_delivered": COMMITTED_VS_DELIVERED_SQL,
    "logged_vs_planned_hours": LOGGED_VS_PLANNED_HOURS_SQL,
    "re_estimated_stories": RE_ESTIMATED_STORIES_SQL,
    "review_turnaround": REVIEW_TURNAROUND_SQL,
    "longest_review_wait": LONGEST_REVIEW_WAIT_SQL,
    "unreviewed_prs": UNREVIEWED_PRS_SQL,
    "long_open_prs": LONG_OPEN_PRS_SQL,
    "commits_for_ticket": COMMITS_FOR_TICKET_SQL,
    "commits_without_ticket": COMMITS_WITHOUT_TICKET_SQL,
    "non_convention_branches": NON_CONVENTION_BRANCHES_SQL,
    "open_connectors_stories": OPEN_CONNECTORS_STORIES_SQL,
}
 
def metric_path(state: AgentState) -> AgentState:
    """Run the SQL for the classified metric and store the rows. No model call."""
    sql = _QUERY_MAP[state["metric_key"]]
    with sqlite3.connect(DB_PATH) as conn:
        # The ticket key goes in as a value, never pasted into the SQL text, so it cannot
        # change what the query does. Queries with no :ticket_key placeholder ignore it.
        cur = conn.execute(sql, {"ticket_key": state.get("ticket_key")})
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
    state["file_path"] = path
    state["answer"] = f"Wrote {len(last_sprint)} row(s) to {path}."
    return state
 
 
def _source(result: SearchResult) -> dict[str, Any]:
    """Turn one retrieved chunk into a source: the page title, its link, the text and the score.

    The title and link are looked up in confluence_pages so a reader sees which
    document a reason came from. If the page is not in the database (or the
    table does not exist yet), the document id is shown instead of a title.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT title, url FROM confluence_pages WHERE id = ?", (result.page_id,)
            ).fetchone()
    except sqlite3.OperationalError:
        row = None
    title, url = row if row else (result.doc_id, None)
    return {"title": title, "url": url, "text": result.text, "score": result.score}


def narrative_path(state: AgentState) -> AgentState:
    """Retrieve the top matching chunks for a narrative question (AGENTS-39, AGENTS-74).

    Gathers the passages into state["sources"] (title, link, text, score).
    state["rows"] stays empty because there are no numbers. Uses AGENTS-38's
    search(), which confines the search to state["doc_type"] and applies
    whatever ticket/date filters were passed (none, here -- this ticket only
    wires the plain lookup in).

    A combined filter or a genuinely unmatched question can make search()
    return nothing (ADR-003: accept lower recall). That is answered honestly
    here rather than handed to compose_answer, the same way refuse_path
    answers honestly instead of guessing -- an empty prompt would tempt the
    strong-tier model to invent an answer from outside the retrieved data.
    """
    doc_type = state["doc_type"]
    # The classifier's rewritten phrase searches better than a short, generic question.
    results = search(state.get("search_query") or state["question"], doc_type)
    state["rows"] = []
    if not results:
        state["sources"] = []
        state["answer"] = f"No matching content found in the project's {doc_type} documents."
        return state
    # Passages go in "sources", not "rows": rows are for numbers a screen can draw as a table.
    state["sources"] = [_source(r) for r in results]
    return state


def hybrid_path(state: AgentState) -> AgentState:
    """Run the SQL metric and the document search, and keep them apart (AGENTS-41, AGENTS-74).

    Reuses metric_path and narrative_path exactly as they already are. The
    numbers stay in state["rows"] and the document passages go in
    state["sources"], so a screen can show "the figures" and "the reason, from
    this document" as two parts, and compose_answer can name the document.
    """
    question, doc_type = state["question"], state["doc_type"]
    metric_state: AgentState = {"metric_key": state["hybrid_metric_key"]}
    # A hybrid answer built on commits_for_ticket needs the ticket key the classifier found.
    if "ticket_key" in state:
        metric_state["ticket_key"] = state["ticket_key"]
    metric_rows = metric_path(metric_state)["rows"]
    narrative_state = narrative_path(
        {"question": question, "doc_type": doc_type, "search_query": state.get("search_query", "")}
    )
    state["rows"] = metric_rows
    state["sources"] = narrative_state.get("sources", [])
    return state


def refuse_path(state: AgentState) -> AgentState:
    """Answer honestly when the question matches none of the known paths."""
    state["rows"] = []
    state["answer"] = (
        "I can answer sprint velocity, committed versus delivered points, logged versus "
        "planned hours, re-estimated stories, PR review turnaround, the longest PR review "
        "wait, PRs merged without review, PRs open for more than three days, "
        "the commits for one ticket, how many commits have no ticket ID, "
        "branches that break the naming convention, open tickets under the Connectors epic, "
        "export the last sprint's velocity to Excel, "
        "questions answered by the project's own documents, or questions that combine a "
        "number with the project's own documents. This question doesn't match any of those."
    )
    return state
 
 
_COMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the question in one short sentence using only the data given. "
     "Do not add any number that is not in the data. "
     "If the data does not answer the question at all, say so plainly. "
     "If the data answers only part of the question, give that part and do not say that the "
     "data does not answer it. "
     "If the data is an empty list, say that no matching records were found. "
     "Never use general knowledge or guess."),
    ("human", "Question: {question}\nData: {rows}"),
])

# Hybrid answers get their own prompt: a number from the database plus a reason from a
# document, with the document named so the reader can check it (AGENTS-74).
_HYBRID_COMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the question in at most two short sentences using only the data given. "
     "The numbers come from the project's database. The documents are passages from the "
     "project's own pages. Give the number first, then the reason or rule from the "
     "documents, and name the document it came from. "
     "Do not add any number that is not in the data. "
     "If no document explains it, give the number and say that no document explains it. "
     "If the numbers list is empty, say that no matching records were found. "
     "Never use general knowledge or guess."),
    ("human", "Question: {question}\nNumbers: {rows}\nDocuments: {sources}"),
])


def _for_prompt(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the document title and text of each source: the link and score are noise here."""
    return [{"document": s["title"], "text": s["text"]} for s in sources]


def compose_answer(state: AgentState) -> AgentState:
    """Phrase the rows and passages as a sentence. Skipped if a path already answered."""
    if state.get("answer"):
        return state
    metric_key = state.get("metric_key")
    sources = _for_prompt(state.get("sources", []))

    if metric_key == "hybrid":
        chain = _HYBRID_COMPOSE_PROMPT | get_llm("strong") | StrOutputParser()
        state["answer"] = chain.invoke(
            {"question": state["question"], "rows": state["rows"], "sources": sources}
        )
        return state

    # A narrative question has passages and no rows; every other path has rows.
    data = sources if metric_key == "narrative" else state["rows"]
    chain = _COMPOSE_PROMPT | get_llm("strong") | StrOutputParser()
    state["answer"] = chain.invoke({"question": state["question"], "rows": data})
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