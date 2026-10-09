"""Tools the agent loop can call (ADR-022).

Each tool wraps code that already exists and has been reviewed. get_metric runs the SQL in
agent._QUERY_MAP, search_documents runs the document search, and export_excel writes rows
with export_to_excel(). The model only chooses a tool and its arguments. It never writes SQL.

A tool never raises an error to the model. A problem such as an unknown key or a database
error comes back as {"error": "..."}, so the model can read it, try again or explain it.
"""

import sqlite3
from collections.abc import Callable
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError

from app import agent
from app.excel_export import export_to_excel
from app.freshness import SOURCE_FOR_METRIC, get_freshness

# One line per metric_key. The model reads these lines to choose a key. A test checks that
# this dict has exactly the keys of agent._QUERY_MAP, so a new query cannot be left out.
METRIC_DESCRIPTIONS: dict[str, str] = {
    "velocity": "Story points delivered in each sprint.",
    "committed_vs_delivered": "Points committed (approximate) and points delivered, per sprint.",
    "logged_vs_planned_hours": "Hours logged against hours planned (estimates), per sprint.",
    "re_estimated_stories": "Story-point re-estimates made while each sprint was running.",
    "review_turnaround": "Average hours from opening a pull request to its first review.",
    "longest_review_wait": "The one pull request whose first review took the longest.",
    "unreviewed_prs": "Pull requests merged with no review at all.",
    "long_open_prs": "Pull requests open for more than three days.",
    "commits_for_ticket": "Commits that mention one ticket. Needs ticket_key, such as AGENTS-14.",
    "commits_without_ticket": "How many commits have no ticket ID in the message.",
    "non_convention_branches": "Pull request branches that do not follow AGENTS-<n>-description.",
    "open_connectors_stories": "Count and keys of the open tickets under the Connectors epic.",
    "spilled_over_tickets": "Tickets that moved from one sprint into another, with the sprints.",
    "dora_per_sprint": "The four DORA metrics per sprint: deployments, lead time (minutes), "
    "change failure rate (%) and time to restore (minutes).",
    "failed_runs_by_stage": "Failed and finished pipeline runs per stage, this sprint.",
}

# The document types the search can filter on. A test checks that this matches the list the
# fixed agent's classifier uses.
DOC_TYPE_DESCRIPTIONS: dict[str, str] = {
    "charter": "The project charter: purpose, scope and success measures.",
    "decision_log": "Architecture decisions (ADRs) and the reasons for them.",
    "general": "Other project pages that fit no specific type.",
    "plan_of_action": "The plan of action and the hand-over notes.",
    "pm_notes": "Project management notes explaining the practices used.",
    "question_catalogue": "The catalogue of questions the agent should answer.",
    "raid_log": "Risks, assumptions, issues and dependencies, with owners.",
    "retro": "Sprint retrospectives: what went well and what to change.",
    "sprint_summary": "Delivery summaries written at the end of each sprint.",
}


def _validate(metric_key: str, ticket_key: str | None) -> tuple[str | None, str | None]:
    """Check a metric request. Return (clean ticket key, error text); error is None if valid."""
    if metric_key not in agent._QUERY_MAP:
        valid = ", ".join(sorted(agent._QUERY_MAP))
        return None, f"Unknown metric_key '{metric_key}'. Valid keys: {valid}."
    cleaned = agent._clean_ticket_key(ticket_key)
    if metric_key == "commits_for_ticket" and cleaned is None:
        return None, "commits_for_ticket needs a ticket_key such as AGENTS-14."
    return cleaned, None


def _rows_for(metric_key: str, ticket_key: str | None) -> list[dict[str, Any]]:
    """Run one reviewed query through the same code the fixed agent uses."""
    state: agent.AgentState = {"metric_key": metric_key}
    if ticket_key:
        state["ticket_key"] = ticket_key
    return agent.metric_path(state)["rows"]


def get_metric(metric_key: str, ticket_key: str | None = None) -> dict[str, Any]:
    """Run the reviewed query for metric_key and return its rows."""
    cleaned, error = _validate(metric_key, ticket_key)
    if error:
        return {"error": error}
    try:
        rows = _rows_for(metric_key, cleaned)
    except sqlite3.Error as exc:
        return {"error": f"Database error: {exc}"}
    result: dict[str, Any] = {
        "metric_key": metric_key,
        "rows": rows,
        "freshness": get_freshness(str(agent.DB_PATH), SOURCE_FOR_METRIC[metric_key]),
    }
    if not rows:
        result["message"] = "The stored data has no rows for this question."
    return result


def search_documents(query: str, doc_type: str) -> dict[str, Any]:
    """Search one type of project document and return the best passages with page titles."""
    if doc_type not in DOC_TYPE_DESCRIPTIONS:
        valid = ", ".join(DOC_TYPE_DESCRIPTIONS)
        return {"error": f"Unknown doc_type '{doc_type}'. Valid types: {valid}."}
    try:
        state = agent.narrative_path(
            {"question": query, "doc_type": doc_type, "search_query": query}
        )
    except Exception as exc:  # a failing search must not end the loop
        return {"error": f"Document search failed: {exc}"}
    sources = state.get("sources", [])
    result: dict[str, Any] = {
        "doc_type": doc_type,
        "sources": sources,
        "freshness": get_freshness(str(agent.DB_PATH), "confluence"),
    }
    if not sources:
        result["message"] = f"No matching content found in the project's {doc_type} documents."
    return result


def export_excel(
    metric_key: str, ticket_key: str | None = None, last_row_only: bool = False
) -> dict[str, Any]:
    """Write the rows of a metric to an .xlsx file and return the file path and the rows."""
    cleaned, error = _validate(metric_key, ticket_key)
    if error:
        return {"error": error}
    try:
        rows = _rows_for(metric_key, cleaned)
    except sqlite3.Error as exc:
        return {"error": f"Database error: {exc}"}
    if last_row_only:
        rows = rows[-1:]
    if not rows:
        return {"message": "Nothing to export: the stored data has no rows for this question."}
    name_parts = [metric_key] + ([cleaned] if cleaned else []) + (["last"] if last_row_only else [])
    try:
        path = export_to_excel(rows, "_".join(name_parts) + ".xlsx")
    except OSError as exc:
        return {"error": f"Could not write the file: {exc}"}
    return {"file_path": path, "rows": rows, "rows_written": len(rows)}


class GetMetricArgs(BaseModel):
    """Arguments the model may give get_metric."""

    metric_key: str = Field(description="One key from the list in the tool description.")
    ticket_key: str | None = Field(
        default=None, description="A ticket key such as AGENTS-14. Only for commits_for_ticket."
    )


class SearchDocumentsArgs(BaseModel):
    """Arguments the model may give search_documents."""

    query: str = Field(description="A short search phrase about what to find.")
    doc_type: str = Field(description="One document type from the list in the tool description.")


class ExportExcelArgs(BaseModel):
    """Arguments the model may give export_excel."""

    metric_key: str = Field(description="One key from the get_metric list.")
    ticket_key: str | None = Field(
        default=None, description="A ticket key such as AGENTS-14. Only for commits_for_ticket."
    )
    last_row_only: bool = Field(
        default=False, description="True to write only the most recent row, such as the last one."
    )


def _listing(items: dict[str, str]) -> str:
    """Format a key-to-description dict as one '- key: text' line per item."""
    return "\n".join(f"- {key}: {text}" for key, text in items.items())


TOOLS: list[StructuredTool] = [
    StructuredTool.from_function(
        func=get_metric,
        name="get_metric",
        description=(
            "Run a reviewed query over the project's stored Jira, GitHub and pipeline data "
            "and return its rows. Use it for any number, count or list about sprints, tickets, "
            "pull requests, commits, branches or pipeline runs. Choose metric_key from:\n"
            + _listing(METRIC_DESCRIPTIONS)
        ),
        args_schema=GetMetricArgs,
    ),
    StructuredTool.from_function(
        func=search_documents,
        name="search_documents",
        description=(
            "Search the project's own documents (Confluence pages) and return the best "
            "matching passages with their page titles. Use it for reasons, decisions, risks "
            "and notes. Choose doc_type from:\n" + _listing(DOC_TYPE_DESCRIPTIONS)
        ),
        args_schema=SearchDocumentsArgs,
    ),
    StructuredTool.from_function(
        func=export_excel,
        name="export_excel",
        description=(
            "Write the rows of a metric to an Excel (.xlsx) file and return the file path. "
            "Use it only when the user asks for a spreadsheet or an Excel file. metric_key "
            "and ticket_key work as in get_metric. Set last_row_only to write only the most "
            "recent row."
        ),
        args_schema=ExportExcelArgs,
    ),
]

_TOOL_TABLE: dict[str, tuple[Callable[..., dict[str, Any]], type[BaseModel]]] = {
    "get_metric": (get_metric, GetMetricArgs),
    "search_documents": (search_documents, SearchDocumentsArgs),
    "export_excel": (export_excel, ExportExcelArgs),
}


def run_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run one tool by name with the model's arguments. Problems come back as {"error": ...}."""
    entry = _TOOL_TABLE.get(name)
    if entry is None:
        return {"error": f"Unknown tool '{name}'. Valid tools: {', '.join(_TOOL_TABLE)}."}
    function, args_model = entry
    try:
        parsed = args_model(**arguments)
    except ValidationError as exc:
        return {"error": f"Bad arguments for {name}: {exc.errors()[0]['msg']}."}
    return function(**parsed.model_dump())
