"""Tests for app/tools.py, the tools the agent loop can call (ADR-022).

Fully offline: a temp SQLite file stands in for projectpulse.db, and the document search is
replaced with a fake. No Jira, GitHub or model call.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, get_args

import pytest
from openpyxl import load_workbook
from sqlmodel import Session, SQLModel, create_engine

from app import agent, excel_export, tools
from app.models import Commit, ConfluencePage, Sprint, Ticket, TicketSprint
from app.retrieval import SearchResult


def _new_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    """Create an empty temp database, point the agent at it, and return an open session."""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(agent, "DB_PATH", str(db_path))
    return Session(engine)


def _add_done_ticket(session: Session) -> None:
    """One sprint with one done 5-point ticket, so velocity returns one known row."""
    session.add(Sprint(id=1, name="Sprint 1", state="closed", start_date=datetime(2026, 1, 1)))
    session.add(
        Ticket(
            key="AGENTS-1",
            summary="Test ticket",
            issue_type="Task",
            hierarchy_level=0,
            status="Done",
            status_category="done",
            story_points=5.0,
            created=datetime(2026, 1, 1),
            updated=datetime(2026, 1, 2),
        )
    )
    session.add(TicketSprint(ticket_key="AGENTS-1", sprint_id=1, position=0))
    session.commit()


def _commit(letter: str, message: str, ticket_key: str | None) -> Commit:
    """A commit with a 40-character id made of one repeated letter."""
    return Commit(
        sha=letter * 40,
        message=message,
        author_login="dev",
        authored_at=datetime(2026, 1, 5),
        ticket_key=ticket_key,
    )


def _fake_result() -> SearchResult:
    """One retrieved passage from the RAID log page."""
    return SearchResult(
        score=0.692,
        text="R2 | ... | Unassigned | Open",
        doc_id="confluence:10092545",
        doc_type="raid_log",
        space_key="ConProK",
        page_id="10092545",
        version=1,
        ticket_ids=[],
        author_id=None,
        updated_at="2026-09-25",
    )


# ---------------------------------------------------------------------------
# What the model reads: the lists must stay in step with the code they describe
# ---------------------------------------------------------------------------


def test_every_query_has_a_description_and_the_other_way_round() -> None:
    """A query added to _QUERY_MAP must also be described, or the model cannot choose it."""
    assert set(tools.METRIC_DESCRIPTIONS) == set(agent._QUERY_MAP)


def test_document_types_match_the_classifiers_list() -> None:
    """The search tool offers the same document types as the fixed agent's classifier."""
    annotation = agent.Classification.model_fields["doc_type"].annotation
    literal = next(arg for arg in get_args(annotation) if arg is not type(None))
    assert set(get_args(literal)) == set(tools.DOC_TYPE_DESCRIPTIONS)


def test_tool_descriptions_list_every_key_the_model_may_choose() -> None:
    """The model sees every metric key and document type in the tool descriptions."""
    by_name = {tool.name: tool for tool in tools.TOOLS}
    assert set(by_name) == {
        "get_metric",
        "search_documents",
        "export_excel",
        "live_jira_ticket",
        "live_jira_search",
        "live_github_pull_requests",
        "live_github_commits",
        "live_confluence_page",
    }
    assert tools.LIVE_TOOL_NAMES == {name for name in by_name if name.startswith("live_")}
    for key in tools.METRIC_DESCRIPTIONS:
        assert key in by_name["get_metric"].description
    for doc_type in tools.DOC_TYPE_DESCRIPTIONS:
        assert doc_type in by_name["search_documents"].description


# ---------------------------------------------------------------------------
# get_metric
# ---------------------------------------------------------------------------


def test_get_metric_returns_the_rows_of_the_reviewed_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """velocity runs the reviewed SQL and returns its rows unchanged."""
    with _new_db(tmp_path, monkeypatch) as session:
        _add_done_ticket(session)

    result = tools.get_metric("velocity")

    assert result["metric_key"] == "velocity"
    assert result["rows"] == [{"sprint": "Sprint 1", "delivered": 5.0}]

def test_get_metric_with_an_unknown_key_lists_the_valid_keys() -> None:
    """The model gets a readable error with the valid keys, not an exception."""
    result = tools.get_metric("drop_table_sprints")

    assert "Unknown metric_key 'drop_table_sprints'" in result["error"]
    assert "velocity" in result["error"]


def test_get_metric_commits_for_ticket_needs_a_usable_ticket_key() -> None:
    """A missing or malformed ticket key is an error, not an empty answer."""
    for bad in (None, "", "no key here"):
        result = tools.get_metric("commits_for_ticket", bad)
        assert result == {"error": "commits_for_ticket needs a ticket_key such as AGENTS-14."}


def test_get_metric_cleans_the_ticket_key_before_use(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """'agents 14' finds the commits stored under AGENTS-14, and other tickets stay out."""
    with _new_db(tmp_path, monkeypatch) as session:
        session.add(_commit("a", "AGENTS-14: first", "AGENTS-14"))
        session.add(_commit("b", "AGENTS-15: other", "AGENTS-15"))
        session.commit()

    result = tools.get_metric("commits_for_ticket", "agents 14")

    assert [row["subject"] for row in result["rows"]] == ["AGENTS-14: first"]


def test_get_metric_says_so_when_there_are_no_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty result comes with a message the model can pass on."""
    with _new_db(tmp_path, monkeypatch):
        pass

    result = tools.get_metric("velocity")

    assert result["rows"] == []
    assert result["message"] == "The stored data has no rows for this question."


def test_get_metric_reports_a_database_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A database with no tables gives an error result, and the loop is not interrupted."""
    monkeypatch.setattr(agent, "DB_PATH", str(tmp_path / "empty.db"))

    result = tools.get_metric("velocity")

    assert result["error"].startswith("Database error:")


# ---------------------------------------------------------------------------
# search_documents
# ---------------------------------------------------------------------------


def test_search_documents_returns_passages_with_the_page_title(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A passage comes back with the title and link of the page it came from."""
    with _new_db(tmp_path, monkeypatch) as session:
        session.add(
            ConfluencePage(
                id="10092545",
                title="RAID Log — ProjectPulse",
                space_key="ConProK",
                url="https://wiki.example/raid",
                version=1,
                updated=datetime(2026, 9, 25),
            )
        )
        session.commit()
    seen: list[tuple[str, str]] = []

    def fake_search(query: str, doc_type: str) -> list[SearchResult]:
        seen.append((query, doc_type))
        return [_fake_result()]

    monkeypatch.setattr(agent, "search", fake_search)

    result = tools.search_documents("risks without an owner", "raid_log")

    assert seen == [("risks without an owner", "raid_log")]
    assert result["sources"][0]["title"] == "RAID Log — ProjectPulse"
    assert result["sources"][0]["url"] == "https://wiki.example/raid"
    assert "message" not in result


def test_search_documents_says_so_when_nothing_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No passage gives an empty list and a message, so the model can admit the gap."""
    monkeypatch.setattr(agent, "DB_PATH", str(tmp_path / "empty.db"))
    monkeypatch.setattr(agent, "search", lambda query, doc_type: [])

    result = tools.search_documents("anything", "retro")

    assert result["sources"] == []
    assert result["message"] == "No matching content found in the project's retro documents."


def test_search_documents_with_an_unknown_type_lists_the_valid_types() -> None:
    """An unknown document type is an error that names the valid ones."""
    result = tools.search_documents("anything", "gossip")

    assert "Unknown doc_type 'gossip'" in result["error"]
    assert "raid_log" in result["error"]


def test_search_documents_reports_a_failing_search(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A search that raises (for example a missing index) becomes an error result."""
    monkeypatch.setattr(agent, "DB_PATH", str(tmp_path / "empty.db"))

    def broken(query: str, doc_type: str) -> list[SearchResult]:
        raise RuntimeError("index is not built")

    monkeypatch.setattr(agent, "search", broken)

    result = tools.search_documents("anything", "retro")

    assert result == {"error": "Document search failed: index is not built"}


# ---------------------------------------------------------------------------
# export_excel
# ---------------------------------------------------------------------------


def test_export_excel_writes_every_row_and_returns_the_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The file holds the rows of the reviewed query, and the path comes back."""
    with _new_db(tmp_path, monkeypatch) as session:
        _add_done_ticket(session)
    monkeypatch.setattr(excel_export, "EXPORTS_DIR", tmp_path)

    result = tools.export_excel("velocity")

    assert result["file_path"] == str(tmp_path / "velocity.xlsx")
    assert result["rows"] == [{"sprint": "Sprint 1", "delivered": 5.0}]
    assert result["rows_written"] == 1
    sheet = load_workbook(tmp_path / "velocity.xlsx").active
    assert sheet is not None
    assert [cell.value for cell in sheet[2]] == ["Sprint 1", 5.0]


def test_export_excel_can_write_only_the_last_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """last_row_only keeps the most recent sprint and names the file so."""
    with _new_db(tmp_path, monkeypatch) as session:
        _add_done_ticket(session)
        session.add(Sprint(id=2, name="Sprint 2", state="active", start_date=datetime(2026, 2, 1)))
        session.add(
            Ticket(
                key="AGENTS-2",
                summary="Second",
                issue_type="Task",
                hierarchy_level=0,
                status="Done",
                status_category="done",
                story_points=8.0,
                created=datetime(2026, 2, 1),
                updated=datetime(2026, 2, 2),
            )
        )
        session.add(TicketSprint(ticket_key="AGENTS-2", sprint_id=2, position=0))
        session.commit()
    monkeypatch.setattr(excel_export, "EXPORTS_DIR", tmp_path)

    result = tools.export_excel("velocity", last_row_only=True)

    assert result["file_path"] == str(tmp_path / "velocity_last.xlsx")
    assert result["rows"] == [{"sprint": "Sprint 2", "delivered": 8.0}]


def test_export_excel_puts_the_ticket_in_the_file_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exports for two tickets do not overwrite each other."""
    with _new_db(tmp_path, monkeypatch) as session:
        session.add(_commit("a", "AGENTS-14: first", "AGENTS-14"))
        session.commit()
    monkeypatch.setattr(excel_export, "EXPORTS_DIR", tmp_path)

    result = tools.export_excel("commits_for_ticket", "AGENTS-14")

    assert result["file_path"] == str(tmp_path / "commits_for_ticket_AGENTS-14.xlsx")


def test_export_excel_writes_no_file_when_there_is_nothing_to_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty result gives a message and no file."""
    with _new_db(tmp_path, monkeypatch):
        pass
    exports = tmp_path / "exports"
    monkeypatch.setattr(excel_export, "EXPORTS_DIR", exports)

    result = tools.export_excel("velocity")

    assert result == {
        "message": "Nothing to export: the stored data has no rows for this question."
    }
    assert not exports.exists()


def test_export_excel_with_an_unknown_key_is_an_error() -> None:
    """The same key check as get_metric applies."""
    assert "Unknown metric_key" in tools.export_excel("nope")["error"]


# ---------------------------------------------------------------------------
# run_tool
# ---------------------------------------------------------------------------


def test_run_tool_runs_the_named_tool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """run_tool passes the model's arguments to the tool and returns its result."""
    with _new_db(tmp_path, monkeypatch) as session:
        _add_done_ticket(session)

    result: dict[str, Any] = tools.run_tool("get_metric", {"metric_key": "velocity"})

    assert result["rows"] == [{"sprint": "Sprint 1", "delivered": 5.0}]


def test_run_tool_with_an_unknown_tool_lists_the_valid_tools() -> None:
    """A tool name the model made up is an error that names the real tools."""
    result = tools.run_tool("run_sql", {"sql": "select 1"})

    assert "Unknown tool 'run_sql'" in result["error"]
    assert "get_metric" in result["error"]


def test_run_tool_with_bad_arguments_is_an_error() -> None:
    """A missing argument comes back as a readable error, not an exception."""
    result = tools.run_tool("get_metric", {})

    assert result["error"].startswith("Bad arguments for get_metric:")
