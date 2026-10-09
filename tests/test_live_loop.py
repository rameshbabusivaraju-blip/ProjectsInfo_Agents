"""Tests for how the loop uses the live tools (ADR-023): the 3-call budget and the result fields.

The model and the live tools are fakes, so nothing here calls a provider, Jira, GitHub or
Confluence.
"""

from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from app import agent_loop, tools
from app.agent_loop import LIVE_LIMIT_ERROR, MAX_LIVE_CALLS

TICKET = {
    "source": "jira",
    "live": True,
    "fetched_at": "2026-10-09 06:00 UTC",
    "rows": [{"key": "AGENTS-95", "status": "To Do"}],
}
PAGE = {
    "source": "confluence",
    "live": True,
    "fetched_at": "2026-10-09 06:00 UTC",
    "sources": [{"title": "RAID Log", "url": "https://x/raid", "text": "Risk 1", "updated": "d"}],
}
STORED = {"metric_key": "velocity", "rows": [{"sprint": "Sprint 4", "delivered": 34.5}]}


class Script:
    """A scripted model: returns the prepared replies in order and keeps what it saw."""

    def __init__(self, replies: list[AIMessage]) -> None:
        self.replies = replies
        self.calls: list[list[BaseMessage]] = []

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(list(messages))
        return self.replies[len(self.calls) - 1].model_copy(update={"id": None})


def _asks(*calls: tuple[str, dict[str, Any]]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}
            for i, (name, args) in enumerate(calls)
        ],
        usage_metadata={"input_tokens": 10, "output_tokens": 0, "total_tokens": 10},
    )


def _says(text: str) -> AIMessage:
    return AIMessage(
        content=text,
        usage_metadata={"input_tokens": 10, "output_tokens": 0, "total_tokens": 10},
    )


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch) -> Callable[..., tuple[Script, list[str]]]:
    """Install a scripted model and a fake run_tool. Returns the model and the names run."""

    def make(*replies: AIMessage) -> tuple[Script, list[str]]:
        model = Script(list(replies))
        ran: list[str] = []

        def fake_run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
            ran.append(name)
            results: dict[str, dict[str, Any]] = {
                "live_jira_ticket": TICKET,
                "live_confluence_page": PAGE,
                "get_metric": STORED,
            }
            return results.get(name, TICKET)

        monkeypatch.setattr(agent_loop, "_model", lambda: model)
        monkeypatch.setattr(agent_loop, "run_tool", fake_run_tool)
        return model, ran

    return make


def test_the_prompt_tells_the_model_when_to_go_live() -> None:
    prompt = agent_loop.SYSTEM_PROMPT
    assert "Use stored data first" in prompt
    assert "live data" in prompt
    assert "fetched_at" in prompt
    assert f"at most {MAX_LIVE_CALLS} live calls" in prompt


def test_a_live_ticket_comes_back_as_rows_with_the_live_metric_key(
    install: Callable[..., tuple[Script, list[str]]],
) -> None:
    install(
        _asks(("live_jira_ticket", {"ticket_key": "AGENTS-95"})),
        _says("Live data (2026-10-09 06:00 UTC): AGENTS-95 is To Do."),
    )

    result = agent_loop.loop_result(agent_loop.run_loop("Status of AGENTS-95?"))

    assert result["metric_key"] == "live"
    assert result["rows"] == TICKET["rows"]
    assert result["sources"] == []


def test_a_live_confluence_page_joins_sources(
    install: Callable[..., tuple[Script, list[str]]],
) -> None:
    install(
        _asks(("live_confluence_page", {"query": "RAID Log"})),
        _says("Live data: the RAID Log lists Risk 1."),
    )

    result = agent_loop.loop_result(agent_loop.run_loop("What is in the RAID Log?"))

    assert result["metric_key"] == "live"
    assert result["sources"] == PAGE["sources"]
    assert result["rows"] == []


def test_stored_data_alone_keeps_its_metric_key(
    install: Callable[..., tuple[Script, list[str]]],
) -> None:
    install(_asks(("get_metric", {"metric_key": "velocity"})), _says("34.5 points."))

    result = agent_loop.loop_result(agent_loop.run_loop("Velocity?"))

    assert result["metric_key"] == "velocity"


def test_a_failed_live_call_does_not_make_the_result_live(
    monkeypatch: pytest.MonkeyPatch,
    install: Callable[..., tuple[Script, list[str]]],
) -> None:
    install(
        _asks(("live_jira_ticket", {"ticket_key": "AGENTS-95"})),
        _says("Live data could not be reached."),
    )
    monkeypatch.setattr(agent_loop, "run_tool", lambda name, args: {"error": "Jira timed out."})

    result = agent_loop.loop_result(agent_loop.run_loop("Status of AGENTS-95?"))

    assert result["metric_key"] == "other"
    assert result["rows"] == []


def test_the_fourth_live_call_is_refused_and_not_run(
    install: Callable[..., tuple[Script, list[str]]],
) -> None:
    model, ran = install(
        _asks(*[("live_jira_ticket", {"ticket_key": f"AGENTS-{n}"}) for n in range(1, 5)]),
        _says("Only three could be read live."),
    )

    state = agent_loop.run_loop("Compare AGENTS-1 to AGENTS-4")

    assert ran == ["live_jira_ticket"] * MAX_LIVE_CALLS
    assert state["live_calls"] == MAX_LIVE_CALLS
    returned = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert len(returned) == 4
    assert LIVE_LIMIT_ERROR in str(returned[3].content)


def test_the_budget_is_counted_across_steps_and_stored_tools_do_not_use_it(
    install: Callable[..., tuple[Script, list[str]]],
) -> None:
    _, ran = install(
        _asks(("get_metric", {"metric_key": "velocity"}), ("live_jira_ticket", {})),
        _asks(("live_jira_ticket", {}), ("live_confluence_page", {"query": "x"})),
        _asks(("live_jira_ticket", {})),
        _says("Done."),
    )

    state = agent_loop.run_loop("Many things")

    assert state["live_calls"] == MAX_LIVE_CALLS
    assert ran == [
        "get_metric",
        "live_jira_ticket",
        "live_jira_ticket",
        "live_confluence_page",
    ]


def test_every_live_tool_is_in_the_budget_and_the_dispatch_table() -> None:
    for name in tools.LIVE_TOOL_NAMES:
        assert name in tools._TOOL_TABLE
