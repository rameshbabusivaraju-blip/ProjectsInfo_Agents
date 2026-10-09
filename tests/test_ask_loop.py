"""Tests for the AGENT_MODE switch on /ask (ADR-022, AGENTS-82).

Both agents are faked, so these tests only prove the plumbing: which agent runs, and that the
loop's result comes back in the same response shape as the fixed agent's.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, ToolMessage

from app import agent_loop, main
from app.agent import agent

client = TestClient(main.app, headers={"X-API-Key": "test-key"})


def _fake_loop_state() -> agent_loop.LoopState:
    """A finished loop state: one get_metric call and its answer."""
    result = {"metric_key": "velocity", "rows": [{"sprint": "Sprint 4", "delivered": 15.0}]}
    return {
        "question": "velocity?",
        "messages": [
            HumanMessage(content="velocity?"),
            ToolMessage(content=json.dumps(result), tool_call_id="call_1", name="get_metric"),
        ],
        "answer": "Sprint 4 delivered 15.0 points.",
    }


def _forbid(*_args: Any) -> Any:
    raise AssertionError("this agent must not run in this mode")


def test_the_fixed_agent_runs_when_agent_mode_is_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_MODE", raising=False)
    monkeypatch.setattr(
        agent, "invoke", lambda state: {"answer": "fixed", "metric_key": "velocity"}
    )
    monkeypatch.setattr(main, "run_loop", _forbid)

    body = client.post("/ask", json={"question": "velocity?"}).json()

    assert body["answer"] == "fixed"


def test_the_fixed_agent_runs_when_agent_mode_is_fixed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MODE", "fixed")
    monkeypatch.setattr(
        agent, "invoke", lambda state: {"answer": "fixed", "metric_key": "velocity"}
    )
    monkeypatch.setattr(main, "run_loop", _forbid)

    assert client.post("/ask", json={"question": "velocity?"}).json()["answer"] == "fixed"


def test_the_loop_runs_when_agent_mode_is_loop_and_keeps_the_response_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_MODE", "loop")
    monkeypatch.setattr(agent, "invoke", _forbid)
    asked: list[str] = []

    def fake_run_loop(question: str) -> agent_loop.LoopState:
        asked.append(question)
        return _fake_loop_state()

    monkeypatch.setattr(main, "run_loop", fake_run_loop)

    response = client.post("/ask", json={"question": "velocity?"})

    assert response.status_code == 200
    assert asked == ["velocity?"]
    assert response.json() == {
        "question": "velocity?",
        "answer": "Sprint 4 delivered 15.0 points.",
        "metric_key": "velocity",
        "rows": [{"sprint": "Sprint 4", "delivered": 15.0}],
        "sources": [],
        "file_url": None,
    }


def test_the_loop_returns_a_download_link_for_an_export(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MODE", "loop")
    export = {
        "file_path": "exports/velocity_last.xlsx",
        "rows": [{"sprint": "Sprint 4", "delivered": 15.0}],
        "rows_written": 1,
    }
    state: agent_loop.LoopState = {
        "question": "export",
        "messages": [
            ToolMessage(content=json.dumps(export), tool_call_id="call_1", name="export_excel")
        ],
        "answer": "The file is ready.",
    }
    monkeypatch.setattr(main, "run_loop", lambda question: state)

    body = client.post("/ask", json={"question": "export"}).json()

    assert body["metric_key"] == "export_excel"
    assert body["file_url"] == "/files/velocity_last.xlsx"


def test_an_unknown_agent_mode_is_refused_with_a_clear_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_MODE", "turbo")
    monkeypatch.setattr(agent, "invoke", _forbid)
    monkeypatch.setattr(main, "run_loop", _forbid)

    response = client.post("/ask", json={"question": "velocity?"})

    assert response.status_code == 503
    assert response.json() == {"detail": "AGENT_MODE must be 'fixed' or 'loop'."}
