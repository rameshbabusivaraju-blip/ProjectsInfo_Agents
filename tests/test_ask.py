"""Tests for the /ask endpoint — AGENTS-42.

agent.invoke() calls live LLM providers (classify_intent, compose_answer), so
it is faked here, the same way test_agent.py fakes agent.search rather than
building a real FAISS index. This only proves the endpoint's own plumbing --
it accepts a question, calls the agent, and shapes the result into
AskResponse -- not that the agent itself answers correctly, which is what
test_agent.py and the golden set already cover.
"""
from app.agent import agent
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)


def test_ask_returns_the_agents_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /ask must call agent.invoke() and return its answer and metric_key."""
    fake_result = {
        "question": "which risks have no owner",
        "metric_key": "narrative",
        "doc_type": "raid_log",
        "rows": [{"text": "R2 has no owner.", "doc_id": "confluence:1", "score": 0.9}],
        "answer": "R2 has no owner.",
    }
    monkeypatch.setattr(main.agent, "invoke", lambda state: fake_result)

    response = client.post("/ask", json={"question": "which risks have no owner"})

    assert response.status_code == 200
    assert response.json() == {
        "question": "which risks have no owner",
        "answer": "R2 has no owner.",
        "metric_key": "narrative",
    }


def test_ask_rejects_a_missing_question() -> None:
    """No question field must fail validation before the route body ever runs."""
    response = client.post("/ask", json={})

    assert response.status_code == 422