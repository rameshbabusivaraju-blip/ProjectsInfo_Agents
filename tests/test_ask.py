"""Tests for the /ask endpoint — AGENTS-42.

agent.invoke() calls live LLM providers (classify_intent, compose_answer), so
it is faked here, the same way test_agent.py fakes agent.search rather than
building a real FAISS index. This only proves the endpoint's own plumbing --
it accepts a question, calls the agent, and shapes the result into
AskResponse -- not that the agent itself answers correctly, which is what
test_agent.py and the golden set already cover.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import main
from app.agent import agent

client = TestClient(main.app, headers={"X-API-Key": "test-key"})


def test_ask_returns_the_agents_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /ask must call agent.invoke() and return its answer and metric_key.

    Patches the object imported directly from app.agent (its defining module)
    rather than main.agent -- main.py imports the same object, but reaching it
    through main's re-export trips mypy strict's no-implicit-reexport check.
    """
    fake_result = {
        "question": "which risks have no owner",
        "metric_key": "narrative",
        "doc_type": "raid_log",
        "rows": [],
        "sources": [
            {
                "title": "RAID Log",
                "url": "https://wiki.example/raid",
                "text": "R2 unowned",
                "score": 0.9,
            }
        ],
        "answer": "R2 has no owner.",
    }
    monkeypatch.setattr(agent, "invoke", lambda state: fake_result)

    response = client.post("/ask", json={"question": "which risks have no owner"})

    assert response.status_code == 200
    assert response.json() == {
        "question": "which risks have no owner",
        "answer": "R2 has no owner.",
        "metric_key": "narrative",
        "rows": [],
        "sources": [
            {
                "title": "RAID Log",
                "url": "https://wiki.example/raid",
                "text": "R2 unowned",
                "score": 0.9,
            }
        ],
        "file_url": None,
    }


def test_ask_returns_numbers_and_sources_apart_for_a_hybrid_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hybrid answer must keep its figures in rows and its document passages in sources."""
    fake_result = {
        "question": "were any PRs merged without review, and what does the decision log say",
        "metric_key": "hybrid",
        "rows": [{"number": 2, "title": "Unreviewed change"}],
        "sources": [{"title": "03 Decision Log", "text": "one approving review", "score": 0.8}],
        "answer": "PR 2 was merged without review; ADR-016 requires one (03 Decision Log).",
    }
    monkeypatch.setattr(agent, "invoke", lambda state: fake_result)

    body = client.post("/ask", json={"question": "hybrid question"}).json()

    assert body["rows"] == [{"number": 2, "title": "Unreviewed change"}]
    assert body["sources"] == [
        {"title": "03 Decision Log", "url": None, "text": "one approving review", "score": 0.8}
    ]


def test_ask_returns_a_download_link_when_the_agent_wrote_a_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An export answer must carry rows and a /files/ link built from the file's name."""
    fake_result = {
        "question": "export the last sprint velocity to Excel",
        "metric_key": "export_excel",
        "rows": [{"sprint": "Sprint 2", "delivered": 8.0}],
        "answer": "Wrote 1 row(s) to exports/last_sprint_velocity.xlsx.",
        "file_path": "exports/last_sprint_velocity.xlsx",
    }
    monkeypatch.setattr(agent, "invoke", lambda state: fake_result)

    response = client.post("/ask", json={"question": "export the last sprint velocity to Excel"})

    body = response.json()
    assert body["rows"] == [{"sprint": "Sprint 2", "delivered": 8.0}]
    assert body["file_url"] == "/files/last_sprint_velocity.xlsx"


def test_files_downloads_an_exported_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A file inside the exports folder must download with its contents."""
    (tmp_path / "report.xlsx").write_bytes(b"file contents")
    monkeypatch.setattr(main, "EXPORTS_DIR", tmp_path)

    response = client.get("/files/report.xlsx")

    assert response.status_code == 200
    assert response.content == b"file contents"


def test_files_returns_404_for_a_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A name that is not in the exports folder must not download anything."""
    monkeypatch.setattr(main, "EXPORTS_DIR", tmp_path)

    response = client.get("/files/nothing.xlsx")

    assert response.status_code == 404


def test_files_refuses_to_leave_the_exports_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A file one level above the exports folder must not be reachable by name."""
    exports = tmp_path / "exports"
    exports.mkdir()
    (tmp_path / "secret.txt").write_text("secret")
    monkeypatch.setattr(main, "EXPORTS_DIR", exports)

    with pytest.raises(HTTPException) as error:
        main.download_file("../secret.txt")

    assert error.value.status_code == 404


def test_ask_rejects_a_missing_question() -> None:
    """No question field must fail validation before the route body ever runs."""
    response = client.post("/ask", json={})

    assert response.status_code == 422

def test_ask_rejects_a_request_without_the_api_key() -> None:
    """No X-API-Key header must stop the request before the agent runs."""
    response = TestClient(main.app).post("/ask", json={"question": "velocity"})

    assert response.status_code == 401


def test_ask_rejects_a_wrong_api_key() -> None:
    """A key that does not match PROJECTPULSE_API_KEY must be refused."""
    response = TestClient(main.app).post(
        "/ask", json={"question": "velocity"}, headers={"X-API-Key": "wrong"}
    )

    assert response.status_code == 401


def test_files_rejects_a_request_without_the_api_key() -> None:
    """Downloads need the key too, not only questions."""
    response = TestClient(main.app).get("/files/report.xlsx")

    assert response.status_code == 401


def test_routes_stay_closed_when_no_key_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With PROJECTPULSE_API_KEY unset the API must refuse, not open up."""
    monkeypatch.delenv("PROJECTPULSE_API_KEY")

    response = client.post("/ask", json={"question": "velocity"})

    assert response.status_code == 503


def test_health_needs_no_api_key() -> None:
    """/health must stay open so the host can probe it."""
    assert TestClient(main.app).get("/health").status_code == 200
