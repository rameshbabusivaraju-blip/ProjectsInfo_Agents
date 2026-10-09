"""Tests for app/live_confluence.py (AGENTS-89). Fully offline: the HTTP helper is replaced."""

from typing import Any

import pytest

from app import live_confluence
from app.live_http import LiveDataError


@pytest.fixture(autouse=True)
def space_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool reads the space key when it runs, so each test sets it."""
    monkeypatch.setenv("CONFLUENCE_SPACE_KEY", "PP")


def _match(page_id: str, title: str, when: str = "2026-10-09T08:00:00.000Z") -> dict[str, Any]:
    """One result of the Confluence content search."""
    return {
        "id": page_id,
        "title": title,
        "version": {"when": when},
        "_links": {"webui": f"/spaces/PP/pages/{page_id}/{title.replace(' ', '+')}"},
    }


def _body(html: str) -> dict[str, Any]:
    """A page as the v2 API returns it with body-format export_view."""
    return {"body": {"export_view": {"value": html}}}


def _fake_get(
    monkeypatch: pytest.MonkeyPatch,
    matches: list[dict[str, Any]] | Exception,
    body: dict[str, Any] | Exception | None = None,
) -> list[dict[str, Any]]:
    """Replace live_get. The search URL gets `matches`, the page URL gets `body`."""
    calls: list[dict[str, Any]] = []

    def fake(source: str, url: str, **kwargs: Any) -> Any:
        calls.append({"source": source, "url": url, **kwargs})
        outcome: Any = matches if "/rest/api/content/search" in url else body
        if isinstance(outcome, Exception):
            raise outcome
        return {"results": outcome} if "/rest/api/content/search" in url else outcome

    monkeypatch.setattr(live_confluence, "live_get", fake)
    return calls


def test_it_returns_the_best_page_as_text_with_its_link(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_get(
        monkeypatch,
        [_match("111", "Sprint 4 Retro"), _match("222", "Sprint 3 Retro")],
        _body("<h1>Sprint 4 Retro</h1><p>Went well: <b>live data</b>.</p>"),
    )

    result = live_confluence.live_confluence_page("Sprint 4 Retro")

    assert result["source"] == "Confluence"
    assert result["live"] is True
    assert result["fetched_at"].endswith(" UTC")
    assert result["sources"] == [
        {
            "title": "Sprint 4 Retro",
            "url": "https://test.atlassian.net/wiki/spaces/PP/pages/111/Sprint+4+Retro",
            "text": "# Sprint 4 Retro\n\nWent well: **live data**.",
            "updated": "2026-10-09T08:00:00",
        }
    ]
    assert result["truncated"] is False
    assert result["other_matches"] == ["Sprint 3 Retro"]
    assert calls[0]["params"]["limit"] == 3
    assert calls[1]["url"].endswith("/api/v2/pages/111")
    assert calls[1]["params"] == {"body-format": "export_view"}


def test_the_search_is_limited_to_the_project_space(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_get(monkeypatch, [_match("1", "A")], _body("<p>x</p>"))

    live_confluence.live_confluence_page("risk log")

    assert calls[0]["params"]["cql"] == (
        'space = "PP" AND type = page AND (title ~ "risk log" OR text ~ "risk log")'
    )


def test_a_long_page_is_cut_and_marked(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_get(monkeypatch, [_match("1", "Big")], _body("<p>" + "word " * 2000 + "</p>"))

    result = live_confluence.live_confluence_page("Big")

    assert len(result["sources"][0]["text"]) == live_confluence.MAX_TEXT_CHARS
    assert result["truncated"] is True


def test_no_matching_page_is_a_message_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_get(monkeypatch, [])

    result = live_confluence.live_confluence_page("nothing like this")

    assert result["sources"] == []
    assert result["message"] == "Confluence has no page in the project space that matches."
    assert "error" not in result
    assert len(calls) == 1  # no page was read


@pytest.mark.parametrize(
    "bad", ['x" OR space = "OTHER', "a" * 61, "x) OR (y", "", "title ~ 'a'"]
)
def test_search_words_that_could_change_the_query_are_rejected(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    calls = _fake_get(monkeypatch, [])

    result = live_confluence.live_confluence_page(bad)

    assert "may only use letters" in result["error"]
    assert calls == []


def test_without_a_space_key_the_tool_answers_with_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CONFLUENCE_SPACE_KEY")
    calls = _fake_get(monkeypatch, [])

    result = live_confluence.live_confluence_page("anything")

    assert "CONFLUENCE_SPACE_KEY is not set" in result["error"]
    assert calls == []


@pytest.mark.parametrize(
    "message",
    [
        "Confluence did not answer within 10 seconds.",
        "Confluence rejected the credentials (HTTP 401).",
        "Confluence refused the request because the rate limit is used up.",
    ],
)
def test_a_failed_search_comes_back_as_an_error_the_model_can_read(
    monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    _fake_get(monkeypatch, LiveDataError(message))

    assert live_confluence.live_confluence_page("retro") == {"error": message}


def test_a_failed_page_read_comes_back_as_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_get(
        monkeypatch,
        [_match("1", "Retro")],
        LiveDataError("Confluence has nothing at this address (HTTP 404)."),
    )

    assert live_confluence.live_confluence_page("retro") == {
        "error": "Confluence has nothing at this address (HTTP 404)."
    }


def test_the_tool_never_sends_anything_but_get() -> None:
    """live_get is the only way out of this module, and it only sends GET (see test_live_http)."""
    import inspect

    source = inspect.getsource(live_confluence)

    for word in ("httpx.post", "httpx.put", "httpx.patch", "httpx.delete", "httpx.request"):
        assert word not in source