"""Tests for app/live_http.py (AGENTS-92). Fully offline: httpx.get is replaced by a fake."""

import time
from typing import Any

import httpx
import pytest

from app import live_http
from app.live_http import LiveDataError, LiveNotFound, live_get

URL = "https://example.test/rest/thing"


def _response(status: int, **kwargs: Any) -> httpx.Response:
    """A real httpx response, so status handling and .json() behave as in production."""
    return httpx.Response(status, request=httpx.Request("GET", URL), **kwargs)


def _install(
    monkeypatch: pytest.MonkeyPatch, *outcomes: httpx.Response | Exception
) -> list[dict[str, Any]]:
    """Make httpx.get return or raise the given outcomes in order. Returns the calls made."""
    calls: list[dict[str, Any]] = []
    queue = list(outcomes)

    def fake_get(url: str, **kwargs: Any) -> httpx.Response:
        calls.append({"url": url, **kwargs})
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    return calls


def test_a_good_answer_is_returned_as_parsed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install(monkeypatch, _response(200, json={"key": "AGENTS-1"}))

    result = live_get("Jira", URL, headers={"Accept": "x"}, auth=("u", "t"), params={"a": 1})

    assert result == {"key": "AGENTS-1"}
    assert calls == [
        {
            "url": URL,
            "headers": {"Accept": "x"},
            "auth": ("u", "t"),
            "params": {"a": 1},
            "timeout": 10,
        }
    ]


def test_an_empty_result_is_returned_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The helper does not decide what an empty list means. The tool does."""
    _install(monkeypatch, _response(200, json={"issues": []}))

    assert live_get("Jira", URL) == {"issues": []}


def test_a_timeout_is_not_repeated(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install(monkeypatch, httpx.ReadTimeout("slow"))

    with pytest.raises(LiveDataError, match="did not answer within 10 seconds"):
        live_get("Jira", URL)
    assert len(calls) == 1


def test_a_dropped_connection_is_tried_again(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install(
        monkeypatch,
        httpx.RemoteProtocolError("Server disconnected without sending a response."),
        _response(200, json={"ok": True}),
    )

    assert live_get("GitHub", URL) == {"ok": True}
    assert len(calls) == 2


def test_it_gives_up_after_three_dropped_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    drop = httpx.ConnectError("refused")
    calls = _install(monkeypatch, drop, drop, drop)

    with pytest.raises(LiveDataError, match="could not be reached after 3 attempts"):
        live_get("GitHub", URL)
    assert len(calls) == 3


def test_a_401_is_not_repeated(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install(monkeypatch, _response(401))

    with pytest.raises(LiveDataError, match=r"Jira rejected the credentials \(HTTP 401\)"):
        live_get("Jira", URL)
    assert len(calls) == 1


def test_a_plain_403_means_bad_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _response(403))

    with pytest.raises(LiveDataError, match="rejected the credentials"):
        live_get("GitHub", URL)


def test_a_403_with_no_requests_left_means_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _response(403, headers={"x-ratelimit-remaining": "0"}))

    with pytest.raises(LiveDataError, match="rate limit is used up"):
        live_get("GitHub", URL)


def test_a_429_is_not_repeated(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install(monkeypatch, _response(429))

    with pytest.raises(LiveDataError, match="rate limit is used up"):
        live_get("Jira", URL)
    assert len(calls) == 1


def test_a_404_is_reported_as_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _response(404))

    with pytest.raises(LiveNotFound, match="nothing at this address"):
        live_get("Jira", URL)


def test_other_error_codes_are_reported_with_the_code(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _response(500))

    with pytest.raises(LiveDataError, match=r"Confluence returned HTTP 500"):
        live_get("Confluence", URL)


def test_an_answer_that_is_not_json_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _response(200, text="<html>login</html>"))

    with pytest.raises(LiveDataError, match="not JSON"):
        live_get("Confluence", URL)


def test_error_text_never_contains_the_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _response(401))

    with pytest.raises(LiveDataError) as caught:
        live_get("Jira", URL, headers={"Authorization": "Bearer SECRET-TOKEN"}, auth=("me", "pw"))

    assert "SECRET-TOKEN" not in str(caught.value)
    assert "pw" not in str(caught.value)


def test_fetched_at_is_utc_text() -> None:
    stamp = live_http.fetched_at()

    assert stamp.endswith(" UTC")
    assert len(stamp) == len("2026-10-09T12:00:00 UTC")