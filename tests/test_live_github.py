"""Tests for app/live_github.py (AGENTS-88). Fully offline: the HTTP helper is replaced."""

from typing import Any

import pytest

from app import live_github
from app.live_http import LiveDataError


def _pr(number: int = 49, **overrides: Any) -> dict[str, Any]:
    """A pull request as the GitHub list endpoint returns it."""
    entry: dict[str, Any] = {
        "number": number,
        "title": "AGENTS-91 Record when each source was last synced",
        "user": {"login": "SivarajuRB-web"},
        "state": "closed",
        "created_at": "2026-10-09T08:00:00Z",
        "merged_at": "2026-10-09T09:30:00Z",
        "closed_at": "2026-10-09T09:30:00Z",
        "head": {"ref": "AGENTS-87-92-live-fallback"},
    }
    entry.update(overrides)
    return entry


def _commit(sha: str = "0f7e96f" + "a" * 33, **overrides: Any) -> dict[str, Any]:
    """A commit as the GitHub list endpoint returns it."""
    entry: dict[str, Any] = {
        "sha": sha,
        "author": {"login": "SivarajuRB-web"},
        "commit": {
            "message": "AGENTS-95 Restore the Render deploy hook\n\nLonger body text.",
            "author": {"date": "2026-10-09T09:10:00Z"},
        },
    }
    entry.update(overrides)
    return entry


def _fake_get(monkeypatch: pytest.MonkeyPatch, outcome: Any) -> list[dict[str, Any]]:
    """Replace live_get. Returns the calls made. An Exception outcome is raised."""
    calls: list[dict[str, Any]] = []

    def fake(source: str, url: str, **kwargs: Any) -> Any:
        calls.append({"source": source, "url": url, **kwargs})
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(live_github, "live_get", fake)
    return calls


# ---------------------------------------------------------------------------
# live_github_pull_requests
# ---------------------------------------------------------------------------


def test_pull_requests_return_the_stored_fields_and_a_live_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_get(monkeypatch, [_pr()])

    result = live_github.live_github_pull_requests()

    assert result["source"] == "GitHub"
    assert result["live"] is True
    assert result["fetched_at"].endswith(" UTC")
    assert result["rows"] == [
        {
            "number": 49,
            "title": "AGENTS-91 Record when each source was last synced",
            "author_login": "SivarajuRB-web",
            "state": "closed",
            "created_at": "2026-10-09T08:00:00",
            "merged_at": "2026-10-09T09:30:00",
            "closed_at": "2026-10-09T09:30:00",
            "head_branch": "AGENTS-87-92-live-fallback",
            "ticket_key": "AGENTS-87",
        }
    ]
    assert calls[0]["url"].endswith("/pulls")
    assert calls[0]["params"] == {
        "state": "all",
        "sort": "created",
        "direction": "desc",
        "per_page": 10,
    }
    assert "Authorization" in calls[0]["headers"]


def test_pull_request_ticket_falls_back_to_the_title(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_get(monkeypatch, [_pr(head={"ref": "feature/x"}, title="AGENTS-20 Fix it")])

    assert live_github.live_github_pull_requests()["rows"][0]["ticket_key"] == "AGENTS-20"


def test_an_open_pull_request_has_no_merge_or_close_time(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_get(monkeypatch, [_pr(state="open", merged_at=None, closed_at=None, user=None)])

    row = live_github.live_github_pull_requests(state="open")["rows"][0]

    assert row["merged_at"] is None
    assert row["closed_at"] is None
    assert row["author_login"] is None


def test_pull_requests_pass_the_state_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_get(monkeypatch, [])

    live_github.live_github_pull_requests(state="open")

    assert calls[0]["params"]["state"] == "open"


def test_pull_requests_reject_an_unknown_state(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_get(monkeypatch, [])

    result = live_github.live_github_pull_requests(state="merged")  # type: ignore[arg-type]

    assert result == {"error": "state must be open, closed or all."}
    assert calls == []


def test_pull_requests_cap_the_number_of_results(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_get(monkeypatch, [])

    live_github.live_github_pull_requests(max_results=500)
    live_github.live_github_pull_requests(max_results=-3)

    assert [call["params"]["per_page"] for call in calls] == [25, 1]


def test_a_full_page_says_there_may_be_more(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_get(monkeypatch, [_pr(1), _pr(2)])

    assert live_github.live_github_pull_requests(max_results=2)["may_be_more"] is True
    assert live_github.live_github_pull_requests(max_results=5)["may_be_more"] is False


def test_no_pull_requests_is_a_message_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_get(monkeypatch, [])

    result = live_github.live_github_pull_requests()

    assert result["rows"] == []
    assert result["message"] == "GitHub has no pull requests that match."
    assert "error" not in result


@pytest.mark.parametrize(
    "message",
    [
        "GitHub did not answer within 10 seconds.",
        "GitHub rejected the credentials (HTTP 401).",
        "GitHub refused the request because the rate limit is used up.",
    ],
)
def test_pull_request_failures_come_back_as_an_error_the_model_can_read(
    monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    _fake_get(monkeypatch, LiveDataError(message))

    assert live_github.live_github_pull_requests() == {"error": message}


# ---------------------------------------------------------------------------
# live_github_commits
# ---------------------------------------------------------------------------


def test_commits_return_the_stored_fields_and_a_live_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_get(monkeypatch, [_commit()])

    result = live_github.live_github_commits()

    assert result["live"] is True
    assert result["rows"] == [
        {
            "sha": "0f7e96f",
            "message": "AGENTS-95 Restore the Render deploy hook",
            "author_login": "SivarajuRB-web",
            "authored_at": "2026-10-09T09:10:00",
            "ticket_key": "AGENTS-95",
        }
    ]
    assert calls[0]["url"].endswith("/commits")
    assert calls[0]["params"] == {"per_page": 10}


def test_commit_without_a_ticket_or_a_github_account(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _commit(author=None)
    entry["commit"]["message"] = "updated yml for failure testing"
    _fake_get(monkeypatch, [entry])

    row = live_github.live_github_commits()["rows"][0]

    assert row["ticket_key"] is None
    assert row["author_login"] is None


def test_commits_since_a_date_ask_from_midnight_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_get(monkeypatch, [])

    live_github.live_github_commits(since="2026-10-09")

    assert calls[0]["params"]["since"] == "2026-10-09T00:00:00Z"


@pytest.mark.parametrize(
    "bad", ["yesterday", "09/10/2026", "2026-13-40", "2026-10-09&per_page=100"]
)
def test_commits_reject_a_since_that_is_not_yyyy_mm_dd(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    calls = _fake_get(monkeypatch, [])

    result = live_github.live_github_commits(since=bad)

    assert result == {"error": "since must be a date written YYYY-MM-DD."}
    assert calls == []


def test_no_commits_is_a_message_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_get(monkeypatch, [])

    result = live_github.live_github_commits(since="2030-01-01")

    assert result["rows"] == []
    assert result["message"] == "GitHub has no commits that match."


def test_commit_failure_comes_back_as_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_get(
        monkeypatch, LiveDataError("GitHub refused the request because the rate limit is used up.")
    )

    assert live_github.live_github_commits() == {
        "error": "GitHub refused the request because the rate limit is used up."
    }


def test_the_tools_never_send_anything_but_get() -> None:
    """live_get is the only way out of this module, and it only sends GET (see test_live_http)."""
    import inspect

    source = inspect.getsource(live_github)

    for word in ("httpx.post", "httpx.put", "httpx.patch", "httpx.delete", "httpx.request"):
        assert word not in source