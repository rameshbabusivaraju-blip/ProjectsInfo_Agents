"""Tests for app/live_jira.py (AGENTS-87). Fully offline: the HTTP helper is replaced."""

from typing import Any

import pytest

from app import live_jira
from app.live_http import LiveDataError, LiveNotFound


def _issue(key: str = "AGENTS-96", **overrides: Any) -> dict[str, Any]:
    """A Jira issue as the REST API returns it, with only the fields the tool asks for."""
    fields: dict[str, Any] = {
        "summary": "Added today",
        "issuetype": {"name": "Task"},
        "status": {"name": "In Progress"},
        "statusCategory": {"key": "indeterminate"},
        "parent": {"key": "AGENTS-10"},
        "assignee": {"displayName": "Ramesh", "accountId": "acc-1"},
        "created": "2026-10-09T10:00:00.000+0530",
        "updated": "2026-10-09T12:30:00.000+0530",
        "resolutiondate": None,
        "customfield_10016": 3.0,
        "customfield_10020": [{"id": 101, "name": "SCRUM Sprint 4"}],
    }
    fields.update(overrides)
    return {"key": key, "fields": fields}


def _fake_get(monkeypatch: pytest.MonkeyPatch, outcome: Any) -> list[dict[str, Any]]:
    """Replace live_get. Returns the calls made. An Exception outcome is raised."""
    calls: list[dict[str, Any]] = []

    def fake(source: str, url: str, **kwargs: Any) -> Any:
        calls.append({"source": source, "url": url, **kwargs})
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(live_jira, "live_get", fake)
    return calls


# ---------------------------------------------------------------------------
# live_jira_ticket
# ---------------------------------------------------------------------------


def test_ticket_returns_the_stored_fields_and_a_live_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_get(monkeypatch, _issue())

    result = live_jira.live_jira_ticket("AGENTS-96")

    assert result["source"] == "Jira"
    assert result["live"] is True
    assert result["fetched_at"].endswith(" UTC")
    assert result["rows"] == [
        {
            "key": "AGENTS-96",
            "summary": "Added today",
            "issue_type": "Task",
            "status": "In Progress",
            "status_category": "indeterminate",
            "story_points": 3.0,
            "parent_key": "AGENTS-10",
            "assignee": "Ramesh",
            "created": "2026-10-09T04:30:00",
            "updated": "2026-10-09T07:00:00",
            "resolved": None,
            "sprints": ["SCRUM Sprint 4"],
        }
    ]
    assert calls[0]["url"].endswith("/rest/api/3/issue/AGENTS-96")
    assert calls[0]["source"] == "Jira"


def test_ticket_accepts_a_loosely_written_key(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_get(monkeypatch, _issue("AGENTS-14"))

    live_jira.live_jira_ticket("agents 14")

    assert calls[0]["url"].endswith("/issue/AGENTS-14")


def test_ticket_without_a_valid_key_makes_no_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_get(monkeypatch, _issue())

    result = live_jira.live_jira_ticket("the latest one")

    assert "needs a ticket key" in result["error"]
    assert calls == []


def test_ticket_that_does_not_exist_is_an_empty_result_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_get(monkeypatch, LiveNotFound("Jira has nothing at this address (HTTP 404)."))

    result = live_jira.live_jira_ticket("AGENTS-999")

    assert result["rows"] == []
    assert result["message"] == "Jira has no ticket AGENTS-999."
    assert "error" not in result


@pytest.mark.parametrize(
    "message",
    [
        "Jira did not answer within 10 seconds.",
        "Jira rejected the credentials (HTTP 401).",
        "Jira refused the request because the rate limit is used up.",
    ],
)
def test_ticket_failures_come_back_as_an_error_the_model_can_read(
    monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    _fake_get(monkeypatch, LiveDataError(message))

    assert live_jira.live_jira_ticket("AGENTS-1") == {"error": message}


def test_ticket_with_no_assignee_parent_or_sprints(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_get(
        monkeypatch,
        _issue(assignee=None, parent=None, customfield_10020=None, customfield_10016=None),
    )

    row = live_jira.live_jira_ticket("AGENTS-96")["rows"][0]

    assert row["assignee"] is None
    assert row["parent_key"] is None
    assert row["sprints"] == []
    assert row["story_points"] is None


# ---------------------------------------------------------------------------
# live_jira_search
# ---------------------------------------------------------------------------


def test_search_builds_the_query_from_the_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_get(monkeypatch, {"issues": [_issue()]})

    result = live_jira.live_jira_search(
        status="In Progress", issue_type="Task", created_since="2026-10-09", text="live data"
    )

    assert calls[0]["params"]["jql"] == (
        'project = TEST AND status = "In Progress" AND issuetype = "Task" '
        'AND created >= "2026-10-09" AND text ~ "live data" ORDER BY created DESC'
    )
    assert calls[0]["url"].endswith("/rest/api/3/search/jql")
    assert [row["key"] for row in result["rows"]] == ["AGENTS-96"]
    assert result["live"] is True
    assert result["more_available"] is False


def test_search_without_filters_lists_the_newest_tickets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_get(monkeypatch, {"issues": [_issue()]})

    live_jira.live_jira_search()

    assert calls[0]["params"]["jql"] == "project = TEST ORDER BY created DESC"
    assert calls[0]["params"]["maxResults"] == 10


def test_search_caps_the_number_of_results(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_get(monkeypatch, {"issues": []})

    live_jira.live_jira_search(max_results=500)
    live_jira.live_jira_search(max_results=0)

    assert [call["params"]["maxResults"] for call in calls] == [25, 1]


def test_search_says_when_more_tickets_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_get(monkeypatch, {"issues": [_issue()], "nextPageToken": "abc"})

    assert live_jira.live_jira_search()["more_available"] is True


def test_search_with_no_match_is_a_message_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_get(monkeypatch, {"issues": []})

    result = live_jira.live_jira_search(status="Done")

    assert result["rows"] == []
    assert result["message"] == "Jira has no tickets that match these filters."


@pytest.mark.parametrize(
    "bad",
    ['Done" OR project = OTHER', "x; DROP", "a" * 61, "in (progress)", 'a"b'],
)
def test_search_rejects_filter_values_that_could_change_the_query(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    calls = _fake_get(monkeypatch, {"issues": []})

    result = live_jira.live_jira_search(status=bad)

    assert "may only use letters" in result["error"]
    assert calls == []


@pytest.mark.parametrize("bad", ["yesterday", "09/10/2026", "2026-13-40", "2026-10-09 OR 1=1"])
def test_search_rejects_a_date_that_is_not_yyyy_mm_dd(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    calls = _fake_get(monkeypatch, {"issues": []})

    result = live_jira.live_jira_search(created_since=bad)

    assert result == {"error": "created must be a date written YYYY-MM-DD."}
    assert calls == []


def test_search_failure_comes_back_as_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_get(monkeypatch, LiveDataError("Jira rejected the credentials (HTTP 401)."))

    assert live_jira.live_jira_search() == {"error": "Jira rejected the credentials (HTTP 401)."}


def test_the_tools_never_send_anything_but_get() -> None:
    """live_get is the only way out of this module, and it only sends GET (see test_live_http)."""
    import inspect

    source = inspect.getsource(live_jira)

    for word in ("httpx.post", "httpx.put", "httpx.patch", "httpx.delete", "httpx.request"):
        assert word not in source