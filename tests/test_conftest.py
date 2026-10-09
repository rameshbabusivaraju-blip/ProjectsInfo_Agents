"""Shared pytest setup.

app.agent imports jira_connector and github_connector at module level, and
both read several environment variables eagerly so a missing one fails loudly
in real use (python -m app.jira_connector, say). No test calls out to either
API — they only need agent.py to import cleanly — so this sets placeholder
values before collection runs, ahead of anything that would otherwise raise
KeyError just from being imported.

The values are set, not defaulted. A developer's shell or .env often holds the real
site, project key and API key, and a test that expects the placeholder would then fail.
Setting them makes the tests give the same result on every machine.
"""

import os

import pytest

os.environ.update(
    {
        "GITHUB_TOKEN": "test-token",
        "GITHUB_OWNER": "test-owner",
        "GITHUB_REPO": "test-repo",
        "ATLASSIAN_SITE": "test.atlassian.net",
        "ATLASSIAN_EMAIL": "test@example.com",
        "ATLASSIAN_API_TOKEN": "test-token",
        "JIRA_PROJECT_KEY": "TEST",
        "CONFLUENCE_SPACE_KEY": "TEST",
        "PROJECTPULSE_API_KEY": "test-key",
    }
)


@pytest.fixture(autouse=True)
def fixed_agent_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test in fixed mode, whatever AGENT_MODE says in the developer's .env.

    Tests that need the loop set AGENT_MODE themselves. Without this, a .env with
    AGENT_MODE=loop sends /ask tests to the real model.
    """
    monkeypatch.setenv("AGENT_MODE", "fixed")