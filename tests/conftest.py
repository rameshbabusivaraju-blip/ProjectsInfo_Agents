"""Shared pytest setup.

app.agent imports jira_connector and github_connector at module level, and
both read several environment variables eagerly so a missing one fails loudly
in real use (python -m app.jira_connector, say). No test calls out to either
API — they only need agent.py to import cleanly — so this fills in
placeholder values before collection runs, ahead of anything that would
otherwise raise KeyError just from being imported.
"""

import os

import pytest

os.environ.setdefault("GITHUB_TOKEN", "test-token")
os.environ.setdefault("GITHUB_OWNER", "test-owner")
os.environ.setdefault("GITHUB_REPO", "test-repo")
os.environ.setdefault("ATLASSIAN_SITE", "test.atlassian.net")
os.environ.setdefault("ATLASSIAN_EMAIL", "test@example.com")
os.environ.setdefault("ATLASSIAN_API_TOKEN", "test-token")
os.environ.setdefault("JIRA_PROJECT_KEY", "TEST")
os.environ.setdefault("PROJECTPULSE_API_KEY", "test-key")

@pytest.fixture(autouse=True)
def fixed_agent_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test in fixed mode, whatever AGENT_MODE says in the developer's .env.

    Tests that need the loop set AGENT_MODE themselves. Without this, a .env with
    AGENT_MODE=loop sends /ask tests to the real model.
    """
    monkeypatch.setenv("AGENT_MODE", "fixed")