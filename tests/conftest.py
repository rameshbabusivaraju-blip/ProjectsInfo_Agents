"""Shared pytest setup.

app.agent imports jira_connector and github_connector at module level, and
both read several environment variables eagerly so a missing one fails loudly
in real use (python -m app.jira_connector, say). No test calls out to either
API — they only need agent.py to import cleanly — so this fills in
placeholder values before collection runs, ahead of anything that would
otherwise raise KeyError just from being imported.
"""

import os

os.environ.setdefault("GITHUB_TOKEN", "test-token")
os.environ.setdefault("GITHUB_OWNER", "test-owner")
os.environ.setdefault("GITHUB_REPO", "test-repo")
os.environ.setdefault("ATLASSIAN_SITE", "test.atlassian.net")
os.environ.setdefault("ATLASSIAN_EMAIL", "test@example.com")
os.environ.setdefault("ATLASSIAN_API_TOKEN", "test-token")
os.environ.setdefault("JIRA_PROJECT_KEY", "TEST")
