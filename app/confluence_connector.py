"""Pull the Confluence space into the local SQLite store.

Confluence stores page content in its own "storage format" — XHTML with
Confluence-specific macro tags — which is awkward to read as-is. This asks
Confluence to render each page first ("export_view", macros already expanded
into plain HTML) and converts that HTML to Markdown text. Same idea as
_adf_text() in jira_connector.py: don't hand-parse a markup format yourself,
flatten it with a small purpose-built step.

Run it with:  python -m app.confluence_connector
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import httpx
from dotenv import load_dotenv
from markdownify import markdownify
from sqlalchemy import delete
from sqlmodel import Session

from app.db import engine, init_db
from app.models import ConfluencePage, Person

load_dotenv()

# Same Atlassian site, email and API token as the Jira connector — Confluence
# and Jira are the same Atlassian Cloud site, so nothing new to add for auth.
ATLASSIAN_SITE = os.environ["ATLASSIAN_SITE"]
ATLASSIAN_EMAIL = os.environ["ATLASSIAN_EMAIL"]
ATLASSIAN_API_TOKEN = os.environ["ATLASSIAN_API_TOKEN"]
CONFLUENCE_SPACE_KEY = os.environ["CONFLUENCE_SPACE_KEY"]

WIKI_URL = f"https://{ATLASSIAN_SITE}/wiki"
AUTH = (ATLASSIAN_EMAIL, ATLASSIAN_API_TOKEN)


def _get(url: str, **params: Any) -> dict[str, Any]:
    """Call a Confluence endpoint and return the parsed JSON.

    url may be a /wiki-relative path or the full next-page URL Confluence
    hands back — passed straight through rather than reconstructed by hand,
    since the cursor value inside it is not meant to be built manually.
    """
    full_url = url if url.startswith("http") else f"{WIKI_URL}{url}"
    response = httpx.get(full_url, auth=AUTH, params=params, timeout=30)
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def _required_dt(value: str) -> datetime:
    """Confluence timestamps are ISO with a Z suffix. Store them as naive UTC."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(UTC).replace(tzinfo=None)


def _space_id(key: str) -> str:
    """Resolve a space key to the numeric id the v2 pages endpoint needs.

    Uses the older v1 endpoint on purpose: Confluence's newer v2 API still has
    coverage gaps for some bulk/list operations, but this specific v1 lookup
    is long-stable. Worth re-checking if Atlassian closes that gap later.
    """
    space: dict[str, Any] = _get(f"/rest/api/space/{key}")
    return str(space["id"])


def fetch_pages(space_id: str) -> list[dict[str, Any]]:
    """Every page in the space — id, title, current version. No body yet."""
    pages: list[dict[str, Any]] = []
    next_url: str | None = f"/api/v2/spaces/{space_id}/pages?limit=100"

    while next_url:
        page = _get(next_url)
        pages.extend(page.get("results", []))
        next_url = page.get("_links", {}).get("next")

    return pages


def fetch_body_markdown(page_id: str) -> str:
    """One page's content, rendered then converted to Markdown text.

    One extra call per page — same shape as fetch_worklogs() in
    jira_connector.py and fetch_reviews() in github_connector.py: a cheap
    list, then an expensive detail call per item.
    """
    detail = _get(f"/api/v2/pages/{page_id}", **{"body-format": "export_view"})
    html: str = detail["body"]["export_view"]["value"]
    text: str = markdownify(html, heading_style="ATX").strip()
    return text


def load() -> None:
    """Refresh the Confluence table from the API.

    Only this table is cleared, so Jira and GitHub data in the same database
    are untouched — same convention as github_connector.load().
    """
    space_id = _space_id(CONFLUENCE_SPACE_KEY)
    pages = fetch_pages(space_id)

    init_db()

    with Session(engine) as session:
        session.execute(delete(ConfluencePage))

        # Confluence authors share the same Atlassian account id as Jira
        # users, so known people are updated in place rather than duplicated.
        # A brand-new author has no display name available from this
        # endpoint; the account id is stored as a placeholder until that
        # person turns up in a Jira load too.
        people: dict[str, Person] = {}

        def remember(account_id: str | None) -> str | None:
            if not account_id:
                return None
            if account_id not in people:
                people[account_id] = Person(account_id=account_id, display_name=account_id)
            return account_id

        for summary in pages:
            page_id: str = summary["id"]
            version = summary["version"]
            links = summary.get("_links", {})

            session.add(
                ConfluencePage(
                    id=page_id,
                    title=summary["title"],
                    space_key=CONFLUENCE_SPACE_KEY,
                    url=f"{WIKI_URL}{links.get('webui', '')}",
                    version=version["number"],
                    version_author_id=remember(version.get("authorId")),
                    updated=_required_dt(version["createdAt"]),
                    body_text=fetch_body_markdown(page_id),
                )
            )

        for person in people.values():
            session.merge(person)

        session.commit()

    print(f"Loaded {len(pages)} Confluence pages from space '{CONFLUENCE_SPACE_KEY}'")


if __name__ == "__main__":
    load()
