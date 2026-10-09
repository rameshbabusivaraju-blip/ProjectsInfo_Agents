"""Live Confluence tool: find a page by title or text and return it as plain text (ADR-023).

The tool reads from Confluence itself, only when the stored pages cannot answer, for example
for a page written after the last sync. It searches the project's space, takes the best match
and returns its text and link. The text is converted the same way the Confluence connector
converts it: Confluence renders the page, then the HTML becomes Markdown. Nothing is written to
the database, and Confluence receives GET requests only (see live_http).

The space key is read when the tool runs, not when the file is imported, so an app that has no
CONFLUENCE_SPACE_KEY still starts. The tool then answers with a plain error.
"""

import os
import re
from datetime import UTC, datetime
from typing import Any

from markdownify import markdownify
from pydantic import BaseModel, Field

from app.jira_connector import AUTH, BASE_URL
from app.live_http import LiveDataError, fetched_at, live_get

WIKI_URL = f"{BASE_URL}/wiki"

# How many pages the search looks at. The best match is read in full. The others are only listed.
MATCHES = 3

# The longest page text returned. A longer page is cut and marked, so one page cannot fill the
# model's context.
MAX_TEXT_CHARS = 4000

# Letters, digits, spaces, dots, dashes and underscores only. A quote or a CQL keyword
# character cannot get through, so the search words cannot change the query.
_SAFE_TEXT = re.compile(r"^[A-Za-z0-9 _.\-]{1,60}$")


def _time(value: str | None) -> str | None:
    """A Confluence timestamp as UTC text, like the stored ones, or None."""
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(UTC).replace(tzinfo=None).isoformat(timespec="seconds")


def build_cql(space_key: str, query: str) -> str:
    """Build the search from the space and the words. Raises ValueError for a bad query."""
    if not _SAFE_TEXT.match(query):
        raise ValueError(
            "The query may only use letters, digits, spaces, dots, dashes and underscores, "
            "up to 60 characters."
        )
    return f'space = "{space_key}" AND type = page AND (title ~ "{query}" OR text ~ "{query}")'


def _page_text(page_id: str) -> str:
    """One page's content: Confluence renders it, then the HTML becomes Markdown."""
    detail = live_get(
        "Confluence",
        f"{WIKI_URL}/api/v2/pages/{page_id}",
        auth=AUTH,
        params={"body-format": "export_view"},
    )
    html: str = detail["body"]["export_view"]["value"]
    text: str = markdownify(html, heading_style="ATX").strip()
    return text


def live_confluence_page(query: str) -> dict[str, Any]:
    """Find the best matching Confluence page for the words given, and read it now."""
    space_key = os.environ.get("CONFLUENCE_SPACE_KEY")
    if not space_key:
        return {"error": "CONFLUENCE_SPACE_KEY is not set, so live Confluence reads are off."}
    try:
        cql = build_cql(space_key, query)
    except ValueError as exc:
        return {"error": str(exc)}

    try:
        found = live_get(
            "Confluence",
            f"{WIKI_URL}/rest/api/content/search",
            auth=AUTH,
            params={"cql": cql, "limit": MATCHES, "expand": "version"},
        )
        matches = found.get("results", [])
        if not matches:
            return {
                "source": "Confluence",
                "live": True,
                "fetched_at": fetched_at(),
                "sources": [],
                "message": "Confluence has no page in the project space that matches.",
            }
        best = matches[0]
        text = _page_text(str(best["id"]))
    except LiveDataError as exc:
        return {"error": str(exc)}

    truncated = len(text) > MAX_TEXT_CHARS
    return {
        "source": "Confluence",
        "live": True,
        "fetched_at": fetched_at(),
        "sources": [
            {
                "title": best["title"],
                "url": f"{WIKI_URL}{best.get('_links', {}).get('webui', '')}",
                "text": text[:MAX_TEXT_CHARS],
                "updated": _time(best.get("version", {}).get("when")),
            }
        ],
        "truncated": truncated,
        "other_matches": [match["title"] for match in matches[1:]],
    }


class LiveConfluencePageArgs(BaseModel):
    """Arguments the model may give live_confluence_page."""

    query: str = Field(
        description="A page title or a few words from the page, up to 60 characters."
    )


PAGE_DESCRIPTION = (
    "Find a page in the project's Confluence space by title or text, straight from "
    "Confluence, right now, and return its text and link. Use it only when "
    "search_documents finds no matching passage, or its freshness says the stored pages "
    "are stale, for example for a page written after the last sync. Returns the best "
    "match, up to 4000 characters, and the titles of other matches."
)