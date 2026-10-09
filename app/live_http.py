"""Read-only HTTP for the live-data tools (ADR-023).

Every live tool reads Jira, GitHub or Confluence through live_get(). It sends GET requests
only, so no live tool has a code path that can create, change or delete anything.

Rules from ADR-023:
- 10 seconds per call. A call that times out is not repeated.
- A dropped connection is tried again, up to 3 attempts in all, as in the Actions connector.
- A 401, 403 or 429 answer is a real answer, so it is not repeated.
- Every failure becomes a LiveDataError whose text is written for the model to read. The text
  never contains a token or a password.
"""

import time
from datetime import UTC, datetime
from typing import Any

import httpx

LIVE_TIMEOUT_SECONDS = 10
LIVE_ATTEMPTS = 3


class LiveDataError(Exception):
    """A live read failed. str(error) says why, in words the model can pass on."""


class LiveNotFound(LiveDataError):
    """The source answered 404: nothing exists at that address."""


def fetched_at() -> str:
    """The current time as UTC text. Live results carry it so the answer can say when."""
    return datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds") + " UTC"


def _fail_for_status(source: str, response: httpx.Response) -> None:
    """Raise LiveDataError (or LiveNotFound) for an error status. Do nothing for a good one."""
    code = response.status_code
    if code < 400:
        return
    if code == 404:
        raise LiveNotFound(f"{source} has nothing at this address (HTTP 404).")
    # GitHub answers 403, not 429, when the rate limit is used up.
    out_of_requests = response.headers.get("x-ratelimit-remaining") == "0"
    if code == 429 or (code == 403 and out_of_requests):
        raise LiveDataError(f"{source} refused the request because the rate limit is used up.")
    if code in (401, 403):
        raise LiveDataError(f"{source} rejected the credentials (HTTP {code}).")
    raise LiveDataError(f"{source} returned HTTP {code}.")


def live_get(
    source: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    auth: tuple[str, str] | None = None,
    params: dict[str, Any] | None = None,
) -> Any:
    """GET one URL and return the parsed JSON. `source` names the service in error text."""
    for attempt in range(1, LIVE_ATTEMPTS + 1):
        try:
            response = httpx.get(
                url,
                headers=headers,
                auth=auth,
                params=params,
                timeout=LIVE_TIMEOUT_SECONDS,
            )
            break
        except httpx.TimeoutException:
            raise LiveDataError(
                f"{source} did not answer within {LIVE_TIMEOUT_SECONDS} seconds."
            ) from None
        except httpx.TransportError:
            if attempt == LIVE_ATTEMPTS:
                raise LiveDataError(
                    f"{source} could not be reached after {LIVE_ATTEMPTS} attempts."
                ) from None
            time.sleep(attempt)
    _fail_for_status(source, response)
    try:
        return response.json()
    except ValueError:
        raise LiveDataError(f"{source} returned an answer that is not JSON.") from None