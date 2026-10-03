"""API key check for the protected routes (AGENTS-53).

A caller proves who it is by sending the key in an X-API-Key header. The key
lives in the PROJECTPULSE_API_KEY environment variable, never in code. In .NET
terms this is a small authorization filter that runs before the action.
"""

from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Reject the request unless X-API-Key matches PROJECTPULSE_API_KEY.

    The key is read on every call so a changed value is picked up without code
    changes. If no key is configured the route stays closed (503) instead of
    open, so a missing setting can never expose the API by accident. The
    comparison is constant-time so response timing does not leak the key.
    """
    expected = os.environ.get("PROJECTPULSE_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="API key is not configured on the server")
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
