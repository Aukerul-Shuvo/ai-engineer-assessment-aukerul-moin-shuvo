"""Optional API-key protection for ``POST /ask``.

Off by default so a reviewer can run the service bare. When ``API_KEY`` is configured, every
request must carry it in ``X-API-Key``. Comparison is constant-time so response timing cannot
be used to guess the key one character at a time.
"""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Header, Request

from app.api.errors import UnauthorizedError


async def require_api_key(
    request: Request, x_api_key: Annotated[str | None, Header()] = None
) -> None:
    """FastAPI dependency. No-op unless an API key is configured."""
    expected = request.app.state.settings.api_key
    if expected is None:
        return
    supplied = x_api_key or ""
    if not hmac.compare_digest(supplied.encode(), expected.get_secret_value().encode()):
        raise UnauthorizedError("Missing or invalid API key.")
