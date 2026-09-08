"""Per-client rate limiting for ``POST /ask``.

Every request costs model calls, and the free tiers behind them are finite. The limit is keyed
by API key when one is supplied, otherwise by client address, and enforced by slowapi with an
in-memory store. That is right for one process; a multi-instance deployment would point slowapi
at Redis through ``storage_uri``, which the ADR names as the scale-up step.

Exceeding the limit returns the standard error envelope with ``Retry-After`` and
``X-RateLimit-*`` headers, so clients can back off precisely.
"""

from __future__ import annotations

from fastapi import Request, Response, status
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.errors import error_response
from app.config import Settings


def client_key(request: Request) -> str:
    """API key if present, else the caller's address."""
    api_key = request.headers.get("x-api-key")
    return f"key:{api_key}" if api_key else f"ip:{get_remote_address(request)}"


def build_limiter(settings: Settings) -> Limiter:
    """A limiter with rate-limit headers on every response it inspects."""
    return Limiter(key_func=client_key, headers_enabled=True, enabled=bool(settings.rate_limit))


async def rate_limit_exceeded(request: Request, exc: Exception) -> Response:
    """Envelope the 429 and let slowapi add its headers."""
    detail = exc.detail if isinstance(exc, RateLimitExceeded) else str(exc)
    response: Response = error_response(
        request,
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        error_type="rate_limited",
        message=f"Rate limit exceeded: {detail}",
    )
    limiter: Limiter = request.app.state.limiter
    view_limit = getattr(request.state, "view_rate_limit", None)
    if view_limit is not None:
        # Same call slowapi's own handler makes; it is the only way to get Retry-After.
        response = limiter._inject_headers(response, view_limit)
    return response
