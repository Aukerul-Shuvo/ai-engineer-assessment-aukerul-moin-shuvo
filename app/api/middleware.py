"""Request-scoped middleware.

``RequestContextMiddleware`` wraps every HTTP request and does three things:

1. Assigns a request id. An incoming ``X-Request-ID`` header is honoured so a caller can
   correlate across services; otherwise a UUID is generated. The id is bound to the logging
   context, stored on the request scope for error handlers, and echoed back in the response.
2. Emits exactly one access-log line per request: method, path, status, duration.
3. Clears the logging context afterwards so nothing leaks into the next request.

It is written as raw ASGI rather than Starlette's ``BaseHTTPMiddleware`` because the latter
buffers response bodies, which would break server-sent event streaming on ``POST /ask``.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

import structlog

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

REQUEST_ID_HEADER = "x-request-id"
_MAX_REQUEST_ID_LEN = 128

log = structlog.get_logger("app.access")


class RequestContextMiddleware:
    """Assign a request id, log one access line, keep logging context per request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI entry point."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _incoming_request_id(scope) or uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        started = time.perf_counter()
        status = {"code": 500}  # overwritten when the response starts; 500 if it never does

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                headers = list(message.get("headers", []))
                # Error responses set the header themselves so that 500s raised above this
                # middleware still carry it. Never emit it twice.
                header_name = REQUEST_ID_HEADER.encode()
                if not any(name.lower() == header_name for name, _ in headers):
                    headers.append((header_name, request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            log.info(
                "request",
                method=scope.get("method"),
                path=scope.get("path"),
                status=status["code"],
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            structlog.contextvars.clear_contextvars()


def _incoming_request_id(scope: Scope) -> str | None:
    """Return a caller-supplied ``X-Request-ID`` if present and sane."""
    for name, value in scope.get("headers", []):
        if name == REQUEST_ID_HEADER.encode():
            candidate = value.decode("latin-1").strip()
            return candidate[:_MAX_REQUEST_ID_LEN] or None
    return None
