"""One error shape for every failure the API returns.

Clients parse a single envelope::

    {"error": {"type": "not_found", "message": "...", "request_id": "...", "details": []}}

``AppError`` subclasses are raised by lower layers (tools, retrieval, graph, model clients) and
mapped to HTTP status codes here, in one place. Anything unexpected becomes a 500 that carries
the request id and never a stack trace; the trace goes to the log under the same request id.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.middleware import REQUEST_ID_HEADER

log = structlog.get_logger(__name__)


class AppError(Exception):
    """Base class for errors that map to a deliberate HTTP response."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type: str = "internal_error"

    def __init__(self, message: str, *, details: list[Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or []

    def describe(self) -> str:
        """Message plus details, for logs and degraded-mode reasons."""
        if not self.details:
            return self.message
        return f"{self.message}: {'; '.join(str(item) for item in self.details)}"


class BadRequestError(AppError):
    """The request was well-formed but cannot be served as asked."""

    status_code = status.HTTP_400_BAD_REQUEST
    error_type = "bad_request"


class UnauthorizedError(AppError):
    """A required API key is missing or wrong."""

    status_code = status.HTTP_401_UNAUTHORIZED
    error_type = "unauthorized"


class NotReadyError(AppError):
    """The service is up but a dependency it needs is not available yet."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_type = "not_ready"


class UpstreamError(AppError):
    """A dependency answered, but with something we could not use."""

    status_code = status.HTTP_502_BAD_GATEWAY
    error_type = "upstream_error"


class UpstreamUnavailableError(AppError):
    """A dependency timed out or refused the connection."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_type = "upstream_unavailable"


class UpstreamRateLimitedError(AppError):
    """A dependency told us to slow down; the caller should retry later."""

    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    error_type = "upstream_rate_limited"


_STATUS_TYPES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    415: "unsupported_media_type",
    429: "rate_limited",
}


def error_response(
    request: Request,
    *,
    status_code: int,
    error_type: str,
    message: str,
    details: list[Any] | None = None,
) -> JSONResponse:
    """Build the standard error envelope, stamping the request id in body and header."""
    request_id: str | None = getattr(request.state, "request_id", None)
    body = {
        "error": {
            "type": error_type,
            "message": message,
            "request_id": request_id,
            "details": jsonable_encoder(details or []),
        }
    }
    headers = {REQUEST_ID_HEADER: request_id} if request_id else None
    return JSONResponse(body, status_code=status_code, headers=headers)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach every exception handler to the application."""

    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        log.warning("app_error", error_type=exc.error_type, message=exc.message)
        return error_response(
            request,
            status_code=exc.status_code,
            error_type=exc.error_type,
            message=exc.message,
            details=exc.details,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(
            request,
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            error_type="validation_error",
            message="Request validation failed.",
            details=list(exc.errors()),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return error_response(
            request,
            status_code=exc.status_code,
            error_type=_STATUS_TYPES.get(exc.status_code, "http_error"),
            message=str(exc.detail),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_error")
        return error_response(
            request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_type="internal_error",
            message="Internal server error.",
        )
