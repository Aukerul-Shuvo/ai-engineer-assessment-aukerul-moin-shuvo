"""``POST /ask``, the one endpoint the assessment asks for.

Content negotiation keeps it a single endpoint while supporting two delivery styles: a JSON body
by default, or a server-sent event stream when the client sends ``Accept: text/event-stream``.
The stream carries one event per graph stage (plan, evidence, answer, grounding) and ends with a
``done`` event whose payload is exactly the JSON body the non-streaming call would return.

The router is built by a function rather than declared at import time because the rate limit
string comes from settings and slowapi binds it when the route is decorated.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, Request, Response, status
from slowapi import Limiter
from sse_starlette import EventSourceResponse

from app.api.auth import require_api_key
from app.api.schemas import AskRequest, AskResponse, ErrorEnvelope
from app.api.service import AskService
from app.config import Settings

_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_200_OK: {
        "description": "The answer with its sources, or an event stream of progress.",
        "content": {
            "application/json": {},
            "text/event-stream": {
                "schema": {"type": "string"},
                "example": 'event: plan\ndata: {"intent": "answerable", ...}\n\n',
            },
        },
    },
    status.HTTP_401_UNAUTHORIZED: {"model": ErrorEnvelope, "description": "API key required."},
    status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorEnvelope, "description": "Bad input."},
    status.HTTP_429_TOO_MANY_REQUESTS: {"model": ErrorEnvelope, "description": "Slow down."},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorEnvelope, "description": "Not ready."},
    status.HTTP_504_GATEWAY_TIMEOUT: {"model": ErrorEnvelope, "description": "Took too long."},
}


def build_ask_router(settings: Settings, limiter: Limiter) -> APIRouter:
    """The ``/ask`` route bound to this process's rate limit."""
    router = APIRouter(tags=["ask"])

    @router.post(
        "/ask",
        response_model=AskResponse,
        responses=_RESPONSES,
        dependencies=[Depends(require_api_key)],
        summary="Ask a question",
    )
    @limiter.limit(settings.rate_limit)
    async def ask(
        request: Request,
        response: Response,
        body: AskRequest,
        accept: Annotated[str | None, Header()] = None,
    ) -> Any:
        """Answer from the text corpus, the Superhero API, or both, with sources.

        Send ``Accept: text/event-stream`` to receive progress events instead of one JSON body.
        """
        service: AskService = request.app.state.ask_service
        request_id = getattr(request.state, "request_id", None)
        if accept and "text/event-stream" in accept:
            # LF separators rather than the CRLF default: valid SSE, and readable in curl output.
            return EventSourceResponse(service.stream(body, request_id=request_id), sep="\n")
        return await service.ask(body, request_id=request_id)

    return router
