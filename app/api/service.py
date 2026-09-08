"""The ask service: run the graph for one question and shape the result into a response.

The HTTP route stays thin; this class owns the sequence:

1. enforce the configured question length
2. serve from the response cache when the request is stateless and cached
3. load session history, run the graph under a deadline
4. turn graph state into ``AskResponse``: sources labelled S1.., plan summary, provenance, meta
5. remember the turn for the session and cache the response when appropriate

``stream`` does the same work while emitting progress events as each graph node finishes, for
clients that ask for ``text/event-stream``. Both paths build the response through one function so
the streamed ``done`` event and the JSON body are identical.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Protocol

import structlog

from app.api.cache import ResponseCache
from app.api.errors import AppError, RequestTimeoutError, ValidationFailedError
from app.api.schemas import (
    AskRequest,
    AskResponse,
    PlanSubQuery,
    PlanSummary,
    ResponseMeta,
    Source,
    SourceLocator,
    SourceRetrieval,
)
from app.api.sessions import SessionStore
from app.graph.schemas import Evidence

if TYPE_CHECKING:
    from app.observability.metrics import AskMetrics

log = structlog.get_logger(__name__)

UNGROUNDED_CAVEAT = " (Some statements could not be verified against the sources.)"


class GraphRunner(Protocol):
    """The two LangGraph methods the service uses, so tests can substitute a stand-in."""

    async def ainvoke(self, input: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Run to completion."""
        ...

    def astream(self, input: dict[str, Any], **kwargs: Any) -> AsyncIterator[Any]:
        """Yield progress."""
        ...


class AskService:
    """Answers questions. One instance per process, built in the lifespan."""

    def __init__(
        self,
        *,
        graph: GraphRunner,
        sessions: SessionStore,
        cache: ResponseCache,
        max_question_chars: int,
        timeout_s: float,
        metrics: AskMetrics | None = None,
    ) -> None:
        self._graph = graph
        self._sessions = sessions
        self._cache = cache
        self._max_question_chars = max_question_chars
        self._timeout_s = timeout_s
        self._metrics = metrics

    # ------------------------------------------------------------------ JSON path
    async def ask(self, request: AskRequest, *, request_id: str | None) -> AskResponse:
        """Answer one question."""
        self._check_length(request.question)
        started = time.perf_counter()

        if request.session_id is None and (cached := self._cache.get(request.question)):
            log.info("ask_cache_hit")
            response = cached.model_copy(
                update={
                    "meta": cached.meta.model_copy(
                        update={
                            "request_id": request_id,
                            "cached": True,
                            "latency_ms": _elapsed_ms(started),
                        }
                    )
                }
            )
            self._record(response)
            return response

        try:
            state = await asyncio.wait_for(
                self._graph.ainvoke(self._graph_input(request)), timeout=self._timeout_s
            )
        except TimeoutError as exc:
            raise RequestTimeoutError(f"Answering took longer than {self._timeout_s:.0f}s") from exc

        response = self.build_response(
            state, request_id=request_id, session_id=request.session_id, started=started
        )
        self._finish(request, response)
        return response

    # ------------------------------------------------------------------ SSE path
    async def stream(
        self, request: AskRequest, *, request_id: str | None
    ) -> AsyncIterator[dict[str, str]]:
        """Progress events per graph node, then ``done`` with the full response."""
        try:
            self._check_length(request.question)
        except AppError as exc:
            yield _event("error", {"type": exc.error_type, "message": exc.message})
            return

        started = time.perf_counter()
        final_state: dict[str, Any] = {}
        try:
            async with asyncio.timeout(self._timeout_s):
                async for mode, chunk in self._graph.astream(
                    self._graph_input(request), stream_mode=["updates", "values"]
                ):
                    if mode == "values":
                        final_state = chunk
                        continue
                    for node, update in (chunk or {}).items():
                        event = _progress_event(node, update or {})
                        if event is not None:
                            yield event
        except TimeoutError:
            yield _event(
                "error", {"type": "timeout", "message": f"Exceeded {self._timeout_s:.0f}s"}
            )
            return
        except AppError as exc:
            yield _event("error", {"type": exc.error_type, "message": exc.message})
            return

        response = self.build_response(
            final_state, request_id=request_id, session_id=request.session_id, started=started
        )
        self._finish(request, response)
        yield _event("done", response.model_dump(mode="json"))

    # ------------------------------------------------------------------ shaping
    def build_response(
        self,
        state: dict[str, Any],
        *,
        request_id: str | None,
        session_id: str | None,
        started: float,
        cached: bool = False,
    ) -> AskResponse:
        """Graph state to API response. The only place the response shape is decided."""
        evidence: list[Evidence] = state.get("cited_evidence") or []
        sources = [_source(f"S{index}", item) for index, item in enumerate(evidence, start=1)]

        answer: str | None = state.get("answer")
        grounded: bool | None = state.get("grounded")
        if answer and grounded is False:
            answer = answer.rstrip() + UNGROUNDED_CAVEAT

        plan = state.get("plan")
        plan_summary = (
            PlanSummary(
                intent=plan.intent,
                sub_queries=[
                    PlanSubQuery(id=sq.id, text=sq.text, source=sq.source, hero_name=sq.hero_name)
                    for sq in plan.sub_queries
                ],
            )
            if plan is not None
            else None
        )

        return AskResponse(
            answer=answer,
            sources=sources,
            citations=list(state.get("citations") or []),
            plan=plan_summary,
            meta=ResponseMeta(
                request_id=request_id,
                session_id=session_id,
                providers=list(state.get("providers") or []),
                latency_ms=_elapsed_ms(started),
                grounded=grounded,
                degraded=bool(state.get("degraded")),
                degraded_reason=state.get("degraded_reason"),
                retrieval_mode=_retrieval_mode(evidence),
                cached=cached,
                notes=list(state.get("notes") or []),
            ),
        )

    # ------------------------------------------------------------------ internals
    def _check_length(self, question: str) -> None:
        if len(question) > self._max_question_chars:
            raise ValidationFailedError(
                f"question exceeds the limit of {self._max_question_chars} characters",
                details=[{"loc": ["body", "question"], "max_length": self._max_question_chars}],
            )

    def _graph_input(self, request: AskRequest) -> dict[str, Any]:
        history = self._sessions.history(request.session_id) if request.session_id else []
        return {"question": request.question, "history": history}

    def _finish(self, request: AskRequest, response: AskResponse) -> None:
        if response.answer and request.session_id:
            self._sessions.record(request.session_id, request.question, response.answer)
        if request.session_id is None and response.answer and not response.meta.degraded:
            self._cache.put(request.question, response)
        self._record(response)

    def _record(self, response: AskResponse) -> None:
        if self._metrics is not None:
            self._metrics.record(response)


def _source(label: str, item: Evidence) -> Source:
    return Source(
        id=label,
        type=item.kind,
        title=item.title,
        reference=item.reference,
        url=item.url,
        excerpt=item.excerpt,
        locator=SourceLocator.model_validate(item.locator) if item.locator else None,
        retrieval=SourceRetrieval.model_validate(item.retrieval) if item.retrieval else None,
        data=item.data,
    )


def _retrieval_mode(evidence: list[Evidence]) -> str | None:
    """Hybrid if any dataset hit came from the dense index, BM25-only otherwise, None if none."""
    retrievals = [
        item.retrieval for item in evidence if item.kind == "dataset" and item.retrieval is not None
    ]
    if not retrievals:
        return None
    dense = any("dense" in retrieval.get("found_by", []) for retrieval in retrievals)
    return "hybrid" if dense else "bm25_only"


def _progress_event(node: str, update: dict[str, Any]) -> dict[str, str] | None:
    """Translate one node's state update into a client-facing event, or nothing."""
    if node == "analyze_query":
        plan = update.get("plan")
        return _event(
            "plan",
            {
                "intent": plan.intent if plan else None,
                "sub_queries": [sq.model_dump() for sq in plan.sub_queries] if plan else [],
                "degraded": bool(update.get("degraded")),
            },
        )
    if node in ("retrieve_documents", "superhero_agent"):
        items = update.get("evidence") or []
        return _event(
            "evidence",
            {
                "branch": node,
                "sub_queries": sorted({item.sub_query_id for item in items}),
                "count": len(items),
                "notes": update.get("notes") or [],
            },
        )
    if node == "resolve_dependencies":
        return _event("wave", {"wave": update.get("wave"), "ready": len(update.get("ready", []))})
    if node == "synthesize":
        return _event(
            "answer",
            {"answer": update.get("answer"), "citations": update.get("citations") or []},
        )
    if node == "check_grounding":
        return _event(
            "grounding",
            {"grounded": update.get("grounded"), "issues": update.get("grounding_issues") or []},
        )
    return None


def _event(name: str, payload: dict[str, Any]) -> dict[str, str]:
    return {"event": name, "data": json.dumps(payload, default=str)}


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
