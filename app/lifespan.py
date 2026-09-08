"""Startup and shutdown.

The lifespan owns every long-lived resource: the shared HTTP client, the Superhero client, the
embedder, the reranker, the retrieval store, the chat models, the agent tools, the compiled
graph, the session store, the response cache and the ask service. They are built once when the
server starts, stored on ``app.state``, and closed cleanly on shutdown. Request handlers never
construct these themselves. Construction lives in ``app.resources`` so the MCP server reuses it.

Startup policy when required secrets are missing:

* ``prod``  refuse to start. A misconfigured deployment should fail loudly, not serve errors.
* ``dev``   start and warn. Degraded mode still serves whatever it can.
* ``test``  silent. Tests inject fakes for every external dependency.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app import __version__
from app.api.cache import ResponseCache
from app.api.readiness import ReadinessRegistry
from app.api.service import AskService
from app.api.sessions import SessionStore
from app.config import Settings
from app.graph.build import GraphDependencies, build_graph
from app.llm.providers import build_chat_models
from app.resources import (
    build_agent_tools,
    build_embedder,
    build_http_client,
    build_reranker,
    build_superhero_client,
    load_store,
)

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build shared resources before serving, release them after."""
    settings: Settings = app.state.settings
    _apply_secret_policy(settings)

    app.state.http = build_http_client(settings)
    app.state.superhero = build_superhero_client(settings, app.state.http)
    app.state.embedder = build_embedder(settings)
    app.state.reranker = await build_reranker(settings)
    app.state.store = load_store(settings, app.state.embedder, app.state.reranker)
    app.state.models = build_chat_models(settings)
    tools, app.state.mcp_bridge = await build_agent_tools(settings, app.state.superhero)

    app.state.graph = build_graph(
        GraphDependencies(
            models=app.state.models,
            store=app.state.store,
            superhero_tools=tools,
            document_titles=(
                [doc.title for doc in app.state.store.list_documents()] if app.state.store else []
            ),
            max_agent_steps=settings.max_agent_steps,
            max_query_rewrites=settings.max_query_rewrites,
            max_regenerations=settings.max_regenerations,
            max_sub_queries=settings.max_sub_queries,
            grade_top_k=settings.grade_top_k,
            evidence_per_sub_query=settings.evidence_per_sub_query,
        )
    )
    app.state.sessions = SessionStore(
        ttl_s=settings.session_ttl_s, max_turns=settings.session_max_turns
    )
    app.state.response_cache = ResponseCache(
        ttl_s=settings.response_cache_ttl_s, max_size=settings.response_cache_size
    )
    app.state.ask_service = AskService(
        graph=app.state.graph,
        sessions=app.state.sessions,
        cache=app.state.response_cache,
        max_question_chars=settings.max_question_chars,
        timeout_s=settings.ask_timeout_s,
    )

    readiness = ReadinessRegistry()
    readiness.register("secrets", lambda: not settings.missing_runtime_secrets())
    readiness.register("superhero_client", lambda: app.state.superhero is not None)
    readiness.register("retrieval_index", lambda: app.state.store is not None)
    readiness.register("llm_configured", lambda: app.state.models is not None)
    app.state.readiness = readiness

    log.info(
        "startup_complete",
        version=__version__,
        environment=settings.environment,
        tools_backend=settings.tools_backend,
        providers=[p.name for p in app.state.models.providers] if app.state.models else [],
        dense_retrieval=bool(app.state.store and app.state.store.dense_enabled),
        reranker=app.state.reranker.model_id if app.state.reranker else None,
    )
    try:
        yield
    finally:
        if app.state.mcp_bridge is not None:
            await app.state.mcp_bridge.__aexit__(None, None, None)
        await app.state.http.aclose()
        log.info("shutdown_complete")


def _apply_secret_policy(settings: Settings) -> None:
    missing = settings.missing_runtime_secrets()
    if not missing or settings.environment == "test":
        return
    if settings.environment == "prod":
        raise RuntimeError(f"Refusing to start: missing required secrets {missing}")
    log.warning("missing_secrets", missing=missing, note="starting in degraded mode")
