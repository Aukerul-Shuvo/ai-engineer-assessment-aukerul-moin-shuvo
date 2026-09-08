"""Startup and shutdown.

The lifespan owns every long-lived resource: the shared HTTP client, the Superhero client, the
embedder, the reranker, the retrieval store. They are built once when the server starts, stored
on ``app.state``, and closed cleanly on shutdown. Request handlers never construct these
themselves. The construction itself lives in ``app.resources`` so the MCP server can reuse it.

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
from app.api.readiness import ReadinessRegistry
from app.config import Settings
from app.resources import (
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

    readiness = ReadinessRegistry()
    readiness.register("secrets", lambda: not settings.missing_runtime_secrets())
    readiness.register("superhero_client", lambda: app.state.superhero is not None)
    readiness.register("retrieval_index", lambda: app.state.store is not None)
    app.state.readiness = readiness

    log.info(
        "startup_complete",
        version=__version__,
        environment=settings.environment,
        tools_backend=settings.tools_backend,
        dense_retrieval=bool(app.state.store and app.state.store.dense_enabled),
        reranker=app.state.reranker.model_id if app.state.reranker else None,
    )
    try:
        yield
    finally:
        await app.state.http.aclose()
        log.info("shutdown_complete")


def _apply_secret_policy(settings: Settings) -> None:
    missing = settings.missing_runtime_secrets()
    if not missing or settings.environment == "test":
        return
    if settings.environment == "prod":
        raise RuntimeError(f"Refusing to start: missing required secrets {missing}")
    log.warning("missing_secrets", missing=missing, note="starting in degraded mode")
