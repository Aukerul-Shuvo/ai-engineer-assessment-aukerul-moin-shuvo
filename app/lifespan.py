"""Startup and shutdown.

The lifespan owns every long-lived resource: the shared HTTP client, the retrieval store, the
model clients. They are built once when the server starts, stored on ``app.state``, and closed
cleanly on shutdown. Request handlers never construct these themselves.

Startup policy when required secrets are missing:

* ``prod``  refuse to start. A misconfigured deployment should fail loudly, not serve errors.
* ``dev``   start and warn. Degraded mode still serves whatever it can.
* ``test``  silent. Tests inject fakes for every external dependency.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI

from app import __version__
from app.api.readiness import ReadinessRegistry
from app.config import Settings

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build shared resources before serving, release them after."""
    settings: Settings = app.state.settings
    _apply_secret_policy(settings)

    # One client for all outbound HTTP. follow_redirects matters: the Superhero API answers
    # every call with a 302 to its www host, and httpx does not follow redirects by default.
    app.state.http = httpx.AsyncClient(
        timeout=settings.superhero_timeout_s,
        follow_redirects=True,
        headers={"User-Agent": f"ai-engineer-assessment/{__version__}"},
    )

    readiness = ReadinessRegistry()
    readiness.register("secrets", lambda: not settings.missing_runtime_secrets())
    app.state.readiness = readiness

    log.info(
        "startup_complete",
        version=__version__,
        environment=settings.environment,
        tools_backend=settings.tools_backend,
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
