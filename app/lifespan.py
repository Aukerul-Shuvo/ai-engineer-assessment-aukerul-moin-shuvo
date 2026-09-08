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
from app.llm.embeddings import Embedder, GeminiEmbedder
from app.retrieval.corpus import CorpusPaths
from app.retrieval.rerank import FlashRankReranker, Reranker
from app.retrieval.store import HybridStore
from app.tools.circuit_breaker import CircuitBreaker
from app.tools.superhero import SuperheroClient

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

    # The superhero source. None when no token is configured; the graph then reports the
    # source as unavailable instead of failing the whole request.
    app.state.superhero = (
        SuperheroClient(
            app.state.http,
            base_url=settings.superhero_base_url,
            token=settings.superhero_api_token.get_secret_value(),
            cache_ttl_s=settings.superhero_cache_ttl_s,
            cache_size=settings.superhero_cache_size,
            max_retries=settings.superhero_max_retries,
            breaker=CircuitBreaker(
                failure_threshold=settings.superhero_breaker_failures,
                recovery_s=settings.superhero_breaker_recovery_s,
            ),
        )
        if settings.superhero_api_token
        else None
    )

    # The text corpus. Embedder and reranker are optional; the store degrades without them.
    app.state.embedder = _build_embedder(settings)
    app.state.reranker = await _build_reranker(settings)
    app.state.store = _load_store(settings, app.state.embedder, app.state.reranker)

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


def _build_embedder(settings: Settings) -> Embedder | None:
    if settings.gemini_api_key is None:
        return None
    return GeminiEmbedder(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.gemini_embedding_model,
        dimensions=settings.embedding_dimensions,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )


async def _build_reranker(settings: Settings) -> Reranker | None:
    if not settings.reranker_enabled:
        return None
    return await FlashRankReranker.create(
        model_name=settings.reranker_model,
        cache_dir=settings.reranker_cache_dir,
        max_length=settings.reranker_max_length,
    )


def _load_store(
    settings: Settings, embedder: Embedder | None, reranker: Reranker | None
) -> HybridStore | None:
    """Load the corpus. A missing corpus is fatal in prod and a warning elsewhere."""
    try:
        return HybridStore.load(
            CorpusPaths(settings.data_dir),
            embedder=embedder,
            reranker=reranker,
            expected_embedding_model=settings.gemini_embedding_model,
            expected_dimensions=settings.embedding_dimensions,
            bm25_top_k=settings.bm25_top_k,
            dense_top_k=settings.dense_top_k,
            rrf_k=settings.rrf_k,
            rerank_candidates=settings.rerank_candidates,
            rerank_top_k=settings.rerank_top_k,
        )
    except (FileNotFoundError, ValueError) as exc:
        if settings.environment == "prod":
            raise RuntimeError(f"Refusing to start: corpus not loadable: {exc}") from exc
        log.warning("corpus_unavailable", reason=str(exc), data_dir=str(settings.data_dir))
        return None
