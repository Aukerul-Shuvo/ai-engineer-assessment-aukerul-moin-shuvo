"""Construction of shared runtime resources from settings.

Both the API (in ``lifespan``) and the standalone MCP server build the same objects: the HTTP
client, the Superhero client, the embedder, the reranker, the retrieval store. Defining that
wiring once here means the two entry points cannot drift apart, and a change to how a dependency
is configured happens in exactly one place.

Every builder returns ``None`` when the dependency is not configured. Callers decide what that
means: the API degrades, readiness reports it, tests inject fakes.
"""

from __future__ import annotations

import httpx
import structlog
from langchain_core.tools import BaseTool

from app import __version__
from app.config import Settings
from app.graph.mcp_tools import MCPToolBridge, bundled_server_parameters
from app.graph.tools import build_superhero_tools
from app.llm.embeddings import Embedder, GeminiEmbedder
from app.retrieval.corpus import CorpusPaths
from app.retrieval.rerank import FlashRankReranker, Reranker
from app.retrieval.store import HybridStore
from app.tools.circuit_breaker import CircuitBreaker
from app.tools.superhero import SuperheroClient

log = structlog.get_logger(__name__)

SUPERHERO_TOOL_NAMES = {"search_superheroes", "get_superhero"}


def build_http_client(settings: Settings) -> httpx.AsyncClient:
    """One client for all outbound HTTP.

    ``follow_redirects`` matters: the Superhero API answers every call with a 302 to its www host,
    and httpx does not follow redirects by default.
    """
    return httpx.AsyncClient(
        timeout=settings.superhero_timeout_s,
        follow_redirects=True,
        headers={"User-Agent": f"ai-engineer-assessment/{__version__}"},
    )


def build_superhero_client(settings: Settings, http: httpx.AsyncClient) -> SuperheroClient | None:
    """The superhero source, or ``None`` when no token is configured."""
    if settings.superhero_api_token is None:
        return None
    return SuperheroClient(
        http,
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


def build_embedder(settings: Settings) -> Embedder | None:
    """Gemini embeddings, or ``None`` without a Gemini key."""
    if settings.gemini_api_key is None:
        return None
    return GeminiEmbedder(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.gemini_embedding_model,
        dimensions=settings.embedding_dimensions,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )


async def build_reranker(settings: Settings) -> Reranker | None:
    """The cross-encoder, or ``None`` when disabled. Loads weights off the event loop."""
    if not settings.reranker_enabled:
        return None
    return await FlashRankReranker.create(
        model_name=settings.reranker_model,
        cache_dir=settings.reranker_cache_dir,
        max_length=settings.reranker_max_length,
    )


async def build_agent_tools(
    settings: Settings, superhero: SuperheroClient | None
) -> tuple[list[BaseTool], MCPToolBridge | None]:
    """The superhero tools for the agent, in-process or over MCP per ``tools_backend``.

    Returns the open bridge too when MCP is used, so the caller can close it on shutdown.
    """
    if settings.tools_backend == "mcp":
        target = settings.mcp_server_url or bundled_server_parameters()
        bridge = MCPToolBridge(target)
        await bridge.__aenter__()
        tools = await bridge.load_tools(SUPERHERO_TOOL_NAMES)
        log.info("agent_tools", backend="mcp", tools=[tool.name for tool in tools])
        return tools, bridge
    tools = build_superhero_tools(superhero) if superhero is not None else []
    log.info("agent_tools", backend="inprocess", tools=[tool.name for tool in tools])
    return tools, None


def load_store(
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
