"""MCP server exposing the same four tools the API uses internally.

Run standalone from the repository root::

    python -m mcp_server.server                                   # stdio, for local MCP hosts
    python -m mcp_server.server --transport streamable-http       # http://127.0.0.1:3001/mcp
    npx @modelcontextprotocol/inspector python -m mcp_server.server

``create_server`` takes already-built dependencies, which is how tests drive it in-process with
fakes. ``build_server_from_settings`` wires real dependencies through ``app.resources``, the same
code the API uses, so the two entry points cannot disagree about configuration.

Every tool is annotated read-only and idempotent. The corpus tools are closed-world (a fixed
dataset); the superhero tools are open-world (a live external API). Failures come back as
``ToolError`` data rather than protocol errors, so a calling agent can recover.

Logging goes to stderr: on the stdio transport, stdout is the protocol channel.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Annotated

import structlog
from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from app import __version__
from app.config import Settings, get_settings
from app.observability.logging import configure_logging
from app.resources import (
    build_embedder,
    build_http_client,
    build_reranker,
    build_superhero_client,
    load_store,
)
from app.retrieval.store import VectorStore
from app.tools.common import ToolError
from app.tools.documents import (
    DocumentCatalog,
    DocumentSearchResult,
    list_documents,
    search_documents,
)
from app.tools.superhero import (
    Hero,
    HeroSearchResult,
    SuperheroClient,
    get_superhero,
    search_superheroes,
)

log = structlog.get_logger(__name__)

SERVER_NAME = "superhero-knowledge"

_INSTRUCTIONS = (
    "Two knowledge sources. (1) A corpus of English Wikipedia paragraphs from the SQuAD dev "
    "set, covering 20 articles such as Super Bowl 50, Nikola Tesla, Oxygen and the Amazon "
    "rainforest: call list_documents to see the topics and search_documents to retrieve "
    "passages. (2) The Superhero API: call search_superheroes with a character name; several "
    "characters may share a name, so use get_superhero with an id for one character's full "
    "record. All tools are read-only."
)

_CLOSED_WORLD = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)
_OPEN_WORLD = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=True)


def create_server(
    *, superhero: SuperheroClient | None, store: VectorStore | None, version: str = __version__
) -> MCPServer:
    """Register the four tools over the given dependencies."""
    server = MCPServer(name=SERVER_NAME, version=version, instructions=_INSTRUCTIONS)

    @server.tool(name="search_superheroes", annotations=_OPEN_WORLD)
    async def search_superheroes_tool(
        name: Annotated[str, Field(min_length=1, description="Character name or alias.")],
        limit: Annotated[int, Field(ge=1, le=20, description="Maximum matches to return.")] = 5,
    ) -> HeroSearchResult | ToolError:
        """Find superhero characters by name.

        Returns id, name, full name, publisher, alignment and power stats for each match, and a
        note when several characters share the name or when none was found.
        """
        if superhero is None:
            return _superhero_unconfigured()
        return await search_superheroes(superhero, name, limit)

    @server.tool(name="get_superhero", annotations=_OPEN_WORLD)
    async def get_superhero_tool(
        hero_id: Annotated[str, Field(min_length=1, description="Id from search_superheroes.")],
    ) -> Hero | ToolError:
        """Full record for one character: biography, appearance, work, connections, image."""
        if superhero is None:
            return _superhero_unconfigured()
        return await get_superhero(superhero, hero_id)

    @server.tool(name="search_documents", annotations=_CLOSED_WORLD)
    async def search_documents_tool(
        query: Annotated[str, Field(min_length=1, description="A self-contained question.")],
        k: Annotated[int, Field(ge=1, le=20, description="Passages to return.")] = 5,
    ) -> DocumentSearchResult | ToolError:
        """Retrieve the most relevant corpus paragraphs for a question.

        Semantic search over the corpus vectors, reranked by a cross-encoder. Each hit
        carries the paragraph text, its article title and URL, and where it ranked.
        """
        if store is None:
            return _store_unavailable()
        return await search_documents(store, query, k)

    @server.tool(name="list_documents", annotations=_CLOSED_WORLD)
    async def list_documents_tool() -> DocumentCatalog | ToolError:
        """List every article in the corpus with its paragraph count and URL."""
        if store is None:
            return _store_unavailable()
        return await list_documents(store)

    return server


def _superhero_unconfigured() -> ToolError:
    return ToolError(
        error="The superhero source is not configured on this server.",
        hint="Set SUPERHERO_API_TOKEN and restart.",
    )


def _store_unavailable() -> ToolError:
    return ToolError(
        error="The document corpus is not loaded on this server.",
        hint="Run `python -m scripts.build_dataset` to create it.",
    )


async def build_server_from_settings(settings: Settings) -> MCPServer:
    """Wire real dependencies exactly as the API does."""
    http = build_http_client(settings)
    superhero = build_superhero_client(settings, http)
    embedder = build_embedder(settings)
    reranker = await build_reranker(settings)
    store = load_store(settings, embedder, reranker)
    log.info(
        "mcp_server_ready",
        superhero=superhero is not None,
        corpus=store is not None,
        searchable=bool(store and store.searchable),
    )
    return create_server(superhero=superhero, store=store)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3001)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Blocks until the host disconnects or the process is stopped."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    settings = get_settings()
    configure_logging(settings.log_level, as_json=False, stream=sys.stderr)
    server = asyncio.run(build_server_from_settings(settings))
    if args.transport == "stdio":
        server.run()
    else:
        server.run(transport="streamable-http", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
