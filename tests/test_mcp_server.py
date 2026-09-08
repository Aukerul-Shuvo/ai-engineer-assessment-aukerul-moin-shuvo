from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import respx
from mcp import Client

from app.config import Settings
from app.retrieval.store import HybridStore
from app.tools.circuit_breaker import CircuitBreaker
from app.tools.superhero import SuperheroClient
from mcp_server.server import SERVER_NAME, build_server_from_settings, create_server
from tests.fakes import FakeEmbedder
from tests.test_store import build_corpus, load
from tests.test_superhero import BASE, BATMEN, TOKEN

EXPECTED_TOOLS = {"search_superheroes", "get_superhero", "search_documents", "list_documents"}


@pytest.fixture
async def store(tmp_path: Path) -> HybridStore:
    embedder = FakeEmbedder(16)
    return load(await build_corpus(tmp_path, embedder=embedder), embedder=embedder)


@pytest.fixture
async def superhero() -> AsyncIterator[SuperheroClient]:
    async with httpx.AsyncClient(follow_redirects=True, timeout=5) as http:
        yield SuperheroClient(
            http,
            base_url=BASE,
            token=TOKEN,
            max_retries=0,
            retry_wait_max_s=0,
            breaker=CircuitBreaker(),
        )


async def test_exposes_four_read_only_tools_with_typed_schemas(store: HybridStore) -> None:
    server = create_server(superhero=None, store=store)

    async with Client(server) as client:
        assert client.server_info is not None
        assert client.server_info.name == SERVER_NAME
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    assert set(tools) == EXPECTED_TOOLS
    assert all(tool.description for tool in tools.values())

    search = tools["search_superheroes"]
    assert search.input_schema["required"] == ["name"]
    assert search.input_schema["properties"]["limit"]["minimum"] == 1
    assert search.input_schema["properties"]["limit"]["maximum"] == 20
    assert search.annotations is not None
    assert search.annotations.read_only_hint is True
    assert search.annotations.open_world_hint is True, "external API"

    docs = tools["search_documents"]
    assert docs.annotations is not None
    assert docs.annotations.open_world_hint is False, "fixed corpus"
    assert docs.output_schema is not None, "union return types still produce a typed schema"


async def test_document_tools_return_structured_results(store: HybridStore) -> None:
    server = create_server(superhero=None, store=store)

    async with Client(server) as client:
        catalog = await client.call_tool("list_documents", {})
        search = await client.call_tool("search_documents", {"query": "Denver Broncos", "k": 2})

    assert catalog.is_error is False
    assert catalog.structured_content is not None
    assert catalog.structured_content["result"]["count"] == 3

    assert search.is_error is False
    assert search.structured_content is not None
    result = search.structured_content["result"]
    assert result["mode"] == "hybrid"
    assert len(result["results"]) == 2
    assert result["results"][0]["paragraph_id"] == "super-bowl-50-000"


@respx.mock
async def test_superhero_tools_call_the_api_through_the_same_client(
    store: HybridStore, superhero: SuperheroClient
) -> None:
    respx.get(f"{BASE}/{TOKEN}/search/batman").mock(
        return_value=httpx.Response(200, json={"response": "success", "results": BATMEN})
    )
    respx.get(f"{BASE}/{TOKEN}/70").mock(
        return_value=httpx.Response(200, json={"response": "success", **BATMEN[1]})
    )
    server = create_server(superhero=superhero, store=store)

    async with Client(server) as client:
        found = await client.call_tool("search_superheroes", {"name": "batman", "limit": 2})
        hero = await client.call_tool("get_superhero", {"hero_id": "70"})

    assert found.structured_content is not None
    assert found.structured_content["result"]["count"] == 3
    assert len(found.structured_content["result"]["results"]) == 2
    assert hero.structured_content is not None
    assert hero.structured_content["result"]["full_name"] == "Bruce Wayne"
    assert hero.structured_content["result"]["powerstats"]["combat"] is None


async def test_unconfigured_sources_answer_with_tool_errors_not_protocol_errors() -> None:
    server = create_server(superhero=None, store=None)

    async with Client(server) as client:
        heroes = await client.call_tool("search_superheroes", {"name": "batman"})
        docs = await client.call_tool("list_documents", {})

    assert heroes.is_error is False
    assert heroes.structured_content is not None
    assert "SUPERHERO_API_TOKEN" in heroes.structured_content["result"]["hint"]
    assert docs.is_error is False
    assert docs.structured_content is not None
    assert "build_dataset" in docs.structured_content["result"]["hint"]


async def test_invalid_arguments_are_rejected_before_the_tool_runs(store: HybridStore) -> None:
    server = create_server(superhero=None, store=store)

    async with Client(server) as client:
        result = await client.call_tool("search_documents", {"query": "x", "k": 0})

    assert result.is_error is True


async def test_build_from_settings_wires_the_real_corpus(test_settings: Settings) -> None:
    server = await build_server_from_settings(test_settings)

    async with Client(server) as client:
        tools = {tool.name for tool in (await client.list_tools()).tools}
        catalog = await client.call_tool("list_documents", {})

    assert tools == EXPECTED_TOOLS
    assert catalog.structured_content is not None
    assert catalog.structured_content["result"]["count"] == 48, "the committed SQuAD corpus"
