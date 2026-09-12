import json
from pathlib import Path

from app.graph.mcp_tools import MCPToolBridge, bundled_server_parameters
from app.retrieval.store import VectorStore
from mcp_server.server import create_server
from tests.fakes import FakeEmbedder
from tests.fakes import build_test_corpus as build_corpus
from tests.fakes import load_test_store as load


async def test_bridge_exposes_server_tools_as_langchain_tools_over_one_session(
    tmp_path: Path,
) -> None:
    embedder = FakeEmbedder(16)
    store: VectorStore = load(await build_corpus(tmp_path, embedder=embedder), embedder=embedder)
    server = create_server(superhero=None, store=store)

    async with MCPToolBridge(server) as bridge:
        tools = await bridge.load_tools({"search_documents", "list_documents"})
        by_name = {tool.name: tool for tool in tools}

        catalog = json.loads(await by_name["list_documents"].ainvoke({}))
        hits = json.loads(
            await by_name["search_documents"].ainvoke({"query": "Denver Broncos", "k": 1})
        )
        unconfigured = json.loads(
            await (await bridge.load_tools({"search_superheroes"}))[0].ainvoke({"name": "batman"})
        )

    assert set(by_name) == {"search_documents", "list_documents"}
    assert by_name["search_documents"].description.startswith("Retrieve the most relevant")
    assert catalog["count"] == 3
    assert hits["results"][0]["paragraph_id"] == "super-bowl-50-000"
    assert "SUPERHERO_API_TOKEN" in unconfigured["hint"], "ToolError flows back as JSON data"


def test_bundled_server_parameters_point_at_this_repository() -> None:
    params = bundled_server_parameters()

    assert params.args == ["-m", "mcp_server.server"]
    assert params.cwd is not None
    assert (Path(params.cwd) / "mcp_server" / "server.py").exists()
