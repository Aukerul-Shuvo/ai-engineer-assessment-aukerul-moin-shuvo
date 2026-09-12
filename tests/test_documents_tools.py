from pathlib import Path

from app.retrieval.store import VectorStore
from app.tools.common import ToolError
from app.tools.documents import (
    DocumentCatalog,
    DocumentSearchResult,
    list_documents,
    search_documents,
)
from tests.fakes import FakeEmbedder
from tests.fakes import build_test_corpus as build_corpus
from tests.fakes import load_test_store as load


async def test_search_documents_returns_compact_hits_with_provenance(tmp_path: Path) -> None:
    embedder = FakeEmbedder(16)
    store = load(await build_corpus(tmp_path, embedder=embedder), embedder=embedder)

    result = await search_documents(store, "Denver Broncos", k=2)

    assert isinstance(result, DocumentSearchResult)
    assert result.mode == "dense"
    assert len(result.results) == 2
    hit = result.results[0]
    assert hit.paragraph_id == "super-bowl-50-000"
    assert hit.title == "Super Bowl 50"
    assert hit.url.endswith("/Super_Bowl_50")
    assert "Denver Broncos" in hit.text
    assert hit.dense_rank == 1
    assert hit.rerank_score is None, "no reranker in this store"


async def test_list_documents_returns_the_catalogue(tmp_path: Path) -> None:
    # The catalogue comes from the paragraphs, so it works even with no vectors to search.
    store = load(await build_corpus(tmp_path, embedder=None), embedder=None)

    catalog = await list_documents(store)

    assert isinstance(catalog, DocumentCatalog)
    assert catalog.count == 3
    assert [d.display_title for d in catalog.documents] == [
        "Super Bowl 50",
        "Nikola Tesla",
        "Oxygen",
    ]


async def test_search_documents_turns_unexpected_failures_into_tool_errors(tmp_path: Path) -> None:
    class BrokenStore(VectorStore):
        async def search(self, query: str, **kwargs: object) -> None:  # type: ignore[override]
            raise RuntimeError("index corrupted")

    store = load(await build_corpus(tmp_path, embedder=None), embedder=None)
    store.__class__ = BrokenStore

    result = await search_documents(store, "anything")

    assert isinstance(result, ToolError)
    assert "RuntimeError" in result.error
