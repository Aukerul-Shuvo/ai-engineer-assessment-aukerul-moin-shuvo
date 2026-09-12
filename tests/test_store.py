"""The vector store: search modes, provenance, reranking, and honest failure."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from tests.fakes import (
    TEST_DIMENSIONS,
    FakeEmbedder,
    build_test_corpus,
)
from tests.fakes import (
    load_test_store as load,
)


class KeywordReranker:
    """Scores a passage 1.0 if it contains the keyword, else a small decreasing value."""

    model_id = "fake-keyword-reranker"

    def __init__(self, keyword: str) -> None:
        self.keyword = keyword
        self.calls: list[int] = []

    async def rerank(self, query: str, passages: Sequence[str]) -> list[tuple[int, float]]:
        self.calls.append(len(passages))
        scored = [
            (i, 1.0 if self.keyword in text else 0.5 - i * 0.01) for i, text in enumerate(passages)
        ]
        return sorted(scored, key=lambda item: -item[1])


class FailingEmbedder(FakeEmbedder):
    async def embed_query(self, text: str) -> np.ndarray:
        raise ConnectionError("embedding service down")


# ---------------------------------------------------------------- unsearchable
async def test_corpus_without_vectors_is_readable_but_not_searchable(tmp_path: Path) -> None:
    paths = await build_test_corpus(tmp_path, embedder=None)
    store = load(paths, embedder=FakeEmbedder(TEST_DIMENSIONS))

    assert not store.searchable
    assert store.get_paragraph("super-bowl-50-000") is not None

    result = await store.search("Denver Broncos", final_k=2)

    assert result.mode == "unavailable"
    assert result.hits == []
    assert result.failure_reason is not None
    assert "vector index" in result.failure_reason


async def test_vectors_without_an_embedder_are_not_searchable(tmp_path: Path) -> None:
    paths = await build_test_corpus(tmp_path, embedder=FakeEmbedder(TEST_DIMENSIONS))
    store = load(paths, embedder=None)

    assert not store.searchable
    assert (await store.search("Denver Broncos")).mode == "unavailable"


async def test_a_failed_query_embedding_degrades_that_query_only(tmp_path: Path) -> None:
    paths = await build_test_corpus(tmp_path, embedder=FakeEmbedder(TEST_DIMENSIONS))
    store = load(paths, embedder=FailingEmbedder(TEST_DIMENSIONS))

    result = await store.search("Denver Broncos", final_k=2)

    assert result.mode == "unavailable"
    assert result.hits == []
    assert result.failure_reason is not None
    assert "embedding service down" in result.failure_reason


async def test_an_index_built_for_a_different_corpus_is_refused(tmp_path: Path) -> None:
    embedder = FakeEmbedder(TEST_DIMENSIONS)
    paths = await build_test_corpus(tmp_path, embedder=embedder)
    # Same shape, different content: the fingerprint no longer matches the paragraphs.
    vectors = np.load(paths.embeddings)
    paragraphs = paths.paragraphs.read_text(encoding="utf-8").replace("Denver", "Chicago")
    paths.paragraphs.write_text(paragraphs, encoding="utf-8")
    np.save(paths.embeddings, vectors)

    store = load(paths, embedder=embedder)

    assert not store.searchable, "a mismatched index is never used"


# ---------------------------------------------------------------- searching
async def test_search_returns_dense_provenance_without_a_reranker(tmp_path: Path) -> None:
    embedder = FakeEmbedder(TEST_DIMENSIONS)
    paths = await build_test_corpus(tmp_path, embedder=embedder)
    store = load(paths, embedder=embedder)

    assert store.searchable
    assert not store.reranker_enabled
    result = await store.search("Denver Broncos Super Bowl", final_k=3)

    assert result.mode == "dense"
    assert result.reranked is False
    assert result.failure_reason is None
    assert len(result.hits) == 3
    assert [hit.dense_rank for hit in result.hits] == [1, 2, 3], "dense order is preserved"
    assert all(hit.rerank_score is None for hit in result.hits)
    assert all(hit.score == hit.dense_score for hit in result.hits)
    top = result.hits[0]
    assert top.paragraph.title == "Super_Bowl_50"
    assert top.dense_score > 0


async def test_the_reranker_reorders_and_records_its_score(tmp_path: Path) -> None:
    embedder = FakeEmbedder(TEST_DIMENSIONS)
    paths = await build_test_corpus(tmp_path, embedder=embedder)
    reranker = KeywordReranker("Wardenclyffe")
    store = load(paths, embedder=embedder, reranker=reranker)

    result = await store.search("Super Bowl", final_k=4)

    assert result.mode == "dense_reranked"
    assert result.reranked is True
    assert result.hits[0].paragraph.id == "nikola-tesla-000", "the reranker's pick wins"
    assert result.hits[0].rerank_score == 1.0
    assert result.hits[0].score == 1.0, "the reported score is the reranker's"
    assert result.hits[0].dense_rank > 1, "which the vector search had ranked lower"


async def test_rerank_candidates_bounds_the_cross_encoder_input(tmp_path: Path) -> None:
    embedder = FakeEmbedder(TEST_DIMENSIONS)
    paths = await build_test_corpus(tmp_path, embedder=embedder)
    reranker = KeywordReranker("Broncos")
    store = load(paths, embedder=embedder, reranker=reranker, rerank_candidates=2)

    await store.search("Super Bowl", final_k=4)

    assert reranker.calls == [2], "only the top two candidates reach the cross-encoder"


async def test_dense_ranking_lists_ids_in_vector_order(tmp_path: Path) -> None:
    embedder = FakeEmbedder(TEST_DIMENSIONS)
    paths = await build_test_corpus(tmp_path, embedder=embedder)
    store = load(paths, embedder=embedder)

    ranking = await store.dense_ranking("Denver Broncos Super Bowl", k=4)

    assert len(ranking) == 4
    assert len(set(ranking)) == 4
    result = await store.search("Denver Broncos Super Bowl", final_k=4)
    assert ranking == [hit.paragraph.id for hit in result.hits], "same order, no reranker"


async def test_dense_ranking_raises_when_the_corpus_cannot_be_searched(tmp_path: Path) -> None:
    paths = await build_test_corpus(tmp_path, embedder=None)
    store = load(paths, embedder=FakeEmbedder(TEST_DIMENSIONS))

    with pytest.raises(RuntimeError, match="vector index"):
        await store.dense_ranking("Denver Broncos", k=2)


# ---------------------------------------------------------------- catalogue
async def test_list_documents_counts_paragraphs_per_article(tmp_path: Path) -> None:
    embedder = FakeEmbedder(TEST_DIMENSIONS)
    paths = await build_test_corpus(tmp_path, embedder=embedder)
    store = load(paths, embedder=embedder)

    documents = store.list_documents()

    assert [(doc.title, doc.paragraphs) for doc in documents] == [
        ("Super_Bowl_50", 2),
        ("Nikola_Tesla", 1),
        ("Oxygen", 1),
    ]
    assert documents[0].display_title == "Super Bowl 50"
    assert documents[0].url == "https://en.wikipedia.org/wiki/Super_Bowl_50"


async def test_get_paragraph_by_id(tmp_path: Path) -> None:
    embedder = FakeEmbedder(TEST_DIMENSIONS)
    paths = await build_test_corpus(tmp_path, embedder=embedder)
    store = load(paths, embedder=embedder)

    assert store.get_paragraph("oxygen-000") is not None
    assert store.get_paragraph("no-such-paragraph") is None
