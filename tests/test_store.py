import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from app.retrieval.build import run_build
from app.retrieval.corpus import CorpusPaths
from app.retrieval.store import HybridStore
from tests.fakes import FakeEmbedder

CORPUS = {
    "version": "1.1",
    "data": [
        {
            "title": "Super_Bowl_50",
            "paragraphs": [
                {
                    "context": (
                        "The AFC champion Denver Broncos defeated the NFC champion Carolina "
                        "Panthers 24-10 to earn their third Super Bowl title."
                    ),
                    "qas": [],
                },
                {
                    "context": (
                        "The Panthers finished the regular season with a 15-1 record and "
                        "quarterback Cam Newton was named MVP."
                    ),
                    "qas": [],
                },
            ],
        },
        {
            "title": "Nikola_Tesla",
            "paragraphs": [
                {
                    "context": (
                        "Tesla later approached Morgan to ask for more funds to build a more "
                        "powerful transmitter at Wardenclyffe."
                    ),
                    "qas": [],
                }
            ],
        },
        {
            "title": "Oxygen",
            "paragraphs": [
                {
                    "context": "Oxygen is a chemical element with symbol O and atomic number 8.",
                    "qas": [],
                }
            ],
        },
    ],
}


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


async def build_corpus(tmp_path: Path, embedder: FakeEmbedder | None) -> CorpusPaths:
    source = tmp_path / "dev.json"
    source.write_text(json.dumps(CORPUS), encoding="utf-8")
    paths = CorpusPaths(tmp_path / "data")
    await run_build(source=str(source), paths=paths, embedder=embedder)
    return paths


def load(
    paths: CorpusPaths, embedder: FakeEmbedder | None, reranker: object | None = None, **kw: int
) -> HybridStore:
    return HybridStore.load(
        paths,
        embedder=embedder,
        reranker=reranker,  # type: ignore[arg-type]
        expected_embedding_model="fake-embedder",
        expected_dimensions=16,
        **kw,
    )


# ---------------------------------------------------------------- modes
async def test_bm25_only_when_no_embeddings_were_built(tmp_path: Path) -> None:
    paths = await build_corpus(tmp_path, embedder=None)
    store = load(paths, embedder=FakeEmbedder(16))

    assert store.size == 4
    assert not store.dense_enabled
    result = await store.search("Denver Broncos", final_k=2)

    assert result.mode == "bm25_only"
    assert result.reranked is False
    assert result.fallback_reason is None
    assert result.hits[0].paragraph.id == "super-bowl-50-000"
    assert result.hits[0].found_by == ["bm25"]
    assert result.hits[0].dense_rank is None


async def test_hybrid_mode_reports_both_retrievers(tmp_path: Path) -> None:
    embedder = FakeEmbedder(16)
    paths = await build_corpus(tmp_path, embedder=embedder)
    store = load(paths, embedder=embedder)

    assert store.dense_enabled
    result = await store.search("Denver Broncos Super Bowl", final_k=3)

    assert result.mode == "hybrid"
    top = result.hits[0]
    assert top.paragraph.id == "super-bowl-50-000"
    assert top.found_by == ["bm25", "dense"]
    assert top.bm25_rank == 1
    assert top.dense_rank is not None


async def test_dense_is_disabled_when_the_vectors_do_not_match_the_corpus(tmp_path: Path) -> None:
    embedder = FakeEmbedder(16)
    paths = await build_corpus(tmp_path, embedder=embedder)
    # Tamper with one paragraph so the fingerprint no longer matches the vectors.
    lines = paths.paragraphs.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("Denver", "Denvor")
    paths.paragraphs.write_text("\n".join(lines) + "\n", encoding="utf-8")

    store = load(paths, embedder=embedder)

    assert not store.dense_enabled, "a stale vector file must never be used"


async def test_dense_is_disabled_when_the_configured_model_differs(tmp_path: Path) -> None:
    embedder = FakeEmbedder(16)
    paths = await build_corpus(tmp_path, embedder=embedder)

    store = HybridStore.load(
        paths,
        embedder=embedder,
        reranker=None,
        expected_embedding_model="some-other-model",
        expected_dimensions=16,
    )

    assert not store.dense_enabled


async def test_embedding_failure_degrades_that_query_only(tmp_path: Path) -> None:
    paths = await build_corpus(tmp_path, embedder=FakeEmbedder(16))
    store = load(paths, embedder=FailingEmbedder(16))
    assert store.dense_enabled, "the index loaded fine; only the query embedding will fail"

    result = await store.search("Denver Broncos", final_k=2)

    assert result.mode == "bm25_only"
    assert result.fallback_reason is not None
    assert "embedding service down" in result.fallback_reason
    assert result.hits, "BM25 still answers"


# ---------------------------------------------------------------- reranking
async def test_reranker_reorders_candidates_and_records_scores(tmp_path: Path) -> None:
    embedder = FakeEmbedder(16)
    paths = await build_corpus(tmp_path, embedder=embedder)
    reranker = KeywordReranker(keyword="Wardenclyffe")
    store = load(paths, embedder=embedder, reranker=reranker)

    result = await store.search("Super Bowl champion", final_k=2)

    assert result.reranked is True
    assert result.hits[0].paragraph.id == "nikola-tesla-000", "the reranker decides the order"
    assert result.hits[0].rerank_score == 1.0
    assert result.hits[0].score == 1.0, "score is the rerank score when reranked"
    assert all(hit.rerank_score is not None for hit in result.hits)


async def test_rerank_candidates_bounds_the_cross_encoder_input(tmp_path: Path) -> None:
    embedder = FakeEmbedder(16)
    paths = await build_corpus(tmp_path, embedder=embedder)
    reranker = KeywordReranker(keyword="nothing")
    store = load(paths, embedder=embedder, reranker=reranker, rerank_candidates=2)

    await store.search("Super Bowl Tesla Oxygen", final_k=4)

    assert reranker.calls == [2], "only the top fused candidates reach the cross-encoder"


# ---------------------------------------------------------------- catalogue
async def test_list_documents_and_get_paragraph(tmp_path: Path) -> None:
    paths = await build_corpus(tmp_path, embedder=None)
    store = load(paths, embedder=None)

    documents = store.list_documents()

    assert [d.title for d in documents] == ["Super_Bowl_50", "Nikola_Tesla", "Oxygen"]
    assert documents[0].paragraphs == 2
    assert documents[0].display_title == "Super Bowl 50"
    assert documents[0].url == "https://en.wikipedia.org/wiki/Super_Bowl_50"
    assert store.get_paragraph("oxygen-000") is not None
    assert store.get_paragraph("nope") is None


async def test_load_refuses_a_bm25_index_that_does_not_match_the_paragraphs(tmp_path: Path) -> None:
    paths = await build_corpus(tmp_path, embedder=None)
    lines = paths.paragraphs.read_text(encoding="utf-8").splitlines()
    paths.paragraphs.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="BM25 index has 4 rows"):
        load(paths, embedder=None)
