"""The hybrid retrieval store: what the RAG branch of the graph calls.

``search`` runs two retrievers over the same indexed text, fuses their rankings with reciprocal
rank fusion, reranks the top fused candidates with a cross-encoder, and returns the best
passages with full provenance: which retriever found each one, at what rank, and its rerank
score. That provenance flows straight into the ``sources`` of every API response.

Degradation is explicit rather than silent:

* no vectors on disk, or vectors that do not match the corpus or model, means BM25-only
* an embedder that fails for one query means BM25-only for that query, with the reason recorded
* no reranker means fused order

The store raises only when the corpus itself cannot be loaded. Everything else degrades.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import structlog
from pydantic import BaseModel

from app.llm.embeddings import Embedder
from app.retrieval.corpus import (
    CorpusPaths,
    Paragraph,
    indexed_text,
    paragraphs_fingerprint,
    read_jsonl,
)
from app.retrieval.dense import DenseIndex, DenseIndexMismatchError
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.lexical import LexicalIndex
from app.retrieval.rerank import Reranker

log = structlog.get_logger(__name__)

RetrievalMode = Literal["hybrid", "bm25_only"]


class RetrievedParagraph(BaseModel):
    """One hit with everything needed to cite it and to explain why it was retrieved."""

    paragraph: Paragraph
    score: float
    found_by: list[str]
    bm25_rank: int | None
    dense_rank: int | None
    rerank_score: float | None


class SearchResult(BaseModel):
    """Hits plus how they were produced, so callers can report degraded modes honestly."""

    query: str
    mode: RetrievalMode
    reranked: bool
    fallback_reason: str | None
    hits: list[RetrievedParagraph]


class DocumentSummary(BaseModel):
    """One article in the corpus."""

    title: str
    display_title: str
    url: str
    paragraphs: int


class HybridStore:
    """Lexical plus dense retrieval, fused and reranked, over the committed corpus."""

    def __init__(
        self,
        paragraphs: Sequence[Paragraph],
        lexical: LexicalIndex,
        dense: DenseIndex | None,
        embedder: Embedder | None,
        reranker: Reranker | None,
        *,
        bm25_top_k: int = 100,
        dense_top_k: int = 100,
        rrf_k: int = 60,
        rerank_candidates: int = 100,
        rerank_top_k: int = 20,
    ) -> None:
        self._paragraphs = list(paragraphs)
        self._by_id = {p.id: p for p in self._paragraphs}
        self._lexical = lexical
        self._dense = dense
        self._embedder = embedder
        self._reranker = reranker
        self._bm25_top_k = bm25_top_k
        self._dense_top_k = dense_top_k
        self._rrf_k = rrf_k
        self._rerank_candidates = rerank_candidates
        self._rerank_top_k = rerank_top_k

    # ------------------------------------------------------------------ construction
    @classmethod
    def load(
        cls,
        paths: CorpusPaths,
        *,
        embedder: Embedder | None,
        reranker: Reranker | None,
        expected_embedding_model: str,
        expected_dimensions: int,
        bm25_top_k: int = 100,
        dense_top_k: int = 100,
        rrf_k: int = 60,
        rerank_candidates: int = 100,
        rerank_top_k: int = 20,
    ) -> HybridStore:
        """Load the corpus and indexes from disk. Dense is optional and verified."""
        paragraphs = read_jsonl(paths.paragraphs, Paragraph)
        lexical = LexicalIndex.load(paths.bm25_dir)
        if lexical.size != len(paragraphs):
            raise ValueError(
                f"BM25 index has {lexical.size} rows but paragraphs.jsonl has {len(paragraphs)}"
            )

        dense: DenseIndex | None = None
        if not paths.embeddings.exists() or not paths.embeddings_meta.exists():
            log.warning("dense_index_absent", note="running BM25-only; build embeddings to enable")
        elif embedder is None:
            log.warning("dense_index_unusable", note="vectors exist but no embedder is configured")
        else:
            try:
                dense = DenseIndex.load(
                    paths,
                    expected_fingerprint=paragraphs_fingerprint(paragraphs),
                    expected_count=len(paragraphs),
                    expected_model=expected_embedding_model,
                    expected_dimensions=expected_dimensions,
                )
            except DenseIndexMismatchError as exc:
                log.warning("dense_index_mismatch", reason=str(exc), note="running BM25-only")

        log.info(
            "retrieval_store_loaded",
            paragraphs=len(paragraphs),
            dense=dense is not None,
            reranker=reranker.model_id if reranker else None,
        )
        return cls(
            paragraphs,
            lexical,
            dense,
            embedder,
            reranker,
            bm25_top_k=bm25_top_k,
            dense_top_k=dense_top_k,
            rrf_k=rrf_k,
            rerank_candidates=rerank_candidates,
            rerank_top_k=rerank_top_k,
        )

    # ------------------------------------------------------------------ properties
    @property
    def size(self) -> int:
        """Number of paragraphs."""
        return len(self._paragraphs)

    @property
    def dense_enabled(self) -> bool:
        """Whether dense retrieval is available at all."""
        return self._dense is not None and self._embedder is not None

    @property
    def reranker_enabled(self) -> bool:
        """Whether a reranker is configured."""
        return self._reranker is not None

    # ------------------------------------------------------------------ queries
    async def search(
        self,
        query: str,
        *,
        final_k: int | None = None,
        bm25_top_k: int | None = None,
        dense_top_k: int | None = None,
        rerank_candidates: int | None = None,
    ) -> SearchResult:
        """Hybrid search. Keyword overrides exist for evaluation sweeps; the API uses defaults."""
        final_k = final_k or self._rerank_top_k
        bm25_hits = self._lexical.search(query, bm25_top_k or self._bm25_top_k)
        bm25_rows = [row for row, _ in bm25_hits]

        mode: RetrievalMode = "bm25_only"
        fallback_reason: str | None = None
        dense_rows: list[int] = []
        if self._dense is not None and self._embedder is not None:
            try:
                query_vector = await self._embedder.embed_query(query)
                dense_rows = [
                    row
                    for row, _ in self._dense.search(query_vector, dense_top_k or self._dense_top_k)
                ]
                mode = "hybrid"
            except Exception as exc:  # any embedder failure degrades this query, never the request
                fallback_reason = f"query embedding failed: {type(exc).__name__}: {exc}"
                log.warning("dense_search_failed", reason=fallback_reason)

        fused = reciprocal_rank_fusion([bm25_rows, dense_rows], k=self._rrf_k)
        candidates = fused[: rerank_candidates or self._rerank_candidates]

        rerank_scores: dict[int, float] = {}
        if self._reranker is not None and candidates:
            texts = [indexed_text(self._paragraphs[row]) for row, _ in candidates]
            ranked = await self._reranker.rerank(query, texts)
            rerank_scores = {candidates[index][0]: score for index, score in ranked}
            ordered = sorted(candidates, key=lambda item: -rerank_scores[item[0]])
        else:
            ordered = candidates

        bm25_position = {row: rank for rank, row in enumerate(bm25_rows, start=1)}
        dense_position = {row: rank for rank, row in enumerate(dense_rows, start=1)}
        hits = [
            RetrievedParagraph(
                paragraph=self._paragraphs[row],
                score=rerank_scores.get(row, fused_score),
                found_by=[
                    name
                    for name, positions in (("bm25", bm25_position), ("dense", dense_position))
                    if row in positions
                ],
                bm25_rank=bm25_position.get(row),
                dense_rank=dense_position.get(row),
                rerank_score=rerank_scores.get(row),
            )
            for row, fused_score in ordered[:final_k]
        ]
        return SearchResult(
            query=query,
            mode=mode,
            reranked=bool(rerank_scores),
            fallback_reason=fallback_reason,
            hits=hits,
        )

    def get_paragraph(self, paragraph_id: str) -> Paragraph | None:
        """Look up one paragraph by its stable id."""
        return self._by_id.get(paragraph_id)

    def list_documents(self) -> list[DocumentSummary]:
        """Every article with its paragraph count, in corpus order."""
        counts: dict[str, int] = {}
        urls: dict[str, str] = {}
        for paragraph in self._paragraphs:
            counts[paragraph.title] = counts.get(paragraph.title, 0) + 1
            urls.setdefault(paragraph.title, paragraph.url)
        return [
            DocumentSummary(
                title=title,
                display_title=title.replace("_", " "),
                url=urls[title],
                paragraphs=count,
            )
            for title, count in counts.items()
        ]
