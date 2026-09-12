"""The vector store: what the retrieval branch of the graph calls.

``search`` embeds the query with the same model that embedded the corpus, takes the nearest
paragraphs by cosine similarity, reranks the top candidates with a cross-encoder, and returns
them with provenance: the rank the vector search gave each passage and the score the reranker
gave it. That provenance flows straight into the ``sources`` of every API response.

Retrieval is semantic only. There is no keyword index to fall back on, so a query that cannot
be embedded cannot be answered from the corpus: ``search`` then returns no hits and says why,
rather than returning something weaker while pretending it searched. The store raises only when
the corpus itself cannot be read; everything else degrades and is reported.
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
from app.retrieval.rerank import Reranker

log = structlog.get_logger(__name__)

RetrievalMode = Literal["dense_reranked", "dense", "unavailable"]


class RetrievedParagraph(BaseModel):
    """One hit with everything needed to cite it and to explain why it was retrieved."""

    paragraph: Paragraph
    score: float
    dense_rank: int
    dense_score: float
    rerank_score: float | None


class SearchResult(BaseModel):
    """Hits plus how they were produced, so callers can report degraded modes honestly."""

    query: str
    mode: RetrievalMode
    reranked: bool
    failure_reason: str | None
    hits: list[RetrievedParagraph]


class DocumentSummary(BaseModel):
    """One article in the corpus."""

    title: str
    display_title: str
    url: str
    paragraphs: int


class VectorStore:
    """Dense retrieval over the committed corpus, reranked by a cross-encoder."""

    def __init__(
        self,
        paragraphs: Sequence[Paragraph],
        dense: DenseIndex | None,
        embedder: Embedder | None,
        reranker: Reranker | None,
        *,
        dense_top_k: int = 100,
        rerank_candidates: int = 50,
        rerank_top_k: int = 20,
    ) -> None:
        self._paragraphs = list(paragraphs)
        self._by_id = {p.id: p for p in self._paragraphs}
        self._dense = dense
        self._embedder = embedder
        self._reranker = reranker
        self._dense_top_k = dense_top_k
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
        dense_top_k: int = 100,
        rerank_candidates: int = 50,
        rerank_top_k: int = 20,
    ) -> VectorStore:
        """Load the corpus and its vectors from disk. The vectors are verified against both.

        A missing or mismatched index leaves the store searchable-in-name-only: it can still
        resolve paragraphs by id and list articles, and every search reports ``unavailable``.
        Readiness reports it too, so this never passes silently in production.
        """
        paragraphs = read_jsonl(paths.paragraphs, Paragraph)

        dense: DenseIndex | None = None
        if not paths.embeddings.exists() or not paths.embeddings_meta.exists():
            log.warning(
                "dense_index_absent",
                note="corpus search is unavailable; run scripts.build_dataset to embed",
            )
        elif embedder is None:
            log.warning(
                "embedder_absent", note="vectors exist but no key is configured to embed queries"
            )
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
                log.warning("dense_index_mismatch", reason=str(exc))

        log.info(
            "retrieval_store_loaded",
            paragraphs=len(paragraphs),
            searchable=dense is not None and embedder is not None,
            reranker=reranker.model_id if reranker else None,
        )
        return cls(
            paragraphs,
            dense,
            embedder,
            reranker,
            dense_top_k=dense_top_k,
            rerank_candidates=rerank_candidates,
            rerank_top_k=rerank_top_k,
        )

    # ------------------------------------------------------------------ properties
    @property
    def searchable(self) -> bool:
        """Whether the corpus can actually be searched: vectors on disk and a query embedder."""
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
        dense_top_k: int | None = None,
        rerank_candidates: int | None = None,
    ) -> SearchResult:
        """Embed, retrieve, rerank. Keyword overrides exist for evaluation sweeps only."""
        final_k = final_k or self._rerank_top_k
        ranked, failure = await self._dense_search(query, dense_top_k or self._dense_top_k)
        if failure is not None:
            return SearchResult(
                query=query, mode="unavailable", reranked=False, failure_reason=failure, hits=[]
            )

        dense_scores = dict(ranked)
        dense_position = {row: rank for rank, (row, _) in enumerate(ranked, start=1)}
        candidates = [row for row, _ in ranked[: rerank_candidates or self._rerank_candidates]]

        rerank_scores: dict[int, float] = {}
        if self._reranker is not None and candidates:
            texts = [indexed_text(self._paragraphs[row]) for row in candidates]
            reranked = await self._reranker.rerank(query, texts)
            rerank_scores = {candidates[index]: score for index, score in reranked}
            ordered = sorted(candidates, key=lambda row: -rerank_scores[row])
        else:
            ordered = candidates

        hits = [
            RetrievedParagraph(
                paragraph=self._paragraphs[row],
                score=rerank_scores.get(row, dense_scores[row]),
                dense_rank=dense_position[row],
                dense_score=dense_scores[row],
                rerank_score=rerank_scores.get(row),
            )
            for row in ordered[:final_k]
        ]
        return SearchResult(
            query=query,
            mode="dense_reranked" if rerank_scores else "dense",
            reranked=bool(rerank_scores),
            failure_reason=None,
            hits=hits,
        )

    async def dense_ranking(self, query: str, *, k: int) -> list[str]:
        """Paragraph ids in vector-search order, before reranking, for evaluation.

        This is the first stage on its own, which is what the recall curve is measured on.
        """
        ranked, failure = await self._dense_search(query, k)
        if failure is not None:
            raise RuntimeError(failure)
        return [self._paragraphs[row].id for row, _ in ranked]

    async def _dense_search(self, query: str, k: int) -> tuple[list[tuple[int, float]], str | None]:
        """Rows and scores nearest the query, or a reason the corpus could not be searched."""
        if self._dense is None or self._embedder is None:
            return [], "the corpus has no usable vector index"
        try:
            query_vector = await self._embedder.embed_query(query)
        except Exception as exc:  # a failed embedding degrades this query, never the request
            reason = f"query embedding failed: {type(exc).__name__}: {exc}"
            log.warning("dense_search_failed", reason=reason)
            return [], reason
        return list(self._dense.search(query_vector, k)), None

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
