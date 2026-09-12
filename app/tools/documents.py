"""Tool functions over the text corpus.

``search_documents`` returns a compact, model-facing view of a corpus search: the passage text,
where it came from, and the retrieval provenance. ``list_documents`` returns the catalogue of
articles so a model can tell whether a topic is even covered before searching for it.
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel

from app.retrieval.store import DocumentSummary, RetrievalMode, VectorStore
from app.tools.common import ToolError

log = structlog.get_logger(__name__)


class PassageHit(BaseModel):
    """One retrieved paragraph with citation fields and provenance."""

    paragraph_id: str
    title: str
    url: str
    text: str
    score: float
    dense_rank: int
    rerank_score: float | None


class DocumentSearchResult(BaseModel):
    """Model-facing search outcome, including how retrieval ran."""

    query: str
    mode: RetrievalMode
    reranked: bool
    failure_reason: str | None
    results: list[PassageHit]


class DocumentCatalog(BaseModel):
    """Every article the corpus covers."""

    count: int
    documents: list[DocumentSummary]


async def search_documents(
    store: VectorStore, query: str, k: int = 5
) -> DocumentSearchResult | ToolError:
    """Search the corpus. Never raises: retrieval problems come back as data."""
    try:
        result = await store.search(query, final_k=k)
    except Exception as exc:  # the store degrades internally; this guards against bugs
        log.exception("search_documents_failed")
        return ToolError(error=f"Document search failed: {type(exc).__name__}", hint=None)
    return DocumentSearchResult(
        query=result.query,
        mode=result.mode,
        reranked=result.reranked,
        failure_reason=result.failure_reason,
        results=[
            PassageHit(
                paragraph_id=hit.paragraph.id,
                title=hit.paragraph.display_title,
                url=hit.paragraph.url,
                text=hit.paragraph.text,
                score=hit.score,
                dense_rank=hit.dense_rank,
                rerank_score=hit.rerank_score,
            )
            for hit in result.hits
        ],
    )


async def list_documents(store: VectorStore) -> DocumentCatalog:
    """The articles in the corpus with paragraph counts."""
    documents = store.list_documents()
    return DocumentCatalog(count=len(documents), documents=documents)
