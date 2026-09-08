"""retrieve_documents: the RAG branch for one dataset sub-query.

Pipeline, not agent: hybrid search, then one structured grading call over the top passages, then
at most one query rewrite and second search when the grader says the passages are insufficient.
Relevant passages become ``Evidence`` carrying the paragraph text, its citation fields and the
retrieval provenance from the store. If the grader finds nothing relevant, the top few passages
are kept anyway so the synthesizer can say what the corpus does contain.

Without a model the branch skips grading and returns the top passages, marked as ungraded.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.api.errors import UpstreamUnavailableError
from app.graph import prompts
from app.graph.schemas import Evidence, RetrievalGrade, RewrittenQuery
from app.graph.state import BranchInput
from app.llm.providers import ChatModels, call_with_failover
from app.retrieval.store import HybridStore, RetrievedParagraph

log = structlog.get_logger(__name__)


def make_retrieve_documents(
    models: ChatModels | None,
    store: HybridStore | None,
    *,
    grade_top_k: int,
    evidence_per_sub_query: int,
    max_query_rewrites: int,
) -> Callable[[BranchInput], Awaitable[dict[str, Any]]]:
    """Build the node."""

    async def retrieve_documents(branch: BranchInput) -> dict[str, Any]:
        sub_query = branch["sub_query"]
        if store is None:
            return {"notes": [f"{sub_query.id}: the text corpus is not loaded"], "evidence": []}

        result = await store.search(sub_query.text, final_k=grade_top_k)
        hits = list(result.hits)
        providers: list[str] = []
        notes: list[str] = []
        if result.fallback_reason:
            notes.append(f"{sub_query.id}: dense retrieval unavailable, used BM25 only")

        if models is None:
            chosen = hits[:evidence_per_sub_query]
            notes.append(f"{sub_query.id}: passages not graded (no model)")
        else:
            chosen, providers, rewrote = await _grade_with_rewrite(
                models, store, sub_query.text, hits, grade_top_k, max_query_rewrites
            )
            if rewrote:
                notes.append(f"{sub_query.id}: query rewritten once")
            chosen = chosen[:evidence_per_sub_query]

        return {
            "evidence": [_to_evidence(sub_query.id, hit) for hit in chosen],
            "notes": notes,
            "providers": providers,
        }

    return retrieve_documents


async def _grade_with_rewrite(
    models: ChatModels,
    store: HybridStore,
    question: str,
    hits: list[RetrievedParagraph],
    grade_top_k: int,
    max_query_rewrites: int,
) -> tuple[list[RetrievedParagraph], list[str], bool]:
    """Grade once; if insufficient and allowed, rewrite the query, search again, grade again."""
    providers: list[str] = []
    relevant, sufficient, provider = await _grade(models, question, hits)
    providers.append(f"grade:{provider}")
    rewrote = False

    if not sufficient and max_query_rewrites > 0:
        try:
            rewritten, provider = await call_with_failover(
                models.structured(RewrittenQuery),
                [SystemMessage(prompts.QUERY_REWRITER), HumanMessage(question)],
            )
            providers.append(f"rewrite:{provider}")
            new_query = (
                rewritten.query if isinstance(rewritten, RewrittenQuery) else str(rewritten)
            ).strip()
        except UpstreamUnavailableError:
            new_query = ""
        if new_query and new_query.lower() != question.lower():
            rewrote = True
            second = await store.search(new_query, final_k=grade_top_k)
            merged = _merge(hits, second.hits)
            relevant, _, provider = await _grade(models, question, merged)
            providers.append(f"grade:{provider}")
            hits = merged

    if not relevant:
        # Nothing graded relevant: keep the top passages so the answer can say what was found.
        relevant = hits[:3]
    return relevant, providers, rewrote


async def _grade(
    models: ChatModels, question: str, hits: list[RetrievedParagraph]
) -> tuple[list[RetrievedParagraph], bool, str]:
    if not hits:
        return [], False, "none"
    listing = "\n\n".join(
        f"[{hit.paragraph.id}] {hit.paragraph.display_title}: {hit.paragraph.text}" for hit in hits
    )
    messages = [
        SystemMessage(prompts.RETRIEVAL_GRADER),
        HumanMessage(f"Question: {question}\n\nPassages:\n{listing}"),
    ]
    try:
        raw, provider = await call_with_failover(models.structured(RetrievalGrade), messages)
    except UpstreamUnavailableError:
        return hits[:3], True, "none"
    grade = raw if isinstance(raw, RetrievalGrade) else RetrievalGrade.model_validate(raw)
    wanted = set(grade.relevant_ids)
    relevant = [hit for hit in hits if hit.paragraph.id in wanted]
    return relevant, grade.sufficient, provider


def _merge(
    first: list[RetrievedParagraph], second: list[RetrievedParagraph]
) -> list[RetrievedParagraph]:
    seen = {hit.paragraph.id for hit in first}
    return first + [hit for hit in second if hit.paragraph.id not in seen]


def _to_evidence(sub_query_id: str, hit: RetrievedParagraph) -> Evidence:
    paragraph = hit.paragraph
    return Evidence(
        sub_query_id=sub_query_id,
        kind="dataset",
        title=paragraph.display_title,
        reference=(f"SQuAD v1.1 dev - article {paragraph.title!r} - paragraph {paragraph.index}"),
        url=paragraph.url,
        excerpt=paragraph.text,
        locator={"file": "data/paragraphs.jsonl", "paragraph_id": paragraph.id},
        retrieval={
            "found_by": hit.found_by,
            "bm25_rank": hit.bm25_rank,
            "dense_rank": hit.dense_rank,
            "rerank_score": hit.rerank_score,
        },
    )
