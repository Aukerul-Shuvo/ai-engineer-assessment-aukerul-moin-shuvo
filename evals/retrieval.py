"""Retrieval quality on the SQuAD gold questions: recall@k and MRR per stage.

Why this exists: the candidate depth and the final k in the store started as literature
defaults. Every SQuAD question records the paragraph it was written from, so recall at k can be
measured exactly on this corpus. Run this, read the curve, and set the numbers from evidence.

Two stages are scored: ``dense``, the vector search on its own, and ``reranked``, the same
candidates after the cross-encoder. Both come from one search per question, so one query
embedding is spent per question and the two rankings are always from the same retrieval.

    python -m evals.retrieval --limit 300           # the vector search alone
    python -m evals.retrieval --limit 300 --rerank  # add the reranked ranking
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Sequence
from typing import Any

import structlog

from app.config import get_settings
from app.observability.logging import configure_logging
from app.resources import build_embedder, build_reranker, load_store
from app.retrieval.corpus import CorpusPaths, GoldQuestion
from app.retrieval.store import VectorStore
from evals.common import load_gold, markdown_table, write_report

log = structlog.get_logger("evals.retrieval")

KS = (1, 3, 5, 10, 20, 50, 100)


class RankingScores:
    """Accumulates hits at each k and reciprocal ranks for one ranking source."""

    def __init__(self, ks: Sequence[int]) -> None:
        self.ks = list(ks)
        self.hits = dict.fromkeys(self.ks, 0)
        self.reciprocal_ranks: list[float] = []
        self.count = 0

    def add(self, ranked_ids: Sequence[str], gold_id: str) -> None:
        """Record one question's ranking."""
        self.count += 1
        try:
            rank = ranked_ids.index(gold_id) + 1
        except ValueError:
            self.reciprocal_ranks.append(0.0)
            return
        self.reciprocal_ranks.append(1.0 / rank)
        for k in self.ks:
            if rank <= k:
                self.hits[k] += 1

    def summary(self) -> dict[str, float]:
        """recall@k for each k, plus MRR."""
        if not self.count:
            return {}
        out = {f"recall@{k}": round(self.hits[k] / self.count, 4) for k in self.ks}
        out["mrr"] = round(sum(self.reciprocal_ranks) / self.count, 4)
        return out


async def evaluate_retrieval(
    store: VectorStore,
    gold: Sequence[GoldQuestion],
    *,
    ks: Sequence[int] = KS,
    pool: int = 100,
    rerank: bool = False,
) -> dict[str, Any]:
    """Score every available ranking source against the gold paragraph ids."""
    ks = [k for k in ks if k <= pool]
    scores = {"dense": RankingScores(ks)}
    if rerank and store.reranker_enabled:
        scores["reranked"] = RankingScores(ks)

    started = time.perf_counter()
    for index, question in enumerate(gold, start=1):
        # The pool governs every depth, so a --pool sweep measures what the reranker actually
        # sees; the store's own defaults are what the API uses. One search gives both rankings:
        # the hits come back in reranked order and each carries the rank the vector search gave
        # it, so no second query embedding is spent.
        result = await store.search(
            question.question, final_k=pool, dense_top_k=pool, rerank_candidates=pool
        )
        if result.failure_reason:
            log.error("search_unavailable", reason=result.failure_reason)
            raise RuntimeError(result.failure_reason)
        by_dense_rank = sorted(result.hits, key=lambda hit: hit.dense_rank)
        scores["dense"].add([hit.paragraph.id for hit in by_dense_rank], question.paragraph_id)
        if "reranked" in scores:
            scores["reranked"].add([hit.paragraph.id for hit in result.hits], question.paragraph_id)
        if index % 500 == 0:
            log.info(
                "progress",
                done=index,
                total=len(gold),
                elapsed_s=round(time.perf_counter() - started),
            )

    return {
        "questions": len(gold),
        "pool": pool,
        "ks": ks,
        "searchable": store.searchable,
        "reranked": "reranked" in scores,
        "elapsed_s": round(time.perf_counter() - started, 1),
        "results": {name: score.summary() for name, score in scores.items()},
    }


def render(report: dict[str, Any]) -> str:
    """Markdown table of the results."""
    ks = report["ks"]
    headers = ["ranking", *[f"R@{k}" for k in ks], "MRR"]
    rows = [
        [name, *[f"{summary[f'recall@{k}']:.3f}" for k in ks], f"{summary['mrr']:.3f}"]
        for name, summary in report["results"].items()
    ]
    return markdown_table(headers, rows)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--limit", type=int, default=0, help="Sample size; 0 means all questions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool", type=int, default=100, help="Candidate depth per retriever")
    parser.add_argument("--rerank", action="store_true", help="Also score the reranked ranking")
    return parser.parse_args(argv)


async def _main(argv: list[str]) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    configure_logging("INFO", as_json=False)
    paths = CorpusPaths(settings.data_dir)

    embedder = build_embedder(settings)
    reranker = await build_reranker(settings) if args.rerank else None
    store = load_store(settings, embedder, reranker)
    if store is None:
        log.error("corpus_not_loadable", data_dir=str(settings.data_dir))
        return 2

    gold = load_gold(paths, limit=args.limit, seed=args.seed)
    log.info("evaluating", questions=len(gold), searchable=store.searchable, rerank=args.rerank)
    report = await evaluate_retrieval(store, gold, pool=args.pool, rerank=args.rerank)
    report["settings"] = {
        "limit": args.limit,
        "seed": args.seed,
        "embedding_model": settings.gemini_embedding_model,
        "reranker_model": settings.reranker_model if args.rerank else None,
        "reranker_max_length": settings.reranker_max_length if args.rerank else None,
    }
    path = write_report("retrieval", report)
    print()
    print(render(report))
    print(f"\nreport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(sys.argv[1:])))
