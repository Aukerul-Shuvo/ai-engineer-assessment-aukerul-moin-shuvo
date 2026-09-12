"""Time one cross-encoder rerank call at several pool depths, input lengths and models.

Why this exists: the reranker is CPU work on every dataset sub-query, so its pool depth and
maximum input length are latency levers as much as recall levers. This prints the wall time of
one ``rerank`` call on real candidates so the depths in ``app/config.py`` can be defended
with a number from the machine at hand. Recall for the same settings comes from
``python -m evals.retrieval --rerank --pool N``; the two tables together are the decision.

    python -m evals.rerank_timing                # defaults below
    python -m evals.rerank_timing --queries 20   # steadier medians

Timings are machine-specific; the report records the CPU name.
"""

import argparse
import asyncio
import platform
import statistics
import sys
import time
from pathlib import Path

from app.config import get_settings
from app.observability.logging import configure_logging
from app.resources import build_embedder, load_store
from app.retrieval.corpus import CorpusPaths, indexed_text
from app.retrieval.rerank import FlashRankReranker
from evals.common import load_gold, write_report

POOLS = (100, 50, 30, 20)
CONFIGS: tuple[tuple[str, int], ...] = (
    ("ms-marco-MiniLM-L-12-v2", 512),
    ("ms-marco-MiniLM-L-12-v2", 256),
    ("ms-marco-TinyBERT-L-2-v2", 512),
)


async def _candidates(queries: int, seed: int) -> list[tuple[str, list[str]]]:
    """Real candidate texts from the vector search, for a sample of gold questions."""
    settings = get_settings()
    store = load_store(settings, build_embedder(settings), reranker=None)
    if store is None:
        raise SystemExit("corpus not loadable; run scripts.build_dataset first")
    gold = load_gold(CorpusPaths(settings.data_dir), limit=queries, seed=seed)
    out: list[tuple[str, list[str]]] = []
    for question in gold:
        ranking = await store.dense_ranking(question.question, k=max(POOLS))
        texts: list[str] = []
        for paragraph_id in ranking:
            paragraph = store.get_paragraph(paragraph_id)
            if paragraph is not None:
                texts.append(indexed_text(paragraph))
        out.append((question.question, texts))
    return out


async def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    configure_logging("WARNING", as_json=False)

    settings = get_settings()
    candidates = await _candidates(args.queries, args.seed)
    words = [len(text.split()) for _, texts in candidates for text in texts]
    print(f"candidate passages: mean {statistics.mean(words):.0f} words, max {max(words)} words")
    print(f"cpu: {platform.processor() or platform.machine()}")

    rows: list[dict[str, object]] = []
    print(f"\n| model | max_length | {' | '.join(f'pool {n}' for n in POOLS)} |")
    print(f"|---|---|{'---|' * len(POOLS)}")
    for model_name, max_length in CONFIGS:
        reranker = await FlashRankReranker.create(
            model_name=model_name,
            cache_dir=Path(settings.reranker_cache_dir),
            max_length=max_length,
        )
        medians: dict[str, float] = {}
        for pool in POOLS:
            timings: list[float] = []
            for query, texts in candidates:
                started = time.perf_counter()
                await reranker.rerank(query, texts[:pool])
                timings.append(time.perf_counter() - started)
            medians[f"pool_{pool}"] = round(statistics.median(timings), 3)
        rows.append({"model": model_name, "max_length": max_length, **medians})
        cells = " | ".join(f"{medians[f'pool_{n}']:.2f}s" for n in POOLS)
        print(f"| {model_name} | {max_length} | {cells} |")

    path = write_report(
        "rerank_timing",
        {"queries": args.queries, "seed": args.seed, "cpu": platform.processor(), "rows": rows},
    )
    print(f"\nreport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(sys.argv[1:])))
