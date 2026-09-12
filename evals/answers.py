"""End-to-end answer quality on a sample of SQuAD questions. Needs model keys.

Runs the compiled graph exactly as the API does, without HTTP, and scores each answer:

* exact match and F1 against the gold answers, the SQuAD standard
* gold-in-sources: whether any returned source contains a gold answer (retrieval succeeded)
* citation precision: fraction of cited sources whose excerpt contains a gold answer
* grounded and degraded rates from the graph's own flags, and which providers answered
* latency percentiles

Exact match is reported but is not the headline: a strong generator paraphrases, and the CMU
project on a sibling corpus saw exact match fall while grounding rose. Citation precision and
the grounding rate say whether the answer is trustworthy; F1 says whether it is on target.

    python -m evals.answers --limit 50 --pause 3
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Any

import structlog

from app.config import get_settings
from app.graph.build import GraphDependencies, build_graph
from app.llm.providers import build_chat_models
from app.observability.logging import configure_logging
from app.resources import (
    build_agent_tools,
    build_embedder,
    build_http_client,
    build_reranker,
    build_superhero_client,
    load_store,
)
from app.retrieval.corpus import CorpusPaths, GoldQuestion
from evals.common import (
    contains_gold,
    exact_match,
    f1,
    load_gold,
    markdown_table,
    percentile,
    write_report,
)

log = structlog.get_logger("evals.answers")


async def evaluate_answers(
    graph: Any, gold: list[GoldQuestion], *, pause_s: float
) -> dict[str, Any]:
    """Answer each question through the graph and score it."""
    rows: list[dict[str, Any]] = []
    for question in gold:
        started = time.perf_counter()
        state = await graph.ainvoke({"question": question.question})
        latency = time.perf_counter() - started

        answer = state.get("answer") or ""
        evidence = state.get("cited_evidence") or []
        labels = {f"S{i}": item for i, item in enumerate(evidence, start=1)}
        cited = [labels[c] for c in state.get("citations") or [] if c in labels]
        rows.append(
            {
                "id": question.id,
                "question": question.question,
                "gold": question.answers,
                "answer": answer,
                "em": exact_match(answer, question.answers) if answer else 0.0,
                "f1": f1(answer, question.answers) if answer else 0.0,
                "gold_in_sources": any(
                    contains_gold(e.excerpt, question.answers) for e in evidence
                ),
                "citation_precision": (
                    sum(contains_gold(e.excerpt, question.answers) for e in cited) / len(cited)
                    if cited
                    else None
                ),
                "grounded": state.get("grounded"),
                "degraded": bool(state.get("degraded")),
                "providers": state.get("providers") or [],
                "latency_s": round(latency, 2),
            }
        )
        log.info(
            "answered", f1=rows[-1]["f1"], latency_s=rows[-1]["latency_s"], q=question.question[:50]
        )
        await asyncio.sleep(pause_s)

    total = len(rows)
    with_citations = [r["citation_precision"] for r in rows if r["citation_precision"] is not None]
    latencies = [r["latency_s"] for r in rows]
    return {
        "questions": total,
        "exact_match": round(sum(r["em"] for r in rows) / total, 4),
        "f1": round(sum(r["f1"] for r in rows) / total, 4),
        "gold_in_sources": round(sum(r["gold_in_sources"] for r in rows) / total, 4),
        "citation_precision": round(sum(with_citations) / len(with_citations), 4)
        if with_citations
        else None,
        "grounded_rate": round(sum(r["grounded"] is True for r in rows) / total, 4),
        "degraded_rate": round(sum(r["degraded"] for r in rows) / total, 4),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "rows": rows,
    }


def render(report: dict[str, Any]) -> str:
    """Summary table."""
    return markdown_table(
        ["metric", "value"],
        [
            ["questions", report["questions"]],
            ["exact match", report["exact_match"]],
            ["F1", report["f1"]],
            ["gold answer in sources", report["gold_in_sources"]],
            ["citation precision", report["citation_precision"]],
            ["grounded rate", report["grounded_rate"]],
            ["degraded rate", report["degraded_rate"]],
            ["latency p50 (s)", report["latency_p50_s"]],
            ["latency p95 (s)", report["latency_p95_s"]],
        ],
    )


async def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pause", type=float, default=3.0, help="Seconds between questions")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging("INFO", as_json=False)
    models = build_chat_models(settings)
    if models is None:
        log.error("no_model_key", hint="set GEMINI_API_KEY")
        return 2

    http = build_http_client(settings)
    superhero = build_superhero_client(settings, http)
    store = load_store(settings, build_embedder(settings), await build_reranker(settings))
    tools, bridge = await build_agent_tools(settings, superhero)
    graph = build_graph(
        GraphDependencies(
            models=models,
            store=store,
            superhero_tools=tools,
            document_titles=[d.title for d in store.list_documents()] if store else [],
            max_agent_steps=settings.max_agent_steps,
            max_query_rewrites=settings.max_query_rewrites,
            max_regenerations=settings.max_regenerations,
            max_sub_queries=settings.max_sub_queries,
            grade_top_k=settings.grade_top_k,
            evidence_per_sub_query=settings.evidence_per_sub_query,
        )
    )
    try:
        gold = load_gold(CorpusPaths(settings.data_dir), limit=args.limit, seed=args.seed)
        report = await evaluate_answers(graph, gold, pause_s=args.pause)
    finally:
        if bridge is not None:
            await bridge.__aexit__(None, None, None)
        await http.aclose()

    report["model"] = models.providers[0].model_id
    report["corpus_searchable"] = bool(store and store.searchable)
    path = write_report("answers", report)
    print()
    print(render(report))
    print(f"\nreport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(sys.argv[1:])))
