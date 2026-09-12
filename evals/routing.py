"""Planner accuracy on a hand-written golden set. Needs a model key.

Each line of ``routing_golden.jsonl`` is a question with the intent the planner should assign,
the set of sources it should route to, and the minimum number of sub-queries a correct
decomposition needs. The planner is the node the API uses, built the same way, so this measures
the real thing. Calls are paced to stay inside free-tier limits.

    python -m evals.routing --pause 2
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel

from app.config import get_settings
from app.graph.nodes.analyze_query import make_analyze_query
from app.llm.providers import build_chat_models
from app.observability.logging import configure_logging
from app.resources import load_store
from evals.common import markdown_table, write_report

log = structlog.get_logger("evals.routing")

GOLDEN_PATH = Path(__file__).with_name("routing_golden.jsonl")


class GoldenCase(BaseModel):
    """One routing expectation."""

    question: str
    intent: str
    sources: list[str]
    min_sub_queries: int


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenCase]:
    """The golden set."""
    with path.open(encoding="utf-8") as handle:
        return [GoldenCase.model_validate_json(line) for line in handle if line.strip()]


async def evaluate_routing(
    analyze: Any, cases: list[GoldenCase], *, pause_s: float
) -> dict[str, Any]:
    """Run the planner on every case and score intent, sources and decomposition."""
    rows: list[dict[str, Any]] = []
    for case in cases:
        out = await analyze({"question": case.question})
        plan = out["plan"]
        sources = sorted({sq.source for sq in plan.sub_queries})
        row = {
            "question": case.question,
            "expected_intent": case.intent,
            "intent": plan.intent,
            "expected_sources": sorted(case.sources),
            "sources": sources,
            "sub_queries": len(plan.sub_queries),
            "intent_ok": plan.intent == case.intent,
            "sources_ok": sources == sorted(case.sources),
            "decomposition_ok": len(plan.sub_queries) >= case.min_sub_queries,
            "provider": out.get("providers", ["?"])[0].split(":")[-1],
        }
        rows.append(row)
        log.info("case", ok=row["intent_ok"] and row["sources_ok"], question=case.question[:60])
        await asyncio.sleep(pause_s)

    total = len(rows)
    return {
        "cases": total,
        "intent_accuracy": round(sum(r["intent_ok"] for r in rows) / total, 4),
        "sources_accuracy": round(sum(r["sources_ok"] for r in rows) / total, 4),
        "decomposition_accuracy": round(sum(r["decomposition_ok"] for r in rows) / total, 4),
        "all_correct": round(
            sum(r["intent_ok"] and r["sources_ok"] and r["decomposition_ok"] for r in rows) / total,
            4,
        ),
        "rows": rows,
    }


def render(report: dict[str, Any]) -> str:
    """Summary plus the failing cases."""
    summary = markdown_table(
        ["metric", "value"],
        [
            ["cases", report["cases"]],
            ["intent accuracy", report["intent_accuracy"]],
            ["sources accuracy", report["sources_accuracy"]],
            ["decomposition accuracy", report["decomposition_accuracy"]],
            ["all three correct", report["all_correct"]],
        ],
    )
    failures = [
        [
            r["question"],
            f"{r['expected_intent']}/{r['expected_sources']}",
            f"{r['intent']}/{r['sources']}",
        ]
        for r in report["rows"]
        if not (r["intent_ok"] and r["sources_ok"] and r["decomposition_ok"])
    ]
    if not failures:
        return summary + "\n\nAll cases correct."
    return summary + "\n\n" + markdown_table(["question", "expected", "got"], failures)


async def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pause", type=float, default=2.0, help="Seconds between model calls")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging("INFO", as_json=False)
    models = build_chat_models(settings)
    if models is None:
        log.error("no_model_key", hint="set GEMINI_API_KEY")
        return 2
    store = load_store(settings, None, None)
    titles = [doc.title for doc in store.list_documents()] if store else []
    analyze = make_analyze_query(models, titles, max_sub_queries=settings.max_sub_queries)

    report = await evaluate_routing(analyze, load_golden(), pause_s=args.pause)
    report["model"] = models.providers[0].model_id
    path = write_report("routing", report)
    print()
    print(render(report))
    print(f"\nreport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(sys.argv[1:])))
