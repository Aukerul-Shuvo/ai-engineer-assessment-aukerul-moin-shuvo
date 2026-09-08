"""analyze_query: classify the question, split it, route each part.

One structured model call returns a ``QueryPlan``. The plan is then sanitised so that downstream
nodes can trust it: ids are unique, texts are non-empty, dependencies point at real ids and form
no cycle, and the count is capped. Sub-queries with no dependencies form the first wave
(``ready``); the rest wait in ``pending`` for ``resolve_dependencies``.

If no model is configured or every provider fails, the node degrades to a single dataset
sub-query containing the whole question, marks the state degraded, and lets retrieval proceed so
the response can still return relevant sources.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.api.errors import UpstreamUnavailableError
from app.graph import prompts
from app.graph.schemas import QueryPlan, SubQuery
from app.graph.state import GraphState
from app.llm.providers import ChatModels, call_with_failover

log = structlog.get_logger(__name__)


def make_analyze_query(
    models: ChatModels | None, document_titles: Sequence[str], *, max_sub_queries: int
) -> Callable[[GraphState], Awaitable[dict[str, Any]]]:
    """Build the node."""
    system = prompts.PLANNER.format(
        count=len(document_titles),
        titles="\n".join(f"- {title.replace('_', ' ')}" for title in document_titles),
        max_sub_queries=max_sub_queries,
    )

    async def analyze_query(state: GraphState) -> dict[str, Any]:
        question = state["question"]
        if models is None:
            return _degraded(question, "no model provider configured")

        messages = [SystemMessage(system), *state.get("history", []), HumanMessage(question)]
        try:
            raw, provider = await call_with_failover(models.structured(QueryPlan), messages)
        except UpstreamUnavailableError as exc:
            return _degraded(question, exc.describe())

        plan = sanitize_plan(
            raw if isinstance(raw, QueryPlan) else QueryPlan.model_validate(raw), max_sub_queries
        )
        ready = [sq for sq in plan.sub_queries if not sq.depends_on]
        pending = [sq for sq in plan.sub_queries if sq.depends_on]
        log.info(
            "plan",
            intent=plan.intent,
            sub_queries=[(sq.id, sq.source) for sq in plan.sub_queries],
            pending=[sq.id for sq in pending],
            provider=provider,
        )
        return {
            "plan": plan,
            "ready": ready,
            "pending": pending,
            "wave": 1,
            "providers": [f"plan:{provider}"],
            "degraded": False,
            "degraded_reason": None,
        }

    return analyze_query


def _degraded(question: str, reason: str) -> dict[str, Any]:
    log.warning("plan_degraded", reason=reason)
    fallback = SubQuery(id="q1", text=question, source="dataset")
    return {
        "plan": QueryPlan(intent="answerable", sub_queries=[fallback]),
        "ready": [fallback],
        "pending": [],
        "wave": 1,
        "degraded": True,
        "degraded_reason": f"planner unavailable ({reason}); searched the corpus directly",
    }


def sanitize_plan(plan: QueryPlan, max_sub_queries: int) -> QueryPlan:
    """Make the plan safe to execute: unique ids, valid dependencies, no cycles, bounded size."""
    if plan.intent != "answerable":
        return plan.model_copy(update={"sub_queries": []})

    cleaned: list[SubQuery] = []
    seen: set[str] = set()
    for index, sq in enumerate(plan.sub_queries[:max_sub_queries], start=1):
        text = sq.text.strip()
        if not text:
            continue
        sq_id = sq.id.strip() or f"q{index}"
        if sq_id in seen:
            sq_id = f"q{index}"
        seen.add(sq_id)
        cleaned.append(sq.model_copy(update={"id": sq_id, "text": text}))

    ids = {sq.id for sq in cleaned}
    cleaned = [
        sq.model_copy(update={"depends_on": [d for d in sq.depends_on if d in ids and d != sq.id]})
        for sq in cleaned
    ]
    if _has_cycle(cleaned) or all(sq.depends_on for sq in cleaned):
        cleaned = [sq.model_copy(update={"depends_on": []}) for sq in cleaned]

    if not cleaned:
        return plan.model_copy(
            update={"intent": "out_of_scope", "sub_queries": [], "direct_reply": None}
        )
    return plan.model_copy(update={"sub_queries": cleaned})


def _has_cycle(sub_queries: Sequence[SubQuery]) -> bool:
    graph = {sq.id: set(sq.depends_on) for sq in sub_queries}
    resolved: set[str] = set()
    while True:
        progress = [
            sq_id for sq_id, deps in graph.items() if sq_id not in resolved and deps <= resolved
        ]
        if not progress:
            return len(resolved) != len(graph)
        resolved.update(progress)
