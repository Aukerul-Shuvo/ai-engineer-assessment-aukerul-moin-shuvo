"""resolve_dependencies: turn wave-one evidence into self-contained wave-two sub-queries.

A sub-query like "Name a superhero from <country from q1>" cannot run until q1 has an answer.
After the first wave, this node shows the model the pending sub-queries and the evidence so far
and asks for rewritten, self-contained versions. Those become ``ready`` for a second dispatch.
There are at most two waves; anything still unresolved after that is dropped with a note.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.api.errors import UpstreamUnavailableError
from app.graph import prompts
from app.graph.nodes.common import label_evidence, render_evidence
from app.graph.schemas import ResolvedSubQueries, SubQuery
from app.graph.state import GraphState
from app.llm.providers import ChatModels, call_with_failover

log = structlog.get_logger(__name__)


def make_resolve_dependencies(
    models: ChatModels | None,
) -> Callable[[GraphState], Awaitable[dict[str, Any]]]:
    """Build the node."""

    async def resolve_dependencies(state: GraphState) -> dict[str, Any]:
        pending = state.get("pending", [])
        wave = state.get("wave", 1) + 1
        if not pending:
            return {"ready": [], "pending": [], "wave": wave}
        if models is None:
            return {"ready": _unblocked(pending), "pending": [], "wave": wave}

        evidence = render_evidence(label_evidence(state.get("evidence", [])))
        listing = "\n".join(
            f"- {sq.id} ({sq.source}, depends on {', '.join(sq.depends_on)}): {sq.text}"
            for sq in pending
        )
        messages = [
            SystemMessage(prompts.DEPENDENCY_RESOLVER),
            HumanMessage(
                f"Original question: {state['question']}\n\nPending sub-queries:\n{listing}"
                f"\n\nEvidence so far:\n{evidence or '(none)'}"
            ),
        ]
        try:
            raw, provider = await call_with_failover(
                models.structured(ResolvedSubQueries), messages
            )
        except UpstreamUnavailableError:
            return {"ready": _unblocked(pending), "pending": [], "wave": wave}

        resolved = (
            raw if isinstance(raw, ResolvedSubQueries) else ResolvedSubQueries.model_validate(raw)
        )
        by_id = {sq.id: sq for sq in resolved.sub_queries}
        ready = [
            original.model_copy(
                update={
                    "text": by_id.get(original.id, original).text.strip() or original.text,
                    "hero_name": by_id.get(original.id, original).hero_name or original.hero_name,
                    "depends_on": [],
                }
            )
            for original in pending
        ]
        log.info("dependencies_resolved", wave=wave, ids=[sq.id for sq in ready], provider=provider)
        return {
            "ready": ready,
            "pending": [],
            "wave": wave,
            "providers": [f"resolve:{provider}"],
        }

    return resolve_dependencies


def _unblocked(pending: list[SubQuery]) -> list[SubQuery]:
    """Run pending sub-queries as written when no model can rewrite them."""
    return [sq.model_copy(update={"depends_on": []}) for sq in pending]
