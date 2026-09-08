"""synthesize: write the answer from the evidence, citing only what exists.

Evidence from all branches is ordered deterministically (by the plan's sub-query order, then by
retrieval order) and labelled S1, S2, ... so the same question yields the same labels. The model
is told to cite a label after every factual statement. Afterwards the text is checked: labels that
do not correspond to evidence are removed, and the set of labels actually used becomes
``citations``. With no evidence at all the node returns a fixed honest answer without a model
call. If every provider fails, the answer is ``None`` and the state is marked degraded so the API
can still return the sources.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.api.errors import UpstreamUnavailableError
from app.graph import prompts
from app.graph.nodes.common import label_evidence, message_text, render_evidence
from app.graph.schemas import Evidence, QueryPlan
from app.graph.state import GraphState
from app.llm.providers import ChatModels, call_with_failover

log = structlog.get_logger(__name__)

_LABEL = re.compile(r"\[(S\d+)\]")


def make_synthesize(
    models: ChatModels | None,
) -> Callable[[GraphState], Awaitable[dict[str, Any]]]:
    """Build the node."""

    async def synthesize(state: GraphState) -> dict[str, Any]:
        ordered = order_evidence(state.get("plan"), state.get("evidence", []))
        labelled = label_evidence(ordered)
        notes = state.get("notes", [])
        earlier_reason = state.get("degraded_reason")

        if not ordered:
            answer = prompts.NO_EVIDENCE_ANSWER
            if notes:
                answer += " " + _notes_sentence(notes)
            return {"answer": answer, "citations": [], "cited_evidence": []}

        if models is None:
            return _degraded(ordered, "no model provider configured", earlier_reason)

        feedback = ""
        if state.get("grounding_issues"):
            feedback = prompts.SYNTHESIS_FEEDBACK.format(
                issues="\n".join(f"- {issue}" for issue in state["grounding_issues"])
            )
        system = prompts.SYNTHESIZER.format(feedback=feedback, evidence=render_evidence(labelled))
        messages = [
            SystemMessage(system),
            *state.get("history", []),
            HumanMessage(state["question"]),
        ]
        try:
            reply, provider = await call_with_failover(models.chat(), messages)
        except UpstreamUnavailableError as exc:
            return _degraded(ordered, exc.describe(), earlier_reason)

        text, citations = clean_citations(message_text(reply), {label for label, _ in labelled})
        if notes and not state.get("grounding_issues"):
            text = text.rstrip() + " " + _notes_sentence(notes)
        return {
            "answer": text,
            "citations": citations,
            "cited_evidence": ordered,
            "providers": [f"synth:{provider}"],
        }

    return synthesize


def order_evidence(plan: QueryPlan | None, evidence: Sequence[Evidence]) -> list[Evidence]:
    """Plan order first, retrieval order second. Stable, so labels are reproducible."""
    position = {sq.id: index for index, sq in enumerate(plan.sub_queries)} if plan else {}
    return sorted(evidence, key=lambda item: position.get(item.sub_query_id, len(position)))


def clean_citations(text: str, valid: set[str]) -> tuple[str, list[str]]:
    """Drop labels that do not exist; return the cleaned text and the labels used, in order."""
    used: list[str] = []
    for label in _LABEL.findall(text):
        if label in valid and label not in used:
            used.append(label)
    cleaned = _LABEL.sub(lambda m: m.group(0) if m.group(1) in valid else "", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, sorted(used, key=lambda label: int(label[1:]))


def _degraded(ordered: list[Evidence], reason: str, earlier: str | None) -> dict[str, Any]:
    """Keep the sources, drop the answer, and add this reason to any earlier one."""
    log.warning("synthesis_degraded", reason=reason)
    this = f"answer synthesis unavailable ({reason}); returning sources only"
    return {
        "answer": None,
        "citations": [],
        "cited_evidence": ordered,
        "degraded": True,
        "degraded_reason": f"{earlier}; {this}" if earlier else this,
    }


def _notes_sentence(notes: Sequence[str]) -> str:
    unique = list(dict.fromkeys(note.split(": ", 1)[-1] for note in notes))
    return "Note: " + "; ".join(unique) + "."
