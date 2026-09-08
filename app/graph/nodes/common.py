"""Helpers shared by the nodes: message text extraction and evidence rendering."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import BaseMessage

from app.graph.schemas import Evidence


def message_text(message: BaseMessage) -> str:
    """The plain text of a message whose content may be a string or a list of parts."""
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "".join(parts)


def label_evidence(evidence: Sequence[Evidence]) -> list[tuple[str, Evidence]]:
    """Assign S1, S2, ... in the given order."""
    return [(f"S{index}", item) for index, item in enumerate(evidence, start=1)]


def render_evidence(labelled: Sequence[tuple[str, Evidence]]) -> str:
    """The evidence block shown to the synthesizer and the grounding checker."""
    return "\n\n".join(
        f"[{label}] {item.title} ({item.kind}): {item.excerpt}" for label, item in labelled
    )
