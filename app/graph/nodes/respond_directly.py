"""respond_directly: answer out-of-scope and chit-chat questions without retrieval.

The planner already wrote the reply. This node just places it in the state with no evidence, no
citations and nothing to ground, so the API returns 200 with an empty sources list.
"""

from __future__ import annotations

from typing import Any

from app.graph import prompts
from app.graph.state import GraphState


async def respond_directly(state: GraphState) -> dict[str, Any]:
    """Use the planner's direct reply, or the standard scope message."""
    plan = state.get("plan")
    reply = (plan.direct_reply if plan else None) or prompts.OUT_OF_SCOPE_REPLY
    return {
        "answer": reply,
        "citations": [],
        "cited_evidence": [],
        "grounded": None,
        "grounding_issues": [],
    }
