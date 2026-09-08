"""check_grounding: verify the answer against the evidence it cites.

One structured model call reads the evidence block and the answer and lists unsupported claims.
If there are any and a regeneration is still allowed, the issues are stored and the graph routes
back to ``synthesize``, which folds them into its prompt. Otherwise the verdict is recorded and
the answer ships with ``grounded`` set accordingly, so the API can flag it. Answers with no
evidence or no text are not checked.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from langchain_core.messages import HumanMessage

from app.api.errors import UpstreamUnavailableError
from app.graph import prompts
from app.graph.nodes.common import label_evidence, render_evidence
from app.graph.schemas import GroundingVerdict
from app.graph.state import GraphState
from app.llm.providers import ChatModels, call_with_failover

log = structlog.get_logger(__name__)


def make_check_grounding(
    models: ChatModels | None,
) -> Callable[[GraphState], Awaitable[dict[str, Any]]]:
    """Build the node."""

    async def check_grounding(state: GraphState) -> dict[str, Any]:
        answer = state.get("answer")
        evidence = state.get("cited_evidence", [])
        if models is None or not answer or not evidence:
            return {"grounded": None, "grounding_issues": []}

        prompt = prompts.GROUNDING_CHECKER.format(
            evidence=render_evidence(label_evidence(evidence)), answer=answer
        )
        try:
            raw, provider = await call_with_failover(
                models.structured(GroundingVerdict), [HumanMessage(prompt)]
            )
        except UpstreamUnavailableError:
            return {"grounded": None, "grounding_issues": []}

        verdict = raw if isinstance(raw, GroundingVerdict) else GroundingVerdict.model_validate(raw)
        regenerations = state.get("regenerations", 0)
        if verdict.supported:
            return {"grounded": True, "grounding_issues": [], "providers": [f"ground:{provider}"]}

        log.info("grounding_failed", issues=verdict.unsupported_claims, regenerations=regenerations)
        return {
            "grounded": False,
            "grounding_issues": verdict.unsupported_claims,
            "regenerations": regenerations + 1,
            "providers": [f"ground:{provider}"],
        }

    return check_grounding
