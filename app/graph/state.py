"""Graph state: the single dictionary every node reads from and writes to.

Fields with an ``operator.add`` reducer accumulate across parallel branches; everything else is
overwritten by the last node to write it. ``BranchInput`` is the payload each ``Send`` carries
into a retrieval branch.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage

from app.graph.schemas import Evidence, QueryPlan, SubQuery


class GraphState(TypedDict, total=False):
    """Everything the graph knows about one question."""

    question: str
    history: list[BaseMessage]

    plan: QueryPlan | None
    ready: list[SubQuery]
    pending: list[SubQuery]
    wave: int

    evidence: Annotated[list[Evidence], operator.add]
    # Two audiences. ``notes`` is the operator's record of everything unusual in this run and
    # is reported in the response metadata. ``caveats`` is the subset the person reading the
    # answer needs to know, because it explains a gap in what could be answered; only those
    # are appended to the answer text. "Query rewritten once" is a note, "the superhero source
    # is not configured" is a caveat.
    notes: Annotated[list[str], operator.add]
    caveats: Annotated[list[str], operator.add]
    providers: Annotated[list[str], operator.add]

    answer: str | None
    citations: list[str]
    cited_evidence: list[Evidence]
    grounded: bool | None
    grounding_issues: list[str]
    regenerations: int

    degraded: bool
    degraded_reason: str | None


class BranchInput(TypedDict):
    """What a retrieval branch receives from ``Send``."""

    question: str
    sub_query: SubQuery
