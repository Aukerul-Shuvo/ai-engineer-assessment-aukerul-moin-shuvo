"""Structured types the graph's model calls return and the evidence the branches produce.

These Pydantic models double as the JSON schemas handed to the model for structured output, so
their field descriptions are prompt material. Keep them precise.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SourceKind = Literal["dataset", "superhero"]


class SubQuery(BaseModel):
    """One self-contained question routed to one source."""

    id: str = Field(description="Short id: q1, q2, ...")
    text: str = Field(description="A question that stands alone, with pronouns resolved.")
    source: SourceKind = Field(description="Which source can answer it.")
    hero_name: str | None = Field(
        default=None, description="For superhero sub-queries: the character name to look up."
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Ids of sub-queries whose answers are needed before this one can be asked.",
    )


class QueryPlan(BaseModel):
    """The planner's output: what kind of question this is and how to split it."""

    intent: Literal["answerable", "out_of_scope", "chitchat"] = Field(
        description="answerable if any source helps; out_of_scope if none; chitchat for greetings."
    )
    sub_queries: list[SubQuery] = Field(default_factory=list)
    direct_reply: str | None = Field(
        default=None, description="For out_of_scope or chitchat: the short reply to give."
    )


class RetrievalGrade(BaseModel):
    """The grader's verdict on a set of retrieved passages."""

    relevant_ids: list[str] = Field(
        default_factory=list, description="Ids of passages that contain useful information."
    )
    sufficient: bool = Field(description="True if the relevant passages can answer the question.")


class RewrittenQuery(BaseModel):
    """A reformulated search query."""

    query: str


class ResolvedSubQueries(BaseModel):
    """Dependent sub-queries rewritten as self-contained ones after the first wave."""

    sub_queries: list[SubQuery]


class GroundingVerdict(BaseModel):
    """Whether an answer's claims are all supported by the cited evidence."""

    supported: bool
    unsupported_claims: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    """One retrieved item: what the model may cite and what the response returns as a source."""

    sub_query_id: str
    kind: SourceKind
    title: str
    reference: str
    url: str | None
    excerpt: str
    locator: dict[str, Any] | None = None
    retrieval: dict[str, Any] | None = None
    data: dict[str, Any] | None = None
