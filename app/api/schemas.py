"""Request and response models for the HTTP API.

Everything the API accepts or returns is a Pydantic model declared here, so the OpenAPI document
at ``/docs`` is generated from the same definitions the code validates against. Request models
forbid unknown fields, which catches client typos early.

The response contract is the heart of the assessment's "say where it came from" requirement.
Every ``Source`` carries a human-readable reference, an openable URL, the verbatim excerpt the
answer used, an exact offline locator into the committed corpus, and the retrieval provenance
that explains why it was retrieved. The answer text cites sources by their ``id``.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ------------------------------------------------------------------ health and errors
class HealthResponse(BaseModel):
    """Body of ``GET /health/live``."""

    status: Literal["ok"]
    version: str


class ReadinessResponse(BaseModel):
    """Body of ``GET /health/ready``: overall status plus each named dependency check."""

    status: Literal["ready", "not_ready"]
    checks: dict[str, bool]


class ErrorDetail(BaseModel):
    """Inner object of the error envelope."""

    type: str
    message: str
    request_id: str | None
    details: list[object]


class ErrorEnvelope(BaseModel):
    """Shape of every non-2xx response. Documented here so it appears in OpenAPI."""

    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail


# ------------------------------------------------------------------ ask: request
class AskRequest(BaseModel):
    """Body of ``POST /ask``."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        max_length=4000,
        description="A natural-language question. The configured limit is usually lower.",
        examples=["Who won Super Bowl 50, and how strong is Batman?"],
    )
    session_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]{1,64}$",
        description="Optional. Reuse across requests so follow-up questions have context.",
    )

    @field_validator("question")
    @classmethod
    def _clean_question(cls, value: str) -> str:
        """Trim whitespace, reject blank text and control characters."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be blank")
        if _CONTROL_CHARS.search(cleaned):
            raise ValueError("question must not contain control characters")
        return cleaned


# ------------------------------------------------------------------ ask: response
class SourceLocator(BaseModel):
    """Exact, offline-reproducible pointer into the committed corpus."""

    file: str
    paragraph_id: str


class SourceRetrieval(BaseModel):
    """Why a passage was retrieved: which retriever found it and how it ranked."""

    found_by: list[str]
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rerank_score: float | None = None


class Source(BaseModel):
    """One piece of evidence the answer may cite."""

    id: str = Field(description="Label cited in the answer text, e.g. S1.")
    type: Literal["dataset", "superhero"]
    title: str
    reference: str = Field(description="Human-readable citation.")
    url: str | None = Field(description="Openable link. API URLs carry a {token} placeholder.")
    excerpt: str = Field(description="The verbatim text the answer was built from.")
    locator: SourceLocator | None = None
    retrieval: SourceRetrieval | None = None
    data: dict[str, Any] | None = Field(default=None, description="Full record, for API sources.")


class PlanSubQuery(BaseModel):
    """One routed sub-question from the planner."""

    id: str
    text: str
    source: Literal["dataset", "superhero"]
    hero_name: str | None = None


class PlanSummary(BaseModel):
    """How the question was understood and split."""

    intent: Literal["answerable", "out_of_scope", "chitchat"]
    sub_queries: list[PlanSubQuery]


class ResponseMeta(BaseModel):
    """Operational facts about how this answer was produced."""

    request_id: str | None
    session_id: str | None
    providers: list[str] = Field(description="Which model provider answered each step.")
    latency_ms: int
    grounded: bool | None = Field(description="Grounding check verdict; null when not checked.")
    degraded: bool
    degraded_reason: str | None
    retrieval_mode: Literal["hybrid", "bm25_only"] | None
    cached: bool
    notes: list[str]


class AskResponse(BaseModel):
    """Body of ``POST /ask``."""

    answer: str | None = Field(description="Null only in degraded mode; sources are still given.")
    sources: list[Source]
    citations: list[str] = Field(description="Source ids the answer actually cites.")
    plan: PlanSummary | None
    meta: ResponseMeta
