"""Request and response models for the HTTP API.

Everything the API accepts or returns is a Pydantic model declared here, so the OpenAPI
document at ``/docs`` is generated from the same definitions the code validates against.
``extra="forbid"`` on request models means unknown fields are rejected rather than silently
ignored, which catches client typos early.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


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
