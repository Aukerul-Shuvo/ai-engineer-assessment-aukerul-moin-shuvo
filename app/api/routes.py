"""Health routes.

* ``GET /health/live``  returns 200 while the process runs. Orchestrators use it to restart a
  dead process.
* ``GET /health/ready`` runs every check registered on ``app.state.readiness`` and returns 200,
  or 503 listing the checks that failed. Load balancers use it to decide whether to route
  traffic here.

``POST /ask`` lives in ``app.api.ask`` because its router is built from settings.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response, status

from app import __version__
from app.api.schemas import ErrorEnvelope, HealthResponse, ReadinessResponse

router = APIRouter()

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorEnvelope,
        "description": "A dependency is not ready.",
    }
}


@router.get("/health/live", response_model=HealthResponse, tags=["health"])
async def liveness() -> HealthResponse:
    """Process is up. Always 200."""
    return HealthResponse(status="ok", version=__version__)


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    tags=["health"],
    responses=_ERROR_RESPONSES,
)
async def readiness(request: Request, response: Response) -> ReadinessResponse:
    """Every registered dependency check, by name. 503 if any fails."""
    checks = await request.app.state.readiness.run()
    ready = all(checks.values())
    response.status_code = status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="ready" if ready else "not_ready", checks=checks)
