"""Readiness checks.

Startup code registers one named check per dependency the service needs: retrieval index
loaded, model provider configured, MCP session connected, and so on. ``GET /health/ready``
runs them all concurrently and reports each by name, so an operator sees exactly which
dependency is failing instead of a bare 503.

A check is any callable returning ``bool`` or an awaitable of ``bool``. A check that raises
counts as failed and the exception is logged.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable

import structlog

log = structlog.get_logger(__name__)

Check = Callable[[], bool | Awaitable[bool]]


class ReadinessRegistry:
    """Named dependency checks, run together on demand."""

    def __init__(self) -> None:
        self._checks: dict[str, Check] = {}

    def register(self, name: str, check: Check) -> None:
        """Add or replace the check for ``name``."""
        self._checks[name] = check

    async def run(self) -> dict[str, bool]:
        """Run every check concurrently and return ``{name: passed}``."""
        names = list(self._checks)
        results = await asyncio.gather(*(self._run_one(name) for name in names))
        return dict(zip(names, results, strict=True))

    async def _run_one(self, name: str) -> bool:
        try:
            outcome = self._checks[name]()
            if inspect.isawaitable(outcome):
                outcome = await outcome
            return bool(outcome)
        except Exception:
            log.exception("readiness_check_failed", check=name)
            return False
