"""A minimal circuit breaker for calls to an external service.

When a dependency is down, every call waits for its full timeout and then fails. Under load
that turns one slow dependency into a slow service. The breaker counts consecutive failures.
After ``failure_threshold`` of them it opens, and calls fail immediately for ``recovery_s``
seconds. Then a single probe call is let through: success closes the breaker, failure opens it
again for another recovery window.

Deliberately tiny and dependency-free. The clock is injectable so tests do not sleep.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal

State = Literal["closed", "open", "half_open"]


class CircuitBreaker:
    """Fail fast after repeated failures; probe once per recovery window."""

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = failure_threshold
        self._recovery_s = recovery_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._probing = False

    @property
    def state(self) -> State:
        """Current state, derived from failure count and elapsed time."""
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self._recovery_s:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        """Whether a call may proceed right now. Half-open lets exactly one probe through."""
        state = self.state
        if state == "closed":
            return True
        if state == "half_open" and not self._probing:
            self._probing = True
            return True
        return False

    def record_success(self) -> None:
        """A call succeeded: close the breaker and forget past failures."""
        self._failures = 0
        self._opened_at = None
        self._probing = False

    def record_failure(self) -> None:
        """A call failed: count it, and open the breaker at the threshold or on a failed probe."""
        self._failures += 1
        if self._probing or self._failures >= self._threshold:
            self._opened_at = self._clock()
            self._probing = False
