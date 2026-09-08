from app.tools.circuit_breaker import CircuitBreaker


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_starts_closed_and_allows_calls() -> None:
    breaker = CircuitBreaker(failure_threshold=3, recovery_s=30, clock=FakeClock())

    assert breaker.state == "closed"
    assert breaker.allow()


def test_opens_after_threshold_consecutive_failures() -> None:
    breaker = CircuitBreaker(failure_threshold=3, recovery_s=30, clock=FakeClock())

    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "closed", "two failures is under the threshold"
    breaker.record_failure()

    assert breaker.state == "open"
    assert not breaker.allow()


def test_success_resets_the_failure_count() -> None:
    breaker = CircuitBreaker(failure_threshold=3, recovery_s=30, clock=FakeClock())

    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()

    assert breaker.state == "closed"


def test_half_open_allows_exactly_one_probe_then_closes_on_success() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, recovery_s=30, clock=clock)
    breaker.record_failure()
    assert not breaker.allow()

    clock.advance(30)

    assert breaker.state == "half_open"
    assert breaker.allow(), "first call after recovery is the probe"
    assert not breaker.allow(), "no second call while the probe is in flight"
    breaker.record_success()
    assert breaker.state == "closed"
    assert breaker.allow()


def test_failed_probe_reopens_for_another_recovery_window() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, recovery_s=30, clock=clock)
    breaker.record_failure()
    clock.advance(30)
    assert breaker.allow()

    breaker.record_failure()

    assert breaker.state == "open"
    clock.advance(29)
    assert not breaker.allow()
    clock.advance(1)
    assert breaker.allow()
