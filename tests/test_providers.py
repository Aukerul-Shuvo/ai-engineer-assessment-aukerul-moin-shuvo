"""Provider chain: per-model rate budgets, failover order, short error summaries."""

import pytest
from langchain_core.runnables import RunnableLambda

from app.api.errors import UpstreamUnavailableError
from app.config import Settings
from app.llm.providers import (
    Candidate,
    RateBudget,
    build_chat_models,
    call_with_failover,
    failure_block_seconds,
    summarize_exception,
)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_rate_budget_allows_the_limit_per_window_and_frees_after_sixty_seconds() -> None:
    clock = Clock()
    budget = RateBudget(2, clock=clock)
    assert budget.try_acquire() and budget.try_acquire()
    assert not budget.try_acquire()
    assert budget.seconds_until_free() == 60.0
    clock.t = 59.9
    assert not budget.try_acquire()
    clock.t = 60.0
    assert budget.seconds_until_free() == 0.0
    assert budget.try_acquire()


async def test_failover_skips_a_spent_budget_without_waiting() -> None:
    clock = Clock()
    spent = RateBudget(1, clock=clock)
    spent.try_acquire()
    calls: list[str] = []

    def runnable(name: str) -> RunnableLambda:
        return RunnableLambda(lambda _x: calls.append(name) or f"{name}-ok")

    candidates: list[Candidate] = [
        ("a", runnable("a"), spent),
        ("b", runnable("b"), RateBudget(5, clock=clock)),
    ]
    result, provider = await call_with_failover(candidates, "q")
    assert (result, provider) == ("b-ok", "b")
    assert calls == ["b"]


async def test_failover_waits_bounded_for_the_soonest_budget_when_all_are_spent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    a = RateBudget(1, clock=clock)
    a.try_acquire()  # frees at t=60
    clock.t = 30.0
    b = RateBudget(1, clock=clock)
    b.try_acquire()  # frees at t=90
    waited: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waited.append(seconds)

    monkeypatch.setattr("app.llm.providers.asyncio.sleep", fake_sleep)
    candidates: list[Candidate] = [
        ("a", RunnableLambda(lambda _x: "a-ok"), a),
        ("b", RunnableLambda(lambda _x: "b-ok"), b),
    ]
    result, provider = await call_with_failover(candidates, "q")
    assert (result, provider) == ("a-ok", "a")
    assert waited == [20.0], "a frees in 30 s, but the wait is capped at 20 s"


async def test_failover_reports_one_short_line_per_provider() -> None:
    body = (
        "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current "
        "quota ... Please retry in 14.2s.', 'details': [" + "x" * 900 + "]}}"
    )

    def boom(_x: str) -> str:
        raise RuntimeError(body)

    with pytest.raises(UpstreamUnavailableError) as info:
        await call_with_failover([("a", RunnableLambda(boom), None)], "q")
    assert info.value.details == ["a: 429 rate limited, retry in 14s"]
    assert len(info.value.describe()) < 120


def test_summarize_exception_variants() -> None:
    assert summarize_exception(RuntimeError("503 UNAVAILABLE. high demand")) == "503 unavailable"
    assert (
        summarize_exception(ValueError("parse failed\nsecond line")) == "ValueError: parse failed"
    )
    assert summarize_exception(ValueError("")) == "ValueError"


def test_settings_expose_the_model_chain_and_budgets() -> None:
    settings = Settings(
        environment="test",
        gemini_model="m1",
        gemini_fallback_models="m2, m1 ,m3",
        gemini_model_rpm="m1:15,m3:5",
        _env_file=None,
    )
    assert settings.gemini_model_chain == ["m1", "m2", "m3"]
    assert settings.gemini_rpm_budgets == {"m1": 15, "m3": 5}


def test_build_chat_models_makes_one_provider_per_gemini_model_with_its_budget() -> None:
    settings = Settings(environment="test", gemini_api_key="k", _env_file=None)
    models = build_chat_models(settings)
    assert models is not None
    assert [p.name for p in models.providers] == [
        "gemini/gemini-3.5-flash-lite",
        "gemini/gemini-3.1-flash-lite",
        "gemini/gemini-3.5-flash",
    ]
    assert [p.budget.per_minute if p.budget else None for p in models.providers] == [15, 15, 5]


def test_rate_budget_block_keeps_a_capped_model_out_of_rotation() -> None:
    clock = Clock()
    budget = RateBudget(15, clock=clock)
    budget.block_for(30.0)
    assert not budget.try_acquire()
    assert budget.seconds_until_free() == 30.0
    clock.t = 30.0
    assert budget.try_acquire()


async def test_a_daily_quota_429_blocks_the_provider_for_an_hour() -> None:
    clock = Clock()
    budget = RateBudget(5, clock=clock)
    daily = "429 RESOURCE_EXHAUSTED ... quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"

    def capped(_x: str) -> str:
        raise RuntimeError(daily)

    candidates: list[Candidate] = [
        ("capped", RunnableLambda(capped), budget),
        ("ok", RunnableLambda(lambda _x: "ok"), None),
    ]
    assert await call_with_failover(candidates, "q") == ("ok", "ok")
    assert budget.seconds_until_free() == 3600.0, "not retried until the quota can have reset"
    assert failure_block_seconds(RuntimeError("429 ... Please retry in 14.2s")) == 15.2
    assert failure_block_seconds(ValueError("parse failed")) is None, "a parse failure is not"


async def test_a_slow_provider_is_parked_so_later_calls_skip_it() -> None:
    """A provider answering 504 is unhealthy: the next call should not pay its timeout again."""
    clock = Clock()
    budget = RateBudget(15, clock=clock)
    attempts: list[str] = []

    def sick(_x: str) -> str:
        attempts.append("sick")
        raise RuntimeError("504 DEADLINE_EXCEEDED. Deadline expired before operation completed.")

    def healthy(_x: str) -> str:
        attempts.append("healthy")
        return "ok"

    candidates: list[Candidate] = [
        ("sick", RunnableLambda(sick), budget),
        ("healthy", RunnableLambda(healthy), None),
    ]
    assert await call_with_failover(candidates, "q") == ("ok", "healthy")
    assert await call_with_failover(candidates, "q") == ("ok", "healthy")

    assert attempts == ["sick", "healthy", "healthy"], "the sick provider is tried once, not twice"
    assert budget.seconds_until_free() == 60.0
    assert summarize_exception(TimeoutError()) == "timed out"
