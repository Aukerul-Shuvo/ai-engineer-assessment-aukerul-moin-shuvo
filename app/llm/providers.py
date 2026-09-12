"""Chat model providers, per-call failover and per-model rate budgets.

Gemini models form a chain, and each model is its own provider, because Google's free tier
meters requests per project *per model*. Measured on the AI Studio dashboard on 2026-09-09: the
Flash Lite models allow 15 requests per minute and 500 per day, the larger Flash models 5 per
minute and 20 per day. One question costs four to nine calls, so the lite models lead and three
small buckets become one usable budget.

Instead of wrapping providers in one model object, each caller asks ``ChatModels`` for an
ordered list of provider-specific runnables built for its purpose: a structured planner, a
tool-calling agent, or plain chat. ``call_with_failover`` tries them in order, skips a provider
whose minute budget is already spent without waiting on it, and returns the result together
with the provider's name, so every response can state which model produced each step.

Failover is per call. One rate-limited call falls to the next model for that call only; the
next call tries the primary again. Any exception from a provider triggers the fallback,
including a structured-output parse failure, since a second model is as likely to recover from
that as a retry of the first. Upstream error bodies are summarised to one short line for the
response and kept in full only in the log.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import structlog
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, SecretStr

from app.api.errors import UpstreamUnavailableError
from app.config import Settings

log = structlog.get_logger(__name__)

# When every provider's minute budget is spent, wait at most this long for the soonest one
# rather than failing the request outright.
_MAX_BUDGET_WAIT_S = 20.0
_RETRY_IN = re.compile(r"retry in ([0-9.]+)s", re.IGNORECASE)


class RateBudget:
    """Sliding-window request budget: at most ``per_minute`` call starts in any 60 seconds.

    Mirrors how the upstream quota is metered, so a call that would be refused with a 429 is
    routed to the next model before it is sent. The clock is injectable for tests.
    """

    def __init__(self, per_minute: int, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._limit = per_minute
        self._clock = clock
        self._starts: deque[float] = deque()
        self._blocked_until = 0.0

    @property
    def per_minute(self) -> int:
        """The configured limit."""
        return self._limit

    def _prune(self, now: float) -> None:
        while self._starts and now - self._starts[0] >= 60.0:
            self._starts.popleft()

    def try_acquire(self) -> bool:
        """Record a call start if the window has room and no block is active. Never waits."""
        now = self._clock()
        self._prune(now)
        if now < self._blocked_until or len(self._starts) >= self._limit:
            return False
        self._starts.append(now)
        return True

    def block_for(self, seconds: float) -> None:
        """Refuse calls for a while: the upstream said so with a 429 and a retry-after."""
        self._blocked_until = max(self._blocked_until, self._clock() + seconds)

    def seconds_until_free(self) -> float:
        """How long until a call would be accepted. Zero when there is room now."""
        now = self._clock()
        self._prune(now)
        blocked = max(0.0, self._blocked_until - now)
        window = 60.0 - (now - self._starts[0]) if len(self._starts) >= self._limit else 0.0
        return max(blocked, window)


# A daily quota answers 429 with a short retry-after that says nothing about when the quota
# returns; an hour keeps a capped model out of the way without abandoning it for the day.
_DAILY_QUOTA_BLOCK_S = 3600.0
# A provider that times out or returns a 5xx is unhealthy for longer than one call. Measured
# 2026-09-09: gemini-3.5-flash-lite spent a day answering a one-word prompt in 30 s instead of
# 1 s, so every call in a request paid the timeout before failing over. One minute in the
# penalty box turns that into one slow call per minute instead of one per step.
_UNHEALTHY_BLOCK_S = 60.0
_UNHEALTHY = ("DEADLINE_EXCEEDED", "UNAVAILABLE", "INTERNAL", " 500", " 502", " 503", " 504")


def failure_block_seconds(exc: BaseException) -> float | None:
    """How long to keep a provider out of rotation after this error, or ``None`` to keep it.

    Rate limits and unhealthy responses say something about the provider, so they park it. A
    parse failure or a bad request says something about this one call, so it does not.
    """
    text = str(exc)
    if "RESOURCE_EXHAUSTED" in text or " 429" in text or text.startswith("429"):
        if "PerDay" in text:
            return _DAILY_QUOTA_BLOCK_S
        match = _RETRY_IN.search(text)
        return float(match.group(1)) + 1.0 if match else 60.0
    if isinstance(exc, TimeoutError) or any(marker in text for marker in _UNHEALTHY):
        return _UNHEALTHY_BLOCK_S
    return None


def summarize_exception(exc: BaseException) -> str:
    """One short line for a provider failure, suitable for a response body.

    Google's 429 and 503 bodies run to a kilobyte of JSON; the caller only needs the status and,
    for a rate limit, how long the upstream asked us to wait.
    """
    text = str(exc)
    if "RESOURCE_EXHAUSTED" in text or " 429" in text or text.startswith("429"):
        match = _RETRY_IN.search(text)
        wait = f", retry in {float(match.group(1)):.0f}s" if match else ""
        return f"429 rate limited{wait}"
    if "DEADLINE_EXCEEDED" in text or " 504" in text or isinstance(exc, TimeoutError):
        return "timed out"
    if "UNAVAILABLE" in text or " 503" in text or text.startswith("503"):
        return "503 unavailable"
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return f"{type(exc).__name__}: {first_line[:120]}" if first_line else type(exc).__name__


Candidate = tuple[str, Runnable[Any, Any], RateBudget | None]


@dataclass(frozen=True)
class Provider:
    """One configured chat model, how to name it in logs and responses, and its minute budget."""

    name: str
    model_id: str
    model: BaseChatModel
    budget: RateBudget | None = None


class ChatModels:
    """Ordered providers, primary first, exposed as purpose-built runnable lists."""

    def __init__(self, providers: Sequence[Provider]) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        self._providers = list(providers)

    @property
    def providers(self) -> list[Provider]:
        """Configured providers in failover order."""
        return list(self._providers)

    def structured(self, schema: type[BaseModel]) -> list[Candidate]:
        """Runnables that return an instance of ``schema``."""
        return [(p.name, p.model.with_structured_output(schema), p.budget) for p in self._providers]

    def chat(self) -> list[Candidate]:
        """Plain chat runnables returning an ``AIMessage``."""
        return [(p.name, p.model, p.budget) for p in self._providers]

    def agents(self, tools: Sequence[BaseTool], system_prompt: str) -> list[Candidate]:
        """Tool-calling ReAct agents, one per provider, as compiled LangGraph runnables."""
        return [
            (p.name, create_agent(p.model, list(tools), system_prompt=system_prompt), p.budget)
            for p in self._providers
        ]


async def call_with_failover(
    candidates: Sequence[Candidate], payload: Any, *, config: RunnableConfig | None = None
) -> tuple[Any, str]:
    """Invoke candidates in order until one succeeds. Returns ``(result, provider_name)``.

    A candidate whose minute budget is spent is skipped without waiting. If every candidate
    was skipped or failed, the one whose budget frees soonest is waited for (bounded) and tried
    once more before giving up.
    """
    failures: list[str] = []
    skipped: list[Candidate] = []
    for name, runnable, budget in candidates:
        if budget is not None and not budget.try_acquire():
            skipped.append((name, runnable, budget))
            log.info("provider_budget_spent", provider=name, per_minute=budget.per_minute)
            continue
        try:
            result = await runnable.ainvoke(payload, config=config)
        except Exception as exc:
            failures.append(f"{name}: {summarize_exception(exc)}")
            log.warning(
                "provider_failed", provider=name, error=type(exc).__name__, detail=str(exc)[:2000]
            )
            block = failure_block_seconds(exc)
            if budget is not None and block is not None:
                budget.block_for(block)
                log.info("provider_blocked", provider=name, seconds=round(block))
            continue
        return result, name

    if skipped:
        name, runnable, budget = min(
            skipped, key=lambda item: item[2].seconds_until_free() if item[2] else 0.0
        )
        wait = min(budget.seconds_until_free() if budget else 0.0, _MAX_BUDGET_WAIT_S)
        log.info("provider_budget_wait", provider=name, seconds=round(wait, 1))
        await asyncio.sleep(wait)
        if budget is not None:
            budget.try_acquire()
        try:
            result = await runnable.ainvoke(payload, config=config)
        except Exception as exc:
            failures.append(f"{name}: {summarize_exception(exc)}")
            log.warning(
                "provider_failed", provider=name, error=type(exc).__name__, detail=str(exc)[:2000]
            )
        else:
            return result, name
        failures.extend(f"{n}: minute budget spent" for n, _, _ in skipped if n != name)

    raise UpstreamUnavailableError(
        "All model providers failed", details=failures or ["no providers configured"]
    )


def _gemini_model(model_id: str, api_key: SecretStr, settings: Settings) -> ChatGoogleGenerativeAI:
    options: dict[str, Any] = {}
    if settings.gemini_thinking_level:
        options["thinking_level"] = settings.gemini_thinking_level
    # Flash Lite models use fixed sampling defaults and warn on every call when a temperature
    # is passed; the larger models honour it.
    if "lite" not in model_id:
        options["temperature"] = settings.llm_temperature
    return ChatGoogleGenerativeAI(
        model=model_id,
        google_api_key=api_key,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        **options,
    )


def build_chat_models(settings: Settings) -> ChatModels | None:
    """The Gemini chain, or ``None`` when no key is configured."""
    providers: list[Provider] = []
    if settings.gemini_api_key is not None:
        key = SecretStr(settings.gemini_api_key.get_secret_value())
        budgets = settings.gemini_rpm_budgets
        for model_id in settings.gemini_model_chain:
            providers.append(
                Provider(
                    name=f"gemini/{model_id}",
                    model_id=model_id,
                    model=_gemini_model(model_id, key, settings),
                    budget=RateBudget(budgets[model_id]) if model_id in budgets else None,
                )
            )
    return ChatModels(providers) if providers else None
