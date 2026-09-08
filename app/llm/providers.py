"""Chat model providers and per-call failover.

Gemini is the primary provider and Groq the fallback. Instead of wrapping both in one model
object, each caller asks ``ChatModels`` for an ordered list of provider-specific runnables built
for its purpose: a structured planner, a tool-calling agent, or plain chat. ``call_with_failover``
tries them in order and returns the result together with the name of the provider that answered,
so every response can state which model produced it.

Failover is per call. One rate-limited Gemini call falls to Groq for that call only; the next
call tries Gemini again. Any exception from a provider triggers the fallback, including a
structured-output parse failure, since a second model is as likely to recover from that as a
retry of the first.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import structlog
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pydantic import BaseModel, SecretStr

from app.api.errors import UpstreamUnavailableError
from app.config import Settings

log = structlog.get_logger(__name__)

Candidate = tuple[str, Runnable[Any, Any]]


@dataclass(frozen=True)
class Provider:
    """One configured chat model and how to name it in logs and responses."""

    name: str
    model_id: str
    model: BaseChatModel


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
        return [(p.name, p.model.with_structured_output(schema)) for p in self._providers]

    def chat(self) -> list[Candidate]:
        """Plain chat runnables returning an ``AIMessage``."""
        return [(p.name, p.model) for p in self._providers]

    def agents(self, tools: Sequence[BaseTool], system_prompt: str) -> list[Candidate]:
        """Tool-calling ReAct agents, one per provider, as compiled LangGraph runnables."""
        return [
            (p.name, create_agent(p.model, list(tools), system_prompt=system_prompt))
            for p in self._providers
        ]


async def call_with_failover(
    candidates: Sequence[Candidate], payload: Any, *, config: RunnableConfig | None = None
) -> tuple[Any, str]:
    """Invoke candidates in order until one succeeds. Returns ``(result, provider_name)``."""
    failures: list[str] = []
    for name, runnable in candidates:
        try:
            result = await runnable.ainvoke(payload, config=config)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            log.warning("provider_failed", provider=name, error=type(exc).__name__, detail=str(exc))
            continue
        return result, name
    raise UpstreamUnavailableError(
        "All model providers failed", details=failures or ["no providers configured"]
    )


def build_chat_models(settings: Settings) -> ChatModels | None:
    """Gemini then Groq, from whichever keys are configured. ``None`` when neither is."""
    providers: list[Provider] = []
    if settings.gemini_api_key is not None:
        providers.append(
            Provider(
                name="gemini",
                model_id=settings.gemini_model,
                model=ChatGoogleGenerativeAI(
                    model=settings.gemini_model,
                    google_api_key=SecretStr(settings.gemini_api_key.get_secret_value()),
                    temperature=settings.llm_temperature,
                    timeout=settings.llm_timeout_s,
                    max_retries=settings.llm_max_retries,
                ),
            )
        )
    if settings.groq_api_key is not None:
        providers.append(
            Provider(
                name="groq",
                model_id=settings.groq_model,
                # Field names rather than their aliases (model, api_key, timeout): the
                # pydantic mypy plugin only knows the former.
                model=ChatGroq(
                    model_name=settings.groq_model,
                    groq_api_key=SecretStr(settings.groq_api_key.get_secret_value()),
                    temperature=settings.llm_temperature,
                    request_timeout=settings.llm_timeout_s,
                    max_retries=settings.llm_max_retries,
                ),
            )
        )
    return ChatModels(providers) if providers else None
