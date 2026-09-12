"""Prometheus metrics.

``GET /metrics`` exposes standard HTTP metrics per route (request count, latency histogram) from
prometheus-fastapi-instrumentator, plus this service's own view of how questions were answered:

* ``ask_requests_total{outcome}``        answered, cached, degraded or out_of_scope
* ``ask_latency_seconds``                end-to-end time per answered question
* ``ask_provider_calls_total{step,provider}``  which model answered which graph step
* ``ask_retrieval_mode_total{mode}``     dense_reranked, dense or unavailable
* ``ask_grounding_total{result}``        supported, unsupported or unchecked
* ``ask_sources_per_response``           how much evidence each answer carried

Each application instance gets its own registry, so tests that build several apps in one
process do not trip Prometheus's duplicate-metric guard.
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI
from prometheus_client import CollectorRegistry, Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.schemas import AskResponse
from app.config import Settings

log = structlog.get_logger(__name__)

_LATENCY_BUCKETS = (0.25, 0.5, 1, 2, 3, 5, 8, 13, 21, 34, 60)
_SOURCE_BUCKETS = (0, 1, 2, 3, 5, 8, 13, 21)


class AskMetrics:
    """The service-level metrics, recorded once per answered request."""

    def __init__(self, registry: CollectorRegistry) -> None:
        self.requests = Counter(
            "ask_requests_total", "Questions answered, by outcome.", ["outcome"], registry=registry
        )
        self.latency = Histogram(
            "ask_latency_seconds",
            "End-to-end time to answer a question.",
            buckets=_LATENCY_BUCKETS,
            registry=registry,
        )
        self.provider_calls = Counter(
            "ask_provider_calls_total",
            "Model calls by graph step and provider that answered.",
            ["step", "provider"],
            registry=registry,
        )
        self.retrieval_mode = Counter(
            "ask_retrieval_mode_total", "Retrieval mode used.", ["mode"], registry=registry
        )
        self.grounding = Counter(
            "ask_grounding_total", "Grounding check verdicts.", ["result"], registry=registry
        )
        self.sources = Histogram(
            "ask_sources_per_response",
            "Evidence items returned per answer.",
            buckets=_SOURCE_BUCKETS,
            registry=registry,
        )

    def record(self, response: AskResponse) -> None:
        """Update every metric from one finished response."""
        meta = response.meta
        self.requests.labels(outcome=_outcome(response)).inc()
        if not meta.cached:
            self.latency.observe(meta.latency_ms / 1000)
            for entry in meta.providers:
                step, _, provider = entry.partition(":")
                self.provider_calls.labels(step=step, provider=provider or "unknown").inc()
            if meta.retrieval_mode:
                self.retrieval_mode.labels(mode=meta.retrieval_mode).inc()
            self.grounding.labels(result=_grounding(meta.grounded)).inc()
            self.sources.observe(len(response.sources))


def configure_metrics(app: FastAPI, settings: Settings) -> AskMetrics | None:
    """Expose ``/metrics`` and return the service metrics, or ``None`` when disabled."""
    if not settings.metrics_enabled:
        log.info("metrics_disabled")
        return None
    registry = CollectorRegistry()
    Instrumentator(
        registry=registry,
        should_group_status_codes=False,
        excluded_handlers=["/metrics", "/health/.*"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    log.info("metrics_enabled", endpoint="/metrics")
    return AskMetrics(registry)


def _outcome(response: AskResponse) -> str:
    if response.meta.cached:
        return "cached"
    if response.meta.degraded:
        return "degraded"
    if response.plan is not None and response.plan.intent != "answerable":
        return "out_of_scope"
    return "answered"


def _grounding(grounded: bool | None) -> str:
    if grounded is None:
        return "unchecked"
    return "supported" if grounded else "unsupported"
