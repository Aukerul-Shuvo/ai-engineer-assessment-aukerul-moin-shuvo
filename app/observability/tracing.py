"""Distributed tracing with OpenTelemetry.

One trace per request: the FastAPI server span, then a span per LangGraph node and per model
call through OpenInference's LangChain instrumentation. Spans export over OTLP to whatever
collector ``OTEL_EXPORTER_OTLP_ENDPOINT`` names; the compose profile ships a collector wired to
Jaeger. With no endpoint configured nothing is installed and there is no overhead, which is the
right default for a reviewer's laptop.

Health and metrics endpoints are excluded from tracing; they would drown real requests.
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
)

from app import __version__
from app.config import Settings

log = structlog.get_logger(__name__)

_UNTRACED = "health/live,health/ready,metrics"


def configure_tracing(
    app: FastAPI,
    settings: Settings,
    *,
    exporter: SpanExporter | None = None,
    synchronous: bool = False,
) -> TracerProvider | None:
    """Install tracing on ``app``. Returns the provider, or ``None`` when tracing is off.

    Args:
        app: The application to instrument. Must not have started yet.
        settings: Source of the endpoint, protocol and service name.
        exporter: Override the OTLP exporter; tests pass an in-memory one.
        synchronous: Export each span as it ends instead of batching; for tests.
    """
    if exporter is None:
        if settings.otel_exporter_otlp_endpoint is None:
            log.info("tracing_disabled", reason="OTEL_EXPORTER_OTLP_ENDPOINT not set")
            return None
        exporter = _otlp_exporter(settings)

    provider = TracerProvider(
        resource=Resource.create(
            {
                SERVICE_NAME: settings.otel_service_name,
                "service.version": __version__,
                "deployment.environment": settings.environment,
            }
        )
    )
    processor = SimpleSpanProcessor(exporter) if synchronous else BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider, excluded_urls=_UNTRACED)
    LangChainInstrumentor().instrument(tracer_provider=provider, skip_dep_check=True)
    log.info(
        "tracing_enabled",
        endpoint=settings.otel_exporter_otlp_endpoint,
        protocol=settings.otel_exporter_otlp_protocol,
    )
    return provider


def _otlp_exporter(settings: Settings) -> SpanExporter:
    endpoint = settings.otel_exporter_otlp_endpoint
    if settings.otel_exporter_otlp_protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        return OTLPSpanExporter(endpoint=endpoint)
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as GrpcSpanExporter,
    )

    insecure = endpoint is not None and endpoint.startswith("http://")
    return GrpcSpanExporter(endpoint=endpoint, insecure=insecure)
