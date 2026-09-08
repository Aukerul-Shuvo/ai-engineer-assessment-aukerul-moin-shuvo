from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import CollectorRegistry

from app.api.schemas import AskResponse, PlanSummary, ResponseMeta, Source
from app.config import Settings
from app.main import create_app
from app.observability.metrics import AskMetrics
from app.observability.tracing import configure_tracing
from tests.test_ask_endpoint import happy_path_script, scripted_service


def test_metrics_endpoint_exposes_http_and_ask_metrics(app: FastAPI, client: TestClient) -> None:
    structured, chat = happy_path_script()
    scripted_service(app, structured=structured, chat=chat)

    assert client.post("/ask", json={"question": "Who won Super Bowl 50?"}).status_code == 200
    assert client.post("/ask", json={"question": "Who won Super Bowl 50?"}).status_code == 200

    body = client.get("/metrics").text

    assert 'ask_requests_total{outcome="answered"} 1.0' in body
    assert 'ask_requests_total{outcome="cached"} 1.0' in body
    assert 'ask_provider_calls_total{provider="fake",step="plan"} 1.0' in body
    assert 'ask_retrieval_mode_total{mode="bm25_only"} 1.0' in body
    assert 'ask_grounding_total{result="supported"} 1.0' in body
    assert "ask_latency_seconds_bucket" in body
    assert 'http_requests_total{handler="/ask",method="POST",status="200"} 2.0' in body
    assert 'handler="/metrics"' not in body, "the metrics route does not measure itself"


def test_metrics_can_be_disabled(test_settings: Settings) -> None:
    app = create_app(test_settings.model_copy(update={"metrics_enabled": False}))

    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 404
    assert app.state.ask_metrics is None


def test_ask_metrics_classify_outcomes() -> None:
    metrics = AskMetrics(CollectorRegistry())

    def response(**overrides: object) -> AskResponse:
        meta = ResponseMeta(
            request_id="r",
            session_id=None,
            providers=["plan:gemini", "synth:groq"],
            latency_ms=1500,
            grounded=True,
            degraded=False,
            degraded_reason=None,
            retrieval_mode="hybrid",
            cached=False,
            notes=[],
        )
        base = {
            "answer": "x [S1]",
            "sources": [
                Source(id="S1", type="dataset", title="t", reference="r", url=None, excerpt="e")
            ],
            "citations": ["S1"],
            "plan": PlanSummary(intent="answerable", sub_queries=[]),
            "meta": meta,
        }
        base.update(overrides)
        return AskResponse(**base)  # type: ignore[arg-type]

    metrics.record(response())
    metrics.record(response(meta=response().meta.model_copy(update={"degraded": True})))
    metrics.record(response(plan=PlanSummary(intent="out_of_scope", sub_queries=[])))
    metrics.record(response(meta=response().meta.model_copy(update={"cached": True})))

    def count(counter, **labels: str) -> float:  # type: ignore[no-untyped-def]
        return counter.labels(**labels)._value.get()

    assert count(metrics.requests, outcome="answered") == 1
    assert count(metrics.requests, outcome="degraded") == 1
    assert count(metrics.requests, outcome="out_of_scope") == 1
    assert count(metrics.requests, outcome="cached") == 1
    assert count(metrics.provider_calls, step="synth", provider="groq") == 3, "cached skipped"
    assert count(metrics.grounding, result="supported") == 3


def test_tracing_is_off_without_an_endpoint(test_settings: Settings) -> None:
    app = create_app(test_settings)

    assert app.state.tracer_provider is None
    assert configure_tracing(app, test_settings) is None


def test_tracing_records_a_span_per_request_when_configured(test_settings: Settings) -> None:
    app = create_app(test_settings)
    exporter = InMemorySpanExporter()
    provider = configure_tracing(app, test_settings, exporter=exporter, synchronous=True)
    assert provider is not None

    with TestClient(app) as client:
        assert client.get("/docs").status_code == 200
        assert client.get("/health/live").status_code == 200

    names = [span.name for span in exporter.get_finished_spans()]
    assert any("/docs" in name for name in names), names
    assert not any("/health/live" in name for name in names), "health checks are not traced"
    resource = exporter.get_finished_spans()[0].resource.attributes
    assert resource["service.name"] == test_settings.otel_service_name
    provider.shutdown()
