from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import __version__


def test_liveness_reports_version_and_request_id(client: TestClient) -> None:
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}
    assert response.headers["x-request-id"]


def test_readiness_passes_with_secrets_configured(client: TestClient) -> None:
    response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"] == {
        "secrets": True,
        "superhero_client": True,
        "retrieval_index": True,
    }


def test_readiness_names_the_failing_check(app: FastAPI, client: TestClient) -> None:
    app.state.readiness.register("retrieval_store", lambda: False)

    response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["retrieval_store"] is False
    assert body["checks"]["secrets"] is True


def test_readiness_treats_a_raising_check_as_failed(app: FastAPI, client: TestClient) -> None:
    def broken() -> bool:
        raise RuntimeError("boom")

    app.state.readiness.register("broken", broken)

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["broken"] is False


def test_caller_supplied_request_id_is_echoed(client: TestClient) -> None:
    response = client.get("/health/live", headers={"X-Request-ID": "trace-abc-123"})

    assert response.headers["x-request-id"] == "trace-abc-123"
