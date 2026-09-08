from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.errors import UpstreamUnavailableError


def _envelope(response) -> dict:  # type: ignore[no-untyped-def]
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"type", "message", "request_id", "details"}
    assert body["error"]["request_id"] == response.headers["x-request-id"]
    return body["error"]


def test_unknown_route_returns_not_found_envelope(client: TestClient) -> None:
    response = client.get("/does-not-exist")

    assert response.status_code == 404
    error = _envelope(response)
    assert error["type"] == "not_found"


def test_validation_failure_returns_422_with_details(app: FastAPI) -> None:
    @app.get("/_test/echo")
    async def echo(n: int) -> dict[str, int]:
        return {"n": n}

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/_test/echo", params={"n": "not-a-number"})

    assert response.status_code == 422
    error = _envelope(response)
    assert error["type"] == "validation_error"
    assert error["details"], "pydantic field errors should be included"
    assert error["details"][0]["loc"] == ["query", "n"]


def test_app_error_maps_to_its_status_and_type(app: FastAPI) -> None:
    @app.get("/_test/upstream-down")
    async def upstream_down() -> None:
        raise UpstreamUnavailableError("Superhero API timed out", details=["timeout after 10s"])

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/_test/upstream-down")

    assert response.status_code == 503
    error = _envelope(response)
    assert error["type"] == "upstream_unavailable"
    assert error["message"] == "Superhero API timed out"
    assert error["details"] == ["timeout after 10s"]


def test_unhandled_exception_returns_500_without_leaking_internals(app: FastAPI) -> None:
    @app.get("/_test/crash")
    async def crash() -> None:
        raise RuntimeError("secret internal detail")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/_test/crash")

    assert response.status_code == 500
    error = _envelope(response)
    assert error["type"] == "internal_error"
    assert "secret internal detail" not in response.text
    assert "Traceback" not in response.text
