"""Shared test fixtures.

Tests never read real secrets or call real services. ``test_settings`` builds a Settings object
with dummy keys and ``environment="test"``, ignoring any local .env file. ``client`` is a
TestClient with the lifespan running, so startup and shutdown code is exercised in every test.
Server exceptions are not re-raised so tests can assert on the error envelope.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.config import Settings
from app.main import create_app


@pytest.fixture
def test_settings() -> Settings:
    # The real committed corpus in data/ is loaded, which doubles as a check that the committed
    # artifacts are valid. The reranker is off so no model weights are needed to run tests.
    return Settings(
        _env_file=None,
        environment="test",
        log_json=False,
        reranker_enabled=False,
        gemini_api_key=SecretStr("test-gemini-key"),
        superhero_api_token=SecretStr("test-superhero-token"),
    )


@pytest.fixture
def app(test_settings: Settings) -> FastAPI:
    return create_app(test_settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
