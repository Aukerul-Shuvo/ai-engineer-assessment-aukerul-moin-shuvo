"""Shared test fixtures.

Tests never read real secrets or call real services. ``test_settings`` builds a Settings object
with dummy keys and ``environment="test"``, ignoring any local .env file. ``client`` is a
TestClient with the lifespan running, so startup and shutdown code is exercised in every test.
Server exceptions are not re-raised so tests can assert on the error envelope.

Retrieval is semantic, so a test corpus needs vectors and something to embed queries with. Both
come from ``FakeEmbedder``: the ``test_corpus`` fixture builds a four-paragraph corpus with it
once per session, and an autouse fixture makes every entry point use it instead of Gemini. No
test can reach the embedding API even if a real key is present in the environment.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.config import Settings
from app.main import create_app
from tests.fakes import (
    TEST_DIMENSIONS,
    TEST_EMBEDDING_MODEL,
    FakeEmbedder,
    build_test_corpus,
)


@pytest.fixture(scope="session")
def test_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The data directory of a small corpus with fake vectors, built once for the session."""
    root = tmp_path_factory.mktemp("corpus")
    paths = asyncio.run(build_test_corpus(root, FakeEmbedder(TEST_DIMENSIONS)))
    return paths.data_dir


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every entry point embeds with the fake, matching the vectors in the test corpus."""

    def build(_settings: Settings) -> FakeEmbedder:
        return FakeEmbedder(TEST_DIMENSIONS)

    for module in ("app.lifespan", "mcp_server.server"):
        monkeypatch.setattr(f"{module}.build_embedder", build)


@pytest.fixture
def test_settings(test_corpus: Path) -> Settings:
    # The reranker is off so no model weights are needed to run tests; the ordering tests inject
    # their own. The embedding model and width match what built the test corpus, which is what
    # the store verifies before it will use an index.
    return Settings(
        _env_file=None,
        environment="test",
        log_json=False,
        data_dir=test_corpus,
        gemini_embedding_model=TEST_EMBEDDING_MODEL,
        embedding_dimensions=TEST_DIMENSIONS,
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
