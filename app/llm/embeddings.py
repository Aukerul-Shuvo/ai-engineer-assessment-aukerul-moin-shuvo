"""Text embeddings for dense retrieval.

One ``Embedder`` protocol, one production implementation. Both the build script (embedding the
corpus) and the running service (embedding each query) go through the same object, which is the
only way to guarantee the two sides use the same model, the same output dimension and the right
task type. Gemini distinguishes ``RETRIEVAL_DOCUMENT`` from ``RETRIEVAL_QUERY``; mixing them up
silently degrades ranking, so the distinction is baked into the two methods.

Vectors are returned as float32 unit vectors, so a dot product is a cosine similarity.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import structlog
from google.genai import errors as genai_errors
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

log = structlog.get_logger(__name__)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_GEMINI_MAX_BATCH = 100


class Embedder(Protocol):
    """Anything that turns text into unit vectors of a fixed size."""

    @property
    def model_id(self) -> str:
        """Identifier recorded next to any index built with this embedder."""
        ...

    @property
    def dimensions(self) -> int:
        """Length of every returned vector."""
        ...

    async def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed corpus passages. Shape ``(len(texts), dimensions)``."""
        ...

    async def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query. Shape ``(dimensions,)``."""
        ...


def unit_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to length 1 so dot product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized: np.ndarray = (vectors / norms).astype(np.float32)
    return normalized


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, genai_errors.APIError):
        return exc.code in _RETRYABLE_STATUS
    return isinstance(exc, (TimeoutError, ConnectionError))


class GeminiEmbedder:
    """Gemini embeddings through LangChain, with retries and explicit task types."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimensions: int,
        timeout_s: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._timeout_s = timeout_s
        self._max_attempts = max_retries + 1
        self._client = GoogleGenerativeAIEmbeddings(
            model=model, google_api_key=SecretStr(api_key), output_dimensionality=dimensions
        )

    @property
    def model_id(self) -> str:
        """The Gemini embedding model name."""
        return self._model

    @property
    def dimensions(self) -> int:
        """Requested output dimensionality."""
        return self._dimensions

    async def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed passages with the document task type."""
        return await self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    async def embed_query(self, text: str) -> np.ndarray:
        """Embed one query with the query task type."""
        vectors = await self._embed([text], task_type="RETRIEVAL_QUERY")
        query_vector: np.ndarray = vectors[0]
        return query_vector

    async def _embed(self, texts: Sequence[str], *, task_type: str) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimensions), dtype=np.float32)
        batches = -(-len(texts) // _GEMINI_MAX_BATCH)
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(initial=1, max=20),
            reraise=True,
        ):
            with attempt:
                if attempt.retry_state.attempt_number > 1:
                    log.warning(
                        "embedding_retry",
                        attempt=attempt.retry_state.attempt_number,
                        task_type=task_type,
                    )
                raw = await asyncio.wait_for(
                    self._client.aembed_documents(
                        list(texts),
                        batch_size=_GEMINI_MAX_BATCH,
                        task_type=task_type,
                        output_dimensionality=self._dimensions,
                    ),
                    timeout=self._timeout_s * batches,
                )
        return unit_normalize(np.asarray(raw, dtype=np.float32))
