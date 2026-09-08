"""Deterministic stand-ins for external dependencies.

Nothing here talks to a network. Each fake implements the same protocol as the real thing so it
can be injected wherever the real one is expected.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from app.llm.embeddings import unit_normalize

_WORD = re.compile(r"\w+")


class FakeEmbedder:
    """Bag-of-hashed-words vectors: texts that share words get similar vectors.

    Good enough to test that indexing, saving, loading and similarity search are wired
    correctly, with no model behind it.
    """

    def __init__(self, dimensions: int = 32, model_id: str = "fake-embedder") -> None:
        self._dimensions = dimensions
        self._model_id = model_id
        self.calls = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        return unit_normalize(np.stack([self._vector(t) for t in texts]))

    async def embed_query(self, text: str) -> np.ndarray:
        self.calls += 1
        return unit_normalize(self._vector(text)[None, :])[0]

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimensions, dtype=np.float32)
        for word in _WORD.findall(text.lower()):
            slot = int(hashlib.md5(word.encode(), usedforsecurity=False).hexdigest(), 16)
            vector[slot % self._dimensions] += 1.0
        return vector
