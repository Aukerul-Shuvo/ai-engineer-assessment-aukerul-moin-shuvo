"""Exact-match response cache.

The same question asked twice costs the same model calls twice. Stateless requests (no
``session_id``) with a complete, non-degraded answer are cached by their normalised text for a
short TTL. Session requests are never cached because history changes the meaning of a question.
"""

from __future__ import annotations

import hashlib

from cachetools import TTLCache

from app.api.schemas import AskResponse


class ResponseCache:
    """TTL cache of finished responses keyed by normalised question."""

    def __init__(self, *, ttl_s: int, max_size: int) -> None:
        self._cache: TTLCache[str, AskResponse] = TTLCache(maxsize=max_size, ttl=ttl_s)

    @staticmethod
    def key(question: str) -> str:
        """Case- and whitespace-insensitive fingerprint of the question."""
        normalised = " ".join(question.lower().split())
        return hashlib.sha256(normalised.encode()).hexdigest()

    def get(self, question: str) -> AskResponse | None:
        """A previous response, or ``None``."""
        return self._cache.get(self.key(question))

    def put(self, question: str, response: AskResponse) -> None:
        """Remember a response."""
        self._cache[self.key(question)] = response

    def __len__(self) -> int:
        return len(self._cache)
