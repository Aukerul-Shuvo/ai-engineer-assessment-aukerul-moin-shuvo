"""Short-lived conversation memory keyed by ``session_id``.

Each question is answered by a fresh graph run; what carries over between turns is only the
recent history, which the planner and synthesizer receive as prior messages so that "and how
tall is he?" can be resolved. History is capped at a few turns and expires after inactivity.

In-memory on purpose: one process, no persistence. A multi-instance deployment would back this
with Redis. LangGraph's checkpointer was considered and rejected because it persists the whole
graph state per thread, which is the wrong shape for one-shot questions with carried context.
"""

from __future__ import annotations

from cachetools import TTLCache
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class SessionStore:
    """Recent turns per session, with TTL expiry."""

    def __init__(self, *, ttl_s: int, max_turns: int = 6, max_sessions: int = 10_000) -> None:
        self._turns: TTLCache[str, list[BaseMessage]] = TTLCache(maxsize=max_sessions, ttl=ttl_s)
        self._max_messages = max_turns * 2

    def history(self, session_id: str) -> list[BaseMessage]:
        """Prior messages for the session, oldest first. Empty for a new session."""
        return list(self._turns.get(session_id, []))

    def record(self, session_id: str, question: str, answer: str) -> None:
        """Append one exchange and drop the oldest beyond the cap."""
        messages = self._turns.get(session_id, [])
        messages = [*messages, HumanMessage(question), AIMessage(answer)][-self._max_messages :]
        self._turns[session_id] = messages

    def __len__(self) -> int:
        return len(self._turns)
