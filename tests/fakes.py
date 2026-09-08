"""Deterministic stand-ins for external dependencies.

Nothing here talks to a network. Each fake implements the same protocol as the real thing so it
can be injected wherever the real one is expected.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

from app.llm.embeddings import unit_normalize

_WORD = re.compile(r"\w+")


class ScriptedChatModel(BaseChatModel):
    """A chat model that replays a script.

    ``script`` feeds plain and tool-calling turns as ``AIMessage`` objects. ``structured_script``
    feeds structured-output calls as Pydantic models or dicts; when it is empty, structured calls
    fall back to ``script``. Two queues matter because parallel graph branches interleave chat
    and structured calls in a non-deterministic global order, while the order within each kind
    is fixed by the graph. Nodes and agents are tested against exact model behaviour, offline.
    """

    script: list[Any] = Field(default_factory=list)
    structured_script: list[Any] = Field(default_factory=list)
    calls: list[Any] = Field(default_factory=list)
    bound_tools: list[Any] = Field(default_factory=list)
    fail_with: Exception | None = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next(self, structured: bool = False) -> Any:
        if self.fail_with is not None:
            raise self.fail_with
        queue = self.structured_script if structured and self.structured_script else self.script
        if not queue:
            raise RuntimeError("ScriptedChatModel: script exhausted")
        return queue.pop(0)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(messages)
        item = self._next()
        message = item if isinstance(item, AIMessage) else AIMessage(content=str(item))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, AIMessage]:
        self.bound_tools = list(tools)
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:
        def produce(payload: Any) -> Any:
            self.calls.append(payload)
            item = self._next(structured=True)
            if isinstance(item, BaseModel):
                return item
            return schema.model_validate(item)

        return RunnableLambda(produce)


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
