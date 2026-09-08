"""Types shared by every tool function."""

from __future__ import annotations

from pydantic import BaseModel


class ToolError(BaseModel):
    """A failure returned to the model as data, not raised.

    An agent that receives one can decide what to do next, for example try an alias after a
    failed lookup, instead of the whole request failing.
    """

    error: str
    hint: str | None = None
