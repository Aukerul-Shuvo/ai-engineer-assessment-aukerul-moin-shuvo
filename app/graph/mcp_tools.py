"""Load tools from an MCP server as LangChain tools, over one persistent session.

This is the ``TOOLS_BACKEND=mcp`` path. The bundled adapter package for this could not be used:
its latest release targets the 1.x MCP SDK and fails to import against the 2.x SDK the server is
built on. The bridge below is small and does two things the adapter did not: it holds one client
session open for the life of the application instead of opening one per tool call, and it
returns tool output exactly as the in-process tools do, as JSON text.

``MCPToolBridge`` connects to a Streamable HTTP URL, spawns the bundled stdio server, or attaches
to an in-process ``MCPServer`` instance for tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import TracebackType
from typing import Any

import structlog
from langchain_core.tools import BaseTool, StructuredTool
from mcp import Client, StdioServerParameters
from mcp.server import MCPServer
from mcp.types import TextContent

from app.tools.common import ToolError

log = structlog.get_logger(__name__)

Target = str | StdioServerParameters | MCPServer


def bundled_server_parameters() -> StdioServerParameters:
    """Spawn this repository's own MCP server with the current interpreter."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        cwd=str(Path(__file__).resolve().parents[2]),
    )


class MCPToolBridge:
    """One MCP client session, exposed as LangChain tools."""

    def __init__(self, target: Target) -> None:
        self._client = Client(target)
        self._entered = False

    async def __aenter__(self) -> MCPToolBridge:
        """Open the session."""
        await self._client.__aenter__()
        self._entered = True
        info = self._client.server_info
        log.info("mcp_connected", server=info.name if info else None)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the session."""
        self._entered = False
        await self._client.__aexit__(exc_type, exc, tb)

    async def load_tools(self, names: set[str] | None = None) -> list[BaseTool]:
        """Every server tool, or only ``names``, as LangChain tools calling through the session."""
        if not self._entered:
            raise RuntimeError("MCPToolBridge must be entered before loading tools")
        listed = (await self._client.list_tools()).tools
        tools = [
            self._wrap(tool.name, tool.description or "", tool.input_schema) for tool in listed
        ]
        if names is not None:
            tools = [tool for tool in tools if tool.name in names]
        return tools

    def _wrap(self, name: str, description: str, input_schema: dict[str, Any]) -> BaseTool:
        client = self._client

        async def _call(**kwargs: Any) -> str:
            result = await client.call_tool(name, kwargs)
            text = "\n".join(
                block.text for block in result.content if isinstance(block, TextContent)
            )
            if result.is_error:
                return ToolError(error=text or f"{name} failed").model_dump_json()
            return text

        return StructuredTool.from_function(
            coroutine=_call, name=name, description=description, args_schema=input_schema
        )
