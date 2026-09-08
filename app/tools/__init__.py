"""The two knowledge sources as plain async functions.

``superhero`` wraps the Superhero API; ``documents`` wraps the retrieval store. Each function
takes its dependency explicitly and returns a Pydantic result, so the same code is called
directly by the LangGraph agent, registered with the MCP server, and unit-tested with fakes.
Nothing here knows about LangChain or MCP.
"""
