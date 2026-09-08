"""The four knowledge tools exposed over the Model Context Protocol.

The API calls the tool functions in ``app.tools`` directly. This package registers the very same
functions with the MCP SDK so any MCP host, such as MCP Inspector, Claude Desktop or another
agent, can call them too. No tool logic lives here.
"""
