from app.mcp_server.server import MCPServer


def test_mcp_server_registers_tools() -> None:
    server = MCPServer()
    server.register_tool("lead_tool")
    server.register_tool("quote_tool")

    assert server.list_tools() == ["lead_tool", "quote_tool"]
