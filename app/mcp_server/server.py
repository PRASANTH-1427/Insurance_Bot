from __future__ import annotations


class MCPServer:
    def __init__(self, name: str = "insureassist-mcp") -> None:
        self.name = name
        self.tools: list[str] = []

    def register_tool(self, tool_name: str) -> None:
        self.tools.append(tool_name)

    def list_tools(self) -> list[str]:
        return list(self.tools)
