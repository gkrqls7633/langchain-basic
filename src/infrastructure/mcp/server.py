from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.application.tool_registry import ToolRegistry
from src.application.mcp_service import MCPService


class MCPServerCore:
    """
    MCP(Model Context Protocol) 서버의 "핵심 로직(Core)"만 분리한 클래스.

    - 실제 전송 계층(stdio/websocket/http 등)은 infrastructure의 별도 adapter로 구현
    - MVP에서는 `src/main.py`의 FastAPI가 HTTP adapter 역할을 수행
    """

    def __init__(self, tool_registry: ToolRegistry, assistant: Optional[MCPService] = None) -> None:
        self._tool_registry = tool_registry
        self._assistant = assistant

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.parameters),
            }
            for tool in self._tool_registry.list()
        ]

    def call_tool(self, name: str, args: Dict[str, Any] | None = None) -> Any:
        tool = self._tool_registry.get(name)
        return tool.execute(**(args or {}))

    def query(self, prompt: str) -> str:
        if not self._assistant:
            raise RuntimeError("Assistant(MCPService) is not configured.")
        return self._assistant.process_query(prompt)
