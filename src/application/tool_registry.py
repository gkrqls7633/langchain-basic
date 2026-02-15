from __future__ import annotations

from typing import Dict, List, Optional

from src.domain.tool import BaseTool


class ToolRegistry:
    """
    Application-level registry for domain tools.

    - LangChain 의존성 없음
    - 동적 등록/조회 지원 (향후 플러그인/설정 기반 로딩 확장 가능)
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        name = (tool.name or "").strip()
        if not name:
            raise ValueError("Tool name must be non-empty.")
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = tool

    def list(self) -> List[BaseTool]:
        return list(self._tools.values())

    def get(self, name: str) -> BaseTool:
        tool = self._tools.get(name)
        if not tool:
            raise KeyError(f"Unknown tool: {name}")
        return tool

    def names(self) -> List[str]:
        return list(self._tools.keys())
