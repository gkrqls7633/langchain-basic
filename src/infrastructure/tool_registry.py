from typing import List, Dict, Any
from langchain.tools import Tool
from src.domain.tool import BaseTool

class ToolRegistry:
    """
    Registry for managing and wrapping tools.
    Separates domain tools from LangChain's Tool class.
    """
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get_langchain_tools(self) -> List[Tool]:
        """
        Wraps domain tools in LangChain's Tool structure.
        Only Infrastructure layer should know about LangChain's Tool class.
        """
        lc_tools = []
        for tool in self._tools.values():
            lc_tools.append(
                Tool(
                    name=tool.name,
                    func=tool.execute,
                    description=tool.description
                )
            )
        return lc_tools

    def get_tool_names(self) -> List[str]:
        return list(self._tools.keys())
