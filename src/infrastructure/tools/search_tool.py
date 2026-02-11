from typing import Any, List
from src.domain.tool import BaseTool

class SimpleSearchTool(BaseTool):
    def __init__(self):
        self._data = [
            {"id": 1, "title": "LangChain", "content": "A framework for building LLM applications."},
            {"id": 2, "title": "MCP", "content": "Model Context Protocol for standardizing LLM tool interactions."},
            {"id": 3, "title": "Python", "content": "A versatile programming language used in AI."},
        ]

    @property
    def name(self) -> str:
        return "search_tool"

    @property
    def description(self) -> str:
        return "Searches for keywords in a dummy database."

    def execute(self, query: str) -> List[Any]:
        from src.infrastructure.logger import logger
        logger.info(f"Executing SimpleSearchTool with query: {query}")
        return [item for item in self._data if query.lower() in item["content"].lower() or query.lower() in item["title"].lower()]
