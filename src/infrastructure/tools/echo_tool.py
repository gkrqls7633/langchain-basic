from typing import Any
from src.domain.tool import BaseTool

class EchoTool(BaseTool):
    @property
    def name(self) -> str:
        return "echo_tool"

    @property
    def description(self) -> str:
        return "Returns the input text as is."

    def execute(self, text: str) -> str:
        return text
