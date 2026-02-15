from __future__ import annotations

from src.domain.llm import LLMProvider
from src.application.tool_registry import ToolRegistry


DEFAULT_SYSTEM_PROMPT = """\
당신은 개인 비서형 AI이며, 가능하면 항상 도구(Tool)를 호출해 문제를 해결합니다.

규칙:
- 사용자의 질문에 직접 답하지 말고, 가능한 경우 반드시 tool을 1회 이상 호출하세요.
- tool 호출 결과를 바탕으로 최종 답변을 간결하게 작성하세요.
- tool로 해결 불가하면, 어떤 tool이 필요한지 1문장으로 제안하고 필요한 정보를 질문하세요.
"""


class MCPService:
    """
    Application layer service that orchestrates LLM and Tools.
    This layer only depends on domain interfaces and application registry.
    """
    def __init__(self, llm: LLMProvider, tool_registry: ToolRegistry):
        self.llm = llm
        self.tool_registry = tool_registry

    def process_query(self, query: str) -> str:
        """
        Processes a user query by using the LLM and registered tools.
        """
        tools = self.tool_registry.list()
        return self.llm.generate(query, tools=tools, system_prompt=DEFAULT_SYSTEM_PROMPT)
