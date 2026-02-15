from __future__ import annotations

import os

from src.application.mcp_service import MCPService
from src.application.tool_registry import ToolRegistry
from src.application.tools import CalculatorTool, EventDbTool, MemoTool, NationTool, SearchTool, TimeTool, TodoTool
from src.infrastructure.llm.fake_llm import FakeLLM
from src.infrastructure.llm.gemini_llm import GeminiLLM
from src.infrastructure.repositories.in_memory import InMemoryEventRepository, InMemoryMemoRepository, InMemoryTodoRepository
from src.infrastructure.repositories.postgres import PostgresEventRepository


def _build_service() -> MCPService:
    tool_registry = ToolRegistry()
    tool_registry.register(TimeTool())
    tool_registry.register(TodoTool(InMemoryTodoRepository()))
    tool_registry.register(MemoTool(InMemoryMemoRepository()))
    tool_registry.register(SearchTool())
    tool_registry.register(CalculatorTool())
    tool_registry.register(NationTool())
    event_db_mode = os.getenv("EVENT_DB_MODE", "postgres").strip().lower()
    if event_db_mode == "memory":
        event_repo = InMemoryEventRepository()
    else:
        try:
            event_repo = PostgresEventRepository(database_url=os.getenv("DATABASE_URL"))
        except Exception:
            event_repo = InMemoryEventRepository()
    tool_registry.register(EventDbTool(event_repo))

    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        llm = GeminiLLM(model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    else:
        llm = FakeLLM()

    return MCPService(llm=llm, tool_registry=tool_registry)


def main() -> None:
    service = _build_service()
    print("=== Personal Assistant CLI Demo ===")
    print("exit / quit 로 종료")
    while True:
        user_input = input("\nYou> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        print("AI>", service.process_query(user_input))


if __name__ == "__main__":
    main()
