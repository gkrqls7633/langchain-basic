import unittest

from src.application.tools import CalculatorTool, MemoTool, SearchTool, TimeTool, TodoTool
from src.infrastructure.repositories.in_memory import InMemoryMemoRepository, InMemoryTodoRepository


class ToolTests(unittest.TestCase):
    def test_time_tool(self):
        tool = TimeTool()
        result = tool.execute(format="%Y")
        self.assertTrue(result.isdigit())

    def test_todo_tool_add_and_list(self):
        repo = InMemoryTodoRepository()
        tool = TodoTool(repo)
        created = tool.execute(action="add", title="buy milk")
        self.assertEqual(created["title"], "buy milk")
        items = tool.execute(action="list")
        self.assertEqual(len(items), 1)

    def test_memo_tool_add_and_get(self):
        repo = InMemoryMemoRepository()
        tool = MemoTool(repo)
        created = tool.execute(action="add", title="t1", content="c1")
        fetched = tool.execute(action="get", memo_id=created["id"])
        self.assertEqual(fetched["content"], "c1")

    def test_search_tool(self):
        tool = SearchTool()
        results = tool.execute(query="mcp", limit=5)
        self.assertGreaterEqual(len(results), 1)

    def test_calculator_tool(self):
        tool = CalculatorTool()
        result = tool.execute(expression="(2+3)*4")
        self.assertEqual(result["result"], 20.0)


if __name__ == "__main__":
    unittest.main()

