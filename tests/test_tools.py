import unittest

from src.application.tools import CalculatorTool, EventDbTool, MemoTool, NationTool, SearchTool, TimeTool, TodoTool
from src.infrastructure.repositories.in_memory import (
    InMemoryEventRepository,
    InMemoryMemoRepository,
    InMemoryTodoRepository,
)


class ToolTests(unittest.TestCase):
    def test_time_tool(self):
        tool = TimeTool()
        value = tool.execute(format="%Y")
        self.assertTrue(str(value["now"]).isdigit())

    def test_time_tool_timezone(self):
        tool = TimeTool()
        out = tool.execute(format="%Y-%m-%d %H:%M", timezone="Europe/London")
        self.assertEqual(out["timezone"], "Europe/London")

    def test_search_tool(self):
        tool = SearchTool()
        results = tool.execute(query="mcp", limit=10)
        self.assertGreaterEqual(len(results), 1)

    def test_calculator_tool(self):
        tool = CalculatorTool()
        out = tool.execute(expression="2+3*4")
        self.assertEqual(out["result"], 14.0)

    def test_calculator_tool_rejects_unsafe(self):
        tool = CalculatorTool()
        with self.assertRaises(ValueError):
            tool.execute(expression="__import__('os').system('echo hi')")

    def test_todo_tool(self):
        repo = InMemoryTodoRepository()
        tool = TodoTool(repo)
        created = tool.execute(action="add", title="buy milk")
        self.assertIn("id", created)
        items = tool.execute(action="list")
        self.assertEqual(len(items), 1)
        done = tool.execute(action="done", todo_id=created["id"], is_done=True)
        self.assertTrue(done["is_done"])

    def test_memo_tool(self):
        repo = InMemoryMemoRepository()
        tool = MemoTool(repo)
        created = tool.execute(action="add", title="t", content="c")
        memo = tool.execute(action="get", memo_id=created["id"])
        self.assertEqual(memo["content"], "c")
        memos = tool.execute(action="list")
        self.assertEqual(len(memos), 1)

    def test_nation_tool(self):
        tool = NationTool()
        out = tool.execute(country="Europe/London")
        self.assertEqual(out["resolved"]["timezone"], "Europe/London")

    def test_event_db_tool_insert_and_list(self):
        repo = InMemoryEventRepository()
        tool = EventDbTool(repo)
        inserted = tool.execute(action="insert", payload='{"kind":"test","x":1}')
        self.assertIn("inserted", inserted)
        listed = tool.execute(action="list", limit=10)
        self.assertEqual(len(listed["events"]), 1)


if __name__ == "__main__":
    unittest.main()
