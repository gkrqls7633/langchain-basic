import unittest

from src.application.tools import CalculatorTool, MemoTool, NationTool, SearchTool, TimeTool, TodoTool
from src.application.tools.event_db_tool import EventDbTool
from src.infrastructure.repositories.in_memory import InMemoryEventRepository, InMemoryMemoRepository, InMemoryTodoRepository


class ToolTests(unittest.TestCase):
    def test_time_tool(self):
        tool = TimeTool()
        result = tool.execute(format="%Y")
        self.assertTrue(str(result["now"]).isdigit())

    def test_time_tool_timezone(self):
        tool = TimeTool()
        result = tool.execute(format="%Y-%m-%d %H:%M", timezone="Europe/London")
        self.assertEqual(result["timezone"], "Europe/London")

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

    def test_event_db_tool_insert_and_list(self):
        repo = InMemoryEventRepository()
        tool = EventDbTool(repo)
        inserted = tool.execute(action="insert", payload='{"kind":"test","x":1}')
        self.assertIn("inserted", inserted)
        listed = tool.execute(action="list", limit=10)
        self.assertEqual(len(listed["events"]), 1)

    def test_nation_tool_schema(self):
        tool = NationTool()
        self.assertIsNotNone(tool.input_model)

    def test_nation_tool_timezone_input(self):
        tool = NationTool()
        out = tool.execute(country="Europe/London")
        self.assertEqual(out["resolved"]["timezone"], "Europe/London")

    def test_nation_tool_country_code_input(self):
        tool = NationTool()
        out = tool.execute(country="GB")
        self.assertEqual(out["resolved"]["country_code"], "GB")


if __name__ == "__main__":
    unittest.main()
