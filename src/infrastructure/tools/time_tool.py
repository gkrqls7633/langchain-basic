from datetime import datetime
from src.domain.tool import BaseTool

class TimeTool(BaseTool):
    @property
    def name(self) -> str:
        return "time_tool"

    @property
    def description(self) -> str:
        return "Returns the current local time."

    def execute(self, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        return datetime.now().strftime(format)
