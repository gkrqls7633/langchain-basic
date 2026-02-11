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
        from src.infrastructure.logger import logger
        # Handle empty or None format by falling back to default
        actual_format = format if format and format.strip() else "%Y-%m-%d %H:%M:%S"
        logger.info(f"Executing TimeTool with format: {actual_format} (original: {format})")
        return datetime.now().strftime(actual_format)
