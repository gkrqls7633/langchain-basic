from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Type

from pydantic import BaseModel, Field

from src.domain.tool import BaseTool

logger = logging.getLogger(__name__)

class TimeToolInput(BaseModel):
    format: str = Field(default="%Y-%m-%d %H:%M:%S", description="Python strftime 포맷")


class TimeTool(BaseTool):
    @property
    def name(self) -> str:
        return "time_tool"

    @property
    def description(self) -> str:
        return "현재 로컬 시간/날짜를 반환합니다."

    @property
    def input_model(self) -> Type[BaseModel]:
        return TimeToolInput

    def execute(self, format: str = "%Y-%m-%d %H:%M:%S", **_: Any) -> str:
        actual_format = format if format and format.strip() else "%Y-%m-%d %H:%M:%S"
        logger.info("TimeTool execute format=%s", actual_format)
        return datetime.now().strftime(actual_format)
