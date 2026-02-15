from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional, Type

import pytz

from pydantic import BaseModel, Field

from src.domain.tool import BaseTool

logger = logging.getLogger(__name__)


class TimeToolInput(BaseModel):
    format: str = Field(default="%Y-%m-%d %H:%M:%S", description="Python strftime 포맷")
    timezone: Optional[str] = Field(
        default=None,
        description="IANA timezone (예: Europe/London, Asia/Seoul). 없으면 서버 로컬 시간.",
    )


class TimeTool(BaseTool):
    @property
    def name(self) -> str:
        return "time_tool"

    @property
    def description(self) -> str:
        return "timezone(선택)을 받아 현재 시간/날짜를 반환합니다. 시간 계산은 이 tool에서만 수행합니다."

    @property
    def input_model(self) -> Type[BaseModel]:
        return TimeToolInput

    def execute(
        self,
        format: str = "%Y-%m-%d %H:%M:%S",
        timezone: Optional[str] = None,
        **_: Any,
    ) -> dict:
        actual_format = format if format and format.strip() else "%Y-%m-%d %H:%M:%S"
        tz_name = (timezone or "").strip() or None

        if tz_name:
            tz = pytz.timezone(tz_name)
            now = datetime.now(tz)
        else:
            now = datetime.now()

        logger.info("TimeTool execute format=%s timezone=%s", actual_format, tz_name)
        return {"timezone": tz_name, "now": now.strftime(actual_format)}
