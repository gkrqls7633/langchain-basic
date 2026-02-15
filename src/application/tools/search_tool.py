from __future__ import annotations

import logging
from typing import Any, List, Type

from pydantic import BaseModel, Field

from src.domain.tool import BaseTool

logger = logging.getLogger(__name__)


class SearchToolInput(BaseModel):
    query: str = Field(..., description="검색 키워드")
    limit: int = Field(default=5, ge=1, le=50, description="최대 반환 개수(기본 5)")


class SearchTool(BaseTool):
    """
    MVP: 더미 데이터 기반 키워드 검색.
    (추후 external API / DB 연동 시 repository로 교체)
    """

    def __init__(self) -> None:
        self._data = [
            {"id": 1, "title": "LangChain", "content": "LLM 애플리케이션 구축 프레임워크"},
            {"id": 2, "title": "MCP", "content": "Model Context Protocol: 툴 상호작용 표준화"},
            {"id": 3, "title": "Clean Architecture", "content": "의존성 규칙과 레이어 분리"},
        ]

    @property
    def name(self) -> str:
        return "search_tool"

    @property
    def description(self) -> str:
        return "더미 데이터에서 키워드 검색 결과를 반환합니다."

    @property
    def input_model(self) -> Type[BaseModel]:
        return SearchToolInput

    def execute(self, query: str, limit: int = 5, **_: Any) -> List[dict]:
        q = (query or "").strip().lower()
        if not q:
            raise ValueError("query must be non-empty")
        logger.info("SearchTool execute query=%s limit=%s", q, limit)
        results = [
            item
            for item in self._data
            if q in item["title"].lower() or q in item["content"].lower()
        ]
        return results[: max(1, int(limit))]
