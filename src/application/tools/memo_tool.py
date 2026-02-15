from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Literal, Optional, Type

from pydantic import BaseModel, Field

from src.domain.repositories import MemoRepository
from src.domain.tool import BaseTool

logger = logging.getLogger(__name__)


class MemoToolInput(BaseModel):
    action: Literal["list", "add", "get"] = Field(..., description="list | add | get")
    title: Optional[str] = Field(default=None, description="add 시 메모 제목")
    content: Optional[str] = Field(default=None, description="add 시 메모 내용")
    memo_id: Optional[str] = Field(default=None, description="get 시 memo id")


class MemoTool(BaseTool):
    def __init__(self, repo: MemoRepository) -> None:
        self._repo = repo

    @property
    def name(self) -> str:
        return "memo_tool"

    @property
    def description(self) -> str:
        return "메모를 저장/조회합니다."

    @property
    def input_model(self) -> Type[BaseModel]:
        return MemoToolInput

    def execute(
        self,
        action: str,
        title: str | None = None,
        content: str | None = None,
        memo_id: str | None = None,
        **_: Any,
    ) -> Any:
        act = (action or "").strip().lower()
        logger.info("MemoTool execute action=%s", act)

        if act == "list":
            return [asdict(memo) for memo in self._repo.list()]

        if act == "add":
            if not title or not title.strip():
                raise ValueError("title is required for action=add")
            if content is None or not str(content).strip():
                raise ValueError("content is required for action=add")
            memo = self._repo.add(title.strip(), str(content))
            return asdict(memo)

        if act == "get":
            if not memo_id or not memo_id.strip():
                raise ValueError("memo_id is required for action=get")
            memo = self._repo.get(memo_id.strip())
            if not memo:
                raise KeyError(f"memo not found: {memo_id.strip()}")
            return asdict(memo)

        raise ValueError("Unsupported action. Use one of: list, add, get")
