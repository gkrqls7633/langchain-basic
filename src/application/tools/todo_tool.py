from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Literal, Optional, Type

from pydantic import BaseModel, Field

from src.domain.repositories import TodoRepository
from src.domain.tool import BaseTool

logger = logging.getLogger(__name__)


class TodoToolInput(BaseModel):
    action: Literal["list", "add", "done"] = Field(..., description="list | add | done")
    title: Optional[str] = Field(default=None, description="add 시 TODO 제목")
    todo_id: Optional[str] = Field(default=None, description="done 시 TODO id")
    is_done: bool = Field(default=True, description="완료 여부(기본 true)")


class TodoTool(BaseTool):
    def __init__(self, repo: TodoRepository) -> None:
        self._repo = repo

    @property
    def name(self) -> str:
        return "todo_tool"

    @property
    def description(self) -> str:
        return "TODO를 조회/추가/완료 처리합니다."

    @property
    def input_model(self) -> Type[BaseModel]:
        return TodoToolInput

    def execute(self, action: str, title: str | None = None, todo_id: str | None = None, is_done: bool = True, **_: Any) -> Any:
        act = (action or "").strip().lower()
        logger.info("TodoTool execute action=%s", act)

        if act == "list":
            return [asdict(item) for item in self._repo.list()]

        if act == "add":
            if not title or not title.strip():
                raise ValueError("title is required for action=add")
            item = self._repo.add(title.strip())
            return asdict(item)

        if act == "done":
            if not todo_id or not todo_id.strip():
                raise ValueError("todo_id is required for action=done")
            item = self._repo.mark_done(todo_id.strip(), bool(is_done))
            return asdict(item)

        raise ValueError("Unsupported action. Use one of: list, add, done")
