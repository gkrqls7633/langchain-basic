from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from src.domain.models import Event, Memo, TodoItem


class TodoRepository(ABC):
    @abstractmethod
    def add(self, title: str) -> TodoItem: ...

    @abstractmethod
    def list(self) -> List[TodoItem]: ...

    @abstractmethod
    def mark_done(self, todo_id: str, is_done: bool = True) -> TodoItem: ...

    @abstractmethod
    def get(self, todo_id: str) -> Optional[TodoItem]: ...


class MemoRepository(ABC):
    @abstractmethod
    def add(self, title: str, content: str) -> Memo: ...

    @abstractmethod
    def list(self) -> List[Memo]: ...

    @abstractmethod
    def get(self, memo_id: str) -> Optional[Memo]: ...


class EventRepository(ABC):
    @abstractmethod
    def insert(self, kind: str, payload: dict) -> Event: ...

    @abstractmethod
    def list(self, limit: int = 50) -> List[Event]: ...
