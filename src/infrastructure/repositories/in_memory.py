from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.domain.models import Event, Memo, TodoItem
from src.domain.repositories import EventRepository, MemoRepository, TodoRepository


class InMemoryTodoRepository(TodoRepository):
    def __init__(self) -> None:
        self._items: Dict[str, TodoItem] = {}

    def add(self, title: str) -> TodoItem:
        todo_id = str(uuid.uuid4())
        item = TodoItem(id=todo_id, title=title, is_done=False, created_at=datetime.utcnow())
        self._items[todo_id] = item
        return item

    def list(self) -> List[TodoItem]:
        return sorted(self._items.values(), key=lambda x: x.created_at)

    def mark_done(self, todo_id: str, is_done: bool = True) -> TodoItem:
        existing = self._items.get(todo_id)
        if not existing:
            raise KeyError(f"Todo not found: {todo_id}")
        updated = TodoItem(
            id=existing.id,
            title=existing.title,
            is_done=is_done,
            created_at=existing.created_at,
        )
        self._items[todo_id] = updated
        return updated

    def get(self, todo_id: str) -> Optional[TodoItem]:
        return self._items.get(todo_id)


class InMemoryMemoRepository(MemoRepository):
    def __init__(self) -> None:
        self._items: Dict[str, Memo] = {}

    def add(self, title: str, content: str) -> Memo:
        memo_id = str(uuid.uuid4())
        memo = Memo(id=memo_id, title=title, content=content, created_at=datetime.utcnow())
        self._items[memo_id] = memo
        return memo

    def list(self) -> List[Memo]:
        return sorted(self._items.values(), key=lambda x: x.created_at)

    def get(self, memo_id: str) -> Optional[Memo]:
        return self._items.get(memo_id)


class InMemoryEventRepository(EventRepository):
    def __init__(self) -> None:
        self._items: Dict[str, Event] = {}

    def insert(self, kind: str, payload: dict) -> Event:
        event_id = str(uuid.uuid4())
        evt = Event(id=event_id, kind=kind, payload=dict(payload or {}), created_at=datetime.utcnow())
        self._items[event_id] = evt
        return evt

    def list(self, limit: int = 50) -> List[Event]:
        events = sorted(self._items.values(), key=lambda x: x.created_at, reverse=True)
        return events[: max(1, int(limit))]
