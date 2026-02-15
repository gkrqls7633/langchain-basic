from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class TodoItem:
    id: str
    title: str
    is_done: bool
    created_at: datetime


@dataclass(frozen=True)
class Memo:
    id: str
    title: str
    content: str
    created_at: datetime
