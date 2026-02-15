from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Dict, Literal, Optional, Type, Union

from pydantic import BaseModel, Field

try:  # Pydantic v2
    from pydantic import field_validator  # type: ignore
    _FIELD_VALIDATOR_V2 = True
except Exception:  # Pydantic v1
    from pydantic import validator as field_validator  # type: ignore
    _FIELD_VALIDATOR_V2 = False

from src.domain.repositories import EventRepository
from src.domain.tool import BaseTool

logger = logging.getLogger(__name__)


class EventDbToolInput(BaseModel):
    action: Literal["insert", "list"] = Field(..., description="insert | list")
    kind: Optional[str] = Field(default=None, description="insert 시 이벤트 종류 (예: weather_query)")
    payload: Union[Dict[str, Any], str] = Field(
        default_factory=dict,
        description="insert 시 저장할 JSON payload (객체 또는 JSON 문자열)",
    )
    limit: int = Field(default=20, description="list 시 최대 반환 개수 (기본 20)")

    if _FIELD_VALIDATOR_V2:
        @field_validator("payload", mode="before")  # type: ignore[misc]
        @classmethod
        def _coerce_payload(cls, v: Any) -> Dict[str, Any]:
            if v is None or v == "":
                return {}
            if isinstance(v, dict):
                return v
            if isinstance(v, str):
                try:
                    parsed = json.loads(v)
                except Exception as e:
                    raise ValueError("payload must be a dict or a JSON object string") from e
                if not isinstance(parsed, dict):
                    raise ValueError("payload JSON must decode to an object/dict")
                return parsed
            raise ValueError("payload must be a dict or a JSON object string")
    else:
        @field_validator("payload", pre=True)  # type: ignore[misc]
        def _coerce_payload(cls, v: Any) -> Dict[str, Any]:
            if v is None or v == "":
                return {}
            if isinstance(v, dict):
                return v
            if isinstance(v, str):
                try:
                    parsed = json.loads(v)
                except Exception as e:
                    raise ValueError("payload must be a dict or a JSON object string") from e
                if not isinstance(parsed, dict):
                    raise ValueError("payload JSON must decode to an object/dict")
                return parsed
            raise ValueError("payload must be a dict or a JSON object string")


class EventDbTool(BaseTool):
    """
    DB에 JSON payload를 insert/list 하는 Tool.

    - 어떤 DB를 쓰는지는 repository 구현(infrastructure)이 결정
    - Tool은 repository interface만 의존
    """

    def __init__(self, repo: EventRepository) -> None:
        self._repo = repo

    @property
    def name(self) -> str:
        return "event_db_tool"

    @property
    def description(self) -> str:
        return "어떤 tools들이 호출되던지 event_db_tool은 무조건 호출합니다. 이는 DB에 이벤트(JSON)를 insert/list 합니다. (MVP: events 테이블)"

    @property
    def input_model(self) -> Type[BaseModel]:
        return EventDbToolInput

    def execute(self, action: str, **kwargs: Any) -> Any:
        args = EventDbToolInput(action=action, **kwargs)
        act = args.action
        logger.info("EventDbTool execute action=%s", act)

        if act == "insert":
            payload = dict(args.payload or {})
            kind = (args.kind or payload.get("kind") or "generic").strip()
            evt = self._repo.insert(kind=kind, payload=payload)
            return {"inserted": asdict(evt)}

        if act == "list":
            events = self._repo.list(limit=args.limit)
            return {"events": [asdict(e) for e in events]}

        raise ValueError("Unsupported action. Use one of: insert, list")
