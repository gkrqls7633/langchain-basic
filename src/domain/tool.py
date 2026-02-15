from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Type

from pydantic import BaseModel


class BaseTool(ABC):
    """
    Interface for all MCP Tools (Domain).
    LangChain 등 외부 프레임워크 의존성 없음.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def input_model(self) -> Type[BaseModel]:
        """
        Tool 입력 스키마(Pydantic).
        Infrastructure(LangChain/MCP adapter)에서 function-calling schema로 사용한다.
        """
        raise NotImplementedError

    @property
    def parameters(self) -> Mapping[str, Any]:
        """
        MCP tool listing / function-calling에 사용할 수 있는 JSON Schema 형태.
        """
        schema = self.input_model.model_json_schema()
        return {"type": "object", **schema}

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """
        Tool 비즈니스 로직.

        tool 내부에서 infrastructure(로거/DB 클라이언트/외부 SDK)를 직접 import 하지 않는다.
        필요 시 Port(Repository/Client interface)를 주입받아 사용한다.
        """
        raise NotImplementedError
