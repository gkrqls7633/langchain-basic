from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from src.domain.tool import BaseTool

class LLMProvider(ABC):
    """
    Interface for LLM models.
    This ensures that the application layer does not depend on specific LLM implementations.
    """
    
    @abstractmethod
    def generate(
        self,
        user_input: str,
        *,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a response for a given input, optionally using tools.

        - tools: domain tools (LangChain 변환은 infrastructure에서 처리)
        - system_prompt: tool 사용을 강제하는 정책 프롬프트 등
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the name of the underlying model.
        """
        pass
