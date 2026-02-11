from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional

class LLMProvider(ABC):
    """
    Interface for LLM models.
    This ensures that the application layer does not depend on specific LLM implementations.
    """
    
    @abstractmethod
    def generate(self, prompt: str, tools: Optional[List[Any]] = None) -> str:
        """
        Generate a response for a given prompt, optionally using tools.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the name of the underlying model.
        """
        pass
