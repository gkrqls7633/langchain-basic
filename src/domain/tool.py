from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseTool(ABC):
    """
    Interface for all MCP Tools.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool logic."""
        pass
