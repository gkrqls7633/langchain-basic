from typing import List, Any
from src.domain.llm import LLMProvider
from src.infrastructure.tool_registry import ToolRegistry
from src.infrastructure.logger import logger    
class MCPService:
    """
    Application layer service that orchestrates LLM and Tools.
    This layer only depends on domain interfaces and the tool registry.
    """
    def __init__(self, llm: LLMProvider, tool_registry: ToolRegistry):
        self.llm = llm
        self.tool_registry = tool_registry

    def process_query(self, query: str) -> str:
        """
        Processes a user query by using the LLM and registered tools.
        """
        # Get tools in a format compatible with the underlying LLM implementation
        # In this case, we get LangChain tools from the registry
        tools = self.tool_registry.get_langchain_tools()
        logger.info(f"Tools: {tools}")
        
        return self.llm.generate(query, tools=tools)
