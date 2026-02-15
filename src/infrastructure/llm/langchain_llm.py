from typing import Any, List, Optional

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.domain.llm import LLMProvider
from src.domain.tool import BaseTool
from src.infrastructure.langchain.tool_adapter import to_langchain_tools

class LangChainLLM(LLMProvider):
    """
    LangChain-based implementation of LLMProvider.
    This class is located in the Infrastructure layer.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0):
        self.model_name = model_name
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

    def generate(
        self,
        user_input: str,
        *,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        In a real-world scenario, this would involve setting up an agent or a chain.
        For this MVP, we simplify the tool usage.
        """
        if tools:
            lc_tools = to_langchain_tools(tools)
            # Simple agent setup for tool calling
            mcp_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt or "You are a helpful assistant with access to tools."),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            agent = create_openai_functions_agent(self.llm, lc_tools, mcp_prompt)
            agent_executor = AgentExecutor(agent=agent, tools=lc_tools, verbose=True)
            result = agent_executor.invoke({"input": user_input})
            return result["output"]
        else:
            response = self.llm.invoke(user_input)
            return response.content

    def get_model_name(self) -> str:
        return self.model_name
