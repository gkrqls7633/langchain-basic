from typing import Any, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.domain.llm import LLMProvider
from src.domain.tool import BaseTool
from src.infrastructure.langchain.tool_adapter import to_langchain_tools
from src.infrastructure.logger import logger

class GeminiLLM(LLMProvider):
    """
    LangChain-based Gemini implementation of LLMProvider.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0):
        self.model_name = model_name
        import os
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            logger.error("Google API Key not found in environment variables.")
            
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=temperature,
            google_api_key=api_key
        )

    def generate(
        self,
        user_input: str,
        *,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        if tools:
            lc_tools = to_langchain_tools(tools)
            # Generic tool calling agent for Gemini
            mcp_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt or "You are a helpful assistant with access to tools."),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # Custom logging callbacks to see detailed tool usage
            from langchain_core.callbacks import StdOutCallbackHandler
            from langchain_core.callbacks.base import BaseCallbackHandler

            class ToolLoggingHandler(BaseCallbackHandler):
                def on_agent_action(self, action: Any, **kwargs: Any) -> Any:
                    logger.info(f"Step [Agent Action]: LLM decided to use tool '{action.tool}' with input: {action.tool_input}")

                def on_tool_end(self, output: str, **kwargs: Any) -> Any:
                    logger.info(f"Step [Tool Result]: Tool returned: {output}")

                def on_agent_finish(self, finish: Any, **kwargs: Any) -> Any:
                    logger.info(f"Step [Agent Final Answer]: {finish.return_values['output']}")

            # create_tool_calling_agent is more modern and supports Gemini well
            agent = create_tool_calling_agent(self.llm, lc_tools, mcp_prompt)
            agent_executor = AgentExecutor(
                agent=agent, 
                tools=lc_tools,
                verbose=True,
                callbacks=[ToolLoggingHandler()] # Attach custom logger
            )
            
            logger.info(f"--- Starting Agent Execution Flow for: {user_input} ---")
            result = agent_executor.invoke({"input": user_input})
            logger.info(f"--- Finished Agent Execution Flow ---")
            return result["output"]
        else:
            logger.info(f"Invoking Gemini without tools for prompt: {user_input}")
            response = self.llm.invoke(user_input)
            return response.content

    def get_model_name(self) -> str:
        return self.model_name
