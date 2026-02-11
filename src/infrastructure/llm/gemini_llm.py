from typing import Any, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.domain.llm import LLMProvider
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

    def generate(self, prompt: str, tools: Optional[List[Any]] = None) -> str:
        if tools:
            # Generic tool calling agent for Gemini
            mcp_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant with access to tools."),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # create_tool_calling_agent is more modern and supports Gemini well
            agent = create_tool_calling_agent(self.llm, tools, mcp_prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            
            logger.info(f"Invoking Gemini with tools for prompt: {prompt}")
            result = agent_executor.invoke({"input": prompt})
            return result["output"]
        else:
            logger.info(f"Invoking Gemini without tools for prompt: {prompt}")
            response = self.llm.invoke(prompt)
            return response.content

    def get_model_name(self) -> str:
        return self.model_name
