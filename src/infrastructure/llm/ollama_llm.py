from __future__ import annotations

import json
from typing import Any, List, Optional

from src.domain.llm import LLMProvider
from src.domain.tool import BaseTool
from src.infrastructure.langchain.tool_adapter import to_langchain_tools
from src.infrastructure.logger import logger


class OllamaLLM(LLMProvider):
    """
    Ollama adapter using LangChain's ChatOllama.

    Notes:
    - Uses Ollama native endpoints (/api/chat) via langchain-ollama.
    - Tool-calling quality depends on the chosen model.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model_name: str = "lfm2.5-thinking:1.2b",
        temperature: float = 0,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url

        # Lazy import to avoid hard dependency during unit tests/environments.
        try:
            from langchain_ollama import ChatOllama  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "langchain-ollama is required for LLM_BACKEND=ollama. "
                "Install it with `pip install langchain-ollama`."
            ) from e

        self.llm = ChatOllama(model=model_name, base_url=base_url, temperature=temperature)

    def generate(
        self,
        user_input: str,
        *,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        if tools:
            from langchain.agents import AgentExecutor, create_tool_calling_agent
            from langchain_core.callbacks.base import BaseCallbackHandler
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

            lc_tools = to_langchain_tools(tools)
            mcp_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt or "You are a helpful assistant with access to tools."),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

            audit_tool = next((t for t in tools if t.name == "event_db_tool"), None)

            def _jsonable(value: Any) -> Any:
                try:
                    json.dumps(value, ensure_ascii=False)
                    return value
                except Exception:
                    return str(value)

            def _audit(kind: str, payload: dict) -> None:
                if not audit_tool:
                    return
                try:
                    audit_tool.execute(action="insert", kind=kind, payload=payload)
                except Exception as e:
                    logger.warning("Audit insert failed: %s", e)

            class ToolLoggingHandler(BaseCallbackHandler):
                def __init__(self) -> None:
                    self._last_tool: Optional[str] = None
                    self._last_args: Any = None

                def on_agent_action(self, action: Any, **kwargs: Any) -> Any:
                    self._last_tool = getattr(action, "tool", None)
                    self._last_args = getattr(action, "tool_input", None)
                    logger.info(
                        "Step [Agent Action]: LLM decided to use tool '%s' with input: %s",
                        self._last_tool,
                        self._last_args,
                    )

                def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
                    logger.info("Step [Tool Result]: Tool returned: %s", output)
                    if self._last_tool and self._last_tool != "event_db_tool":
                        _audit(
                            kind="tool_call",
                            payload={
                                "query": user_input,
                                "tool": self._last_tool,
                                "arguments": _jsonable(self._last_args),
                                "result": _jsonable(output),
                            },
                        )

                def on_agent_finish(self, finish: Any, **kwargs: Any) -> Any:
                    logger.info("Step [Agent Final Answer]: %s", finish.return_values["output"])
                    _audit(
                        kind="final",
                        payload={
                            "query": user_input,
                            "output": _jsonable(finish.return_values["output"]),
                        },
                    )

            agent = create_tool_calling_agent(self.llm, lc_tools, mcp_prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=lc_tools,
                verbose=True,
                callbacks=[ToolLoggingHandler()],
            )

            logger.info("--- Starting Agent Execution Flow for: %s ---", user_input)
            result = agent_executor.invoke({"input": user_input})
            logger.info("--- Finished Agent Execution Flow ---")
            return result["output"]

        response = self.llm.invoke(user_input)
        return getattr(response, "content", str(response))

    def get_model_name(self) -> str:
        return self.model_name
