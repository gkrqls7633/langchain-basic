from __future__ import annotations

from typing import Any, List

from src.domain.tool import BaseTool


def to_langchain_tools(tools: List[BaseTool]) -> List[Any]:
    """
    Convert domain tools to LangChain StructuredTool.
    LangChain 의존성은 infrastructure 레이어에만 존재.
    """
    from langchain_core.tools import StructuredTool

    lc_tools: List[Any] = []
    for tool in tools:
        args_schema = tool.input_model

        def _make_func(t: BaseTool):
            def _func(**kwargs: Any) -> Any:
                return t.execute(**kwargs)

            return _func

        lc_tools.append(
            StructuredTool.from_function(
                func=_make_func(tool),
                name=tool.name,
                description=tool.description,
                args_schema=args_schema,
            )
        )

    return lc_tools
