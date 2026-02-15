from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from src.domain.tool import BaseTool


def to_langchain_tools(tools: List[BaseTool]) -> List[Any]:
    """
    Convert domain tools to LangChain StructuredTool.
    LangChain 의존성은 infrastructure 레이어에만 존재.
    """
    from pydantic import BaseModel, Field, create_model
    from langchain_core.tools import StructuredTool

    def _json_schema_type_to_python(schema: Dict[str, Any]) -> type:
        t = schema.get("type")
        if t == "string":
            return str
        if t == "integer":
            return int
        if t == "number":
            return float
        if t == "boolean":
            return bool
        if t == "array":
            items = schema.get("items") or {}
            if items.get("type") == "string":
                return list[str]
            return list
        return Any

    def _build_args_schema(tool: BaseTool) -> Optional[Type[BaseModel]]:
        params = dict(tool.parameters or {})
        if not params:
            return None
        if params.get("type") != "object":
            return None

        properties = params.get("properties") or {}
        required = set(params.get("required") or [])
        if not properties:
            return None

        fields: Dict[str, tuple] = {}
        for name, prop_schema in properties.items():
            python_type = _json_schema_type_to_python(prop_schema)
            description = prop_schema.get("description") or ""
            default_value = ... if name in required else None
            fields[name] = (python_type, Field(default=default_value, description=description))

        return create_model(f"{tool.name}_Args", **fields)  # type: ignore[arg-type]

    lc_tools: List[Any] = []
    for tool in tools:
        # Prefer explicit Pydantic input model if tool provides it (권장)
        args_schema = getattr(tool, "input_model", None)
        if args_schema is None:
            args_schema = _build_args_schema(tool)

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
