from __future__ import annotations

import json
from typing import List, Optional

from src.domain.llm import LLMProvider
from src.domain.tool import BaseTool


class FakeLLM(LLMProvider):
    """
    API Key 없이도 아키텍처를 검증하기 위한 로컬용 LLM 스텁.

    - 실제 LLM처럼 "tool을 선택 -> 실행 -> 결과 기반 응답" 흐름을 흉내냄
    - 프로덕션에서는 LangChain 기반 구현으로 교체
    """

    def __init__(self, model_name: str = "fake-llm") -> None:
        self.model_name = model_name

    def get_model_name(self) -> str:
        return self.model_name

    def generate(
        self,
        user_input: str,
        *,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        tools_by_name = {t.name: t for t in (tools or [])}
        text = (user_input or "").strip().lower()

        def _call(name: str, **kwargs):
            tool = tools_by_name.get(name)
            if not tool:
                return {"error": f"tool not available: {name}"}
            return tool.execute(**kwargs)

        # 매우 단순한 규칙 기반 라우팅 (MVP 데모용)
        if any(k in text for k in ["시간", "time", "date", "날짜"]):
            result = _call("time_tool", format="%Y-%m-%d %H:%M:%S")
            return f"[time_tool 결과] {result}"

        if any(k in text for k in ["todo", "할일", "할 일", "작업"]):
            if any(k in text for k in ["추가", "add"]):
                title = user_input.split(":", 1)[-1].strip() if ":" in user_input else "새 TODO"
                result = _call("todo_tool", action="add", title=title)
                return f"[todo_tool 결과]\n{json.dumps(result, ensure_ascii=False)}"
            result = _call("todo_tool", action="list")
            return f"[todo_tool 결과]\n{json.dumps(result, ensure_ascii=False)}"

        if any(k in text for k in ["메모", "memo", "note"]):
            if any(k in text for k in ["저장", "add"]):
                result = _call("memo_tool", action="add", title="메모", content=user_input)
                return f"[memo_tool 결과]\n{json.dumps(result, ensure_ascii=False)}"
            result = _call("memo_tool", action="list")
            return f"[memo_tool 결과]\n{json.dumps(result, ensure_ascii=False)}"

        if any(k in text for k in ["계산", "calc", "calculator"]):
            expr = user_input.split(":", 1)[-1].strip() if ":" in user_input else user_input
            result = _call("calculator_tool", expression=expr)
            return f"[calculator_tool 결과]\n{json.dumps(result, ensure_ascii=False)}"

        if any(k in text for k in ["검색", "search"]):
            q = user_input.split(":", 1)[-1].strip() if ":" in user_input else user_input
            result = _call("search_tool", query=q, limit=5)
            return f"[search_tool 결과]\n{json.dumps(result, ensure_ascii=False)}"

        # 기본: 검색으로 처리
        result = _call("search_tool", query=user_input, limit=5)
        return f"[search_tool 결과]\n{json.dumps(result, ensure_ascii=False)}"
