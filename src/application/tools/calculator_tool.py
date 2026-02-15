from __future__ import annotations

import ast
import logging
from typing import Any, Type

from pydantic import BaseModel, Field

from src.domain.tool import BaseTool

logger = logging.getLogger(__name__)


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def _safe_eval(expression: str) -> float:
    """
    안전한 산술식 계산(MVP).
    허용: 숫자, + - * / // % **, 괄호, 단항 +/-
    금지: 함수 호출/속성 접근/변수/컨테이너 등
    """
    tree = ast.parse(expression, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARYOPS):
            value = _eval(node.operand)
            return +value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left**right
        raise ValueError("Unsupported expression")

    return _eval(tree)


class CalculatorTool(BaseTool):
    class Input(BaseModel):
        expression: str = Field(..., description="계산할 수식 (예: (12+3)*4)")

    @property
    def name(self) -> str:
        return "calculator_tool"

    @property
    def description(self) -> str:
        return "간단한 산술 수식을 계산합니다."

    @property
    def input_model(self) -> Type[BaseModel]:
        return CalculatorTool.Input

    def execute(self, expression: str, **_: Any) -> dict:
        expr = (expression or "").strip()
        if not expr:
            raise ValueError("expression must be non-empty")
        logger.info("CalculatorTool execute expression=%s", expr)
        value = _safe_eval(expr)
        return {"expression": expr, "result": value}
