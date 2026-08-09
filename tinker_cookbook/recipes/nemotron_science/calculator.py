"""A safe in-process calculator tool for the science RL recipe.

The science questions occasionally require evaluating a physics/chemistry
formula numerically (e.g. relativistic speed from energy, a Nernst potential).
Rather than a full code sandbox, this exposes a single `calculator` tool that
evaluates one arithmetic expression via a restricted AST walk: only numeric
literals, arithmetic/comparison operators, and a whitelist of ``math`` functions
and constants are allowed. There is no attribute access, no names outside the
whitelist, no imports, no calls to anything but the whitelisted functions, so
arbitrary-code execution is not possible.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Annotated

from tinker_cookbook.tool_use import ToolResult, simple_tool_result, tool

# Whitelisted binary / unary operators.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Whitelisted names: math functions + constants (callables and floats only).
_ALLOWED_NAMES: dict[str, object] = {
    name: getattr(math, name)
    for name in (
        "sqrt", "exp", "log", "log10", "log2", "sin", "cos", "tan",
        "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh",
        "degrees", "radians", "factorial", "gcd", "hypot", "fabs",
        "floor", "ceil", "pi", "e", "tau", "inf",
    )
}
_ALLOWED_NAMES.update({"abs": abs, "round": round, "min": min, "max": max})


class _SafeEval(ast.NodeVisitor):
    """Evaluate an arithmetic expression AST, rejecting anything unlisted."""

    def visit(self, node: ast.AST):
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            raise ValueError(f"disallowed expression element: {type(node).__name__}")
        return method(node)

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numeric constants are allowed")

    def visit_BinOp(self, node: ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"disallowed operator: {type(node.op).__name__}")
        return op(self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"disallowed unary operator: {type(node.op).__name__}")
        return op(self.visit(node.operand))

    def visit_Name(self, node: ast.Name):
        if node.id in _ALLOWED_NAMES:
            val = _ALLOWED_NAMES[node.id]
            if callable(val):
                raise ValueError(f"'{node.id}' is a function; call it, don't reference it")
            return val
        raise ValueError(f"unknown name: {node.id}")

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("only direct calls to whitelisted functions are allowed")
        fn = _ALLOWED_NAMES.get(node.func.id)
        if fn is None or not callable(fn):
            raise ValueError(f"unknown function: {node.func.id}")
        if node.keywords:
            raise ValueError("keyword arguments are not allowed")
        return fn(*[self.visit(a) for a in node.args])


def safe_eval(expr: str) -> float:
    """Evaluate a single arithmetic expression safely. Raises ValueError on
    anything outside the whitelist."""
    tree = ast.parse(expr, mode="eval")
    result = _SafeEval().visit(tree)
    if not isinstance(result, (int, float)):
        raise ValueError("expression did not evaluate to a number")
    return float(result)


class Calculator:
    """Holds the `calculator` tool. Tracks invocation count for observability."""

    def __init__(self) -> None:
        self.call_count = 0

    @tool
    async def calculator(
        self,
        expression: Annotated[
            str,
            "A single arithmetic expression to evaluate, e.g. "
            "'sqrt(2*1.6e-19*1000/9.11e-31)'. Supports + - * / ** % and math "
            "functions (sqrt, exp, log, sin, ...) and constants (pi, e).",
        ],
    ) -> ToolResult:
        """Evaluate a numeric arithmetic expression and return the result."""
        self.call_count += 1
        try:
            value = safe_eval(expression)
        except Exception as e:
            return simple_tool_result(f"[calculator error: {e}]")
        return simple_tool_result(f"{expression} = {value}")
