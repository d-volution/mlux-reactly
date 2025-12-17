import traceback
from typing import Any
import numexpr as ne # type: ignore
import math
import re
from ..types import Tool

symbol_re = re.compile("[a-zA-Z_][a-zA-Z_0-9]*")
allowed_symbols = [
    "pi",
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2",
    "sinh", "cosh", "tanh",
    "exp", "log", "log10", "log2",
    "abs", "floor", "ceil", "round", "trunc",
    "sqrt"
]
local_dict = {"pi": math.pi}

class CalculatorTool(Tool):
    def __init__(self):
        self._name = "calculator"
        self._description = """
This calculator tool evaluates an arithmetic expression and returns the result. It takes as input a JSON-encoded object and returns the result as a number. It uses numexpr.

Us the tool the following way:
---
Action: calculator
Action Input: { "input": "your expression to be evaluated by numexpr" }
---

As an example:
---
Action: calculator
Action Input: {"input": "sin(12.345)-2**3"}
---

Functions that can be used in expression (Replace a, b, x, y, ... by a number or numeric expression as you wish):
`sin(x)`: sine of angle x. Alternatively: 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'arctan2', 'sinh', 'cosh', 'tanh'. All functions assume x to be in radians!
`exp(x)`, `log(x)`, `log10(x)`, `log2(x)`: exponential and logarithms
`abs(x)`, `floor(x)`, `ceil(x)`, `round(x)`, `trunc(x)`
`sqrt(x)`

Only use numbers. Not variable names, like `sqrt(100 + q)`! Exception: 'pi' is defined.
Don't use any functions not named here before!
Do never write `np.sin`, `np.sqrt`, ... in the input!
"""

    def run(self, input: dict[str, Any]) -> tuple[str, bool]:
        expr = input["input"]
        e = ne.validate(expr, local_dict)

        undef = [s for s in symbol_re.findall(expr) if s not in allowed_symbols]
        if len(undef) > 0:
            return f"The following used symbols are not defined: {undef}", False

        if e is not None:
            return f"Tool input '{expr}' is invalid. {e}", False
        
        result = ne.evaluate(expr, local_dict)
        return result, True