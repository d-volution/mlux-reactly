from typing import Any
import numexpr as ne # type: ignore
from ..types import Tool

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

Do never write `np.sin`, `np.sqrt`, ... in the input!
"""

    def run(self, input: dict[str, Any]) -> tuple[str, bool]:
        expr = input["input"]
        result = ne.evaluate(expr)
        return result, True