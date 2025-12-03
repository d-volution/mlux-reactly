from typing import Any
import numexpr as ne
from ..types import Tool

class CalculatorTool(Tool):
    _name: str
    _desc: str
    
    def __init__(self):
        self._name = "calculator"
        self._desc = """
This calculator tool evaluates an arithmetic expression and returns the result. It takes as input a JSON-encoded object and returns the result as a number. It uses numexpr.

Us the tool the following way:
---
Action: calculator
Action Input: {{ "input": "your expression to be evaluated by numexpr" }}
---

As an example:
---
Action: calculator
Action Input: {{ "input": "sin(12.345)-2**3" }}
---

Do never write `np.sin`, `np.sqrt`, ... in the input!
"""

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def desc(self) -> str:
        return self._desc

    def run(self, input: dict[str, Any]) -> tuple[str, bool]:
        expr = input["input"]
        result = ne.evaluate(expr)
        return result, True