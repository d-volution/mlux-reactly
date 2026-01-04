from typing import Any, Annotated
import numexpr as ne # type: ignore
import math
import re
import json
from mlux_reactly import Tool

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


def calculator(
        expression: Annotated[str, """The expression to be evaluated. It has to be in numexpr format.
Example Input: "12 - sin((1+3)/2)"

Only use the following functions (replace a, b, x, y, ... by a number or numeric expression as you wish):
`sin(x)`: sine of angle x. Alternatively: 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'arctan2', 'sinh', 'cosh', 'tanh'. All functions assume x to be in radians!
`exp(x)`, `log(x)`, `log10(x)`, `log2(x)`: exponential and logarithms
`abs(x)`, `floor(x)`, `ceil(x)`, `round(x)`, `trunc(x)`
`sqrt(x)`

Only use numbers. Not variable names, like `100 + q`! Exception: 'pi' is defined.
Do never write `np.sin`, `np.sqrt`, ... in the input!"""]
) -> str:
    """This calculator tool evaluates an arithmetic expression and returns the result. It uses numexpr under the hood."""

    e = ne.validate(expression, local_dict)

    undef = [s for s in symbol_re.findall(expression) if s not in allowed_symbols]
    if len(undef) > 0:
        raise f"The following used symbols are not defined: {undef}"

    if e is not None:
        raise f"Tool input '{expression}' is invalid. {e}"
    
    result = ne.evaluate(expression, local_dict)
    return result