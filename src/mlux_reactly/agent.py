from typing import Any, List, Dict
from typing import Callable
from typing import get_origin, get_args, Annotated
import inspect
from io import StringIO
from .types import LLM, Tool, ChatQA, Tracer, ZeroTracer, AgentConfig
from .core import run_query


def describe(fn):
    sig = inspect.signature(fn)
    out = {}

    for name, p in sig.parameters.items():
        t = p.annotation
        desc = ""

        if get_origin(t) is Annotated:
            base, *meta = get_args(t)
            t = base
            desc = meta[0] if meta else ""

        if get_origin(t) is list:
            inner = get_args(t)[0]
            out[name] = [f"{inner.__name__}: {desc}"]
        else:
            if desc == "":
                out[name] = t.__name__
            else:
                out[name] = f"{t.__name__}: {desc}"

    return out


def tool_from_function(tool_fn: Tool | Callable) -> Tool:
    if isinstance(tool_fn, Tool):
        return tool_fn
    else:
        return Tool(tool_fn.__name__, tool_fn.__doc__, describe(tool_fn), tool_fn)


class ReactlyAgent:
    def __init__(
            self, 
            tools: List[Tool | Callable], *, 
            tracer: Tracer = ZeroTracer(),
            llm = LLM("qwen2.5:7b-instruct-q8_0"),
            config: AgentConfig = AgentConfig()):
        self.tools = [tool_from_function(tool_fn) for tool_fn in tools]
        self.llm = llm
        self.tracer = tracer
        self.history: List[ChatQA] = []
        self.config = config

    def query(self, user_question: str) -> str:
        response = run_query(user_question, self.history, self.tools, self.llm, self.tracer, self.config)
        self.history.append(ChatQA(user_question, response))
        return response

