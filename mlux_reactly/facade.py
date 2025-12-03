from typing import Any, Callable, Dict, List, Sequence, Optional, Union, Type, cast
from .types import Tool, LLM
from .react import run_react

class ReactlyAgent:
    name: str
    description: str
    llm: LLM
    tools: dict[str, Tool]

    def __init__(
        self,
        llm: LLM,
        name: str = "ReactlyAgent",
        description: str = "",
        tools: List[Tool] = [],
    ):
        self.name = name
        self.description = description
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}

    def query(self, user_query: str) -> str:
        return run_react(user_query, llm=self.llm, tools=self.tools)