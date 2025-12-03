from typing import Any, Callable, Dict, List, Sequence, Optional, Union, Type, cast
from .types import Tool, LLM


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
        tools: list[Tool] = [],
    ):
        self.name = name
        self.description = description
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}