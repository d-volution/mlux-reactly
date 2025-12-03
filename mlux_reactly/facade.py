from typing import Any, Callable, Dict, List, Sequence, Optional, Union, Type, cast
from .types import Tool, LLM
from .react import run_react
from .recorder import Recorder, ZeroRecorder

class ReactlyAgent:
    name: str
    description: str
    llm: LLM
    tools: dict[str, Tool]
    recorder: Recorder

    def __init__(
        self,
        llm: LLM,
        name: str = "ReactlyAgent",
        description: str = "",
        tools: List[Tool] = [],
        recorder: Recorder = ZeroRecorder()
    ):
        self.name = name
        self.description = description
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.recorder = recorder

    def query(self, user_query: str) -> str:
        self.recorder.record_query(user_query)
        response = run_react(user_query, llm=self.llm, tools=self.tools)
        self.recorder.on_response(response)
        return response
