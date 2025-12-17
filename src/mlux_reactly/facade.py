from typing import Any, Callable, Dict, List, Sequence, Optional, Union, Type, cast
from io import StringIO
from .types import Tool, LLM, BaseAgent, RunConfig
from .react import run_react
from .recorder import Recorder, ZeroRecorder
from .prompts import generate_react_prompt

class ReactlyAgent(BaseAgent):
    runConfig: RunConfig

    def __init__(
        self,
        llm: LLM,
        tools: List[Tool] = [],
        *,
        name: str = "ReactlyAgent",
        recorder: Recorder = ZeroRecorder(),
        stream: StringIO|None = None
    ):
        self.name = name
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.system_prompt = generate_react_prompt(self.tools)

        self.runConfig = RunConfig(recorder=recorder, stream=stream)

    def query(self, user_query: str) -> str:
        record = self.runConfig.recorder.record_query(user_query)
        response = run_react(user_query, self, self.runConfig)
        self.runConfig.recorder.on_response(response)
        return response
