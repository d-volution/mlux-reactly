from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from io import StringIO, TextIOWrapper
import json
from .types import Tracer, Tool





class Diagnostics:
    counters: dict[str, int]
    timepoints: dict[str, float]

    def __init__(self):
        self.counters = {}
        self.timepoints = {}

    def as_dict(self):
        return {
            'counters': self.counters,
            'timepoints': self.timepoints
        }
    
    def get_timepoint(self, key) -> datetime|None:
        if not key in self.timepoints:
            return None
        else:
            return datetime.fromtimestamp(self.timepoints[key])

    def set_timepoint(self, key: str, timepoint: datetime = datetime.now()):
        self.timepoints[key] = timepoint.timestamp()
    
    def increment_counter(self, key: str):
        if key not in self.counters:
            self.counters[key] = 0
        self.counters[key] += 1

    def on_event(event_key: str):
        pass


@dataclass
class Event:
    key: str
    args: Dict[str, Any]
    sub: List["Event"]
    time: datetime

    def as_dict(self) -> Dict:
        return asdict(self)

    def arg(self, name: str, alternative: Any = None) -> Any:
        return self.args.get(name, alternative)


class NormalTracer(Tracer):
    event: Event
    level: int
    diagnostics: Diagnostics
    stream: StringIO|None
    record_file: TextIOWrapper|None

    def __init__(self, *, 
                 name: str|None = None,
                 event: Event = Event("", {}, [], datetime.now()), 
                 level: int = 0, 
                 diagnostics = Diagnostics(),
                 stream: StringIO|None = None,
                 record_file: TextIOWrapper|None = None):
        self.event = event
        if name is not None:
            self.event.key = name
        self.level = level
        self.diagnostics = diagnostics
        self.stream = stream
        self.record_file = record_file

    def on(self, key: str, args: Dict[str, Any]) -> "NormalTracer":
        time = datetime.now()
        self.diagnostics.increment_counter(key)
        if key == "run_query":
            args['session_name'] = self.event.key
        event = Event(key, args, [], time)
        self.event.sub.append(event)
        self._log(event)
        return NormalTracer(event=event, level=self.level+1, diagnostics=self.diagnostics, stream=self.stream, record_file=self.record_file)
    
    def add_arg(self, arg_name: str, arg: Any):
        self.event.args[arg_name] = arg

    def reset(self):
        self.diagnostics.set_timepoint("finished")
        self_as_dict = {
            'session': str(self.event.time.timestamp()) + self.event.arg('session_name', "session"),
            'query': self.event.arg('user_question', ""),
            'response': self.event.arg("agent_response", ""),
            'diagnostics': self.diagnostics.as_dict()
        }

        if self.record_file != None:
            self.record_file.write(json.dumps(self_as_dict) + "\n")

    def _log(self, event: Event):
        key = event.key
        args = event.args

        NONE_TOOL = Tool("None", "None", {}, lambda *a, **k: None)

        details = ""
        if key == "run_query":
            self.diagnostics.set_timepoint("run_query")
        elif key == "run_subtask":
            details = f"{args['task_description']}"
        elif key == "choose_tool":
            as_dict = {tool.name: tool.doc for tool in (tool or NONE_TOOL for tool in event.arg("tools"))}
            details = f"tools: {json.dumps(as_dict)}"
        elif key == "answer":
            details = f"{args['answer']}"
        elif key in ["result", "stage_result", "tool_result", "tool_failure"]:
            details = str(event.arg("result"))[:200]
        elif key == "run_tool":
            tool: Tool = event.arg("tool", Tool("None", "None", {}, lambda *args, **kwargs: None))
            details = tool.name + " " + json.dumps(event.arg("input"))

        if key.startswith('stage_run_'):
            """--"""
            if key in ["stage_run_"]:
                details = f"\n\nprompt:\n----------\n{event.arg('sys_prompt', "<<- tracer could not get system prompt ->>")}{event.arg('conversation', "<<- tracer could not get conversation ->>")}\n----------"

        print(f"{"  "*self.level}* {key}{": " if details != "" else ""}{details}", file=self.stream)




    

def tracer_initialize_on_query(agent_tracer: Tracer, user_question: str) -> Tracer:
    tracer = agent_tracer.on("run_query", {
        'user_question': user_question
    })
    return tracer

def tracer_finish_with_response(tracer: Tracer, agent_response: str):
    tracer.add_arg("agent_response", agent_response)
    tracer.reset()

