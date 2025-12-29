from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from io import StringIO, TextIOWrapper
import json
from .types import Tracer





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

    def set_timepoint(self, key: str, timepoint: datetime):
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
        return self.args[name] or alternative


class NormalTracer(Tracer):
    event: Event
    level: int
    diagnostics: Diagnostics
    stream: StringIO|None
    record_file: TextIOWrapper|None

    def __init__(self, *, 
                 event: Event = Event("root", {}, [], datetime.now()), 
                 level: int = 0, 
                 diagnostics = Diagnostics(),
                 stream: StringIO|None = None,
                 record_file: TextIOWrapper|None = None):
        self.event = event
        self.level = level
        self.diagnostics = diagnostics
        self.stream = stream
        self.record_file = record_file

    def on(self, key: str, args: Dict[str, Any]) -> "NormalTracer":
        time = datetime.now()
        self.diagnostics.increment_counter(key)
        event = Event(key, args, [], time)
        self.event.sub.append(event)
        self._log(event)
        return NormalTracer(event=event, level=self.level+1, diagnostics=self.diagnostics, stream=self.stream, record_file=self.record_file)
    
    def add_arg(self, arg_name: str, arg: Any):
        self.event.args[arg_name] = arg

    def reset(self):
        self_as_dict = {
            'session': str(self.event.time.timestamp()) + self.event.key,
            'query': self.event.args['user_question'],
            'response': self.event.arg("answer", ""),
            'diagnostics': self.diagnostics.as_dict()
        }

        if self.record_file != None:
            self.record_file.write(json.dumps(self_as_dict) + "\n")

    def _log(self, event: Event):
        key = event.key
        args = event.args

        details = ""
        if key == "run_subtask":
            details = f"{args['task_description']}"
        elif key == "answer":
            details = f"{args['answer']}"

        print(f"{"  "*self.level}* {key}{": " if details != "" else ""}{details}", file=self.stream)



class ZeroTracer(Tracer):
    def on(self, key: str, args: Dict[str, Any]) -> "ZeroTracer":
        return self
    def reset(self):
        pass
    def add_arg(self, arg_name, arg):
        return