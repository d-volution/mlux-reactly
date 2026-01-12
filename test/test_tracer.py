from dataclasses import dataclass, asdict, is_dataclass
from typing import Any, Dict, List
from datetime import datetime
from io import StringIO, TextIOWrapper
import json
from mlux_reactly import Tracer, Tool




@dataclass
class Event:
    key: str
    args: Dict[str, Any]
    sub: List["Event"]
    time: datetime
    endtime: datetime|None = None

    def as_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TraceConfig:
    session: str
    record_file: TextIOWrapper|None = None

known_init_keys = ['stage', 'task', 'llmcall', 'toolrun']


########################

def make_json_serializable(data: Any):
    if type(data) == list:
        return [make_json_serializable(element) for element in data]
    if isinstance(data, dict):
        return {k: make_json_serializable(v) for k, v in data.items()}
    if is_dataclass(data) and not isinstance(data, type):
        return make_json_serializable(asdict(data))
    if type(data) in [str, int, float, bool]:
        return data
    if hasattr(data, "tolist"):
        return make_json_serializable(data.tolist())
    return str(data)

def format_json_line(data: Any) :
    if data is None:
        return ""
    return json.dumps(make_json_serializable(data))

@dataclass
class FormatConfig:
    colored: bool = True


def format_event_compact(event: Event, *, level: int = 0, format_config: FormatConfig = FormatConfig()) -> str:
    lines = []
    key = event.key
    nextlevel = level+1

    RESET = '\033[0m'
    NCOLOR = '\033[33m' if format_config.colored else ''
    ERRCOLOR = '\033[31m' if format_config.colored else ''
    arg_nr = event.args.get('nr', -100)

    headline = f"{"  "*level}* {key}"
    if key=='stage':
        headline += f" {NCOLOR}'{event.args.get('name', '')}'{RESET} => {format_json_line(event.args.get('result'))}"
    elif key == 'toolrun':
        tool: Tool = event.args.get('tool') or Tool()
        headline += f" {NCOLOR}'{tool.name}'{RESET} => {format_json_line(event.args.get('result'))}"
    elif key == 'try' and arg_nr == 0:
        headline = ""
    elif key == 'try' and arg_nr != 0:
        headline = f"{ERRCOLOR}{"  "*level}* retry: {arg_nr}{RESET}"
    elif key == 'failed':
        msg = event.args.get('reason_code', str(event.args.get('exception', '')))
        headline = f"{ERRCOLOR}{headline}: {msg}{RESET}"
    lines.append(headline)
    for ev in event.sub:
        if ev.key in ['complete', 'llmcall']:
            continue
        if ev.key == 'try':
            nextlevel=level
        lines.append(format_event_compact(ev, level=nextlevel))
    return "\n".join([line for line in lines if line != ""])
    




##########################

class TestTracer(Tracer):
    config: TraceConfig
    event: Event

    def __init__(self, *, 
                 config: TraceConfig|None = None, 
                 event: Event|None = None,
                 session: str|None = None,
                 record_file: TextIOWrapper|None = None):
        self.event = event or Event("root", {}, sub=[], time=datetime.now())
        self.config = config or TraceConfig(session=session or f"default", record_file=record_file)

    def on(self, key: str, args: Dict[str, Any]) -> "TestTracer":
        time = datetime.now()
        event = Event(key, args, [], time)
        self.event.sub.append(event)

        if key == 'complete':
            self.add_arg('result', args.get('result'))
            if self.event.key == 'query':
                self._record_to_file()

        return TestTracer(config=self.config, event=event)
    
    def add_arg(self, arg_name: str, arg: Any):
        self.event.args[arg_name] = arg

    def format_compact(self) -> str:
        return format_event_compact(self.event)

    def _record_to_file(self) -> None:
        if self.config.record_file is not None:
            self_as_dict = {
                'session': str(self.event.time.timestamp()) + self.config.session,
                'query': self.event.args.get('user_question', ""),
                'response': self.event.args.get("result", ""),
                'diagnostics': {}
            }
            self.config.record_file.write(json.dumps(self_as_dict) + "\n")
            self.config.record_file.flush()
            print('query recorded')