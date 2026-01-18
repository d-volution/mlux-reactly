from dataclasses import dataclass, asdict, is_dataclass, field
import traceback
from typing import Any, Dict, List, Optional
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
    nr: int
    endtime: datetime|None = None

    def as_dict(self) -> Dict:
        return asdict(self)

@dataclass(frozen=True)
class FormatConfig:
    colored: bool = True
    compact: bool = True
    show: Dict[str, bool] = field(default_factory=dict)
    show_other: bool = True


@dataclass
class TraceConfig:
    session: str
    record_file: TextIOWrapper|None = None
    live_format: FormatConfig = FormatConfig(show_other=False)

known_init_keys = ['stage', 'task', 'llmcall', 'toolrun']


########################



def make_json_serializable(data: Any):
    if type(data) == list:
        return [make_json_serializable(element) for element in data]
    if isinstance(data, dict):
        return {k: make_json_serializable(v) for k, v in data.items()}
    if is_dataclass(data) and not isinstance(data, type):
        if isinstance(data, Tool):
            return make_json_serializable(data.name)
        return make_json_serializable(vars(data))
    if type(data) in [str, int, float, bool]:
        return data
    if hasattr(data, "tolist"):
        return make_json_serializable(data.tolist())
    return str(data)

def format_json_line(data: Any):
    if data is None:
        return ""
    return json.dumps(make_json_serializable(data), ensure_ascii=False)


def format_failed_event_msg(event: Event) -> str:
    reason_code = event.args.get('reason_code', '')
    error_msg = str(event.args.get('exception', ''))
    error_type = str(type(event.args.get('exception', None)))
    tries = str(event.args.get('tries', ''))

    parts: List[str] = [
        reason_code,
        ': ' if reason_code and error_msg else '',
        error_msg,
        f", {tries} tries" if tries else '',
        f", errtype {error_type}" if error_msg else ''
    ]
    return ''.join(parts)# + f"\n{"".join(traceback.format_exception(event.args.get('exception', ValueError('no exception'))))}"


def _format_text(prefix: str, text: str) -> str:
    return f"{prefix}{text.replace('\n', '\n'+prefix)}\n"

def format_event(event: Event, *, level: int = 0, format_config: FormatConfig = FormatConfig()) -> str:
    lines = []
    key = event.key

    if not (format_config.show_other or format_config.show.get(key, False)):
        return

    RESET = '\033[0m'
    NCOLOR = '\033[33m' if format_config.colored else ''
    ERRCOLOR = '\033[31m' if format_config.colored else ''
    arg_nr = event.args.get('nr', -100)

    headline = f"{"  "*level}* {key}"
    details = ''
    if key == 'query':
        headline += f" {NCOLOR}{format_json_line(event.args.get('user_question'))}{RESET}"
    elif key == 'task':
        headline += f" {NCOLOR}{format_json_line(event.args.get('task'))}{RESET}"
    elif key == 'stage':
        headline += f" {NCOLOR}'{event.args.get('name', '')}'{RESET} => {format_json_line(event.args.get('result'))}"
    elif key == 'toolrun':
        tool: Tool = event.args.get('tool') or Tool()
        headline += f" {NCOLOR}'{tool.name}'{RESET} => {format_json_line(event.args.get('result'))}"
    elif key == 'try' and arg_nr == 0:
        headline = ""
    elif key == 'try' and arg_nr != 0:
        headline = f"{ERRCOLOR}{"  "*level}* retry: {arg_nr}{RESET}"
    elif key == 'llmcall':
        if not format_config.compact:
            details += _format_text('    => ', str(event.args.get('sys_prompt', '<--- sys prompt not available --->')))
            details += _format_text('    -> ', str(event.args.get('prompt', '<--- prompt not available --->'))) + '<END of PROMPT>'
    elif key == 'complete':
        details += _format_text('    -> ', str(event.args.get('result', '<--- sys prompt not available --->')))
    elif key == 'failed':
        headline = f"{ERRCOLOR}{headline}: {format_failed_event_msg(event)}{RESET}"

    if headline:
        lines.append(f"{(str(event.nr)+':').ljust(4)} {headline}")
    if details:
        lines.append(details)

    for ev in event.sub:
        nextlevel = level+1
        if ev.key in ['complete', 'llmcall'] and not (format_config.show.get(ev.key, False) or not format_config.compact):
            continue
        if ev.key in ['try']:
            nextlevel=level
        lines.append(format_event(ev, level=nextlevel, format_config=format_config))

    if key == 'query':
        lines.append(f"{''.ljust(4)}{"  "*level} * query answer: {format_json_line(event.args.get('result'))}")
    return "\n".join([line for line in lines if line != ""])
    



def format_tracer(tracer: Tracer, format_config: FormatConfig = FormatConfig()) -> str:
    if isinstance(tracer, TestTracer):
        event = tracer.event
        for _ in range(3):
            if event.key == 'root' and len(event.sub) > 0 and False:
                event = event.sub[len(event.sub)-1]
        return format_event(event, format_config=format_config)
    else:
        return ""
    

def find_event_with_nr_not_itself(event: Event, nr: int) -> Event|None:
    for sub in event.sub:
        if sub.nr == nr:
            return sub
        found = find_event_with_nr_not_itself(sub, nr)
        if found is not None:
            return found
    return None

def find_event_with_nr(event: Event, nr: int) -> Event|None:
    if event.nr == nr:
        return event
    else:
        return find_event_with_nr_not_itself(event, nr)

def format_tracer_with_nr(tracer: Tracer, nr: int, format_config: FormatConfig = FormatConfig(compact=False)) -> str:
    if isinstance(tracer, TestTracer):
        event = find_event_with_nr(tracer.event, nr)
        if event is not None:
            return format_event(event, format_config=format_config)
        return f"no such event with nr {nr}"
    return ""

##########################

class TestTracer(Tracer):
    config: TraceConfig
    event: Event
    root_tracer: 'TestTracer'
    event_count: int = 0

    def __init__(self, *, 
                 config: TraceConfig|None = None, 
                 event: Event|None = None,
                 root_tracer: Optional['TestTracer'] = None,
                 session: str|None = None,
                 record_file: TextIOWrapper|None = None):
        self.event = event or Event("root", {}, sub=[], time=datetime.now(), nr=0)
        self.config = config or TraceConfig(session=session or f"default", record_file=record_file)
        self.root_tracer = root_tracer or self

    def on(self, key: str, args: Dict[str, Any]) -> "TestTracer":
        time = datetime.now()
        event = Event(key, args, [], time, nr=self.root_tracer.event_count)
        self.root_tracer.event_count += 1
        self.event.sub.append(event)

        if key == 'complete':
            self.add_arg('result', args.get('result'))
            if self.event.key == 'query':
                self._record_to_file()

        if key == 'failed':
            self.root_tracer.event.args['flag_has_failed_event'] = True

        live_out = format_event(event, format_config=self.config.live_format)
        if live_out:
            print(live_out)

        return TestTracer(config=self.config, event=event, root_tracer=self.root_tracer)
    
    def add_arg(self, arg_name: str, arg: Any):
        self.event.args[arg_name] = arg

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