from dataclasses import dataclass, asdict, is_dataclass, field
import traceback
from typing import Any, Dict, List, Optional
from datetime import datetime
from io import StringIO, TextIOWrapper
import json
from mlux_reactly import Tracer, Tool


SHOW_ALL = '_all'

@dataclass
class Event:
    key: str
    args: Dict[str, Any]
    sub: List["Event"]
    time: datetime
    nr: int
    level: int = 0
    endtime: datetime|None = None

    def as_dict(self) -> Dict:
        return asdict(self)

@dataclass(frozen=True)
class FormatConfig:
    colored: bool = True
    compact: bool = True

    # a list of all event keys to show. Events with keys not in this list will be hidden, except if SHOW_ALL is in list 
    show: List[str] = field(default_factory=lambda: [SHOW_ALL])


@dataclass
class TraceConfig:
    session: str
    record_file: TextIOWrapper|None = None
    live_format: FormatConfig = FormatConfig(show=[])

LIVE_VERBOSE = FormatConfig(show=['query', 'stage', 'task', 'toolrun', 'llmcall', 'failed', 'result'])

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

def format_json_line(data: Any, keep_length_max: int = 500):
    if data is None:
        return ""
    j = json.dumps(make_json_serializable(data), ensure_ascii=False)
    half_length = int(keep_length_max/2)
    if len(j) > keep_length_max:
        return f"{j[:half_length]}.....{j[len(j)-half_length]:}"
    else:
        return j


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
    return ''.join(parts)


def _format_text_manyline(prefix: str, text: str) -> str:
    return f"{prefix}{text.replace('\n', '\n'+prefix)}\n"

def _format_text_fewline(line_prefix: str, text: str) -> str:
    return f"{text.replace('\n', '\n'+line_prefix)}"

def should_show(key: str, show_list: List[str]) -> bool:
    return (SHOW_ALL in show_list or key in show_list)

def format_event(event: Event, *, level: int|None = None, format_config: FormatConfig = FormatConfig(show=[SHOW_ALL]), parent_key: str = "") -> str:
    lines = []
    key = event.key
    level = level if level is not None else event.level

    if not should_show(key, format_config.show) or (key == 'result' and not should_show(parent_key, format_config.show)):
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
            details += _format_text_manyline('    => ', str(event.args.get('sys_prompt', '<--- sys prompt not available --->')))
            details += _format_text_manyline('    -> ', str(event.args.get('prompt', '<--- prompt not available --->')))
    elif key == 'result':
        headline += ': ' + _format_text_fewline('    -> ', str(event.args.get('result', '<--- prompt not available --->')))
    elif key == 'failed':
        headline = f"{ERRCOLOR}{headline}: {format_failed_event_msg(event)}{RESET}"

    if headline:
        lines.append(f"{(str(event.nr)+':').ljust(4)} {headline}")
    if details:
        lines.append(details)

    for ev in event.sub:
        nextlevel = level+1
        if ev.key in ['result', 'llmcall'] and not (ev.key in format_config.show or not format_config.compact):
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
    _prevent_print_live: bool = False

    def __init__(self, *, 
                 config: TraceConfig|None = None, 
                 event: Event|None = None,
                 root_tracer: Optional['TestTracer'] = None,
                 session: str|None = None,
                 record_file: TextIOWrapper|None = None,
                 live_format: FormatConfig = FormatConfig(show=[]),
                 _prevent_print_live: bool = False):
        self.event = event or Event("root", {}, sub=[], time=datetime.now(), nr=0)
        self.config = config or TraceConfig(session=session or f"default", record_file=record_file, live_format=live_format)
        self.root_tracer = root_tracer or self
        self._prevent_print_live = _prevent_print_live

    def on(self, key: str, args: Dict[str, Any]) -> "TestTracer":
        time = datetime.now()
        event = Event(key, args, [], time, nr=self.root_tracer.event_count, level=self.event.level+1)
        self.root_tracer.event_count += 1
        self.event.sub.append(event)

        if key == 'result':
            self.add_arg('result', args.get('result'))
            if self.event.key == 'query':
                self._record_to_file()

        if key == 'failed':
            self.root_tracer.event.args['flag_has_failed_event'] = True

        if True or not self._prevent_print_live:
            live_out = format_event(event, format_config=self.config.live_format, parent_key=self.event.key)
            if live_out:
                print(live_out)

        return TestTracer(config=self.config, event=event, root_tracer=self.root_tracer, 
                          _prevent_print_live=self._prevent_print_live or not should_show(key, self.config.live_format.show))
    
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