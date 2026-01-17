from typing import Callable, Any, Dict, Tuple, List
from typing import Protocol, TypeAlias, TypeVar
from dataclasses import dataclass, field

_UNUSED_sentinel = object()
AnyBasic: TypeAlias = None | str | int | float | bool | list['AnyBasic'] | tuple[str, 'AnyBasic']
T = TypeVar('T')

def at(l: List[T], index, default: T|None = None) -> T|None:
    return l[index] if -len(l) <= index < len(l) else default


@dataclass
class Tool:
    name: str = ''
    doc: str = 'This tool does not exist and does nothing when called'
    input_doc: Dict = field(default_factory=dict)
    run: Callable[..., Any] = lambda **kwargs: ""

NO_TOOL = Tool("", "The No Tool. This tool does not exist and does nothing when called.", {}, lambda **kwargs: "")

@dataclass
class LLM:
    model: str

@dataclass
class Task:
    description: str = ""

@dataclass
class TaskResult:
    task: str = ""
    result: str = ""

@dataclass
class Answer:
    answer: str
    satisfaction: float = 0.0
    reason: str = ''

@dataclass
class ChatQA:
    question: str
    response: str

class Tracer(Protocol):
    def on(self, key: str, args: Dict[str, Any]) -> "Tracer": ...
    def add_arg(self, arg_name: str, arg: Any): ...

class ZeroTracer(Tracer):
    def on(self, key: str, args: Dict[str, Any]) -> "ZeroTracer":
        return self
    def add_arg(self, arg_name, arg):
        return
    
@dataclass
class Ctx:
    llm: LLM
    tracer: Tracer = ZeroTracer()