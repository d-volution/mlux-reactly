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
    satisfaction: float = 0.0

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
class AgentConfig:
    """Various parameters to configure agent behaviour"""

    max_nr_tries_per_task: int = 3
    """Maximal number a single task is tried to solve/answer it"""

    tool_use_rating_threshold: float = 0.5
    """Minimum rating (score between 0 and 1) a tool needs to be rated for a specific task in order to be used for this task"""

    task_answer_satisfaction_threshold: float = 0.5
    """Minimum satisfaction score (between 0 and 1) an answer of a task must have in order to be accepted, otherwise the task will be retried."""
