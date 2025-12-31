from typing import Callable, Any, Dict, Tuple
from typing import Protocol
from dataclasses import dataclass

@dataclass
class Tool:
    name: str
    doc: str
    input_doc: Dict
    run: Callable[..., str]

NO_TOOL = Tool("", "The No Tool. This tool does not exist and does nothing when called.", {}, lambda **kwargs: "")

@dataclass
class LLM:
    model: str

@dataclass
class TaskResult:
    task: str
    result: str

@dataclass
class ChatQA:
    question: str
    response: str

class Tracer(Protocol):
    def on(self, key: str, args: Dict[str, Any]) -> "Tracer": ...
    def add_arg(self, arg_name: str, arg: Any): ...
    def reset(self): ...