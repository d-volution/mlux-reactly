from typing import List, Protocol, Callable, Any, TypeVar
from dataclasses import dataclass
from mlux_reactly import LLM, Tool

class Agent(Protocol):
    def query(self, user_question: str) -> str: ...

AgentContructor = Callable[..., Agent]



TestFunc = Callable[[str, AgentContructor, LLM], Any]


T = TypeVar("T")
def at_or(l: List[T], index: int, default_value: T) -> T:
    return default_value if len(l) <= index or not l[index] else l[index]

def as_list(x: T|List[T]) -> List[T]:
    return x if isinstance(x, list) else [x]


@dataclass
class Example:
    id: str
    question: str
    answer: str


@dataclass
class AgentConfig:
    tools: List[Tool]


@dataclass
class ExampleCase:
    example: Example
    agent_config: AgentConfig
