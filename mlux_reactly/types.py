from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class Role(Enum):
    System = "system"
    User = "user"
    Assistant = "assistant"

@dataclass
class Message:
    role: Role
    content: str

@dataclass
class LLM:
    model: str

class StepKind(Enum):
    ToolCall = "tool_call"
    Answer = "answer"
    Other = "other"
    SyntaxError = "syntax_error"

class Tool(ABC):
    name: str
    desc: str

    @abstractmethod
    def run(input: dict) -> tuple[str, bool]:
        pass
