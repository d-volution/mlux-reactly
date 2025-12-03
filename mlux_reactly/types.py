from typing import Any
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
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def desc(self) -> str:
        pass

    @abstractmethod
    def run(self, input: dict[str, Any]) -> tuple[str, bool]:
        pass


class DiagnosticHandler(ABC):
    @abstractmethod
    def event(self, name: str):
        pass
