from typing import Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from .diagnostics import Diagnostics
from .recorder import Recorder

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
    _name: str
    _description: str

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @abstractmethod
    def run(self, input: dict[str, Any]) -> tuple[str, bool]:
        pass


class DiagnosticHandler(ABC):
    @abstractmethod
    def event(self, name: str):
        pass

@dataclass
class BaseAgent:
    agent_name: str
    llm: LLM
    system_prompt: str
    tools: dict[str, Tool]

@dataclass
class RunConfig:
    diagnostics: Diagnostics = Diagnostics()
    stream: StringIO|None = None
    recorder: Recorder = Recorder