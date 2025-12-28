from typing import Callable, Any, Dict
from dataclasses import dataclass

@dataclass
class Tool:
    name: str
    doc: str
    input_doc: Dict
    run: Callable

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
