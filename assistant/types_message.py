from dataclasses import dataclass
from enum import Enum

class Role(Enum):
    User = 1
    Assistant = 2
    System = 3

@dataclass
class Message:
    role: Role
    text: str
