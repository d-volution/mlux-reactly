
from typing import List
from .types_message import Message



class Context:
    def __init__(self):
        self.list: List[Message] = []

    def __iter__(self):
        return iter(self.list)

    def append(self, message: Message):
        self.list.append(message)

    def with_appended(self, message: Message):
        new_context = Context()
        new_context.list = self.list.copy()
        new_context.append(message)
        return new_context

    def last(self) -> Message:
        return self.list[-1]