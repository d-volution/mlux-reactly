from typing import Any
import json
from ..types import Tool

class TextCountTool(Tool):
    _name: str
    _desc: str
    
    def __init__(self):
        self._name = "textcount"
        self._desc = """
The textcount tool counts numbers of characters, words and lines in a string.

Use the tool like this:
---
Action: textcount
Action Input: {"input": "The text/string you want to know the counts about.\nIt can go over multiple lines."}
---

The tool will respond like this:
---
Observation: {"characters": number of characters, "words": number of words, "lines": number of lines }
---
"""

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def desc(self) -> str:
        return self._desc

    def run(self, input: dict[str, Any]) -> tuple[str, bool]:
        s: str = input["input"]
        r = {
            'characters': len(s),
            'words': len(s.split()),
            'lines': len(s.splitlines())
        }
        return json.dumps(r), True