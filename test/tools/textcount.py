from typing import Any, Annotated, Dict
import json


def text_count(text: Annotated[str, """The text string to be counted"""]) -> Dict[str, int]:
    """The text_count tool counts numbers of characters, words and lines in a string."""

    return {
        'characters': len(text),
        'words': len(text.split()),
        'lines': len(text.splitlines())
    }
