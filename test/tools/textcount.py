from typing import Any, Annotated
import json


def text_count(text: Annotated[str, """The text string to be counted"""]) -> str:
    """The text_count tool counts numbers of characters, words and lines in a string."""

    r = {
        'characters': len(text),
        'words': len(text.split()),
        'lines': len(text.splitlines())
    }
    return json.dumps(r)