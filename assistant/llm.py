from dataclasses import dataclass
from typing import Callable, List, Optional
from abc import ABC, abstractmethod
import ollama
from .types_message import Message
from .types_exported import ResponseStream






@dataclass
class LLMOptions:
    model: str
    system_prompt: str
    response_stream: ResponseStream = None

def llm_query(context: List[Message], options: LLMOptions) -> str:

    llm_context = [{'role': 'system', 'content': options.system_prompt}]
    for message in context:
        llm_context.append({
            'role': message.role.name.lower(),
            'content': message.text
        })
    is_stream_set = options.response_stream is not None

    response = ollama.chat(
        model=options.model,
        messages=llm_context,
        stream=is_stream_set
    )
    if is_stream_set:
        response_content = ""
        for chunk in response:
            chunk_content = chunk["message"]["content"]
            response_content += chunk_content
            options.response_stream.write(chunk_content)
    else:
        response_content = response["message"]["content"]

    return response_content


