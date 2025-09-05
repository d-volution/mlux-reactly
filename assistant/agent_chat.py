from typing import List
from .types_message import Message, Role
from .types_exported import Agent, ResponseStream
from .llm import llm_query, LLMOptions
from .types_context import Context

SYSTEM_PROMPT = """You are a helpful assistant.
==========
"""




class Agent_Chat(Agent):
    def __init__(self, *, response_stream: ResponseStream = None):
        self.context = Context()
        self.llm_options = LLMOptions('qwen2.5:7b-instruct-q8_0', SYSTEM_PROMPT, response_stream)
    
    def query(self, user_question: str):
        self.context.append(Message(Role.User, user_question))
        llm_response = llm_query(self.context, self.llm_options)
        llm_response_msg = Message(Role.Assistant, llm_response)
        self.context.append(llm_response_msg)
        return llm_response_msg


