from typing import List, Callable
import asyncio
from mlux_reactly import Tracer, ZeroTracer, Tool
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import FunctionAgent, ReActAgent
from llama_index.core.llms import LLM


default_llm = Ollama(
    model="qwen2.5:7b-instruct-q8_0"
)



class LlamaFunctionAgentWrapper:
    agent: FunctionAgent

    def __init__(
            self, 
            tools: List[Tool | Callable], *, 
            tracer: Tracer = ZeroTracer(),
            llm: LLM = default_llm):
        self.agent = FunctionAgent(
            tools=tools,
            llm=llm,
            #system_prompt="You are a helpful assistant that can multiply two numbers.",
        )

    def query(self, user_question: str):
        return str(self.agent.run(user_question))


class LlamaReActAgentWrapper:
    agent: ReActAgent

    def __init__(
            self, 
            tools: List[Tool | Callable], *, 
            tracer: Tracer = ZeroTracer(),
            default_llm: LLM = default_llm):
        self.agent = ReActAgent(
            tools=tools,
            llm=default_llm,
            #system_prompt="You are a helpful assistant that can multiply two numbers.",
        )


    async def query(self, user_question: str):
        handler = self.agent.run(user_question)
        answer = await handler
        return str(answer)
