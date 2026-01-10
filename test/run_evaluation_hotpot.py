from typing import List, Protocol, Type, Dict
import sys
import asyncio
import inspect
from hotpot_based_evaluation import eval_results, Result
from hotpot_parser import HotpotDatapoint, HotpotDocument, parse_hotspot_file
from mlux_reactly import ReactlyAgent, LLM, Tracer
from tools import calculator, text_count, make_rag_for_folder, make_rag_for_documents, Document
from test_tracer import TestTracer
from llama_index_agent import LlamaFunctionAgentWrapper, LlamaReActAgentWrapper

import json

class Agent(Protocol):
    def query(self, user_question: str) -> str: ...

class Example(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def question(self) -> str: ...
    @property
    def answer(self) -> str: ...


async def run_example_on_agent(example: Example, agent: Agent) -> Result:
    # TODO: maybe retry, logging
    try:
        result = agent.query(example.question)
        answer: str
        if inspect.isawaitable(result):
            answer = await result
        else:
            answer = result

        return Result(example.id, answer, None)
    except Exception as e:
        print(e)
        return Result(example.id, "", None)



async def run_hotpot_examples(examples: List[HotpotDatapoint], agent_constr, tracer: Tracer) -> Dict[str, float]:
    agent_results: List[Result] = []

    for example in examples:
        documents = [Document(text=doc.text(), metadata={'document_name': doc.title}) 
                     for doc in example.documents]
        eval_rag_tool = make_rag_for_documents(documents)
        agent = agent_constr(tools=[calculator, eval_rag_tool], tracer=tracer)

        agent_result = await run_example_on_agent(example, agent)
        print('45', type(agent_result), agent_result)
        agent_results.append(agent_result)

    correct_results = [Result(example.id, example.answer, None) for example in examples]
    evaluation: Dict[str, float] = eval_results(correct_results, agent_results)

    return evaluation


async def main() -> None:

    with open('test-files/hotpot/hotpot_train_v1.1_FIRST_1000.json', 'r') as file:
        data = json.load(file)
        datapoints = parse_hotspot_file(data)[80:85]

        print("evaluate hotpot llama react")
        evaluation_llama_react = await run_hotpot_examples(datapoints, LlamaReActAgentWrapper, TestTracer(stream=sys.stdout))

        print("evaluate hotpot my reactly")
        evaluation_my = await run_hotpot_examples(datapoints, ReactlyAgent, TestTracer(stream=sys.stdout))

        print('evaluation_my', evaluation_my)
        print('evaluation_llama_react', evaluation_llama_react)

    
asyncio.run(main())