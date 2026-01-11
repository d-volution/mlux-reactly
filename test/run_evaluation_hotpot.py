from typing import List, Protocol, Type, Dict
import sys
import asyncio
import inspect
import json
from hotpot_based_evaluation import eval_results, Result
from hotpot_parser import HotpotDatapoint, HotpotDocument, parse_hotspot_file
from mlux_reactly import ReactlyAgent, LLM, Tracer
from tools import calculator, text_count, make_rag_for_folder, make_rag_for_documents, Document
from test_tracer import TestTracer
from test_types import Agent, AgentContructor, at_or
from llama_index_agent import LlamaFunctionAgentWrapper, LlamaReActAgentWrapper


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



async def run_hotpot_examples(examples: List[HotpotDatapoint], agent_constr: AgentContructor, tracer: Tracer, llm: LLM|None = None) -> Dict[str, float]:
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


available_example_files: Dict[str, str] = {
    'train': 'test-files/hotpot/hotpot_train_v1.1_FIRST_1000.json'
}

async def hotpot_test_fn(test_param: str, agent_constr: AgentContructor, llm: LLM):
    print(f"test p: '{test_param}'")
    param_splits = test_param.split(':')
    set_name = at_or(param_splits, 0, "train")
    start_pos = at_or(param_splits, 1, 0)
    size = at_or(param_splits, 2, 1)
    if set_name not in available_example_files.keys():
        raise AssertionError(f"hotpot test: param set name '{set_name}' not available")
    file_name = available_example_files[set_name]

    with open(file_name, 'r') as file:
        data = json.load(file)
        datapoints = parse_hotspot_file(data)[start_pos:start_pos+size]
        evaluation = await run_hotpot_examples(datapoints, agent_constr, TestTracer(stream=sys.stdout))
        return evaluation