from typing import List, Protocol, Type, Dict, Tuple
import sys
import inspect
import json
import time
import math
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


async def run_example_on_agent(example: Example, agent: Agent) -> Tuple[bool, Result, float]:
    # TODO: maybe retry, logging
    start_time = time.perf_counter()
    try:
        result = agent.query(example.question)
        answer: str
        duration: float
        if inspect.isawaitable(result):
            answer = await result
        else:
            answer = result
        duration = time.perf_counter() - start_time
        return True, Result(example.id, answer, None), duration
    except Exception as e:
        print(e)
        return False, Result(example.id, "", None), math.nan



async def run_hotpot_examples(examples: List[HotpotDatapoint], agent_constr: AgentContructor, tracer: Tracer, llm: LLM|None = None) -> Dict[str, float]:
    agent_results: List[Result] = []

    duration_total: float = 0
    duration_min: float = math.inf
    duration_max: float = -math.inf
    nr_finished: int = 0

    for example in examples:
        documents = [Document(text=doc.text(), metadata={'document_name': doc.title}) 
                     for doc in example.documents]
        eval_rag_tool = make_rag_for_documents(documents)
        agent = agent_constr(tools=[calculator, eval_rag_tool], tracer=tracer)

        finished, agent_result, duration = await run_example_on_agent(example, agent)

        agent_results.append(agent_result)
        if finished:
            duration_total += duration
            duration_min = min(duration_min, duration)
            duration_max = max(duration_max, duration)
            nr_finished += 1

    correct_results = [Result(example.id, example.answer, None) for example in examples]
    evaluation: Dict[str, float] = eval_results(correct_results, agent_results)

    evaluation |= {
        'duration_total': duration_total,
        'duration_min': duration_min,
        'duration_max': duration_max,
        'duration_avg': duration_total / nr_finished,
        'nr_total': len(examples),
        'nr_finished': nr_finished,
        'nr_failed': len(examples) - nr_finished
    }

    return evaluation


available_example_files: Dict[str, str] = {
    'train': 'test-files/hotpot/hotpot_train_v1.1_FIRST_1000.json'
}

async def hotpot_test_fn(test_param: str|None, agent_constr: AgentContructor, llm: LLM):
    param_splits = (test_param or "").split(':')
    set_name = at_or(param_splits, 0, "train")
    start_pos = int(at_or(param_splits, 1, 0))
    size = int(at_or(param_splits, 2, 1))
    if set_name not in available_example_files.keys():
        raise AssertionError(f"hotpot test: param set name '{set_name}' not available")
    file_name = available_example_files[set_name]

    with open(file_name, 'r') as file:
        data = json.load(file)
        datapoints = parse_hotspot_file(data)[start_pos:start_pos+size]
        evaluation = await run_hotpot_examples(datapoints, agent_constr, TestTracer(stream=sys.stdout))
        return evaluation