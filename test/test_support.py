from typing import List, Dict, Tuple
import inspect
import time
import math
from hotpot_based_evaluation import eval_results, Result
from mlux_reactly import LLM, Tracer
from test_types import Agent, AgentContructor, Example, ExampleCase



async def run_example_on_agent(example: Example, agent: Agent) -> Tuple[bool, Result, float]:
    # TODO: maybe retry, logging
    start_time = time.perf_counter()
    try:
        result = agent.query("Answer the following question by *just* stating the questioned fact in a few words (For example, 'Where is the Eiffel Tower' would be best answered by 'Paris, France'. No explanation.)."
                             " Question: " + example.question)
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
    

async def run_hotpotlike_examples(example_cases: List[ExampleCase], agent_constr: AgentContructor, tracer: Tracer, llm: LLM|None = None, talky: bool = True) -> Dict[str, float]:
    agent_results: List[Result] = []

    duration_total: float = 0
    duration_min: float = math.inf
    duration_max: float = -math.inf
    nr_finished: int = 0

    for example_case in example_cases:
        agent = agent_constr(tools=example_case.agent_config.tools, tracer=tracer, llm=llm)

        finished, agent_result, duration = await run_example_on_agent(example_case.example, agent)

        if talky:
            print(f"=> {example_case.example.id[:6]}: '{example_case.example.question}' good: '{example_case.example.answer}' A: '{agent_result.answer}'")

        agent_results.append(agent_result)
        if finished:
            duration_total += duration
            duration_min = min(duration_min, duration)
            duration_max = max(duration_max, duration)
            nr_finished += 1

    correct_results = [Result(example_case.example.id, example_case.example.answer, None) for example_case in example_cases]
    evaluation: Dict[str, float] = eval_results(correct_results, agent_results)

    evaluation |= {
        'duration_total': duration_total,
        'duration_min': duration_min,
        'duration_max': duration_max,
        'duration_avg': (duration_total / nr_finished) if nr_finished else math.nan,
        'nr_total': len(example_cases),
        'nr_finished': nr_finished,
        'nr_failed': len(example_cases) - nr_finished
    }

    return evaluation