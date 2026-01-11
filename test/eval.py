from typing import Dict, List, Any, Callable
from dataclasses import dataclass
import argparse
import asyncio
from test_types import Agent, AgentContructor, TestFunc, as_list
from mlux_reactly import ReactlyAgent, LLM, Tracer
from llama_index_agent import LlamaFunctionAgentWrapper, LlamaReActAgentWrapper
from run_evaluation_hotpot import hotpot_test_fn



argp = argparse.ArgumentParser()

argp.add_argument(
    "-tests", "-test",
    nargs="+",
    required=True,
    help="Test set name"
)
argp.add_argument(
    "-agents", "-agent",
    nargs="+",
    help="List of agent names"
)
argp.add_argument(
    "-llms", "-llm",
    nargs="+",
    help="List of LLM identifiers",
    default="qwen2.5:7b-instruct-q8_0"
)
argp.add_argument(
    "-comment",
    help="Free-form comment"
)
args = argp.parse_args()




@dataclass
class EvalRun:
    test: str
    test_param: str
    agent_name: str
    llm: str


available_tests: Dict[str, TestFunc] = {
    'hotpot': hotpot_test_fn,
}
available_agents: Dict[str, AgentContructor] = {
    'reactly': ReactlyAgent,
    'llama-react': LlamaReActAgentWrapper,
    'llama-func': LlamaFunctionAgentWrapper,
}
available_llms: Dict[str, LLM] = {
    'qwen2.5:7b-instruct-q8_0': LLM('qwen2.5:7b-instruct-q8_0')
}

def assert_arg(name: str, arg: str, availables: Dict[str, Any]):
    if arg not in availables.keys():
        raise AssertionError(f"command line argument {name} '{arg}' not available")

for test in as_list(args.tests):
    assert_arg('tests', test.split('/')[0], available_tests)
for agent in as_list(args.agents):
    assert_arg('agents', agent, available_agents)
for llm in as_list(args.llms):
    assert_arg('llms', llm, available_llms)



runs: List[EvalRun] = []


for test in as_list(args.tests):
    for agent in as_list(args.agents):
        for llm in as_list(args.llms):
            runs.append(EvalRun(
                test=test.split('/')[0],
                test_param="/".join(test.split('/')[1:]),
                agent_name=agent,
                llm=llm
            ))

for i, run in enumerate(runs):
    print(f"{i}: {run}")

print()

# run evaluations







async def main() -> None:
    for i, run in enumerate(runs):
        print(f"running {i}: {run}")

        test_fn = available_tests[run.test]
        agent = available_agents[run.agent_name]
        llm = available_llms[run.llm]

        evaluation = await test_fn(run.test_param, agent, llm)

        print('eval:', run, evaluation)

    print("Done")

if __name__ == '__main__':
    asyncio.run(main())