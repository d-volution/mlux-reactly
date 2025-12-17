import math
import random
from io import StringIO
from mlux_reactly import ReactlyAgent, Recorder
from mlux_reactly.types import LLM
from mlux_reactly.tools import CalculatorTool

random.seed(12042)


def test_examples(agent: ReactlyAgent) -> None:
    examples = [
        calc_1_sinus_of,
        calc_1_sinus_of_degrees
    ]

    for example in examples:
        for _ in range(3):
            agent.runConfig.stream = StringIO()
            a = random.randint(-100000000, 100000000)
            question, truth = example(a)
            res = agent.query(question + " Answer with the result as a number only.")
            print("==> ", question, " --> ", res, " -- ", truth)
            if res != truth:
                print(agent.runConfig.stream.getvalue())
                print("--------------------------\n")


def calc_1_sinus_of(a):
    return f"What is the sine of {a}?", str(math.sin(a))

def calc_1_sinus_of_degrees(a):
    return f"What is the sine of {a} degrees?", str(math.sin(math.radians(a)))


# ---------------------------------------------------------------

llm = LLM("qwen2.5:7b-instruct-q8_0")
math_agent = ReactlyAgent(llm = llm, tools=[CalculatorTool()], 
                          recorder=Recorder("calc1", file=open("reactly_eval_query_record.jsonl", "+a")))

test_examples(math_agent)
