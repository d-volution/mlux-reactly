import math
import random
import re
import numexpr as ne
from mlux_reactly import ReactlyAgent, NormalTracer
from mlux_reactly.types import LLM
from tools import calculator, text_count, make_rag_for_folder

random.seed(12042)
record_file=open("reactly_eval_query_record.jsonl", "+a")
llm = LLM("qwen2.5:7b-instruct-q8_0")




def calc_1_sinus_of(a):
    return f"What is the sine of {a}?", str(ne.evaluate(f"sin({a})"))

def calc_1_sinus_of_degrees(a):
    return f"What is the sine of {a} degrees?", str(ne.evaluate(f"sin(deg * pi / 180)", {"deg": a, "pi": math.pi}))



def evaluate_calc_1():
    examples = [
        calc_1_sinus_of,
        calc_1_sinus_of_degrees
    ]

    for example in examples:
        per_ex_stats = {
            'example_name': example.__name__,
            'nr_correct': 0,
            'nr_incorrect': 0
        }

        for _ in range(10):
            tracer = NormalTracer(name=f"eval_{example.__name__}", record_file=record_file)
            agent = ReactlyAgent(llm = llm, tools=[calculator], tracer=tracer)
            
            a = random.randint(-100000000, 100000000)
            question, truth = example(a)

            res = agent.query(question + " Answer with the result as a number only.")

            match_nr = re.search(r"-?\d+(\.\d+)?", res).group()
            correct = match_nr == truth[:len(match_nr)]

            print(f"==> Q: '{question}' res: {match_nr} truth: {truth} => {"correct" if correct else "incorrect"}")
            if correct:
                per_ex_stats['nr_correct'] += 1
            else:
                per_ex_stats['nr_incorrect'] += 1

        print("------------------------------------------------------------------")
        for key, val in per_ex_stats.items():
            print(f"  {key}: {val}")


evaluate_calc_1()