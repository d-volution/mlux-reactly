import traceback
import sys
import io
import json
from dotenv import load_dotenv
load_dotenv()

from mlux_reactly import ReactlyAgent, LLM, ZeroTracer
from tools import calculator, text_count, make_rag_for_folder, wikipedia_search
from test_tracer import TestTracer, make_json_serializable, format_tracer_compact

test_tracer = TestTracer(session='chat', record_file=open("reactly_query_record.jsonl", "+a"))
tracer = test_tracer

llm = LLM("qwen2.5:7b-instruct-q8_0")
agent = ReactlyAgent(tools=[calculator, text_count, make_rag_for_folder("./test-files/rag-docs-default"), wikipedia_search], tracer=tracer, llm=llm)


# chat loop:

while True:
    user_input = input(">>> ")

    if user_input == "":
        continue

    if user_input.startswith("/"):
        if user_input == "/bye" or user_input == "/q":
            break
        if user_input.startswith("/trace ") or user_input.startswith("/t "):
            cmds = user_input.split()
            if cmds[1] == "on":
                tracer = test_tracer
            elif cmds[1] == "off":
                tracer = ZeroTracer()
            elif cmds[1] == "dump":
                print(format_tracer_compact(tracer))
            elif cmds[1] == "json":
                print(json.dumps(make_json_serializable(tracer.event), indent=2))

        continue

    try:
        response = agent.query(user_input)
        print(response + "\n")
    except Exception as e:
        print(format_tracer_compact(tracer))
        print(f"\033[31m[query crashed]\033[0m")
        traceback.print_exception(e)
        print()
