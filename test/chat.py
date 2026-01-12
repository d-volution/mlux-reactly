import traceback
import sys
import io
import json
from dotenv import load_dotenv
load_dotenv()

from mlux_reactly import ReactlyAgent, LLM
from tools import calculator, text_count, make_rag_for_folder, wikipedia_search
from test_tracer import TestTracer, make_json_serializable

tracer = TestTracer(session='chat', record_file=open("reactly_query_record.jsonl", "+a"))

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
        if user_input.startswith("/trace on"):
            pass
        if user_input.startswith("/trace off"):
            pass
        if user_input.startswith('/trace dump'):
            print(tracer.format_compact())
        if user_input.startswith('/trace j'):
            print(json.dumps(make_json_serializable(tracer.event), indent=2))

        continue

    try:
        response = agent.query(user_input)
        print(response + "\n")
    except Exception as e:
        print(tracer.format_compact())
        print(f"\033[31m[query crashed]\033[0m")
        traceback.print_exception(e)
        print()
