import traceback
import sys
import io
from mlux_reactly import ReactlyAgent, NormalTracer, LLM
from tools import calculator, text_count, make_rag_tool


class StdoutStringIO(io.StringIO):
    trace_setting: bool = True

    def write(self, s):
        if self.trace_setting:
            sys.stdout.write(s)
        return len(s)

stream = StdoutStringIO()
tracer = NormalTracer(name="chat", stream=stream, record_file=open("reactly_query_record.jsonl", "+a"))

llm = LLM("qwen2.5:7b-instruct-q8_0")
agent = ReactlyAgent(tools=[calculator, text_count, make_rag_tool("./test-files-rag")], tracer=tracer, llm=llm)


# chat loop:

while True:
    user_input = input(">>> ")

    if user_input == "":
        continue

    if user_input.startswith("/"):
        if user_input == "/bye" or user_input == "/q":
            break
        if user_input.startswith("/trace on"):
            stream.trace_setting = True
        if user_input.startswith("/trace off"):
            stream.trace_setting = False

        continue

    try:
        response = agent.query(user_input)
        print(response + "\n")
    except Exception as e:
        print(f"\033[31m[query crashed]\033[0m")
        traceback.print_exception(e)
        print()
