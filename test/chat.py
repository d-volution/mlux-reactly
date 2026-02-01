import asyncio
import traceback
import json

from mlux_reactly import ReactlyAgent, LLM, ZeroTracer
from tools import calculator, text_count, make_rag_for_folder, wikipedia_search
from test_tracer import TestTracer, make_json_serializable, format_tracer, format_tracer_with_nr, FormatConfig, TraceConfig
from eval import main_eval, args_to_eval_runs, EvalRun

record_file = open("reactly_query_record.jsonl", "+a")
test_tracer_format = FormatConfig()
test_tracer = TestTracer(session='chat', record_file=record_file)
eval_tracer = test_tracer #TestTracer(config=TraceConfig(session='chateval', record_file=record_file, live_format=FormatConfig(show={'query': True, 'query_answer': True}, show_other=False)))
tracer = test_tracer

llm = LLM("qwen2.5:7b-instruct-q8_0")
agent = ReactlyAgent(tools=[calculator, text_count, make_rag_for_folder("./test-files/rag-docs-default"), wikipedia_search], tracer=tracer, llm=llm)

# helper

def to_int(s: str, default: int | None = None) -> int | None:
    try:
        return int(s)
    except ValueError:
        return default

# chat loop:

while True:
    user_input = input(">>>> ")

    if user_input == "":
        continue

    if user_input.startswith("/"):
        if user_input == "/bye" or user_input == "/q":
            break
        if user_input.startswith("/trace ") or user_input.startswith("/t "):
            cmds = user_input.split()
            if cmds[1] == "on":
                tracer = test_tracer
                print("-- tracing on")
            elif cmds[1] == "off":
                tracer = ZeroTracer()
                print("-- tracing off")
            elif cmds[1] == "llm:on":
                test_tracer_format.show['llmcall'] = True
                print('-- showing llmcall on')
            elif cmds[1] == "llm:off":
                test_tracer_format.show['llmcall'] = False
                print('-- showing llmcall off')
            elif cmds[1] == "compact":
                test_tracer_format.compact = True
                print('-- showing compact on')
            elif cmds[1] == "verbose":
                test_tracer_format.compact = False
                print('-- showing compact off')
            elif cmds[1] == "dump":
                print(format_tracer(tracer, test_tracer_format))
            elif cmds[1] == "show":
                if len(cmds) <= 2:
                    print('usage:   /trace show <event nr>')
                    continue
                nr = to_int(cmds[2], -100)
                if nr < 0:
                    print('invalid nr. usage:   /trace show <event nr>')
                    continue
                print(format_tracer_with_nr(tracer, nr))
            elif cmds[1] == "json":
                print(json.dumps(make_json_serializable(tracer.event), indent=2))

        if user_input.startswith('//1'):
            user_input = '/eval -agents reactly llama-react -tests hotpot/train:100:5'
        if user_input.startswith('//2'):
            user_input = '/eval -agents reactly llama-react -tests qa/wikipedia-1:0:2'

        if user_input.startswith("/eval ") or user_input.startswith("/e "):
            argstr = user_input.split(maxsplit=1)
            if len(argstr) < 2:
                print('usage:    /eval -agents agent1 ... -tests test1 ...')
                continue
            try:
                runs = args_to_eval_runs(argstr[1].split())
                result = asyncio.run(main_eval(runs, eval_tracer))
            except Exception as e:
                print(f"\033[31m[eval crashed]\033[0m")
                traceback.print_exception(e)
                print()

        continue

    try:
        response = agent.query(user_input)

        if isinstance(tracer, TestTracer) and tracer.event.args.get('flag_has_failed_event', False) == True:
            print(format_tracer(tracer, test_tracer_format))

        print(response + "\n")
    except Exception as e:
        print(format_tracer(tracer, test_tracer_format))
        print(f"\033[31m[query crashed]\033[0m")
        traceback.print_exception(e)
        print()
