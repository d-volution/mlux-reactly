from mlux_reactly import ReactlyAgent, Recorder
from mlux_reactly.types import LLM


llm = LLM("qwen2.5:7b-instruct-q8_0")
agent = ReactlyAgent(llm = llm, tools=[], recorder=Recorder("testchat", file=open("reactly_query_record.jsonl", "+a")))


# chat loop:

while True:
    user_input = input(">>> ")

    if user_input == "":
        continue

    if user_input.startswith("/bye"):
        break

    response = agent.query(user_input)
    print(response)
