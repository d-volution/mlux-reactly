from mlux_reactly.react import run_react
from mlux_reactly.types import LLM

print("start")

llm = LLM("qwen2.5:7b-instruct-q8_0")

print(run_react("Is 20 larger than 3?", llm, {}))
print(run_react("How high is the Eiffel Tower?", llm, {}))