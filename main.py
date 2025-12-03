from mlux_reactly.react import run_react
from mlux_reactly.types import LLM
from mlux_reactly.tools import CalculatorTool
from mlux_reactly import ReactlyAgent

print("start")

llm = LLM("qwen2.5:7b-instruct-q8_0")
agent = ReactlyAgent(llm = llm, tools=[CalculatorTool()])

print(agent.query("Is 20 larger than 3?"))
print(agent.query("How high is the Eiffel Tower?"))
print(agent.query("What is the square root of sin(0.1274)?"))