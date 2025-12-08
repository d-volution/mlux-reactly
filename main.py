from mlux_reactly.react import run_react
from mlux_reactly.types import LLM
from mlux_reactly.tools import CalculatorTool, TextCountTool, RetrieverTool
from mlux_reactly import ReactlyAgent
from mlux_reactly.recorder import Recorder

print("start")

llm = LLM("qwen2.5:7b-instruct-q8_0")
tools = [
    CalculatorTool(), 
    TextCountTool(), 
    RetrieverTool.from_directory_path("test-files")
    ]
agent = ReactlyAgent(llm = llm, tools=tools, recorder=Recorder(file=open("reactly_query_record.jsonl", "+a")))

print(agent.query("Is 20 larger than 3?"))
print(agent.query("How high is the Eiffel Tower?"))
print(agent.query("What is the square root of sin(0.1274)?"))
print(agent.query("How many characters does the word rasperry have?"))
print(agent.query("What degrees does Philip J. Pierre holds from the University of the West Indies?"))