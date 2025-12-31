from mlux_reactly import ReactlyAgent, NormalTracer
import sys
import io
from tools import calculator, text_count

class StdoutStringIO(io.StringIO):
    def write(self, s):
        sys.stdout.write(s)
        return len(s)


tracer = NormalTracer(stream=StdoutStringIO(), record_file=open("reactly_query_record.jsonl", "+a"))
agent = ReactlyAgent(tools=[calculator, text_count], tracer=tracer)


def query(user_question: str):
    print(">>> " + user_question)
    response = agent.query(user_question)
    print(response + "\n")



query("How many characters does the word asinusobelix have?")
query("What is the square root of 358079929?")
query("Is 20 larger than 3?")
query("How high is the Eiffel Tower?")
query("What is the square root of sin(0.1274)?")
query("How many characters does the word rasperry have?")
query("What degrees does Philip J. Pierre holds from the University of the West Indies?")