from mlux_reactly import ReactlyAgent
from mlux_reactly.types import LLM
from typing import Annotated, List
import json

print("start")

def candle_light(candle_type: str):
    """This tool calculates the brightness for a candle light of some candle type."""
    return

def history_calendar(historic_event: str, precision_in_hours: int):
    """A tool to query the date some historic event occurred."""

def calculator(expression: Annotated[str, "some kind of math expr"]):
    """A tool to evaluate a math expression."""

def geo_3d(from_system: str, to_system: str, from_coordinates: str):
    """A tool to convert between different kinds of earth coordinate systems."""

agent = ReactlyAgent(tools=[candle_light, history_calendar, geo_3d, calculator])


"""
print(agent.query("Is 20 larger than 3?"))
print(agent.query("How high is the Eiffel Tower?"))
print(agent.query("What is the square root of sin(0.1274)?"))
print(agent.query("How many characters does the word rasperry have?"))
print(agent.query("What degrees does Philip J. Pierre holds from the University of the West Indies?"))
"""

#print(agent.query("What does, if anything, Philip J. Pierre and Anton Zeilinger have in common? Try invoke at least one subagent before answering this."))

agent.query("How many years passed between when Napoleon became King of Italy and when the first world war ended?")