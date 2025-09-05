import json
import numexpr as ne
import requests
from .types_context import Context
from .types_message import Message, Role
from .llm import llm_query, LLMOptions
from .types_exported import Agent, ResponseStream
from .agent_react_system_prompt import SYSTEM_PROMPT

class Agent_ReAct(Agent):
    def __init__(self, *, response_stream: ResponseStream = None):
        self.context = Context()
        self.llm_options = LLMOptions('qwen2.5:7b-instruct-q8_0', SYSTEM_PROMPT, response_stream)

    def query(self, user_question):
        self.context.append(Message(Role.User, user_question))

        cycle_nr = 0

        while True:
            cycle_nr += 1
            thought_line_start = "Thought " + str(cycle_nr) + ":"
            if self.llm_options.response_stream != None:
                self.llm_options.response_stream.write(thought_line_start)

            llm_result = llm_query(self.context.with_appended(Message(Role.Assistant, thought_line_start)), self.llm_options)

            llm_result_lines = llm_result.split("\n")
            self.context.append(Message(Role.Assistant, thought_line_start + llm_result_lines[0]))

            action_line_found = False
            action_line_without_start: str = ""
            for line in llm_result_lines[1:]:
                if line.startswith("Action "):
                    self.context.append(Message(Role.Assistant, line + "\n"))
                    self.llm_options.response_stream.write("\n")
                    action_line_found = True
                    action_line_without_start = ":".join(line.split(":")[1:])
                    break
            
            if not action_line_found:
                action_line_start = "Action " + str(cycle_nr) + ":"
                if self.llm_options.response_stream.write != None:
                    self.llm_options.response_stream.write(action_line_start)

                llm_result = llm_query(self.context.with_appended(Message(Role.Assistant, action_line_start)), self.llm_options)

                action_line_without_start = llm_result.split("\n")[0]
                self.context.append(Message(Role.Assistant, action_line_start + action_line_without_start + "\n"))
                self.llm_options.response_stream.write("\n")

            action_args = json.loads(action_line_without_start)
            action_kind = str(action_args['action']).lower()
            action_input = action_args['input']

            # run tool and make observation
        
            if action_kind == "answer":
                return action_input
            elif action_kind == "calculator":
                tool_result = ne.evaluate(action_input)
                self.context.append(Message(Role.User, f"Observation {cycle_nr}: {{\"result\": \"{str(tool_result)}\"}}\n"))
                self.llm_options.response_stream.write(self.context.last().text)
            elif action_kind == "wikipedia":
                tool_result = wikipedia_search(action_input)
                self.context.append(Message(Role.User, f"Observation {cycle_nr}: {{\"result\": {tool_result} }}\n"))
                self.llm_options.response_stream.write(self.context.last().text)
            else:
                return "[ERROR] action '" + action_kind + "' not implemented."


def wikipedia_search(query: str, limit: int = 3):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json"
    }
    headers = {
        "User-Agent": "mlux-reactly/0.1" # https://phabricator.wikimedia.org/T400119 required by Wikipedia 
    }
    response = requests.get(url, params=params, headers=headers)

    response_json = response.json()
    results = response_json.get("query", {}).get("search", [])
    return [{"title": r["title"], "snippet": r["snippet"]} for r in results]
