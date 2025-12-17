from typing import Optional, Any
from io import StringIO
import json
import ollama
from .types import LLM, Message, Role, Tool, BaseAgent, RunConfig
from .diagnostics import Diagnostics


def run_react(query: str, agent: BaseAgent, run_config: RunConfig) -> str:
    llm = agent.llm
    tools = agent.tools
    diagnostics = run_config.diagnostics

    history: list[Message] = []

    def stream_print(*values: object):
        print(*values, file=run_config.stream)

    def append_msg(message: Message):
        if run_config.stream != None:
            if len(message.content) > 500:
                stream_print(f"-> {message.role.name}: {message.content[:500]}... (total size {len(message.content)})")
            else:
                stream_print(f"-> {message.role.name}: {message.content}")

        history.append(message)

    def call_llm(history: list[Message], llm: LLM) -> str:
        response = ollama.chat(
            model=llm.model,
            messages=[{
                'role': msg.role.name.lower(),
                'content': msg.content
            } for msg in history]
        )
        return response["message"]["content"]

    def run_tool(tool: Optional[Tool], input_as_json: str) -> tuple[str, bool]:
        if tool == None:
            diagnostics.increment_counter("error_internal_no_tool_to_run")
            return "The agent had some internal error. Tool could not be run.", False
        else:
            try:
                input = json.loads(input_as_json)
            except:
                diagnostics.increment_counter("error_action_input_json_invalid")
                return "The Action Input was not encoded as valid JSON.", False

            try:
                return tool.run(input)
            except:
                diagnostics.increment_counter("error_tool_crash")
                return "The tool crashed while running.", False

    append_msg(Message(Role.System, agent.system_prompt))
    append_msg(Message(Role.User, f"Question: {query}"))
    selected_tool: Optional[Tool] = None
    expect_steps: list[str] = ["Thought", "Answer"]


    while True:
        diagnostics.increment_counter("steps")
        output = call_llm(history, llm)
        
        for line in output.splitlines():
            if line == "":
                continue

            append_msg(Message(Role.Assistant, f"{line}"))

            step_body_parts = line.split(": ", 1)
            if len(step_body_parts) < 2:
                append_msg(Message(Role.User, f"Error: This step has invalid format! Stick to format: `role: ...`"))
                diagnostics.increment_counter("error_invalid_format")
                break

            step, body = step_body_parts

            if step not in expect_steps:
                append_msg(Message(Role.User, f"Error: Step '{step}' not expected at this point. Expect one of [{", ".join(expect_steps)}]."))
                diagnostics.increment_counter("error_invalid_step")
                break

            diagnostics.increment_counter("step_" + step.lower().replace(" ", "_"))

            if step == "Thought":
                expect_steps = ["Action", "Answer"]

            elif step == "Action":
                tool_name = body.lower()
                if tools.get(tool_name) == None:
                    append_msg(Message(Role.User, f"Error: Tool '{tool_name}' not found. Choose other Action or answer!"))
                    diagnostics.increment_counter("error_tool_not_found")
                    break
                
                selected_tool = tools.get(tool_name)
                expect_steps = ["Action Input"]

            elif step == "Action Input":
                response, ok = run_tool(selected_tool, body)
                if ok:
                    append_msg(Message(Role.User, f"Observation: {response}"))
                else:
                    append_msg(Message(Role.User, f"Observation: Tool responded with error: {response}"))
                expect_steps = ["Thought", "Answer"]
                selected_tool = None

            elif step == "Answer":
                return body



