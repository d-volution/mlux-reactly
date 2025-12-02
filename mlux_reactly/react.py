
from .prompts import generate_react_prompt
from .types import LLM, Message, Role, Tool, DiagnosticHandler
from .diagnostics import DiagnosticHandlerDefault
import ollama



def run_react(query: str, llm: LLM, tools: dict[str, Tool], *, diagnostic: DiagnosticHandler = DiagnosticHandlerDefault) -> str:
    history: list[Message] = []
    append_msg(history, Message(Role.System, generate_react_prompt(tools)))
    append_msg(history, Message(Role.User, f"Question: {query}"))
    selected_tool: Tool = None
    expect_steps: list[str] = ["Thought", "Answer"]

    while True:
        output = call_llm(history, llm)
        diagnostic.event("llm_call_finished")
        
        for line in output.splitlines():
            append_msg(history, Message(Role.Assistant, f"{line}"))

            step_body_parts = line.split(": ", 1)
            if len(step_body_parts) < 2:
                append_msg(history, Message(Role.User, f"Error: This step has invalid format! Stick to format: `role: ...`"))
                break

            step, body = step_body_parts

            if step not in expect_steps:
                append_msg(history, Message(Role.User, f"Error: Step '{step}' not expected at this point. Expect one of [{", ".join(expect_steps)}]."))
                break

            if step == "Thought":
                expect_steps = ["Action", "Answer"]

            elif step == "Action":
                tool_name = body.lower()
                if tools.get(tool_name) == None:
                    append_msg(history, Message(Role.User, f"Error: Tool '{tool_name}' not found. Choose other Action or answer!"))
                    break
                
                selected_tool = tools.get(tool_name)
                expect_steps = ["Action Input"]

            elif step == "Action Input":
                response, ok = selected_tool.run(body)
                if ok:
                    append_msg(history, Message(Role.User, f"Observation: {response}"))
                else:
                    append_msg(history, Message(Role.User, f"Observation: Tool responded with error: {response}"))
                expect_steps = ["Thought", "Answer"]
                selected_tool = None

            elif step == "Answer":
                return body

def append_msg(history: list[Message], message: Message):
    print(f"-> {message.role.name}: {message.content}")
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