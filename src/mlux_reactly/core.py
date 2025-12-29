from typing import Any, List, Dict
from enum import Enum
from dataclasses import dataclass, asdict
import json
import ollama
from .types import LLM, Tool, TaskResult, ChatQA, Tracer
from .diagnostics import Diagnostics

# helper

class Role(Enum):
    System = "system"
    User = "user"
    Assistant = "assistant"

@dataclass
class CtxMessage:
    role: Role
    content: str


def call_llm(context: List[CtxMessage], llm: LLM) -> str:
    response = ollama.chat(
            model=llm.model,
            messages=[{
                'role': msg.role.name.lower(),
                'content': msg.content
            } for msg in context]
        )
    return response["message"]["content"]


def stage_call_llm(sys_prompt: str, query_msg: str, llm: LLM, *, decode_json: bool = True) -> Any:
    llm_response = call_llm([
        CtxMessage(Role.System, sys_prompt),
        CtxMessage(Role.User, query_msg)
    ], llm)
    #print(f"-> {query_msg}{llm_response}\n")
    if not decode_json:
        return llm_response
    
    try:
        decoded = json.loads(llm_response)
        return decoded
    except:
        return json.dumps(llm_response)    

@dataclass
class ToolCall:
    tool: str
    input: Any
    output: Any


def run_tool(tool: Tool, input: Dict[str, Any]) -> str:
    #print("run tool", tool, input)
    return "no result"


def format_tools(tools: List[Tool]) -> str:
    #as_dict = {tool.name: {'description': tool.doc, 'input format': tool.input_doc} for tool in tools}
    as_dict = {tool.name: tool.doc for tool in tools}
    return json.dumps(as_dict)

def format_chat_history(history: List[ChatQA]) -> str:
    return json.dumps([asdict(qa) for qa in history])

def format_subtask_results(results: List[TaskResult]) -> str:
    return json.dumps([asdict(result) for result in results])

def format_tool_calls(tool_calls: List[ToolCall]) -> str:
    return json.dumps([asdict(tool_call) for tool_call in tool_calls])

# stages

def split_user_question(user_question: str, history: List[ChatQA], tools: List[Tool], llm: LLM, caller_tracer: Tracer) -> List[str]:
    tracer = caller_tracer.on("split_user_question", locals())
    SYS_PROMPT = """You are a task-splitting system.

Your job is to decompose the user Question into the minimal ordered list of atomic Tasks required to answer it.

# Rules
- Do NOT answer the Question.
- ONLY output the Tasks array in valid JSON. No extra text.
- Tasks must be strictly ordered.
- Do not merge multiple actions into one task.
- Do not invent facts or assume missing information.
- If required information is missing, create a task to retrieve it.
- If the Question can be answered without tools, output [].
- Prefer fewer tasks over more.
- Use neutral, imperative phrasing.
- NEVER explain your reasoning.

# Format

The format looks like this:

Tools: {"tool name": "tool description", ...more tools...}
Question: the question (i.e. main task) from the user to be splitted into (sub) tasks
Tasks: ["task 1", "task 2", ...]

# GOOD Examples

Tools: {"sqr_math": "A tool to square any real number.", "wood": "This tool returns basic properties of a kind of wood.", "cowork_db": "DB to look up coworker info.", "hex_color": "Retuns the color code for an inputted color name"}
Question: What is the sqare of the year the coworker Mike was born?
Tasks: ["Find out the date of birth for coworker Mike.", "Square the year number of the date of birth."]

# Conversation

"""
    query_msg = f"""
Tools: {format_tools(tools)}
Question: {user_question}
Tasks: """
    return stage_call_llm(SYS_PROMPT, query_msg, llm)


def answer_user_question(user_question: str, history: List[ChatQA], subtask_results: List[TaskResult], llm: LLM) -> str:
    SYS_PROMPT = """You are a helpful answer generator.

Your job is to answer the user Question by using the Results of the subtasks.

# Format

The format looks like this:

History: [{"question": "First question asked by user on a previous query.", "response": "response from the agent for the first question"}, ...]
Question: the user Question you should answer
Results: [{"task": "description of subtask 1", "result": "result of subtask 1"}, ...]
Answer: your answer based on the Results

# GOOD Examples

History: []
Question: Do Mark and Elisa drive the same type of car?
Results: [{"task": "Determine what type of car Mark drives.", "result": "Mark drives an Audi A4."}, {"task": "Determine what type of car Elisa drives.", "result": "Elisa drives both a Tesla Model 3 and a Audi A4."}]
Answer: Yes, both drive a Audi A4.

# Conversation

"""
    query_msg = f"""
History: {format_chat_history(history)}
Question: {user_question}
Results: {format_subtask_results(subtask_results)}
Answer: """
    return stage_call_llm(SYS_PROMPT, query_msg, llm)



def choose_tool(task_description: str, tools: List[Tool], llm: LLM, caller_tracer: Tracer) -> Tool:
    SYS_PROMPT = """You are a tool choser.

Your job is to select one of the Tools to be called next. The History shows the previous Tool calls including the input to the Tool and the output from it.

# Format

The format looks like this:

Tools: {"tool name": "tool description", ...more tools...}
Task: the description of the task
History: [{"tool": "tool name", "input": "input to tool", "output": "the response of the tool"}, ...]
Chosen Tool: name of the chosen tool

# GOOD Examples

Tools: {"sqr_math": "A tool to square any real number.", "wood": "This tool returns basic properties of a kind of wood.", "cowork_db": "DB to look up coworker info.", "hex_color": "Retuns the color code for an inputted color name"}
Task: Square the number of unknown guests. There were 123 unknown guest at the party.
History: []
Chosen Tool: sqr_math

# Conversation

"""
    query_msg = f"""
Tools: {format_tools(tools)}
Task: {task_description}
History: []
Chosen Tool: """
    tool_name = stage_call_llm(SYS_PROMPT, query_msg, llm, decode_json=False)
    tool = next((tool for tool in tools if tool.name == tool_name), None)
    return tool


def make_tool_input(task_description: str, tool: Tool, llm: LLM) -> Any:
    EXAMPLE_TOOL_SQR_MATH_DESCR = """A tool to square any real number.

Input only a single number (as a valid JSON-formatted number). The tool will return the square of the number.
"""
    SYS_PROMPT = """You are an input generator.

Your job is to parse the Task description and generate a valid Input for the provided Tool.

# Rules

- ONLY output the Input in valid JSON. No extra text.

# Format

The format looks like this:

Task: the description of the task
Tool: {"name of provided tool": "tool description"}
Input: the JSON-encoded input formatted as the tool description demands

# GOOD Examples

Task: Find the square of 123.
Tool: {"sqr_math": """ + EXAMPLE_TOOL_SQR_MATH_DESCR + """}
Input: 123

# BAD Examples

Task: What is the square of one-hundred and three.
Tool: {"sqr_math": """ + EXAMPLE_TOOL_SQR_MATH_DESCR + """}
Input: one-hundred

# Conversation

"""
    query_msg = f"""
Task: {task_description}
Tool: {json.dumps({tool.name: {'description': tool.doc, 'input format': tool.input_doc}})}
Input: """
    return stage_call_llm(SYS_PROMPT, query_msg, llm)


def answer_subtask(task_description: str, tool_calls: List[ToolCall], llm: LLM) -> Any:
    SYS_PROMPT = """You are a context summarizer.

Your job is to Answer the Task using the History of performed tool calls.

# Format

The format looks like this:

Task: the description of the task
History: [{"tool": "tool name", "input": "input to tool", "output": "the response of the tool"}, ...]
Answer: your answer of the Task

# GOOD Examples

Task: Square the number of unknown guests. There were 123 unknown guest at the party.
History: [{"tool": "sqr_math", "input": "123", "output": 15129}]
Answer: The quare of the number of 123 unknown guests is 15129.

# Conversation

"""
    query_msg = f"""
Task: {task_description}
History: {format_tool_calls(tool_calls)}
Answer: """
    return stage_call_llm(SYS_PROMPT, query_msg, llm)


# main

def run_subtask(task_description: str, tools: List[Tool], results: List[TaskResult], llm: LLM, caller_tracer = Tracer) -> str:
    tracer = caller_tracer.on("run_subtask", locals())

    chosen_tool = choose_tool(task_description, tools, llm, tracer)
    tool_input = make_tool_input(task_description, chosen_tool, llm)
    tool_result = run_tool(chosen_tool, tool_input)
    tool_calls = [ToolCall(chosen_tool.name, tool_input, tool_result)]
    subanswer = answer_subtask(task_description, tool_calls, llm)
    tracer.on("answer", {'answer': subanswer})
    return subanswer


def run_query(user_question: str, history: List[ChatQA], tools: List[Tool], llm: LLM, agent_tracer: Tracer) -> str:
    tracer = agent_tracer.on("run_query", locals())
    subtask_results: List[TaskResult] = []
    
    subtasks = split_user_question(user_question, history, tools, llm, tracer)

    for subtask in subtasks:
        result = run_subtask(subtask, tools, subtask_results, llm, tracer)
        subtask_results.append(TaskResult(subtask, result))

    answer = answer_user_question(user_question, history, subtask_results, llm)
    tracer.add_arg("answer", answer)
    tracer.reset()
    return answer