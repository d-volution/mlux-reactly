from typing import Any, List, Dict
from enum import Enum
from dataclasses import dataclass, asdict
from .types import LLM, Tool, TaskResult, ChatQA, Tracer
from .framework import make_stage, make_ffff, ffff_as_list, Encoding
from .diagnostics import Diagnostics, tracer_finish_with_response

# helper


@dataclass
class ToolRunRecord:
    tool: str
    input: Any
    output: Any


def run_tool(tool: Tool, input: Dict[str, Any], caller_tracer: Tracer) -> Any:
    tracer = caller_tracer.on("run_tool", locals())
    try:
        result = tool.run(**input)
        tracer.on("tool_result", {'result': result})
    except Exception as e:
        result = f"tool failed: {e}"
        tracer.on("tool_failure", {'result': result})
    return result


TOOLS_FORMAT_LINE = """{"tool name": "tool description", ...more tools...}"""
EXAMPLE_TOOLS = [
    Tool("sqr_math", "A tool to square any real number.", {"number": "The number that should be squared"}, lambda **kwargs: ""),
    Tool("wood", "This tool returns basic properties of a kind of wood", {"kind": "name of wood kind"}, lambda **kwargs: ""),
    Tool("cowork_db", "DB to look up coworker info.", {"": ""}, lambda **kwargs: ""),
    Tool("hex_color", "Retuns the color code for an inputted color name", {"": ""}, lambda **kwargs: "")
]

SUBTASK_RESULTS_FORMAT_LINE = """[{"task": "description of a finished task", "result": "result of the finished task"}, ...]"""

TOOL_VERBOSE_FORMAT_LINE = """{"tool_name": "name of tool", "description": "describes the tool", "input format": {"parameter name": "description per input parameter"}}"""


# stages


tools_concise_ffff = make_ffff(
    {"name of tool": "description of tool"},
    preshape_fn=lambda tools: {tool.name: tool.doc for tool in tools}
)

tool_verbose_ffff = make_ffff(
    {"tool_name": "name of tool", "description": "description of tool", "input_format": {"parameter name": "description per input parameter"}},
    preshape_fn=lambda tool: {'tool_name': tool.name, 'description': tool.doc, 'input_format': tool.input_doc}
)

task_description_ffff = make_ffff(
    "description of the task",
    encoding=Encoding.rawtextline
)

task_descriptions_ffff = ffff_as_list(task_description_ffff)

task_result_ffff = make_ffff(
    {"task": "description of a finished task", "result": "result of the finished task"}
)

task_results_ffff = ffff_as_list(task_result_ffff)

tool_run_record = make_ffff(
    {"tool": "tool name", "input": "input to tool", "output": "the response of the tool"}
)

tool_run_history = ffff_as_list(tool_run_record)

user_chat_QA_ffff = make_ffff(
    {"question": "First question asked by user on a previous query.", "response": "response from the agent for the first question"}
)

user_chat_history_ffff = ffff_as_list(user_chat_QA_ffff)

yes_no_ffff = make_ffff(
    "true or false"
)

###

OBJECTIVITY_RULES = [
    "Do not invent facts or assume missing information.",
    "Use neutral, imperative phrasing."
    "NEVER explain your reasoning."
]

split_user_question = make_stage(
    "split_user_question",
    """You are a task-splitting system.\n\nYour job is to decompose the user Question into the minimal ordered list of atomic Tasks required to answer it.""",
    rules = OBJECTIVITY_RULES + [
        "Do NOT answer the Question.",
        "Tasks must be strictly ordered.",
        "Do not merge multiple actions into one task.",
        "If required information is missing, create a task to retrieve it.",
        "Prefer fewer tasks over more.",
        "Do NOT say which Tools to use in the Tasks."
    ],
    inputs=[
        ("Tools", tools_concise_ffff),
        ("Question", "the question (i.e. main task) from the user to be splitted into (sub) tasks")
    ],
    output=("Tasks", task_descriptions_ffff),
    good_examples=[
        {
            'Tools': EXAMPLE_TOOLS, 
            'Question': "What is the sqare of the year the coworker Mike was born?", 
            'Tasks': ["Find out the date of birth for coworker Mike.", "Square the year number of the date of birth."]
        }
    ],
    bad_examples=[
        {
            'Tools': EXAMPLE_TOOLS, 
            'Question': "What kind of wood is heavy?", 
            'Tasks': ["Find a heavy kind of wood using the wood tool."]
        }
    ])

print("sys prompt split user q\n", split_user_question.sys_prompt)

enhance_task_description = make_stage(
    "enhance_task_description",
    "You are a task reinterpreter.\n" +
        "Your job is to generate an Enhanced task description, based on the current Task description and enhanced with the knowledge of the Results of the finished tasks.\n" +
        "The task is answered in the following stages only by the Enhanced description you generate. Keep all necessary information in.",
    rules = OBJECTIVITY_RULES + [
        "If the Results are unnecessary verbose, summarize them and filter out relevant information."
    ],
    inputs=[
        ("Tools", tools_concise_ffff),
        ("Task", "the description of the current Task"),
        ("Results", task_results_ffff)
    ],
    output=("Enhanced", "your enhanced task description in plain text"),
    good_examples=[
        {
            'Tools': EXAMPLE_TOOLS,
            'Task': "Determine when the author of Some Example Book was born.",
            'Results': [TaskResult("Determine the author of Some Example Book", "The author is James B. Clark")],
            'Enhanced': "Determine when James B. Clark, the author of Some Example Book, was born."
        }
    ]
)

answer_user_question = make_stage(
    "answer_user_question",
    "You are a helpful answer generator.\n\nYour job is to answer the user Question by using the Results of the subtasks.",
    rules = [],
    inputs=[
        ("History", user_chat_history_ffff),
        ("Question", "the user Question"),
        ("Results", task_results_ffff)
    ],
    output=("Answer", "your answer of the Question"),
    good_examples=[
        {
            'History': [],
            'Question': "Do Mark and Elisa drive the same type of car?",
            'Results': [
                TaskResult("Determine what type of car Mark drives.", "Mark drives an Audi A4."), 
                TaskResult("Determine what type of car Elisa drives.", "Elisa drives both a Tesla Model 3 and a Audi A4.")
            ],
            'Answer': "Yes, both drive a Audi A4."
        }
    ]
)

choose_tool_for_task = make_stage(
    "choose_tool_for_task",
    "You are a tool choser.\n\nYour job is to select one of the Tools to be called next. The History shows the previous Tool runs including their inputs and outputs.",
    rules=["If the Task can be answered without a tool, output *just* the JSON keyword null!"],
    inputs=[("Tools", tools_concise_ffff), ("Task", task_description_ffff), ("History", tool_run_history)],
    output=("Chosen Tool", "name of the tool you chose"),
    good_examples=[
        {
            'Tools': EXAMPLE_TOOLS,
            'Task': "Square the number of unknown guests. There were 123 unknown guest at the party.",
            'History': [],
            'Chosen Tool': "sqr_math"
        }
    ]
)

generate_tool_input = make_stage(
    "generate_tool_input",
    "You are an input generator.\n\nYour job is to parse the Task description and generate a valid Input for the provided Tool.",
    rules=["Format the Input according to the input_format of the Tool."],
    inputs=[('Task', task_description_ffff), ('Tool', tool_verbose_ffff)],
    output=('Input', make_ffff("your generated JSON-encoded input", encoding=Encoding.JSON)),
    good_examples=[
        {'Task': "Find the square of 123.", 'Tool': EXAMPLE_TOOLS[0], 'Input': {'number': 123}}
    ],
    bad_examples=[
        {'Task': "What is the square of one-hundred and three.", 'Tool': EXAMPLE_TOOLS[0], 'Input': {'number': "103"}}
    ]
)



answer_task = make_stage(
    "answer_task",
    "You are a context summarizer.\n\nYour job is to Answer the Task using the History of performed tool calls.",
    rules=[],
    inputs=[('Task', task_description_ffff), ('History', tool_run_history)],
    output=('Answer', "Your answer of the Task"),
    good_examples=[
        {
            'Task': "Square the number of unknown guests. There were 123 unknown guest at the party.",
            'History': [ToolRunRecord("sqr_math", {'number': 123}, 15129)],
            'Answer': "The square of the number of 123 unknown guests is 15129."
        }
    ]
)

can_answer_task = make_stage(
    "can_answer_task",
    "Your job is to determine if the Task can be answered by the information obtained.",
    rules=[],
    inputs=[('Task', task_description_ffff), ('History', tool_run_history)],
    output=('Answerable', yes_no_ffff),
    good_examples=[
        {
            'Task': "Square the number of unknown guests. There were 123 unknown guest at the party.",
            'History': [ToolRunRecord("sqr_math", {'number': 123}, 15129)],
            'Answerable': True
        },
        {
            'Task': "Find out who wrote Some Example Book.",
            'History': [ToolRunRecord("sqr_math", {'number': 123}, 15129), ToolRunRecord("book_db", {"title": "Some Example Book"}, [])],
            'Answerable': False
        }
    ]
)


# main

def run_subtask(task_description: str, tools: List[Tool], subtask_results: List[TaskResult], llm: LLM, caller_tracer: Tracer) -> str:
    task_tracer = caller_tracer.on("run_subtask", locals())
    tool_runs: List[ToolRunRecord] = []

    enhanced_description = enhance_task_description({'Task': task_description, 'Tools': tools, 'Results': subtask_results}, llm, task_tracer)
    for i in range(10):
        tracer = task_tracer.on("task_round", {'round': i})
        chosen_tool_name = choose_tool_for_task({'Task': enhanced_description, 'Tools': tools}, llm, tracer)
        chosen_tool = next((tool for tool in tools if tool.name == chosen_tool_name), None)
        if chosen_tool is None:
            break
        tool_input = generate_tool_input({'Task': enhanced_description, 'Tool': chosen_tool}, llm, tracer)
        tool_result = run_tool(chosen_tool, tool_input, tracer)
        tool_runs.append(ToolRunRecord(chosen_tool.name, tool_input, tool_result))
        can_answer = can_answer_task({'Task': enhanced_description, 'History': tool_runs}, llm, tracer)
        if can_answer != False:
            break
    subanswer = answer_task({'Task': enhanced_description, 'History': tool_runs}, llm, task_tracer)
    task_tracer.on("answer", {'answer': subanswer})
    return str(subanswer)


def run_query(user_question: str, history: List[ChatQA], tools: List[Tool], llm: LLM, agent_tracer: Tracer) -> str:
    tracer = agent_tracer.on("run_query", locals())
    subtask_results: List[TaskResult] = []
    
    subtasks = split_user_question({'Question': user_question, 'Tools': tools}, llm=llm, tracer=tracer)

    for subtask in subtasks:
        result = run_subtask(subtask, tools, subtask_results, llm, tracer)
        subtask_results.append(TaskResult(subtask, result))

    answer = str(answer_user_question({'Question': user_question, 'History': history, 'Results': subtask_results}, llm, tracer))

    tracer_finish_with_response(tracer, answer)
    return answer