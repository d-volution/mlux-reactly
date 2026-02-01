from typing import List, Dict, Any
from dataclasses import dataclass

from .types import AnyBasic, T, Task, TaskResult, Answer, LLM, Tracer, Tool, ChatQA
from .framework import make_format, FormatDescr, make_stage, run_stage




@dataclass
class RatedTool:
    tool: Tool
    score: float

@dataclass
class ToolRunRecord:
    tool: str
    input: Any
    result: Any

_qa_format = make_format(
    {'question': 'a question from the user', 'answer': 'answer from the agent'},
    preshape_fn=lambda qa: {'question': qa.question, 'answer': qa.response}
)

_tools_concise_format = make_format(
    {"name of tool": "description of tool"},
    preshape_fn=lambda tools: {tool.name: tool.doc for tool in tools},
    label='Tools'
)

_tool_verbose_format = make_format(
    {"tool_name": "name of tool", "description": "description of tool", "input_format": {"parameter name": "description per input parameter"}},
    preshape_fn=lambda tool: {'tool_name': tool.name, 'description': tool.doc, 'input_format': tool.input_doc},
    label='Tool'
)

_task_description_format = make_format(
    "description of the task",
    label='Task'
)

_task_specification_format = make_format(
    {'description': 'description of the task', 'tools': ['name of tool 1', '...']},
    label='Tool'
)

_task_result_format = make_format(
    {'task': 'description of the task', 'result': 'the task result (i.e. answer)'},
    preshape_fn=lambda task: {'task': task.task, 'result': task.result},
    label='Result'
)

_rated_tools_format = make_format(
    {"name of tool1": 0.123, "tool2": 0.456},
    preshape_fn=lambda rated_tools: {rated_tool.tool.name: rated_tool.score for rated_tool in rated_tools},
    preshape_requires_type=RatedTool,
    label='Ratings'
)

_tool_runs_format = make_format(
    [{'tool': 'name of tool', 'input': 'input to tool (does not have to be a string)', 'result': 'the output of the tool run'}],
)

_answer_format = make_format('The answer of the Task.', label='Answer', is_plain_text=True)



_EXAMPLE_TOOL_SQR = Tool("sqr_math", "A tool to square any real number.", {"number": "The number that should be squared"})
_EXAMPLE_TOOL_WOOD = Tool("wood", "This tool returns basic properties of a kind of wood", {"kind": "name of wood kind"})
_EXAMPLE_TOOL_COWORKDB = Tool("cowork_db", "DB to look up coworker info.", {"person_name": "name of coworker"})
_EXAMPLE_TOOL_HEXCOLOR = Tool("hex_color", "Retuns the color code for an inputted color name", {"color_name": "name of color"})
_EXAMPLE_TOOLS: List[Tool] = [_EXAMPLE_TOOL_SQR, _EXAMPLE_TOOL_WOOD, _EXAMPLE_TOOL_COWORKDB, _EXAMPLE_TOOL_HEXCOLOR]

_EXAMPLE_TOOL_RATINGS = [
    RatedTool(_EXAMPLE_TOOL_SQR, 0.41),
    RatedTool(_EXAMPLE_TOOL_WOOD, 0.0),
    RatedTool(_EXAMPLE_TOOL_COWORKDB, 0.02),
    RatedTool(_EXAMPLE_TOOL_HEXCOLOR, 0.96),
]



OBJECTIVITY_RULES = [
    "Do not invent facts or assume missing information.",
    "Use neutral, imperative phrasing.",
    "NEVER explain your reasoning."
]









_split_question_into_tasks_stage = make_stage(
    'split_question_into_tasks',
    """You are a task-splitting system.\nYour job is to decompose the user Question into an ordered list of atomic Tasks required to answer it.""",
    rules = OBJECTIVITY_RULES + [
        "Do NOT answer the Question.",
        "Tasks must be strictly ordered.",
        "Do not merge multiple actions into one task.",
        "If required information is missing, create a task to retrieve it.",
        "Each Task should only need a single Tool call to answer it.",
        "Prefer many small tasks over few large tasks. For example, if the Question asks about different people or places, use one Task for each.",
    ],
    inputs = [
        ('Question', 'The user question'),
        _tools_concise_format,
    ],
    output=_task_description_format.as_list(label='Tasks'),
    good_examples=[
        {
            'Tools': _EXAMPLE_TOOLS, 
            'Question': "What is the sqare of the year the coworker Mike was born?",
            'Tasks': ['Find out the date of birth for coworker Mike', "Square the year number of the date of birth."],
        },
         {
            'Tools': _EXAMPLE_TOOLS, 
            'Question': "How much more does a ton of oak wood cost than a ton of beech wood in Jessie's home country?",
            'Tasks': [
                "Determine the home country of Jessie.", 
                "Determine the price of a ton of oak wood in Jessie's home country.", 
                "Determine the price of a ton of beech wood in Jessie's home country.",
                "Calculate the difference between the prices of wood."
            ],
        }
    ],
    tries = 3,
    or_return=[]
)

def split_question_into_tasks(user_question: str, tools: List[Tool], llm: LLM, tracer: Tracer) -> List[Task]:
    def post_process(parsed: List[str]) -> List[Task]:
        return [Task(description) for description in parsed]

    tasks: List[Task] = run_stage(_split_question_into_tasks_stage, {'Question': user_question, 'Tools': tools}, llm, tracer, post_fn=post_process)
    return tasks










_enhance_user_question_stage = make_stage(
    'enhance_user_question',
    """You are a question enhancing stage.\nYour job is to Enhance the current user Question using information of previous interactions in the chat History."""
    """The following stages will try to answer the Question using only your Enhanced question.""",
    rules=OBJECTIVITY_RULES+[
        "Include all informations from previous chat interactions *relevant* for answering this user Question."
    ],
    inputs=[
        _qa_format.as_list('History'),
        ('Question', 'the user question'),
    ],
    output=('Enhanced', 'Your enhanced user Question'),
    good_examples=[
        {
            'History': [
                ChatQA('Where does the prime minister live?', '10 Downing Street, London.'),
                ChatQA('How old is he?', "51 years"),
                ChatQA('Does he pay rent there?', "No, it's an official residence paided by the state.")
            ],
            'Question': "How much rent would someone pay for such a residence?",
            'Enhanced': "How much rent would someone pay for a residence like the one that the Prime Minister lives in (in 10 Downing Street, London where the rent is paid for him by the state)."
        }
    ],
    or_return="",
    tries=3
)

def enhance_user_question(user_question: str, history: List[ChatQA], llm: LLM, tracer: Tracer) -> str:
    if len(history) == 0:
        return user_question
    enhanced = run_stage(_enhance_user_question_stage, {'History': history, 'Question': user_question}, llm, tracer)
    return enhanced or user_question








_enhance_task_description_stage = make_stage(
    'enhance_task_description',
    """You are a task enhancing stage.\nYour job is to Enhance the description of a Task with previous Results. The following stages will try to answer the task using only your Enhanced description.""",
    rules=OBJECTIVITY_RULES+[
        "If the Results are unnecessary verbose, summarize them and filter out relevant information.",
        "The Enhanced description should explicitly state which information is asked for."
    ],
    inputs=[
        _task_result_format.as_list('Results'),
        _task_description_format,
    ],
    output=('Enhanced', 'Your task description enhanced by the relevant knowledge learned from the Results'),
    good_examples=[
        {
            'Results': [TaskResult("Determine the author of Some Example Book", "The author is James B. Clark")],
            'Task': "Determine when the author of Some Example Book was born.",
            'Enhanced': "Determine when James B. Clark, the author of Some Example Book, was born."
        },
        {
            'Results': [
                TaskResult("What regulations exist for playing grounds.", "Documents EXAMPLE-123 and EXAMPLE-456 regulate indoor playing grounds."),
                TaskResult("Determine the capital of contry X", "The capital of country X is 'Bani'. Five months ago there was a rock concert in Bani."),
            ],
            'Task': "Find the population of the capital of country X.",
            'Enhanced': "Find the population of 'Bani', the capital of country X."
        }
    ]
)

def enhance_task_description(task_descr: str, previous_tasks: List[TaskResult], llm: LLM, tracer: Tracer) -> str:
    return run_stage(_enhance_task_description_stage, {'Results': previous_tasks, 'Task': task_descr}, llm, tracer)






_rate_tools_for_task_stage = make_stage(
    'rate_tools_for_task',
    """You are a tool rater.\nYour job is to rate each Tool with a score between 0.0 and 1.0, regarding how well it is suited for answering the Task. 0.0 means bad for task, 1.0 means well suited.""",
    rules = OBJECTIVITY_RULES + [
        'Rate every available Tool.'
    ],
    inputs = [
        ('Task', _task_description_format),
        _tools_concise_format
    ],
    output=_rated_tools_format,
    good_examples=[
        {
            'Task': 'Lookup the ANSI color code of magenta.',
            'Tools': _EXAMPLE_TOOLS,
            'Ratings': _EXAMPLE_TOOL_RATINGS,
        }
    ],
    tries = 2
)


def rate_tools_for_task(task: Task, tools: List[Tool], llm: LLM, tracer: Tracer) -> List[RatedTool]:
    available_tools_by_name = {t.name: t for t in tools}
    def post_process(parsed: AnyBasic) -> List[RatedTool]:
        result: List[RatedTool] = []
        assert isinstance(parsed, dict)
        for tool_name, score in parsed.items():
            tool = available_tools_by_name.get(tool_name)
            assert tool is not None
            result.append(RatedTool(tool, score))
        return result


    return run_stage(_rate_tools_for_task_stage, {'Task': task.description, 'Tools': tools}, llm, tracer, post_fn=post_process)








_make_tool_input_stage = make_stage(
    'make_tool_input',
    """You are an input generator.\n\nYour job is to parse the Task description and generate a valid Input for the provided Tool.""",
    rules = OBJECTIVITY_RULES + [
        "Format the Input according to the input_format of the Tool."
    ],
    inputs = [
        ('Task', _task_description_format),
        _tool_verbose_format
    ],
    output=('Input', make_format({'first parameter name': "some input value (of JSON-type specified by input format)", "another parameter name": "another input value"})),
    good_examples=[
        {'Task': "Find the square of 123.", 'Tool': _EXAMPLE_TOOL_SQR, 'Input': {'number': 123}}
    ],
    tries = 2
)

def make_tool_input(task: Task, tool: Tool, llm: LLM, tracer: Tracer) -> AnyBasic:
    return run_stage(_make_tool_input_stage, {'Task': task.description, 'Tool': tool}, llm, tracer)








_try_answer_task_stage = make_stage(
    'try_answer_task',
    """You are a answer generator of an agent.\nYour job is to Answer a Task only using the information from the Results.""",
    rules = OBJECTIVITY_RULES + [
        "Do NOT invent any facts or missing information.",
    ],
    inputs = [
        ('Results', _tool_runs_format),
        ('Task', _task_description_format),
    ],
    output=_answer_format,
    good_examples=[
        {
            'Task': "Find the names of daughter and brother of Joe.", 
            'Results': [{'tool': 'people_tool', 'input': 'Joe', 'result': {'name': 'Joe', 'brother': 'Maximilian'}}], 
            'Answer': 'The brother of Joe is Maximilian.'
        },
        {
            'Task': "Who wrote The Blue-Pink Song?", 
            'Results': [
                {'tool': 'song_db', 'input': 'The Blue-Pink Song', 'result': {'title': 'The Blue-Pink Song', 'author': 'Helene Josh'}},
                {'tool': 'music_tool', 'input': {'search': 'The Blue-Pink Song'}, 'result': {'work': 'The Blue-Pink Song', 'author': 'Laura Taylor'}}
            ], 
            'Answer': 'The Blue-Pink Song was probably written by Helene Josh or Laura Taylor.'
        },
        {
            'Task': "Determine the sensor size of a A123-camera.", 
            'Results': [
                {'tool': 'tech_lookup', 'input': {'topic': 'cameras', 'model': 'A123'}, 'result': {'full_name': 'Ycam Solo A123', 'weight_kg': 1.5, 'sensor_size': 'full-frame'}},
                {'tool': 'some_geo_tool', 'input': 'A123', 'result': 'The highway A123 in Australia connects cities in the north and south.'},
            ], 
            'Answer': "The A123-camera uses a 'full-frame' sized sensor."
        },
        {
            'Task': "What movie stares Mike Simon in the role of Roy.", 
            'Results': [
                {'tool': 'cinema', 'input': {'query': 'Mike Simon as Roy'}, 'result': [
                    {'page': 'Mike Simon (actor)', 'text': "Mike L. Simon is an US-american actor from New Jersey. He is known for various roles in TV and cinema."},
                    {'page': 'Finish Winter Comes Back (movie)', 'text': "Finish Winter Comes Back is an action thriller, where a teenager called 'Roy' breaks into his own home."},
                ]}
            ], 
            'Answer': "The movie 'Finish Winter Comes Back' probably starres Mike Simon."
        },
    ],
    tries = 2
)

def try_answer(task_description: str, results: List[ToolRunRecord], llm: LLM, tracer: Tracer) -> str:
    return run_stage(_try_answer_task_stage, {'Task': task_description, 'Results': results}, llm, tracer)






_rate_task_answer_stage = make_stage(
    'rate_task_answer',
    """You are an Answer rater of an agent.\nYour job is to rate an Answer for a given Task with a satisfaction score between 0 and 1.""",
    rules = OBJECTIVITY_RULES + [
        "Do not answer the Task."
    ],
    inputs = [
        ('Task', _task_description_format),
        ('Answer', _tool_runs_format),
    ],
    output=('Rating', make_format({'satisfaction': 0.12, 'reason': "Why the answer can't fully satisfy the Task. (This attribute can be ommitted when satisfaction is high.)"})),
    good_examples=[
        {
            'Task': "Find the names of daughter and brother of Joe.", 
            'Answer': 'The brother of Joe is Maximilian.',
            'Rating': {'satisfaction': 0.4, 'reason': 'The name of the daugther is missing.'},
        },
        {
            'Task': "Who wrote The Blue-Pink Song?", 
            'Answer': 'The Blue-Pink Song was probably written by Helene Josh or Laura Taylor.',
            'Rating': {'satisfaction': 0.35, 'reason': 'The results contradict each other regarding the author.'},
        },
        {
            'Task': "Determine the sensor size of a A123-camera.", 
            'Answer': "The A123-camera uses a 'full-frame' sized sensor.",
            'Rating': {'satisfaction': 0.9},
        },
        {
            'Task': "What movie stares Mike Simon in the role of Roy.", 
            'Answer': "The movie 'Finish Winter Comes Back' probably starres Mike Simon.",
            'Rating': {'satisfaction': 0.24, 'reason': 'Results ambiguous about requested information.'},
        },
    ],
    tries = 2
)

def rate_task_answer(task_description: str, answer: str, llm: LLM, tracer: Tracer) -> int:
    def post_process(output: Dict[str, str|float]) -> Answer:
        return output.get('satisfaction', 0.0)
    return run_stage(_rate_task_answer_stage, {'Task': task_description, 'Answer': answer}, llm, tracer, post_fn=post_process)


