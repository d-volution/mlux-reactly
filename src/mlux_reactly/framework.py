from typing import Callable, Any, Tuple, List, Dict
from dataclasses import dataclass, asdict, is_dataclass
from enum import Enum
import json
import ollama
from .types import LLM, Tracer, ZeroTracer

class Role(Enum):
    System = "system"
    User = "user"
    Assistant = "assistant"

@dataclass
class CtxMessage:
    role: Role
    content: str

class Encoding(Enum):
    JSON = "json"
    rawtextline = "rawtextline"


def call_llm(context: List[CtxMessage], llm: LLM) -> str:
    response = ollama.chat(
            model=llm.model,
            messages=[{
                'role': msg.role.name.lower(),
                'content': msg.content
            } for msg in context]
        )
    return str(response["message"]["content"])




def make_json_serializable(data: Any):
    if type(data) == list:
        return [make_json_serializable(element) for element in data]
    if is_dataclass(data) and not isinstance(data, type):
        return asdict(data)
    if type(data) in [str, int, float, bool, dict]:
        return data
    return str(data)


@dataclass
class FFFF:
    """FFFF something about format"""
    format_template: str
    encoding: Encoding
    rules: List[str] # FFF specific rules
    preshape_fn: Callable[[Any], Any] | None
    default: Any


def serialize_data(data, encoding: Encoding) -> str:
    if encoding == Encoding.JSON:
        serializable = make_json_serializable(data)
        return json.dumps(serializable)
    else:
        return str(data)

def format_data_explicit(data, preshape_fn: Callable[[Any], Any] | None, encoding: Encoding, *, ctx: str = "") -> str:
    try:
        data_preshaped = data if preshape_fn is None else preshape_fn(data)
    except BaseException as e:
        e.add_note(f"in {ctx}: format_data_explicit")
        raise e
    return serialize_data(data_preshaped, encoding)

def format_data(data, ff: FFFF, *, ctx: str = "") -> str:
    return format_data_explicit(data, ff.preshape_fn, ff.encoding, ctx=ctx)


def make_ffff(template: Any, encoding = Encoding.JSON, *, rules = [], preshape_fn = None, default: Any = None) -> FFFF:
    return FFFF(serialize_data(template, encoding), encoding=encoding, rules=rules, preshape_fn=preshape_fn, default=default)

def ffff_as_list(ff: FFFF, *, add_rules: List[str] = []) -> FFFF:
    return FFFF(
        format_template=f"[{ff.format_template}, ...]",
        encoding=Encoding.JSON,
        rules=ff.rules + add_rules,
        preshape_fn=(lambda elements: [ff.preshape_fn(element) for element in elements]) if ff.preshape_fn is not None else None,
        default=[]
    )

def generate_conversation(data: Dict[str, Any], inputs: List[Tuple[str, FFFF]], output: Tuple[str, FFFF], with_output: bool = False, ctx: str = ""):
    unhandled_input_labels = set(data.keys())

    conversation_section: str = ""
    for input in inputs:
        input_label = input[0]
        input_ff = input[1]
        input_value = data.get(input_label, input_ff.default)
        unhandled_input_labels.discard(input_label)
        conversation_section += f"{input_label}: {format_data(input_value, input_ff, ctx=ctx+f": generate_conversation input {input_label}")}\n"

    output_label = output[0]
    output_ff = output[1]
    conversation_section += f"{output_label}: "
    if with_output:
        output_value = data.get(output_label)
        unhandled_input_labels.remove(output_label)
        conversation_section += format_data(output_value, output_ff,  ctx=ctx+f": generate_conversation output {output_label}")

    if len(unhandled_input_labels) > 0:
        e = ValueError(ctx+f"got input values with unknown labels")
        e.add_note(f"unknown labels: {unhandled_input_labels}")
        raise e

    return conversation_section




FIXED_STATIC_RULES: List[str] = [
    "NEVER explain your reasoning."
]

def generate_sys_prompt(
        *,
        stage_name: str, 
        stage_description: str, 
        rules: List[str] = [],
        inputs: List[Tuple[str, FFFF]],
        output: Tuple[str, FFFF],
        good_examples: List[Dict[str, Any]],
        **kwargs) -> str:
    
    prompt_head = f"{stage_description}\n"

    all_rules = FIXED_STATIC_RULES[:]
    if output[1].encoding == Encoding.JSON:
        all_rules.append("Output valid JSON. No extra text!")
    all_rules.extend(rules)
    for input in inputs:
        all_rules.extend(input[1].rules)

    rules_section = "# Rules\n\n"
    for rule in all_rules:
        rules_section += f"- {rule}\n"

    format_section = "# Format\n\nThe format looks like this:\n\n"
    for input in inputs:
        format_section += f"{input[0]}: {input[1].format_template}\n"
    format_section += f"{output[0]}: {output[1].format_template}\n"

    good_examples_section = "# GOOD Examples\n\n" if len(good_examples) > 0 else ""
    for example in good_examples:
        good_examples_section += generate_conversation(example, inputs=inputs, output=output, with_output=True, ctx="system_prompt") + "\n"

    sys_prompt = "\n".join([sec for sec in [prompt_head, rules_section, format_section, good_examples_section, "# Conversation\n\n"] if sec != ""])
    return sys_prompt




@dataclass
class Stage:
    name: str
    sys_prompt: str
    inputs: List[Tuple[str, FFFF]]
    output: Tuple[str, FFFF]
    tries: int
    llm: LLM

    def __call__(self, input_data: Dict[str, Any], llm: LLM|None = None, tracer: Tracer = ZeroTracer(), *, tries: int|None = None):
        conversation_section = generate_conversation(input_data, inputs=self.inputs, output=self.output, ctx="stage_run")

        local_tracer = tracer.on("stage_run_" + self.name, {'sys_prompt': self.sys_prompt, 'conversation': conversation_section})

        for try_nr in range(tries or self.tries):
            context = [
                CtxMessage(Role.System, self.sys_prompt),
                CtxMessage(Role.User, conversation_section)
            ]
            llm_response = call_llm(context, llm or self.llm)
            
            deserialized = llm_response.replace("\n", "") if self.output[1].encoding == Encoding.rawtextline else json.loads(llm_response)
            local_tracer.on("stage_result", {'result': deserialized})
            return deserialized

        return None # if no try was successful


def as_labeled_format(original: Tuple[str, FFFF|str]) -> Tuple[str, FFFF]:
    assert(isinstance(original, tuple))
    return original if isinstance(original[1], FFFF) else (original[0], make_ffff(original[1], encoding=Encoding.rawtextline)) 


def make_stage(
        stage_name: str, 
        stage_description: str, 
        *, 
        rules: List[str] = [],
        inputs: List[Tuple[str, FFFF|str]],
        output: Tuple[str, FFFF|str],
        llm: LLM = LLM(""),
        tries: int = 1,
        good_examples: List[Dict[str, Any]]) -> Stage:
    inputs = inputs_ = [as_labeled_format(input) for input in inputs]
    output = output_ = as_labeled_format(output)
    args = locals()

    sys_prompt = generate_sys_prompt(**args)
    stage = Stage(
        stage_name, 
        sys_prompt, 
        inputs=inputs_,
        output=output_, 
        tries=tries, 
        llm=llm)
    return stage


