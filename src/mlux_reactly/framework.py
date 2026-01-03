from typing import Callable, Any, Tuple, List
from dataclasses import dataclass, asdict, is_dataclass
from enum import Enum
import json
import ollama
from .types import LLM, Tracer

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
    return response["message"]["content"]




def make_json_serializable(data):
    if type(data) == list:
        return [make_json_serializable(element) for element in data]
    if is_dataclass(data):
        return asdict(data)
    return data



@dataclass
class FFFF:
    """FFFF something about format"""
    format_template: str
    encoding: Encoding
    rules: List[str] # FFF specific rules
    preshape_fn: Callable[[Any], Any] | None


def serialize_data(data, encoding: Encoding) -> str:
    if encoding == Encoding.JSON:
        serializable = make_json_serializable(data)
        return json.dumps(serializable)
    else:
        return str(data)

def format_data_explicit(data, preshape_fn: Callable[[Any], Any] | None, encoding: Encoding) -> str:
    data_preshaped = data if preshape_fn == None else preshape_fn(data)
    return serialize_data(data_preshaped, encoding)

def format_data(data, ff: FFFF) -> str:
    return format_data_explicit(data, ff.preshape_fn, ff.encoding)


def make_ffff(template: Any, encoding = Encoding.JSON, rules = [], preshape_fn = None) -> FFFF:
    return FFFF(serialize_data(template, encoding), encoding=encoding, rules=rules, preshape_fn=preshape_fn)

def ffff_as_list(ff: FFFF) -> FFFF:
    return FFFF(
        format_template=f"[{ff.format_template}, ...]",
        encoding=Encoding.JSON,
        rules=ff.rules
    )


def generate_conversation(*args, inputs: List[Tuple[str, FFFF]], output: Tuple[str, FFFF], with_output: bool = False):
    required_len_of_args = len(inputs) + (1 if with_output else 0)
    if len(args) != required_len_of_args:
        raise ValueError(f"incompatible number of input arguments. should be {required_len_of_args}, is {len(args)}")

    conversation_section: str = ""
    for i in range(len(inputs)):
        input_value = args[i]
        input_label = inputs[i][0]
        input_ff = inputs[i][1]
        conversation_section += f"{input_label}: {format_data(input_value, input_ff)}\n"
    
    output_label = output[0]
    output_ff = output[1]
    conversation_section += f"{output_label}: "
    if with_output:
        conversation_section += format_data(args[len(inputs)], output_ff)
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
        good_examples: List[Tuple],
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
        good_examples_section += generate_conversation(*example, inputs=inputs, output=output, with_output=True) + "\n"

    sys_prompt = "\n".join([sec for sec in [prompt_head, rules_section, format_section, good_examples_section, "# Conversation\n"] if sec != ""])

    return sys_prompt




@dataclass
class Stage:
    name: str
    sys_prompt: str
    inputs: List[Tuple[str, FFFF]]
    output: Tuple[str, FFFF]
    tries: int
    llm: LLM

    def __call__(self, *args, tries: int|None = None):
        if len(args) != len(self.inputs):
            raise ValueError(f"stage_call_error: {self.name}: incompatible number of input arguments")

        conversation_section = generate_conversation(*args, inputs=self.inputs, output=self.output)

        for try_nr in range(tries or self.tries):
            context = [
                CtxMessage(Role.System, self.sys_prompt),
                CtxMessage(Role.User, conversation_section)
            ]
            llm_response = call_llm(context, self.llm)
            
            deserialized = llm_response if self.output[1].encoding == Encoding.rawtextline else json.loads(llm_response)
            return deserialized



def make_stage(
        stage_name: str, 
        stage_description: str, 
        *, 
        rules: List[str] = [],
        inputs: List[Tuple[str, FFFF]],
        output: Tuple[str, FFFF],
        tracer: Tracer,
        llm: LLM,
        tries: int = 0,
        good_examples: List[Tuple]) -> Stage:
    args = locals()
    local_tracer = tracer.on("stage_" + stage_name, {})

    sys_prompt = generate_sys_prompt(**args)
    stage = Stage(
        stage_name, 
        sys_prompt, 
        inputs=inputs,
        output=output, 
        tries=tries, 
        llm=llm)
    return stage


