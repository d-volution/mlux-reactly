from typing import Callable, Any, Tuple, List, Dict
from dataclasses import dataclass, asdict, is_dataclass, replace, field
from enum import Enum
import json
import ollama
from .types import Ctx, LLM, Tracer, ZeroTracer, _UNUSED_sentinel


@dataclass
class FormatDescr:
    template: str
    rules: List[str]
    default: Any
    preshape_fn: Callable[[Any], Any] | None
    preshape_requires_type: type|None
    label: str

    def as_list(self, label: str = '', *, add_rules: List[str] = []):
        return FormatDescr(
            template=f"[{self.template}]",
            rules=self.rules + add_rules,
            preshape_fn=(lambda elements: [self.preshape_fn(element) for element in elements]) 
            if self.preshape_fn is not None else None,
            preshape_requires_type=list,
            default=[],
            label=label
        )

    def with_label(self, new_label: str) -> 'FormatDescr':
        return replace(self, label=new_label)

def make_format(
        template: Any,
        *, 
        rules: List[str] = [],
        default: Any = None,
        preshape_fn: Callable[[Any], Any] | None = None,
        preshape_requires_type: type|None = None,
        label: str = ""
    ) -> FormatDescr:
    return FormatDescr(
        template=format_data_explicit(template, None, ctx='make_format_descr:' + label),
        rules=rules,
        default=default,
        preshape_fn=preshape_fn,
        preshape_requires_type=preshape_requires_type,
        label=label,
    )

@dataclass
class Stage:
    name: str
    static_prompt: str
    input_formats: List[FormatDescr]
    output_format: FormatDescr
    tries: int
    or_return: Any # return this if stage failed



def call_llm(sys_prompt: str, conversation_section: str, llm: LLM, *, tracer: Tracer) -> str:
    tracer = tracer.on('llmcall', {'sys_prompt': sys_prompt, 'prompt': conversation_section})

    messages = []
    if sys_prompt:
        messages.append({'role': 'system', 'content': sys_prompt})
    messages.append({'role': 'user','content': conversation_section})


    #print(f"<----PROMPT VON HIER>{conversation_section}<----PROMPT BIS HIER>")



    response_ollama = ollama.chat(
            model=llm.model,
            messages=messages,
    )
    response_content = str(response_ollama["message"]["content"])
    #print(f"<----VON HIER>{'<-|||->'.join([m['content'] for m in messages])}<----BIS HIER>")
    

    tracer.on('complete', {'result': response_content})
    return response_content



def make_json_serializable(data: Any):
    if type(data) == list:
        return [make_json_serializable(element) for element in data]
    if isinstance(data, dict):
        return {k: make_json_serializable(v) for k, v in data.items()}
    if is_dataclass(data) and not isinstance(data, type):
        return make_json_serializable(asdict(data))
    if type(data) in [str, int, float, bool]:
        return data
    if hasattr(data, "tolist"):
        return make_json_serializable(data.tolist())
    return str(data)

def serialize_data(data) -> str:
    serializable = make_json_serializable(data)
    return json.dumps(serializable)


def format_data_explicit(data, preshape_fn: Callable[[Any], Any] | None, *, ctx: str = "", preshape_required: bool = True) -> str:
    try:
        data = data if preshape_fn is None else preshape_fn(data)
    except BaseException as e:
        if preshape_required:
            e.add_note(f"in {ctx}: format_data_explicit: preshape")
            raise e
    try:
        return serialize_data(data)
    except BaseException as e:
        e.add_note(f"in {ctx}: format_data_explicit: serialize")
        raise e
            

def format_data(data, format: FormatDescr, *, ctx: str = "", preshape_required: bool = True) -> str:
    return format_data_explicit(data, format.preshape_fn, ctx=ctx, preshape_required=preshape_required)



def generate_conversation(data: Dict[str, Any], inputs: List[FormatDescr], output: FormatDescr, with_output: bool = False, ctx: str = ""):
    unhandled_input_labels = set(data.keys())

    conversation_section: str = ""
    for input in inputs:
        input_value = data.get(input.label, input.default)
        unhandled_input_labels.discard(input.label)
        conversation_section += f"{input.label}: {format_data(input_value, input, ctx=ctx+f": generate_conversation input {input.label}")}\n"

    conversation_section += f"{output.label}: "
    if with_output:
        output_value = data.get(output.label)
        unhandled_input_labels.remove(output.label)
        conversation_section += format_data(output_value, output,  ctx=ctx+f": generate_conversation output {output.label}", preshape_required=False)+'\n'

    if len(unhandled_input_labels) > 0:
        e = ValueError(ctx+f" got input values with unknown labels")
        e.add_note(f"unknown labels: {unhandled_input_labels}")
        raise e

    return conversation_section


def generate_static_prompt(
        stage_description: str, 
        rules: List[str],
        inputs: List[FormatDescr],
        output: FormatDescr,
        good_examples: List[Dict[str, Any]],
        bad_examples: List[Dict[str, Any]]) -> str:
    
    prompt_head = f"{stage_description}\n"

    all_rules = [
        "Output valid JSON. No extra text!",
        #"Output a single line of valid JSON. No extra text!",
        #"STRICTLY follow the Format!"
    ]
    all_rules.extend(rules)
    for input in inputs:
        all_rules.extend(input.rules)

    rules_section = "# Rules\n\n"
    for rule in all_rules:
        rules_section += f"- {rule}\n"

    format_section = "# Format\n\nThe format looks like this:\n\n"
    for input in inputs:
        format_section += f"{input.label}: {input.template}\n"
    format_section += f"{output.label}: {output.template}\n"

    good_examples_section = "# GOOD Examples\n\n" if len(good_examples) > 0 else ""
    for example in good_examples:
        good_examples_section += generate_conversation(example, inputs=inputs, output=output, with_output=True, ctx="system_prompt good example") + "\n"

    bad_examples_section = "# BAD Examples\n\n" if len(bad_examples) > 0 else ""
    for example in bad_examples:
        bad_examples_section += generate_conversation(example, inputs=inputs, output=output, with_output=True, ctx="system_prompt bad example") + "\n"

    sys_prompt = "\n".join([sec for sec in [prompt_head, rules_section, format_section, good_examples_section, bad_examples_section, "# Conversation\n"] if sec != ""])
    return sys_prompt







def as_format_descr(original: Tuple[str, FormatDescr|str]|FormatDescr) -> FormatDescr:
    if isinstance(original, FormatDescr):
        return original
    elif isinstance(original, tuple):
        if isinstance(original[1], FormatDescr):
            return original[1].with_label(original[0])
        else:
            return make_format(template=original[1], label=original[0])




def make_stage(
        stage_name: str, 
        stage_description: str, 
        *, 
        rules: List[str] = [],
        inputs: List[Tuple[str, FormatDescr|str]|FormatDescr],
        output: Tuple[str, FormatDescr|str]|FormatDescr,
        tries: int = 1,
        good_examples: List[Dict[str, Any]] = [],
        bad_examples: List[Dict[str, Any]] = [],
        or_return: Any = _UNUSED_sentinel
):
    inputs = input_formats = [as_format_descr(format) for format in inputs]
    output = output_format = as_format_descr(output)
    return Stage(
        stage_name,
        generate_static_prompt(stage_description, rules, input_formats, output_format, good_examples, bad_examples),
        input_formats,
        output_format,
        tries,
        or_return
    )


def run_stage(stage: Stage, input_data: Dict[str, Any], llm: LLM, tracer: Tracer, *, post_fn: Callable|None = None):
    tracer = tracer.on("stage", {'name': stage.name})

    try:
        conversation_section = generate_conversation(input_data, inputs=stage.input_formats, output=stage.output_format, ctx="stage_run")
    except Exception as e:
        tracer.on('failed', {'reason_code': 'generate_conversation', 'exception': e})
        return stage.or_return

    last_err: Exception | None = None
    err_reason_code = ''
    for try_nr in range(stage.tries):
        try:
            try_tracer = tracer.on('try', {'nr': try_nr})

            err_reason_code = 'llm'
            llm_response = call_llm(stage.static_prompt, conversation_section, llm, tracer=try_tracer)
            err_reason_code = 'json_parsing'
            parsed_result = json.loads(llm_response)

            err_reason_code = 'post_processing'
            if post_fn is not None:
                result = post_fn(parsed_result)
            else:
                result = parsed_result
            tracer.on('complete', {'result': result})
            return result
        except Exception as e:
            last_err = e
            try_tracer.on('failed', {'reason_code': err_reason_code, 'exception': e})
    assert last_err is not None
    tracer.on('failed', {'reason_code': err_reason_code, 'exception': last_err, 'tries': stage.tries})
    if stage.or_return is _UNUSED_sentinel:
        raise last_err
    else:
        return stage.or_return