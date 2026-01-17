from typing import Any, List, Dict
from enum import Enum
from dataclasses import dataclass, asdict
from .types import LLM, Tool, Task, TaskResult, ChatQA, Tracer
from .stages import split_question_into_tasks, enhance_task_description, rate_tools_for_task, make_tool_input, try_answer, ToolRunRecord






def run_tool(tool: Tool, input: Dict[str, Any], caller_tracer: Tracer) -> Any:
    tracer = caller_tracer.on("toolrun", {'tool': tool})
    try:
        result = tool.run(**input)
        tracer.on("complete", {'result': result})
    except Exception as e:
        result = f"tool failed: {e}"
        tracer.on("failed", {'result': result, 'exception': e})
    return result




def run_query(user_question: str, history: List[ChatQA], tools: List[Tool], llm: LLM, agent_tracer: Tracer) -> str:
    query_tracer = agent_tracer.on("query", {'user_question': user_question})

    tasks: List[Task] = split_question_into_tasks(user_question, tools, llm, query_tracer)
    #print('Tasks::', tasks)
    task_results: List[TaskResult] = []

    for original_task in tasks:
        tracer = query_tracer.on("task", {'task': original_task})
        tool_results: List[ToolRunRecord] = []

        task = Task(enhance_task_description(original_task.description, task_results, llm, tracer))

        rated_tools = rate_tools_for_task(task, tools, llm, tracer, include_model=True)
        #print('rated::', {t.tool.name: t.score for t in rated_tools})
        
        rated_tools.sort(key=lambda rt: -rt.score)
        #print('rated sorted::', {t.tool.name: t.score for t in rated_tools})

        
        for rated_tool in rated_tools:
            if rated_tool.score < 0.5:
                break

            tool_input = make_tool_input(task, rated_tool.tool, llm, tracer)
            #print('input::', tool_input)

            tool_result = run_tool(rated_tool.tool, tool_input, tracer)
            #print('tool result::', action_result)
            tool_results.append(ToolRunRecord(rated_tool.tool, tool_input, tool_result))

        task_answer = try_answer(task.description, tool_results, llm, tracer)
        task_results.append(TaskResult(task.description, task_answer))
        #print('answer::', task.answer)
        

    answer = try_answer(user_question, [ToolRunRecord('subtask', t.task, t.result) for t in task_results], llm, query_tracer)
    query_tracer.on('query_answer', {'answer': answer.answer})
    return answer.answer
