from typing import Any, List, Dict
from enum import Enum
from dataclasses import dataclass, asdict
from .types import LLM, Tool, Task, TaskResult, ChatQA, Tracer, Answer, AgentConfig
from .stages import enhance_user_question, split_question_into_tasks, enhance_task_description
from .stages import rate_tools_for_task, make_tool_input, try_answer, rate_task_answer, ToolRunRecord






def run_tool(tool: Tool, input: Dict[str, Any], caller_tracer: Tracer) -> Any:
    tracer = caller_tracer.on("toolrun", {'tool': tool})
    try:
        result = tool.run(**input)
        tracer.on("complete", {'result': result})
    except Exception as e:
        result = f"tool failed: {e}"
        tracer.on("failed", {'result': result, 'exception': e})
    return result




def run_query(user_question: str, history: List[ChatQA], tools: List[Tool], llm: LLM, agent_tracer: Tracer, agent_config: AgentConfig) -> str:
    query_tracer = agent_tracer.on("query", {'user_question': user_question})

    enhanced_user_question = enhance_user_question(user_question, history, llm, query_tracer)

    tasks: List[Task] = split_question_into_tasks(enhanced_user_question, tools, llm, query_tracer)
    task_results: List[TaskResult] = []

    for original_task in tasks:
        tracer = query_tracer.on("task", {'task': original_task.description})
        tool_results: List[ToolRunRecord] = []

        task = Task(enhance_task_description(original_task.description, task_results, llm, tracer))

        proposed_task_answers: List[TaskResult] = []
        for try_nr in range(agent_config.max_nr_tries_per_task):
            tracer.on('try', {'nr': try_nr})
            rated_tools = rate_tools_for_task(task, tools, llm, tracer)
            rated_tools.sort(key=lambda rt: -rt.score)

            for rated_tool in rated_tools:
                if rated_tool.score < agent_config.tool_use_rating_threshold:
                    break
                tool_input = make_tool_input(task, rated_tool.tool, llm, tracer)
                tool_result = run_tool(rated_tool.tool, tool_input, tracer)
                tool_results.append(ToolRunRecord(rated_tool.tool.name, tool_input, tool_result))

            task_answer = try_answer(task.description, tool_results, llm, tracer)
            satisfaction = rate_task_answer(task.description, task_answer, llm, tracer)

            if satisfaction >= agent_config.task_answer_satisfaction_threshold:
                task_results.append(TaskResult(task.description, task_answer, satisfaction))
                break
            else:
                proposed_task_answers.append(TaskResult(task.description, task_answer, satisfaction))
                task = Task(enhance_task_description(original_task.description, proposed_task_answers, llm, tracer))
        
    answer = try_answer(enhanced_user_question, [ToolRunRecord('subtask', t.task, t.result) for t in task_results], llm, query_tracer)
    query_tracer.on('complete', {'result': answer})
    return answer
