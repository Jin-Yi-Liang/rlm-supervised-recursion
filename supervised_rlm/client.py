import os
from openai import OpenAI
import json
from rich.console import Console
from rich.markdown import Markdown
import sys
from copy import deepcopy

from supervised_rlm.repl import SimpleREPL
from supervised_rlm.subagent import SubAgentDispatcher

models={
    "qwen3-32B":{
        "base_url":"http://10.100.1.114:8001/v1",
        "api_key":"NO_API_KEY",
        "model":"Qwen/Qwen3-32B"
    },
    "qwen3.5-9B":{
        "base_url":"http://10.100.1.114:8000/v1",
        "api_key":"NO_API_KEY",
        "model":"Qwen/Qwen3.5-9B"
    },
    "deepseek-v4.1":{
        "base_url":"https://api.deepseek.com",
        "api_key":os.environ.get("LLM_API_KEY"),
        "model":"https://api.deepseek.com"
    }
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": (
                "Execute Python code in a persistent REPL. "
                "Use print() to return results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"],
                "additionalProperties": False
            }
        }
    }
]

def _terminal_print(content):
    stream=sys.__stdout__ or sys.stdout
    stream.write(content)
    stream.flush()

def _extract_final_answer(content):
    if "<think>" in content:
        if "</think>" not in content:
            return ""
        return content.rsplit("</think>",1)[1].strip()
    return content.strip()

def call_llm(client,model,message,tools,show_stream=True):
    finish_reason=None
    full_content=""
    tool_calls={}
    response=client.chat.completions.create(model=model,messages=message,tools=tools,stream=True)
    for chunk in response:
        if not chunk.choices:
            continue
        delta=chunk.choices[0].delta
        if delta.content:
            full_content+=delta.content
            if show_stream:
                _terminal_print(delta.content)
        if delta.tool_calls:
            for tool in delta.tool_calls or []:
                index = tool.index
                if index not in tool_calls:
                    tool_calls[index] = {
                        "id": "",
                        "type": "function",
                        "function": {
                            "name": "",
                            "arguments": ""
                        }
                    }
                current_tool_call=tool_calls[index]
                if tool.id:
                    current_tool_call["id"]+=tool.id
                if tool.function:
                    if tool.function.name:
                        current_tool_call["function"]["name"]+=tool.function.name
                    if tool.function.arguments:
                        current_tool_call["function"]["arguments"]+=tool.function.arguments
        if chunk.choices[0].finish_reason is not None:
            finish_reason=chunk.choices[0].finish_reason
    if show_stream:
        _terminal_print(f"Finish reason:{finish_reason}\n")
    return full_content,tool_calls,finish_reason

def do_repl(tool_calls,repl,require_approval=True,show_log=True):
    results={}
    for call in tool_calls.values():
        if call["function"]["name"]!="execute_python":
            results[call["id"]] = (
                "Unknown tool: " + call["function"]["name"]
            )
            print("unknown tool:",call["function"]["name"])
            continue
        try:
            arguments=json.loads(call["function"]["arguments"])
            code=arguments["code"]
            if not isinstance(code,str):
                raise TypeError("code must be a string")
            if show_log:
                _terminal_print(f"Code will be execute:\n{code}\n")
            user_decision=""
            result=""
            while user_decision not in ["y","n"]:
                if not require_approval:
                    user_decision = "y"
                else:
                    user_decision=input("Do tool call[y/n]?").strip().lower()
                if user_decision=="n":
                    result="Execution denied by user"
                elif user_decision=="y":
                    result=repl.execute(code)
                    if not result:
                        result="Execution without output"
                else:
                    continue
        except Exception as e:
            result=f"{type(e).__name__}:{e}"
        if show_log:
            _terminal_print(f"REPL's output:\n{result}\n")
        results[call["id"]]=result
    return results

def agent_loop(client,model,messages,tools,repl,history,
               max_step=8,require_approval=True,show_stream=True,role="root"):
    if show_stream==True and role=="root":
        print(f"Model:{model}")
        print(f"Max_steps:{max_step}")
        print(f"require_approval:{require_approval}")
    for step in range(1,max_step+1):
        if show_stream==True and role=="root":
            print(f"==================== step{step} ====================")
        content,tool_calls,finish_reason=call_llm(client,model,messages,tools,show_stream=show_stream)
        calls = [
            tool_calls[index]
            for index in sorted(tool_calls)
        ]
        entry={
            "role":role,
            "step":step,
            "finish_reason":finish_reason,
            "content":content,
            "tool_calls":deepcopy(calls),
            "results":{},
            "repl_state":repl.snapshot_variables(),
            "status":"unknown"
        }
        if finish_reason in ("length","content_filter"):
            entry["status"]="incomplete"
            history.append(entry)
            return {
                "status":"incomplete",
                "reason":finish_reason
            }
        if finish_reason=="tool_calls":
            if not tool_calls:
                entry["status"]="protocol_error"
                history.append(entry)
                return {
                    "status":"protocol_error",
                    "reason":"finish_reason is tool_call but calls is empty"
                }
            messages.append({
                "role":"assistant",
                "content":content or None,
                "tool_calls":calls
            })
            results=do_repl(tool_calls,repl,require_approval=require_approval,show_log=show_stream)
            for call in calls:
                tool_id=call["id"]
                result=results.get(tool_id,"Tool execution failed or return no result")
                messages.append({
                    "role":"tool",
                    "tool_call_id":tool_id,
                    "content":str(result)
                })
            entry["status"]="tool_executed"
            entry["results"]=dict(results)
            entry["repl_state"]=repl.snapshot_variables()
            history.append(entry)
            continue
        if finish_reason=="stop" and not tool_calls:
            answer=_extract_final_answer(content)
            if answer:
                messages.append({
                    "role":"assistant",
                    "content":content,
                })
                entry["status"]="completed"
                history.append(entry)
                return {
                    "status":"completed",
                    "answer":answer
                }
            else:
                entry["status"]="incomplete"
                history.append(entry)
                return {
                    "status":"incomplete",
                    "reason":"no final answer can be extracted"
                }
        else:
            entry["status"]="protocol_error"
            history.append(entry)
            return {
                "status":"protocol_error",
                "reason":(
                    f"unexpected finish reason:{finish_reason},"
                    f"tool_calls={len(calls)}"
                )
            }
    return {
        "status":"max_step",
        "reason":f"Task reached max model turn:{max_step}"
    }

def main():
    model_config=models["qwen3-32B"]
    base_url=model_config["base_url"]
    api_key=model_config["api_key"]
    model=model_config["model"]
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60, max_retries=3)
    root_variables={
        "context":[
            "i am a student in CS",
            "i am doing research on RLM Paper's optimized",
            "My work folder is /home/michael/workspace/research/rlm-supervised-recursion/supervised_rlm",
            "list all files in that Path into standard output, and store them into global variables:file_names"
        ]
    }
    root_repl=SimpleREPL(root_variables,root_variables)
    dispatcher=SubAgentDispatcher(client,model,tools,agent_loop,max_subcalls=4,max_content_chars=200_000,required_approval=False)
    root_repl.set_global_variable("rlm_query",dispatcher.rlm_query)
    root_repl.set_global_variable("rlm_query_batched",dispatcher.rlm_query_batch)
    system_prompt = (
        "You are a Root Agent with a persistent Python REPL. "
        "The REPL contains a variable named context, "
        "holding external task data. "
        "Use execute_python to inspect and process it. "
        "Two Python functions are available inside the REPL: "
        "rlm_query(instruction, context) creates an independent "
        "Child Agent and returns its final answer. "
        "rlm_query_batched(queries) sequentially executes "
        "multiple child tasks, where queries is a list of "
        "(instruction, context) pairs. "
        "Each Child Agent has its own messages and REPL. "
        "Delegate independent subtasks when appropriate. "
        "Pass only relevant context to children. "
        "Use Python to aggregate child results. "
        "When the overall task is complete, provide a final "
        "answer directly without calling tools. "
        "Do not claim success without sufficient evidence."
    )
    messages=[
        {
            "role":"system",
            "content":system_prompt
        }
    ]
    history=[]
    console=Console()
    while True:
        user_input=input(">>").strip()
        if user_input.lower() in ["exit","quit"]:
            break
        else:
            dispatcher.reset_task()
            messages.append({
                "role":"user",
                "content":user_input
            })
            outcome=agent_loop(client,model,messages,tools,root_repl,history,max_step=20,require_approval=False,show_stream=True,role="root")
            _terminal_print(f"Task status:{outcome['status']}\n")
            _terminal_print(f"Total child calls:{len(dispatcher.records)}\n")
            if outcome["status"]=="completed":
                console.print(Markdown(outcome["answer"]))
            else:
                _terminal_print(f"Reason:{outcome['reason']}\n")
    _terminal_print("Agent session finished\n")


if __name__ == "__main__":
    main()