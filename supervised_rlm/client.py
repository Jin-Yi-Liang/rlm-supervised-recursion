import os
from openai import OpenAI
import json
from rich.console import Console
from rich.markdown import Markdown

from supervised_rlm.repl import SimpleREPL

base_url=os.environ.get("LLM_URL") or "http://10.100.1.114:8001/v1"
api_key=os.environ.get("LLM_API_KEY") or "NO_API_KEY"
model=os.environ.get("MODEL") or "Qwen/Qwen3-32B"

client=OpenAI(base_url=base_url,api_key=api_key,timeout=60,max_retries=3)
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
messages = [
    {
        "role": "system",
        "content": (
            "You are a Python coding assistant. "
            "You can call execute_python to run Python code. "
            "The REPL preserves variables across calls. "
            "Use print() to inspect results. "
            "For this task, perform read-only filesystem operations. "
            "Do not modify or delete files. "
            "When the task is complete, provide a final answer."
        )
    },
    {
        "role": "user",
        "content": (
            "List the files in /home/michael/workspace/research/rlm-supervised-recursion/supervised_rlm"
            "using the Python REPL and store all files name in 'file_name' variable and print all file names "
            "in stand output"
        )
    }
]
history=[]
global_variables={}
local_variables={}
repl=SimpleREPL(global_variables,local_variables)

max_step=10
require_approval=True

def call_llm(client,model,message,tools):
    finish_reason=None
    full_content=""
    tool_calls={}
    response=client.chat.completions.create(model=model,messages=message,tools=tools,stream=True)
    for chunk in response:
        if not chunk.choices:
            continue
        delta=chunk.choices[0].delta
        if delta.content is not None:
            full_content+=delta.content
            print(delta.content,end="",flush=True)
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
            print(f"Finish reason:{finish_reason}")
    return full_content,tool_calls,finish_reason

def do_repl(tool_calls,repl,require_approval=True):
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
            print(f"Code will be execute:\n{code}")
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
        print(f"REPL's output:\n{result}")
        results[call["id"]]=result
    return results

def agent_loop(client,model,messages,tools,repl,history,max_step=8,require_approval=True):
    print(f"Model:{model}")
    print(f"Max_steps:{max_step}")
    print(f"require_approval:{require_approval}")
    for step in range(max_step):
        print(f"====================step{step+1}====================")
        content,tool_calls,finish_reason=call_llm(client,model,messages,tools)
        if finish_reason=="length":
            print("Model output chunked. Task not complete")
        elif finish_reason=="tool_calls" and tool_calls:
            results=do_repl(tool_calls,repl,require_approval)
            calls=[
                tool_calls[index]
                for index in sorted(tool_calls)
            ]
            messages.append({
                "role":"assistant",
                "content":content,
                "tool_calls":calls
            })
            for call in calls:
                tool_id=call["id"]
                result=results.get(tool_id,"Tool execution failed or return no result")
                messages.append({
                    "role":"tool",
                    "tool_call_id":tool_id,
                    "content":str(result)
                })
        else:
            messages.append({
                "role":"assistant",
                "content":content,
            })
            break
        history.append({
            "step":step+1,
            "finish_reason":finish_reason,
            "content":content,
            "tool_calls":tool_calls,
            "result":results,
            "global_variables":repl.get_pure_global_variables(),
            "local_variables":repl.get_local_variables()
        })
    print("Agent loop finished")

def render_markdown(content:str):
    console=Console()
    markdown=Markdown(content)
    console.print(markdown)

if __name__ == "__main__":
    while True:
        user_input=input(">>")
        if user_input.strip().lower() in ["exit","quit"]:
            break
        if not user_input:
            continue
        messages.append({
            "role":"user",
            "content":user_input
        })
        agent_loop(client,model,messages,tools,repl,history,max_step=20,require_approval=False)
    print("Bye!")

    print("=======================================================")
    render_markdown(str(history))
    render_markdown(str(messages))
    repl.show_variables()