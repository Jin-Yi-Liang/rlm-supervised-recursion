import os
from openai import OpenAI
import json
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

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
            "List the files in /home/michael/workspace/research/RLM/rlm-supervised-recursion/supervised_rlm "
            "using the Python REPL and store all files name in 'file_name' variable"
        )
    }
]
global_variables={}
local_variables={}
repl=SimpleREPL(global_variables,local_variables)

def call_llm(client,model,message,tools):
    full_content=""
    tool_calls={}
    response=client.chat.completions.create(model=model,messages=message,tools=tools,stream=True)
    for chunk in response:
        if not chunk.choices:
            continue
        delta=chunk.choices[0].delta
        if delta.content is not None:
            content=chunk.choices[0].delta.content
            full_content+=content
            print(content,end="",flush=True)
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
    results=do_repl(tool_calls)
    return full_content,tool_calls,results

def do_repl(tool_calls):
    results={}
    for call in tool_calls.values():
        if call["function"]["name"]!="execute_python":
            print("unknown tool:",call["function"]["name"])
            continue
        try:
            arguments=json.loads(call["function"]["arguments"])
            code=arguments["code"]
            print(f"Code will be execute:\n{code}")
            user_decision=""
            while user_decision not in ["y","n"]:
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

def stream_markdown(client,model,message):
    console = Console()
    response=client.chat.completions.create(model=model,messages=message,stream=True)
    full_content=""
    with Live(
        Markdown(""),
        console=console,
        refresh_per_second=1,
    )as live:
        for chunk in response:
            if not chunk.choices:
                continue
            content=chunk.choices[0].delta.content
            if content:
                full_content+=content
                live.update(Markdown(full_content))
    return full_content

content,tool_calls,results=call_llm(client,model,messages,tools)
print(content,tool_calls,results)
repl.show_variables()