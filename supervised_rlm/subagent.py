import json

from supervised_rlm.repl import SimpleREPL

class SubAgentDispatcher:
    def __init__(self,client,model,tools,agent_loop,*,
                 max_subcalls=4,child_max_step=8,max_content_chars=200_000,required_approval=True):
        self.client=client
        self.model=model
        self.tools=tools
        self.agent_loop=agent_loop
        self.max_subcalls=max_subcalls
        self.child_max_step=child_max_step
        self.max_content_chars=max_content_chars
        self.required_approval=required_approval

        self.calls_this_task=0
        self.records=[]

    def reset_task(self):
        self.calls_this_task=0

    def rlm_query(self,instruction,context):
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a nonempty string")
        if self.calls_this_task>=self.max_subcalls:
            raise RuntimeError("Sub-Agent call budget exceeded")
        try:
            encoded=json.dumps(context,ensure_ascii=False,allow_nan=False)
        except (TypeError,ValueError) as e:
            raise ValueError("context must be a valid json object") from e
        if len(encoded)>self.max_content_chars:
            raise ValueError(f"Child context must not exceed max_content_chars:{self.max_content_chars}")
        self.calls_this_task+=1
        task_id=f"sub-{len(self.records)+1:04d}"
        child_variables={
            "context":json.loads(encoded)
        }
        child_repl=SimpleREPL(child_variables,child_variables)
        child_messages = [
            {
                "role": "system",
                "content": (
                    "You are a child agent. "
                    "Your persistent Python REPL has a "
                    "context variable holding task data. "
                    "Call execute_python to inspect or "
                    "compute with this data. "
                    "You cannot delegate further. "
                    "When finished, provide a final answer "
                    "without tool calls."
                )
            },
            {
                "role": "user",
                "content": instruction
            }
        ]
        child_history=[]
        record={
            "task_id":task_id,
            "instruction":instruction,
            "depth":1,
            "context_chars":len(encoded),
            "status":"running",
            "history":child_history,
            "messages":child_messages,
        }
        self.records.append(record)
        try:
            outcome=self.agent_loop(self.client,self.model,child_messages,self.tools,child_repl,child_history,
                                    max_step=self.child_max_step,require_approval=self.required_approval,show_stream=False,role="child")
        except Exception as e:
            record["status"]="error"
            record["reason"]=(f"{type(e).__name__}:{e}")
            raise RuntimeError(f"Child{task_id} failed:{record['reason']}") from e
        record["status"]=outcome["status"]
        record["reason"]=outcome.get("reason")
        record["answer"]=outcome.get("answer")
        if outcome["status"]!="completed":
            raise RuntimeError(f"Child{task_id} incomplete:{record['status']},{record['reason']}")
        return outcome["answer"]

    def rlm_query_batch(self,queries):
        if not isinstance(queries,list) or not queries:
            raise ValueError("queries must be a nonempty list")
        results=[]
        for query in queries:
            if not isinstance(query,(list,tuple)) or len(query)!=2:
                raise ValueError("queries must be (instruction,context)")
            instruction,context=query
            result=self.rlm_query(instruction,context)
            results.append(result)
        return results