import io
from contextlib import redirect_stdout

class SimpleREPL:
    def __init__(self,global_vari=None,local_vari=None):
        if global_vari is None:
            global_vari={}
        if local_vari is None:
            local_vari=global_vari
        self.global_variables=global_vari
        self.local_variables=local_vari

    def set_global_variable(self,name,value):
        self.global_variables[name]=value

    def set_local_variable(self,name,value):
        self.local_variables[name]=value

    def get_global_variable(self,name):
        return self.global_variables[name]

    def get_global_variables(self):
        return self.global_variables

    def get_local_variable(self,name):
        return self.local_variables[name]

    def get_local_variables(self):
        return self.local_variables

    def execute(self,code):
        result=""
        output=io.StringIO()
        with redirect_stdout(output):
            try:
                exec(code,self.global_variables,self.local_variables)
                result=output.getvalue()
            except Exception as e:
                result+=f"error:{e}"
        return result

    def snapshot_variables(self):
        def summarize(namespace):
            items={}
            for name,value in namespace.items():
                if name=="__builtins__":
                    continue
                item={
                    "type":type(value).__name__
                }
                if isinstance(value,(str,int,float,bool,type(None))):
                    item["preview"]=repr(value)[:200]
                elif isinstance(value,(list,tuple,dict,set)):
                    item["length"]=len(value)
                items[name]=item
            return items
        return {
            "global":summarize(self.global_variables),
            "local":summarize(self.local_variables),
        }

    def show_variables(self):
        print(self.snapshot_variables())

if __name__=="__main__":
    global_variables={}
    local_variables={}
    repl=SimpleREPL(global_variables,local_variables)
    while True:
        code=input(">>")
        if code=="exit":
            break
        result=repl.execute(code)
        print(result)
    repl.show_variables()