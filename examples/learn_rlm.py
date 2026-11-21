import io
from contextlib import redirect_stdout

class SimpleREPL:
    def __init__(self,global_vari,local_vari):
        self.global_variables=global_vari
        self.local_variables=local_vari

    def set_variable(self,name,value):
        self.global_variables[name]=value

    def get_variable(self,name):
        return self.global_variables[name]

    def execute(self,code):
        output=io.StringIO()
        with redirect_stdout(output):
            exec(code,self.global_variables,self.local_variables)
        return output.getvalue()

if __name__=="__main__":
    global_variables={}
    local_variables={}
    repl=SimpleREPL(global_variables,local_variables)
    while True:
        code=input(">>")
        if code=="exit":
            break
        try:
            result=repl.execute(code)
            print(result)
        except Exception as e:
            print(f"error: {e}")
    print("global_variables:")
    print({
        k:v
        for k,v in global_variables.items()
        if k!="__builtins__"
    })
    print(f"local_variables:{local_variables}")