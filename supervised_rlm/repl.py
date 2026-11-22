import io
from contextlib import redirect_stdout

class SimpleREPL:
    def __init__(self,global_vari,local_vari):
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

    def show_variables(self):
        print("global_variables:")
        print({
            k: v
            for k, v in self.global_variables.items()
            if k != "__builtins__"
        })
        print("local_variables:")
        print({
            k:v
            for k,v in self.local_variables.items()
        })

    def get_pure_global_variables(self):
        pure_global_variables=[]
        pure_global_variables.append({
            k: v
            for k, v in self.global_variables.items()
            if k != "__builtins__"
        })
        return pure_global_variables

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