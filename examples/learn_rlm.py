class SimpleREPL:
    def __init__(self):
        self.variables={}

    def set_variable(self,name,value):
        self.variables[name]=value

    def get_variable(self,name):
        return self.variables[name]

    def execute(self,code):
        exec(code,self.variables,self.variables)

if __name__=="__main__":
    repl=SimpleREPL()
    repl.set_variable("context","This is my full context variable that contains all the external context of your context")
    context=repl.get_variable("context")
    print(f"context:{context}")

    repl.execute("""
a_num=10
b_num=20
total=a_num+b_num
    """)
    total1=repl.get_variable("total")
    repl.execute("total+=a_num")
    total2=repl.get_variable("total")
    print(f"total1:{total1}\ntotal2:{total2}")

    for key,value in repl.variables.items():
        if key !=  "__builtins__":
            print(f"{key}:{value}")