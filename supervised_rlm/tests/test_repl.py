from supervised_rlm.repl import SimpleREPL

root_variables={}
root=SimpleREPL(root_variables,root_variables)

child_variables={}
child=SimpleREPL(child_variables,child_variables)

root.execute("a=10")
child.execute("a=220")

print(root.get_global_variable("a"))
print(child.get_global_variable("a"))

root.execute("""
def get_a():
    return a
""")

print(root.execute("print(get_a())"))

root.show_variables()
child.show_variables()