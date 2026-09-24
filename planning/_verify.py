import ast

files = [
    "planning_core.py",
    "planning_kev_mcp.py",
    "planning_engine.py",
    "planning_backends.py",
    "planning_standalone.py",
]

for name in files:
    with open(name, encoding="utf-8") as f:
        text = f.read()
    ast.parse(text, filename=name)
    print(name, "OK")

print("ALL PARSE OK")
