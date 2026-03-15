import ast
s=open('app.py','r',encoding='utf-8').read()
try:
    tree=ast.parse(s)
except Exception as e:
    print('PARSE ERROR',e)
    raise

for node in ast.walk(tree):
    if isinstance(node, ast.Try):
        print(f'Try at {node.lineno}, handlers: {[ (h.lineno) for h in node.handlers ]}, orelse_len:{len(node.orelse)}, final_len:{len(node.finalbody)}')
