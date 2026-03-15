with open('app.py','r',encoding='utf-8') as f:
    for i,l in enumerate(f, start=1):
        print(f"{i:03d}: {l.rstrip()}")
