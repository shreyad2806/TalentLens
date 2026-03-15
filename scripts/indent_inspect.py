with open('app.py','r',encoding='utf-8') as f:
    for i,l in enumerate(f, start=1):
        if 30<=i<=180:
            lead=''
            for ch in l:
                if ch==' ':
                    lead+='.'
                elif ch=='\t':
                    lead+='\t'
                else:
                    break
            print(f"{i:03d}: lead='{lead}' len={len(lead)} line={l[len(lead):].rstrip()}")
