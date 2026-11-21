import sys, random
def brute(s):
    q=[[],[]]; op=[True,True]; cnt=0; out=[]
    for c in s:
        if c in 'ab': cnt+=1; q['ab'.index(c)].append(cnt)
        elif c in 'AB': out.append(str(q['AB'.index(c)].pop(0)%10))
        elif c in '>]':
            k='>]'.index(c); q[1-k]+=q[k][::-1]; q[k]=[]; op[k]=False
        else:
            k='<['.index(c); o=q[1-k]; h=len(o)//2
            q[k]=o[len(o)-h:][::-1]; q[1-k]=o[:len(o)-h]; op[k]=True
    return ''.join(out)
def gen(n,rng,w):
    q=[0,0]; op=[True,True]; s=[]; served=False
    while len(s)<n:
        ch=[]
        for k in (0,1):
            if op[k]:
                ch+=['ab'[k]]*w[0]
                if q[k]>0: ch+=['AB'[k]]*w[1]
                if op[1-k]: ch+=['>]'[k]]*w[2]
            else: ch+=['<['[k]]*w[3]
        c=rng.choice(ch); s.append(c)
        if c in 'ab': q['ab'.index(c)]+=1
        elif c in 'AB': q['AB'.index(c)]-=1; served=True
        elif c in '>]': k='>]'.index(c); q[1-k]+=q[k]; q[k]=0; op[k]=False
        else: k='<['.index(c); h=q[1-k]//2; q[k]=h; q[1-k]-=h; op[k]=True
    return ''.join(s), served
if __name__=='__main__':
    import subprocess
    rng=random.Random(int(sys.argv[1])); bad=0; tot=0
    for it in range(int(sys.argv[2])):
        n=rng.choice([2,3,5,8,15,40,200,1000]); w=[rng.randint(1,6) for _ in range(4)]
        s,ok=gen(n,rng,w)
        if not ok: continue
        tot+=1
        r=subprocess.run(['./solution'],input=f'{n}\n{s}\n',capture_output=True,text=True).stdout.strip()
        if r!=brute(s): bad+=1; print('MISMATCH',n,s[:80],r[:40],brute(s)[:40]); break
    print('tests',tot,'mismatches',bad)
