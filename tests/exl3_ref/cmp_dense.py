import json,struct,os,hashlib,sys,random
E=os.path.expanduser('~/models/inkling-2-small-exl3-2.59bpw'); O=os.path.expanduser('~/models/inkling-2-small')
def hdr(p):
    with open(p,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n)); h.pop('__metadata__',None); return h,8+n
def tbytes(path,name,cache={}):
    if path not in cache: cache[path]=hdr(path)
    h,off=cache[path]; e=h[name]; a,b=e['data_offsets']
    fd=os.open(path,os.O_RDONLY)
    try:
        data=os.pread(fd,b-a,off+a)
        os.posix_fadvise(fd,off+a,b-a,os.POSIX_FADV_DONTNEED)
    finally: os.close(fd)
    return e['dtype'],e['shape'],data
r=json.load(open(E+'/checkpoint.json')); wm=r['weight_map']
oi=json.load(open(O+'/model.safetensors.index.json'))['weight_map']
names=[k for k in wm if not k.startswith('model.visual')]
missing=[k for k in names if k not in oi]
print('dense names', len(names), 'missing in original index', len(missing), missing[:5])
random.seed(7)
big=['model.llm.layers.0.attn.wq_du.weight','model.llm.layers.47.attn.wo_ud.weight','model.llm.layers.5.attn.k_sconv.weight_proj','model.llm.attn_res.output_query','model.llm.layers.40.mlp.gate.bias']
sample=big+random.sample(names,40)
same=diff=0
for k in sample:
    d1,s1,b1=tbytes(E+'/'+wm[k],k); d2,s2,b2=tbytes(O+'/'+oi[k],k)
    ok=(d1==d2 and s1==s2 and hashlib.sha256(b1).digest()==hashlib.sha256(b2).digest())
    same+=ok; diff+=not ok
    if not ok: print('DIFF',k,d1,s1,d2,s2)
print('sampled',len(sample),'identical',same,'different',diff)
# all dtypes/shapes (headers only) for every dense name
import collections
bad=0
hcache={}
for k in names:
    pe=E+'/'+wm[k]; po=O+'/'+oi[k]
    for p in (pe,po):
        if p not in hcache: hcache[p]=hdr(p)[0]
    a=hcache[pe][k]; b=hcache[po][k]
    if a['dtype']!=b['dtype'] or a['shape']!=b['shape']: bad+=1; print('HDR DIFF',k,a['dtype'],a['shape'],b['dtype'],b['shape'])
print('header dtype/shape mismatches over all', len(names), ':', bad)
