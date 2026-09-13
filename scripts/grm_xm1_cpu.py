"""XM1 numerical model/loader doubles; no CUDA import or allocation.

Prior art: GRM C7/LT1 loader doubles (GRM contributors, 2026), numpy attention
fixtures and RS4 probability partitions. Taken: loader-only dependency seam;
ours: per-adapter payload ranks/rotary subsets and cache-shape stimulus.
This does NOT execute native attention class bodies or the GPT RS4 ladder.
No language quality, weight numerics or GPU mass equality is established.
"""
from contextlib import nullcontext
from types import SimpleNamespace, ModuleType
import ast
from pathlib import Path
import re
import numpy as np
from scripts.grm_xm1_parity import Loaded

class Tensor:
    def __init__(self, value):
        self.a=np.asarray(value,dtype=np.float32)
    @property
    def shape(self): return self.a.shape
    @property
    def dtype(self): return 'float32'
    def numpy(self): return self.a.copy()
    def float(self): return self
    def astype(self, dtype): return self
    def slice(self, axis, start, length):
        s=[slice(None)]*self.a.ndim; s[axis]=slice(start,start+length)
        return Tensor(self.a[tuple(s)])
    def softmax(self, axis):
        a=np.exp(self.a-self.a.max(axis=axis,keepdims=True))
        return Tensor(a/a.sum(axis=axis,keepdims=True))
    @property
    def device(self): return 'cpu:0'
    def reshape(self, shape): return Tensor(self.a.reshape(shape))
    def transpose(self, a, b): return Tensor(self.a.swapaxes(a,b))
    def expand(self, shape): return Tensor(np.broadcast_to(self.a,shape))
    def mean(self, axes, keep=False): return Tensor(self.a.mean(axis=tuple(axes),keepdims=keep))
    def pow(self, value): return Tensor(self.a**value)
    def sigmoid(self): return Tensor(1/(1+np.exp(-self.a)))
    def half(self): return self
    def __mul__(self,other): return Tensor(self.a*(other.a if isinstance(other,Tensor) else other))
    def __add__(self, other): return Tensor(self.a+(other.a if isinstance(other,Tensor) else other))

class Engine:
    no_grad=staticmethod(nullcontext)
    tensor=staticmethod(lambda a, **kw: Tensor(a))
    is_grad_enabled=staticmethod(lambda:False)
    cat=staticmethod(lambda xs,dim=0:Tensor(np.concatenate([x.a for x in xs],axis=dim)))
    @staticmethod
    def matmul(a,b,alpha=1.,trans_b=False):
        return Tensor(np.matmul(a.a,b.a.swapaxes(-1,-2) if trans_b else b.a)*alpha)

class Codec:
    eos_token_id=0
    def __init__(self):
        self.words=['<eos>']; self.ids={'<eos>':0}
    def encode(self,text,**kw):
        out=[]
        for t in re.findall(r'\S+|\s+', text):
            if t not in self.ids:
                self.ids[t]=len(self.words); self.words.append(t)
            out.append(self.ids[t])
        return out
    def decode(self,ids,skip_special_tokens=False,**kw):
        return ''.join(self.words[int(i)] for i in ids if not(skip_special_tokens and int(i)==0))
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False, enable_thinking=None):
        text=''.join(f"[{m['role']}] {m['content']}\n" for m in messages)
        if add_generation_prompt: text+='[assistant] '
        return self.encode(text) if tokenize else text


def sdpa(q,k,v,**kw):
    scores=Engine.matmul(q,k,alpha=kw.get('scale') or q.shape[-1]**-.5,trans_b=True)
    if kw.get('attn_mask') is not None: scores=scores+kw['attn_mask']
    return Engine.matmul(scores.softmax(-1),v)

class Model:
    compute_dtype='float32'
    def __init__(self,kind,codec,owner):
        self.kind,self.codec,self.owner=kind,codec,owner
        self.layers=[]; self.calls=[]
        self.output=codec.encode('CPU-double')[0]
        for i in range(2):
            att=SimpleNamespace(inject_kv=None,graft_seats=0,live_shift=None,_capture=False,
                is_local_attention=(kind=='nope_mixed' and i==0))
            self.layers.append(SimpleNamespace(is_attn=True,mixer=att) if kind=='partial_rope' else SimpleNamespace(self_attn=att))
        self.extend_rope(4096)
    def extend_rope(self,n):
        R=2 if self.kind in ('mla','partial_rope') else 4
        freqs=1/(10000**(np.arange(0,R,2,dtype=np.float32)/R))
        theta=np.arange(n,dtype=np.float32)[:,None]*freqs[None,:]
        theta=np.concatenate([theta,theta],axis=-1)
        self.rope_cos=Tensor(np.cos(theta));self.rope_sin=Tensor(np.sin(theta))
    def rotate(self,x,pos,r):
        if not r:return x
        n=x.shape[2];cos=self.rope_cos.a[pos:pos+n];sin=self.rope_sin.a[pos:pos+n]
        head=x[...,:r];h=r//2
        y=x.copy();y[...,:r]=head*cos+np.concatenate([-head[...,h:],head[...,:h]],axis=-1)*sin
        return y
    def __call__(self,ids,caches=None,kv_caches=None,position_offset=0,**kw):
        caches=caches if caches is not None else kv_caches
        n=ids.shape[1];out=[]
        raw=np.sin(ids.astype(np.float32)[...,None]*np.array([.1,.2,.3,.4],np.float32))
        x=Tensor(raw)
        for i,layer in enumerate(self.layers):
            att=getattr(layer,'self_attn',getattr(layer,'mixer',None))
            _,cache=att(x,self.rope_cos,self.rope_sin,position_offset,
                        None if caches is None else caches[i])
            out.append(cache)
        self.calls.append(dict(offset=position_offset,n=n,capture=bool(getattr(att,'_capture',False)),injected=att.inject_kv is not None))
        logits=np.zeros((1,1,len(self.codec.words)),np.float32)
        logits[0,0,0 if n==1 else self.output]=1
        return Tensor(logits),out


def source_function(path,name,namespace,cls=None):
    """Compile the unmodified, pinned native function AST; stub imports only.

    Prior art: Python ast/compile (Python contributors); GRM CPU numerical
    seams (2026). No prior art known to me for this exact loader composition.
    It avoids importing a CUDA-initializing module. File/line provenance stays
    native, but constructors, weights and the surrounding model are doubles.
    """
    root=Path(__file__).resolve().parents[1]
    tree=ast.parse((root/path).read_text())
    if cls is not None:
        tree=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(root/path),'exec'),namespace)
    return namespace[name]


def rotary(x,cos,sin):
    r=x.shape[-1];h=r//2
    return Tensor(x.a*cos.a+np.concatenate([-x.a[...,h:],x.a[...,:h]],axis=-1)*sin.a)


def native_layers(model,kind,module):
    root=Path(__file__).resolve().parents[1]
    module.tc=Engine;module.np=np
    module.BlockTC=SimpleNamespace(COMPUTE_DTYPE='float32')
    module._cast=lambda x:x
    module._to_dtype=lambda x,dtype:x
    module._repeat_kv=lambda x,n:Tensor(np.repeat(x.a,n,axis=1))
    module.F=SimpleNamespace(apply_rotary=rotary,scaled_dot_product_attention=sdpa)
    cfg=SimpleNamespace(num_heads=1,num_kv_heads=1,head_dim=4,partial_rotary_dim=2,rms_norm_eps=1e-5,
                        qk_nope_head_dim=2,qk_rope_head_dim=2,v_head_dim=2,q_head_dim=4,kv_lora_rank=2)
    if kind=='partial_rope':
        path='core/qwen35_tc.py';cls='Qwen35AttentionTC'
        source_function(path,'_per_head_rmsnorm',module.__dict__)
        gate=source_function(path,'_apply_output_gate',module.__dict__,cls)
    elif kind=='mla':
        path='core/minicpm3_tc.py';cls='MLAAttentionTC'
    elif kind=='gqa_sink':
        path='core/gpt_oss20b_tc.py';cls='GptOssAttentionTC'
        for name in ('sink_attention_tc','sliding_sink_attention_tc','_gpt_oss_attention_mask'):
            source_function(path,name,module.__dict__)
    else:
        path='scripts/trinity_nope_graft_width_sweep.py';cls=None
        source_function(path,'_apply_rope',module.__dict__)
        module._trinity_scaled_attention=sdpa
        module.GraftF=module.F
        module._band_mask=lambda L,S,w,*args:Tensor(np.where(np.arange(S)[None,:]>(np.arange(S-L,S)[:,None]-w),0,-1e4)[None,None])
    call=source_function(path,'__call__' if cls else '_arena_attention_call',module.__dict__,cls)
    Native=type('PinnedNativeAttention',(),{'__call__':call})
    if kind=='partial_rope':Native._apply_output_gate=gate
    model.layers=[]
    for i in range(2):
        a=Native();a.cfg=cfg;a.inject_kv=None;a.graft_seats=0;a.live_shift=None;a._capture=False
        a.attention_mode='standard';a.absorbed_decode=False
        a.q_proj=lambda x: x;a.k_proj=lambda x:x;a.v_proj=lambda x:x;a.o_proj=lambda x:x
        a.num_heads=a.num_kv_heads=a.num_key_value_heads=a.num_key_value_groups=a.num_heads_per_kv=1
        a.head_dim=4;a.scaling=.5;a.sinks=Tensor([0.1]);a.attn_block=128
        a.sliding_window=64 if kind=='gqa_sink' and i==0 else None
        a.is_local_attention=kind=='nope_mixed' and i==0
        if a.is_local_attention:a.sliding_window=4096
        a.gate_proj=lambda x:x;a.q_norm=lambda x:x;a.k_norm=lambda x:x
        if kind=='partial_rope':
            a.q_proj=lambda x:Tensor(np.concatenate([x.a,x.a],axis=-1))
            a.q_norm_w=a.k_norm_w=Tensor(np.ones(4))
        if kind=='mla':
            a.q_a=lambda x:x;a.q_a_norm=lambda x:x;a.q_b=lambda x:x
            a.kv_a=lambda x:x;a.kv_a_norm=lambda x:x
            a.kv_b=lambda x:Tensor(np.concatenate([x.a,x.a],axis=-1))
        model.layers.append(SimpleNamespace(is_attn=True,mixer=a) if kind=='partial_rope' else SimpleNamespace(self_attn=a))
    return path


def cpu_loader(name,reg):
    kind=reg['adapters'][name]['kind']
    codec=Codec();module=ModuleType('xm1_cpu_native_'+name)
    model=Model(kind,codec,module)
    path=native_layers(model,kind,module)
    if kind=='gqa_sink':owner,fn=module,'sink_attention_tc'
    elif kind=='nope_mixed':owner,fn=module,'_trinity_scaled_attention'
    else:owner,fn=module.F,'scaled_dot_product_attention'
    return Loaded(model,codec,Engine,module,owner,fn,kind,
                  dict(model=name,weights='CPU numerical double',native_attention_source_executed=path,
                       source_loading='unmodified function AST, stub engine and projections; no native constructor/full model'))
