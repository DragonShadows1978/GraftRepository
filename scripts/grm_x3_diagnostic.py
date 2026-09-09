#!/usr/bin/env python3
"""Pure X3 fork/score contracts; no model import or serving changes.
Prior art: Jain & Wallace (2019), https://aclanthology.org/N19-1357/ motivates
attention/intervention separation; Meng et al. (2022), arXiv:2202.05262 supplies
controlled activation-intervention precedent. House DET1 (2026) supplies the
canonical host snapshot representation. Own adapter edits one active mount.
"""
from __future__ import annotations
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
OUT=ROOT/'artifacts/grm_x3'
ARMS=('zero','remove','swap','same_payload','sham')

class X3Error(RuntimeError): pass

def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def digest(x): return hashlib.sha256(canonical(x)).hexdigest()

def create(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(canonical(x))

def freeze_array(a):
    a=np.ascontiguousarray(a).copy()
    if a.dtype.hasobject: raise X3Error('object array forbidden')
    a.flags.writeable=False
    return a

def array_digest(a):
    return digest({'dtype':a.dtype.str,'shape':a.shape,'bytes':hashlib.sha256(a.tobytes()).hexdigest()})

@dataclass(frozen=True)
class Snapshot:
    state: dict
    arrays: dict[str,np.ndarray]
    spans: dict[str,tuple[int,int]]
    active_prefix: str
    layers: int

    def fingerprint(self):
        return digest({'state':self.state,'spans':self.spans,'prefix':self.active_prefix,
            'layers':self.layers,'arrays':{k:array_digest(v) for k,v in self.arrays.items()}})

def load_capture(path, spans):
    from scripts.grm_det1_3_snapshot import load_snapshot,load_snapshot_array,require_snapshot_member_coverage
    m=load_snapshot(Path(path));require_snapshot_member_coverage(m,'X3 live source')
    if not m['complete']: raise X3Error('incomplete live capture')
    return Snapshot(copy.deepcopy(m['state']),{k:freeze_array(load_snapshot_array(m,k)) for k in m['arrays']},
        spans,'cache' if m['state']['arena.caches_present'] else 'injection',m['identity']['layer_count'])

def fork(base,arm,replacement=None):
    """One-mount intervention: exclusion via mask, replacement via active K/V.
    Prior art: house DET1 fork masks and RS3 constant-position law (2026),
    Meng et al. (2022) controlled interventions. Mask-only removal is X3's
    implementation choice; leaves all non-target columns and tensor bytes fixed.
    """
    if arm not in ARMS: raise X3Error(f'unknown arm {arm}')
    if set(base.spans)!={'target','sham'} or base.layers<1: raise X3Error('missing spans/layers')
    t=base.spans['target'];s=base.spans['sham']
    if max(t[0],s[0])<min(t[1],s[1]): raise X3Error('overlapping mount spans')
    arrays=dict(base.arrays)
    span=base.spans['sham' if arm=='sham' else 'target'];lo,hi=span
    if not 0<=lo<hi: raise X3Error('empty or negative span')
    for li in range(base.layers):
        mask_key=f'mask.layer_{li:02d}.allowed'
        mask=base.arrays[mask_key]
        if mask.ndim!=4 or mask.dtype!=np.uint8 or not np.isin(mask,[0,1]).all(): raise X3Error('invalid canonical mask')
        if hi>mask.shape[3]: raise X3Error('mask span out of bounds')
        if arm in ('remove','sham'):
            mask=mask.copy();mask[...,lo:hi]=0;arrays[mask_key]=freeze_array(mask)
        for name in ('k','v'):
            key=f'{base.active_prefix}.layer_{li:02d}.{name}'
            src=base.arrays[key]
            if src.ndim!=4 or hi>src.shape[2]:raise X3Error('payload span out of bounds')
            if arm in ('swap','same_payload'):
                donor=src[:,:,lo:hi,:].copy() if arm=='same_payload' else np.asarray((replacement or {}).get(key))
                if donor.shape!=src[:,:,lo:hi,:].shape or donor.dtype!=src.dtype:raise X3Error(f'donor shape/dtype mismatch: {key}')
                dst=src.copy();dst[:,:,lo:hi,:]=donor;arrays[key]=freeze_array(dst)
    result=Snapshot(copy.deepcopy(base.state),arrays,copy.deepcopy(base.spans),base.active_prefix,base.layers)
    verify_delta(base,result,arm,replacement)
    return result

def verify_delta(base,other,arm,replacement=None):
    """Independent byte oracle; unexpected mutations fail rather than be ignored.
    Prior art: house DET1 snapshot delta allowlisting (2026); own X3 field policy.
    """
    if canonical(base.state)!=canonical(other.state) or base.spans!=other.spans or base.layers!=other.layers or base.active_prefix!=other.active_prefix:
        raise X3Error('nonintervention scalar state changed')
    if base.arrays.keys()!=other.arrays.keys():raise X3Error('array field set changed')
    lo,hi=base.spans['sham' if arm=='sham' else 'target']
    changed=[]
    for key,src in base.arrays.items():
        dst=other.arrays[key]
        if src.shape!=dst.shape or src.dtype!=dst.dtype:raise X3Error('shape/dtype changed')
        expected=src.copy()
        if arm in ('remove','sham') and key.startswith('mask.layer_') and key.endswith('.allowed'):
            expected[...,lo:hi]=0
        if arm=='swap' and key.startswith(base.active_prefix+'.layer_') and key.endswith(('.k','.v')):
            if replacement is None or key not in replacement:raise X3Error('missing donor')
            expected[:,:,lo:hi,:]=replacement[key]
        if expected.tobytes()!=dst.tobytes():raise X3Error(f'unauthorized byte delta: {key}')
        if src.tobytes()!=dst.tobytes():changed.append(key)
    return {'evidence_class':'unit test' if base.state.get('synthetic') else 'E2E session receipt',
        'canonical_host_bytes_pinned':True,'literal_device_storage_bytes':False,
        'changed_fields':changed,'base_fingerprint':base.fingerprint(),'fork_fingerprint':other.fingerprint()}

def log_probs(logits):
    x=np.asarray(logits,dtype=np.float64)
    if x.ndim!=1 or x.size<2 or not np.isfinite(x).all():raise X3Error('invalid full-vocabulary logits')
    z=x-x.max();return z-np.log(np.exp(z).sum())

def distribution_delta(base,other):
    # Prior art: Kullback & Leibler (1951), On Information and Sufficiency.
    # unverified — lead to check: doi 10.1214/aoms/1177729694. Standard KL;
    # float64 log-softmax avoids epsilon clipping and underflowed log(0).
    p,q=log_probs(base),log_probs(other)
    if p.shape!=q.shape:raise X3Error('vocabulary mismatch')
    kl=float(np.sum(np.exp(p)*(p-q)))
    if kl < -1e-10:raise X3Error('negative KL beyond rounding')
    return {'kl_nats':max(0.0,kl),'top1_changed':bool(np.argmax(p)!=np.argmax(q)),
        'base_top1':int(np.argmax(p)),'fork_top1':int(np.argmax(q)),
        'raw_logits_byte_equal':np.asarray(base).dtype==np.asarray(other).dtype and np.asarray(base).tobytes()==np.asarray(other).tobytes()}

def outcome(text,fixture,truncated=False):
    # Prior art: house battery answer checks (2026); whole-answer equality is
    # deliberately stricter than DET1's whole-value containment. X3 registration.
    a=text.strip().casefold()
    exact=any(a==v.strip().casefold() for v in fixture['expected_values'])
    decoy=a==fixture['decoy_value'].strip().casefold()
    refusal=any(a.startswith(v) for v in ("not in memory","i don't know","i do not know","i cannot","i can't","unknown"))
    realized=exact if fixture['intended_group']=='correct' else decoy if fixture['intended_group']=='decoy' else refusal
    return {'answer':text,'exact_error':not exact,'is_refusal':refusal,'is_exact_decoy':decoy,'category_realized':realized,'truncated':bool(truncated)}

def validate_fixtures():
    manifest=json.loads((OUT/'fixtures/manifest.json').read_text())
    for r in [manifest['registration'],*manifest['fixtures'],*manifest['source_pins']]:
        if sha(ROOT/r['path'])!=r['sha256']:raise X3Error(f'frozen input drift: {r["path"]}')
    reg=json.loads((OUT/'registration.json').read_text())
    if sha(reg['tokenizer']['path'])!=reg['tokenizer']['sha256']:raise X3Error('tokenizer drift')
    rows=[json.loads((ROOT/r['path']).read_text()) for r in manifest['fixtures']]
    if len(rows)!=20 or len({r['id'] for r in rows})!=20:raise X3Error('fixture count/duplicates')
    counts={g:sum(r['intended_group']==g for r in rows) for g in ('correct','decoy','refusal')}
    if counts!={'correct':8,'decoy':6,'refusal':6}:raise X3Error('strata count')
    for r in rows:
        if sha(ROOT/r['battery']['source']['path'])!=r['battery']['source']['sha256']:raise X3Error('battery drift')
        if len(r['base_token_ids'])!=len(r['swap_token_ids']) or not 0<len(r['base_token_ids'])+len(r['sham_token_ids'])<=96:raise X3Error('geometry')
    return reg,rows

def cells(rows):
    return [{'cell':f'cell_{i:02d}','snapshot_ids':[r['id'] for r in rows[i*5:(i+1)*5]],
        'arms':['unforked',*ARMS],'worker_cap_seconds':285,'outer_cap_seconds':590,
        'estimated_worker_seconds':[180,285],'cooldown_seconds':30,
        'evidence_class':'reasoning; enumeration, no GPU measurements'} for i in range(4)]

def summarize(rows,reg):
    """Fixed-direction paired classification; no fitting, no truth in features.
    Prior art: standard accuracy and contingency tables, no specific prior art
    known to me for X3's scoring direction/thresholds. House D-NGH mass direction.
    """
    if len({r['id'] for r in rows})!=len(rows):raise X3Error('duplicate result rows')
    threshold=reg['scoring']['mass_threshold']
    table={f'{m}/{e}':{'mass':m,'effect':e,'n':0,'exact_errors':0,'error_rate':None} for m in ('low','high') for e in ('low','high')}
    dec=[];correct=[];scored=[];comparable=0;control_good=0;equal=0;sham_good=0
    for r in rows:
        mass=float(r['mass']);effect=float(r['forks']['remove']['kl_nats'])
        if not np.isfinite(mass) or not 0<=mass<=1 or not np.isfinite(effect) or effect<0:raise X3Error('invalid observed mass/effect')
        mh=mass>=threshold;eh=effect>1;err=r['outcome']['exact_error']
        cell=table[f'{"high" if mh else "low"}/{"high" if eh else "low"}'];cell['n']+=1;cell['exact_errors']+=int(err)
        score={'mass_correct':int((not mh)==err),'lesion_correct':int(eh==err)};scored.append(score)
        if r['intended_group']=='decoy':dec.append((mh,score))
        if r['intended_group']=='correct':correct.append(eh)
        f=r['forks']
        if set(f)!=set(ARMS) or any(not np.isfinite(float(v['kl_nats'])) or float(v['kl_nats'])<0 for v in f.values()):raise X3Error('missing/nonfinite fork metrics')
        control_good+=int(f['same_payload']['kl_nats']<.05 and f['sham']['kl_nats']<.05)
        sham_good+=int(f['sham']['kl_nats']<.05);equal+=int(f['same_payload']['raw_logits_byte_equal'])
        factual=max(f['remove']['kl_nats'],f['swap']['kl_nats']);controls=max(f[k]['kl_nats'] for k in ('same_payload','sham','zero'))
        comparable+=int(factual>=.05 and controls>=factual)
    for c in table.values():
        if c['n']:c['error_rate']=c['exact_errors']/c['n']
    full=len(rows)==20 and {r['id'] for r in rows}=={f'x3_{i:02d}' for i in range(20)}
    realized=full and all(r['outcome']['category_realized'] and not r['outcome']['truncated'] for r in rows)
    valid=realized and len(correct)==8 and len(dec)==6
    def acc(s,key):return sum(x[key] for x in s)/len(s) if s else None
    accuracy={'all':{k:acc(scored,k+'_correct') for k in ('mass','lesion')},'decoy':{k:acc([x[1] for x in dec],k+'_correct') for k in ('mass','lesion')}}
    predictions={'P1':control_good>=18 if full else None,'P2':sum(correct)>=6 if valid else None,
        'P3':sum(x[0] for x in dec)>=4 and accuracy['decoy']['lesion']>accuracy['decoy']['mass'] if valid else None,
        'Q1':equal==20 and sham_good>=18 if full else None,'Q2':sum(correct)<=5 if valid else None,
        'Q3':accuracy['all']['lesion']<=accuracy['all']['mass'] if valid else None}
    killed=full and (comparable>=3 or accuracy['all']['lesion']<=accuracy['all']['mass'])
    return {'schema':'grm.x3.summary.v1','evidence_class':'E2E session receipt' if rows else 'reasoning; GPU blocked',
        'status':'RED_KILL' if killed else 'RED_STRATA_OR_INCOMPLETE' if not valid else 'COMPLETE',
        'observed_snapshots':len(rows),'realized_strata':realized,'table':list(table.values()),'predictions':predictions,
        'accuracy':accuracy,'controls_comparable_count':comparable,'P1_joint_count':control_good,'P2_high_count':sum(correct),
        'decoy_mass_high_count':sum(x[0] for x in dec),'same_payload_equal_count':equal,
        'no_serving_adoption':True,'limitations':['canonical host value bytes only','fixed first token may precede fact','tiny or constant-class decoy subset; accuracy not discrimination/AUROC','controlled diagnostic attempts, not natural routing acceptance']}

if __name__=='__main__':
    reg,fixtures=validate_fixtures()
    if sys.argv[1:]==['--dry-run']:print(json.dumps({'cells':cells(fixtures),'total_snapshot_recipes':20,'raw_GPU_snapshots':0,'maximum_gpu_seconds':1140,'registration_sha256':sha(OUT/'registration.json')},indent=2))
    else:raise SystemExit('usage: grm_x3_diagnostic.py --dry-run')
