#!/usr/bin/env python3
"""CPU diagnosis and immutable registration; no model imports at module scope.

Prior art: GRM C7/FIX4 receipt joins, SC1.1 glyph projection, A-DEC own-text
binding and RD1 amendment-1 fixed-residency replay (GRM contributors, 2026),
verified local sources. Reuse their contracts; ours is this cohort/prompt.
No prior art known to me for this exact composition; no retrieval algorithm.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'artifacts/grm_c7/r3/amendment_7'
R3 = ROOT/'artifacts/grm_c7/r3'
REG = OUT/'registration.json'
MIDDLE = 'Answer using only the mounted memory records. If those records do not contain the requested information, reply unknown. '
PRIOR = ('GRM SC1.1/DET1.4, A-DEC/RT1, C7/FIX4 and RD1 amendment-1 '
         '(GRM contributors, 2026), locally verified; reuse glyph projection, '
         'own-text binding, immutable checkpoints, fixed mounts, production reader '
         'and scorer. Ours: receipt join and middle prompt. No prior art known '
         'to me for this exact composition.')

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(value,f,indent=2,sort_keys=True,ensure_ascii=False);f.write('\n')

def need(ok, message):
    if not ok: raise ValueError(message)

def census(version):
    # Prior art: FIX4 cell_directory (GRM, 2026). Explicitly select the retained
    # mixed r2 campaign, never recursively pick a stale/failed duplicate.
    out={}
    for c in read(R3/'registration.json')['cells']:
        base=ROOT/'artifacts/grm_c7'/version
        if version=='r2' and c['start']>=24: base=base/'fix4_attempt_1'
        d=base/'cells'/c['id']
        p=d/'probes.jsonl'
        if not p.exists():continue
        for line,text in enumerate(p.read_text().splitlines(),1):
            r=json.loads(text)
            need(r['probe_id'] not in out,'DUPLICATE_PROBE')
            out[r['probe_id']]={'receipt':str(p.relative_to(ROOT)), 'line':line,
                'cell':c,'row':r,'checkpoint':str((d/'checkpoint').relative_to(ROOT))}
    return out

def diagnose():
    fixtures={p['id']:p for p in read(R3/'fixture.json')['probes']}
    versions={v:census(v) for v in ('r2','r3')}
    folds=[]
    for p in sorted((R3/'cells').glob('*/folds/*/result.json')):
        r=read(p);start=read(p.with_name('start.json'))
        folds.append({'receipt':str(p.relative_to(ROOT)),**r,'source_ids':[s['id'] for s in start['sources']]})
    out=[]
    for pid,p in sorted(fixtures.items()):
        if p['class'] not in ('fresh','folded'):continue
        item={'probe_id':pid,'turn':p['turn'],'expected':p['expected'],'class':p['class'],'versions':{}}
        for v in versions:
            src=versions[v][pid];r=src['row'];cp=ROOT/src['checkpoint']
            nodes=read(cp/'repository/manifest.json')['nodes'];state=read(cp/'state.json')
            source=[]
            for t in p['source_turns']:
                rec=state['turn_records'][str(t)];i=rec.get('memory_node_id',rec.get('chat_node_id'))
                n=nodes[i];events=[f for f in folds if f['accepted'] and i in f['source_ids']] if v=='r3' else []
                # These selected raw nodes have no supersession edge. Their
                # only retirement event is a receipted accepted fold.
                need(not n.get('metadata',{}).get('superseded_by'),'UNEXPECTED_SUPERSESSION')
                retired=any(f['turn']<p['turn'] for f in events)
                need(bool(n['retired'])==any(f['turn']<=src['cell']['stop'] for f in events),'RETIREMENT_NOT_EXPLAINED')
                source.append({'source_turn':t,'id':i,'retired_at_probe':retired,
                    'checkpoint_retired':n['retired'],'retirement_events':[{'turn':f['turn'],'receipt':f['receipt']} for f in events],
                    'text':n['text']})
            info=r['memory']['route_info'];ap=info['_route_observation']['admission_profile']
            if v=='r2':
                mechanism='ASCII raw source remains eligible; '+('reader abstains' if r['memory']['score']['abstained'] else 'reader output recorded; see frozen score')
            elif info.get('abstain_reason'):
                mechanism='accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation'
            elif info.get('served_from_node_ids'):
                mechanism='outbound digest does not bind; FIX4 serves the ASCII inbound recency node only' if p['class']=='folded' else 'FIX4 recency binding'
            elif p['class']=='fresh':mechanism='before turn-18 fold; ASCII raw source binds and mounts'
            else:mechanism='outbound Unicode digest does not bind; ASCII inbound source binds; missing outbound at readout'
            item['versions'][v]={'receipt':src['receipt']+':'+str(src['line']),'checkpoint':src['checkpoint'],
                'mounted_ids':r['memory']['residency']['mounted_ids'],'identified':ap['identified_candidates'],
                'ranking':ap['ranking'],'identifier_tokens':ap['identifier_tokens'],'excluded_live_ids':info['_route_observation']['excluded_live_ids'],
                'sources':source,'answer':r['memory']['answer'],'score':r['memory']['score'],
                'mechanism':mechanism,'policy_branch':ap['policy_branch'],
                'served_from_node_ids':info.get('served_from_node_ids',[])}
        out.append(item)
    write(OUT/'per_probe.json',out);write(OUT/'fold_receipts.json',folds)
    md=['# Fresh and folded: all 30 rows (20 answerable, 10 controls)','',
        'Evidence: lead model receipts joined on probe ID; exact recorded rankings, identifiers and mounts. R = retired, A = active. Raw-source retirement at probe is reconstructed from accepted fold events, checked against the cell-end manifest; source IDs are joined through turn_records, never equated across revisions. Only folds strictly before the probe count. No ranking is recomputed.','',
        '| Probe / turn / expected | r2 mounted; identified; ranking | r2 raw source ID:state | r3 mounted; identified; ranking | r3 raw source ID:state | Mechanism |',
        '|---|---|---|---|---|---|']
    for r in out:
        a,b=(r['versions'][v] for v in ('r2','r3'))
        fmt=lambda x:' / '.join(json.dumps(x[k]) for k in ('mounted_ids','identified','ranking'))
        state=lambda x:', '.join(f"{n['id']}:{'R' if n['retired_at_probe'] else 'A'}" for n in x['sources'])
        md.append(f"| {r['probe_id']} / {r['turn']} / {r['expected'].replace('|',' / ')} | {fmt(a)} | {state(a)} | {fmt(b)} | {state(b)} | {b['mechanism']} |")
    for r in out:
        md += ['',f"## {r['probe_id']}"]
        for v,x in r['versions'].items():
            md += ['',f"{v}: `{x['receipt']}`; checkpoint `{x['checkpoint']}`.",
                f"Identifiers: `{x['identifier_tokens']}`; excluded live: `{x['excluded_live_ids']}`; FIX4 served: `{x['served_from_node_ids']}`.",
                '```text',x['answer'],'```',x['mechanism']+'.']
    (OUT/'PER_PROBE.md').write_text('\n'.join(md)+'\n')
    return out

def register():
    c=census('r3');fixture={p['id']:p for p in read(R3/'fixture.json')['probes']}
    requests=[]
    for pid,x in sorted(c.items(),key=lambda kv:kv[1]['row']['turn']):
        p=fixture[pid];r=x['row']
        if p['expected']!='UNKNOWN' and p['class'] not in ('fresh','folded'):continue
        cp=ROOT/x['checkpoint'];descriptor=read(cp/'checkpoint.json')
        mounts=r['memory']['residency']['mounted_ids']
        requests.append({'probe_id':pid,'probe':p,'historical':r['memory'],
            'plain':r['question'],'middle':MIDDLE+r['question'],
            'receipt':x['receipt'],'receipt_line':x['line'], 'checkpoint':x['checkpoint'],
            'checkpoint_sha256':sha(cp/'checkpoint.json'),
            'mounted_payload_sha256':{str(i):descriptor['files'][f'repository/nodes/{i:04d}.npz'] for i in mounts},
            'mounted_ids':mounts,'mode':'fixed_mount_reader' if mounts else 'recorded_admission_refusal'})
    write(OUT/'requests.json',requests)
    # Prior art: RD1/C2 pessimistic lease reservation (2026). Six bounded
    # sequential batches; round-robin by mounted vs refused rows balances
    # actual paired forwards, with deterministic membership fixed pre-gate.
    groups=[[] for _ in range(6)]
    for mode in ('fixed_mount_reader','recorded_admission_refusal'):
        for j,q in enumerate(x for x in requests if x['mode']==mode):groups[j%6].append(q['probe_id'])
    source_manifest=read(R3/'source_manifest.json')
    paths=[ROOT/p for p in source_manifest if p.startswith(('core/','scripts/','config/'))]
    paths += [Path(p) for p in source_manifest if p.startswith('/home/vader/.cache/') or p.endswith('libgrm_runtime.so')]
    paths += [Path(__file__), ROOT/'scripts/grm_c7_middle_replay.py', ROOT/'tests/test_grm_c7_amendment7.py',
        OUT/'requests.json',OUT/'lead_commands.txt',ROOT/'orders/GRM_C7_AMENDMENT_7.md',R3/'registration.json',R3/'fixture.json',R3/'source_manifest.json']
    paths += [ROOT/q['receipt'] for q in requests]
    r3=read(R3/'registration.json')
    reg={'schema':'grm.c7.amendment7.middle.v1','status':'REGISTERED_NOT_RUN',
        'agent_model':'gpt-6-astra','agent_effort':'high','identity_receipt':'logs/grm_c7_a7.log:6,10',
        'cohort':{'controls':20,'fresh_answerable':10,'folded_answerable':10,'total':40},
        'middle_prefix':MIDDLE,'plain_example':requests[0]['plain'],'middle_example':requests[0]['middle'],
        'flags':r3['arms']['A']['flags'],'model':r3['model'],'model_frame':r3['model_frame'],
        'native_library':'/mnt/ForgeRealm/GraftRepository/cpp/build/libgrm_runtime.so',
        'ngen':32,'batch_ids':[f'B{i+1}' for i in range(6)],
        'batches':{f'B{i+1}':g for i,g in enumerate(groups)},'worker_seconds':280,'lease_seconds':280,
        'max_lease_seconds':285,'outer_envelope_seconds':590,'cooldown_seconds':30,'gpu_cap_seconds':1800,
        'maximum_reserved_seconds':1680,'free_space_min_bytes':20*1024**3,
        'budget_basis':'Six pessimistic 280-second reservations = 1680s = 0.4667 GPU-h; not measured throughput. 30s cooldown outside GPU lease. Non-fit/timeout stops without retry or cohort trimming; native alarm overrun is RED and charged in full.',
        'replay_scope':'Fixed recorded residency reader contrast from each end-of-cell checkpoint. Not a pre-probe routing replay. Cold payload restoration uses copied checkpoint. Clean ephemeral _attempt for A0/A1. No reconstruction, rerouting, deposits, source text in live prompt, or raw-source rescue.',
        'refusal_rule':'Rows with no mounts and identifier_unbound retain recorded refusal on both sides; zero forwards, explicitly policy carry-forward, not generated evidence.',
        'baseline_rule':'For every generated pair A0 UTF-8 answer must equal historical r3 before A1 runs. Record mismatch then STOP entire campaign. All 40 pairs required for any cohort verdict.',
        'score_rule':'Unmodified scripts.grm_c7_common.score; preserve exact/wrong/abstention/unsupported separately. No additional case or glyph normalization.',
        'predictions':{'controls_unsupported_answer_error_at_most':5,'fresh_exact_correct_at_least':2,
            'fresh_admission_refusals':8,'folded_exact_correct':0,'folded_admission_refusals':6,
            'interpretation':'Middle prompt reduces unsupported controls but cannot recover absent mounted information or bypass admission.'},
        'success_rule':'All 40 pairs complete, generated A0 raw parity, controls unsupported <=5/20, fresh exact >=2/10. Report folded residual independently; no product certification.',
        'gates':['40 unique rows with 20/10/10 cohort','prompt entity rare-token parity (no uppercase UNKNOWN trap)',
            'checkpoint descriptor and selected payload SHA','frozen scorer hand cases','actual predicate Unicode RED and existing normalizer-only counterfactual',
            'fake numerical attempt: same ordered mounts, clean state, no deposits','A0 mismatch receipt and stop; missing row never PASS',
            'six reservations <=1800; unsafe duplicate launch rejected'],
        'prior_art':PRIOR,'inputs':{str(p):sha(p) for p in sorted(set(paths))}}
    write(REG,reg)
    with REG.with_suffix('.sha256').open('x') as f:f.write(sha(REG)+'  registration.json\n')
    return reg

def verify(payloads=True):
    need(sha(REG)==REG.with_suffix('.sha256').read_text().split()[0],'REGISTRATION_SHA_MISMATCH')
    r=read(REG)
    for p,d in r['inputs'].items():need(sha(p)==d,'INPUT_SHA_MISMATCH: '+p)
    for q in read(OUT/'requests.json'):
        cp=ROOT/q['checkpoint'];need(sha(cp/'checkpoint.json')==q['checkpoint_sha256'],'CHECKPOINT_SHA_MISMATCH')
        d=read(cp/'checkpoint.json')
        need(d['binding']['registration_sha256']==sha(R3/'registration.json'),'CHECKPOINT_BINDING_MISMATCH')
        for name in ('repository/manifest.json','state.json'):
            need(sha(cp/name)==d['files'][name],'CHECKPOINT_METADATA_MISMATCH')
        if payloads:
            for i,h in q['mounted_payload_sha256'].items():need(sha(cp/f'repository/nodes/{int(i):04d}.npz')==h,'PAYLOAD_SHA_MISMATCH')
    return r

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--diagnose',action='store_true');ap.add_argument('--register',action='store_true');ap.add_argument('--verify',action='store_true');a=ap.parse_args()
    if a.diagnose:print(json.dumps({'diagnosis_rows':len(diagnose()),'gpu_executed':False}))
    if a.register:print(json.dumps({'registration':str(REG),'sha256':sha(REG)} if REG.exists() else {'status':register()['status'],'sha256':sha(REG)}))
    if a.verify:print(json.dumps({'status':verify()['status'],'gpu_executed':False}))
