#!/usr/bin/env python3
"""RD2 receipt joins and CPU metadata replay; never load numerical backends.

Prior art: GRM RD1/C7 receipt/checkpoint contracts, ADM1/A-DEC, L2/M5,
RS3 and RT1 (GRM contributors, 2026), verified in local source. Reuse exact
stdlib-only code or AST-extracted methods. Ours: joins and presentation.
No prior art known to me for this exact diagnostic composition. No novelty claim.
Fix candidates are reported only: reuse explicit revision links (C2/DET1,
GRM 2026); alias dependency retrieval is a lead to check, no implementation.
"""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import runpy
import types

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
C7 = Path('/mnt/ForgeRealm/wt/grm-c7/artifacts/grm_c7')
CAM = ROOT/'artifacts/grm_rd1/amendment_1/gpu_1'
inputs = {}
def read(p):
    p = Path(p); raw = p.read_bytes()
    inputs[str(p)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)
def write(name, obj):
    p = OUT/name
    with p.open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False); f.write('\n')
def method(path, name, globals_):
    # Prior art: Python AST extraction; unchanged method bodies, no module
    # numerical imports. This tests metadata decisions, not native scoring/KV.
    path = ROOT/path; raw = path.read_bytes()
    inputs[str(path)] = hashlib.sha256(raw).hexdigest()
    tree = ast.parse(raw)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    env = dict(globals_); exec(compile(module, str(path), 'exec'), env)
    return env[name]
def projection(n):
    return {k:n.get(k) for k in ('text','ntok','retired','no_fold','sources')} | {
        'metadata':{k:n.get('metadata',{}).get(k) for k in ('active','supersedes','superseded_by','width_guard_child','width_guard_parent')},
        'capture':n.get('capture')}
def content(n):
    t=n['text']
    return t.split('<|start|>user<|message|>',1)[1].split('<|end|>',1)[0] if '<|start|>user<|message|>' in t else t

os.environ['CUDA_VISIBLE_DEVICES']=''
reg=read(OUT/'registration.json')
for p,h in reg['input_sha256'].items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
assert hashlib.sha256((OUT/'registration.json').read_bytes()).hexdigest()==(OUT/'registration.sha256').read_text().split()[0]
adm=runpy.run_path(str(ROOT/'core/grm_admission.py'))
frame=runpy.run_path(str(ROOT/'core/grm_frame.py'))
for p in ('core/grm_admission.py','core/grm_frame.py'):
    inputs[str(ROOT/p)]=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
class Arena: pass
for name in ('_metadata_node_ids','_revision_mount_heads','_resolve_revision_mounts','_rs3_seat_plan','_split_family_members'):
    setattr(Arena,name,method('core/graft_arena.py',name,frame))
co=read(ROOT/'artifacts/grm_rd1/cohort.json')
fixture=read(C7/'fixture.json'); turns={x['turn']:x for x in fixture['turns']}
rows=[]
for h in co:
    probe=h['probe']; path=CAM/('A2_'+h['cell_id'])/(probe['id']+'_memory.json')
    if probe['class'] not in ('alias','correction') or not path.exists(): continue
    actual=read(path); assert actual['scores']['frozen']['wrong_value_error']==1
    a0=read(CAM/('A0_'+h['cell_id'])/(probe['id']+'_memory.json'))
    assert a0['r2_byte_equal'] and a0['mounted_payload_sha256']==actual['mounted_payload_sha256']
    sourcefile,lineno=h['receipt'].rsplit(':',1)
    raw=Path(sourcefile).read_bytes();inputs[sourcefile]=hashlib.sha256(raw).hexdigest()
    original=json.loads(raw.splitlines()[int(lineno)-1]); assert original==h['historical']
    cp=Path(h['checkpoint']); before=read(cp/'repository/manifest.json')['nodes']
    state=read(cp/'state.json'); after=read(path.parent/'session/repository/manifest.json')['nodes']
    ids=actual['mounted_ids']; assert ids==original['memory']['residency']['mounted_ids']
    related=[0,1,4,8,9,11,12,13,14]
    for i in related: assert projection(before[i])==projection(after[i]),(probe['id'],i)
    # Snapshot boundary + unchanged relevant node metadata on both sides;
    # prefix event/command scan guards against an intervening correction.
    prefix=[turns[t] for t in h['prefix_turns'] if t>=state['next_turn']]
    for event in prefix:
        assert not (event.get('kind')=='supersede' and 'C7-Vesper' in event.get('correction_command',''))
    a=Arena(); a.grafts=before; a.n_sink=19;a.width=96;a.live_shift=115
    a._rs3_seat_explicit=True;a.revision_resolution=True
    rt=original['memory']['route_info'];prof=rt['_route_observation']['admission_profile']
    ranking=rt['admission_ranking_before_demotion']; identified=rt['admission_identified_candidates']
    assert not a._split_family_members(ranking)
    eligible=[i for i,n in enumerate(before) if not n['retired'] and n.get('kind')!='recall']
    bindings=[i for i in eligible if adm['is_identifier_binding'](candidate_text=before[i]['text'],
       ordered_identifier_tokens=prof['identifier_tokens'],rare_identifier_tokens=prof['rare_identifier_tokens'])]
    assert set(bindings)==set(identified),(probe['id'],bindings,identified)
    plan,branch=adm['policy_plan'](ranking=ranking,identified_candidates=identified,route_margin_1_2=rt['admission_route_margin_1_2'])
    assert plan==rt['admission_rank_plan'] and branch==rt['admission_policy_branch']
    resolved=a._resolve_revision_mounts(plan)
    fit=adm['plan_priority_fit'](plan=plan,candidates=resolved,ntok={i:n['ntok'] for i,n in enumerate(before)},budget=96)
    assert fit['fit_seated']==ids
    assert a._resolve_revision_mounts(ids)==ids
    seat_order,seat=a._rs3_seat_plan(ids);pos=seat['mount_pos0'];positions=[]
    for i in seat_order:
        positions.append({'id':i,'start_inclusive':pos,'end_exclusive':pos+before[i]['ntok']});pos+=before[i]['ntok']
    assert pos==115
    roles={0:'SUPERSEDED semantically (Mica), still active',1:'intermediate Flint, retired',4:'ALIAS BASE / CURRENT Jasper',8:'ALIAS EDGE Signal-0',9:'ALIAS EDGE Signal-1',11:'CURRENT Onyx-911 full turn',12:'CURRENT Onyx-911 fact',13:'CURRENT Onyx-912 full turn',14:'CURRENT Onyx-912 fact'}
    row={'probe_id':probe['id'],'turn':probe['turn'],'distance':probe['distance'],'class':'c' if probe['class']=='alias' else 'b',
      'question':actual['request']['prompt'],'expected':probe['expected'],'served':actual['served_text'],'frozen_scores':actual['scores'],
      'A2_receipt':str(path.relative_to(ROOT)), 'original_receipt':h['receipt'],'checkpoint_before':str(cp),
      'checkpoint_next_turn':state['next_turn'],'prefix_turns':h['prefix_turns'],'metadata_evidence':'predecessor checkpoint; stable projection verified against A2 end manifest, prefix has no relevant correction; not a contemporaneous per-probe metadata dump',
      'mounted_ids':ids,'mounted_nodes':[{'id':i,'role':roles[i],'content':content(before[i]),**projection(before[i])} for i in ids],
      'related_nodes':[{'id':i,'role':roles[i],'content':content(before[i]),**projection(before[i])} for i in related],
      'historical_route':rt,'residency':original['memory']['residency'],
      'CPU_replay':{'evidence_class':'CPU extracted production metadata methods; original ranking supplied, no route score or KV recomputation',
        'identified':bindings,'rank_plan':plan,'branch':branch,'L2_resolved':resolved,'fit':fit,'rs3':seat,'seat_positions':positions},
      'actual_payload_matches_A0':True,'retired_mounted_ids':[i for i in ids if before[i]['retired']]}
    rows.append(row)
assert len(rows)==16 and sum(x['class']=='b' for x in rows)==8 and sum(x['class']=='c' for x in rows)==8
assert not any(x['retired_mounted_ids'] for x in rows)
write('rows.json',rows)

# Prior art: exact correct_memory query-target semantics (GRM M5, 2026).
# Execute its unchanged Python body with persistence/model operations stubbed.
# Neither native active-match lookup nor numerical remember is exercised.
class Repo:
    def _snapshot_state(self):return None
    def _native_active_text_matches(self,q):return None
    def _native_memory_mutation_plan(self,*a,**kw):return None
    def _default_metadata(self,g):return {'active':True}
    def _mark_dirty(self,*a,**kw):pass
    def _native_apply_revision(self,*a):pass
    def _append_wal(self,*a,**kw):pass
    def _mark_mutations(self,*a):pass
    def remember(self,text,metadata,**kw):
        n={'text':text,'metadata':dict(metadata,active=True),'retired':False}
        self.arena.grafts.append(n);return len(self.arena.grafts)-1
Repo.correct_memory=method('core/graft_repository.py','correct_memory',{})
r=Repo();r.arena=types.SimpleNamespace(grafts=[])
# IDs match checkpoint before turn17; no need to invent payloads or regenerate turns.
cp=read(C7/'r2/cells/A-009-016/checkpoint/repository/manifest.json')['nodes']
r.arena.grafts=copy.deepcopy(cp);assert len(cp)==11
mutations=[]
for t,expected_id in [(17,12),(18,14)]:
    event=turns[t];r.arena.grafts.append({'text':event['user'],'retired':False,'metadata':{'active':True}})
    command=event['correction_command'];query,replacement=command.removeprefix('correct memory: ').split(' => ')
    i=r.correct_memory(query,replacement);assert i==expected_id
    mutations.append({'turn':t,'command':command,'replacement_id':i,'supersedes':r.arena.grafts[i]['metadata']['supersedes']})
assert mutations[0]['supersedes']==[1] and mutations[1]['supersedes']==[]
assert not r.arena.grafts[0]['retired'] and r.arena.grafts[1]['retired']
write('correction_CPU_replay.json',{'evidence_class':'CPU exact Python correct_memory body with I/O stubs; checkpoint starting state; metadata only','turn2':turns[2], 'mutation_results':mutations,'node0_retired':r.arena.grafts[0]['retired'],'node1_retired':r.arena.grafts[1]['retired']})

c2path=Path('/mnt/ForgeRealm/wt/grm-c2/artifacts/grm_c2/epochs/scout-fix-2/scores/sup.json')
c2=read(c2path);compare={'receipt':str(c2path),'overall_status':c2['status'],'arms':{}}
for arm,d in c2['result']['arms'].items():
    compare['arms'][arm]={phase:{'correct':v['correct'],'rows':len(v['rows']),'probes':[{k:x.get(k) for k in ('probe_id','correct','served_answer','mounted_ids','metadata_retained')} for x in v['rows'].values()]} for phase,v in d['phases'].items()}
write('C2_comparison.json',compare)

lines=['# GRM-RD2: 16 wrong rows','', 'Evidence: actual A2 responses; original C7 route plans; CPU-checked predecessor metadata. IDs are zero-based. “Content” below is the exact user-content span; full stored Harmony texts are in rows.json. All mounts, including metadata and effective RS3 positions, are detailed there.','', '| Turn / probe | Question (verbatim A2) | Expected | Served (verbatim) | Admission plan → actual mounts; exact mounted content | Class |','|---|---|---|---|---|---|']
for x in rows:
    texts='<br>'.join(f"{n['id']} **{n['role']}**: {n['content']}" for n in x['mounted_nodes'])
    lines.append(f"| {x['turn']} / {x['probe_id']} | {x['question']} | {x['expected']} | {x['served']} | {x['CPU_replay']['rank_plan']} → {x['mounted_ids']}<br>{texts} | ({x['class']}) |")
lines += ['', 'Class counts: (a) 0; (b) 8; (c) 8; (d) 0. There is no unique dominant class. Alias rows are refusals classified wrong by the frozen scorer, not stale values.','', '## Per-turn route and state evidence','']
for x in rows:
    rt=x['historical_route'];cpu=x['CPU_replay']
    lines += [f"### {x['probe_id']} — turn {x['turn']}",f"- A2: `{x['A2_receipt']}`; original route/residency: `{x['original_receipt']}`.", f"- Metadata predecessor: `{x['checkpoint_before']}/repository/manifest.json`; next turn {x['checkpoint_next_turn']}; prefix {x['prefix_turns']}. Relevant projection agrees with A2 session end. Full texts and role mapping: `rows.json`.",f"- Ranking before demotion: `{rt['admission_ranking_before_demotion']}`; identified `{rt['admission_identified_candidates']}`; branch `{rt['admission_policy_branch']}`; rank plan `{cpu['rank_plan']}`; L2 `{cpu['L2_resolved']}`.",f"- RT1: demoted `{rt['admission_split_child_demoted']}`, demoted IDs `{rt['admission_split_demoted_ids']}`, family IDs `{rt['admission_split_family_ids']}`; margin {rt['admission_route_margin_1_2']}, evaluated `{rt['admission_route_margin_evaluated']}` (zero is not evidence of a measured tie).",f"- Fit: seated `{rt['fit_seated']}`; pending shuttle `{rt['fit_shuttle_trips']}`; dropped filler `{rt['fit_dropped_filler']}`. Original trip trace: `{json.dumps(rt['_route_observation']['trips'])}`.",f"- RS3 effective positions, half-open: `{cpu['seat_positions']}`; live starts 115. Retired mounted IDs `{x['retired_mounted_ids']}`."]
    if x['class']=='c':
        lines += ['- Missing CURRENT/base is node 4: “The current C7-AliasBase-0 value is Jasper-711. The current C7-AliasBase-1 value is Jasper-712.” It is active, no_fold=true; no superseded alias value exists. The query binds only Signal edge 8/9. Node 4 does not contain the Signal identifier. Edge costs 49 and base costs 61: 110 > 96; expansion alone would also require a fitting/dependency strategy.']
    else:
        lines += ['- Node 0 (Mica) is active, retired=false, no_fold=true. Intermediate node 1 (Flint) is retired=true/no_fold=true and absent. Node 12 supersedes [1]; node 14 supersedes []; neither has a path to node 0. Onyx full-turn 11/13 is active but pending for width; Onyx fact 12/14 is mounted. These are ordinary turn/fact nodes, not RT1 split children.']
    lines += ['']
(OUT/'ROW_TABLE.md').write_text('\n'.join(lines)+'\n')
write('CPU_receipt.json',{'status':'PASS_METADATA_DIAGNOSIS','rows':16,'class_counts':{'a':0,'b':8,'c':8,'d':0},'retired_mounted':0,'A0_payload_matches':16,'gates':['original receipt identity','A2 wrong census','predecessor/end metadata projection','identifier binding','A-DEC policy','L2 resolution','plan priority fit','RS3 order/offset','correction metadata replay','C2 census'],'limits':'No model, GPU, native router, logits or numerical KV replay; unchanged source methods with original rankings and I/O stubs. No treatment effect tested.'})
write('INPUT_SHA256.json',inputs)
print(json.dumps({'status':'PASS_METADATA_DIAGNOSIS','rows':16,'classes':{'b':8,'c':8},'retired_mounted':0,'correction_replay':mutations},indent=2))
