#!/usr/bin/env python3
"""LT1 CPU contracts and registered margin-first resumable lead entrypoint.

Prior art: GRM contributors, C7/C2/EB1 and amendment 3 (2026), verified
local source: checkpoint hashes, real serving ladder, isolated live oracle,
profile flags, immutable budget registration. Taken unchanged below.
New: natural dialogue and its reporting adapter; no new retrieval algorithm.
Scoring follows the ordered contiguous value-span rule specified by C5 arm S
(GRM, 2026); NFKC is Unicode Standard Annex 15 (Unicode Consortium, 2001;
unverified — lead to check: Unicode UAX15 normalization history).
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import sys
import unicodedata
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.grm_c7_common import create, read, sha, checkpoint, validate_checkpoint
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/grm_lt1'
FIX=ROOT/'fixtures/lt1/dialogue.json'
REG=OUT/'registration.json'
AMEND=OUT/'amendment1/registration_amendment.json'
AMEND2=OUT/'amendment2/registration_amendment.json'
AMEND3=OUT/'amendment3/registration_amendment_r3.json'
RUN=OUT/'amendment2/run_margin_first'


def normalize(value):
    text=unicodedata.normalize('NFKC',str(value))
    text=text.translate(str.maketrans({c:'-' for c in '\u2010\u2011\u2012\u2013\u2014\u2212'}))
    return ' '.join(text.split())


def score(answer, expected):
    # Prior art: C5 arm S (GRM, 2026), as specified in LT1 order. Borrow
    # contiguous ordered spans, not full-string equality. Boundary checks
    # avoid treating 16 as an answer inside 116; case remains significant.
    text=normalize(answer); value=normalize(expected)
    exact=bool(value and re.search(r'(?<!\w)'+re.escape(value)+r'(?!\w)',text))
    low=text.casefold().replace('’',"'")
    refusal=bool(re.search(r"\b(?:i (?:cannot|can't|won't) (?:help|assist|comply)|not able to assist|against (?:my|the) policy|i must refuse)\b",low))
    abstain=bool(re.search(r"\b(?:unknown|not in memory|not (?:provided|specified)|(?:do not|don't|cannot|can't) (?:know|recall|find)|no (?:stored )?record|not enough information)\b",low)) or not text
    category='correct' if exact else 'refusal' if refusal else 'abstention' if abstain else 'wrong_value'
    return dict(exact_correct=exact,category=category,wrong_value_error=int(category=='wrong_value'),
        abstention_error=int(category=='abstention'),refusal_error=int(category=='refusal'),
        refusal_style=refusal,abstention_style=abstain,answer=str(answer))


def binding(arm):
    return dict(arm=arm,registration_sha256=sha(REG),fixture_sha256=sha(FIX),
                amendment_sha256=sha(AMEND3),admission_rule="margin_first",
                core_shas={n:c["after_sha256"] for n,c in read(AMEND3)["core_shas"].items()})


def verify():
    if sha(REG)!=REG.with_suffix('.sha256').read_text().split()[0]: raise ValueError('REGISTRATION_SHA_MISMATCH')
    r=read(REG)
    # Prior art: C7 SHA-bound amendments (GRM, 2026), unchanged original plan.
    inputs=dict(r['immutable_inputs'])
    if AMEND.exists():
        if sha(AMEND)!=AMEND.with_suffix('.sha256').read_text().split()[0]:raise ValueError('AMENDMENT_SHA_MISMATCH')
        a=read(AMEND)
        if a['registration_sha256']!=sha(REG):raise ValueError('AMENDMENT_CHAIN_MISMATCH')
        for name,change in a['overrides'].items():
            if name.startswith('core/'):raise ValueError('CORE_OVERRIDE_FORBIDDEN')
            if inputs[name]!=change['before_sha256'] or sha(ROOT/change['before_archive'])!=inputs[name]:
                raise ValueError('AMENDMENT_BEFORE_MISMATCH')
            inputs[name]=change['after_sha256']
        inputs.update(a['new_inputs'])
        r['status']='FIT_ESTIMATE';r['projection']['budget_seconds']=10800
        for arm in r['arms'].values():arm['status']='FIT_ESTIMATE'
        r['effective_admission_rule']=0
    # Prior art: C7/C2 SHA-chain amendments (GRM, 2026). FIX6 alone authorizes
    # these core overrides; preserve both preceding registrations byte-for-byte.
    if sha(AMEND2)!=AMEND2.with_suffix('.sha256').read_text().split()[0]:
        raise ValueError('AMENDMENT2_SHA_MISMATCH')
    a2=read(AMEND2)
    if a2['previous_amendment_sha256']!=sha(AMEND) or a2['registration_sha256']!=sha(REG):
        raise ValueError('AMENDMENT2_CHAIN_MISMATCH')
    if a2['admission_rule']!='margin_first' or a2['budget_gpu_seconds']!=10800 or a2['cells']!=r['cells']:
        raise ValueError('AMENDMENT2_PROTOCOL_MISMATCH')
    for name,change in a2['overrides'].items():
        if inputs[name]!=change['before_sha256'] or sha(ROOT/change['before_archive'])!=inputs[name]:
            raise ValueError('AMENDMENT2_BEFORE_MISMATCH')
        inputs[name]=change['after_sha256']
    inputs.update(a2['new_inputs'])
    # Prior art: LT1/C7 SHA-chain source amendments (GRM, 2026), reused.
    from scripts.grm_lt1_amendment3 import apply
    apply(sys.modules[__name__], inputs, a2)
    from scripts.grm_lt1_amendment4 import apply as apply4
    apply4(sys.modules[__name__], inputs)
    r['effective_admission_rule']='margin_first'
    for arm in r['arms'].values():arm['admission_rule']='margin_first'
    for name,digest in inputs.items():
        if sha(ROOT/name)!=digest: raise ValueError('INPUT_SHA_MISMATCH: '+name)
    m=read(ROOT/'fixtures/lt1/manifest.json')
    for name,digest in m['files'].items():
        if sha(ROOT/'fixtures/lt1'/name)!=digest: raise ValueError('FIXTURE_SHA_MISMATCH')
    r['effective_inputs']=inputs
    return r


def fixture_gate(f):
    # Prior art: EB1 eb1_begin_turn (GRM, 2026), use actual nominee code
    # against a plan-derived arena; no alternative recency policy.
    from types import SimpleNamespace
    from core.graft_arena import ArenaCache
    events=f['turns']; ps={p['turn']:p for p in f['probes']}
    if [e['turn'] for e in events]!=list(range(1,201)): raise ValueError('TURN_COVERAGE')
    if Counter(e['kind'] for e in events)!=dict(fact=60,correction=15,alias=10,ordinary=79,probe=35,recap=1): raise ValueError('TURN_MIX')
    a=SimpleNamespace(grafts=[],ephemeral=True,recency_mounts=2,live_segs=[],cur_mounts=[],cur_mount_n=0)
    node_turns=[]; receipt=[]
    for e in events:
        rec=ArenaCache.eb1_begin_turn(a)
        if e['turn'] in ps:
            p=ps[e['turn']]; src=p['source_turns']
            if min(e['turn']-t for t in src)<10: raise ValueError('SOURCE_TOO_RECENT')
            if e['turn']-max(src)!=p['distance']: raise ValueError('DISTANCE_MISMATCH')
            if set(src)&{node_turns[i] for i in rec}: raise ValueError('SOURCE_IS_RECENCY_NOMINEE')
            if re.search(r'\b[A-Z][A-Z_]+\b',p['question']): raise ValueError('UPPERCASE_INSTRUCTION_TOKEN')
            if p['oracle_source_texts']!=[events[t-1]['user'] for t in src]: raise ValueError('ORACLE_SOURCE_MISMATCH')
            values=[x for x in events[:e['turn']-1] if x.get('entity')==p['entity'] and x.get('attribute')==p['attribute']]
            if values[-1]['value']!=p['expected']: raise ValueError('STALE_EXPECTATION')
            receipt.append(dict(probe_id=p['id'],sources=src,recency_turns=[node_turns[i] for i in rec],distance=p['distance']))
        if e['kind'] not in ('probe','recap'):
            a.grafts.append(dict(kind='turn',ntok=1)); node_turns.append(e['turn'])
    if Counter(p['distance'] for p in ps.values())!={10:7,25:7,50:7,100:7,150:7}: raise ValueError('DISTANCE_COVERAGE')
    if events[-1]['user']!='recap the five biggest decisions we made': raise ValueError('RECAP_WORDING')
    return receipt


def summary(arm):
    r=verify(); f=read(FIX); ps={p['id']:p for p in f['probes']}
    rows=[]
    for p in sorted((RUN/'cells').glob(f'{arm}-*/probes.jsonl')):
        # Prior art: LT1 stop-on-RED cell accounting (GRM, 2026). Partial
        # crashed-cell rows are evidence, not committed/scorable turns.
        ctl=p.parent/'controller.json'
        if not ctl.exists() or read(ctl).get('status')!='COMPLETE': continue
        rows.extend(json.loads(line) for line in p.read_text().splitlines())
    ids=[x['probe_id'] for x in rows]
    if len(ids)!=len(set(ids)) or set(ids)-set(ps): raise ValueError('DUPLICATE_OR_UNKNOWN_PROBE')
    table=[]
    for d in f['distances']:
        for cls in ['all','fresh','correction','alias']:
            expected=[p for p in ps.values() if p['distance']==d and (cls=='all' or p['class']==cls)]
            selected=[x for x in rows if x['probe_id'] in {p['id'] for p in expected}]
            row=dict(distance=d,category=cls,expected_n=len(expected),n=len(selected))
            for side in ('memory','oracle'):
                scores=[score(x[side]['answer'],ps[x['probe_id']]['expected']) for x in selected]
                counts=Counter(s['category'] for s in scores)
                row[side]=dict(counts,exact_rate=counts['correct']/len(scores) if scores else None)
            table.append(row)
    recaps=list((RUN/'cells').glob(f'{arm}-*/recap.json'))
    recap=None
    if len(recaps)>1: raise ValueError('DUPLICATE_RECAP')
    if recaps:
        answer=read(recaps[0])['answer']
        recap=dict(answer=answer,out_of=5,matched=sum(score(answer,p['expected'])['exact_correct'] for p in f['decisions']))
    # Prior art: C7 complete-cell plus seat/restart receipts (GRM, 2026).
    # Completeness alone never establishes quality, residency or restart pass.
    from scripts.grm_lt1_amendment4 import cell_directory
    completed=[cell_directory(RUN/'cells',c['id'])/'controller.json'
               for c in r['cells'] if c['arm']==arm
               and (cell_directory(RUN/'cells',c['id'])/'controller.json').exists()]
    workers=[read(p.with_name('worker.json')) for p in completed if p.with_name('worker.json').exists()]
    complete=len(completed)==26 and len(workers)==26 and len(rows)==35 and recap is not None and all(read(p)['status']=='COMPLETE' for p in completed)
    seats=[]
    for path in (RUN/'cells').glob(f'{arm}-*/residency.jsonl'):
        seats.extend(json.loads(line) for line in path.read_text().splitlines())
    residency=bool(seats) and all(x['summed_token_seats']<=2*x['width']+x['actual_recency_token_seats'] for x in seats) if complete else None
    restarts=[w for w in workers if w['cell']['start'] in (71,141)]
    retained=len(restarts)==2 and all(w['restart_retained_scores'] and w['metadata_retained'] and w['pid']!=w['previous_pid'] for w in restarts) if complete else None
    quality=all(row['memory']['exact_rate']>=0.8 for row in table if row['category'] in ('fresh','correction')) if complete else None
    status=('PASS' if quality and residency and retained else 'RED') if complete else 'NOT_RUN' if not rows else 'INCOMPLETE_UNVALIDATED'
    return dict(arm=arm,status=status,complete=complete,quality_threshold_met=quality,
        registration_status=r['status'],evidence_class='no GPU evidence' if not rows else 'partial raw E2E rows',
        measured_recalls=len(rows),by_distance=table,recap=recap,
        residency_bounded=residency,restart_retained=retained,binding=binding(arm))


def preflight():
    r=verify(); free=shutil.disk_usage(ROOT).free
    reasons=[]
    from scripts.grm_lt1_amendment4 import check_original_evidence, cpu_ready
    check_original_evidence()
    if not cpu_ready(): reasons.append('AMENDMENT4_CPU_GATES_NOT_GREEN')
    if free<20_000_000_000: reasons.append('FREE_SPACE_BELOW_20_GB')
    if r['status']!='FIT_ESTIMATE': reasons.append(r['status'])
    receipt=OUT/'amendment3/r3/cpu_receipt.json'
    if os.environ.get('GRM_ADMISSION_RULE')!='margin_first': reasons.append('REGISTERED_ADMISSION_RULE_MISMATCH')
    if not receipt.exists() or read(receipt).get('status')!='PASS': reasons.append('CPU_GATES_NOT_GREEN')
    elif read(receipt).get('binding')!=binding('CPU'): reasons.append('CPU_RECEIPT_BINDING_MISMATCH')
    elif read(receipt).get('fix4_check')!='PASS': reasons.append('FIX4_PREREQUISITE_NOT_GREEN')
    return dict(status='BLOCKED' if reasons else 'READY',reasons=reasons,free_bytes=free,
        minimum_free_bytes=20_000_000_000,gpu_executed=False,projection=r['projection'],
        model=r['model'],agent_model=r['agent_model'],agent_effort=r['agent_effort'],
        binding=binding('CPU'),fix4_check=read(receipt).get('fix4_check') if receipt.exists() else 'NOT_RUN')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--preflight',action='store_true')
    p.add_argument('--summary',choices=['A','B','both']); p.add_argument('--cell')
    p.add_argument('--resume',action='store_true'); p.add_argument('--dry-run',action='store_true'); args=p.parse_args()
    if args.dry_run and args.resume: p.error('--dry-run and --resume are mutually exclusive')
    if args.summary:
        value=summary(args.summary) if args.summary!='both' else dict(A=summary('A'),B=summary('B'),
            comparison='See per-arm complete/status and matched per-distance rates; no reader claim from offline admission',recap_comparison='See per-arm recap; null means NOT_RUN')
        print(json.dumps(value,indent=2)); return 0
    result=preflight(); print(json.dumps(result,indent=2))
    if result['status']!='READY': return 2
    if args.dry_run:
        from scripts.grm_lt1_worker import pending
        r=verify()
        print(json.dumps(dict(status='PASS',mode='CPU_DRY_RUN',gpu_executed=False,
            next_cell=pending(r),cells=len(r['cells']),binding=binding('CPU')),indent=2))
        return 0
    if args.cell: raise ValueError('USE_RESUME_FOR_REGISTERED_INTERLEAVING')
    if args.resume:
        from scripts.grm_lt1_worker import resume
        return resume()
    return 0
if __name__=='__main__': raise SystemExit(main())
