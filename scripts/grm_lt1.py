#!/usr/bin/env python3
"""LT1 CPU contracts and fail-closed lead entrypoint.

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
    return dict(arm=arm,registration_sha256=sha(REG),fixture_sha256=sha(FIX))


def verify():
    if sha(REG)!=REG.with_suffix('.sha256').read_text().split()[0]: raise ValueError('REGISTRATION_SHA_MISMATCH')
    r=read(REG)
    for name,digest in r['immutable_inputs'].items():
        if sha(ROOT/name)!=digest: raise ValueError('INPUT_SHA_MISMATCH: '+name)
    m=read(ROOT/'fixtures/lt1/manifest.json')
    for name,digest in m['files'].items():
        if sha(ROOT/'fixtures/lt1'/name)!=digest: raise ValueError('FIXTURE_SHA_MISMATCH')
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
    for p in sorted((OUT/'cells').glob(f'{arm}-*/probes.jsonl')):
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
    recaps=list((OUT/'cells').glob(f'{arm}-*/recap.json'))
    recap=None
    if len(recaps)>1: raise ValueError('DUPLICATE_RECAP')
    if recaps:
        answer=read(recaps[0])['answer']
        recap=dict(answer=answer,out_of=5,matched=sum(score(answer,p['expected'])['exact_correct'] for p in f['decisions']))
    # Completeness alone never establishes a residency or restart pass.
    return dict(arm=arm,status='NOT_RUN' if not rows else 'INCOMPLETE_UNVALIDATED',
        registration_status=r['status'],evidence_class='no GPU evidence' if not rows else 'partial raw E2E rows',
        measured_recalls=len(rows),by_distance=table,recap=recap,
        residency_bounded=None,restart_retained=None,binding=binding(arm))


def preflight():
    r=verify(); free=shutil.disk_usage(ROOT).free
    reasons=[]
    if free<20_000_000_000: reasons.append('FREE_SPACE_BELOW_20_GB')
    if r['status']!='FIT_ESTIMATE': reasons.append(r['status'])
    receipt=OUT/'cpu_receipt.json'
    if not receipt.exists() or read(receipt).get('status')!='PASS': reasons.append('CPU_GATES_NOT_GREEN')
    elif read(receipt).get('registration_sha256')!=sha(REG): reasons.append('CPU_RECEIPT_BINDING_MISMATCH')
    return dict(status='BLOCKED' if reasons else 'READY',reasons=reasons,free_bytes=free,
        minimum_free_bytes=20_000_000_000,gpu_executed=False,projection=r['projection'],
        model=r['model'],agent_model=r['agent_model'],agent_effort=r['agent_effort'])


def main():
    p=argparse.ArgumentParser(); p.add_argument('--preflight',action='store_true')
    p.add_argument('--summary',choices=['A','B','both']); p.add_argument('--cell')
    p.add_argument('--resume',action='store_true'); args=p.parse_args()
    if args.summary:
        value=summary(args.summary) if args.summary!='both' else dict(A=summary('A'),B=summary('B'),
            comparison='NOT_MEASURED: no paired complete GPU run',recap_comparison='NOT_MEASURED')
        print(json.dumps(value,indent=2)); return 0
    result=preflight(); print(json.dumps(result,indent=2))
    if result['status']!='READY': return 2
    # NON_FIT is a stop rail, not permission to run a subset or invent a
    # bounded launch. A future authorized amendment must register a launcher
    # satisfying strict never-kill and cooperative worker/outer deadlines.
    if args.cell or args.resume: raise ValueError('NO_GPU_LAUNCH_AUTHORIZED_BY_THIS_REGISTRATION')
    return 0
if __name__=='__main__': raise SystemExit(main())
