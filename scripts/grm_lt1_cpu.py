#!/usr/bin/env python3
"""Author CPU baseline; never a language-model quality measurement.
Prior art: GRM C7 r2/a3 numerical doubles and pytest (GRM/Pytest contributors,
2026). Retain actual repository/admission/ladder and attention constructor;
replace numerical reader with a visible-prose parser. No prior art known to
me for this exact parser. It cannot access expected answers at readout.
"""
from pathlib import Path
import argparse
import json
import os
import re
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.grm_lt1 import ROOT,OUT,FIX,REG,read,create,sha,score,verify


def visible_answer(visible,question):
    # Prior art: C7 visible-value regex fake (GRM, 2026). Extend only the
    # numerical stimulus to natural possessive decisions and aliases.
    q=re.search(r"for (.+?)'s (.+?)\?",question)
    if not q: return 'unknown'
    entity,attr=q.groups()
    aliases=dict((alias.casefold(),base) for base,alias in re.findall(r"Let's call (\w+) '([^']+)' from now on",visible))
    entity=aliases.get(entity.casefold(),entity)
    matches=re.findall(r"\b"+re.escape(entity)+"'s "+re.escape(attr)+r" will be ([^.\n]+)",visible)
    return matches[-1].split(', replacing the earlier choice')[0] if matches else 'unknown'


def stage(directory,arm,start,stop):
    import pytest
    from scripts.grm_c7_diagnose import repository,Model
    from scripts.grm_e2e_session import harmony_turn,_probe_ladder_chat
    from scripts.grm_c7_run import oracle,seats
    from core.grm_admission import ordered_identifier_tokens
    patch=pytest.MonkeyPatch()
    repo=repository(directory/'repository',patch)
    a=repo.arena
    a.width=96 if arm=='A' else 256
    a.recency_mounts=2
    statepath=directory/'state.json'
    state=read(statepath) if statepath.exists() else dict(next_turn=1,turn_nodes={},process_ids=[],rows=[])
    if state['next_turn']!=start: raise ValueError('CPU_RESTART_NEXT_TURN')
    if start>1:
        a.m.codec.words=state['codec_words']; a.m.codec.ids={w:i for i,w in enumerate(a.m.codec.words)}
    state['process_ids'].append(os.getpid())
    class ProseModel(Model):
        def __call__(self,ids,kv_caches=None,position_offset=0,**kwargs):
            if kv_caches is None:
                text=self.codec.decode(ids[0])
                visible=(self.injected if a.cur_mounts else '')+'\n'+text
                self.fold_output=visible_answer(visible,text)
            return super().__call__(ids,kv_caches,position_offset,**kwargs)
    a.m.__class__=ProseModel
    f=read(FIX); probes={p['turn']:p for p in f['probes']}
    try:
        for e in f['turns'][start-1:stop]:
            t=e['turn']
            if e['kind']=='probe':
                p=probes[t]
                before_calls=len(a.m.calls)
                source_ids=[state['turn_nodes'][str(t)] for t in p['source_turns']]
                nominees=a.eb1_begin_turn()
                if set(source_ids)&set(nominees): raise ValueError('ACTUAL_SOURCE_IS_RECENCY_NOMINEE')
                ordered,rare=ordered_identifier_tokens(a,p['question'])
                if {'unknown','reply','answer'} & rare: raise ValueError('INSTRUCTION_IDENTIFIER_COLLISION')
                answer,info=_probe_ladder_chat(repo,p['question'],topk=3,ngen=32,max_trips=1,defer_memory=True)
                mounted=list(a.cur_mounts)
                memory_calls=a.m.calls[before_calls:]
                upper=oracle(a,p,32)
                if not all(s in upper['wrapped_prompt'] for s in p['oracle_source_texts']): raise ValueError('ORACLE_PROMPT_MISSING_SOURCE')
                row=dict(probe_id=p['id'],turn=t,expected=p['expected'],answer=answer,
                    memory_score=score(answer,p['expected']),oracle_score=score(upper['answer'],p['expected']),
                    source_ids=source_ids,mounted_ids=mounted,recency_ids=nominees,
                    rare_tokens=sorted(rare),route_info=info,wrapped_oracle_prompt=upper['wrapped_prompt'],
                    memory_initial_inputs=[x for x in memory_calls if x['initial']])
                state['rows'].append(row)
            elif e['kind']!='recap':
                # Same frozen complete-turn replay contract as EB1/C7. No
                # synthetic correction command, expected value or source ID
                # is passed to the serving path or used to influence routing.
                idx=a.feed(harmony_turn(e['user'],e['assistant']))
                a.grafts[idx]['kind']='turn'
                state['turn_nodes'][str(t)]=idx
        repo.flush_now()
        state['next_turn']=stop+1
        state['codec_words']=a.m.codec.words
        # Mutable fake continuation state is test scratch only; registration
        # and actual C7 checkpoint contract remain immutable.
        statepath.write_text(json.dumps(state,indent=2,default=str)+'\n')
    finally:
        repo.close(); patch.undo()


def run():
    verify(); OUT.mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    log=OUT/'cpu_baseline.log'
    with log.open('x') as stream:
        result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_grm_lt1.py','--tb=short','-rA'],
            cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    receipt=dict(status='PASS' if result.returncode==0 else 'RED',returncode=result.returncode,
        registration_sha256=sha(REG),log_sha256=sha(log),evidence_class='author CPU suite; no GPU, no blind validation')
    create(OUT/'cpu_receipt.json',receipt)
    print(json.dumps(receipt,indent=2)); return result.returncode
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--stage',type=Path);p.add_argument('--arm',choices=['A','B']);p.add_argument('--start',type=int);p.add_argument('--stop',type=int)
    args=p.parse_args()
    if args.stage: stage(args.stage,args.arm,args.start,args.stop)
    else: raise SystemExit(run())
