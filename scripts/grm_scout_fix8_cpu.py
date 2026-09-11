"""FIX8 CPU receipts on recorded own text and frozen C2 policy inputs.

Prior art: C7 numerical-boundary doubles and LT1 frozen_c2/evaluate Rule0
(GRM contributors, 2026). Reuse their state/eligibility/ranking contracts and
real core predicates; ours is the per-probe normalization treatment receipt.
No prior art known to me for this exact composition. No GPU/model quality.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.grm_c7_amendment7 import read, sha, write
OUT = ROOT/'artifacts/grm_scout_fix8'
# GRM-F6: was Path('/mnt/ForgeRealm/wt/grm-c2/artifacts/grm_c2/epochs/
# scout-fix-2') -- a pruned seat worktree (H2's finding); the glob below
# returned ZERO rows and the C2 identity census passed vacuously. The epoch
# is canonical in-repo. Prior art: repo-root-from-__file__ (setuptools /
# pytest rootdir, 2009-); vacuous-zero-is-an-error (dbt / Great
# Expectations row-count assertions, 2018). See scripts/grm_repo_paths.py.
from scripts import grm_repo_paths as repo_paths
C2 = repo_paths.receipt_root('artifacts/grm_c2/epochs/scout-fix-2', 'GRM_C2_EPOCH_ROOT',
    'artifacts/grm_scout_fix8/green.json (c2[].source pins the absolute path)',
    '/mnt/ForgeRealm/wt/grm-c2/artifacts/grm_c2/epochs/scout-fix-2')


def requests():
    qs = read(ROOT/'artifacts/grm_c7/r3/amendment_7/requests.json')
    qs = [q for q in qs if q['historical']['route_info'].get('abstain_reason') == 'identifier_unbound']
    assert len(qs) == len({q['probe_id'] for q in qs}) == 24
    return qs


def replay_cpu(q, path, patch):
    from scripts.grm_c7_diagnose import repository
    from core.grm_admission import decisive_admission_profile, identifier_serving_decision
    from scripts.grm_e2e_session import _probe_ladder_chat
    cp = ROOT/q['checkpoint']
    nodes = read(cp/'repository/manifest.json')['nodes']
    repo = repository(path, patch)
    a = repo.arena
    try:
        # Keep stored own text, retirement, kind, lineage and inherited keys.
        # Only payload, tokenizer, embeddings and numerical reader are fakes.
        for node in nodes:
            i = a.deposit(node['text'] or '')
            h, cent, ntok = (a.grafts[i][k] for k in ('h','cent','ntok'))
            a.grafts[i].update(copy.deepcopy(node), h=h, cent=cent, ntok=ntok)
            a.grafts[i]['rare'] = set(node.get('rare', []))
        a._bump_cuda_gqa_epoch()
        a.recency_mounts = 2
        excluded = a.eb1_begin_turn()
        p = decisive_admission_profile(a, q['plain'], exclude=excluded, route_limit=6)
        decision = identifier_serving_decision(a, q['plain'], p, exclude=excluded)
        answer, info = _probe_ladder_chat(repo, q['plain'], topk=3, ngen=32, max_trips=1, defer_memory=True)
        return {'probe_id':q['probe_id'], 'checkpoint':q['checkpoint'],
            'checkpoint_sha256':q['checkpoint_sha256'], 'manifest_sha256':sha(cp/'repository/manifest.json'),
            'historical_refusal':q['historical']['answer'], 'identified':p['identified_candidates'],
            'identifier_tokens':p['identifier_tokens'], 'rank_plan':p['rank_plan'],
            'refused':bool(decision and not decision.get('served_from')),
            'mounted_ids':list(a.cur_mounts),'forward_calls':len(a.m.calls),
            'answer_fake_only':answer,'info':info,
            'binding_texts':{str(i):a.grafts[i]['text'] for i in p['identified_candidates']},
            'evidence_class':'CPU fake numerical boundaries; real route, admission, fit and attempt; cell-end state, not historical pre-probe geometry'}
    finally:
        repo.close()


def c2_plans():
    # Prior art: LT1 offline_supplement.c2_replay/frozen_c2/evaluate (2026).
    # Full manifest is required: the projection omits recall kind. Recompute
    # binding and RT1 with real core, keep measured rank/margin as frozen inputs.
    from core import grm_admission as adm
    from core.graft_arena import ArenaCache
    rows=[]
    for path in repo_paths.require_rows(sorted(C2.glob('cells/*/worker.json')), C2, 'cells/*/worker.json', 'FIX8 C2 identity census'):
        w=read(path);c=w['cell']
        for ordinal,r in enumerate(w['rows']):
            pid=r['probe_id'];cp=C2/'checkpoints'/c['side']/c['battery']/pid/'checkpoint.json'
            descriptor=read(cp);mp=cp.parent/'repository/manifest.json'
            assert sha(mp)==descriptor['files']['manifest.json']
            nodes=read(mp)['nodes'];context=descriptor['context']
            question=context['probe']['question'] if 'probe' in context else context['event']['user']
            assert question.isascii()
            excluded=[] if 'probe' in context else ArenaCache.eb1_begin_turn(SimpleNamespace(
                grafts=nodes,ephemeral=True,recency_mounts=2,live_segs=[],cur_mounts=[],cur_mount_n=0))
            eligible=[i for i,n in enumerate(nodes) if not n.get('retired') and n.get('kind','turn')!='recall' and i not in excluded]
            rr=r['route_receipt'];ad=rr['admission'];info=rr['fit']['info_pass_through']
            ranking=ad.get('ranking_before_demotion',info.get('admission_ranking_before_demotion',rr['route']['ranking_ids']))
            assert set(ranking)<=set(eligible)
            ordered,rare=adm.ordered_identifier_tokens(ArenaCache,question)
            hits=[i for i in eligible if adm.is_identifier_binding(candidate_text=nodes[i]['text'] or '',ordered_identifier_tokens=ordered,rare_identifier_tokens=rare)]
            ranking,demoted=adm.demote_non_binding_split_members(ranking=ranking,binding=hits,
                split_members=ad.get('split_family_ids',info.get('admission_split_family_ids',[])))
            hits=[i for i in ranking if i in hits]+sorted(set(hits)-set(ranking))
            margin=ad['route_margin_1_2'] if ad.get('route_margin_evaluated') else None
            assert margin is not None or not (len(hits)==1 and ranking and hits[0]!=ranking[0])
            plan,branch=adm.policy_plan(ranking=ranking,identified_candidates=hits,route_margin_1_2=margin or 0.)
            before=json.dumps(ad['rank_plan'],separators=(',',':')).encode()
            after=json.dumps(plan,separators=(',',':')).encode()
            rows.append(dict(execution_id=c['id']+':'+pid,source=str(path),source_sha256=sha(path),ordinal=ordinal,
                checkpoint=str(cp),checkpoint_sha256=sha(cp),manifest_sha256=sha(mp),
                recorded_hex=before.hex(),replayed_hex=after.hex(),identical=before==after))
    assert len(rows)==len({r['execution_id'] for r in rows})==132
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=['red','green']);args=p.parse_args()
    import pytest
    with tempfile.TemporaryDirectory(prefix='fix8-cpu-') as tmp, pytest.MonkeyPatch.context() as patch:
        rows=[replay_cpu(q,Path(tmp)/q['probe_id'],patch) for q in requests()]
        c2=c2_plans()
    result=dict(phase=args.phase,rows=rows,c2=c2,bind_count=sum(bool(r['identified']) for r in rows),
        generated_count=sum(r['forward_calls']>0 for r in rows),c2_identical=sum(r['identical'] for r in c2),gpu_executed=False)
    write(OUT/(args.phase+'.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','c2')}))
    return 0 if result['bind_count']==24 and result['generated_count']==24 and result['c2_identical']==132 else 1

if __name__=='__main__':raise SystemExit(main())
