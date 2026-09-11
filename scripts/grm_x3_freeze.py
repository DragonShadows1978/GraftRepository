#!/usr/bin/env python3
"""Author GRM-X3 registration/inputs before gates; exclusive creation only.
Prior art: house DET1/RS3 registration and fixture lineage (2026); reused freeze
and chronological battery construction. Fixed 8/6/6 allocation is the order's;
no prior art known to me for this particular allocation or digit-decoy rule.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/grm_x3'
MODEL = Path('/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee')

def canonical(x):
    return (json.dumps(x, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def create(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        f.write(canonical(x))

def record(p):
    return {'path': str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p), 'sha256': sha(p)}

def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(MODEL / 'tokenizer.json'))
    sources = []
    for p in sorted((ROOT / 'tests/fixtures/supersession_battery').glob('*.json')):
        d = json.loads(p.read_text())
        for probe in d['probes']:
            target = next(n for n in d['nodes'] if n['node_id'] == probe['target_node'])
            sources.append({'key': d['session_id'] + '/' + probe['probe_id'], 'source':record(p),
                'nodes':d['nodes'], 'target_node':target['node_id'], 'target_text':target['text'],
                'value':target['value'], 'question':probe['question']})
    # Extract only the battery's pure constructors, without importing GPU code.
    p = ROOT / 'scripts/grm_e2e_session.py'
    names = {'plant_acceptance','fact_turn','supersede_turn','filler_turn','probe_turn','build_full_script'}
    tree = ast.parse(p.read_text())
    scope = {'Any':Any}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), str(p), 'exec'), scope)
    events = scope['build_full_script']()
    for i in [5,9,13]:
        e = events[i]; target = events[e['source_turn']]
        nodes = [{'node_id':f't{j:02d}', 'text':'User: '+n['user']+'\nAssistant: '+n['assistant'], 'supersedes':[]}
                 for j,n in enumerate(events[:i]) if n['kind'] in ('fact','supersede')]
        # This derivative uses known completed fact turns only; no invented live replies.
        sources.append({'key':f'e2e/t{i:02d}', 'source':record(p), 'nodes':nodes,
            'target_node':f't{e["source_turn"]:02d}',
            'target_text':'User: '+target['user']+'\nAssistant: '+target['assistant'],
            'value':e['expected'], 'question':e['user']})
    assert len(sources) == 8
    sham = 'Weather calm.'
    def harmony(text):
        u,a=text[6:].split('\nAssistant: ',1)
        return '<|start|>user<|message|>'+u+'<|end|>\n<|start|>assistant<|channel|>final<|message|>'+a+'<|end|>\n'
    # Pin the real formatter verbatim as a separate frozen source receipt.
    formatter = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='harmony_turn')
    fs={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'SYSTEM_PREFIX','ASSISTANT_FINAL'}};exec(compile(ast.Module(body=[formatter],type_ignores=[]),str(p),'exec'),fs)
    def formatted(text):
        u,a=text[6:].split('\nAssistant: ',1);return fs['harmony_turn'](u,a)
    recipes=[]
    for group,n in [('correct',8),('decoy',6),('refusal',6)]:
        for idx,s in enumerate(sources[:n]):
            text=s['target_text']; value=s['value']
            digit=next(c for c in value if c.isdigit()); decoy=value.replace(digit,str((int(digit)+1)%10))
            dt=text.replace(value,decoy)
            # Refusal setup mounts two unrelated facts; the queried source is omitted.
            base_text = text if group=='correct' else dt if group=='decoy' else 'User: The unrelated window log says closed.\nAssistant: Recorded closed window.'
            swap_text = dt if group=='correct' else text if group=='decoy' else base_text.replace('closed','opened')
            base_ids=tok.encode(formatted(base_text),add_special_tokens=False).ids
            swap_ids=tok.encode(formatted(swap_text),add_special_tokens=False).ids
            sham_ids=tok.encode(sham,add_special_tokens=False).ids
            if len(base_ids)!=len(swap_ids) or len(base_ids)+len(sham_ids)>96:
                raise RuntimeError(f'equal length/96-row rail: {group} {s["key"]}: {len(base_ids)}, {len(swap_ids)}, {len(sham_ids)}')
            recipes.append({'schema':'grm.x3.fixture.v1', 'id':f'x3_{len(recipes):02d}',
                'intended_group':group,'evidence_class':'reasoning; frozen input recipe, NOT a captured GPU snapshot',
                'battery':s, 'question':s['question'], 'expected_values':[value], 'decoy_value':decoy,
                'base_text':base_text, 'swap_text':swap_text, 'sham_text':sham,
                'base_token_ids':base_ids, 'swap_token_ids':swap_ids, 'sham_token_ids':sham_ids,
                'controlled_mount_plan':['target','sham'],
                'sham_semantics':'Unrelated factual negative control; zero causal influence is a hypothesis, not assumed.',
                'category_guard':'Observed answers must realize group; no replacement or post-hoc selection.'})
    reg={
        'schema':'grm.x3.registration.v1','immutable':True,'order':record(ROOT/'orders/GRM_X3_MEMORY_LESIONS.md'),
        'evidence_class':'reasoning; preregistration before any CPU/GPU gate',
        'model_id':'openai/gpt-oss-20b','revision':MODEL.name,'tokenizer':record(MODEL/'tokenizer.json'),
        'frame':{'arena_width':96,'GRM_PERSISTENT_BOAT':'0','GRM_CAPTURE_PIN':'live','GRM_SEAT_NEAR_LIVE':'1',
                 'GRM_LSR_FIXES':'1','GRM_DEMAND_NGH':'0','routing':'RT1 source retained; diagnostic controlled picks after chronological feed'},
        'lead_predictions':{
            'P1':'same_payload AND sham KL < 0.05 nats jointly on >=18/20 snapshots',
            'P2':'remove-source KL > 1 nat on >=6/8 observed-correct, correct-mount cases',
            'P3':'decoy mass high on >=4/6 observed-decoy cases AND lesion classifier strictly better exact-error accuracy than mass classifier on those six'},
        'seat_predictions':{
            'Q1':'same_payload raw first-token logits bitwise equal on 20/20; sham KL <0.05 on >=18/20',
            'Q2':'remove-source KL >1 on <=5/8 correct-mount cases (first-token statistic may miss later value tokens)',
            'Q3':'lesion classifier accuracy <= mass classifier accuracy across all 20; reliance alone does not identify truth'},
        'scoring':{'kl':'KL(P_unforked || P_fork), float64 log-softmax over full vocabulary, natural log; no clipping',
            'token_position':'first predicted token after the identical full question prefill, no teacher forcing',
            'effect_high':'remove KL > 1.0 (strict)', 'mass_high':'>= '+str(json.loads((ROOT/'config/grm_demand_registered.json').read_text())['threshold']),
            'mass_threshold':json.loads((ROOT/'config/grm_demand_registered.json').read_text())['threshold'],
            'mass_caveat':'D-NGH threshold carried from TWO full-mount calibration turns; applied to target mount here, never refit.',
            'mass_statistic':'unforked first-position mean over full-attention layers and heads, target span only; D-NGH arithmetic',
            'error':'casefold+outer-whitespace exact whole answer differs from every expected value; refusals count as factual errors',
            'refusal':'normalized answer starts with not in memory, i don\'t know, i do not know, i cannot, i can\'t, or unknown',
            'truncation':'ngen=40, no stop reached => truncated; prediction gates unevaluable (RED), keep rows',
            'mass_error_classifier':'mass < registered mass threshold (D-NGH direction)',
            'lesion_error_classifier':'remove KL > 1.0; fixed positive association, no label-dependent inversion',
            'metric':'paired accuracy (#correct predictions / N); P3 decoy subset, primary overall20 also reported; ties fail improvement',
            'table':'4 cells on all20 unforked target mass vs remove KL; count, exact_errors, error_rate; empty cells null, never zero'},
        'forks':['unforked','zero','remove','swap','same_payload','sham'],
        'fork_contract':'remove/sham disable only target columns in canonical attention masks (no row compaction); swaps change only same-length active K/V span; all remaining canonical host bytes identical',
        'control_scope':'unrelated sham can compete for attention; not a mathematical zero; ZERO replay measures hydration noise separately',
        'fixture_phase':'freeze 20 recipes now; lead captures 20 actual model states before fork gates, in four cells of five. Actual strata are pending.',
        'deviations':['No GPU snapshots exist in sandbox; recipes are not snapshots.',
            'Controlled direct _attempt after chronological fact feeds; not a natural routed full-battery serving receipt.',
            'Mask exclusion replaces physical deletion to preserve all other row positions.',
            'E2E-derived recipes retain completed fact turns; filler and previous generated probes are not replayed.'],
        'rails':{'worker_seconds':285,'outer_seconds':590,'cooldown_seconds':30,'lock_wait_seconds':240,
            'gpu_budget_seconds':1800,'maximum_launches':4,'cells':4,'snapshots_per_cell':5,
            'retry':'none; failed or abandoned starts remain consumed; resume only never-started cells',
            'kill':'controls comparable to factual interventions (max(same,sham,zero) >= max(remove,swap) with factual max>=0.05 on >=3/20) OR no strict overall accuracy improvement; no serving change'},
        'prior_art':[
            {'work':'Jain and Wallace 2019 Attention is not Explanation','url':'https://aclanthology.org/N19-1357/','used':'attention weight is not sufficient causal attribution; verified primary abstract'},
            {'work':'Meng et al. 2022 Locating and Editing Factual Associations in GPT','url':'https://arxiv.org/abs/2202.05262','used':'controlled activation interventions; no ROME weight editing; verified primary abstract'},
            {'work':'Kullback and Leibler 1951 On Information and Sufficiency','status':'unverified — lead to check','search_terms':'Kullback Leibler 1951 On Information and Sufficiency doi 10.1214/aoms/1177729694','used':'KL divergence'},
            {'work':'house DET1 S3/S4 D-NGH RS3/RS4 EB1 RT1 (2026)','used':'capture/hydration, arithmetic, geometry, repository feed; own code is scoped mask/payload fork adapter and negative controls'},
            {'work':'no prior art known to me','used':'particular fixed accuracy direction, thresholds beyond lead values, digit decoy rule and 8/6/6 allocation; no novelty claim'}],
        'author':{'model_id':'GPT-6 (Codex; exact deployment subvariant unavailable)','reasoning_effort':'not exposed in this session'}
    }
    # All content calculated before exclusive first registration write. No gates above.
    create(OUT/'registration.json',reg)
    rows=[]
    for recipe in recipes:
        p=OUT/'fixtures'/f'{recipe["id"]}.json';create(p,recipe);rows.append(record(p))
    create(OUT/'fixtures/manifest.json',{'schema':'grm.x3.fixture_manifest.v1','counts':{'correct':8,'decoy':6,'refusal':6},'registration':record(OUT/'registration.json'),'fixtures':rows,
        'source_pins':[record(ROOT/p) for p in ['config/grm_live_registered_baselines.json','config/grm_demand_registered.json','core/grm_frame.py','core/graft_arena.py','core/gpt_oss20b_tc.py','scripts/grm_det1_3_snapshot.py','scripts/grm_det1_3_gpu.py','scripts/grm_det1_common.py','scripts/grm_e2e_session.py']]})
    (OUT/'fixtures/SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted((OUT/'fixtures').glob('*.json'))))
    print(json.dumps({'registration_sha256':sha(OUT/'registration.json'),'fixture_count':len(rows)}))

if __name__=='__main__': main()
