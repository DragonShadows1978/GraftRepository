#!/usr/bin/env python3
"""Exclusive r2 authoring, before gates; never updates r1 or an existing r2.
Prior art: house X3/DET1 (2026) manifest SHA256 registration; reused schema and
8/6/6 layout. New deterministic relational recipes and prior-run sham calibration
are amendment 2 choices; no prior art known to me for these particular choices.
"""
import ast
import copy
import json
from pathlib import Path
import unicodedata
from scripts.grm_x3_diagnostic import ROOT, create, sha, digest
from scripts.grm_x3_r2_scorer import DASH_POINTS, REFUSALS
OUT=ROOT/'artifacts/grm_x3/r2'

def record(p):
    p=Path(p)
    return {'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'sha256':sha(p)}

def main():
    from tokenizers import Tokenizer
    old=json.loads((ROOT/'artifacts/grm_x3/registration.json').read_text())
    tok=Tokenizer.from_file(old['tokenizer']['path'])
    p=ROOT/'scripts/grm_e2e_session.py';tree=ast.parse(p.read_text())
    scope={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'SYSTEM_PREFIX','ASSISTANT_FINAL'}}
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='harmony_turn')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),scope)
    def formatted(text):
        u,a=text[6:].split('\nAssistant: ',1);return scope['harmony_turn'](u,a)
    entities=['Asterwick','Brindleford','Cairnvale','Dunmere','Elmspire','Fenquay','Glimmerfen','Hearthwick','Iverstone','Juniperwell','Kestrelgate','Lumenbrook','Morrowfield','Northwisp','Orchardreach','Peregrinebay','Quillhaven','Rillstone','Silverweald','Thornmere']
    recipes=[];controls=[]
    for i,entity in enumerate(entities):
        group='correct' if i<8 else 'decoy' if i<14 else 'refusal'
        value=f'copper-{731+i} before indigo-{851+i}'
        decoy=f'copper-{831+i} before indigo-{851+i}'
        text=f'User: The {entity} ordering is {value}.\nAssistant: Recorded: {value}.'
        dt=text.replace(value,decoy)
        base=text if group=='correct' else dt if group=='decoy' else 'User: The unrelated window log says closed.\nAssistant: Recorded closed window.'
        swap=dt if group=='correct' else text if group=='decoy' else base.replace('closed','opened')
        f={'schema':'grm.x3.fixture.v2','id':f'x3_r2_{i:02d}','entity':entity,'intended_group':group,
           'evidence_class':'reasoning; frozen new recipe, not GPU snapshot',
           'question':f'What is the recorded ordering for {entity}?',
           'expected_values':[value],'decoy_value':decoy,'base_text':base,'swap_text':swap,'sham_text':'Weather calm.',
           'battery':{'key':f'r2/{entity}','source':record(ROOT/'orders/GRM_X3_AMENDMENT_2.md'),
               'lineage':'X3 completed fact-turn constructor; independently authored new entity/value/question',
               'nodes':[{'node_id':'target','text':text,'value':value,'supersedes':[]}],
               'target_node':'target','target_text':text,'value':value},
           'controlled_mount_plan':['target','sham'],
           'category_guard':'No replacement or post-hoc selection; registered value-span quotas',
           'sham_semantics':old['control_scope']}
        for k,t in [('base',formatted(base)),('swap',formatted(swap)),('sham',f['sham_text'])]:
            f[k+'_token_ids']=tok.encode(t,add_special_tokens=False).ids
        # Recipe construction constraints, before registration; no outcome gates.
        if len(f['base_token_ids'])!=len(f['swap_token_ids']) or len(f['base_token_ids'])+len(f['sham_token_ids'])>96:
            raise RuntimeError('construction geometry: '+f['id'])
        recipes.append(f)
        for v in (value,decoy):
            left,_,right=v.split()
            controls.extend([
                {'kind':'positive_exact','answer':v,'value':v,'accept':True},
                {'kind':'positive_wrapped','answer':'The ordering is **'+v.upper().replace('-', '\u2011').replace(' ', '\n\t')+'**.','value':v,'accept':True},
                {'kind':'changed_digit','answer':v.replace('7','9',1) if '7' in v else v.replace('8','9',1),'value':v,'accept':False},
                {'kind':'omitted_token','answer':left+' '+right,'value':v,'accept':False},
                {'kind':'swapped_relation','answer':right+' before '+left,'value':v,'accept':False},
                {'kind':'changed_relation','answer':left+' after '+right,'value':v,'accept':False},
                {'kind':'negation','answer':'The ordering is not '+v+'.','value':v,'accept':False},
                {'kind':'negation_suffix','answer':v+' is incorrect.','value':v,'accept':False},
                {'kind':'token_boundary','answer':v+'0','value':v,'accept':False}])
    for phrase in REFUSALS:
        controls.extend([{'kind':'refusal_positive','answer':'Answer: **'+phrase+'**.','value':phrase,'refusal':True,'accept':True},
                         {'kind':'refusal_negation','answer':'It is not true that '+phrase,'value':phrase,'refusal':True,'accept':False}])
    create(OUT/'scorer_controls.json',{'cases':controls,'evidence_class':'reasoning; registered before CPU tests'})
    reg=copy.deepcopy(old);reg.update(schema='grm.x3.registration.r2.v1',order=record(ROOT/'orders/GRM_X3_AMENDMENT_2.md'),
        parent_registration=record(ROOT/'artifacts/grm_x3/registration.json'),r1_evidence=record(OUT/'r1_evidence_before.json'),
        author={'model_id':'GPT-6 / Codex; exact deployment subvariant unavailable','reasoning_effort':'high requested'},
        scorer={'source':record(ROOT/'scripts/grm_x3_r2_scorer.py'),'controls':record(OUT/'scorer_controls.json'),
            'unicode_version':unicodedata.unidata_version,'dash_codepoints':list(DASH_POINTS),
            'rule':'NFKC, Dash+legacy Hyphen map to ASCII hyphen, whitespace collapse, casefold; contiguous ordered span with word/hyphen boundaries; whole-answer negation veto except the matched refusal phrase; competing classes veto realization',
            'refusal_phrases':list(REFUSALS),'duplication':'GRM-C5 arm S independent local implementation; no worktree import; lead reconciliation pending'},
        sham_calibration={'prior_run':'r1','reported_mean_nats':.037,'reported_max_nats':.078,'threshold_nats':.10,'required_below':18,
            'statement':'0.10 nats was chosen from r1 as prior-run calibration. Each r2 sham KL >= 0.10 is a FAIL for that snapshot, never a recalibration; P1-prime fails if fewer than 18/20 pass or same-payload is not byte-identical 20/20.'},
        strata_minimum={'correct':6,'decoy':4,'refusal':4})
    reg['lead_predictions']={"P1'":'same-payload raw logits byte-identical 20/20 AND sham KL <0.10 on >=18/20',
        "P2'":'remove KL >1 on >=6 realized correct cases',
        "P3'":'mass high on >=4 realized decoy cases AND lesion accuracy >= mass accuracy on realized decoys (amendment 2 inclusive comparison)'}
    reg['seat_predictions']={"Q1'":reg['lead_predictions']["P1'"],"Q2'":'remove KL >1 on <=5 realized correct cases',
        "Q3'":'lesion accuracy <= mass accuracy on all realized cases (r1 competing witness prediction)'}
    reg['scoring'].update(realization='value-span; frozen r1 exact-equality realization separately labeled',
        metric='Frozen exact-answer error label retained. Paired classifier accuracy on realized strata; all20 descriptive accuracy separately. P3-prime uses >= per amendment; unchanged kill requires strict overall improvement.',
        table='All20 mass/effect four-way table with frozen exact-answer error rates; separately report value-span errors')
    reg['rails'].update(gpu_budget_seconds=1140,campaign_gpu_budget_hours=.64)
    reg['deviations'].append('New relational ordering values to exercise swapped-relation negative controls; single completed fact-turn recipes retain the same capture frame.')
    reg['cpu_gates']=['test_registered_scorer_controls','test_all_registered_dashes','test_r2_registration_rejects_forged_and_stale',
        'test_r1_evidence_byte_unchanged','test_r2_disjoint_recipes_and_dry_run','test_r2_summary_guards_and_floor','test_r2_summary_existing_run_layout']
    reg['prior_art'].extend([{'work':'Unicode Consortium UAX #15 (1998 onward)','url':'https://www.unicode.org/reports/tr15/','used':'NFKC normalization; primary source verified'},
        {'work':'house C5 arm S / X3 amendment 2 (2026)','used':'ordered span rule; local duplication requires lead reconciliation'},
        {'work':'no prior art known to me','used':'conservative negation veto, relation recipe construction and r1-calibrated 0.10 floor; no novelty claim'}])
    create(OUT/'registration.json',reg)
    for f in recipes:create(OUT/'fixtures'/f'{f["id"]}.json',f)
    old_manifest=json.loads((ROOT/'artifacts/grm_x3/fixtures/manifest.json').read_text())
    create(OUT/'fixtures/manifest.json',{'schema':'grm.x3.fixture_manifest.r2.v1','registration':record(OUT/'registration.json'),
        'counts':{'correct':8,'decoy':6,'refusal':6},'fixtures':[record(OUT/'fixtures'/f'{f["id"]}.json') for f in recipes],
        'source_pins':old_manifest['source_pins']})
    # Trust anchor is independent of user-supplied registration/manifest bytes.
    anchor=ROOT/'scripts/grm_x3_r2_anchor.py'
    with anchor.open('x') as stream:
        stream.write('# Prior art: house DET1/X3 (2026), pinned content identity; no new algorithm.\n')
        stream.write(f'REGISTRATION_SHA256 = {sha(OUT/"registration.json")!r}\nMANIFEST_SHA256 = {sha(OUT/"fixtures/manifest.json")!r}\n')
    print(json.dumps({'registration_sha256':sha(OUT/'registration.json'),'manifest_sha256':sha(OUT/'fixtures/manifest.json'),'controls':len(controls)}))

if __name__=='__main__':main()
