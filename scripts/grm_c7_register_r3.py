"""GRM-C7 amendment 6: fresh immutable CPU registration and launch checks.

Prior art: GRM contributors (2026), C7/r2 SHA-bound manifests, C2 cell
reservations, production supersession and RD1 plain prompt contrast. Reuse
those contracts; r3 composes existing mechanisms, no new memory algorithm.
No prior art known to me for this exact composition. Resume validation uses
existing checkpoint tree hashes, not a new recovery algorithm.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil

from scripts.grm_c7_common import ROOT, OUT as BASE, create, read, sha, validate_checkpoint

OUT = BASE/'r3'
REG = OUT/'registration.json'
GATES = OUT/'amendment_6/gates_registration.json'
ORDER = ROOT/'orders/GRM_C7_AMENDMENT_6_R3.md'
MIN_FREE_BYTES = 20_000_000_000
TESTS = ['tests/test_grm_c7_r3.py', 'tests/test_grm_scout_fix3.py',
         'tests/test_grm_scout_fix5.py',
         'tests/test_grm_scout_fix2_capture.py', 'tests/test_grm_fold_recovered_guard.py',
         'tests/test_grm_s4_fold_order.py',
         'tests/test_grm_c7_r2.py::test_oracle_sets_restores_layer_positions_and_records_wrapped_ids',
         'tests/test_grm_c7_amendment3.py']
LEGACY_TESTS = ['tests/test_grm_scout_fix4.py']
PREDICTION = {'fresh':'>= 10/12', 'folded':'>= 10/15',
              'corrections':'>= 6/13', 'aliases':'<= 2/12',
              'residency':'bounded', 'restarts':'retained',
              'aliases_expected_to_fail':True}
DENOMINATOR_NOTE = ('Order predictions retained verbatim. Frozen fixture has 15 probes per class '
                    '(10 answerable + 5 unanswerable controls), 60 total. No subset for '
                    '12/15/13/12 is specified: report all 60 and actual denominators; '
                    'prediction evaluation is unresolved, not a PASS criterion.')


def fixture_bytes():
    # Prior art: production supersede_turn/run_turn (GRM contributors, 2026).
    # Match the shared old source using Vesper-0; replacement preserves BOTH
    # Flint values because retirement is node-wide. No other turn is changed.
    raw = (BASE/'fixture.json').read_bytes()
    f = json.loads(raw)
    assert raw == (json.dumps(f, sort_keys=True, indent=2)+'\n').encode()
    event = f['turns'][1]
    event.update(kind='supersede', old_value='Mica-431',
        correction_command='correct memory: current C7-Vesper-0 value is Mica-431 => '+event['user'])
    return (json.dumps(f, sort_keys=True, indent=2)+'\n').encode()


def prepare_fixture():
    with (OUT/'fixture.json').open('xb') as f:
        f.write(fixture_bytes())
    original = read(BASE/'fixture_manifest.json')
    create(OUT/'fixture_manifest.json', dict(original,
        fixture_sha256=sha(OUT/'fixture.json'),
        parent_fixture_sha256=sha(BASE/'fixture.json'),
        amendment='Only turn 2 receives supersede, old_value and correction_command; correction index list retained verbatim; effective questions separate'))
    from scripts.grm_c7_run import effective_question
    create(OUT/'effective_questions.json', [dict(probe_id=p['id'],
        registered_question=p['question'], question=effective_question(p['question']),
        expected=p['expected']) for p in read(OUT/'fixture.json')['probes']])


def preflight(path=OUT, minimum=MIN_FREE_BYTES):
    # Prior art: C7 lead disk preflight (GRM contributors, 2026), ordinary
    # filesystem free-byte check. New bound is the order's 20 decimal GB.
    free = shutil.disk_usage(path).free
    if free < minimum:
        raise ValueError(f'R3_FREE_SPACE_RED: {free} < {minimum} bytes')
    return {'path':str(path), 'free_bytes':free, 'minimum_bytes':minimum,
            'units':'decimal GB', 'pass':True}


def verify_r3(path=REG):
    if os.environ.get('GRM_C7_FIX4') == '1' or os.environ.get('GRM_ALIAS_FOLLOW', '0') != '0':
        raise ValueError('R3_FORBIDS_OLD_BRIDGE_OR_FIX7')
    if sha(path) != path.with_suffix('.sha256').read_text().split()[0]:
        raise ValueError('R3_REGISTRATION_SHA_MISMATCH')
    r = read(path)
    if r['schema'] != 'grm.c7.r3-registration.v1' or r['budget_seconds_arm_A'] != 7920:
        raise ValueError('R3_PROTOCOL_MISMATCH')
    parent = read(BASE/'registration.json')
    if (r['cells'] != [c for c in parent['cells'] if c['arm']=='A']
            or len(r['cells']) != 39 or set(r['arms']) != {'A'}
            or r['arms']['A']['flags'] != parent['arms']['A']['flags']
            or r['model'] != parent['model']
            or r['fixes'] != {'FIX-3':True,'FIX-4':True,'FIX-5':True,'FIX-7':False}
            or r['prediction'] != PREDICTION
            or r['acceptance'] != read(BASE/'amendment_lead_1.json')['acceptance']):
        raise ValueError('R3_PROTOCOL_MISMATCH')
    for name, digest in r['immutable_inputs'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('R3_INPUT_SHA_MISMATCH: '+name)
    if (sha(OUT/'fixture.json') != r['fixture_sha256']
            or (OUT/'fixture.json').read_bytes() != fixture_bytes()):
        raise ValueError('R3_FIXTURE_MISMATCH')
    from scripts.grm_c7_run import effective_question
    expected = [dict(probe_id=p['id'], registered_question=p['question'],
        question=effective_question(p['question']), expected=p['expected'])
        for p in read(OUT/'fixture.json')['probes']]
    if read(OUT/'effective_questions.json') != expected:
        raise ValueError('R3_PROMPT_MISMATCH')
    r['effective_inputs'] = dict(r['immutable_inputs'])
    return r


def check_ready():
    r = verify_r3()
    ready = OUT/'amendment_6/ready.json'
    if sha(ready) != ready.with_suffix('.sha256').read_text().split()[0]:
        raise ValueError('R3_READY_SHA_MISMATCH')
    receipt = read(ready)
    if receipt['registration_sha256'] != sha(REG) or receipt['status'] != 'GREEN_CPU':
        raise ValueError('R3_READY_BINDING_MISMATCH')
    for name, digest in receipt['receipts'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('R3_GATE_RECEIPT_MISMATCH')
    return r


def resume_check(cell_id, directory=OUT, binding=None):
    from scripts.grm_c7_run import binding as live_binding
    r = verify_r3()
    cell = next(c for c in r['cells'] if c['id']==cell_id)
    target = directory/'cells'/cell_id
    expected_binding = binding if binding is not None else live_binding('A')
    if (directory/'A.active').exists():
        raise ValueError('R3_OWNER_PRESENT: never clear an owner marker')
    if not target.exists():
        return {'cell':cell_id,'status':'UNSTARTED'}
    if not (target/'controller.json').is_file():
        raise ValueError('R3_ORPHAN_STARTED_CELL: never retry')
    controller = read(target/'controller.json')
    if (controller['status'] != 'COMPLETE' or controller['binding'] != expected_binding
            or controller['cell'] != cell):
        raise ValueError('R3_INCOMPLETE_OR_STALE_CELL: never retry')
    worker = read(target/'worker.json')
    if worker['binding'] != expected_binding or worker['cell'] != cell:
        raise ValueError('R3_WORKER_BINDING_MISMATCH')
    cp = target/'checkpoint'
    if sha(cp/'checkpoint.json') != worker['checkpoint_sha256']:
        raise ValueError('R3_CHECKPOINT_SHA_MISMATCH')
    validate_checkpoint(cp, cell['stop']+1, expected_binding)
    return {'cell':cell_id,'status':'COMPLETE'}


def register():
    if os.environ.get('GRM_C7_REVISION') != 'r3':
        raise ValueError('R3_ENVIRONMENT_REQUIRED')
    if REG.exists() or (OUT/'cells').exists():
        raise ValueError('R3_ALREADY_REGISTERED_OR_STARTED')
    r = copy.deepcopy(read(BASE/'registration.json'))
    r.pop('B_execution', None)
    r.pop('B_within_combined_7200_status', None)
    r['cells'] = [c for c in r['cells'] if c['arm']=='A']
    r['arms'] = {'A':r['arms']['A']}
    r['acceptance'] = read(BASE/'amendment_lead_1.json')['acceptance']
    # Pin current executable tree plus original external runtime/config pins.
    # Historical source hashes are archived inputs, never silently overwritten.
    paths = {str(p.relative_to(ROOT)) for d in ('core','scripts','config','tests')
             for p in (ROOT/d).rglob('*') if p.is_file() and p.suffix in ('.py','.json')}
    paths.update(n for n in r['immutable_inputs'] if Path(n).is_absolute())
    paths.update(t.split('::')[0] for t in TESTS + LEGACY_TESTS)
    paths.update(str(p.relative_to(ROOT)) for p in (OUT/'amendment_6/before').rglob('*') if p.is_file())
    paths.update(['artifacts/grm_c7/registration.json','artifacts/grm_c7/registration.sha256',
        'artifacts/grm_c7/fixture.json','artifacts/grm_c7/fixture_manifest.json',
        'artifacts/grm_c7/amendment_lead_1.json','artifacts/grm_c7/r2/amendment.json',
        'artifacts/grm_c7/r2/fix4_continuation/continuation_registration.json',
        'artifacts/grm_c7/r3/stop_receipt.json','artifacts/grm_scout_fix5/registration.json',
        'artifacts/grm_scout_fix5/gpu/summary.json',
        str(ORDER.relative_to(ROOT)), str(GATES.relative_to(ROOT)),
        'artifacts/grm_c7/r3/fixture.json','artifacts/grm_c7/r3/fixture_manifest.json',
        'artifacts/grm_c7/r3/effective_questions.json','artifacts/grm_c7/r3/lead_commands_r3.txt',
        'artifacts/grm_c7/r3/ALIAS_DESIGN_OPTIONS.md',
        'artifacts/grm_c7/r3/amendment_6/test_context_amendment.json'])
    inputs = {n:sha(ROOT/n) for n in sorted(paths)}
    create(OUT/'source_manifest.json', inputs)
    r.update(schema='grm.c7.r3-registration.v1', immutable=True,
        order=str(ORDER.relative_to(ROOT)), parent_registration_sha256=sha(BASE/'registration.json'),
        gates_registration_sha256=sha(GATES), fixture_sha256=sha(OUT/'fixture.json'),
        source_manifest_sha256=sha(OUT/'source_manifest.json'),
        immutable_inputs=dict(inputs, **{'artifacts/grm_c7/r3/source_manifest.json':sha(OUT/'source_manifest.json')}),
        fixes={'FIX-3':True,'FIX-4':True,'FIX-5':True,'FIX-7':False},
        fix_enablement='FIX3 Harmony fold, FIX4 recency binder, FIX5 enumeration/budget are in the pinned core; no environment flag required. GRM_C7_FIX4 is the OLD r2 checkpoint bridge and is forbidden.',
        budget_seconds_arm_A=7920, budget_gpu_hours=2.2, receipt_directory='artifacts/grm_c7/r3',
        prediction=PREDICTION, denominator_note=DENOMINATOR_NOTE,
        tests=TESTS, legacy_prompt_tests=LEGACY_TESTS, free_space_minimum_bytes=MIN_FREE_BYTES,
        cpu_gates=read(GATES)['gates'],
        mutations={'ids':['case_sensitive_control','retain_abstention_clause','wrong_budget_7200',
                          'drop_child_revision','weaken_space_preflight'],
                   'threshold':.80,'only_after_passing_baseline':True,'scope':'author CPU source copies, no blind review'},
        stop_rule='No retries of started/RED/orphan cells; fresh same-binding checkpoints only. Reserve each 280s cell against 7920s cap. 285s lease, 240s foreground lease wait bound, 590s lead outer timeout, 30s foreground cooldown. Stop on failure, changed input, timeout, missing durability or <20 GB.',
        cost_note='r2 39-cell timing proxy retained, not a fresh fit proof. FIX5 max(120,24*N) budget and turn2 correction can increase runtime; rails unchanged, stop rather than retune.',
        prediction_is_acceptance=False,
        prompt_protocol='Plain effective_questions used identically for memory, oracle and restart sentinels; immutable fixture probe bytes retained.',
        probe_protocol='All 60 probes, arm A only. No probe Q&A deposit, isolated exact-source oracle. Alias failures retained in the original acceptance calculation.',
        scorer_protocol='Normalized full answer equality; UNKNOWN unanswerable control comparison casefolds only. Answerable values remain case-sensitive. No synonym or substring credit.',
        agent_model='gpt-6-astra', agent_effort='high', agent_identity_receipt='logs/grm_c7_a6.log:6,10',
        prior_art=__doc__)
    create(REG, r)
    with REG.with_suffix('.sha256').open('x') as f:
        f.write(sha(REG)+'  registration.json\n')
    print(json.dumps({'registration':str(REG),'sha256':sha(REG)}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare-fixture',action='store_true')
    p.add_argument('--register',action='store_true')
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--ready',action='store_true')
    p.add_argument('--resume-check')
    a=p.parse_args()
    if a.prepare_fixture: prepare_fixture()
    elif a.register: register()
    elif a.preflight: print(json.dumps(preflight()))
    elif a.ready: print(json.dumps({'status':'GREEN_CPU','registration_sha256':sha(REG),'cells':len(check_ready()['cells'])}))
    elif a.resume_check: print(json.dumps(resume_check(a.resume_check)))
    else: print(json.dumps({'registration_sha256':sha(REG),'cells':len(verify_r3()['cells'])}))


if __name__ == '__main__':
    main()
