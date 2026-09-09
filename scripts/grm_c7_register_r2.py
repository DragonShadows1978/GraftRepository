"""Immutable FIX-3/C7 r2 amendment, CPU only.

Prior art: C7 A1/A2/lead-1 SHA-bound amendments and C2 receipt isolation
(GRM contributors, 2026), local source verified. Taken: hash chain, explicit
overrides, before archives, inherited cells/thresholds and fail-closed checks.
New: r2 core/harness binding and effective query manifest. No prior art known
to me for this exact composition; no new cryptographic or memory algorithm.
"""
import os
import sys
from pathlib import Path

from scripts.grm_c7_common import ROOT, OUT, REG, FIX, create, read, sha

ORDER = 'orders/GRM_SCOUT_FIX_3_C7_R2.md'
CHANGED = {'core/graft_arena.py', 'scripts/grm_c7_common.py',
           'scripts/grm_c7_run.py', 'tests/test_grm_c7.py'}
NEW = {ORDER, 'scripts/grm_c7_register_r2.py', 'tests/test_grm_scout_fix3.py',
       'tests/test_grm_c7_r2.py', 'tests/test_grm_c7_r2_registration.py',
       'artifacts/grm_c7/r2/execution_registration.json',
       'artifacts/grm_c7/r2/effective_questions.json',
       'artifacts/grm_c7/r2/lead_commands_r2.txt',
       'scripts/grm_c7_r2_cpu.py'}
LAUNCH_CHANGED = {'scripts/grm_c7_run.py', 'scripts/grm_c7_register_r2.py',
                  'scripts/grm_c7_r2_cpu.py', 'tests/test_grm_c7_r2_registration.py',
                  'artifacts/grm_c7/r2/lead_commands_r2.txt'}
LAUNCH_NEW = {'tests/test_grm_c7_r2_launch.py',
              'artifacts/grm_c7/r2/launch_amendment/registration.json'}


def verify_r2(r, inputs, root=ROOT, base=OUT):
    directory = base / 'r2'
    path = directory / 'amendment.json'
    checksum = path.with_suffix('.sha256')
    if not path.is_file() or not checksum.is_file():
        raise ValueError('R2_AMENDMENT_REQUIRED')
    if sha(path) != checksum.read_text().split()[0]:
        raise ValueError('R2_SHA_MISMATCH')
    a = read(path)
    if (a['schema'] != 'grm.c7.r2-amendment.v1'
            or a['registration_sha256'] != sha(base/'registration.json')
            or a['previous_amendment_sha256'] != r['amendment_lead_1_sha256']):
        raise ValueError('R2_CHAIN_MISMATCH')
    if set(a['overrides']) != CHANGED or set(a['new_inputs']) != NEW:
        raise ValueError('R2_SCOPE_MISMATCH')
    if (a['order'] != {'path': ORDER, 'sha256': sha(root/ORDER)}
            or a['acceptance'] != r['acceptance'] or a['cells'] != r['cells']
            or a['arms'] != r['arms'] or a['model'] != r['model']
            or a['fixture_sha256'] != sha(base/'fixture.json')
            or a['fixture_manifest_sha256'] != sha(base/'fixture_manifest.json')
            or a['budget_seconds_arm_A'] != 7200 or a['total_C7_gpu_hours'] != 3.9
            or a['digest_ngen'] != 120 or a['answer_ngen'] != 32
            or a['receipt_directory'] != 'artifacts/grm_c7/r2'):
        raise ValueError('R2_PROTOCOL_MISMATCH')
    for name, change in a['overrides'].items():
        if (change['before_sha256'] != inputs[name]
                or change['before_archive'] != 'artifacts/grm_c7/r2/before/' + name
                or sha(root/change['before_archive']) != inputs[name]):
            raise ValueError('R2_BEFORE_MISMATCH')
        inputs[name] = change['after_sha256']
    inputs.update(a['new_inputs'])
    if a['core_sha256'] != {name: inputs[name] for name in
                           ('core/graft_arena.py', 'core/graft_repository.py')}:
        raise ValueError('R2_CORE_MISMATCH')
    from scripts.grm_c7_run import effective_question
    expected = [{'probe_id': p['id'], 'registered_question': p['question'],
                 'question': effective_question(p['question']), 'expected': p['expected']}
                for p in read(base/'fixture.json')['probes']]
    if read(directory/'effective_questions.json') != expected:
        raise ValueError('R2_QUESTION_MISMATCH')
    r['amendment_r2_sha256'] = sha(path)
    r['fold_core_sha256'] = a['core_sha256']['core/graft_arena.py']
    # Prior art: C7 chained amendments (GRM, 2026). Preserve the first r2
    # registration and bind a separately receipted launch-path correction.
    launch = directory/'launch_amendment/amendment.json'
    if launch.exists():
        verify_launch(r, inputs, launch, root)


def verify_launch(r, inputs, path, root=ROOT):
    if sha(path) != path.with_suffix('.sha256').read_text().split()[0]:
        raise ValueError('R2_LAUNCH_SHA_MISMATCH')
    a = read(path)
    if (a['schema'] != 'grm.c7.r2-launch-amendment.v1'
            or a['previous_amendment_sha256'] != r['amendment_r2_sha256']):
        raise ValueError('R2_LAUNCH_CHAIN_MISMATCH')
    if set(a['overrides']) != LAUNCH_CHANGED or set(a['new_inputs']) != LAUNCH_NEW:
        raise ValueError('R2_LAUNCH_SCOPE_MISMATCH')
    for name, change in a['overrides'].items():
        expected_archive = 'artifacts/grm_c7/r2/launch_amendment/before/' + name
        if (change['before_sha256'] != inputs[name]
                or change['before_archive'] != expected_archive
                or sha(root/expected_archive) != inputs[name]):
            raise ValueError('R2_LAUNCH_BEFORE_MISMATCH')
        inputs[name] = change['after_sha256']
    inputs.update(a['new_inputs'])
    r['amendment_r2_base_sha256'] = r['amendment_r2_sha256']
    r['amendment_r2_sha256'] = sha(path)


def register_launch():
    directory = OUT/'r2/launch_amendment'
    base = OUT/'r2/amendment.json'
    a = read(base)
    inputs = {name:change['after_sha256'] for name,change in a['overrides'].items()}
    inputs.update(a['new_inputs'])
    overrides = {}
    for name in sorted(LAUNCH_CHANGED):
        before = 'artifacts/grm_c7/r2/launch_amendment/before/' + name
        if sha(ROOT/before) != inputs[name]:
            raise ValueError('R2_LAUNCH_BEFORE_MISMATCH: '+name)
        overrides[name] = {'before_archive':before, 'before_sha256':inputs[name],
                           'after_sha256':sha(ROOT/name)}
    path = directory/'amendment.json'
    create(path, {'schema':'grm.c7.r2-launch-amendment.v1', 'immutable':True,
        'previous_amendment_sha256':sha(base), 'overrides':overrides,
        'new_inputs':{name:sha(ROOT/name) for name in sorted(LAUNCH_NEW)},
        'finding':'C2 environment filtering drops GRM_C7_REVISION; explicitly restore it for the owned worker',
        'registered_protocol':'Base r2 fixtures, cells, core SHA, budgets and thresholds unchanged',
        'gates':read(directory/'registration.json'),
        'prior_art':'C7 lease-parent propagation and C7 amendment chaining (GRM contributors, 2026); reuse explicit forwarding and SHA overrides, no new algorithm'})
    with path.with_suffix('.sha256').open('x') as f:
        f.write(sha(path)+'  amendment.json\n')
    print(sha(path))


def register():
    directory = OUT/'r2'
    if (directory/'amendment.json').exists() or (directory/'cells').exists():
        raise ValueError('R2_ALREADY_REGISTERED_OR_STARTED')
    r = read(REG)
    inputs = dict(r['immutable_inputs'])
    for name in ('amendment_A1', 'amendment_A2', 'amendment_lead_1'):
        path = OUT/(name+'.json')
        if sha(path) != path.with_suffix('.sha256').read_text().split()[0]:
            raise ValueError('PREDECESSOR_SHA_MISMATCH')
        a = read(path)
        for key, change in a['overrides'].items():
            if (inputs[key] != change['before_sha256']
                    or sha(ROOT/change['before_archive']) != inputs[key]):
                raise ValueError('PREDECESSOR_BEFORE_MISMATCH')
            inputs[key] = change['after_sha256']
        inputs.update(a['new_inputs'])
    r['acceptance'] = a['acceptance']
    overrides = {}
    for name in sorted(CHANGED):
        before = 'artifacts/grm_c7/r2/before/' + name
        if sha(ROOT/before) != inputs[name]:
            raise ValueError('R2_BEFORE_MISMATCH: '+name)
        overrides[name] = {'before_sha256': inputs[name], 'before_archive': before,
                           'after_sha256': sha(ROOT/name)}
    create(directory/'amendment.json', {
        'schema': 'grm.c7.r2-amendment.v1', 'immutable': True,
        'order': {'path': ORDER, 'sha256': sha(ROOT/ORDER)},
        'registration_sha256': sha(REG),
        'previous_amendment_sha256': sha(OUT/'amendment_lead_1.json'),
        'diagnosis_report_sha256': sha(OUT/'diagnosis_lead_2/REPORT.md'),
        'overrides': overrides, 'new_inputs': {p: sha(ROOT/p) for p in sorted(NEW)},
        'core_sha256': {p: sha(ROOT/p) for p in ('core/graft_arena.py','core/graft_repository.py')},
        'fixture_sha256': sha(FIX), 'fixture_manifest_sha256': sha(OUT/'fixture_manifest.json'),
        'acceptance': r['acceptance'], 'cells': r['cells'], 'arms': r['arms'], 'model': r['model'],
        'budget_seconds_arm_A': 7200, 'total_C7_gpu_hours': 3.9,
        'receipt_directory': 'artifacts/grm_c7/r2', 'digest_ngen': 120, 'answer_ngen': 32,
        'methodology_budget': 'docs/GRM_Methodology.md gives no alternate digest generation budget; 120 retained',
        'question_amendment': 'Only final fallback instruction UNKNOWN -> unknown at execution for memory, sentinels and oracle. Sources, expected answers, source turns, classes and distances remain identical. Original/effective question bytes are explicit in effective_questions.json.',
        'oracle_amendment': 'Same live-source oracle and exact scorer; establish/restore layer live_shift and record actual wrapped token IDs. GPT-OSS UNKNOWN cause remains RED unresolved; measure again in r2.',
        'gates': read(directory/'execution_registration.json')['gates_before_edits'],
        'cpu_command': "GRM_C7_REVISION=r2 CUDA_VISIBLE_DEVICES='' python -m scripts.grm_c7_r2_cpu",
        'stop_rule': 'Original per-cell 280s worker/285s lease/590s outer/30s foreground cooldown. No retry of started/RED/NON_FIT cells. r2 fresh 7200s cap, r1 consumed 1.9 GPU-h per lead. B remains NON_FIT.',
        'agent_model': 'GPT-6 (system identity; exact deployment model ID not exposed)',
        'agent_effort': 'high requested by order; actual deployment setting not independently exposed',
        'prior_art': __doc__,
    })
    path = directory/'amendment.json'
    with path.with_suffix('.sha256').open('x') as f:
        f.write(sha(path)+'  amendment.json\n')
    print(sha(path))


if __name__ == '__main__':
    if sys.argv[1:] == ['--launch-amendment']:
        register_launch()
    else:
        register()
