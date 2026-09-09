"""FIX-4 continuation, CPU registration and fail-closed checkpoint bridge.

Prior art: GRM C7 SHA-bound amendments, checkpoint tree hashes, pessimistic
reservation accounting (GRM contributors, 2026), verified local source.
Reuse those contracts; add an explicit old-boundary compatibility exception
and attempt namespace. No prior art known to me for this exact composition.
"""
import os
from pathlib import Path

from scripts.grm_c7_common import ROOT, OUT, read, sha, validate_checkpoint

DIRECTORY = OUT/'r2/fix4_continuation'
REGISTRATION = DIRECTORY/'continuation_registration.json'
ATTEMPT = OUT/'r2/fix4_attempt_1'
OLD_CELLS = ('A-001-008', 'A-009-016', 'A-017-023')
CHANGED = {'core/graft_arena.py', 'core/grm_admission.py', 'core/grm_three_pass.py',
           'scripts/grm_e2e_session.py', 'scripts/grm_c7_common.py',
           'scripts/grm_c7_run.py', 'tests/test_grm_c7_r2_registration.py', 'tests/test_grm_c7.py'}
NEW = {'scripts/grm_c7_fix4.py', 'tests/test_grm_scout_fix4.py',
       'tests/test_grm_scout_fix4_continuation.py', 'tests/test_grm_c7_amendment3.py',
       'scripts/grm_c7_diagnose.py', 'orders/GRM_SCOUT_FIX_4_RECENCY_BINDER.md',
       'artifacts/grm_c7/r2/lead_commands_r2_resume.txt',
       'artifacts/grm_scout_fix4/registration.json',
       'artifacts/grm_scout_fix4/nonrecency_before.json'}


def enabled():
    return os.environ.get('GRM_C7_FIX4') == '1'


def verify_fix4(r, inputs, path=REGISTRATION, root=ROOT):
    if os.environ.get('GRM_C7_REVISION') != 'r2':
        raise ValueError('FIX4_REQUIRES_R2')
    if not path.is_file() or sha(path) != path.with_suffix('.sha256').read_text().split()[0]:
        raise ValueError('FIX4_REGISTRATION_SHA_MISMATCH')
    a = read(path)
    if (a['schema'] != 'grm.c7.fix4-continuation.v1' or a['status'] != 'REARMED'
            or a['previous_amendment_sha256'] != r['amendment_r2_sha256']
            or a['blocked_predecessor_sha256'] != sha(OUT/'r2/amendment_3/continuation_registration.json')):
        raise ValueError('FIX4_CHAIN_MISMATCH')
    if set(a['overrides']) != CHANGED or set(a['new_inputs']) != NEW:
        raise ValueError('FIX4_SCOPE_MISMATCH')
    if (a['cells'] != r['cells'] or a['acceptance'] != r['acceptance']
            or a['arms'] != r['arms'] or a['model'] != r['model']
            or a['budget_seconds_arm_A'] != 7200 or a['resume_cell'] != 'A-024-031'
            or a['attempt_directory'] != str(ATTEMPT.relative_to(ROOT))
            or a['quarantine_memory_probe_ids'] != ['c7_alias_0_d005', 'c7_alias_1_d005']
            or a['quarantine_oracle_cells'] != list(OLD_CELLS)):
        raise ValueError('FIX4_PROTOCOL_MISMATCH')
    for name, change in a['overrides'].items():
        expected = str((DIRECTORY/'before'/name).relative_to(ROOT))
        if (change['before_sha256'] != inputs[name] or change['before_archive'] != expected
                or sha(root/expected) != inputs[name]):
            raise ValueError('FIX4_BEFORE_MISMATCH: '+name)
        inputs[name] = change['after_sha256']
    inputs.update(a['new_inputs'])
    # Bind all historical receipt bytes, including the failed cell's charge;
    # never silently reclassify an old attempt or import a different boundary.
    for name, digest in a['historical_files'].items():
        if sha(root/name) != digest:
            raise ValueError('FIX4_HISTORY_MISMATCH: '+name)
    old = OUT/'r2/cells'
    actual_charge = sum(read(p)['charged_seconds'] for p in old.glob('A-*/controller.json'))
    actual_reserved = sum(read(p)['seconds'] for p in old.glob('A-*/reservation.json')
                          if not p.with_name('controller.json').exists())
    if actual_charge != a['historical_charged_seconds'] or actual_reserved != a['historical_reserved_seconds']:
        raise ValueError('FIX4_HISTORICAL_CHARGE_MISMATCH')
    r['amendment_fix4_base_sha256'] = sha(path)
    # Prior art: C7 explicit amendment chain (GRM contributors, 2026).
    # Preserve the initial registration; amendment1 binds the exact charge
    # test correction, additional receipt pins and this verification code.
    amendment = DIRECTORY/'amendment_1/amendment.json'
    if sha(amendment) != amendment.with_suffix('.sha256').read_text().split()[0]:
        raise ValueError('FIX4_A1_SHA_MISMATCH')
    patch = read(amendment)
    if patch['previous_sha256'] != sha(REGISTRATION):
        raise ValueError('FIX4_A1_CHAIN_MISMATCH')
    if (set(patch['overrides']) != {'scripts/grm_c7_fix4.py',
                                  'tests/test_grm_scout_fix4_continuation.py'}
            or set(patch['new_inputs']) != {'tests/test_grm_scout_fix4_boundaries.py'}):
        raise ValueError('FIX4_A1_SCOPE_MISMATCH')
    for name, change in patch['overrides'].items():
        expected = str((DIRECTORY/'amendment_1/before'/name).relative_to(ROOT))
        if (change['before_sha256'] != inputs[name]
                or change['before_archive'] != expected
                or sha(root/expected) != inputs[name]):
            raise ValueError('FIX4_A1_BEFORE_MISMATCH')
        inputs[name] = change['after_sha256']
    inputs.update(patch['new_inputs'])
    r['amendment_fix4_sha256'] = sha(amendment)
    r['fold_core_sha256'] = inputs['core/graft_arena.py']
    r['fix4'] = a


def cell_directory(output, cell_id):
    if enabled() and output == ATTEMPT and cell_id in OLD_CELLS:
        return OUT/'r2/cells'/cell_id
    return output/'cells'/cell_id


def accounting_directories(output):
    return [OUT/'r2/cells', output/'cells'] if enabled() and output == ATTEMPT else [output/'cells']


def resume_state(cp, next_turn, binding, registration):
    """Only the registered turn24 boundary may cross the old source binding."""
    if enabled() and cp == OUT/'r2/cells/A-017-023/checkpoint':
        a = registration['fix4']
        if (next_turn != 24 or sha(cp/'checkpoint.json') != a['checkpoint_sha256']
                or read(cp/'checkpoint.json')['binding'] != a['checkpoint_binding']):
            raise ValueError('FIX4_CHECKPOINT_COMPATIBILITY_MISMATCH')
        return validate_checkpoint(cp, 24, a['checkpoint_binding'])
    return validate_checkpoint(cp, next_turn, binding)


def quarantine_summary(value):
    # Raw counts remain evidence of the old policy; they cannot certify the
    # new one. No scorer normalization, missing-row substitution or PASS.
    value['raw_mixed_policy_by_distance'] = value.pop('by_distance')
    value['raw_mixed_policy_by_distance_and_class'] = value.pop('by_distance_and_class')
    value['status'] = 'NOT_MEASURED'
    value['oracle_relative_acceptance'] = 'NOT_MEASURED'
    value['quarantine'] = {
        'memory_probe_ids': ['c7_alias_0_d005', 'c7_alias_1_d005'],
        'oracle_cells': list(OLD_CELLS),
        'reason': 'Pre-FIX4 alias refusals and amendment3 oracle quarantine; raw rows untouched.'}
    return value


def check_ready():
    from scripts.grm_c7_common import verify
    r = verify()
    ready = DIRECTORY/'ready.json'
    if sha(ready) != ready.with_suffix('.sha256').read_text().split()[0]:
        raise ValueError('FIX4_READY_SHA_MISMATCH')
    seal = read(ready)
    if seal['registration_sha256'] != r['amendment_fix4_sha256'] or seal['status'] != 'GREEN_CPU':
        raise ValueError('FIX4_READY_BINDING_MISMATCH')
    for name, digest in seal['receipts'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('FIX4_GATE_RECEIPT_MISMATCH')
    return r


if __name__ == '__main__':
    import json
    r = check_ready()
    print(json.dumps({'status': 'REARMED', 'resume_cell': r['fix4']['resume_cell'],
                      'historical_charged_seconds': r['fix4']['historical_charged_seconds'],
                      'gpu_executed': False}, indent=2))
