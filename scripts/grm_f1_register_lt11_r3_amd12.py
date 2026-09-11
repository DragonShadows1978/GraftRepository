#!/usr/bin/env python3
"""GRM-F1 follow-up 3: amendment 12 -- rebind the LT1.1 RUNNER for `--out`.

WHY THIS AMENDMENT EXISTS.  The lead's arm-A' run died before taking any
lease (`artifacts/grm_f1/lt1_1_r3/lead_Aprime_try1.log`):

    scripts/grm_lt1_1.py:1206  resume -> worker.pending(reg, run=root)
    scripts/grm_lt1_worker.py:240  ValueError: COMPLETED_CELL_BINDING_OR_STATUS

Settled by receipt (`artifacts/grm_f1/red/`), the cause was NEITHER candidate
the lead offered.  `main()` read `return resume(args.arm)` -- `--out` was
accepted by argparse and DROPPED on the one route that spends GPU, so the
campaign resumed into r2's FROZEN `artifacts/grm_d1/lt1_1/run_A`, whose 26
COMPLETE cells carry r2's binding, and `pending()` raised on the FIRST cell.

Fixing that required three runner changes, so the runner sha moves and the
sha-bound preflight must be rebound.  Amendment 12 chains to 11 and carries
amendment 11's core_rebind forward VERBATIM: no core file moves here, and
this script RAISES if one did.

THE THREE RUNNER CHANGES (all in `scripts/grm_lt1_1.py`; core untouched):

  1. `main()` forwards `--out` to the real `resume`, and `--out`'s help now
     states that it governs every root.
  2. `lt1_1_seams(arm, root=None)` pins `worker.RUN` to the campaign root
     rather than `out_dir(arm)`.  `run_cell`, `accounting` and the leased
     child all read `worker.RUN`, so without this a campaign could SELECT a
     cell from one root and EXECUTE it into another.  `resolved_roots()` and
     `assert_root_isolation()` make that unrepresentable and print the
     resolved roots into both the receipt and the operator log.
  3. `spawn_env` carries every r3 TREATMENT FLAG across the
     `environment()` strip into the leased child.  Measured before the fix
     (`artifacts/grm_f1/red/red_child_env_strip.json`): F1, F2 and F5 were
     ALL absent from the child environment, so a treatment arm would have
     EXECUTED AS THE CONTROL while its receipts recorded the flags as on.

PRIOR ART.
  * `scripts/grm_f1_register_lt11_r3_merged.py` (this seat, 2026-09-11) --
    the amendment/document structure and the measured-not-typed sha
    discipline, reused.
  * `artifacts/grm_d1/lt1_1/amendment9.json` (GRM contributors, 2026) --
    the `runner` block shape, reused field for field, including
    `superseded_sha256`.
  * Mine: the root-isolation contract and the treatment-flag carry list.
  * No prior art known to me for this exact composition.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

R2 = Path('artifacts/grm_d1/lt1_1')
OUT_REL = Path('artifacts/grm_f1/lt1_1_r3')
OUT = ROOT / OUT_REL

RUNNER_CHANGE = (
    'GRM-F1 follow-up 3, THREE changes, all confined to the runner; core is '
    'untouched. (1) main() forwards --out to the real resume(): it read '
    '`return resume(args.arm)` and DROPPED the flag on the only route that '
    'spends GPU, which is why the lead\'s arm-A\' run resumed into r2\'s '
    'frozen run root and died in pending() with '
    'COMPLETED_CELL_BINDING_OR_STATUS on the FIRST cell. (2) '
    'lt1_1_seams(arm, root) pins worker.RUN -- read by run_cell, accounting '
    'and the leased child -- to the campaign root instead of out_dir(arm), '
    'with resolved_roots()/assert_root_isolation() refusing a root that is, '
    'or sits inside, the arm default and printing the resolved roots into '
    'the receipt and the operator log. (3) spawn_env carries '
    'GRM_FOLD_RETAIN_SOURCES / GRM_FOLD_ALIAS_GUARD / '
    'GRM_ROUTE_SOLE_BINDER_INSURANCE across the environment() strip into the '
    'leased child; measured before the fix, all three were ABSENT there, so '
    'a treatment arm would have executed as the control while its receipts '
    'claimed otherwise. Each flag is carried only when actually set, so a '
    'caller with none set gets a byte-identical child environment.')

#: Markers that must be present for the rebind to claim change (3) honestly.
RUNNER_MARKERS = (
    'return resume(args.arm, root=args.out,',
    'def lt1_1_seams(arm, root=None):',
    'def assert_root_isolation(',
    'def resolved_roots(',
    'TREATMENT_FLAGS = (',
)


def sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha_file(rel):
    return sha_bytes((ROOT / rel).read_bytes())


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, indent=2) + '\n').encode()


def amd_doc(name):
    return json.loads((ROOT / R2 / name).read_text())


def core_rebind():
    """Amendment 11's block, carried forward and RE-VERIFIED, unchanged.

    Follow-up 3 touches no core file. Every pin is re-measured against the
    tree; if one moved, something happened outside this order's scope and
    this raises rather than rebinding it silently.
    """
    prior = amd_doc('amendment11.json')['core_rebind']
    block = json.loads(json.dumps(prior))
    checked = 0
    for section, key in (('changed', 'after_sha256'),
                         ('new_inputs', 'sha256'),
                         ('unchanged', 'sha256')):
        for name, entry in sorted(block.get(section, {}).items()):
            now = sha_file(name)
            if now != entry[key]:
                raise ValueError(
                    'F1_UNEXPECTED_CORE_DRIFT: %s (amendment 11 pinned %s, '
                    'on tree %s) -- GRM-F1 follow-up 3 edits no core file. '
                    'STOP and report.' % (name, entry[key], now))
            checked += 1
    block['supersedes'] = 'amendment11.core_rebind'
    block['source_binding'] = str(R2 / 'amendment11.json')
    block['source_binding_sha256'] = sha_file(R2 / 'amendment11.json')
    block['attribution_note'] = (
        'CARRIED FORWARD VERBATIM from amendment 11. GRM-F1 follow-up 3 is a '
        'RUNNER change only; all %d core pins were re-measured against the '
        'tree at registration time and every one matches. The script raises '
        'F1_UNEXPECTED_CORE_DRIFT rather than rebinding a core file this '
        'order had no business touching.' % checked)
    return block


def runner_binding():
    """Amendment 12's runner block: the rebind this amendment exists for."""
    prior = dict(amd_doc('amendment11.json')['runner'])

    worker_now = sha_file(prior['worker'])
    if worker_now != prior['worker_sha256']:
        raise ValueError(
            'F1_WORKER_DRIFT: %s (amendment 11 pinned %s, on tree %s) -- '
            'follow-up 3 edits the RUNNER, never the worker. STOP and report.'
            % (prior['worker'], prior['worker_sha256'], worker_now))

    runner_path = prior['path']
    text = (ROOT / runner_path).read_text()
    missing = [m for m in RUNNER_MARKERS if m not in text]
    if missing:
        raise ValueError(
            'F1_RUNNER_CHANGE_ABSENT: %s lacks %r -- the rebind would claim '
            'a change the file does not carry.' % (runner_path, missing))
    if 'return resume(args.arm)\n' in text:
        raise ValueError(
            'F1_RUNNER_DEFECT_PRESENT: %s still contains the argument-'
            'dropping call this amendment exists to retire.' % runner_path)
    runner_now = sha_file(runner_path)
    if runner_now == prior['sha256']:
        raise ValueError('F1_RUNNER_DID_NOT_MOVE: %s' % runner_path)

    prior['carried_from'] = 'amendment11.json'
    prior['superseded_sha256'] = prior['sha256']
    prior['sha256'] = runner_now
    prior['change'] = RUNNER_CHANGE
    prior['worker_change'] = (
        'UNCHANGED by follow-up 3; amendment 11\'s worker sha re-measured '
        'against the tree and carried verbatim.')
    prior['tests_added_by_follow_up_3'] = [
        'tests/test_grm_f1_resume_out_root.py']
    prior['red_receipts'] = [
        'artifacts/grm_f1/red/red_resume_drops_out.log',
        'artifacts/grm_f1/red/red_child_env_strip.json',
    ]
    return prior


def amendment12():
    return dict(
        amendment=12,
        arms=amd_doc('amendment11.json')['arms'],
        previous_amendment='amendment11.json',
        previous_amendment_sha256=sha_file(R2 / 'amendment11.json'),
        order='orders/GRM_F1_FOLD_RETAIN_SOURCES.md',
        order_sha256=sha_file('orders/GRM_F1_FOLD_RETAIN_SOURCES.md'),
        purpose=(
            'Rebind the LT1.1 RUNNER after GRM-F1 follow-up 3 made --out '
            'govern every root the runner and the worker consult on the '
            'resume route, and made the r3 treatment flags cross into the '
            'leased child. No core file moves; amendment 11\'s core_rebind '
            'is carried forward verbatim and re-verified.'),
        follow_up_instruction=(
            'Amendment 12 supersedes amendment 11\'s core_rebind and runner '
            'under the chain rule grm_lt1_1.governing_core_rebind '
            'implements. The THREE-ARM registration (A0 / A\' / A+\') from '
            'amendment 11 stands unchanged; only the runner sha moves.'),
        root_isolation_contract=dict(
            statement=(
                'With --out given, the arm default root is never read. '
                'resolved_roots() reports campaign_root, cells_root, '
                'owner_file, worker_RUN and arm_default_root; '
                'assert_root_isolation() raises LT11_OUT_EQUALS_ARM_DEFAULT, '
                'LT11_OUT_INSIDE_ARM_DEFAULT or '
                'LT11_WORKER_RUN_NOT_ISOLATED rather than proceeding.'),
            printed_in=['the --dry-lease JSON receipt (resolved_roots)',
                        'the operator log line "LT1.1 roots {...}"'],
            proof='tests/test_grm_f1_resume_out_root.py'),
        treatment_flag_carry=dict(
            flags=['GRM_FOLD_RETAIN_SOURCES', 'GRM_FOLD_ALIAS_GUARD',
                   'GRM_ROUTE_SOLE_BINDER_INSURANCE'],
            statement=(
                'grm_c2_cells.environment() strips every ambient GRM_* name; '
                'arm_environment (in-process) and spawn_env (into the leased '
                'child) re-apply these three when, and only when, they are '
                'actually set.'),
            measured_before_fix='all three ABSENT from the child environment',
            receipt='artifacts/grm_f1/red/red_child_env_strip.json',
            proof='tests/test_grm_f1_resume_out_root.py'),
        f2_pin_relocation=dict(
            pin='r2_arm_a_source_store',
            moved_from=('artifacts/grm_d1/lt1_1/run_A/cells/A-025-032/'
                        'session/repository/manifest.json'),
            moved_to=('artifacts/grm_f2/'
                      'r2_arm_a_A-025-032_session_manifest.json'),
            why=('a contract pin must not require a file inside a LIVE '
                 'campaign run root; the copy is byte-identical and the '
                 'stray restored session/ directory has been removed from '
                 'the run root'),
            run_root_state='artifacts/grm_d1/lt1_1/run_A holds no session/ '
                           'directories; the 26 r2 cells are otherwise '
                           'exactly as the campaign left them'),
        runner=runner_binding(),
        core_rebind=core_rebind(),
        process_safety=dict(
            gpu=('single lease via flock --wait; the operator has absolute '
                 'right of way; no process this run did not start is ever '
                 'signalled'),
            display='no windowed gate; CPU gates only',
            git='the lead commits; the seat never runs git',
            core_writes='none -- follow-up 3 edits the runner only',
            scratch=('pytest uses --basetemp artifacts/grm_f1/tmp and the '
                     'directory is removed afterwards'),
            subagents='none spawned'),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    amd = amendment12()
    payload = canonical(amd)
    digest = sha_bytes(payload)

    chain = ROOT / R2 / 'amendment12.json'
    if chain.exists():
        existing = json.loads(chain.read_text())
        if existing.get('order') != amd['order']:
            raise ValueError(
                'F1_AMENDMENT12_FOREIGN: %s was written by order %r, not by '
                'this one (%r) -- refusing to overwrite another order\'s '
                'chain receipt. STOP and report.'
                % (chain, existing.get('order'), amd['order']))
    chain.write_bytes(payload)
    (ROOT / R2 / 'amendment12.sha256').write_text(
        '%s  amendment12.json\n' % digest)
    (OUT / 'amendment12.json').write_bytes(payload)
    (OUT / 'amendment12.sha256').write_text(
        '%s  amendment12.json\n' % digest)

    for path in (chain, OUT / 'amendment12.json'):
        text = path.read_text()
        for bad in ('/mnt/', '/home/', 'wt/'):
            if bad in text:
                raise ValueError('F1_ABSOLUTE_PIN: %s contains %r'
                                 % (path.name, bad))

    print('amendment12 sha256 %s' % digest)
    print('runner %s -> %s'
          % (amd['runner']['superseded_sha256'][:12],
             amd['runner']['sha256'][:12]))
    rb = amd['core_rebind']
    print('core_rebind CARRIED: %d changed, %d new, %d unchanged (none moved)'
          % (len(rb['changed']), len(rb['new_inputs']), len(rb['unchanged'])))


if __name__ == '__main__':
    main()
