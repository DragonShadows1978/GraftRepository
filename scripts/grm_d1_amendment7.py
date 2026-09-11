#!/usr/bin/env python3
"""GRM-D1 follow-up 7: LT1.1 amendment 7 — production turn semantics.

THE FINDING, and it is the cause of the A+ = LT1 null.

The lead ran LT1.1 arm A+ on the card. 24 of 26 cells COMPLETE, then
`A-189-196` RED with `KeyError: 'assistant'` on the first `recap_probe`.
`--summary` over the 24 completed cells returned fresh 14/15, corrections
5/10, aliases 5/10 -- byte-for-byte LT1's numbers.

`grm_lt1_worker.execute` branched on `event['kind']` for `probe` and `recap`
only. Every other kind -- `fact`, `ordinary`, and critically `supersede` and
`alias` -- fell to `a.feed(e2e.harmony_turn(...))`, whose own comment says it:
"User corrections are ordinary prose, not hidden supersede calls." So:

  * the 15 `supersede` turns LT1.1's fixture registered were deposited as
    PROSE. Nothing was retired, the stale node stayed in `_route_cand_base`,
    and the correction rows failed for exactly the reason D1 diagnosed in LT1
    -- the fixture-lineage trap, re-created one layer down, inside the fix
    for it;
  * no `alias_fold` decision appears anywhere in the A+ receipts, because
    A1's merge runs inside the production turn funnel
    (`runtime._finish_turn_event` -> `_guard_deposit_width` ->
    `_alias_fold_deposits`) and `arena.feed()` never calls it. The flag was
    pinned and the mechanism was inert. My CPU A+ check called
    `alias_fold_pass()` explicitly; the GPU worker never did.

LT1.1 as run measured nothing new. That is a null with a known cause, not a
result about memory.

THE FIX. Two more seams on `grm_lt1_worker.execute`, byte-identical defaults:

  `deposit_turn`      supersede -> `repo.apply_memory_command(...)`, the
                      production correction path; every deposit ->
                      `feed()` THEN `runtime._finish_turn_event(...)`, the
                      funnel `scripts/grm_chat.py:240` calls, where the width
                      guard, A1's fold and the librarian live.
  `recap_probe_turn`  the probe path: asked, scored, never deposited. LT1 has
                      no such kind, so the default REFUSES
                      (`UNREGISTERED_TURN_KIND`) instead of falling through to
                      the deposit branch and raising KeyError.

FAIRNESS. Arms A and A+ now share THIS worker. A-vs-A+ is therefore a fair
same-worker contrast, differing by one pinned environment variable. LT1's
numbers are the PARENT BASELINE -- a different worker over a different
fixture -- and must not be read as a same-worker control.

MEASURED ON THE CPU DOUBLE, 26 cells, arm A: corrections 10/10 (the lead run
scored 5/10), fresh 8/15, aliases 0/10, recap 0/5. The correction result is
the supersession fix working end to end through the real `execute`. The fresh
and alias numbers are the CPU prose double's reading ability, not a memory
claim -- that is what the GPU arms are for.

Prior art:
  * The production turn funnel, and the finding that a battery may stop at
    `feed()` while a product may not: `scripts/grm_chat.py:224-245` (GRM-P1,
    GRM contributors 2026). Taken with its reasoning.
  * The supersede turn: `scripts/grm_e2e_session.py:2590-2597`
    (GRM contributors, 2026), reused unchanged.
  * `_finish_turn_event` -> `_guard_deposit_width` -> `_alias_fold_deposits`
    / `_librarian`: `core/grm_runtime.py:69-115` (A1 + GRM contributors,
    2026). Read, never modified.
  * Seam with a byte-identical default: this campaign's own pattern, from
    amendment 5.
  * Mine: routing a frozen fixture's kinds to these existing paths.
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

OUT = ROOT / 'artifacts/grm_d1/lt1_1'
AMENDMENT6 = OUT / 'amendment6.json'
ORDER = ROOT / 'orders/GRM_D1_LT1_RESIDUAL_DIAGNOSIS.md'

RUNNER = 'scripts/grm_lt1_1.py'
WORKER = 'scripts/grm_lt1_worker.py'
RUNNER_TESTS = ('tests/test_grm_lt1_1_runner.py',
                'tests/test_grm_lt1_1_resume_route.py',
                'tests/test_grm_lt1_1_preflight.py',
                'tests/test_grm_lt1_1_child_spawn.py',
                'tests/test_grm_lt1_1_idle_wait.py',
                'tests/test_grm_lt1_1_production_turns.py')

ALIAS_FLAG = 'GRM_ALIAS_FOLD_MERGE'
ARCHIVE_LABEL = 'r1_harness_null'

FOLLOW_UP = (
    'Lead follow-up 7 to GRM-D1 (2026-09-11): LT1.1 A+ ran 24/26 cells then '
    'died on the first recap_probe with KeyError: assistant, and --summary '
    'returned LT1 numbers byte-for-byte because execute() fed supersede and '
    'alias turns as prose and never called the production turn funnel where '
    'A1 fold-merge lives. Route fixture kinds through production semantics; '
    'gate on the CPU double through the REAL execute; state that arms A and '
    'A+ now share this worker so A-vs-A+ is the fair contrast and LT1 is the '
    'parent baseline; archive both runs as r1_harness_null and re-arm.')


def sha_path(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, indent=2) + '\n').encode()


def amendment():
    if not AMENDMENT6.exists():
        raise ValueError('LT11_AMENDMENT6_MISSING')
    parent_sha = sha_path(AMENDMENT6)
    recorded = (OUT / 'amendment6.sha256').read_text().split()[0]
    if parent_sha != recorded:
        raise ValueError('LT11_AMENDMENT6_SHA_MISMATCH')
    parent = json.loads(AMENDMENT6.read_text())

    return dict(
        schema='grm.lt1_1.amendment7.v1',
        amendment=7,
        previous_amendment='artifacts/grm_d1/lt1_1/amendment6.json',
        previous_amendment_sha256=parent_sha,
        registration='artifacts/grm_d1/lt1_1/registration.json',
        registration_sha256=parent['registration_sha256'],
        order='orders/GRM_D1_LT1_RESIDUAL_DIAGNOSIS.md',
        order_sha256=sha_path(ORDER),
        follow_up_instruction=FOLLOW_UP,
        purpose=('Route fixture kinds through production semantics so '
                 'supersede retires and A1 fold-merge actually runs.'),
        finding=dict(
            headline='the A+ = LT1 null was a harness null, not a memory result',
            symptom='24/26 cells COMPLETE then A-189-196 RED with '
                    "KeyError: 'assistant'; --summary over the 24 gave "
                    'fresh 14/15, corrections 5/10, aliases 5/10 -- LT1 '
                    'numbers byte-for-byte',
            cause='grm_lt1_worker.execute branched on kind only for probe and '
                  'recap; supersede and alias fell to '
                  'a.feed(harmony_turn(...)), so nothing was retired and the '
                  'production turn funnel -- where A1 alias_fold runs -- was '
                  'never called',
            consequences=[
                'the 15 supersede turns were deposited as prose: the stale '
                'node stayed in _route_cand_base, which is the fixture-'
                'lineage trap D1 diagnosed, re-created inside its own fix',
                'no alias_fold decision appears in any A+ receipt: the flag '
                'was pinned and the mechanism was inert',
                'the D1 CPU A+ check called alias_fold_pass() explicitly, so '
                'it proved A1 works but NOT that the GPU worker runs it',
            ],
            verdict='LT1.1 as run measured nothing new'),
        fix=dict(
            seams_added=['worker.deposit_turn -> deposit_turn',
                         'worker.recap_probe_turn -> recap_probe_turn'],
            supersede='repo.apply_memory_command(correction_command), the '
                      'production path scripts/grm_e2e_session.py:2590-2597 '
                      'runs for kind == "supersede"',
            deposits='feed() THEN runtime._finish_turn_event("chat", before, '
                     'autosave=True), the funnel scripts/grm_chat.py:240 '
                     'calls; it runs the width guard, _alias_fold_deposits '
                     '(A1) and _librarian',
            recap_probe='the probe path: asked, scored with the value-span '
                        'scorer into probes.jsonl with kind="recap_probe", '
                        'never deposited',
            unchanged=['probe', 'recap'],
            lt1_default='byte-identical; the prose deposit and its comment '
                        'are preserved verbatim and LT1 own campaign is '
                        'unaffected',
            also_fixed='summary() classified every probes.jsonl row against '
                       'the recall-probe map, which raised KeyError: '
                       "'recap_1' once recap rows existed; it now splits by "
                       'the row kind and scores a recap class'),
        fairness=dict(
            statement='arms A and A+ now share THIS worker, so A-vs-A+ is a '
                      'fair same-worker contrast differing by one pinned '
                      'environment variable',
            lt1='LT1 numbers are the PARENT BASELINE -- a different worker '
                'over a different fixture -- and are NOT a same-worker '
                'control',
            gate='tests/test_grm_lt1_1_production_turns.py::'
                 'test_the_two_arms_share_this_worker'),
        cpu_measurement=dict(
            scope='26 cells, arm A, CPU double, through the real execute',
            corrections='10/10 (the lead GPU run scored 5/10)',
            fresh='8/15',
            aliases='0/10',
            recap='0/5',
            reading='the correction result is the supersession fix working '
                    'end to end through the real worker. The fresh, alias and '
                    'recap numbers are the CPU prose double reading ability, '
                    'NOT a memory claim -- that is what the GPU arms measure.',
            evidence_class='CPU fake worker (lineage and plumbing; NOT a '
                           'language-model quality measurement)'),
        alias_attribution=dict(
            observed='under BOTH arms, an active node names the alias and its '
                     'base after the alias turns',
            honest_reading='on this fixture the librarian chronicle fold '
                           'reaches the same pairing the alias fold-merge '
                           'would, so an outcome test cannot attribute the '
                           'pairing to A1',
            what_the_gate_asserts='the OUTCOME the alias rows need (one '
                                  'active node naming both), never which '
                                  'mechanism produced it',
            open='whether A1 fold-merge changes alias recall on the GPU arms '
                 'is unmeasured; it is what the A/A+ contrast is for'),
        archive=dict(
            label=ARCHIVE_LABEL,
            runs=['run_Aplus (24 COMPLETE + 1 RED, the null)',
                  'run_A (in progress at the time)'],
            policy='create-only: each run is copied under '
                   'archive/<label>/<run>, never deleted or overwritten, and '
                   'the live cells are cleared so both arms re-arm from '
                   'cell 1',
            why='the null is evidence about the harness and is kept as such'),
        runner=dict(
            path=RUNNER,
            sha256=sha_path(ROOT / RUNNER),
            superseded_sha256=parent['runner']['sha256'],
            worker=WORKER,
            worker_sha256=sha_path(ROOT / WORKER),
            tests=list(RUNNER_TESTS),
            tests_sha256={name: sha_path(ROOT / name) for name in RUNNER_TESTS
                          if (ROOT / name).exists()}),
        seams=dict(count=8,
                   added=['worker.deposit_turn', 'worker.recap_probe_turn'],
                   existing=['lt.FIX', 'worker.bind', 'worker.RUN',
                             'worker.worker', 'worker.spawn_argv',
                             'worker.spawn_env', 'worker.await_idle']),
        ruling=parent['ruling'],
        arms=parent['arms'],
        budget_gpu_seconds_per_arm=parent['budget_gpu_seconds_per_arm'],
        budget_gpu_seconds_total=parent['budget_gpu_seconds_total'],
        budget_gpu_hours_total=parent['budget_gpu_hours_total'],
        process_safety=parent['process_safety'],
        prior_art=[
            'Production turn funnel and its finding: scripts/grm_chat.py:'
            '224-245 (GRM-P1, GRM contributors 2026)',
            'Supersede turn: scripts/grm_e2e_session.py:2590-2597 '
            '(GRM contributors, 2026), reused unchanged',
            '_finish_turn_event -> _guard_deposit_width -> '
            '_alias_fold_deposits / _librarian: core/grm_runtime.py:69-115 '
            '(A1 + GRM contributors, 2026), read not modified',
            'Seam with a byte-identical default: this campaign own pattern '
            'from amendment 5',
            'Routing a frozen fixture kinds to these existing paths: no prior '
            'art known to me',
        ],
    )


def archive_runs():
    """Copy both runs under `archive/r1_harness_null/`, then clear them."""
    import shutil
    out = []
    base = OUT / 'archive' / ARCHIVE_LABEL
    for name in ('run_Aplus', 'run_A'):
        live = OUT / name
        cells = live / 'cells'
        if not cells.exists() or not any(cells.iterdir()):
            out.append(dict(run=name, status='NOTHING_TO_ARCHIVE'))
            continue
        target = base / name
        if target.exists():
            out.append(dict(run=name, status='ALREADY_ARCHIVED',
                            archive=str(target)))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(live, target)
        shutil.rmtree(cells)
        cells.mkdir(parents=True)
        out.append(dict(run=name, status='ARCHIVED', archive=str(target),
                        cells_archived=len(list((target / 'cells').iterdir()))))
    return out


def lead_commands(amend_sha):
    doc = amendment()
    lines = [
        '# LT1.1 — arm A and arm A+ (amendment 7)',
        '# registration sha256: %s' % doc['registration_sha256'],
        '# amendment 6  sha256: %s' % doc['previous_amendment_sha256'],
        '# amendment 7  sha256: %s' % amend_sha,
        '# runner            : %s (sha256 %s)'
        % (RUNNER, doc['runner']['sha256']),
        '# worker            : %s (sha256 %s)'
        % (WORKER, doc['runner']['worker_sha256']),
        '#',
        '# 26 cells per arm. Reservation ceiling %d s (%.2f GPU-h) PER ARM;'
        % (doc['budget_gpu_seconds_per_arm'],
           doc['budget_gpu_seconds_per_arm'] / 3600.0),
        '# both arms: %.2f GPU-h ceiling. Separately resumable.'
        % doc['budget_gpu_hours_total'],
        '#',
        '# THE PREVIOUS A+ RUN WAS A HARNESS NULL (amendment 7). execute()',
        '# fed supersede and alias turns as prose and never called the',
        '# production turn funnel, so nothing was retired and A1 fold-merge',
        '# never ran -- which is why A+ reproduced LT1 numbers exactly.',
        '# Both runs are archived under',
        '#   artifacts/grm_d1/lt1_1/archive/%s/' % ARCHIVE_LABEL,
        '# and both arms re-arm from cell 1.',
        '#',
        '# FAIRNESS: arms A and A+ now share this worker, so A-vs-A+ is the',
        '# fair contrast. LT1 numbers are the PARENT BASELINE (different',
        '# worker, different fixture), not a same-worker control.',
        '#',
        '# ---- CPU gates (no GPU, no lease) ----',
        'cd %s' % ROOT,
        'python3 -m pytest -q tests/test_grm_d1*.py tests/test_grm_lt1_1*.py',
        'python3 scripts/grm_d1_recap.py          # recap oracle 5/5 PASS',
        'python3 scripts/grm_d1_alias_cpu.py      # A+ alias fold/admit/mount',
        '',
    ]
    for arm in ('A', 'A+'):
        spec = doc['arms'][arm]
        label = ('alias fold OFF' if not spec['alias_fold_merge']
                 else '%s=1 pinned' % ALIAS_FLAG)
        lines += [
            '# ---- arm %s (%s) ----' % (arm, label),
            'python3 %s --arm %s --preflight' % (RUNNER, arm),
            'python3 %s --arm %s --dry-run' % (RUNNER, arm),
            '# Route check on CPU: walks the REAL resume path and stops at',
            '# the lease boundary.',
            'python3 %s --arm %s --resume --dry-lease' % (RUNNER, arm),
            '# GPU campaign (takes the shared lease; resumable -- run again',
            '# to continue). A busy card makes it WAIT, not fail:',
            'python3 %s --arm %s --resume' % (RUNNER, arm),
            'python3 %s --arm %s --summary' % (RUNNER, arm),
            '',
        ]
    lines += [
        '# The runner pins the arm itself and carries it into the leased',
        '# child in GRM_LT1_1_ARM. Supersede turns now go through',
        '# apply_memory_command, and every deposit through',
        '# runtime._finish_turn_event, where A1 fold-merge lives:',
        '#   arm A   -> admission_rule=margin_first, alias_fold_merge=False',
        '#   arm A+  -> admission_rule=margin_first, alias_fold_merge=True',
        '',
        '# Registered predictions (before any run):',
    ]
    for arm in ('A', 'A+'):
        lines.append('#   %-3s %s' % (arm, json.dumps(
            doc['arms'][arm]['predictions'], sort_keys=True)))
    return '\n'.join(lines) + '\n'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archived = archive_runs()
    payload = canonical(amendment())
    (OUT / 'amendment7.json').write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (OUT / 'amendment7.sha256').write_text('%s  amendment7.json\n' % digest)
    (OUT / 'lead_commands.txt').write_text(lead_commands(digest))

    doc = amendment()
    print('amendment7 sha256 %s' % digest)
    print('parent amendment6 %s' % doc['previous_amendment_sha256'])
    print('runner %s' % doc['runner']['sha256'])
    print('worker %s' % doc['runner']['worker_sha256'])
    print('finding: %s' % doc['finding']['headline'])
    print('CPU 26-cell arm A: corrections %s, fresh %s, aliases %s, recap %s'
          % (doc['cpu_measurement']['corrections'],
             doc['cpu_measurement']['fresh'],
             doc['cpu_measurement']['aliases'],
             doc['cpu_measurement']['recap']))
    for row in archived:
        print('archive %-10s %s' % (row['run'], row['status']))


if __name__ == '__main__':
    main()
