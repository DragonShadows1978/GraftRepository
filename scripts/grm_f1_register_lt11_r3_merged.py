#!/usr/bin/env python3
"""GRM-F1 follow-up: LT1.1 r3 on the MERGED core -- amendment 11, three arms.

WHY THIS SUPERSEDES THE SINGLE-ARM r3 REGISTRATION.  The first r3 document
(`scripts/grm_f1_register_lt11_r3.py`, amendment 10) registered ONE arm whose
only variable was `GRM_FOLD_RETAIN_SOURCES`.  Since then the lead merged
lc1-wip into this worktree and three treatments now sit on one core:

    F1  GRM_FOLD_RETAIN_SOURCES        a fold's sources stay routable
    F2  GRM_FOLD_ALIAS_GUARD           fact-less alias turns leave the fold
                                       window; attribution QC rejects a
                                       digest that filed facts under the
                                       wrong entity
    F5  GRM_ROUTE_SOLE_BINDER_INSURANCE  the sole binder is inserted into
                                       the k=3 rank plan

All three default OFF.  A campaign that turned them on together and reported
one set of columns could not attribute a single row, so this registers THREE
SEPARATELY RESUMABLE ARMS against the same frozen 200-turn conversation, the
same fixture sha, the same 26-cell arm-A schedule and the same worker:

    A0   everything OFF on THIS core -- the CONTROL that separates the three
         treatments from any drift since r2.  Registered; run only if the
         lead decides.
    A'   F1 + F2 + F5 ON, GRM-A1 OFF
    A+'  A1 + F1 + F2 + F5 ON

A0 is what makes A' and A+' readable.  Without it a moved column could be the
treatments or it could be six weeks of unrelated core drift, and no receipt
could tell the two apart.

AMENDMENT 11 rebinds the core pins to the merged tree with per-file
attribution (F1 / F2 / F5 / unchanged) and chains to amendment 10.  Without
it every arm's child blocks with `LT11_CHILD_PREFLIGHT_BLOCKED:
INPUT_SHA_MISMATCH`, which is the sha pin doing its job.

BUDGET.  The lead accepted the amendment-10 reading: each arm reserves r2's
per-arm lease sum (7410 s) so `run_cell`'s rail behaves exactly as it did in
r2, and the order's 4680 s (1.30 GPU-h) cap binds on MEASURED charged work.
Both numbers are registered per arm.

REPO-RELATIVE PINS ONLY (round-1 lesson).  The single allow-listed absolute
path is `model.path`, the HF weights snapshot carried verbatim from r2 -- an
external model cache, not a prunable fork.

PRIOR ART.
  * `scripts/grm_f1_register_lt11_r3.py` (this seat, 2026-09-11) -- the
    document structure, the budget-conflict block, the measured-not-typed sha
    discipline and the `--dry-run` gate, all reused.
  * `scripts/grm_d1_register_lt11.py` and
    `artifacts/grm_d1/lt1_1/amendment8.json` (GRM contributors, 2026) --
    the registration and core_rebind shapes, reused field for field.
  * Multi-arm one-variable-at-a-time campaign design with an explicit
    control arm is ordinary experimental practice (Fisher, 1935, factorial
    design), taken in kind only -- no specific mechanism borrowed.
    UNVERIFIED -- lead to check; search terms ``factorial experiment design
    control arm``, ``one factor at a time vs factorial``.
  * Mine: the A0/A'/A+' arm split and its attribution argument, the
    per-arm flag matrix, and the amendment-11 pin set.
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

from core import grm_fold_retain as _f1                       # noqa: E402
from core import grm_fold_alias_guard as _f2                  # noqa: E402
from core import grm_admission as _adm                        # noqa: E402

R2 = Path('artifacts/grm_d1/lt1_1')
OUT_REL = Path('artifacts/grm_f1/lt1_1_r3')
OUT = ROOT / OUT_REL
COMMANDS = ROOT / 'artifacts/grm_f1/lead_commands.txt'

F1_FLAG = _f1.ENV_NAME
F2_FLAG = _f2.ENV_NAME
F5_FLAG = _adm.ROUTE_SOLE_BINDER_INSURANCE_ENV
ALIAS_FLAG = 'GRM_ALIAS_FOLD_MERGE'

#: Per arm. The lead accepted this reading at amendment 10.
BUDGET_GPU_SECONDS = 7410              # r2's reservation ceiling, unchanged
ORDER_CAP_GPU_SECONDS = 4680           # 1.30 GPU-h, binding on MEASURED work

#: The three arms. `alias` is GRM_ALIAS_FOLD_MERGE, which selects the A/A+
#: campaign variant the runner already knows.
ARMS = {
    'A0': dict(
        base_arm='A', alias=False, f1=False, f2=False, f5=False,
        out_dir='artifacts/grm_f1/lt1_1_r3/run_A0',
        role='CONTROL',
        purpose=('Everything OFF on the MERGED core. Separates F1/F2/F5 '
                 'from any core drift between r2 and today. Registered now '
                 'so the comparison exists; RUN ONLY IF THE LEAD DECIDES.')),
    "A'": dict(
        base_arm='A', alias=False, f1=True, f2=True, f5=True,
        out_dir="artifacts/grm_f1/lt1_1_r3/run_Aprime",
        role='TREATMENT',
        purpose=('F1 + F2 + F5 ON, GRM-A1 OFF. Comparable to r2 arm A, '
                 'which also ran alias_fold_merge=false.')),
    "A+'": dict(
        base_arm='A+', alias=True, f1=True, f2=True, f5=True,
        out_dir="artifacts/grm_f1/lt1_1_r3/run_Aplusprime",
        role='TREATMENT',
        purpose=('A1 + F1 + F2 + F5 ON. Comparable to r2 arm A+, which ran '
                 'alias_fold_merge=true.')),
}

#: Core inputs the merge moves or adds, with the attribution for each.
MERGED_CHANGED = {
    'core/graft_arena.py':
        'GRM-F1 + GRM-F2. F1: `_deposit_consolidation(..., retain=)` and the '
        'once-resolved `fold_retain_sources`. F2: the '
        '`fold_attribution_guard` hook and `_fold_attribution_ok`, which '
        'rejects on BOTH fold paths before `_deposit_consolidation` is '
        'reached. Verified at the code site by this seat: the attribution '
        'check returns (None, None) ahead of every deposit, so an F2 '
        'rejection can never leave F1 lineage behind.',
    'core/graft_repository.py':
        'GRM-F1 + GRM-F2, merge resolved by this seat. F1: `_fold_once` '
        'retain lifecycle + fold-event lineage; `_alias_consolidate` pins '
        'retain=False so A1 keeps retiring its alias edge. F2: '
        '`_guard_fold_windows` on BOTH `_librarian_jobs` return paths, the '
        'attribution hook install, and `fold_guard_history`. Composition: '
        'F2 excludes from the window BEFORE job selection; F1 applies its '
        'lifecycle to whatever the guarded window folded; the shared abort '
        'branch absorbs an attribution rejection before any F1 lineage is '
        'written.',
    'core/grm_admission.py':
        'GRM-F5. `GRM_ROUTE_SOLE_BINDER_INSURANCE` inserts the sole binder '
        'into the k=3 rank plan. Not modified by this seat; pinned here '
        'because r3 must run on the merged tree and an unpinned input is an '
        'unnoticed input.',
}
MERGED_NEW = {
    'core/grm_fold_alias_guard.py':
        'GRM-F2 NEW INPUT. The alias-guard flag resolver, the entity/clause '
        'vocabulary, `fold_window_excludes` and `attribution_violations`. '
        'Not authored by this seat.',
}


def sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha_file(rel):
    return sha_bytes((ROOT / rel).read_bytes())


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, indent=2) + '\n').encode()


def r2_doc(name):
    return json.loads((ROOT / R2 / name).read_text())


def amd_doc(name):
    return json.loads((ROOT / R2 / name).read_text())


def core_rebind():
    """Amendment 11's block: the merged tree's pins, nothing dropped.

    Starts from amendment 10's COMPLETE pin set (changed + new + unchanged),
    which is exactly what `governing_core_pins` checks, and re-files each
    entry by what the tree says today. A file that moved without an entry in
    MERGED_CHANGED raises rather than being silently rebound.
    """
    prior = amd_doc('amendment10.json')['core_rebind']
    carried = {}
    carried.update({n: c['after_sha256'] for n, c in prior['changed'].items()})
    carried.update({n: e['sha256'] for n, e in prior['new_inputs'].items()})
    carried.update({n: e['sha256']
                    for n, e in prior.get('unchanged', {}).items()})

    changed, unchanged = {}, {}
    for name, before in sorted(carried.items()):
        now = sha_file(name)
        if now == before:
            unchanged[name] = dict(
                sha256=now, attribution='unchanged by the lc1-wip merge')
            continue
        if name not in MERGED_CHANGED:
            raise ValueError(
                'F1_UNEXPECTED_CORE_DRIFT: %s (amendment 10 pinned %s, on '
                'tree %s) -- a file with no registered attribution moved. '
                'STOP and report rather than rebinding it silently.'
                % (name, before, now))
        changed[name] = dict(before_sha256=before, after_sha256=now,
                             attribution=MERGED_CHANGED[name])
    missing = set(MERGED_CHANGED) - set(changed)
    if missing:
        raise ValueError('F1_CHANGED_PIN_DID_NOT_MOVE: %s' % sorted(missing))

    new_inputs = {}
    for name, why in sorted(MERGED_NEW.items()):
        if name in carried:
            raise ValueError('F1_NEW_INPUT_ALREADY_PINNED: %s' % name)
        new_inputs[name] = dict(sha256=sha_file(name), attribution=why)

    return dict(
        supersedes='amendment10.core_rebind',
        source_binding=str(R2 / 'amendment10.json'),
        source_binding_sha256=sha_file(R2 / 'amendment10.json'),
        attribution_note=(
            'The lc1-wip merge moves %d of amendment 10\'s pins and adds %d '
            'new core input(s). Every attribution names the mechanism at the '
            'code site, not a merge label. The %d pins nothing touched are '
            'recorded as unchanged rather than dropped, and this script '
            'RAISES F1_UNEXPECTED_CORE_DRIFT on any file that moved without '
            'a registered attribution.'
            % (len(changed), len(new_inputs), len(unchanged))),
        changed=changed,
        new_inputs=new_inputs,
        unchanged=unchanged,
        unchanged_count=len(unchanged),
    )


def runner_binding():
    """Amendment 10's runner block, carried forward and RE-VERIFIED.

    `lt1_1_preflight` reads `runner` off the LATEST amendment. Neither this
    follow-up nor the merge changes `scripts/grm_lt1_1.py` or
    `scripts/grm_lt1_worker.py`, so amendment 10's binding is carried
    verbatim and BOTH shas are re-measured against the tree: if either moved,
    something happened that this document must not paper over.
    """
    prior = dict(amd_doc('amendment10.json')['runner'])
    for path_key, sha_key in (('path', 'sha256'),
                              ('worker', 'worker_sha256')):
        name = prior[path_key]
        now = sha_file(name)
        if now != prior[sha_key]:
            raise ValueError(
                'F1_RUNNER_DRIFT: %s (amendment 10 pinned %s, on tree %s) -- '
                'this follow-up must not modify the LT1.1 runner or worker. '
                'STOP and report.' % (name, prior[sha_key], now))
    prior['carried_from'] = 'amendment10.json'
    prior['change'] = (
        'UNCHANGED since amendment 10. Both shas re-measured against the '
        'tree at registration time. The runner already carries F1\'s '
        'GRM_FOLD_RETAIN_SOURCES env-carry; F2 and F5 are read from the '
        'environment by core, so no further runner change is needed.')
    return prior


def arm_env(spec):
    """The flag matrix for one arm, as the lead will export it."""
    env = {'GRM_ADMISSION_RULE': 'margin_first', 'GRM_PROFILE': 'eb1_c2'}
    env[F1_FLAG] = '1' if spec['f1'] else '0'
    env[F2_FLAG] = '1' if spec['f2'] else '0'
    env[F5_FLAG] = '1' if spec['f5'] else '0'
    if spec['alias']:
        env[ALIAS_FLAG] = '1'
    return env


def arm_predictions(label, spec):
    """Registered BEFORE the run, per arm, against the r2 arm-A/A+ baseline.

    r2 measured (lead_A_r2_summary.json / lead_Aplus_r2_summary.json):
    arm A  fresh 10/15, corrections 9/10, aliases 10/10, recap 4/5.
    """
    common = dict(
        evidence_class=('prediction registered BEFORE the run, per house '
                        'rules; nothing here is claimed from the CPU double'),
        residency=('BOUNDED and REPORTED. The width guard is unchanged, so '
                   'the registered bound (summed_token_seats <= 2*width + '
                   'recency) must hold on EVERY row. Max seats must be '
                   'reported; retained-source seats are reported separately '
                   'via retained_source_mounted_ids / '
                   'retained_source_token_seats. r2 arm A measured max 94 '
                   'against width 96 over 323 rows.'))
    if label == 'A0':
        return dict(common,
            role='CONTROL',
            fresh='within +/-1 of r2 arm A (10/15)',
            corrections='within +/-1 of r2 arm A (9/10)',
            aliases='within +/-1 of r2 arm A (10/10)',
            recap='within +/-1 of r2 arm A (4/5)',
            statement=(
                'A0 predicts the r2 arm-A numbers reproduce within +/-1 per '
                'class on the merged core with every flag OFF. That is the '
                'whole job of this arm: if A0 does NOT reproduce r2, then '
                'core drift is in play and NO movement in A\' or A+\' can be '
                'attributed to F1/F2/F5 until that is explained.'),
            falsifier=('any class more than 1 off r2 arm A means the OFF '
                       'path is not byte-identical in practice and the '
                       'treatment arms are uninterpretable as registered'))
    if label == "A'":
        return dict(common,
            role='TREATMENT',
            fresh=('>= 14/15 (r2 arm A measured 10/15). F1: the retained '
                   'source turn carries the fact in the wording the probe '
                   'asks for and was retired by the fold in r2. F5: the sole '
                   'binder is inserted into the k=3 plan.'),
            corrections=('>= 9/10 (r2 arm A measured 9/10 -- NO-REGRESSION, '
                         'not an improvement claim). Under F1 a correction '
                         'retires the stale SOURCE as well as the digest, '
                         'which is strictly more than r2 retired. The '
                         'paraphrasing-digest gap stays open in both arms, '
                         'which is why this is not 10/10.'),
            aliases=('10/10 (r2 arm A measured 10/10). GRM-A1 is OFF in this '
                     'arm and F1 pins retain=False on the alias path, so '
                     'this column must not move at all.'),
            recap=('>= 4/5 (r2 arm A measured 4/5). No mechanism predicts a '
                   'specific gain; no-regression only.'),
            attribution_caveat=(
                'THREE flags move together in this arm. A column that moves '
                'is attributable to the SET {F1, F2, F5}, never to one of '
                'them, unless A0 and a single-flag follow-up separate them.'),
            falsifier=('fresh below 14/15 REFUTES the retention hypothesis '
                       'as stated. A drop in ANY class, or one '
                       'RESIDENCY_BOUND_EXCEEDED, is a FAILURE of the set, '
                       'not a tuning opportunity.'))
    return dict(common,
        role='TREATMENT',
        fresh=">= 14/15, same mechanism as A'",
        corrections=">= 9/10, same mechanism as A'",
        aliases=('10/10. A1 is ON here; r2 arm A+ already scored its alias '
                 'column at 10/10 and F2\'s window exclusion must not cost '
                 'it -- the guard drops FACT-LESS alias turns, and A1\'s '
                 'merged digests are fact-bearing by construction.'),
        recap=">= 4/5, no-regression only",
        attribution_caveat=(
            'FOUR flags move together in this arm (A1 + F1 + F2 + F5). Read '
            "it against A', not against r2, to isolate A1."),
        falsifier=("any alias row below A+' 10/10 means F2's window "
                   'exclusion is interacting with A1 and the pair must be '
                   'separated before either is merged'))


def amendment11():
    return dict(
        amendment=11,
        arms=sorted(ARMS),
        previous_amendment='amendment10.json',
        previous_amendment_sha256=sha_file(R2 / 'amendment10.json'),
        order='orders/GRM_F1_FOLD_RETAIN_SOURCES.md',
        order_sha256=sha_file('orders/GRM_F1_FOLD_RETAIN_SOURCES.md'),
        purpose=(
            'Rebind LT1.1\'s sha-bound core inputs to the MERGED tree (F1 + '
            'F2 + F5) and register LT1.1 r3 as THREE separately resumable '
            'arms: A0 (control, all OFF), A\' (F1+F2+F5) and A+\' (A1 + '
            'F1+F2+F5). Amendment 9\'s scoring columns are carried forward '
            'unchanged.'),
        follow_up_instruction=(
            'Amendment 11 supersedes amendment 10\'s core_rebind under the '
            'chain rule grm_lt1_1.governing_core_rebind implements (the '
            'LATEST amendment carrying a core_rebind governs). Amendment '
            '10\'s single-arm r3 registration is SUPERSEDED by the '
            'three-arm document; its budget reading is carried forward, as '
            'the lead accepted it.'),
        scoring_columns_unchanged=dict(
            binding='amendment9.json',
            sha256=sha_file(R2 / 'amendment9.json'),
            statement=('scripts/grm_lt1.score remains the PRIMARY verdict '
                       'and is not modified; amendment 9\'s column 2 is '
                       'reported beside it, unchanged.')),
        budget_gpu_seconds_per_arm=BUDGET_GPU_SECONDS,
        order_cap_gpu_seconds_per_arm=ORDER_CAP_GPU_SECONDS,
        budget_decision=(
            'ACCEPTED BY THE LEAD: each arm reserves r2\'s per-arm lease sum '
            '(7410 s) so run_cell\'s rail behaves exactly as in r2, and the '
            'order\'s 4680 s (1.30 GPU-h) cap binds on MEASURED charged '
            'work. Registering a budget below the reservation sum would '
            'reproduce the mid-campaign rail trip amendment 2 was written '
            'to correct.'),
        arm_flags={label: arm_env(spec) for label, spec in ARMS.items()},
        runner=runner_binding(),
        core_rebind=core_rebind(),
        process_safety=dict(
            gpu=('single lease via flock --wait; the operator has absolute '
                 'right of way; no process this run did not start is ever '
                 'signalled'),
            display='no windowed gate; CPU gates only',
            git='the lead commits; the seat never runs git',
            core_writes=('this seat edited core/ only inside the merge '
                         'conflict hunk, under the existing flags'),
            scratch=('pytest uses --basetemp artifacts/grm_f1/tmp and the '
                     'directory is removed afterwards'),
            subagents='none spawned'),
    )


def registration(amd_sha):
    parent = r2_doc('registration.json')
    cells = [c for c in parent['cells'] if c['arm'] == 'A']
    if len(cells) != 26:
        raise ValueError('F1_R3_CELL_COUNT: %d' % len(cells))
    lease = sum(c['lease_seconds'] for c in cells)
    estimate = round(sum(c['estimate_seconds'] for c in cells), 1)

    arms = {}
    for label, spec in ARMS.items():
        arms[label] = dict(
            role=spec['role'],
            purpose=spec['purpose'],
            base_arm=spec['base_arm'],
            out_dir=spec['out_dir'],
            env=arm_env(spec),
            flags=dict(f1=spec['f1'], f2=spec['f2'], f5=spec['f5'],
                       alias_fold_merge=spec['alias']),
            budget_gpu_seconds=BUDGET_GPU_SECONDS,
            budget_gpu_hours=round(BUDGET_GPU_SECONDS / 3600.0, 2),
            order_cap_gpu_seconds=ORDER_CAP_GPU_SECONDS,
            order_cap_gpu_hours=round(ORDER_CAP_GPU_SECONDS / 3600.0, 2),
            lease_seconds_sum=lease,
            estimate_seconds_sum=estimate,
            within_budget=lease <= BUDGET_GPU_SECONDS,
            within_order_cap=estimate <= ORDER_CAP_GPU_SECONDS,
            separately_resumable=True,
            predictions=arm_predictions(label, spec))

    return dict(
        schema='grm.lt1_1_r3.registration.v2',
        status='FIT_ESTIMATE',
        gpu_executed=False,
        supersedes=dict(
            schema='grm.lt1_1_r3.registration.v1',
            amendment=10,
            why=('amendment 10 registered ONE arm whose only variable was '
                 'GRM_FOLD_RETAIN_SOURCES. The lc1-wip merge put F2 and F5 '
                 'on the same core, so a one-arm campaign could no longer '
                 'attribute a moved column.')),
        amendment=dict(
            number=11,
            chain_path=str(R2 / 'amendment11.json'),
            copy_path=str(OUT_REL / 'amendment11.json'),
            sha256=amd_sha,
            statement=('written into LT1.1\'s own amendment chain, which is '
                       'where grm_lt1_1.governing_core_rebind reads it. '
                       'Amendments 1-10 and every other r2 receipt are '
                       'untouched.')),
        arms=sorted(ARMS),
        arm_detail=arms,
        control_arm='A0',
        treatment_arms=["A'", "A+'"],
        attribution_rule=(
            'A0 is what makes the treatment arms readable. If A0 does not '
            'reproduce r2 arm A within +/-1 per class, core drift is in play '
            'and NO movement in A\' or A+\' may be attributed to F1/F2/F5 '
            'until that is explained. Within an arm, three (or four) flags '
            'move together: a moved column is attributable to the SET, never '
            'to one flag.'),
        cells=cells,
        model=parent['model'],
        model_frame=parent['model_frame'],
        fixture=dict(path=str(R2 / 'dialogue.json'),
                     sha256=sha_file(R2 / 'dialogue.json'),
                     statement=('byte-identical to r2; all three arms replay '
                                'the same frozen conversation')),
        worker=dict(path='scripts/grm_lt1_1.py',
                    sha256=sha_file('scripts/grm_lt1_1.py'),
                    statement='the REAL LT1.1 worker, same for all arms'),
        baseline=dict(
            arm_A=dict(
                path=str(R2 / 'lead_A_r2_summary.json'),
                sha256=sha_file(R2 / 'lead_A_r2_summary.json'),
                measured=dict(fresh='10/15', corrections='9/10',
                              aliases='10/10', recap='4/5'),
                residency=dict(rows=323, max_summed_token_seats=94, width=96),
                nodes=dict(total=199, active=36, inactive=163,
                           retired_by_fold_alone=148,
                           retired_by_correction=15)),
            arm_Aplus=dict(
                path=str(R2 / 'lead_Aplus_r2_summary.json'),
                sha256=sha_file(R2 / 'lead_Aplus_r2_summary.json'))),
        flags=[
            dict(name=F1_FLAG, treatment='F1', default='OFF',
                 what='a fold\'s sources stay ACTIVE and routable; the '
                      'digest is ADDED, not substituted',
                 evidence='tests/test_grm_f1_fold_retain.py'),
            dict(name=F2_FLAG, treatment='F2', default='OFF',
                 what='fact-less alias/rename turns leave the fold window; '
                      'attribution QC rejects a digest that filed facts '
                      'under the wrong entity',
                 evidence='tests/test_grm_f2_alias_guard.py'),
            dict(name=F5_FLAG, treatment='F5', default='OFF',
                 what='the sole binder is inserted into the k=3 rank plan',
                 evidence='tests/test_grm_f5_sole_binder_insurance.py'),
            dict(name=ALIAS_FLAG, treatment='A1', default='OFF',
                 what='alias resolution by fold-merge',
                 evidence='tests/test_grm_a1_alias_fold.py'),
        ],
        composition=dict(
            statement=(
                'F2 excludes fact-less alias turns from each fold WINDOW '
                'before job selection (_guard_fold_windows, on BOTH '
                '_librarian_jobs return paths). F1\'s retain lifecycle then '
                'applies to whatever the guarded window actually folded. '
                'F2\'s attribution QC rejects inside consolidate(), BEFORE '
                '_deposit_consolidation, so a rejected digest never leaves '
                'F1 lineage on any node. A1 pins retain=False on the alias '
                'path so its edge is always superseded.'),
            proof=['tests/test_grm_f1_fold_retain.py',
                   'tests/test_grm_f2_alias_guard.py',
                   'artifacts/grm_f1/off_identity/'],
            resolved_by='this seat, in the merge conflict hunk only'),
        known_gaps=[dict(
            id='F1-N1',
            title='a PARAPHRASING fold digest escapes correct_memory',
            statement=(
                'correct_memory matches the query against node TEXT, so a '
                'digest that paraphrases a fact rather than copying its '
                'wording is never superseded. Measured in BOTH F1 arms: OFF '
                'supersedes nothing at all, ON supersedes the retained '
                'source but the digest still escapes.'),
            caused_by_f1=False,
            owner='F2 successor (entity-scoped digest supersession)',
            bearing_on_r3=('corrections predicted at >= 9/10, not 10/10, in '
                           'every treatment arm'))],
        cpu_gates=[
            'python3 -m pytest -q tests/test_grm_f1_fold_retain.py '
            'tests/test_grm_f2_alias_guard.py '
            'tests/test_grm_f5_sole_binder_insurance.py',
            'python3 -m pytest -q tests/test_grm_scout_fix5.py '
            'tests/test_grm_scout_fix9.py tests/test_grm_a1_alias_fold.py',
            'python3 -m pytest -q tests/test_grm_s4_fold_order.py '
            'tests/test_grm_lt1_1_production_turns.py',
            'python3 scripts/grm_f1_register_lt11_r3_merged.py',
        ],
        process_safety=dict(
            gpu=('single lease via flock --wait; the operator has absolute '
                 'right of way; no process this run did not start is ever '
                 'signalled'),
            display='no windowed gate; CPU gates only',
            git='the lead commits; the seat never runs git',
            paths='every pin in this document is repository-relative'),
        prior_art=[
            'Single-arm r3 document: scripts/grm_f1_register_lt11_r3.py '
            '(this seat, 2026-09-11), reused as the structure',
            'LT1.1 registration + core_rebind shapes: '
            'scripts/grm_d1_register_lt11.py and '
            'artifacts/grm_d1/lt1_1/amendment8.json (GRM contributors, 2026)',
            'Factorial design with an explicit control arm (Fisher, 1935), '
            'in kind only -- UNVERIFIED, lead to check',
        ],
    )


def lead_commands(reg_sha, amd_sha, amd_number=11):
    lines = [
        '# LT1.1 r3 on the MERGED core -- THREE separately resumable arms',
        '# registration sha256: %s' % reg_sha,
        '# amendment%-2d   sha256: %s' % (amd_number, amd_sha),
        '# (the GOVERNING amendment -- verify against this one)',
        '#',
        '# Per arm: reservation ceiling %d s (%.2f GPU-h), r2 unchanged;'
        % (BUDGET_GPU_SECONDS, BUDGET_GPU_SECONDS / 3600.0),
        '#          ORDER CAP on MEASURED work %d s (%.2f GPU-h).'
        % (ORDER_CAP_GPU_SECONDS, ORDER_CAP_GPU_SECONDS / 3600.0),
        '#',
        '# ATTRIBUTION: A0 is the control. If A0 does not reproduce r2 arm A',
        '# within +/-1 per class, core drift is in play and NO movement in',
        "# A' or A+' may be attributed to F1/F2/F5 until that is explained.",
        '# Within an arm the flags move TOGETHER: a moved column belongs to',
        '# the SET, never to one flag.',
        '#',
        '# --out GOVERNS EVERY ROOT the runner and the worker consult:',
        '# pending(), run_cell(), accounting(), the cell directories, the',
        '# checkpoints and the leased child. With --out given the arm',
        '# default root is never read -- asserted at start, and the',
        '# resolved roots are printed in the receipt and the operator log',
        '# ("LT1.1 roots {...}"). Follow-up 3 fixed the drop that sent the',
        "# lead's first arm-A' run into r2's frozen run root.",
        '#',
        '# EVERY runnable line below ends in --dry-run. Removing it is a',
        '# deliberate act by the lead after reading the registration.',
    ]
    for label in ('A0', "A'", "A+'"):
        spec = ARMS[label]
        env = arm_env(spec)
        lines += ['#', '# ---------- arm %s (%s) ----------' % (label,
                                                                spec['role'])]
        if label == 'A0':
            lines.append('# REGISTERED ONLY -- run if the lead decides.')
        for key in sorted(env):
            lines.append('export %s=%s' % (key, env[key]))
        if not spec['alias']:
            lines.append('unset %s' % ALIAS_FLAG)
        base = spec['base_arm']
        lines += [
            'python3 -m scripts.grm_lt1_1 --arm %s --preflight --dry-run'
            % base,
            'python3 -m scripts.grm_lt1_1 --arm %s --out %s --dry-run'
            % (base, spec['out_dir']),
            'flock --wait 7200 /var/lock/grm_gpu.lock \\',
            '  python3 -m scripts.grm_lt1_1 --arm %s --resume '
            '--out %s --dry-run' % (base, spec['out_dir']),
        ]
    return lines


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    COMMANDS.parent.mkdir(parents=True, exist_ok=True)

    # AMENDMENT 11 IS FROZEN ONCE EMITTED.  Follow-up 3 legitimately moved the
    # runner sha, and `runner_binding()` below RAISES F1_RUNNER_DRIFT on that
    # -- correctly: amendment 11 recorded the runner as unchanged, and
    # re-emitting it against a moved runner would rewrite a receipt to say
    # something it did not say on its day.  Amendment 12 owns that rebind.
    # So when the chain document already exists, this script re-reads it
    # instead of regenerating it, and only the registration and the lead
    # command file are refreshed.
    chain11 = ROOT / R2 / 'amendment11.json'
    if chain11.exists():
        amd_bytes = chain11.read_bytes()
        amd = json.loads(amd_bytes)
        amd_sha = sha_bytes(amd_bytes)
        regenerated = False
    else:
        amd = amendment11()
        amd_bytes = canonical(amd)
        amd_sha = sha_bytes(amd_bytes)
        regenerated = True

    # Follow-up 3: amendment 12 rebinds the RUNNER (see
    # scripts/grm_f1_register_lt11_r3_amd12.py). The three-arm registration
    # this script writes is unchanged by it, but the lead command file must
    # name the amendment that actually GOVERNS, or an operator would verify
    # against a superseded document.
    chain12 = ROOT / R2 / 'amendment12.json'
    governing_amd, governing_sha = ((12, sha_file(R2 / 'amendment12.json'))
                                    if chain12.exists()
                                    else (11, amd_sha))

    chain = ROOT / R2 / 'amendment11.json'
    if regenerated:
        chain.write_bytes(amd_bytes)
        (ROOT / R2 / 'amendment11.sha256').write_text(
            '%s  amendment11.json\n' % amd_sha)
        (OUT / 'amendment11.json').write_bytes(amd_bytes)
        (OUT / 'amendment11.sha256').write_text(
            '%s  amendment11.json\n' % amd_sha)
    elif amd.get('order') != 'orders/GRM_F1_FOLD_RETAIN_SOURCES.md':
        raise ValueError(
            'F1_AMENDMENT11_FOREIGN: %s was written by order %r, not by this '
            'one -- refusing to build a registration on another order\'s '
            'chain receipt. STOP and report.' % (chain, amd.get('order')))

    reg = registration(amd_sha)
    reg_bytes = canonical(reg)
    (OUT / 'registration_merged.json').write_bytes(reg_bytes)
    reg_sha = sha_bytes(reg_bytes)
    (OUT / 'registration_merged.sha256').write_text(
        '%s  registration_merged.json\n' % reg_sha)

    COMMANDS.write_text('\n'.join(
        lead_commands(reg_sha, governing_sha, governing_amd)) + '\n')

    allowed = {reg['model']['path']}
    for path in (OUT / 'registration_merged.json', chain, COMMANDS):
        text = path.read_text()
        for permitted in allowed:
            text = text.replace(json.dumps(permitted)[1:-1], '<model.path>')
        for bad in ('/mnt/', '/home/', 'wt/'):
            if bad in text:
                raise ValueError('F1_ABSOLUTE_PIN: %s contains %r'
                                 % (path.name, bad))

    print('registration sha256 %s' % reg_sha)
    print('amendment11  sha256 %s' % amd_sha)
    rb = amd['core_rebind']
    print('core_rebind: %d changed, %d new, %d unchanged'
          % (len(rb['changed']), len(rb['new_inputs']), len(rb['unchanged'])))
    for name, e in sorted(rb['changed'].items()):
        print('   %-30s %s -> %s' % (name, e['before_sha256'][:12],
                                     e['after_sha256'][:12]))
    for name, e in sorted(rb['new_inputs'].items()):
        print('   %-30s NEW %s' % (name, e['sha256'][:12]))
    for label in ('A0', "A'", "A+'"):
        d = reg['arm_detail'][label]
        print('arm %-4s %-9s base=%-2s within_budget=%s within_cap=%s  %s'
              % (label, d['role'], d['base_arm'], d['within_budget'],
                 d['within_order_cap'],
                 ' '.join('%s=%s' % (k.replace('GRM_', ''), v)
                          for k, v in sorted(d['env'].items())
                          if k.startswith('GRM_FOLD') or k.startswith('GRM_ROUTE')
                          or k == ALIAS_FLAG)))


if __name__ == '__main__':
    main()
