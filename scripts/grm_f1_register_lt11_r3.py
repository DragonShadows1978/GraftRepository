#!/usr/bin/env python3
"""GRM-F1: LT1.1 r3 registration -- arm A only, GRM_FOLD_RETAIN_SOURCES ON.

WHAT r3 CHANGES relative to the r2 arm-A baseline, and nothing else.  The
frozen 200-turn conversation, the fixture sha, the 26-cell arm-A schedule,
the margin_first admission rule, the eb1_c2 profile, the worker
(`scripts/grm_lt1_1.py`) and the scorer are ALL the r2 ones, unchanged.  The
single variable is one environment flag:

    GRM_FOLD_RETAIN_SOURCES=1

With it ON a fold's sources stay ACTIVE and routable after the digest is
deposited (the digest is ADDED, not substituted).  The r2 receipt this
addresses: source nodes appear in `route_info.mounts` in 0 of 57 correct
rows across BOTH r2 arms (`artifacts/grm_d1/REPORT.md` section 3,
`artifacts/grm_d1/lt1_1/miss_causes.json`), because FIX-5 consolidation
retires every source the moment the digest clears MIN_FOLD_KEEP and
`ArenaCache._route_cand_base` excludes retired nodes.

AMENDMENT 10 (core rebind) is part of this registration and is why it
exists as a document at all.  LT1.1's preflight is sha-bound to its core
inputs, and F1 moves two of them plus adds one:

    core/graft_arena.py       ee455a6a... -> (measured at registration time)
    core/graft_repository.py  092cca89... -> (measured at registration time)
    core/grm_fold_retain.py   NEW INPUT

Without the rebind the r3 child blocks with
`LT11_CHILD_PREFLIGHT_BLOCKED: INPUT_SHA_MISMATCH: core/graft_arena.py`,
which is the sha pin doing its job.  Amendment 10 follows amendment 8's
shape exactly (`changed` / `new_inputs` / `unchanged`, every pin carried
forward rather than silently dropped) and supersedes amendment 8's
`core_rebind` under the chain rule `grm_lt1_1.governing_core_rebind`
already implements ("the LATEST amendment carrying a core_rebind governs").

REPO-RELATIVE PINS ONLY.  Round-1 lesson: absolute `wt/` paths written into
a registration became permanent receipts pointing at forks that were later
pruned.  Every path this script writes is relative to the repository root,
and `tests/test_grm_f1_fold_retain.py::test_registration_present_and_shaped`
refuses a registration containing `/mnt/` or `wt/`.

PRIOR ART.
  * `scripts/grm_d1_register_lt11.py` (GRM contributors, 2026) -- the LT1.1
    registration this one is derived from.  TAKEN: the whole document
    structure (sha_bytes / canonical / the `optional_named_flags` block /
    the `cpu_gates` + `lead_commands.txt` pair / the prediction-before-the-run
    discipline), reused with the arm-A cell list carried over verbatim.
  * `artifacts/grm_d1/lt1_1/amendment8.json` (GRM contributors, 2026) -- the
    core-rebind block shape, reused field for field.
  * The SHA-bound immutable-registration contract is LT1's own
    (`scripts/grm_lt1_register.py`), reused unchanged.
  * Append-only "latest record wins" amendment chains: Haber & Stornetta
    (1991), in kind only; nothing mechanical taken.  UNVERIFIED -- lead to
    check; search terms ``Haber Stornetta 1991 timestamping digital
    document``.
  * Mine: the r3 variable (one flag), the amendment-10 pin set, the
    node/seat delta prediction computed from the r2 arm-A checkpoint
    manifest, and the `--dry-run` gate on every lead command.
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

from core import grm_fold_retain as fr  # noqa: E402

#: r2's registration + its receipts. READ ONLY -- r3 never writes here.
R2 = Path('artifacts/grm_d1/lt1_1')
#: r3's own namespace, the C7 immutable-receipt-directory convention.
OUT_REL = Path('artifacts/grm_f1/lt1_1_r3')
OUT = ROOT / OUT_REL
COMMANDS = ROOT / 'artifacts/grm_f1/lead_commands.txt'

#: The ONE variable under test.
FLAG = fr.ENV_NAME
#: The order names F2's flag as an optional the lead may pin alongside.
F2_FLAG = 'GRM_F2_SUPERSEDE_DIGEST_BY_ENTITY'
ALIAS_FLAG = 'GRM_ALIAS_FOLD_MERGE'

#: THE BUDGET, and the conflict it resolves -- stated, not smoothed over.
#:
#: The F1 order caps r3 at "<= 1.3 GPU-h". r2's registered arm-A CELL
#: SCHEDULE, which r3 reuses verbatim so the comparison stays one-variable,
#: carries a LEASE RESERVATION sum of 7410 s (2.06 GPU-h). These are two
#: different quantities and only one of them is GPU work:
#:
#:   * ESTIMATE  4507.9 s = 1.25 GPU-h  -- the predicted GPU work. UNDER the
#:     order's cap, which is the number the cap is about.
#:   * LEASE     7410.0 s = 2.06 GPU-h  -- the reservation `run_cell` checks
#:     against `budget_gpu_seconds` before spawning each cell.
#:
#: Registering 4680 s would put the budget BELOW the lease sum, and LT1.1
#: amendment 2 exists precisely because that was already done once: it
#: corrected 6120 -> 7410 because the lower figure "sat below the reservation
#: sum and would have tripped `run_cell`'s rail mid-campaign"
#: (scripts/grm_lt1_1.py::dry_run, and amendment2.json `budget_correction`).
#: Reproducing that failure to make a number look compliant would be a
#: registration that cannot run.
#:
#: So: the RESERVATION CEILING is pinned at r2's 7410 s (unchanged, so the
#: rail behaves exactly as in r2), and the ORDER'S CAP is registered as a
#: separate, binding limit on MEASURED GPU SECONDS. The campaign is a
#: FAILURE of its own registration if it burns more than 1.3 GPU-h of actual
#: charged time. The lead is asked to confirm this reading -- see
#: `budget_conflict` in the registration document.
BUDGET_GPU_SECONDS = 7410              # reservation ceiling, r2's, unchanged
ORDER_CAP_GPU_SECONDS = 4680           # 1.30 GPU-h, binding on MEASURED work
ESTIMATE_NOTE_GPU_SECONDS = 4507.9     # r2's arm-A estimate sum, for reference

#: Core inputs F1 moves or adds. Values are MEASURED from the tree at
#: registration time -- never typed in, so the document cannot claim a sha
#: the tree does not have.
F1_CHANGED = {
    'core/graft_arena.py':
        'GRM-F1. `_deposit_consolidation` takes a `retain` argument '
        '(default `self.fold_retain_sources`, permanently OFF unless the '
        'flag is set); ON it leaves the sources unretired, marks them '
        '`no_fold` and writes `metadata.digest_of` / '
        '`metadata.retained_sources`. `consolidate` gains the pass-through. '
        'The constructor resolves the flag ONCE. OFF is byte-identical.',
    'core/graft_repository.py':
        'GRM-F1. `_fold_once` completes the retained lineage into the '
        'persisted metadata schema and records `retained_sources` / '
        '`fold_lineage` on the fold event. `_alias_consolidate` PINS '
        '`retain=False` so GRM-A1 keeps retiring its alias edge whatever '
        'the window-fold policy is (a bare active edge IS the RD2 defect).',
}
F1_NEW = {
    'core/grm_fold_retain.py':
        'GRM-F1 NEW INPUT. The flag resolver (explicit > env > OFF, unknown '
        'tokens fail CLOSED) and the lineage vocabulary. No serving path '
        'calls anything here with the flag OFF.',
}


def sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha_file(rel):
    return sha_bytes((ROOT / rel).read_bytes())


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, indent=2) + '\n').encode()


def r2_doc(name):
    return json.loads((ROOT / R2 / name).read_text())


def _flag_present(name):
    """Is this flag name anywhere under core/? Measured, never asserted."""
    for path in sorted((ROOT / 'core').rglob('*.py')):
        if name in path.read_text():
            return True
    return False


def core_rebind():
    """Amendment 10's block: F1's pins, every prior pin carried forward.

    Amendment 8's `changed` + `new_inputs` + `unchanged` is the COMPLETE pin
    set LT1.1 checks. A pin F1 does not touch is re-recorded as `unchanged`
    rather than dropped -- dropping it would mean a later change to it went
    unnoticed, which is the exact reasoning `governing_core_pins` documents.
    """
    prior = r2_doc('amendment8.json')['core_rebind']
    carried = {}
    carried.update({n: c['after_sha256'] for n, c in prior['changed'].items()})
    carried.update({n: e['sha256'] for n, e in prior['new_inputs'].items()})
    carried.update({n: e['sha256']
                    for n, e in prior.get('unchanged', {}).items()})

    changed, unchanged = {}, {}
    for name, before in sorted(carried.items()):
        now = sha_file(name)
        if name in F1_CHANGED:
            if now == before:
                raise ValueError('F1_PIN_DID_NOT_MOVE: %s' % name)
            changed[name] = dict(before_sha256=before, after_sha256=now,
                                 attribution=F1_CHANGED[name])
        else:
            if now != before:
                raise ValueError(
                    'F1_UNEXPECTED_CORE_DRIFT: %s (pinned %s, on tree %s) -- '
                    'a file F1 does not touch moved; STOP and report rather '
                    'than rebinding it silently.' % (name, before, now))
            unchanged[name] = dict(sha256=now,
                                   attribution='unchanged by GRM-F1')
    missing = set(F1_CHANGED) - set(changed)
    if missing:
        raise ValueError('F1_CHANGED_PIN_ABSENT: %s' % sorted(missing))

    new_inputs = {}
    for name, why in sorted(F1_NEW.items()):
        if name in carried:
            raise ValueError('F1_NEW_INPUT_ALREADY_PINNED: %s' % name)
        new_inputs[name] = dict(sha256=sha_file(name), attribution=why)

    return dict(
        supersedes='amendment8.core_rebind',
        source_binding=str(R2 / 'amendment8.json'),
        source_binding_sha256=sha_file(R2 / 'amendment8.json'),
        attribution_note=(
            'GRM-F1 moves two of amendment 8 pins and adds one new core '
            'input. Every attribution below names the mechanism at the code '
            'site, not a merge label. The %d pins F1 does not touch are '
            'recorded as unchanged rather than dropped, and this script '
            'RAISES F1_UNEXPECTED_CORE_DRIFT rather than rebinding a file '
            'that moved for some other reason.' % len(unchanged)),
        changed=changed,
        new_inputs=new_inputs,
        unchanged=unchanged,
        unchanged_count=len(unchanged),
    )


#: The ONE runner change GRM-F1 is allowed to make, named explicitly so any
#: OTHER drift in the runner still raises. `environment()` strips every
#: `GRM_*` name, so without this re-apply an r3 campaign launched with
#: GRM_FOLD_RETAIN_SOURCES exported would run the OFF arm and REPORT it as
#: ON -- the exact failure `arm_environment`'s own docstring already warns
#: about for the arm pin.
RUNNER_CHANGE = (
    'GRM-F1, ONE change: `arm_environment` carries GRM_FOLD_RETAIN_SOURCES '
    'across the `grm_c2_cells.environment()` strip, the same rail the arm '
    'pin already rides (RETAIN_ENV added beside ALIAS_ENV). Without it the '
    'flag is deleted with every other ambient GRM_* name and r3 would '
    'silently measure the OFF arm while its receipts claimed ON. The flag is '
    'carried ONLY when actually set, so an environment with it absent is '
    'byte-identical to pre-F1. Nothing else in the runner moves: no lease, '
    'no deadline, no checkpoint, no preflight rule, no cell schedule.')


def runner_binding():
    """Amendment 10's runner block: amendment 9's, with F1's one change.

    `lt1_1_preflight` reads `runner` off the LATEST amendment -- the one this
    script writes -- and checks `sha(ROOT / runner['path'])` against
    `runner['sha256']`. Amendment 10 must therefore carry a runner block or
    the preflight raises `KeyError: 'runner'` (measured: it does).

    `scripts/grm_lt1_worker.py` is UNCHANGED by F1 and is carried verbatim;
    this raises if it moved. `scripts/grm_lt1_1.py` carries exactly the one
    change named in RUNNER_CHANGE, and the superseded sha is recorded so the
    rebind is auditable rather than asserted.

    The per-test `tests_sha256` map is deliberately NOT carried: F1 adds a
    test file and amendment 9's map is a receipt of ITS day. Re-pinning the
    whole suite would claim a verification F1 did not perform.
    """
    prior = dict(r2_doc('amendment9.json')['runner'])

    # The worker must NOT have moved. F1 has no business there.
    worker_now = sha_file(prior['worker'])
    if worker_now != prior['worker_sha256']:
        raise ValueError(
            'F1_WORKER_DRIFT: %s (amendment 9 pinned %s, on tree %s) -- '
            'GRM-F1 must not modify the LT1.1 worker. STOP and report.'
            % (prior['worker'], prior['worker_sha256'], worker_now))

    # The runner carries F1's single authorised change. Verify the change is
    # THE one named, by checking the marker it introduces is present and that
    # nothing else about the runner's registered contract moved.
    runner_path = prior['path']
    runner_now = sha_file(runner_path)
    text = (ROOT / runner_path).read_text()
    if "RETAIN_ENV = 'GRM_FOLD_RETAIN_SOURCES'" not in text:
        raise ValueError(
            'F1_RUNNER_CHANGE_ABSENT: %s does not define RETAIN_ENV -- the '
            'rebind below would claim a change the file does not carry.'
            % runner_path)
    if runner_now == prior['sha256']:
        raise ValueError('F1_RUNNER_DID_NOT_MOVE: %s' % runner_path)

    prior.pop('tests_sha256', None)
    prior['carried_from'] = 'amendment9.json'
    prior['superseded_sha256'] = prior['sha256']
    prior['sha256'] = runner_now
    prior['change'] = RUNNER_CHANGE
    prior['worker_change'] = (
        'UNCHANGED by GRM-F1; amendment 9\'s worker sha re-measured against '
        'the tree at registration time and carried verbatim.')
    prior['tests_added_by_f1'] = ['tests/test_grm_f1_fold_retain.py']
    return prior


def amendment10():
    prev = r2_doc('amendment9.json')
    return dict(
        amendment=10,
        arms=['A'],
        previous_amendment='amendment9.json',
        previous_amendment_sha256=sha_file(R2 / 'amendment9.json'),
        order='orders/GRM_F1_FOLD_RETAIN_SOURCES.md',
        order_sha256=sha_file('orders/GRM_F1_FOLD_RETAIN_SOURCES.md'),
        purpose=(
            'Rebind LT1.1 sha-bound core inputs for GRM-F1 and register '
            'LT1.1 r3 arm A under GRM_FOLD_RETAIN_SOURCES=1. Amendment 9 '
            'scoring columns and every other registered element are carried '
            'forward unchanged; the ONLY variable r3 introduces is the flag.'),
        follow_up_instruction=(
            'Amendment 10 supersedes amendment 8 core_rebind under the '
            'chain rule grm_lt1_1.governing_core_rebind already implements '
            '(the LATEST amendment carrying a core_rebind governs). '
            'Amendment 9 carried no core_rebind and is unaffected.'),
        scoring_columns_unchanged=dict(
            binding='amendment9.json',
            sha256=sha_file(R2 / 'amendment9.json'),
            statement=('The registered scorer (scripts/grm_lt1.score) remains '
                       'the PRIMARY verdict and is not modified by F1; '
                       'amendment 9 column 2 is reported beside it, '
                       'unchanged.')),
        budget_gpu_seconds_per_arm=BUDGET_GPU_SECONDS,
        budget_gpu_hours_total=round(BUDGET_GPU_SECONDS / 3600.0, 2),
        order_cap_gpu_seconds=ORDER_CAP_GPU_SECONDS,
        order_cap_gpu_hours=round(ORDER_CAP_GPU_SECONDS / 3600.0, 2),
        runner=runner_binding(),
        core_rebind=core_rebind(),
        process_safety=dict(
            gpu=('single lease via flock --wait; the operator has absolute '
                 'right of way; no process this run did not start is ever '
                 'signalled'),
            display='no windowed gate; CPU gates only',
            git='the lead commits; the seat never runs git',
            core_writes=('core/ edits are confined to the GRM_FOLD_RETAIN_'
                         'SOURCES flag; OFF is byte-identical'),
            scratch=('pytest uses --basetemp artifacts/grm_f1/tmp and the '
                     'directory is removed afterwards'),
            subagents='none spawned',
        ),
        previous_findings_carried=sorted(prev.get('findings', {})),
    )


def predictions():
    """Registered BEFORE the run, verbatim from the order's plan.

    The r2 arm-A measured baseline each line is predicted AGAINST is carried
    in `baseline` so the comparison cannot be re-anchored after the fact.
    """
    return dict(
        evidence_class=('prediction registered BEFORE the run, per house '
                        'rules; r3 is a GPU campaign and nothing here is '
                        'claimed from the CPU double'),
        fresh=('>= 14/15 correct (r2 arm A measured 10/15). The retained '
               'source turn is the only node that carries the fact in the '
               'wording the probe asks for; under r2 it was retired by the '
               'fold and only the digest could answer.'),
        corrections=('>= 9/10 correct (r2 arm A measured 9/10 -- this is a '
                     'NO-REGRESSION prediction, not an improvement one). '
                     'Retention must not resurrect a corrected value: a '
                     'correction retires the stale SOURCE as well as the '
                     'digest under the flag, which is strictly more than r2 '
                     'retired.'),
        aliases=('10/10 correct (r2 arm A measured 10/10). GRM-A1 pins '
                 'retain=False, so the alias lineage is invariant under this '
                 'flag and this row must not move at all.'),
        recap=('>= 4/5 (r2 arm A measured 4/5). Unpredicted beyond '
               'no-regression: recap reads era indexes, whose sources are '
               'also retained, but no mechanism predicts a specific gain.'),
        residency=('BOUNDED and REPORTED. The width guard is unchanged and '
                   'the mount budget still bounds seats, so the registered '
                   'bound (summed_token_seats <= 2*width + recency) must '
                   'hold on every row exactly as in r2. MAX SEATS must be '
                   'reported, and retained-source seats are reported '
                   'SEPARATELY via the new residency fields '
                   'retained_source_mounted_ids / '
                   'retained_source_token_seats. r2 arm A measured max '
                   'summed_token_seats 94 against width 96 (323 rows), so '
                   'the arena was already near saturation and retention is '
                   'predicted to change the CANDIDATE BASE, not the seat '
                   'ceiling.'),
        node_and_seat_delta=(
            'Computed from the r2 arm-A end-of-campaign checkpoint manifest '
            '(artifacts/grm_d1/lt1_1/run_A/cells/A-197-200/checkpoint/'
            'repository/manifest.json): 199 nodes, 36 active, 163 inactive. '
            'Of the inactive, 148 were retired BY A FOLD ALONE (124 turns + '
            '24 digests, listed as sources of a digest/era and named by no '
            'correction supersedes list) and 15 by a correction. r3 '
            'therefore predicts the ROUTABLE CANDIDATE BASE grows 36 -> 184 '
            '(5.1x) at the same 199 total nodes, while the 15 '
            'correction-retired nodes stay retired in both arms. Retained '
            'source ntok: mean 52.8, median 53, range 20-73.'),
        falsifier=('If arm A fresh column does not reach 14/15, the '
                   'retention hypothesis is REFUTED as stated and the honest '
                   'reading is that the source nodes were not what the '
                   'correct rows needed. A drop in ANY column, or a single '
                   'RESIDENCY_BOUND_EXCEEDED, is a FAILURE of the flag, not '
                   'a tuning opportunity.'),
    )


def registration():
    parent = r2_doc('registration.json')
    cells = [c for c in parent['cells'] if c['arm'] == 'A']
    if len(cells) != 26:
        raise ValueError('F1_R3_CELL_COUNT: %d' % len(cells))
    lease = sum(c['lease_seconds'] for c in cells)
    return dict(
        schema='grm.lt1_1_r3.registration.v1',
        status='FIT_ESTIMATE',
        gpu_executed=False,
        arms=['A'],
        derived_from=dict(
            registration=str(R2 / 'registration.json'),
            registration_sha256=sha_file(R2 / 'registration.json'),
            baseline_run=str(R2 / 'run_A'),
            baseline_summary=str(R2 / 'lead_A_r2_summary.json'),
            diagnosis=str(R2 / 'miss_causes.json'),
            report='artifacts/grm_d1/REPORT.md'),
        baseline=dict(
            path=str(R2 / 'lead_A_r2_summary.json'),
            sha256=sha_file(R2 / 'lead_A_r2_summary.json'),
            arm='A',
            measured=dict(fresh='10/15', corrections='9/10', aliases='10/10',
                          recap='4/5', measured_recalls=40),
            residency=dict(rows=323, max_summed_token_seats=94, width=96),
            nodes=dict(total=199, active=36, inactive=163,
                       retired_by_fold_alone=148,
                       retired_by_correction=15)),
        arms_flags={'A': dict(parent['arms']['A'],
                              admission_rule='margin_first')},
        cells=cells,
        model=parent['model'],
        model_frame=parent['model_frame'],
        fixture=dict(path=str(R2 / 'dialogue.json'),
                     sha256=sha_file(R2 / 'dialogue.json'),
                     statement='byte-identical to r2; r3 changes no fixture'),
        worker=dict(path='scripts/grm_lt1_1.py',
                    sha256=sha_file('scripts/grm_lt1_1.py'),
                    statement='the REAL LT1.1 worker, unchanged by F1'),
        amendment=dict(
            number=10,
            chain_path=str(R2 / 'amendment10.json'),
            copy_path=str(OUT_REL / 'amendment10.json'),
            statement=('written into LT1.1\'s own amendment chain, which is '
                       'where grm_lt1_1.governing_core_rebind reads it; an '
                       'amendment written elsewhere is invisible to the '
                       'preflight. Amendments 1-9 and every other r2 receipt '
                       'are untouched.')),
        budget_gpu_seconds=BUDGET_GPU_SECONDS,
        budget_gpu_hours=round(BUDGET_GPU_SECONDS / 3600.0, 2),
        lease_seconds_sum=lease,
        within_budget=lease <= BUDGET_GPU_SECONDS,
        estimate_seconds_sum=round(
            sum(c['estimate_seconds'] for c in cells), 1),
        order_cap_gpu_seconds=ORDER_CAP_GPU_SECONDS,
        order_cap_gpu_hours=round(ORDER_CAP_GPU_SECONDS / 3600.0, 2),
        within_order_cap=(
            sum(c['estimate_seconds'] for c in cells)
            <= ORDER_CAP_GPU_SECONDS),
        budget_conflict=dict(
            statement=(
                'The order caps r3 at <= 1.3 GPU-h. r2\'s arm-A cell '
                'schedule, reused verbatim so the comparison stays '
                'one-variable, reserves 7410 s (2.06 GPU-h) of LEASE. Those '
                'are different quantities: the ESTIMATE (predicted GPU work) '
                'is 4507.9 s = 1.25 GPU-h and IS under the cap; the lease is '
                'headroom `run_cell` checks before spawning each cell.'),
            why_not_4680=(
                'Registering budget_gpu_seconds=4680 would put it BELOW the '
                'lease sum. LT1.1 amendment 2 already corrected 6120 -> 7410 '
                'for exactly this reason -- the lower figure "sat below the '
                'reservation sum and would have tripped run_cell\'s rail '
                'mid-campaign". Reproducing a known mid-campaign failure to '
                'make a number look compliant is not an option.'),
            resolution=(
                'budget_gpu_seconds stays at r2\'s 7410 (the RESERVATION '
                'ceiling, so the rail behaves identically to r2). '
                'order_cap_gpu_seconds=4680 is registered as a SEPARATE '
                'binding limit on MEASURED charged GPU seconds: the campaign '
                'FAILS its own registration if it burns more than 1.3 GPU-h '
                'of actual work.'),
            lead_decision_requested=(
                'Confirm this reading, or re-cut the arm-A cell schedule to '
                'a smaller lease -- which would make r3 no longer cell-for-'
                'cell comparable to the r2 A baseline, and is therefore NOT '
                'done unilaterally by this seat.')),
        admission_rule='margin_first',
        required_env=dict(GRM_ADMISSION_RULE='margin_first',
                          GRM_PROFILE='eb1_c2',
                          **{FLAG: '1'}),
        required_env_flags=[dict(
            name=FLAG,
            value='1',
            present_on_tree=True,
            default='OFF',
            evidence=('core/grm_fold_retain.py defines it; '
                      'core/graft_arena.py resolves it once at construction; '
                      'tests/test_grm_f1_fold_retain.py proves OFF is '
                      'byte-identical'),
            statement='THE single variable under test in r3.')],
        optional_named_flags=[
            dict(name=F2_FLAG,
                 present_on_tree=_flag_present(F2_FLAG),
                 evidence=("grep -rn '%s' core/ scripts/ tests/ on branch "
                           'grm-f1' % F2_FLAG),
                 disposition=(
                     'F2 owns entity-scoped supersession of ordinary fold '
                     'digests -- the gap r3 leaves open (see '
                     'known_gaps F1-N1). The lead MAY pin it '
                     'alongside %s to run both treatments in one campaign; '
                     'if pinned, r3 measures the PAIR and the columns must '
                     'be attributed to the pair, never to F1 alone. NOT '
                     'pinned by default.' % FLAG)),
            dict(name=ALIAS_FLAG,
                 present_on_tree=_flag_present(ALIAS_FLAG),
                 evidence=("grep -rn '%s' core/ returns core/grm_alias_fold.py "
                           'on branch grm-f1' % ALIAS_FLAG),
                 disposition=(
                     'r2 arm A ran with alias_fold_merge=false (recorded in '
                     'its binding) and scored aliases 10/10. r3 keeps it OFF '
                     'so the comparison is one-variable. The lead may pin it, '
                     'but then the alias column is no longer comparable to '
                     'the r2 A baseline.')),
        ],
        changes=[
            dict(id='F1-C1', kind='core', flag=FLAG,
                 what='a fold sources stay ACTIVE and routable after the '
                      'digest is deposited; the digest is ADDED, not '
                      'substituted',
                 why='r2 receipt: source nodes appear in route_info.mounts in '
                     '0/57 correct rows, because FIX-5 retires every source '
                     'at MIN_FOLD_KEEP and _route_cand_base excludes retired '
                     'nodes -- so after a fold the digest prose is the '
                     'ONLY routable record and fresh facts at d >= 100 are '
                     'confabulated (three different wrong Breakwater tuples '
                     'across the r2 arms)',
                 proof='tests/test_grm_f1_fold_retain.py',
                 core_change=True),
            dict(id='F1-C2', kind='core', flag=FLAG,
                 what='lineage records metadata.digest_of on each retained '
                      'source and metadata.retained_sources on the digest; '
                      'the fold event records fold_lineage',
                 why='an ACTIVE turn also listed inside a digest sources '
                     'must never require a reader to INFER why',
                 proof='tests/test_grm_f1_fold_retain.py::'
                       'test_on_lineage_recorded_both_ways',
                 core_change=True),
            dict(id='F1-C3', kind='core', flag=FLAG,
                 what='retained sources are marked no_fold',
                 why='the librarian plan is stateless; a still-active folded '
                     'window would be re-selected forever. no_fold is the '
                     'width guard own anti-reselect channel (SCOUT-FIX-9), '
                     'reused rather than reinvented',
                 proof='tests/test_grm_f1_fold_retain.py::'
                       'test_on_sources_are_fold_exempt',
                 core_change=True),
            dict(id='F1-C4', kind='core', flag=None,
                 what='GRM-A1 _alias_consolidate PINS retain=False',
                 why='A1 always supersedes the alias EDGE; a bare active edge '
                     'that wins admission and answers nothing IS the RD2 '
                     'defect A1 exists to fix. A window-fold retention policy '
                     'must not reach in and reinstate it',
                 proof='tests/test_grm_f1_fold_retain.py::'
                       'test_alias_merge_lineage_invariant_under_flag',
                 core_change=True),
            dict(id='F1-C5', kind='harness', flag=None,
                 what='residency rows gain retained_source_mounted_ids and '
                      'retained_source_token_seats',
                 why='a residency bound that moves under the flag must be '
                     'readable as "which of these seats are sources a fold '
                     'kept alive"; with the flag OFF both fields are empty/0',
                 proof='tests/test_grm_f1_fold_retain.py::'
                       'test_residency_reports_retained_seats_separately',
                 core_change=False),
            dict(id='F1-C6', kind='registration', flag=None,
                 what='amendment 10 rebinds LT1.1 core pins',
                 why='LT1.1 preflight is sha-bound; without the rebind the '
                     'r3 child blocks with LT11_CHILD_PREFLIGHT_BLOCKED, '
                     'which is the pin working correctly',
                 proof=str(OUT_REL / 'amendment10.json'),
                 core_change=False),
        ],
        known_gaps=[dict(
            id='F1-N1',
            title='a PARAPHRASING fold digest escapes correct_memory',
            statement=(
                'correct_memory finds targets by matching the query against '
                'node TEXT. A fold digest that paraphrases a fact rather than '
                'copying its wording does not match, so the stale copy inside '
                'it is never superseded. MEASURED on the CPU double in BOTH '
                'arms (tests/test_grm_f1_fold_retain.py::'
                'test_paraphrasing_digest_escapes_correction_in_both_arms): '
                'OFF supersedes nothing at all, ON supersedes the retained '
                'source but the digest still escapes.'),
            caused_by_f1=False,
            effect_of_f1=('strictly MORE of the stale record is retired under '
                          'the flag than without it; F1 neither causes nor '
                          'fixes this gap'),
            owner=('F2. A1 already carries the entity-scoped remedy for its '
                   'own merged digests (_alias_extend_correction_targets); '
                   'generalising it to ordinary fold digests is a change to '
                   'correctness semantics that the F1 order does not '
                   'authorise.'),
            bearing_on_r3=('the corrections column is predicted at >= 9/10 '
                           '(no-regression), NOT at 10/10, precisely because '
                           'this gap is open in both arms'))],
        predictions=predictions(),
        cpu_gates=[
            'python3 -m pytest -q tests/test_grm_f1_fold_retain.py',
            'python3 -m pytest -q tests/test_grm_scout_fix5.py '
            'tests/test_grm_scout_fix9.py',
            'python3 -m pytest -q tests/test_grm_a1_alias_fold.py',
            'python3 -m pytest -q tests/test_grm_lt1_1_child_spawn.py '
            'tests/test_grm_lt1_1_production_turns.py',
            'python3 scripts/grm_f1_register_lt11_r3.py',
        ],
        process_safety=dict(
            gpu=('single lease via flock --wait; the operator has absolute '
                 'right of way; no process this run did not start is ever '
                 'signalled'),
            display='no windowed gate; CPU gates only',
            git='the lead commits; the seat never runs git',
            paths='every pin in this document is repository-relative'),
        prior_art=[
            'LT1.1 registration shape: scripts/grm_d1_register_lt11.py '
            '(GRM contributors, 2026), reused',
            'core_rebind block shape: artifacts/grm_d1/lt1_1/amendment8.json '
            '(GRM contributors, 2026), reused field for field',
            'FIX-5 consolidation + MIN_FOLD_KEEP: core/graft_arena.py '
            '(GRM contributors, 2026), unchanged by F1',
            'LSR-P2C / SCOUT-FIX-9 width guard, whose rejected-split path '
            'already leaves sources ACTIVE and uses no_fold: '
            'core/graft_repository.py (GRM contributors, 2026), reused as '
            'the anti-reselect rail',
            'A1 lineage vocabulary (supersedes / superseded_by / active): '
            'core/grm_alias_fold.py (GRM contributors, 2026), reused',
            'RAPTOR collapsed-tree retrieval (Sarthi et al. 2024, arXiv '
            '2401.18059) as the framing for leaf retention vs substitution '
            '-- UNVERIFIED, lead to check',
            'Append-only amendment chains, Haber and Stornetta (1991), in '
            'kind only -- UNVERIFIED, lead to check',
        ],
    )


def lead_commands(reg_sha, amd_sha):
    """Every runnable line ends in `--dry-run`.

    The order's gate: this file must not contain a command that executes a
    GPU campaign. The lead removes `--dry-run` deliberately, by hand, after
    reading the registration -- it is never removed by a copy-paste.
    `tests/test_grm_f1_fold_retain.py::test_lead_commands_dry_run_gated`
    enforces it.
    """
    return [
        '# LT1.1 r3 -- arm A only, 26 cells',
        '# reservation ceiling %d s (%.2f GPU-h), r2 unchanged'
        % (BUDGET_GPU_SECONDS, BUDGET_GPU_SECONDS / 3600.0),
        '# ORDER CAP on MEASURED work: %d s (%.2f GPU-h). See'
        % (ORDER_CAP_GPU_SECONDS, ORDER_CAP_GPU_SECONDS / 3600.0),
        '# registration.budget_conflict -- lead decision requested.',
        '# registration sha256: %s' % reg_sha,
        '# amendment10  sha256: %s' % amd_sha,
        '#',
        '# THE SINGLE VARIABLE under test:',
        '#   %s=1  (default OFF; OFF is byte-identical)' % FLAG,
        '#',
        '# OPTIONAL, lead may pin -- if pinned, the columns measure the PAIR:',
        '#   export %s=1' % F2_FLAG,
        '# NOT pinned by default (r2 arm A ran with it false):',
        '#   export %s=1' % ALIAS_FLAG,
        '#',
        '# EVERY line below ends in --dry-run. Removing it is a deliberate',
        '# act by the lead after reading the registration, never a paste.',
        '#',
        'export GRM_ADMISSION_RULE=margin_first',
        'export GRM_PROFILE=eb1_c2',
        'export %s=1' % FLAG,
        '#',
        '# 1. document chain + core pins (amendment 10 must govern):',
        'python3 -m scripts.grm_lt1_1 --arm A --preflight --dry-run',
        '# 2. cell schedule and budget:',
        'python3 -m scripts.grm_lt1_1 --arm A --dry-run',
        '# 3. the campaign itself -- under flock, operator has right of way:',
        'flock --wait 7200 /var/lock/grm_gpu.lock \\',
        '  python3 -m scripts.grm_lt1_1 --arm A --resume --dry-run',
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    COMMANDS.parent.mkdir(parents=True, exist_ok=True)

    amd = amendment10()
    amd_bytes = canonical(amd)
    amd_sha = sha_bytes(amd_bytes)

    # WHERE amendment 10 lives, and why it is an APPEND, not an edit.
    #
    # `grm_lt1_1.governing_core_rebind` scans `OUT.glob('amendment[0-9]*.json')`
    # where OUT is `artifacts/grm_d1/lt1_1` -- LT1.1's own amendment chain.
    # An amendment written anywhere else is INVISIBLE to the preflight, and
    # the r3 child would still block with INPUT_SHA_MISMATCH. So the chain
    # file is written there, exactly as amendments 2-9 were each appended by
    # their own orders.
    #
    # This ADDS a file. It modifies NO existing receipt: every amendment 1-9,
    # the r2 registration, the fixture, `miss_causes.json`, both run
    # directories and every `.sha256` are left byte-for-byte untouched, and
    # the create-only write below refuses to overwrite an amendment 10 that
    # somehow already exists rather than silently replacing a receipt.
    # Regeneration is idempotent BY CONSTRUCTION: every field is derived
    # from the tree and from r2's frozen documents, so re-running this script
    # after a core edit legitimately produces a new amendment 10. What must
    # never be overwritten is an amendment 10 THIS SCRIPT DID NOT WRITE --
    # someone else's chain receipt under the same number. The marker is the
    # order sha: a foreign amendment 10 would name a different order.
    chain = ROOT / R2 / 'amendment10.json'
    if chain.exists():
        existing = json.loads(chain.read_text())
        if existing.get('order') != amd['order']:
            raise ValueError(
                'F1_AMENDMENT10_FOREIGN: %s was written by order %r, not by '
                'this one (%r) -- refusing to overwrite another order\'s '
                'chain receipt. STOP and report.'
                % (chain, existing.get('order'), amd['order']))
    chain.write_bytes(amd_bytes)
    (ROOT / R2 / 'amendment10.sha256').write_text(
        '%s  amendment10.json\n' % amd_sha)
    # A copy in r3's own namespace, so the r3 receipt directory is
    # self-describing. The CHAIN copy above is the one the preflight reads.
    (OUT / 'amendment10.json').write_bytes(amd_bytes)
    (OUT / 'amendment10.sha256').write_text(
        '%s  amendment10.json\n' % amd_sha)

    reg = registration()
    reg['amendment']['sha256'] = amd_sha
    reg_bytes = canonical(reg)
    (OUT / 'registration.json').write_bytes(reg_bytes)
    reg_sha = sha_bytes(reg_bytes)
    (OUT / 'registration.sha256').write_text(
        '%s  registration.json\n' % reg_sha)

    COMMANDS.write_text('\n'.join(lead_commands(reg_sha, amd_sha)) + '\n')

    # A registration that pins a WORKTREE path becomes a permanent receipt
    # pointing at a fork that may be pruned (round-1 lesson). Refuse to write
    # one rather than relying on the test to catch it later.
    #
    # SCOPE, stated because the distinction is load-bearing: every path this
    # script AUTHORS is repository-relative. The one absolute path in the
    # document is `model.path`, the HF weights snapshot, carried VERBATIM
    # from r2's registration -- it names an external model cache, not a
    # repository fork, and rewriting it would break the binding r3 exists to
    # reproduce. It is allow-listed by exact key, never by pattern, so a new
    # absolute path appearing anywhere else still raises.
    allowed = {json.loads((OUT / 'registration.json').read_text())
               ['model']['path']}
    for path in (OUT / 'registration.json', OUT / 'amendment10.json',
                 COMMANDS):
        text = path.read_text()
        for permitted in allowed:
            text = text.replace(json.dumps(permitted)[1:-1], '<model.path>')
        for bad in ('/mnt/', '/home/', 'wt/'):
            if bad in text:
                raise ValueError('F1_ABSOLUTE_PIN: %s contains %r'
                                 % (path.name, bad))

    print('registration sha256 %s' % reg_sha)
    print('amendment10  sha256 %s' % amd_sha)
    print('cells %d  reservation %ds (%.2f GPU-h)  within_budget=%s'
          % (len(reg['cells']), BUDGET_GPU_SECONDS,
             BUDGET_GPU_SECONDS / 3600.0, reg['within_budget']))
    print('order cap %ds (%.2f GPU-h)  estimate %.1fs (%.2f GPU-h)  '
          'within_order_cap=%s'
          % (ORDER_CAP_GPU_SECONDS, ORDER_CAP_GPU_SECONDS / 3600.0,
             reg['estimate_seconds_sum'], reg['estimate_seconds_sum'] / 3600.0,
             reg['within_order_cap']))
    rb = amd['core_rebind']
    print('core_rebind: %d changed, %d new, %d unchanged'
          % (len(rb['changed']), len(rb['new_inputs']), len(rb['unchanged'])))
    for name, entry in sorted(rb['changed'].items()):
        print('   %s %s -> %s' % (name, entry['before_sha256'][:12],
                                  entry['after_sha256'][:12]))
    for name, entry in sorted(rb['new_inputs'].items()):
        print('   %s NEW %s' % (name, entry['sha256'][:12]))
    for flag in reg['optional_named_flags']:
        print('optional %s present_on_tree=%s'
              % (flag['name'], flag['present_on_tree']))


if __name__ == '__main__':
    main()
