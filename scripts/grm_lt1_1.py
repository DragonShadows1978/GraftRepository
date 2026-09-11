#!/usr/bin/env python3
"""LT1.1 runner — drives the EXISTING LT1 worker machinery, one arm per run.

WHY THIS EXISTS. The LT1.1 registration, its amendment 1 and its
`dialogue.json` described a campaign that nothing could execute:
`scripts/grm_lt1.py` has no `--run/--arm/--registration/--amendment/--fixture/
--out`; it reads fixed module paths and drives a single hard-wired campaign.
A registration is not runnable until a worker executes it. This module is that
worker entry point.

PARAMETERIZED, NOT FORKED. The cell/lease/checkpoint logic is NOT copied. The
reusable core already exists and is already parameterized:

    scripts/grm_lt1_worker.execute(cell, directory, registration, loader,
                                   *, run=..., deadline=..., fake=...)

`execute` takes its cell, output directory, registration, model loader, run
root and fake flag as arguments -- `scripts/grm_lt1_worker_cpu.py` already
reuses it exactly this way. Only THREE things in that module are bound to
module-level LT1 state rather than passed in:

    lt.FIX      the fixture path `execute` reads for turns/probes/decisions
    lt.binding  the SHA binding stamped into every receipt (via worker.bind)
    RUN         the campaign root -- already a keyword argument

So this runner redirects exactly those three and calls the existing functions.
`run_cell`'s lease/flock/nvidia-smi/charge logic and `pending`'s
resume-and-verify logic are reused through the same redirection, not
reimplemented. The one thing genuinely new here is the A/A+ arm split: the
alias flag pinned AFTER `environment(flags)` and read back (R1 `pin_rule`),
then restored -- LT1 has no concept of two variants of one arm.

WHAT IS NOT REUSED AND WHY. `grm_lt1_worker.run_cell` re-enters the worker as
`-m scripts.grm_lt1_worker --worker`, which would load LT1's module state in
the child. The child is therefore re-entered through THIS module
(`--worker`), so it inherits the same redirection. That is a different entry
point, not a second copy of the lease logic.

ARMS. `A` and `A+` run the same 26 cells over the same frozen conversation and
differ by exactly one environment variable. Their receipts go to different
directories, so either arm may be run, interrupted and resumed alone.

Prior art:
  * `scripts/grm_lt1_worker.py` (execute / run_cell / pending / worker) and
    `scripts/grm_lt1.py` (verify / preflight / score / create / read) — GRM
    contributors, 2026. REUSED by redirection; no cell, lease, checkpoint,
    accounting or scoring logic is reimplemented here.
  * `scripts/grm_lt1_worker_cpu.py` (GRM contributors, 2026) — the pattern of
    calling `execute` with a CPU loader, an arbitrary run root and
    `fake=True`. Taken verbatim as the fake-path shape.
  * R1 `pin_rule` (`scripts/grm_r1_replay.py:242`, GRM contributors 2026) —
    pin the arm variable AFTER `environment(flags)` strips ambient `GRM_*`,
    then read back and assert. Taken; extended to a second variable and to
    restoring the prior value on exit.
  * C2 `environment(flags)` (`scripts/grm_c2_cells.py:47`) — the ambient strip.
  * Mine: the arm-variant split over one registration, the three-seam
    redirection, and the `--arm`-scoped resumable campaign root.
  * No prior art known to me for this exact composition.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_lt1 as lt                      # noqa: E402
from scripts import grm_lt1_worker as worker           # noqa: E402

OUT = ROOT / 'artifacts/grm_d1/lt1_1'
REGISTRATION = OUT / 'registration.json'
AMENDMENT1 = OUT / 'amendment1.json'
FIXTURE = OUT / 'dialogue.json'

ALIAS_ENV = 'GRM_ALIAS_FOLD_MERGE'
RULE_ENV = 'GRM_ADMISSION_RULE'
ARMS = ('A', 'A+')

#: Arm -> alias flag value (None = unset). The ONLY difference between arms.
ARM_ALIAS = {'A': None, 'A+': '1'}

#: LT1.1 inherits LT1's restart sentinels; the base registration is a delta
#: over LT1's and does not restate them.
RESTART_SENTINELS = ('recall_1_10', 'recall_4_10')


def out_dir(arm):
    """Per-arm campaign root. Distinct roots are what make arms resumable.

    A leased child inherits the parent's root through `GRM_LT1_1_RUN`, so it
    writes into the exact directory the parent reserved.
    """
    inherited = os.environ.get(RUN_ENV)
    if inherited:
        return Path(inherited)
    return OUT / ('run_A' if arm == 'A' else 'run_Aplus')


def sha(path):
    return lt.sha(Path(path))


def _relative(path):
    """Repo-relative when it can be; absolute otherwise (gate temp dirs)."""
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------- arm state

#: The arm this process is running. Set by `pinned_arm` / `lt1_1_seams` and
#: read by `binding`, because LT1's callers pass a BACKEND LABEL, not an arm,
#: and the seam must not change their signature. `GRM_LT1_1_ARM` carries it
#: across the `--worker` / `--fake-cell` subprocess boundary the same way
#: `GRM_LT1_LEASE_PARENT` carries the lease-parent contract.
ARM_ENV = 'GRM_LT1_1_ARM'
#: Campaign root, carried into the leased child so it writes into the exact
#: directory the parent reserved instead of recomputing a default.
RUN_ENV = 'GRM_LT1_1_RUN'
_ARM = None


def current_arm(fallback=None):
    """The pinned arm: explicit state, then the environment, then a refusal.

    Never guesses. A binding computed under the wrong arm would stamp the
    wrong `alias_fold_merge` into a receipt, which is exactly the kind of
    silent mislabelling the arm split exists to prevent.
    """
    if _ARM is not None:
        return _ARM
    from_env = os.environ.get(ARM_ENV)
    if from_env in ARMS:
        return from_env
    if fallback in ARMS:
        # A caller that named an arm outright (`binding('A+')`) has said which
        # campaign it means. LT1's own callers pass 'CPU'/'GPU', which are not
        # arms and so never reach this branch.
        return fallback
    raise ValueError('LT11_ARM_NOT_PINNED: binding() needs a pinned arm; '
                     'call inside pinned_arm()/lt1_1_seams() or set %s'
                     % ARM_ENV)


@contextlib.contextmanager
def _arm_state(arm):
    """Hold the pinned arm for this process and its children."""
    global _ARM
    previous, previous_env = _ARM, os.environ.get(ARM_ENV)
    try:
        _ARM = arm
        os.environ[ARM_ENV] = arm
        yield arm
    finally:
        _ARM = previous
        if previous_env is None:
            os.environ.pop(ARM_ENV, None)
        else:
            os.environ[ARM_ENV] = previous_env


# ------------------------------------------------------------------ binding

def binding(label):
    """The SHA binding stamped into every LT1.1 receipt.

    SIGNATURE CONTRACT. This replaces `lt.binding` while the seams are
    redirected, so it MUST accept everything LT1's callers pass and treat it
    the way LT1 does. LT1's `binding(arm)` is label-agnostic: it echoes its
    argument into `arm=` and never interprets it. Its callers pass TWO
    different kinds of token:

        worker.bind(cell['arm'])     -> an ARM name   ('A')
        apply4 / preflight / summary -> a BACKEND label ('CPU')

    An earlier version of this function indexed `ARM_ALIAS[arm]`, which only
    accepts arm names, so `lt.binding('CPU')` from
    `grm_lt1_amendment4.apply` raised `KeyError: 'CPU'` and killed
    `--resume` before it reached the lease. The fake and dry-run paths never
    traversed `apply4`, so nothing caught it.

    The fix: the parameter keeps LT1's meaning (an opaque label, echoed), and
    the ARM -- the thing that actually selects the alias-fold variant -- is
    read from the runner's pinned state, which is where it truly lives.
    """
    amendment = json.loads(AMENDMENT1.read_text())
    core = {n: c['after_sha256']
            for n, c in amendment['core_rebind']['changed'].items()}
    core.update({n: e['sha256']
                 for n, e in amendment['core_rebind']['new_inputs'].items()})
    arm = current_arm(label)
    return dict(label=label,
                arm=label if label in ARMS else arm,
                campaign_arm=arm,
                registration_sha256=sha(REGISTRATION),
                amendment1_sha256=sha(AMENDMENT1),
                fixture_sha256=sha(FIXTURE),
                admission_rule='margin_first',
                alias_fold_merge=(ARM_ALIAS[arm] is not None),
                rebound_core_shas=core)


def registration(arm):
    """The registration `execute` consumes, assembled for one arm."""
    if arm not in ARMS:
        raise ValueError('LT11_UNKNOWN_ARM: ' + str(arm))
    if sha(AMENDMENT1) != (OUT / 'amendment1.sha256').read_text().split()[0]:
        raise ValueError('LT11_AMENDMENT1_SHA_MISMATCH')
    amendment = json.loads(AMENDMENT1.read_text())
    if amendment['registration_sha256'] != sha(REGISTRATION):
        raise ValueError('LT11_AMENDMENT1_CHAIN_MISMATCH')
    base = json.loads(REGISTRATION.read_text())
    spec = amendment['arms'][arm]
    if sha(FIXTURE) != spec['fixture']['sha256']:
        raise ValueError('LT11_FIXTURE_SHA_MISMATCH')

    value = dict(base)
    # `execute` indexes `registration['arms'][cell['arm']]`, and every cell
    # carries arm "A" in both variants (same schedule). Expose the running
    # arm's flags under that key rather than renaming 26 cells.
    value['arms'] = {'A': dict(base['arms']['A'], flags=spec['flags'],
                               admission_rule='margin_first')}
    value['cells'] = spec['cells']
    value['effective_admission_rule'] = 'margin_first'
    value['restart_sentinels'] = list(RESTART_SENTINELS)
    value['lt1_1'] = dict(arm=arm, alias_fold_merge=ARM_ALIAS[arm] is not None,
                          out_dir=_relative(out_dir(arm)),
                          amendment1_sha256=sha(AMENDMENT1))
    return value


# ------------------------------------------------------------- the arm pin

def arm_environment(arm, flags):
    """`environment(flags)` then the arm pin, in that order (R1 contract).

    `environment(flags)` deletes every ambient `GRM_*`, so the pin MUST come
    after it or the arm silently runs the default.
    """
    from scripts.grm_c2_cells import environment
    env = environment(flags)
    env[RULE_ENV] = 'margin_first'
    env.pop(ALIAS_ENV, None)
    if ARM_ALIAS[arm] is not None:
        env[ALIAS_ENV] = ARM_ALIAS[arm]
    return env


@contextlib.contextmanager
def pinned_arm(arm):
    """Pin the arm in THIS process, read back, and restore on exit.

    Prior art: R1 `pin_rule` (scripts/grm_r1_replay.py:242). The readback is
    the point: a pin that did not take must be RED, not silent.
    """
    from core.grm_admission import admission_rule
    from core.grm_alias_fold import alias_fold_enabled
    previous = {k: os.environ.get(k) for k in (RULE_ENV, ALIAS_ENV)}
    try:
        os.environ[RULE_ENV] = 'margin_first'
        os.environ.pop(ALIAS_ENV, None)
        if ARM_ALIAS[arm] is not None:
            os.environ[ALIAS_ENV] = ARM_ALIAS[arm]
        observed_rule = admission_rule()
        if observed_rule != 'margin_first':
            raise ValueError('LT11_RULE_PIN_FAILED: ' + observed_rule)
        observed_alias = alias_fold_enabled()
        if observed_alias is not (ARM_ALIAS[arm] is not None):
            raise ValueError('LT11_ALIAS_PIN_FAILED: arm=%s observed=%s'
                             % (arm, observed_alias))
        # Hold the arm for `binding()`, which cannot take it as a parameter
        # without breaking LT1's signature.
        with _arm_state(arm):
            yield dict(arm=arm, admission_rule=observed_rule,
                       alias_fold_merge=observed_alias)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def lt1_1_seams(arm):
    """Redirect the module seams `execute` and `run_cell` read from LT1 state.

    Everything else in `grm_lt1_worker` -- cells, leases, checkpoints,
    accounting -- is used unchanged through these.

    SIX seams. Follow-up 6 added `await_idle`, so a card another process is
    still using becomes a bounded WAIT rather than a RED cell -- a busy card
    is not a cell failure. Follow-up 5 added the two before it, after a
    lead-run
    `--resume` put cell A-001-008 RED with `WORKER_EXIT_1`:

        run_cell -> Popen([... '-m', 'scripts.grm_lt1_worker', '--worker', id])
                 -> that module's __main__ -> lt.verify()
                 -> ValueError: INPUT_SHA_MISMATCH: core/graft_arena.py

    The CHILD re-ran LT1's own chain verification -- the exact gate the lead's
    ruling removed from the parent. Three reasons nothing caught it:

      * `--dry-lease` stops at the lease boundary, before `run_cell` spawns;
      * the `--fake` proof used `--fake-cell`, a path that never enters
        `run_cell` at all;
      * my own follow-up-2 docstring ASSERTED the child came back through this
        module. It did not. That claim was wrong and this is the correction.

    `lt.verify()` is reachable from the worker module at three points:
    `worker()` line 188, `resume()` line 273, and `__main__` line 289. The
    runner never calls `worker.resume`; the other two are covered by
    redirecting `worker.worker` (which owns line 188 and is what the child
    entry point invokes) and `worker.spawn_argv`, so the child comes back
    through `scripts/grm_lt1_1.py --worker`.
    """
    saved = (lt.FIX, worker.bind, worker.RUN,
             getattr(worker, 'worker', None),
             getattr(worker, 'spawn_argv', None),
             getattr(worker, 'spawn_env', None),
             getattr(worker, 'await_idle', None),
             getattr(worker, 'deposit_turn', None),
             getattr(worker, 'recap_probe_turn', None))
    try:
        lt.FIX = FIXTURE
        # Redirect `worker.bind`, NOT `lt.binding`.
        #
        # These look interchangeable (`worker.bind` is a one-line delegate to
        # `lt.binding`) and they are not. They serve opposite masters:
        #
        #   worker.bind(cell['arm'])  stamps LT1.1 RECEIPTS  -> must be ours
        #   lt.binding('CPU')         is how LT1 VALIDATES ITS OWN chain, in
        #                             verify/preflight/apply4 -> must stay LT1's
        #
        # Replacing `lt.binding` made `grm_lt1_amendment4.apply` compare its
        # recorded `protocol_binding` against an LT1.1-shaped dict, so
        # `--resume` died on AMENDMENT4_PROTOCOL_MISMATCH (and, before the
        # signature fix, on KeyError: 'CPU'). The narrower seam leaves LT1's
        # self-validation untouched and still stamps every receipt ours.
        worker.bind = binding
        worker.RUN = out_dir(arm)
        # The child entry point and the argv that reaches it.
        worker.worker = lt1_1_worker
        worker.spawn_argv = spawn_argv
        worker.spawn_env = spawn_env
        worker.await_idle = await_idle
        # Production turn semantics: supersede through the correction path,
        # every deposit through the funnel where A1's fold-merge lives.
        worker.deposit_turn = deposit_turn
        worker.recap_probe_turn = recap_probe_turn
        # The arm must be resolvable for the whole window: `binding` cannot
        # take it as a parameter without breaking LT1's signature.
        with _arm_state(arm):
            yield
    finally:
        lt.FIX, worker.bind, worker.RUN = saved[:3]
        for name, value in (('worker', saved[3]), ('spawn_argv', saved[4]),
                            ('spawn_env', saved[5]),
                            ('await_idle', saved[6]),
                            ('deposit_turn', saved[7]),
                            ('recap_probe_turn', saved[8])):
            if value is None:
                if hasattr(worker, name):
                    delattr(worker, name)
            else:
                setattr(worker, name, value)


# ------------------------------------------------------------------- loaders

def deposit_turn(repo, event, state, turn):
    """One non-probe fixture turn, through PRODUCTION semantics.

    WHY THIS EXISTS. LT1's worker fed every non-probe kind as ordinary prose:

        idx = a.feed(e2e.harmony_turn(event['user'], event['assistant']))

    with the comment "User corrections are ordinary prose, not hidden
    supersede calls". That is correct for LT1, which replays a frozen
    transcript. It is wrong for LT1.1, whose whole point is that the
    `supersede` and `alias` kinds carry production semantics. Run under that
    worker, LT1.1 arm A+ reproduced LT1's numbers byte-for-byte on
    2026-09-11 -- fresh 14/15, corrections 5/10, aliases 5/10 -- because:

      * the 15 `supersede` turns were deposited as prose, so nothing was
        retired and the stale node stayed in the candidate base: exactly the
        fixture-lineage trap D1 diagnosed, re-created one layer down;
      * no `alias_fold` decision appears anywhere in those receipts, because
        A1's `alias_fold_pass` runs inside `runtime._finish_turn_event` and
        `arena.feed()` never calls it. The flag was pinned and the mechanism
        never ran.

    WHAT THIS DOES.

      `supersede` -> `repo.apply_memory_command(event['correction_command'])`,
        the production correction path (`scripts/grm_e2e_session.py:2590-2597`
        runs exactly this for `kind == "supersede"`). `correct_memory` retires
        the matched node, sets `superseded_by`, and bumps the route epoch, so
        the old value leaves `_route_cand_base`.

      every deposit -> `arena.feed()` AND THEN
        `repo.runtime._finish_turn_event('chat', before, autosave=True)`,
        the funnel every production deposit passes through, exactly as
        `scripts/grm_chat.py:240` does it. That funnel runs the width guard,
        `_alias_fold_deposits` (A1) and `_librarian`, and is what assigns
        `native_node_id`. Under A the alias fold is byte-inert (the flag is
        off, `_alias_fold_jobs` returns `()` from its first line); under A+ it
        merges.

    FAIRNESS. Arms A and A+ now share THIS worker, so A-vs-A+ is a fair
    same-worker contrast: one pinned environment variable is the only
    difference. LT1's numbers are the PARENT BASELINE -- a different worker on
    a different fixture -- and must never be read as a same-worker control.

    Prior art:
      * The production turn funnel and the reason a battery may stop at
        `feed()` while a product may not: `scripts/grm_chat.py:224-245`
        (GRM-P1, GRM contributors 2026), taken with its finding.
      * The supersede turn: `scripts/grm_e2e_session.py:2590-2597`
        (GRM contributors, 2026), reused unchanged.
      * `_finish_turn_event` -> `_alias_fold_deposits` / `_librarian`:
        `core/grm_runtime.py:95-111` (A1 + GRM contributors, 2026). Read, not
        modified.
      * Mine: routing a frozen fixture's kinds to these existing paths, and
        the A/A+ fairness statement above.
      * No prior art known to me for this exact composition.
    """
    from scripts import grm_e2e_session as e2e
    arena = repo.arena
    before = repo._snapshot_state()
    kind = event.get('kind')

    if kind == 'supersede':
        command = event.get('correction_command')
        if not command:
            raise ValueError('LT11_SUPERSEDE_WITHOUT_COMMAND: turn %s' % turn)
        # The production correction path. It deposits the replacement itself
        # and retires the stale node, so there is no separate feed() here.
        repo.apply_memory_command(command)
        repo.runtime._finish_turn_event('chat', before, autosave=True)
        new = [i for i, g in enumerate(arena.grafts)
               if not g.get('retired') and i >= len(before)]
        idx = new[-1] if new else None
        if idx is not None:
            state['turn_nodes'][str(turn)] = idx
        return idx

    idx = int(arena.feed(e2e.harmony_turn(event['user'], event['assistant'])))
    arena.grafts[idx]['kind'] = 'turn'
    state['turn_nodes'][str(turn)] = idx
    # THEN the production funnel: width guard, alias fold (A1), librarian,
    # native publication. Skipping it is what made the A+ arm inert.
    repo.runtime._finish_turn_event('chat', before, autosave=True)
    return idx


def recap_probe_turn(repo, event, directory, binding_value):
    """One `recap_probe` turn: the probe path, scored, no assistant field.

    LT1 has no `recap_probe` kind, so the old worker fell through to the
    deposit branch and raised `KeyError: 'assistant'`, killing cell
    A-189-196. These are questions, not turns: they are asked, scored with the
    existing value-span scorer, and never deposited.

    Prior art: the probe branch of `scripts/grm_lt1_worker.execute`
    (GRM contributors, 2026) and the C5 arm S scorer via `grm_lt1.score`,
    both reused. Mine: the row shape carrying `kind='recap_probe'`.
    """
    from scripts import grm_e2e_session as e2e
    from scripts.grm_c7_run import emit
    fixture = json.loads(FIXTURE.read_text())
    probe = next(q for q in fixture['recap_probes']
                 if q['id'] == event['recap_probe_id'])
    answer, info = e2e._probe_ladder_chat(repo, probe['question'], topk=3,
                                          ngen=32, max_trips=1,
                                          defer_memory=True)
    emit(directory / 'probes.jsonl',
         dict(probe_id=probe['id'], turn=event['turn'], kind='recap_probe',
              question=probe['question'], expected=probe['expected'],
              targets=probe['targets'],
              memory=dict(answer=str(answer),
                          score=lt.score(answer, probe['expected']),
                          route_info=info, admission_rule='margin_first'),
              binding=binding_value, admission_rule='margin_first'))
    return 1


#: Framebuffer ceiling below which the card counts as free to start, and the
#: bound on how long we will wait for somebody else to finish. A busy card is
#: a resource another process holds, not a failure of our cell.
IDLE_LIMIT_MIB = 512
IDLE_WAIT_SECONDS = 900
IDLE_POLL_SECONDS = 15


def device_snapshot():
    """Total framebuffer used, plus the compute list, as a receipt.

    Prior art: R1 `device_snapshot` (`scripts/grm_r1_replay.py:352`, GRM
    contributors 2026) and, through it, FIX-8
    `grm_scout_fix8_resume.parse_memory` and the NVIDIA nvidia-smi XML
    framebuffer report. TAKEN verbatim: read TOTAL framebuffer used and never
    infer an idle card from an empty compute list -- R4's OOM happened with
    ~3,144 MiB held by a display-side program that listed no compute process.
    OURS: nothing; this is a straight reuse.
    """
    import subprocess
    import time
    import xml.etree.ElementTree as ET
    value = dict(time_unix=time.time(), status='ERROR',
                 scope='point sample, not peak; device total includes '
                       'unattributed memory such as display-side programs')
    try:
        probe = subprocess.run(['nvidia-smi', '-q', '-x'],
                               capture_output=True, text=True, timeout=60)
    except Exception as exc:                                # pragma: no cover
        value['error'] = '%s: %s' % (type(exc).__name__, exc)
        return value
    if probe.returncode:
        value['error'] = 'nvidia-smi exit %d' % probe.returncode
        return value
    try:
        root = ET.fromstring(probe.stdout)
        gpu = root.find('gpu')
        fb = gpu.find('fb_memory_usage')

        def mib(node, field):
            return int(str(node.find(field).text).split()[0])
        pids = [int(p.find('pid').text)
                for p in gpu.findall('./processes/process_info')]
        value.update(status='OK',
                     memory_used_mib=mib(fb, 'used'),
                     memory_free_mib=mib(fb, 'free'),
                     memory_total_mib=mib(fb, 'total'),
                     other_process_pids=pids,
                     compute_list_empty=not pids)
    except Exception as exc:                                # pragma: no cover
        value['error'] = 'parse: %s: %s' % (type(exc).__name__, exc)
    return value


def idle_gate(snapshot, limit=None):
    """Is the card free enough to start? Returns `(ok, reason)`.

    Keys on TOTAL used, never on the compute-process list. Declines only:
    it never signals, kills or waits on another process.

    Prior art: R1 `idle_gate` (`scripts/grm_r1_replay.py:420`, GRM
    contributors 2026), taken unchanged.
    """
    limit = IDLE_LIMIT_MIB if limit is None else int(limit)
    if snapshot.get('status') != 'OK':
        return False, 'DEVICE_PROBE_FAILED: %s' % snapshot.get('error',
                                                               'unknown')
    used = int(snapshot['memory_used_mib'])
    if used > limit:
        return False, ('DEVICE_BUSY: memory.used=%d MiB > %d MiB (free=%s MiB; '
                       'compute_list_empty=%s; other_pids=%s)'
                       % (used, limit, snapshot['memory_free_mib'],
                          snapshot['compute_list_empty'],
                          snapshot['other_process_pids']))
    return True, 'IDLE: memory.used=%d MiB <= %d MiB' % (used, limit)


def await_idle(limit=None, wait_seconds=None, poll_seconds=None):
    """Wait a BOUNDED time for the card to fall below the limit.

    A busy card is not a cell failure. The 2026-09-11 lead run started arm A
    while the A1 contrast was still leaving the card, and the single-probe
    check turned "somebody else is finishing" into a RED cell with a charged
    reservation. This waits instead, and only reds out after the bound.

    The bound is STRUCTURAL: `attempts_allowed` caps the number of probes up
    front, so there is no unbounded loop. This is a bounded WAIT for a
    resource somebody else holds, never a retry of our own failed work --
    nothing here re-runs a cell. It never signals anything.

    Prior art: R1 `await_idle` (`scripts/grm_r1_replay.py:440`, GRM
    contributors 2026), taken with its counted-loop bound and its decline-only
    contract. OURS: the LT1.1 defaults and the receipt returned to `run_cell`.
    """
    import time
    limit = IDLE_LIMIT_MIB if limit is None else int(limit)
    wait_seconds = max(0, int(IDLE_WAIT_SECONDS if wait_seconds is None
                              else wait_seconds))
    poll_seconds = max(1, int(IDLE_POLL_SECONDS if poll_seconds is None
                              else poll_seconds))
    attempts_allowed = 1 + (wait_seconds // poll_seconds)
    deadline = time.monotonic() + wait_seconds
    attempts = []
    snapshot, ok, reason = {}, False, 'NOT_PROBED'
    for _ in range(int(attempts_allowed)):
        snapshot = device_snapshot()
        ok, reason = idle_gate(snapshot, limit)
        attempts.append(dict(time_unix=snapshot.get('time_unix'),
                             memory_used_mib=snapshot.get('memory_used_mib'),
                             ok=ok, reason=reason))
        if ok:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        print('LT1.1 waiting for the card: %s' % reason,
              file=sys.stderr, flush=True)
        time.sleep(min(poll_seconds, max(1, remaining)))
    return ok, dict(attempts=attempts, final=snapshot, reason=reason,
                    limit_mib=limit, waited_for_seconds=wait_seconds,
                    attempts_allowed=int(attempts_allowed),
                    policy='bounded wait for a card another process holds; '
                           'declines only, never signals')


def spawn_argv(cell):
    """The child `run_cell` launches: THIS module, not LT1's worker module.

    `grm_lt1_worker.__main__` resolves its cell with `lt.verify()` -- LT1's own
    chain -- which is the gate the lead's ruling removed. Routing the child
    here means it resolves its cell from LT1.1's chain instead.
    """
    return [sys.executable, '-m', 'scripts.grm_lt1_1',
            '--arm', current_arm(), '--worker', cell['id']]


def spawn_env(env, cell):
    """Carry the pinned arm into the child.

    `environment(flags)` strips every ambient `GRM_*`, and `run_cell` rebuilds
    the child environment from it, so the arm pin must be re-applied here or
    the child cannot tell A from A+. This is the same pin-after-the-strip
    contract R1's `pin_rule` states, applied across a process boundary.
    """
    arm = current_arm()
    env[ARM_ENV] = arm
    env.pop(ALIAS_ENV, None)
    if ARM_ALIAS[arm] is not None:
        env[ALIAS_ENV] = ARM_ALIAS[arm]
    env[RULE_ENV] = 'margin_first'
    # The campaign root, so the child writes into the SAME directory the
    # parent reserved rather than recomputing a default from the arm.
    env[RUN_ENV] = str(worker.RUN)
    return env


def lt1_1_worker_cpu(cell):
    """`lt1_1_worker` with the CPU double in place of the model.

    Identical to `lt1_1_worker` except for the loader: the lease-parent
    contract, the cooperative deadline, the cell directory `run_cell` already
    created, and the create-only `worker.json` are all the same. It exists so
    a gate can spawn a REAL leased child on a machine with no GPU -- the model
    is the only thing that cannot be exercised here, so it is the only thing
    replaced.
    """
    import signal
    import time
    import pytest
    if os.environ.get('GRM_LT1_LEASE_PARENT') != str(os.getppid()):
        raise ValueError('WORKER_REQUIRES_LEASE_PARENT')
    arm = current_arm()
    deadline = time.monotonic() + cell['worker_seconds']

    def expired(*_):
        raise TimeoutError('WORKER_COOPERATIVE_DEADLINE')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(cell['worker_seconds'])
    directory = worker.cell_directory(worker.RUN / 'cells', cell['id'])
    try:
        with pytest.MonkeyPatch.context() as patch:
            result = worker.execute(cell, directory, registration(arm),
                                    fake_loader(patch),
                                    run=worker.RUN / 'cells', deadline=deadline,
                                    fake=True)
        lt.create(directory / 'worker.json', result)
    finally:
        signal.alarm(0)


def lt1_1_worker(cell):
    """The leased child body: LT1's `worker`, with LT1.1's verification.

    `grm_lt1_worker.worker` calls `lt.verify()` at its line 188 to obtain the
    registration it hands to `execute`. That is LT1's chain. This replacement
    keeps every other part of that function -- the lease-parent contract, the
    cooperative SIGALRM deadline with no kill syscall, the GPU loader, the
    cell directory, the create-only `worker.json` -- and swaps only the
    registration source for `registration(arm)`, which is LT1.1's chain
    checked by `lt1_1_preflight`.

    Prior art: `scripts/grm_lt1_worker.py:worker` (GRM contributors, 2026),
    reproduced in structure with one substitution; the deadline mechanism is
    POSIX `setitimer` via Python's `signal`, unchanged. No new lease, timeout
    or checkpoint logic.
    """
    import signal
    import time
    if os.environ.get('GRM_LT1_LEASE_PARENT') != str(os.getppid()):
        raise ValueError('WORKER_REQUIRES_LEASE_PARENT')
    arm = current_arm()
    deadline = time.monotonic() + cell['worker_seconds']

    def expired(*_):
        raise TimeoutError('WORKER_COOPERATIVE_DEADLINE')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(cell['worker_seconds'])
    directory = worker.cell_directory(worker.RUN / 'cells', cell['id'])
    try:
        result = worker.execute(cell, directory, registration(arm), gpu_loader,
                                run=worker.RUN / 'cells', deadline=deadline)
        lt.create(directory / 'worker.json', result)
    finally:
        signal.alarm(0)


def gpu_loader(session, flags, state):
    """The registered loader: the real model, exactly as LT1's worker uses."""
    from scripts import grm_e2e_session as e2e
    from scripts.grm_c2_cells import args_for
    _, _, repo, info = e2e.load_model_and_repo(args_for(e2e, session, flags),
                                               session)
    return repo, info


def fake_loader(patch):
    """CPU double loader.

    Prior art: scripts/grm_lt1_worker_cpu.py (GRM contributors, 2026), taken
    verbatim, plus the alias-aware dual reader D1 established: the
    consolidation path must receive a faithful extractive digest, or the
    coverage gate in `ArenaCache.consolidate` aborts every alias fold and the
    A+ fake arm measures nothing. Receipt:
    artifacts/grm_d1/alias_cpu_aplus.json.
    """
    from scripts.grm_c7_diagnose import repository, Model
    from scripts.grm_lt1_cpu import visible_answer
    from core.grm_alias_fold import alias_scan_text

    def loader(session, flags, state):
        repo = repository(session / 'repository', patch)
        arena = repo.arena
        arena.width = flags['arena_width']
        arena.recency_mounts = 2
        if state.get('codec_words'):
            arena.m.codec.words = state['codec_words']
            arena.m.codec.ids = {w: i for i, w in enumerate(arena.m.codec.words)}

        consolidating = {'on': False, 'sources': []}

        class ProseModel(Model):
            def __call__(self, ids, kv_caches=None, position_offset=0, **kw):
                if kv_caches is None:
                    text = self.codec.decode(ids[0])
                    if consolidating['on']:
                        self.fold_output = (
                            ' '.join(consolidating['sources']) + '<|end|>')
                    else:
                        visible = ((self.injected if arena.cur_mounts else '')
                                   + '\n' + text)
                        self.fold_output = visible_answer(visible, text)
                return super().__call__(ids, kv_caches, position_offset, **kw)
        arena.m.__class__ = ProseModel

        original = arena.consolidate

        def consolidate(idxs, *args, **kwargs):
            consolidating['on'] = True
            consolidating['sources'] = [
                ' '.join(alias_scan_text(
                    arena.grafts[i].get('text', '')).split()) for i in idxs]
            try:
                return original(idxs, *args, **kwargs)
            finally:
                consolidating['on'] = False
        arena.consolidate = consolidate
        return repo, dict(model='CPU prose fake', gpu=False)
    return loader


# ------------------------------------------------------------------ actions

def preflight(arm):
    """Document-chain checks only. GPU readiness is LT1's own `preflight`."""
    reasons = []
    for path in (REGISTRATION, AMENDMENT1, FIXTURE):
        if not path.exists():
            reasons.append('MISSING_INPUT: ' + path.name)
    if not reasons:
        try:
            registration(arm)
        except ValueError as exc:
            reasons.append(str(exc))
    return dict(status='BLOCKED' if reasons else 'READY', reasons=reasons,
                arm=arm, gpu_executed=False,
                registration_sha256=sha(REGISTRATION) if REGISTRATION.exists() else None,
                amendment1_sha256=sha(AMENDMENT1) if AMENDMENT1.exists() else None,
                fixture_sha256=sha(FIXTURE) if FIXTURE.exists() else None)


def dry_run(arm, *, root=None):
    """Enumerate the arm's cells and estimates. No GPU, no lease, no writes."""
    reg = registration(arm)
    root = Path(root) if root else out_dir(arm)
    with pinned_arm(arm) as pin:
        cells = reg['cells']
        done = [c['id'] for c in cells
                if (root / 'cells' / c['id'] / 'controller.json').exists()]
        # The governing ceiling is the LATEST amendment's. Amendment 2
        # corrected amendment 1's 6120 s, which sat below the reservation sum
        # and would have tripped `run_cell`'s rail mid-campaign.
        amendment2 = OUT / 'amendment2.json'
        source = amendment2 if amendment2.exists() else AMENDMENT1
        budget = json.loads(source.read_text())['arms'][arm]
        lease = sum(c['lease_seconds'] for c in cells)
        return dict(status='PASS', mode='CPU_DRY_RUN', gpu_executed=False,
                    arm=arm, pinned=pin, cells=len(cells),
                    completed_cells=len(done),
                    estimate_seconds=round(
                        sum(c['estimate_seconds'] for c in cells), 3),
                    lease_seconds=lease,
                    budget_gpu_seconds=budget['budget_gpu_seconds'],
                    budget_source=str(source.relative_to(ROOT)),
                    within_budget=lease <= budget['budget_gpu_seconds'],
                    out_dir=str(root),
                    next_cell=next((c['id'] for c in cells
                                    if c['id'] not in done), None),
                    cell_ids=[c['id'] for c in cells],
                    binding=binding(arm))


def fake_cell(arm, cell_id, root):
    """Run ONE cell on the CPU double, in this process. Writes real receipts.

    `execute` enforces `RESTART_REQUIRES_NEW_PROCESS`: a resumed cell must
    observe a different pid/process_id than the checkpoint it loads. So one
    cell per process, exactly as the GPU controller spawns one child per cell.
    `run_fake` is the parent that spawns these.
    """
    import pytest
    reg = registration(arm)
    root = Path(root)
    cell = next(c for c in reg['cells'] if c['id'] == cell_id)
    directory = root / 'cells' / cell['id']
    with pytest.MonkeyPatch.context() as patch:
        loader = fake_loader(patch)
        # ONE seam definition. This used to install a narrow subset inline,
        # which silently drifted from `lt1_1_seams` when seams were added:
        # the fake path kept LT1's prose-deposit behaviour after the
        # production turn semantics landed, so it proved the wrong thing.
        with lt1_1_seams(arm), pinned_arm(arm):
            worker.RUN = root
            directory.mkdir(parents=True)
            lt.create(directory / 'reservation.json',
                      dict(seconds=cell['lease_seconds'], cell=cell,
                           binding=binding(arm), fake=True,
                           admission_rule='margin_first'))
            value = worker.execute(cell, directory, reg, loader,
                                   run=root / 'cells', fake=True)
            lt.create(directory / 'worker.json', value)
            lt.create(directory / 'controller.json',
                      dict(status='COMPLETE', error=None,
                           charged_seconds=0.0, cell=cell,
                           binding=binding(arm), fake=True,
                           admission_rule='margin_first'))
    return str(directory)


def run_fake(arm, *, root=None, limit=None):
    """Execute cells on the CPU double, one subprocess per cell.

    Mirrors the GPU controller's structure (one child process per cell, parent
    accumulates receipts) minus the lease, the flock and the GPU check, so the
    receipt plumbing `summary` reads is exercised end to end. Cells already
    COMPLETE are skipped, which is what makes this path resumable.

    Prior art: the one-subprocess-per-cell fake loop in
    tests/test_grm_lt1_fix6.py and tests/test_grm_lt1_amendment1.py
    (GRM contributors, 2026), reused.
    """
    import subprocess
    reg = registration(arm)
    root = Path(root) if root else out_dir(arm)
    written, skipped = [], []
    todo = reg['cells'][:limit] if limit else reg['cells']
    for cell in todo:
        directory = root / 'cells' / cell['id']
        if (directory / 'controller.json').exists():
            skipped.append(cell['id'])
            continue
        env = arm_environment(arm, reg['arms']['A']['flags'])
        env['PATH'] = os.environ.get('PATH', '')
        env['HOME'] = os.environ.get('HOME', '')
        result = subprocess.run(
            [sys.executable, '-m', 'scripts.grm_lt1_1', '--arm', arm,
             '--fake-cell', cell['id'], '--out', str(root)],
            cwd=ROOT, env=env, capture_output=True, text=True)
        if result.returncode:
            raise ValueError('LT11_FAKE_CELL_FAILED: %s\n%s'
                             % (cell['id'], result.stdout + result.stderr))
        written.append(str(directory))
    return dict(arm=arm, mode='CPU_FAKE', gpu_executed=False,
                cells_written=written, cells_skipped=skipped,
                out_dir=str(root),
                evidence_class='CPU fake worker (receipt plumbing; NOT a '
                               'language-model quality measurement)')


#: The ruling this preflight implements, recorded verbatim so the code and the
#: amendment cannot drift apart. Lead, 2026-09-11.
RULING = (
    "LT1's original registration is a frozen receipt of its day; it is NOT "
    're-validated against today\'s core. LT1.1\'s host preflight must validate '
    "LT1.1's own chain — registration 02b44d02 + amendments 1–3, whose core "
    'pins were rebound to this tree with attribution — using the same '
    'verification functions (`grm_lt1.verify`-class checks: sha-bound inputs, '
    'fixture sha, cell schedule, budget) but pointed at LT1.1\'s '
    "registration/amendment set. LT1's registration sha and its recorded core "
    'pins are carried as `parent` lineage in the LT1.1 receipt (recorded, with '
    'the drift table you already attributed), not as a gate.')


def parent_lineage():
    """LT1's recorded identity, carried as lineage. Never gated.

    Every value here is READ from LT1's frozen documents and reported. Nothing
    in this function compares anything to today's core: that comparison is
    exactly what the ruling removes.
    """
    lt1_registration = ROOT / 'artifacts/grm_lt1/registration.json'
    lt1_binding = ROOT / 'artifacts/grm_lt1/amendment3/registration_amendment_r3.json'
    amendment1 = json.loads(AMENDMENT1.read_text())
    recorded = {n: c['after_sha256']
                for n, c in json.loads(lt1_binding.read_text())['core_shas'].items()}
    drift = []
    for name, change in sorted(amendment1['core_rebind']['changed'].items()):
        drift.append(dict(input=name,
                          lt1_recorded=change['before_sha256'],
                          lt1_1_rebound=change['after_sha256'],
                          on_tree_now=sha(ROOT / name),
                          attribution=change['attribution']))
    for name, entry in sorted(amendment1['core_rebind']['new_inputs'].items()):
        drift.append(dict(input=name, lt1_recorded=None,
                          lt1_1_rebound=entry['sha256'],
                          on_tree_now=sha(ROOT / name),
                          attribution=entry['attribution']))
    return dict(
        lt1_registration='artifacts/grm_lt1/registration.json',
        lt1_registration_sha256=sha(lt1_registration),
        lt1_core_binding='artifacts/grm_lt1/amendment3/registration_amendment_r3.json',
        lt1_core_binding_sha256=sha(lt1_binding),
        lt1_recorded_core_pins=len(recorded),
        lt1_amendment4_protocol_binding=json.loads(
            (ROOT / 'artifacts/grm_lt1/amendment4/resume_registration.json')
            .read_text())['protocol_binding'],
        drift_table=drift,
        status='RECORDED as parent lineage, not gated (see RULING)')


def lt1_1_preflight(arm='A'):
    """LT1.1's own chain preflight — the same verify-class checks, our chain.

    Per the lead's ruling (see RULING), this validates LT1.1's registration +
    amendments 1-3 against THIS tree, and carries LT1's identity as lineage.
    It performs the same classes of check `grm_lt1.verify` / `grm_lt1.preflight`
    perform, pointed at our documents:

      * sha-bound documents      registration/amendment1/2/3 vs their sidecars
      * chain continuity         each amendment names its parent's sha
      * sha-bound inputs         every core pin amendment 1 rebound, plus the
                                 runner amendment 3 bound, vs the file on disk
      * fixture sha              dialogue.json vs the registered digest
      * cell schedule            26 arm-A cells, matching the amendment
      * budget                   reservation sum within the registered ceiling
      * host readiness           free space and the pinned admission rule

    What it does NOT do is re-validate LT1's day-of core pins. That is the
    ruling: those are a frozen receipt, reported in `parent_lineage`.

    This is the check that replaced `lt.preflight()`. `lt.preflight` failed on
    this tree with `INPUT_SHA_MISMATCH: core/graft_arena.py` -- not because
    anything about LT1.1 was wrong, but because it re-hashes core against SHAs
    recorded before `grm-merge` moved. NOTE for the record: `apply4`'s
    protocol-binding check (`a['protocol_binding'] != lt.binding('CPU')`)
    PASSES on this tree unchanged, because `lt.binding` reports AMEND3's
    RECORDED core shas rather than live ones. The amendment-4 chain was never
    the problem; only that final input loop was.
    """
    reasons = []
    # Every amendment present on disk is verified, so a new one is covered the
    # moment it is emitted rather than when someone remembers to list it.
    documents = {'registration.json': (REGISTRATION,
                                       OUT / 'registration.sha256')}
    for path in sorted(OUT.glob('amendment[0-9]*.json')):
        documents[path.name] = (path, path.with_suffix('.sha256'))
    shas = {}
    for name, (path, sidecar) in documents.items():
        if not path.exists():
            reasons.append('MISSING_DOCUMENT: ' + name)
            continue
        shas[name] = sha(path)
        if not sidecar.exists():
            reasons.append('MISSING_SIDECAR: ' + name)
        elif sidecar.read_text().split()[0] != shas[name]:
            reasons.append('DOCUMENT_SHA_MISMATCH: ' + name)
    if reasons:
        return dict(status='BLOCKED', reasons=reasons, arm=arm,
                    gpu_executed=False, ruling=RULING)

    loaded = {name: json.loads(path.read_text())
              for name, (path, _) in documents.items()
              if name != 'registration.json'}
    a1 = loaded['amendment1.json']

    # chain continuity: amendment 1 names the registration; every later
    # amendment names its immediate parent.
    if a1['registration_sha256'] != shas['registration.json']:
        reasons.append('AMENDMENT1_CHAIN_MISMATCH')
    ordered = sorted(loaded, key=lambda n: loaded[n]['amendment'])
    for previous, name in zip(ordered, ordered[1:]):
        if loaded[name].get('previous_amendment_sha256') != shas[previous]:
            reasons.append('AMENDMENT%d_CHAIN_MISMATCH'
                           % loaded[name]['amendment'])

    # The LATEST amendment governs the runner binding and the budget.
    latest = loaded[ordered[-1]]
    a2 = json.loads((OUT / 'amendment2.json').read_text())

    # sha-bound inputs: LT1.1's OWN rebound core pins, checked against the tree
    checked_inputs = 0
    for name, change in sorted(a1['core_rebind']['changed'].items()):
        checked_inputs += 1
        if sha(ROOT / name) != change['after_sha256']:
            reasons.append('INPUT_SHA_MISMATCH: ' + name)
    for name, entry in sorted(a1['core_rebind']['new_inputs'].items()):
        checked_inputs += 1
        if sha(ROOT / name) != entry['sha256']:
            reasons.append('INPUT_SHA_MISMATCH: ' + name)
    # ... and the runner the LATEST amendment bound.
    runner = latest['runner']
    checked_inputs += 1
    if sha(ROOT / runner['path']) != runner['sha256']:
        reasons.append('RUNNER_SHA_MISMATCH: ' + runner['path'])

    # fixture sha
    if not FIXTURE.exists():
        reasons.append('MISSING_FIXTURE')
    elif sha(FIXTURE) != a1['fixture']['sha256']:
        reasons.append('FIXTURE_SHA_MISMATCH')

    # cell schedule + budget
    cells = a1['arms'][arm]['cells'] if arm in a1['arms'] else []
    if len(cells) != 26:
        reasons.append('CELL_SCHEDULE_MISMATCH: %d' % len(cells))
    lease = sum(c['lease_seconds'] for c in cells)
    ceiling = a2['arms'][arm]['budget_gpu_seconds'] if arm in a2['arms'] else 0
    if lease > ceiling:
        reasons.append('BUDGET_BELOW_RESERVATION: %d > %d' % (lease, ceiling))

    # host readiness (the parts of LT1's preflight that are about the machine)
    free = shutil.disk_usage(ROOT).free
    if free < 20_000_000_000:
        reasons.append('FREE_SPACE_BELOW_20_GB')
    if os.environ.get(RULE_ENV) != 'margin_first':
        reasons.append('REGISTERED_ADMISSION_RULE_MISMATCH')

    return dict(status='BLOCKED' if reasons else 'READY', reasons=reasons,
                arm=arm, gpu_executed=False,
                document_sha256=shas,
                inputs_checked=checked_inputs,
                fixture_sha256=sha(FIXTURE) if FIXTURE.exists() else None,
                cells=len(cells),
                reservation_seconds=lease, budget_gpu_seconds=ceiling,
                free_bytes=free, minimum_free_bytes=20_000_000_000,
                ruling=RULING,
                parent=parent_lineage())


def host_preflight(arm='A'):
    """Back-compatible name for `lt1_1_preflight` (the ruling renamed it)."""
    return lt1_1_preflight(arm)


def resume(arm, *, root=None, dry_lease=False, host_gate=True):
    """Registered GPU campaign for one arm. Reuses LT1's controller loop.

    `dry_lease=True` walks THE SAME code path -- amendment loading, `apply4`,
    binding, preflight, seam redirection, arm pin, campaign-owner file, cell
    selection via `worker.pending` -- and stops at the lease boundary instead
    of calling `worker.run_cell`. That is the only substitution: nothing about
    the route before it is mocked, which is what makes it a gate rather than a
    rehearsal. It is how `--resume --dry-lease` reproduces on CPU the failure
    the lead hit on the card.

    `host_gate=False` skips the chain preflight. Since the lead's ruling that
    gate is `lt1_1_preflight` -- LT1.1's OWN chain, which passes on this tree
    -- so the escape hatch is no longer needed to get a green route and is
    kept only for isolating the route from its documents in tests. The GPU
    path always runs the gate.
    """
    reg = registration(arm)
    root = Path(root) if root else out_dir(arm)
    # The chain preflight runs OUTSIDE the seams -- `lt1_1_preflight` hashes
    # documents directly and must not see redirected module state -- but
    # INSIDE the arm pin, because one of its checks is that the registered
    # admission rule is actually in force. LT1's `preflight` reads the same
    # variable for the same reason.
    with pinned_arm(arm):
        gate = (lt1_1_preflight(arm) if host_gate
                else dict(status='SKIPPED', reasons=['host_gate=False']))
    if host_gate and gate['status'] != 'READY':
        raise ValueError(json.dumps(gate))
    with lt1_1_seams(arm), pinned_arm(arm) as pin:
        root.mkdir(parents=True, exist_ok=True)
        owner = root / 'campaign.active'
        fd = os.open(owner, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        # Progress goes to stderr so stdout stays a single parseable JSON
        # document for --dry-lease; the GPU path's operator log is unchanged.
        print('LT1.1 arm %s pinned %s' % (arm, json.dumps(pin)),
              file=sys.stderr, flush=True)
        selected = []
        try:
            while True:
                cell = worker.pending(reg, run=root)
                if cell is None:
                    break
                print('LT1.1 %s cell %s' % (arm, cell['id']),
                      file=sys.stderr, flush=True)
                if dry_lease:
                    # The lease boundary. Everything above this line is the
                    # production route; `run_cell` is where the GPU lease,
                    # the flock and the model would be taken.
                    selected.append(cell['id'])
                    return dict(status='PASS', mode='RESUME_DRY_LEASE',
                                gpu_executed=False, arm=arm, pinned=pin,
                                host_preflight=gate['status'],
                                next_cell=cell['id'],
                                selected=selected,
                                receipt_binding=worker.bind(cell['arm']),
                                out_dir=str(root),
                                stopped_at='worker.run_cell (lease boundary)')
                if not worker.run_cell(cell, reg):
                    return 2
            if dry_lease:
                return dict(status='PASS', mode='RESUME_DRY_LEASE',
                            gpu_executed=False, arm=arm, pinned=pin,
                            host_preflight=gate['status'], next_cell=None,
                            selected=selected, out_dir=str(root),
                            stopped_at='campaign complete')
            return 0
        finally:
            owner.unlink()


def summary(arm, *, root=None):
    """Score the arm's receipts with LT1's own scorer, unchanged."""
    reg = registration(arm)
    root = Path(root) if root else out_dir(arm)
    fixture = json.loads(FIXTURE.read_text())
    probes = {p['id']: p for p in fixture['probes']}
    rows, complete = [], []
    for cell in reg['cells']:
        controller = root / 'cells' / cell['id'] / 'controller.json'
        if not controller.exists() or lt.read(controller)['status'] != 'COMPLETE':
            continue
        complete.append(cell['id'])
        path = root / 'cells' / cell['id'] / 'probes.jsonl'
        if path.exists():
            rows.extend(json.loads(line) for line in
                        path.read_text().splitlines() if line.strip())
    # Recall probes and recap probes share `probes.jsonl` but live in
    # different fixture lists, so classify by the row's own `kind` rather
    # than indexing every id into the recall map (which raised
    # KeyError: 'recap_1' the first time recap rows were written).
    recall_rows = [r for r in rows if r.get('kind') != 'recap_probe']
    recap_rows = [r for r in rows if r.get('kind') == 'recap_probe']
    table = {}
    for cls in ('fresh', 'correction', 'alias'):
        want = [p for p in fixture['probes'] if p['class'] == cls]
        got = [r for r in recall_rows
               if probes[r['probe_id']]['class'] == cls]
        correct = sum(1 for r in got
                      if lt.score(r['memory']['answer'],
                                  probes[r['probe_id']]['expected'])['exact_correct'])
        table[cls] = dict(expected_n=len(want), n=len(got), correct=correct,
                          exact_rate=(correct / len(got)) if got else None)
    recap_correct = sum(1 for r in recap_rows
                        if lt.score(r['memory']['answer'],
                                    r['expected'])['exact_correct'])
    table['recap'] = dict(expected_n=len(fixture.get('recap_probes', [])),
                          n=len(recap_rows), correct=recap_correct,
                          exact_rate=(recap_correct / len(recap_rows)
                                      if recap_rows else None))
    # `binding` reads the pinned arm rather than taking one, so that LT1's
    # callers can keep passing a backend label. Hold it for this call.
    with _arm_state(arm):
        stamp = binding(arm)
    return dict(arm=arm, out_dir=str(root), complete_cells=len(complete),
                total_cells=len(reg['cells']),
                complete=len(complete) == len(reg['cells']),
                completed_cell_ids=complete,
                measured_recalls=len(rows), by_class=table,
                binding=stamp,
                evidence_class=('partial raw rows' if rows else 'no rows'))


# ---------------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog='scripts/grm_lt1_1.py',
        description='LT1.1 runner: one arm per invocation, over the LT1 '
                    'worker machinery.')
    p.add_argument('--arm', choices=ARMS, required=True,
                   help='A (alias fold OFF) or A+ (GRM_ALIAS_FOLD_MERGE=1)')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--resume', action='store_true',
                      help='registered GPU campaign (takes the shared lease)')
    mode.add_argument('--summary', action='store_true')
    mode.add_argument('--fake', action='store_true',
                      help='CPU double execution; writes real receipts')
    mode.add_argument('--worker', help='leased child entry point (internal)')
    mode.add_argument('--fake-cell', help='one-cell CPU child (internal)')
    mode.add_argument('--worker-cpu',
                      help='leased child with the CPU double (gate only); '
                           'same route as --worker, model replaced')
    p.add_argument('--out', help='override the campaign root (dry-run/fake/summary)')
    p.add_argument('--limit', type=int, help='fake mode: run only the first N cells')
    p.add_argument('--dry-lease', action='store_true',
                   help='with --resume: walk the real resume route (amendment '
                        'load, apply4, binding, preflight, cell selection) and '
                        'stop at the lease boundary; no GPU, no flock')
    p.add_argument('--no-host-gate', action='store_true',
                   help="with --resume --dry-lease: skip LT1's host preflight, "
                        'which validates the HOST tree rather than the LT1.1 '
                        'route (use when core has drifted past LT1 SHAs)')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.worker_cpu:
        # Same leased-child route as --worker; the model is the CPU double.
        gate = lt1_1_preflight(args.arm)
        if gate['status'] != 'READY':
            print(json.dumps(gate), file=sys.stderr, flush=True)
            raise ValueError('LT11_CHILD_PREFLIGHT_BLOCKED: %s'
                             % ', '.join(gate['reasons']))
        with lt1_1_seams(args.arm):
            worker.worker = lt1_1_worker_cpu
            cell = next(c for c in registration(args.arm)['cells']
                        if c['id'] == args.worker_cpu)
            worker.worker(cell)
        return 0
    if args.worker:
        # Leased child. The parent pinned the arm into our environment via
        # `spawn_env`; verify OUR chain (never LT1's -- that is the ruling)
        # before doing any work, so a drifted LT1.1 document stops the child
        # exactly as it would stop the parent.
        gate = lt1_1_preflight(args.arm)
        if gate['status'] != 'READY':
            print(json.dumps(gate), file=sys.stderr, flush=True)
            raise ValueError('LT11_CHILD_PREFLIGHT_BLOCKED: %s'
                             % ', '.join(gate['reasons']))
        with lt1_1_seams(args.arm):
            cell = next(c for c in registration(args.arm)['cells']
                        if c['id'] == args.worker)
            worker.worker(cell)
        return 0
    if args.fake_cell:
        print(fake_cell(args.arm, args.fake_cell,
                        args.out or out_dir(args.arm)))
        return 0
    if args.preflight:
        value = preflight(args.arm)
        print(json.dumps(value, indent=2))
        return 0 if value['status'] == 'READY' else 2
    if args.dry_run:
        gate = preflight(args.arm)
        if gate['status'] != 'READY':
            print(json.dumps(gate, indent=2))
            return 2
        print(json.dumps(dry_run(args.arm, root=args.out), indent=2))
        return 0
    if args.summary:
        print(json.dumps(summary(args.arm, root=args.out), indent=2))
        return 0
    if args.fake:
        print(json.dumps(run_fake(args.arm, root=args.out, limit=args.limit),
                         indent=2))
        return 0
    if args.dry_lease:
        print(json.dumps(resume(args.arm, root=args.out, dry_lease=True,
                                host_gate=not args.no_host_gate), indent=2))
        return 0
    return resume(args.arm)


if __name__ == '__main__':
    raise SystemExit(main())
