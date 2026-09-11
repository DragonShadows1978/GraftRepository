#!/usr/bin/env python3
"""GRM-R1 margin-first regression replay worker.

Runs the 31 SCOUT-FIX-6 registered execution cells twice from ONE recorded
C2 checkpoint each: arm OFF (today's ``all_tokens_bind``) and arm ON
(``GRM_ADMISSION_RULE=margin_first``).  Both arms use the PRODUCTION ladder
(``scripts/grm_e2e_session``), the production scorer
(``lsr_p2c_replay_gpu.answer_verdict`` -> ``det1_common.contains_value``),
and the C2 frame flags of that cell's own side (profile vs defaults).

Prior art (paper/system/year; taken vs ours) -- see REPORT.md "Prior art":
  * GRM C2 restart cells / ``grm_c2_cells.py`` (GRM contributors, 2026):
    TAKEN verbatim -- ``environment(flags)`` frame pinning, ``flags_for``,
    ``args_for``, ``observe`` and ``score_probe``'s traced ``run_turn``
    shape.  Imported, not re-implemented.
  * GRM C7 amendment-7 / ``grm_c7_middle_replay.open_copy`` (GRM, 2026):
    TAKEN -- copy the recorded repository per arm and load the GPU model
    ONCE per batch, reusing the ordinary ``GraftRepository`` constructor for
    every later open.  OURS: two arms per cell off one load, not one.
  * GRM C7/FIX8 lease + receipt discipline (GRM, 2026): TAKEN -- O_EXCL
    single-owner file, pessimistic reservations charged even on failure,
    create-only receipts, orphan/failed-campaign fail-closed, foreground
    cooldown outside the lease.  The lease itself is ``grm_cmc1_gpu_arms.
    gpu_lease`` IMPORTED, not a third copy (order 1 forbids one).
  * GRM RD1 amendment-1 / C7 ``A0_R3_BYTE_MISMATCH_STOP`` (GRM, 2026):
    TAKEN -- the arm-OFF parity barrier.  OURS: the barrier compares the
    OFF ``rank_plan`` to the FIX-6 recorded ``off_plan`` bytes, and a
    mismatch is a RED receipt that STOPS the run (order 1).
  * GRM LT1 ``grm_lt1_admission.evaluate`` / FIX-6 ``FrozenArena``
    (GRM, 2026): TAKEN -- replay stored eligibility/ranking/margin through
    the REAL ``core.grm_admission`` boundary.  OURS: the ``--fake`` CPU seam
    below, which must synthesize a score vector because the CPU double's
    ``_node_key`` is a constant and cannot reproduce GPU cosine scores.
  * SQuAD / SQuAD 2.0 answerability (Rajpurkar et al. 2016 arXiv:1606.05250;
    Rajpurkar/Jia/Liang 2018 arXiv:1806.03822) -- the value-span + abstention
    ancestry behind ``contains_value``.  Concept only; the normalization and
    the transition classes below are not SQuAD's.  Unverified in-sandbox
    (no network) -- lead to check.
  * ARIES (Mohan et al., 1992) -- checkpoint/restart recovery concept behind
    the C2 checkpoints this worker consumes.  Concept only, no WAL algorithm.
  * Paired/counterfactual A-B evaluation off one held state is standard
    experimental practice; no specific prior art known to me for the exact
    composition of (recorded-plan parity barrier + in-process rule flip +
    per-arm fresh repository copy + five-way transition classification).

NO NEW ROUTING, SCORING OR ADMISSION ALGORITHM IS INTRODUCED HERE.  This
file is glue, receipts and gates.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from contextlib import contextmanager
import shutil
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.grm_c7_amendment7 import read, sha, write, need  # noqa: E402

OUT = ROOT / 'artifacts/grm_r1'
REG = OUT / 'registration.json'
FIX6 = ROOT / 'artifacts/grm_scout_fix6'
CELLS = FIX6 / 'c2_replay_cells.json'

#: The two arms.  OFF is today's shipped default; ON is the FIX-6 opt-in.
ARMS = ('off', 'on')
RULE_ENV = 'GRM_ADMISSION_RULE'
ARM_RULE = {'off': None, 'on': 'margin_first'}

#: Transition classes for one execution (OFF answer -> ON answer).
TRANSITIONS = ('unchanged_correct', 'unchanged_wrong', 'correct_to_wrong',
               'wrong_to_correct', 'abstain_gained', 'abstain_lost')

#: Paths an amendment may bind that the registration could not list, because
#: they did not exist when it was written.  Strictly the amendment's own
#: builder: an amendment can never smuggle in a new core or scoring source.
REBINDABLE_NEW = ('scripts/grm_r1_amend_1.py', 'scripts/grm_r1_amend_2.py',
                  'scripts/grm_r1_amend_3.py')


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------

def verify(root=None):
    """Registration + every sha-bound source, cells file and cohort contract."""
    need(REG.is_file(), 'R1_REGISTRATION_MISSING')
    need(sha(REG) == REG.with_suffix('.sha256').read_text().split()[0],
         'R1_REGISTRATION_SHA_MISMATCH')
    r = read(REG)
    need(r['schema'] == 'grm.r1.replay.v1', 'R1_SCHEMA_MISMATCH')
    inputs = dict(r['inputs'])
    # An amendment may RE-BIND a source whose hash legitimately changed (the
    # worker itself, when the amendment's fix lives in it).  The re-bind is
    # explicit, sha-bound and scoped: it can only replace a hash for a path
    # the registration already listed, never add or drop one.  Every other
    # input keeps its original registered hash and still fails closed.
    a = amendment(root)
    if a:
        for path, digest in a.get('rebound_inputs', {}).items():
            need(path in inputs or path in REBINDABLE_NEW,
                 'R1_AMENDMENT_REBIND_OUT_OF_SCOPE: ' + path)
            inputs[path] = digest
        r['amendment_sha256'] = a['_self_sha256']
        r['amendment_chain_sha256'] = a.get('_chain', [])
        r['amendment_index'] = a['amendment']
    for path, digest in inputs.items():
        p = Path(path) if Path(path).is_absolute() else ROOT / path
        need(sha(p) == digest, 'R1_INPUT_SHA_MISMATCH: ' + path)
    need(sha(CELLS) == r['cells_sha256'], 'R1_CELLS_SHA_MISMATCH')
    registered = read(CELLS)
    need(registered['admission_rule'] == 'margin_first'
         and registered['schema'] == 'grm.fix6.c2.replay.v1',
         'R1_CELLS_CONTRACT_MISMATCH')
    need(len(registered['cells']) == r['cell_count'] == 31, 'R1_COHORT_MISMATCH')
    ids = [c['id'] for c in registered['cells']]
    need(len(ids) == len(set(ids)), 'R1_DUPLICATE_CELL_ID')
    need([i for b in r['batch_ids'] for i in r['batches'][b]] == ids,
         'R1_BATCH_COHORT_MISMATCH')
    need(len({c['question_id'] for c in registered['cells']})
         == r['distinct_question_ids'] == 9, 'R1_QUESTION_COHORT_MISMATCH')
    # Budget rail: pessimistic per-batch reservation, charged even on success.
    reserved = sum(r['batch_lease_seconds'][b] for b in r['batch_ids'])
    need(reserved == r['reserved_seconds'] <= r['gpu_cap_seconds'] <= 3600,
         'R1_BUDGET_MISMATCH')
    return r


def cells():
    return read(CELLS)['cells']


def cell_by_id(cell_id):
    for c in cells():
        if c['id'] == cell_id:
            return c
    raise ValueError('R1_UNKNOWN_CELL: ' + str(cell_id))


def verify_cell_inputs(c):
    """Every hash in ``checkpoint_files`` FIRST.  A mismatch is RED, no retry."""
    cp = Path(c['checkpoint'])
    need(cp.is_file(), 'R1_CHECKPOINT_MISSING: ' + str(cp))
    need(sha(cp) == c['checkpoint_sha256'],
         'R1_CHECKPOINT_SHA_MISMATCH: ' + c['id'])
    descriptor = read(cp)
    need(descriptor['files'] == c['checkpoint_files'],
         'R1_CHECKPOINT_FILE_TABLE_MISMATCH: ' + c['id'])
    repository = cp.parent / 'repository'
    for name, digest in c['checkpoint_files'].items():
        p = repository / name
        need(p.is_file(), 'R1_CHECKPOINT_FILE_MISSING: ' + c['id'] + ':' + name)
        need(sha(p) == digest,
             'R1_CHECKPOINT_FILE_SHA_MISMATCH: ' + c['id'] + ':' + name)
    need(sha(Path(c['state_manifest'])) == c['state_manifest_sha256'],
         'R1_STATE_MANIFEST_SHA_MISMATCH: ' + c['id'])
    need(sha(ROOT / c['policy_state']) == c['policy_state_sha256'],
         'R1_POLICY_STATE_SHA_MISMATCH: ' + c['id'])
    need(sha(Path(c['source_worker'])) == c['source_worker_sha256'],
         'R1_SOURCE_WORKER_SHA_MISMATCH: ' + c['id'])
    return descriptor


def recorded_row(c):
    """The C2 row this cell was registered from: recorded answer + verdict."""
    worker = read(Path(c['source_worker']))
    row = worker['rows'][c['row_ordinal']]
    need(row['probe_id'] == c['question_id'], 'R1_ROW_PROBE_MISMATCH: ' + c['id'])
    need(row['side'] == c['side'] and row['battery'] == c['battery'],
         'R1_ROW_FRAME_MISMATCH: ' + c['id'])
    return row


# --------------------------------------------------------------------------
# scoring: the LT1/C5 value-span rule, unchanged
# --------------------------------------------------------------------------

def expected_values(context, battery):
    """The cell's recorded expected value, read from the RECORDED checkpoint.

    Never from the model's answer and never imputed.  Census/longhistory
    carry ``event.expected``; sup carries ``probe.expected_values`` plus its
    ``rejected_values`` (stale/competitor) guards.
    """
    if battery == 'sup':
        probe = context['probe']
        return ([str(v) for v in probe['expected_values']],
                [str(v) for v in probe['rejected_values']])
    return [str(context['event']['expected'])], []


def score_answer(answer, expected, rejected):
    """The LT1/C5 value-span verdict + the DET1.3 abstention grammar.

    ``answer_verdict`` is the production DET1 comparator C2 itself scored
    with; ``_is_refusal`` is the DET1.3 refusal grammar.  Neither is
    re-implemented here -- both are imported.
    """
    from scripts.lsr_p2c_replay_gpu import answer_verdict
    from scripts.grm_det1_3_gpu import _is_refusal
    verdict = answer_verdict(str(answer), expected_values=expected,
                             rejected_values=rejected)
    verdict['abstained'] = bool(_is_refusal(str(answer)))
    verdict['answer'] = str(answer)
    return verdict


def classify(off, on):
    """Five-way transition for one execution, plus the abstention moves.

    ``correct`` is the value-span verdict (expected hit, no rejected hit).
    Abstention transitions are reported ALONGSIDE, never folded into, the
    correctness classes: an OFF-correct answer that becomes an ON refusal is
    a ``correct_to_wrong`` AND an ``abstain_gained``.
    """
    a, b = bool(off['correct']), bool(on['correct'])
    if a and b:
        primary = 'unchanged_correct'
    elif a and not b:
        primary = 'correct_to_wrong'
    elif b and not a:
        primary = 'wrong_to_correct'
    else:
        primary = 'unchanged_wrong'
    abstain = None
    if on['abstained'] and not off['abstained']:
        abstain = 'abstain_gained'
    elif off['abstained'] and not on['abstained']:
        abstain = 'abstain_lost'
    return {'transition': primary, 'abstain_transition': abstain,
            'answer_changed': str(off['answer']) != str(on['answer']),
            'off_correct': a, 'on_correct': b}


# --------------------------------------------------------------------------
# one arm of one cell
# --------------------------------------------------------------------------

def pin_rule(arm):
    """Pin GRM_ADMISSION_RULE AFTER environment(flags) stripped every GRM_*.

    FIX-6's state contract: ``environment(flags)`` removes every ambient
    ``GRM_`` variable, so the rule MUST be pinned after it or the arm
    silently runs the default.  ``core.grm_admission.admission_rule()`` reads
    ``os.environ`` at call time, so this single variable is the ONLY
    difference between the two arms.  The pin is read back and asserted.
    """
    need(arm in ARM_RULE, 'R1_UNKNOWN_ARM: ' + str(arm))
    os.environ.pop(RULE_ENV, None)
    value = ARM_RULE[arm]
    if value is not None:
        os.environ[RULE_ENV] = value
    from core.grm_admission import admission_rule
    observed = admission_rule()
    need(observed == ('margin_first' if arm == 'on' else 'all_tokens_bind'),
         'R1_RULE_PIN_FAILED: arm=' + arm + ' observed=' + observed)
    return observed


@contextmanager
def scoped_env(keys=None):
    """Restore the EXACT prior environment on the way out, always.

    AMENDMENT-2 DEFECT FIX.  ``pin_rule`` and ``environment(flags)`` both
    mutate ``os.environ`` process-wide, and ``core.grm_admission.
    admission_rule()`` reads ``os.environ`` at CALL time.  Without
    restoration the pin outlives the run: in one pytest process every module
    imported after a R1 test saw ``GRM_ADMISSION_RULE=margin_first`` and was
    silently re-ruled (the lead observed 3 FIX-4 and 20 A1 failures that
    vanish when R1 runs alone).  A leaked experimental flag that changes
    OTHER suites' results is a correctness defect, not untidiness.

    ``keys=None`` snapshots and restores every ``GRM_`` variable plus the
    rule; a explicit key list narrows that.  Restoration is exact: a key
    absent before is REMOVED, not set to ''.

    Prior art: ``unittest.mock.patch.dict(os.environ)`` and pytest's
    ``monkeypatch.setenv`` (Python/pytest contributors) -- the standard
    snapshot/restore idiom, reproduced here because the worker must restore
    in production too, where no pytest fixture exists.  Nothing novel.
    """
    if keys is None:
        names = {k for k in os.environ if k.startswith('GRM_')} | {RULE_ENV}
    else:
        names = set(keys)
    saved = {k: os.environ.get(k) for k in names}
    try:
        yield
    finally:
        # Re-read: the body may have ADDED GRM_ keys that were not in the
        # snapshot (environment(flags) sets a whole frame), so drop those too.
        current = {k for k in os.environ if k.startswith('GRM_')} | names
        for k in current:
            before = saved.get(k, None) if k in saved else None
            if before is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = before


def open_arm(c, session, flags, loaded):
    """Copy the RECORDED repository for this arm and open it.

    Each arm gets its OWN copytree of the same hash-verified checkpoint,
    because the production ``run_turn`` deposits into the repository: a
    shared repo would leak arm OFF's deposits into arm ON.  The GPU model is
    loaded ONCE per batch and reused (C7 ``open_copy`` shape), so "same
    state" is guaranteed by the shared checkpoint hash, not by a shared
    mutable object.
    """
    from scripts import grm_e2e_session as e2e
    from scripts.grm_c2_cells import args_for
    from core.graft_repository import GraftRepository
    source = Path(c['checkpoint']).parent / 'repository'
    session.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, session / 'repository')
    if loaded is None:
        recorded = []
        original_ctor = e2e.GraftRepository

        def recording_ctor(*a, **kw):
            recorded.append((a, kw))
            return original_ctor(*a, **kw)

        e2e.GraftRepository = recording_ctor
        try:
            model, tokenizer, repo, info = e2e.load_model_and_repo(
                args_for(e2e, session, flags), session)
        finally:
            e2e.GraftRepository = original_ctor
        need(recorded, 'R1_CONSTRUCTOR_NOT_OBSERVED')
        return repo, (model, tokenizer, info, recorded[0])
    model, tokenizer, info, (ctor_args, ctor_kwargs) = loaded
    ctor_args = list(ctor_args)
    ctor_args[3] = str(session / 'repository')
    return GraftRepository(*ctor_args, **ctor_kwargs), loaded


def device_memory_mib():
    """Point-sample of device framebuffer use, in MiB, or None off-GPU.

    Prior art: FIX8 ``grm_scout_fix8_resume.parse_memory`` (GRM, 2026) and
    the NVIDIA nvidia-smi XML framebuffer report (docs.nvidia.com, accessed
    2026-09-09).  TAKEN: read total FB used, which conservatively includes
    unattributed memory.  This is a POINT SAMPLE, never a peak, and it is a
    receipt field only -- nothing gates on it.
    """
    import subprocess
    try:
        proc = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5, check=False)
        if proc.returncode != 0:
            return None
        return int(proc.stdout.strip().splitlines()[0])
    except Exception:                        # noqa: BLE001 -- receipt only
        return None


def release_arm(repo):
    """Return this arm's device memory before the next arm/cell allocates.

    THE R1 AMENDMENT-1 FIX.  ``GraftRepository.close()`` releases only the
    NATIVE STORE (``core/graft_repository.py:368-372``); it never touches the
    arena.  But ``_graft_block`` documents that "Grafts are device-resident
    tc tensors", so every ``grafts[i]['h']`` harvested by an arm stays pinned
    in VRAM after ``close()``.  With a fresh arena per arm and two arms per
    cell, the batch's resident payload count is CUMULATIVE across cells --
    which is exactly what the lead's R1 receipts show (243 node payloads
    across 12 arm-passes before ``cudaMalloc failed: out of memory``).

    The release uses the repository pager's own idiom -- ``g['h'] = None``
    (``_free_retired`` / ``_page`` / ``_mark_payload_missing``) -- so nothing
    novel is invented, followed by the engine's ``tc.empty_cache()`` to
    return the freed blocks to the driver.  Persisted node TEXT, metadata and
    lineage are untouched: only the device payload is dropped, and the
    checkpoint on disk remains the source of truth for the next arm.
    """
    freed = 0
    arena = getattr(repo, 'arena', None)
    if arena is not None:
        try:
            arena.reset_live_cache()         # drops self.caches (live KV)
        except Exception:                    # noqa: BLE001 -- best effort
            pass
        arena._deferred_route_keys = {}
        for g in getattr(arena, 'grafts', []) or []:
            if g.get('h') is not None:
                g['h'] = None                # pager idiom: device payload out
                freed += 1
    try:
        repo.close()                         # native store (its own scope)
    except Exception:                        # noqa: BLE001 -- best effort
        pass
    pooled = False
    try:
        import core.graft_arena as ga
        if hasattr(ga.tc, 'empty_cache'):
            ga.tc.empty_cache()
            pooled = True
    except Exception:                        # noqa: BLE001 -- best effort
        pass
    return {'payloads_freed': freed, 'pool_emptied': pooled}


def run_arm(repo, c, descriptor, flags, arm, session, process_id):
    """One arm: the PRODUCTION ladder, observed exactly as C2 observed it."""
    from scripts import grm_e2e_session as e2e
    from scripts.grm_c2_cells import observe, args_for
    context = descriptor['context']
    started = time.monotonic()
    if c['battery'] == 'sup':
        probe = context['probe']
        answer, info = e2e._probe_ladder_chat(
            repo, probe['question'], topk=flags['topk'], ngen=flags['ngen'],
            max_trips=flags['max_trips'], defer_memory=True)
        row = observe(repo, info, c, c['question_id'], descriptor, process_id)
    else:
        captured = []
        original = e2e._probe_ladder_chat

        def traced(*a, **kw):
            value, info = original(*a, **kw)
            captured.append(observe(repo, info, c, c['question_id'],
                                    descriptor, process_id))
            return value, info

        e2e._probe_ladder_chat = traced
        rows = []
        try:
            e2e.run_turn(repo, context['event'], context['turn'],
                         paths=e2e.stage_paths(session),
                         transcript=copy.deepcopy(context['transcript']),
                         turn_records={int(k): v for k, v
                                       in context['turn_records'].items()},
                         probe_rows=rows, args=args_for(e2e, session, flags),
                         resumed=True)
        finally:
            e2e._probe_ladder_chat = original
        need(len(captured) == 1 and len(rows) == 1,
             'R1_EXPECTED_ONE_SERVED_PROBE: ' + c['id'] + ':' + arm)
        row = captured[0]
        answer = rows[0]['answer']
    wall = time.monotonic() - started
    admission = row['route_receipt']['admission']
    expected, rejected = expected_values(context, c['battery'])
    verdict = score_answer(answer, expected, rejected)
    return {'cell': c['id'], 'arm': arm, 'question_id': c['question_id'],
            'question': c['question'], 'side': c['side'],
            'battery': c['battery'], 'admission_rule': admission['policy'],
            'rank_plan': admission['rank_plan'],
            'policy_branch': admission['policy_branch'],
            'route_margin_1_2': admission['route_margin_1_2'],
            'route_margin_evaluated': admission['route_margin_evaluated'],
            'identified_candidates': admission['identified_candidates'],
            'mounted_ids': row['mounted_ids'],
            'token_seats': row['token_seats'],
            'arena_cur_mount_n': row['arena_cur_mount_n'],
            'geometry': row['geometry'], 'rt1': row['rt1'],
            'capture_valid': row['capture_valid'],
            'answer': str(answer), 'expected_values': expected,
            'rejected_values': rejected, 'verdict': verdict,
            'correct': bool(verdict['correct']),
            'route_receipt': row['route_receipt'],
            'answer_measured': True,
            'wall_seconds': wall, 'checkpoint': c['checkpoint'],
            'checkpoint_sha256': c['checkpoint_sha256'],
            'evidence_class': 'production ladder on a recorded C2 checkpoint; '
                              'the admission rule is the only difference '
                              'between the two arms',
            'registration_sha256': sha(REG)}


def archive_stale_sessions(destination, c):
    """Move a re-issued cell's stale session dirs aside before it runs.

    AMENDMENT-2 DEFECT FIX.  A cell that died part-way (cell 6 of batch R1:
    arm OFF finished, arm ON OOMed) leaves its ``sessions/<cell>/off`` and
    ``.../on`` behind.  ``open_arm`` creates the session with
    ``mkdir(exist_ok=False)``, so the re-issue collided with its own debris:
    ``FileExistsError: .../sessions/<cell>/off``.

    THE DISTINCTION THAT MATTERS: cell RECEIPTS are create-only evidence and
    are never touched -- a retained receipt stays byte-identical, and a cell
    that already HAS a receipt is never re-run, so this function never sees
    it.  Session directories are SCRATCH (a copied repository plus driver
    stage files), so they are ARCHIVED, not deleted: the failed attempt's
    scratch stays inspectable under ``sessions/<cell>/attempt_<n>/``.

    Nothing outside this cell's own session directory is touched.

    Prior art: FIX8/C7 "archive the prior attempt, never overwrite it"
    (GRM contributors, 2026) -- the same rule amendment 1 applies to the
    batch controller, here applied to per-cell scratch.  Nothing novel.
    """
    root = Path(destination) / 'sessions' / c['id']
    stale = [p for p in (root / arm for arm in ARMS) if p.exists()]
    if not stale:
        return None
    attempt = 1
    while (root / f'attempt_{attempt}').exists():
        attempt += 1
    target = root / f'attempt_{attempt}'
    target.mkdir(parents=True, exist_ok=False)
    moved = []
    for p in stale:
        p.rename(target / p.name)
        moved.append(p.name)
    return {'attempt': attempt, 'archived_arms': moved,
            'archived_to': str(target),
            'note': 'scratch session dirs only; no cell receipt is touched'}


class _FakeArena:
    """Minimal arena stand-in so the CPU gate exercises the real release path.

    It holds device-payload-shaped entries (``{'h': ...}``) exactly as the
    real arena does, so ``release_arm`` walks the same branches and its
    ``payloads_freed`` count is a real count, not a hard-coded 0.
    """

    def __init__(self, payloads=3):
        self.grafts = [{'h': object(), 'text': f'node {i}'}
                       for i in range(payloads)]
        self.reset_calls = 0

    def reset_live_cache(self):
        self.reset_calls += 1


class _FakeRepo:
    """Repository stand-in for the CPU gate; ``close`` is a real no-op close."""

    def __init__(self, payloads=3):
        self.arena = _FakeArena(payloads)
        self.closed = False

    def close(self):
        self.closed = True


def run_cell(c, destination, loaded, fake=False):
    """Both arms of one cell from the SAME recorded checkpoint."""
    from scripts.grm_c2_cells import flags_for
    descriptor = verify_cell_inputs(c)
    recorded = recorded_row(c)
    flags = flags_for(c['side'])
    process_id = str(uuid.uuid4())
    arms = {}
    pins = {}
    # AMENDMENT 1: device-memory receipt, so the NEXT run shows whether peak
    # is per-cell or cumulative.  Point samples, never peaks; nothing gates
    # on them.  ``None`` off-GPU (the fake path) rather than a fabricated 0.
    device = {'before_cell_mib': None if fake else device_memory_mib(),
              'arms': {}, 'after_cell_mib': None}
    archived = archive_stale_sessions(destination, c)
    for arm in ARMS:
        session = destination / 'sessions' / c['id'] / arm
        # AMENDMENT 2: the rule pin is scoped.  It mutates os.environ, which
        # admission_rule() reads at call time, so an unrestored pin re-rules
        # every module that runs later in the process.
        with scoped_env():
            pins[arm] = pin_rule(arm)
            if fake:
                arms[arm] = fake_arm(c, descriptor, arm, session)
                device['arms'][arm] = {'before_mib': None, 'after_mib': None,
                                       'released': release_arm(_FakeRepo())}
            else:
                before = device_memory_mib()
                repo, loaded = open_arm(c, session, flags, loaded)
                try:
                    arms[arm] = run_arm(repo, c, descriptor, flags, arm,
                                        session, process_id)
                finally:
                    # Release BEFORE the next arm allocates, not at cell end:
                    # otherwise arm ON runs on arm OFF's resident payloads.
                    released = release_arm(repo)
                device['arms'][arm] = {'before_mib': before,
                                       'after_mib': device_memory_mib(),
                                       'released': released}
    device['after_cell_mib'] = None if fake else device_memory_mib()
    # PARITY BARRIER (RD1 A0 shape): arm OFF must reproduce the FIX-6 recorded
    # plan byte-for-byte.  A mismatch means the replayed frame is NOT the
    # recorded frame, so the contrast would be meaningless: it is a RED
    # receipt and the run STOPS.  Never a retry, never a re-blessing.
    off_bytes = json.dumps(arms['off']['rank_plan'], separators=(',', ':')).encode()
    recorded_bytes = json.dumps(c['off_plan'], separators=(',', ':')).encode()
    parity = off_bytes == recorded_bytes
    on_bytes = json.dumps(arms['on']['rank_plan'], separators=(',', ':')).encode()
    value = {'cell': c, 'arms': arms, 'process_id': process_id,
             'rule_pins': pins, 'device_memory': device,
             'archived_stale_sessions': archived,
             'off_plan_parity': parity,
             'off_plan_bytes_hex': off_bytes.hex(),
             'recorded_off_plan_bytes_hex': recorded_bytes.hex(),
             'on_plan_bytes_hex': on_bytes.hex(),
             'recorded_on_plan': c['on_plan'],
             'on_plan_matches_registered': arms['on']['rank_plan'] == c['on_plan'],
             'recorded_c2_answer': recorded['served_answer'],
             'recorded_c2_correct': bool(recorded['correct']),
             'same_state_proof': {
                 'checkpoint': c['checkpoint'],
                 'checkpoint_sha256': c['checkpoint_sha256'],
                 'checkpoint_files_verified': len(c['checkpoint_files']),
                 'arms_share_one_checkpoint': True,
                 'per_arm_private_copy': True,
                 'differing_environment_keys': [RULE_ENV],
                 'note': 'Both arms copytree the SAME hash-verified recorded '
                         'repository before any model call; GRM_ADMISSION_RULE '
                         'is the only environment difference between them.'},
             **classify(arms['off']['verdict'], arms['on']['verdict']),
             'fake': bool(fake), 'registration_sha256': sha(REG)}
    write(destination / 'cells' / (c['id'] + '.json'), value)
    need(parity, 'R1_OFF_PLAN_PARITY_RED_STOP: ' + c['id']
         + ' replayed=' + str(arms['off']['rank_plan'])
         + ' recorded=' + str(c['off_plan']))
    return value, loaded


# --------------------------------------------------------------------------
# CPU fake-model seam
# --------------------------------------------------------------------------

def frozen_scores(state):
    """A score vector that EXACTLY reproduces the recorded ranking and margin.

    The CPU double's ``_node_key`` is a constant unit vector (by design -- it
    is a numerical boundary stub), so real cosine scores cannot reproduce the
    recorded GPU ranking and ``decisive_admission_profile``'s frozen
    score-reconstruction guard fires.  FIX-6's ``FrozenArena`` solved the same
    problem by replaying stored scores.  The 31 registered states record
    ``ranking`` and ``margin`` but not the full score vector, so this builds
    the vector consistent with both: rank order strictly descending, and
    (rank1 - rank2) equal to the recorded margin.

    TEST SEAM FOR THE CPU GATE ONLY.  It never runs on GPU and it changes no
    admission logic -- ``core.grm_admission`` still computes the plan.  No
    model-quality claim is ever made from a fake-model run.
    """
    ranking = [int(i) for i in state['ranking']]
    eligible = [int(i) for i in state['eligible']]
    margin = float(state['margin'])
    scores = {}
    for position, index in enumerate(ranking):
        scores[index] = -float(position)
    if len(ranking) > 1:
        scores[ranking[1]] = scores[ranking[0]] - margin
        for position, index in enumerate(ranking[2:], start=2):
            scores[index] = scores[ranking[1]] - float(position)
    for index in eligible:
        scores.setdefault(index, -1000.0 - float(index))
    return scores


def fake_arm(c, descriptor, arm, session):
    """CPU arm: REAL core admission over the recorded state; fake numerics."""
    import tempfile
    import _pytest.monkeypatch as monkeypatch_module
    from scripts.grm_c7_diagnose import repository
    from core.grm_admission import decisive_admission_profile
    state = read(ROOT / c['policy_state'])
    patch = monkeypatch_module.MonkeyPatch()
    session.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        repo = repository(Path(tempfile.mkdtemp(dir=session)) / 'repo', patch)
        arena = repo.arena
        try:
            for node in state['nodes']:
                index = arena.deposit(node['text'] or '')
                kept = {k: arena.grafts[index][k] for k in ('h', 'cent', 'ntok')}
                arena.grafts[index].update(copy.deepcopy(node), **kept)
                arena.grafts[index]['rare'] = set(node.get('rare', []))
            arena._bump_cuda_gqa_epoch()
            ranking = [int(i) for i in state['ranking']]
            eligible = [int(i) for i in state['eligible']]
            members = list(state.get('split_members', []))
            scores = frozen_scores(state)
            arena._route_cand_base = lambda: list(eligible)
            arena.route = (lambda question, **kw:
                           list(ranking)[:kw.get('limit', 3)])
            arena._vector_route_scores = (lambda key, candidates:
                                          {int(i): float(scores[int(i)])
                                           for i in candidates})
            arena._length_debias_scores = lambda base, candidates: base
            arena._normalize_scores = lambda base: base
            arena._lex_bonus = lambda *a: 0.0
            arena._split_family_members = lambda r: list(members)
            profile = decisive_admission_profile(
                arena, state['question'], exclude=state['exclude'],
                route_limit=6)
        finally:
            repo.close()
    finally:
        patch.undo()
    expected, rejected = expected_values(descriptor['context'], c['battery'])
    # The CPU double has NO reader, so the answer is NOT measured here: the
    # recorded C2 answer is carried forward so the transition machinery is
    # exercised end to end without any fake-model quality claim.
    answer = recorded_row(c)['served_answer']
    verdict = score_answer(answer, expected, rejected)
    return {'cell': c['id'], 'arm': arm, 'question_id': c['question_id'],
            'question': state['question'], 'side': c['side'],
            'battery': c['battery'],
            'admission_rule': profile['admission_rule'],
            'rank_plan': profile['rank_plan'],
            'policy_branch': profile['policy_branch'],
            'route_margin_1_2': profile['route_margin_1_2'],
            'route_margin_evaluated': profile['route_margin_evaluated'],
            'identified_candidates': profile['identified_candidates'],
            'mounted_ids': [], 'token_seats': None, 'arena_cur_mount_n': None,
            'geometry': None, 'rt1': None, 'capture_valid': None,
            'answer': str(answer), 'expected_values': expected,
            'rejected_values': rejected, 'verdict': verdict,
            'correct': bool(verdict['correct']),
            'route_receipt': None, 'answer_measured': False,
            'wall_seconds': time.monotonic() - started,
            'checkpoint': c['checkpoint'],
            'checkpoint_sha256': c['checkpoint_sha256'],
            'evidence_class': 'CPU fake numerical boundary; REAL core '
                              'admission plan over the recorded state; the '
                              'answer is the recorded C2 answer carried '
                              'forward, NOT a measurement -- no quality claim',
            'registration_sha256': sha(REG)}


# --------------------------------------------------------------------------
# campaign: leases, reservations, resume
# --------------------------------------------------------------------------

def _load_amendment(path):
    """One link: sha-bound, create-only, schema- and order-checked."""
    checksum = path.with_suffix('.sha256')
    need(checksum.is_file(), 'R1_AMENDMENT_SHA_MISSING')
    need(sha(path) == checksum.read_text().split()[0],
         'R1_AMENDMENT_SHA_MISMATCH')
    a = read(path)
    need(a['schema'] == 'grm.r1.amendment.v1', 'R1_AMENDMENT_SCHEMA_MISMATCH')
    need(a['registration_sha256'] == sha(REG), 'R1_AMENDMENT_CHAIN_MISMATCH')
    need(a['order_sha256'] == sha(ROOT / 'orders/GRM_R1_MARGIN_FIRST_REPLAY.md'),
         'R1_AMENDMENT_ORDER_MISMATCH')
    a['_self_sha256'] = sha(path)
    a['_path'] = str(path)
    return a


def amendment(root=None):
    """Load the LATEST amendment in the chain: sha-bound, never inferred.

    Amendments chain: each one binds the registration hash, and every one
    after the first also binds its PREDECESSOR's hash, so a link cannot be
    swapped or skipped.  The newest link is the effective one; its
    ``rebound_inputs`` supersede earlier re-binds of the same path.

    Prior art: GRM C7/C2 sha-bound amendment chains (GRM contributors, 2026).
    TAKEN verbatim: a SEPARATE create-only file bound to the registration
    hash (and, from A2 on, to the previous amendment) which widens scope
    explicitly and never edits the immutable registration.  OURS: the
    "re-arm one FAILED batch for its MISSING cells only, retaining completed
    receipts" scope.
    """
    root = Path(root) if root is not None else OUT
    if not (root / 'amendment_1.json').is_file():
        # An amendment binds the CAMPAIGN (registration + sources), not a
        # particular receipt directory.  A gate or test pointed at an
        # alternate --out still runs the amended worker, so it must see the
        # canonical amendment; otherwise the re-bound source hash would fail
        # closed against the pre-amendment registration.
        if not (OUT / 'amendment_1.json').is_file():
            return None
        root = OUT
    chain = []
    index = 1
    while (root / f'amendment_{index}.json').is_file():
        link = _load_amendment(root / f'amendment_{index}.json')
        need(link['amendment'] == index, 'R1_AMENDMENT_INDEX_MISMATCH')
        if index > 1:
            need(link.get('previous_amendment_sha256')
                 == chain[-1]['_self_sha256'],
                 f'R1_AMENDMENT_{index}_CHAIN_MISMATCH')
        chain.append(link)
        index += 1
    need(chain, 'R1_AMENDMENT_CHAIN_EMPTY')
    latest = dict(chain[-1])
    # Re-binds accumulate along the chain; a later link wins on a path both
    # touched, and an earlier link's re-bind is NOT silently dropped.
    rebound = {}
    for link in chain:
        rebound.update(link.get('rebound_inputs', {}))
    latest['rebound_inputs'] = rebound
    latest['_chain'] = [link['_self_sha256'] for link in chain]
    return latest


def resume_scope(r, root, a=None):
    """Which cells of a re-armed batch still need running, and which are kept.

    A completed cell receipt is a VALID create-only receipt: it is retained
    byte-for-byte and never re-run.  Only cells with no receipt are re-issued.
    """
    explicit = a is not None
    if a is None:
        a = amendment(root)
    scope = {}
    if not a:
        return scope
    root = Path(root) if root is not None else OUT
    # An amendment's retain/reissue lists describe ONE receipt root: the one
    # it was built against.  A gate or test pointed at a different --out is
    # running the amended WORKER (so the re-bound source hashes apply) but
    # not the amended CAMPAIGN, so there is no re-arm scope to honour there.
    # Passing the amendment explicitly overrides this, for fixtures that
    # deliberately construct a matching state.
    if not explicit and a.get('_path') and Path(a['_path']).parent != root:
        return scope
    for batch_id, entry in a['rearm'].items():
        need(batch_id in r['batches'], 'R1_AMENDMENT_UNKNOWN_BATCH: ' + batch_id)
        d = root / 'gpu' / batch_id
        retained, missing = [], []
        for cell_id in r['batches'][batch_id]:
            p = d / 'cells' / (cell_id + '.json')
            if p.exists():
                row = read(p)
                need(row['registration_sha256'] == sha(REG),
                     'R1_RETAINED_RECEIPT_BINDING_MISMATCH: ' + cell_id)
                need(row['off_plan_parity'],
                     'R1_RETAINED_RECEIPT_NOT_PARITY_CLEAN: ' + cell_id)
                retained.append(cell_id)
            else:
                missing.append(cell_id)
        # AMENDMENT-3 FIX.  The retain/reissue cross-check is a PRECONDITION
        # for re-issuing an OPEN batch: it asks "is the state I am about to
        # act on still the state this amendment was written against?".  Once
        # the batch's controller says COMPLETE the re-issue has happened and
        # the amendment's scope is SATISFIED -- every registered cell now has
        # a receipt, which is the outcome the amendment existed to produce.
        # Re-applying the check then compares the finished state against the
        # pre-re-issue expectation and closes the campaign on its own success
        # (the lead hit exactly this: R1 COMPLETE with 8/8 receipts,
        # on_disk=8 vs amendment retain=5, and R2 refused to start).
        controller = d / 'controller.json'
        done = controller.exists() and read(controller)['status'] == 'COMPLETE'
        if done:
            # Satisfied means satisfied ON THE REGISTERED COHORT: every cell
            # has a parity-clean, registration-bound receipt.  Anything less
            # is still a genuine disagreement and still STOPS.
            need(not missing,
                 'R1_AMENDMENT_SCOPE_UNSATISFIED: ' + batch_id
                 + ' controller=COMPLETE but missing=' + str(missing))
            scope[batch_id] = {
                'retain': retained, 'reissue': [],
                'lease_seconds': entry['lease_seconds'],
                'scope_satisfied': True,
                'receipts': {cell_id: sha(d / 'cells' / (cell_id + '.json'))
                             for cell_id in retained},
                'note': ('re-issue complete; the amendment\'s retain/reissue '
                         'precondition no longer applies to this batch')}
            continue
        # Still OPEN: the amendment states what it expects to find.  If the
        # receipts on disk disagree, that is a RED stop, not a silent
        # re-scope -- the amendment was written against a state that no
        # longer holds, and a new amendment is required.
        need(retained == entry['retain'],
             'R1_AMENDMENT_RETAIN_MISMATCH: ' + batch_id
             + ' on_disk=' + str(retained) + ' amendment=' + str(entry['retain']))
        need(missing == entry['reissue'],
             'R1_AMENDMENT_REISSUE_MISMATCH: ' + batch_id
             + ' on_disk=' + str(missing) + ' amendment=' + str(entry['reissue']))
        scope[batch_id] = {'retain': retained, 'reissue': missing,
                           'lease_seconds': entry['lease_seconds'],
                           'scope_satisfied': False}
    return scope


def campaign_state(r, root):
    """Charged seconds so far; fail closed on orphans and failed batches.

    A FAILED batch normally closes the campaign (``R1_FAILED_CAMPAIGN_STOP``).
    Amendment 1 UN-CLOSES exactly the batches it re-arms, and nothing else:
    an unlisted FAILED batch still stops everything.
    """
    gpu = root / 'gpu'
    charged = 0.0
    complete = []
    a = amendment(root)
    # Un-closing is CAMPAIGN state, so it applies only to the receipt root the
    # amendment was built against -- the same rule resume_scope uses.  A gate
    # or test pointed at a different --out gets the unamended fail-closed
    # behaviour, which is what it is there to check.
    if a and a.get('_path') and Path(a['_path']).parent != Path(root):
        a = None
    rearmed = set(a['rearm']) if a else set()
    if not gpu.exists():
        return charged, complete
    need(all(p.name in r['batches'] and p.is_dir() for p in gpu.iterdir()),
         'R1_UNKNOWN_BATCH_RECEIPT')
    for p in sorted(gpu.glob('*/reservation.json')):
        need(p.with_name('controller.json').exists()
             or p.parent.name in rearmed,
             'R1_ORPHAN_RESERVATION_STOP: ' + p.parent.name)
    # AMENDMENT-2 ACCOUNTING FIX: an ARCHIVED attempt's charge is still real
    # GPU time and stays on the books.  Reading only controller.json would
    # silently refund every re-issued attempt -- batch R1 showed 134 s after
    # its second attempt, having actually spent 263 + 134 = 397 s.
    for p in sorted(gpu.glob('*/controller_attempt_*.json')):
        archived = read(p)
        need(archived['registration_sha256'] == sha(REG),
             'R1_ARCHIVED_RESULT_BINDING_MISMATCH: ' + p.parent.name)
        charged += float(archived['charged_seconds'])
    for p in sorted(gpu.glob('*/controller.json')):
        controller = read(p)
        need(controller['registration_sha256'] == sha(REG),
             'R1_RESULT_BINDING_MISMATCH: ' + p.parent.name)
        # The failed batch's charge is REAL and stays on the books; only its
        # campaign-stopping effect is lifted, and only for a re-armed batch.
        charged += float(controller['charged_seconds'])
        if controller['status'] == 'COMPLETE':
            complete.append(p.parent.name)
        else:
            need(p.parent.name in rearmed,
                 'R1_FAILED_CAMPAIGN_STOP: ' + p.parent.name)
    return charged, complete


def _gpu_lease(seconds):
    from scripts.grm_cmc1_gpu_arms import gpu_lease
    return gpu_lease(int(seconds), 0)


def _cpu_context():
    from contextlib import nullcontext
    return nullcontext()


def batch(batch_id, fake=False, root=None):
    """Run one batch of cells under ONE foreground GPU lease.

    Resumable: a batch whose controller already says COMPLETE is skipped and
    returns; a create-only ``mkdir(exist_ok=False)`` makes a half-written
    batch fail closed rather than silently re-run (C7 no-retry discipline).
    """
    root = Path(root) if root is not None else OUT
    r = verify(root)
    need(batch_id in r['batches'], 'R1_UNKNOWN_BATCH: ' + str(batch_id))
    destination = root / 'gpu' / batch_id
    scope = resume_scope(r, root).get(batch_id)
    # A SATISFIED scope is a record, not a re-issue instruction: the batch it
    # describes is already COMPLETE, so it must not put this run on the
    # amendment's re-issue path (archive controllers, narrow the cell list).
    if scope is not None and scope.get('scope_satisfied'):
        scope = None
    if (destination / 'controller.json').exists():
        controller = read(destination / 'controller.json')
        if controller['status'] == 'COMPLETE':
            print(json.dumps({'status': 'ALREADY_COMPLETE', 'batch': batch_id}))
            return 0
        # A FAILED batch is re-issuable ONLY under a matching amendment, and
        # only for the cells that have no receipt.  Its old controller is
        # preserved under an attempt name; receipts are never overwritten.
        need(scope is not None, 'R1_FAILED_CAMPAIGN_STOP: ' + batch_id)
    owner = root / 'replay.active'
    owner.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(owner, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(handle, str(os.getpid()).encode())
    os.close(handle)
    try:
        charged, complete = campaign_state(r, root)
        index = r['batch_ids'].index(batch_id)
        for prior in r['batch_ids'][:index]:
            need(prior in complete, 'R1_PRIOR_BATCH_INCOMPLETE: ' + prior)
        attempt = 1
        if scope is None:
            lease = int(r['batch_lease_seconds'][batch_id])
        else:
            lease = int(scope['lease_seconds'])
            while (destination / f'controller_attempt_{attempt}.json').exists():
                attempt += 1
        need(charged + lease <= r['gpu_cap_seconds'], 'R1_GPU_BUDGET_RAIL')
        need(shutil.disk_usage(ROOT).free >= r['free_space_min_bytes'],
             'R1_FREE_SPACE_BELOW_MINIMUM')
        if scope is None:
            destination.mkdir(parents=True, exist_ok=False)
            write(destination / 'reservation.json',
                  {'seconds': lease, 'batch': batch_id,
                   'registration_sha256': sha(REG)})
        else:
            # Archive the prior attempt's receipts create-only; the RETAINED
            # cell receipts are left exactly where they are, byte-for-byte.
            attempt += 1
            for name in ('controller.json', 'reservation.json'):
                p = destination / name
                if p.exists():
                    p.rename(destination
                             / f'{p.stem}_attempt_{attempt - 1}{p.suffix}')
            write(destination / 'reservation.json',
                  {'seconds': lease, 'batch': batch_id, 'attempt': attempt,
                   'amendment': 'amendment_1',
                   'registration_sha256': sha(REG)})
        started = None
        elapsed = 0.0
        status = 'FAILED'
        error = None
        loaded = None
        red = []
        ids = []
        try:
            # FIX-6 state contract: environment(flags) strips every ambient
            # GRM_ var, so the rule is pinned AFTER it, per arm, in pin_rule.
            from scripts.grm_c2_cells import environment, flags_for
            ids = (r['batches'][batch_id] if scope is None
                   else list(scope['reissue']))
            need(ids, 'R1_NOTHING_TO_REISSUE: ' + batch_id)
            sides = {cell_by_id(i)['side'] for i in ids}
            need(len(sides) == 1, 'R1_MIXED_SIDE_BATCH: ' + batch_id)
            env = environment(flags_for(sides.pop()))
            # AMENDMENT 2: the whole frame pin is scoped too, so a batch (in
            # a test, a gate, or any caller that runs more afterwards) cannot
            # leave the process re-framed or re-ruled on the way out.
            with scoped_env():
                for key in list(os.environ):
                    if key.startswith('GRM_'):
                        del os.environ[key]
                os.environ.update({k: v for k, v in env.items()
                                   if k.startswith('GRM_')})
                with (_cpu_context() if fake else _gpu_lease(lease)):
                    started = time.monotonic()
                    for cell_id in ids:
                        try:
                            _, loaded = run_cell(cell_by_id(cell_id),
                                                 destination, loaded, fake=fake)
                        except ValueError as exc:
                            if 'R1_OFF_PLAN_PARITY_RED_STOP' in str(exc):
                                red.append(str(exc))
                            raise
                    elapsed = time.monotonic() - started
                    need(fake or elapsed <= lease, 'R1_LEASE_OVERRUN_RED')
                    status = 'COMPLETE'
        except BaseException as exc:            # noqa: BLE001 -- receipt first
            error = f'{type(exc).__name__}: {exc}'
        finally:
            if started is not None:
                elapsed = time.monotonic() - started
            write(destination / 'controller.json',
                  {'status': status, 'error': error, 'red': red,
                   'elapsed_seconds': elapsed,
                   'charged_seconds': (elapsed if status == 'COMPLETE'
                                       else max(float(lease), elapsed)),
                   'reservation_seconds': lease, 'fake': bool(fake),
                   'attempt': attempt,
                   'amendment': None if scope is None else 'amendment_1',
                   'cells_run': list(ids),
                   'cells_retained': [] if scope is None else list(scope['retain']),
                   'registration_sha256': sha(REG)})
            if not fake:
                time.sleep(30)  # foreground cooldown, outside the lease
        if status != 'COMPLETE':
            raise RuntimeError(error)
    finally:
        owner.unlink()          # only this process's own O_EXCL file
    return 0


# --------------------------------------------------------------------------
# dry run + summary
# --------------------------------------------------------------------------

def dry_run():
    r = verify()
    rows = []
    for c in cells():
        verify_cell_inputs(c)
        rows.append({'id': c['id'], 'side': c['side'], 'battery': c['battery'],
                     'question_id': c['question_id'],
                     'off_plan': c['off_plan'], 'on_plan': c['on_plan'],
                     'estimate_seconds': r['per_cell_estimate_seconds'][c['id']],
                     'checkpoint_sha256': c['checkpoint_sha256'],
                     'checkpoint_files': len(c['checkpoint_files'])})
    return {'status': 'READY_FOR_LEAD', 'gpu_executed': False,
            'registration_sha256': sha(REG), 'cells': rows,
            'cell_count': len(rows), 'batches': r['batches'],
            'batch_ids': r['batch_ids'],
            'batch_lease_seconds': r['batch_lease_seconds'],
            'reserved_seconds': r['reserved_seconds'],
            'reserved_gpu_hours': round(r['reserved_seconds'] / 3600.0, 4),
            'gpu_cap_seconds': r['gpu_cap_seconds'],
            'estimated_seconds': sum(row['estimate_seconds'] for row in rows),
            'prediction': r['prediction'], 'verdict_rule': r['verdict_rule'],
            'measurement_status': 'NOT_RUN'}


def summary(root=None):
    root = Path(root) if root is not None else OUT
    r = verify(root)
    values = {}
    complete = True
    charged = 0.0
    for batch_id in r['batch_ids']:
        d = root / 'gpu' / batch_id
        controller = d / 'controller.json'
        if not controller.exists():
            complete = False
        else:
            value = read(controller)
            need(value['registration_sha256'] == sha(REG),
                 'R1_RESULT_BINDING_MISMATCH: ' + batch_id)
            complete &= value['status'] == 'COMPLETE'
            charged += float(value['charged_seconds'])
        # Archived attempts are real GPU time too (amendment 2's accounting
        # fix, applied here as well so summary and campaign_state agree).
        for p in sorted(d.glob('controller_attempt_*.json')):
            archived = read(p)
            need(archived['registration_sha256'] == sha(REG),
                 'R1_ARCHIVED_RESULT_BINDING_MISMATCH: ' + batch_id)
            charged += float(archived['charged_seconds'])
        for cell_id in r['batches'][batch_id]:
            p = d / 'cells' / (cell_id + '.json')
            if not p.exists():
                complete = False
                continue
            row = read(p)
            need(row['registration_sha256'] == sha(REG),
                 'R1_ROW_BINDING_MISMATCH: ' + cell_id)
            values[cell_id] = row
    complete &= len(values) == r['cell_count'] and charged <= r['gpu_cap_seconds']
    parity = bool(values) and all(v['off_plan_parity'] for v in values.values())
    counts = {k: 0 for k in TRANSITIONS}
    for v in values.values():
        counts[v['transition']] += 1
        if v['abstain_transition']:
            counts[v['abstain_transition']] += 1
    sup = [v for v in values.values() if v['cell']['battery'] == 'sup']
    other = [v for v in values.values() if v['cell']['battery'] != 'sup']
    sup_c2w = sum(v['transition'] == 'correct_to_wrong' for v in sup)
    other_c2w = sum(v['transition'] == 'correct_to_wrong' for v in other)
    changed = sum(v['answer_changed'] for v in values.values())
    fake = any(v.get('fake') for v in values.values())
    measured = all(a['answer_measured'] for v in values.values()
                   for a in v['arms'].values()) if values else False
    # The registered verdict rule applied VERBATIM -- never adjusted after
    # seeing results (house rule: thresholds registered before the gate).
    adopt = complete and parity and measured and sup_c2w == 0 and other_c2w <= 1
    return {'status': ('NOT_MEASURED' if not (complete and measured) else
                       'RED' if not parity else
                       'ADOPT' if adopt else 'DO_NOT_ADOPT'),
            'complete': complete, 'answers_measured': measured,
            'cells_measured': len(values), 'off_plan_parity': parity,
            'charged_seconds': charged,
            'charged_gpu_hours': round(charged / 3600.0, 4),
            'answers_changed': changed, 'transitions': counts,
            'correct_to_wrong_sup': sup_c2w,
            'correct_to_wrong_other': other_c2w,
            'prediction': r['prediction'],
            'prediction_held': ((changed <= 2 and sup_c2w == 0)
                                if (complete and measured) else None),
            'verdict_rule': r['verdict_rule'], 'adopt': adopt,
            'fake_run': fake,
            'scope': 'Admission-rule replay on 31 recorded C2 executions over '
                     '9 distinct questions. Repeated question ids across '
                     'arm/phase are retained as separate executions. Not a '
                     'fresh battery, not a product certification.'
            + (' FAKE CPU RUN: plans are real, answers are NOT measured.'
               if fake else '')}


def main():
    p = argparse.ArgumentParser(description='GRM-R1 margin-first replay')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--batch')
    p.add_argument('--summary', action='store_true')
    p.add_argument('--fake', action='store_true',
                   help='CPU fake-model gate: real admission, no GPU, no reader')
    p.add_argument('--out', help='alternate receipt root (gates/tests only)')
    a = p.parse_args()
    if a.dry_run:
        print(json.dumps(dry_run(), indent=2))
        return 0
    if a.summary:
        print(json.dumps(summary(a.out), indent=2))
        return 0
    if a.batch:
        return batch(a.batch, fake=a.fake, root=a.out)
    r = verify()
    print(json.dumps({'status': 'REGISTERED_NOT_RUN', 'gpu_executed': False,
                      'registration_sha256': sha(REG),
                      'cells': r['cell_count'],
                      'reserved_seconds': r['reserved_seconds'],
                      'gpu_cap_seconds': r['gpu_cap_seconds']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
