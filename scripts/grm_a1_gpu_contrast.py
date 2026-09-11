"""GRM-A1 — registered alias fold-merge GPU contrast worker.

WHO RUNS THIS.  The LEAD, on the card.  The implementation seat is forbidden
to touch the GPU by its order, so the seat gates this worker on ``--fake``,
which drives EVERY branch below against the ``grm_c7_diagnose`` CPU doubles.
``--fake`` and the real path share one code path: the ONLY difference is
which ``load`` function builds the repository (``_load_fake`` vs
``_load_gpu``) and which tokenizer measures the width receipt.  Everything
else — the drift check, the resume ledger, the merge, the width receipt, the
serve, the scoring, the create-only receipts — is the same code in both
modes, so a green fake run exercises the real control flow.

CONTRACT (from artifacts/grm_a1/gpu_contrast_registration.json, sha-bound):

  1. SHA DRIFT CHECK FIRST.  Every pinned input is re-hashed before any
     model load or lease.  A mismatch is a hard RED (``A1_INPUT_SHA_MISMATCH``)
     and nothing else runs.
  2. LEASED.  ``grm_cmc1_gpu_arms.gpu_lease`` — the same helper the C7/R1
     workers import — wraps the GPU section.  The operator has absolute
     right of way; the lease waits on the flock and never kills anything.
  3. RESUMABLE.  One receipt file per row, written create-only ('x').  A row
     whose receipt already exists is SKIPPED, so an interrupted run resumes
     by re-invocation with no flag and no cleanup.  Receipts are never
     overwritten, so a resumed run cannot silently rewrite history.
  4. FLAG PINNED ON AFTER ``environment(flags)``.  ``grm_c2_cells.environment``
     strips every ``GRM_*`` key and re-pins a FIXED set that contains neither
     ``GRM_ALIAS_FOLD_MERGE`` nor ``GRM_ADMISSION_RULE``.  Both are therefore
     pinned AFTER it, the way ``scripts/grm_scout_fix6_replay.py:55`` records
     the contract and ``grm_lt1_worker.py:245`` implements it.  Pinning
     before ``environment()`` would be silently erased — that exact mistake
     produced the retracted LT1 RED in this arc, and it is why this worker
     also ASSERTS the resolved values after pinning.
  5. WIDTH RECEIPT IN GPT-OSS TOKENS BEFORE ANY SINGLE-MOUNT CLAIM.  Each
     merged digest is re-tokenized with the loaded tokenizer and compared to
     the arena width.  Over-width digests take the existing LSR-P2C width
     guard and the receipt says ``fits: false`` with the measured count.
  6. PLAIN PRODUCTION PROMPT.  Reads use the C7 r3 effective question (the A2
     form, no abstention clause), matching what the RD2 receipts record.

ROWS (23): 8 RD2 rows + 10 answerable C7 r3 alias probes + 5 C7 r3
unanswerable alias controls, all from the pinned C7 r3 checkpoints.

Prior art.
  * ``scripts/grm_cmc1_gpu_arms.gpu_lease`` (GRM contributors, 2026) —
    imported, not reimplemented.
  * ``scripts/grm_c2_cells.environment`` (GRM, 2026) — the flag frame; this
    worker adds only the two post-``environment`` pins named above.
  * ``scripts/grm_c7_common.create`` (GRM, 2026) — create-only receipts,
    used verbatim; it is the resume ledger.
  * ``scripts/grm_c7_run.effective_question`` (GRM, 2026) — the plain-prompt
    projection, reused rather than retyped.
  * ``scripts/grm_rs1_read_strength_gpu._load_lived_repo`` (GRM, 2026) — the
    lived GPU repository construction, delegated to.
  * ``scripts/grm_c7_diagnose`` (GRM, 2026) — the CPU doubles behind
    ``--fake``.
  * No prior art known to me for this exact composition (a one-file worker
    whose fake path is the gate for its own GPU path).  The alias merge
    itself is annotated in ``core/grm_alias_fold.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_c7_common import create, read, sha  # noqa: E402

REGISTRATION = ROOT / 'artifacts/grm_a1/gpu_contrast_registration.json'
DEFAULT_OUT = ROOT / 'artifacts/grm_a1/gpu'

#: Both pins live AFTER environment(); see the module docstring, item 4.
FLAG_ENV = 'GRM_ALIAS_FOLD_MERGE'
RULE_ENV = 'GRM_ADMISSION_RULE'

#: HOUSE LEASE CAP. ``grm_cmc1_gpu_arms.MAX_LEASE_SECONDS`` is 590 and
#: ``gpu_lease`` REFUSES anything above it. The first version of this worker
#: defaulted to 2400 and the lead's first GPU attempt died on
#: ``LiveError: lease must be in 1..590s, got 2400`` before a single row ran.
#: 560 is the value the C7/LT1 workers use, leaving ~30 s of headroom under
#: the cap. The rows are resumable and create-only, so a short lease costs
#: nothing: the operator re-runs the same line until pending=0.
LEASE_SECONDS = 560
LOCK_WAIT_SECONDS = 7200

#: Device budget for NODE PAYLOADS (not the model). Arms the pager that
#: ``vram_budget_mb=None`` leaves disabled: with a budget set,
#: ``load()`` stops materialising payloads past it and ``_page()`` spills the
#: least-recently-mounted ones, while ``arena._ensure_h`` pages a mounted node
#: back in through ``node_loader``.
#:
#: 64 MiB is ~14x the largest possible single mount plan (arena_width 96
#: tokens = 4.5 MiB at this dialect's 8 kv heads x 64 head_dim x (K,V) x 24
#: layers x fp16), so a plan always fits with room for the fold's sources,
#: and ~10x under the ~1.3 GB the resident model leaves free. The d250 rows
#: would otherwise pin ~659 MB each — see artifacts/grm_a1/
#: payload_accounting.json.
NODE_VRAM_BUDGET_MB = 64


def _lease_seconds(value):
    """argparse type: reject an over-cap lease AT PARSE TIME.

    Failing here rather than inside ``gpu_lease`` means an impossible lease
    is refused before the flock is even attempted, and the message names the
    cap instead of surfacing as a mid-run LiveError.
    """
    from scripts.grm_cmc1_gpu_arms import MAX_LEASE_SECONDS

    seconds = int(value)
    if not 1 <= seconds <= MAX_LEASE_SECONDS:
        raise argparse.ArgumentTypeError(
            f'lease must be in 1..{MAX_LEASE_SECONDS}s (house cap), '
            f'got {seconds}')
    return seconds


class ContrastError(RuntimeError):
    """A registered precondition failed. Always fatal, never retried."""


# --------------------------------------------------------------- 1. drift

def load_amendments(registration_path):
    """Walk the sha-bound amendment chain in order.

    Each amendment is a SEPARATE json + .sha256 pair that names the
    registration it binds to.  An amendment whose own checksum file
    disagrees, or that binds to a different registration, is a hard RED —
    the same chain discipline ``grm_c7_common.verify_lead_1`` applies.
    Prior art: that function (GRM contributors, 2026); taken verbatim in
    shape, with the C7-specific scope allowlists left behind.
    """
    base = Path(registration_path)
    base_sha = sha(base)
    out = []
    index = 1
    while True:
        path = base.parent / f'gpu_contrast_amendment_{index}.json'
        checksum = path.with_suffix('.sha256')
        if not path.is_file():
            break
        if not checksum.is_file():
            raise ContrastError(f'A1_AMENDMENT_CHECKSUM_MISSING: {path}')
        if sha(path) != checksum.read_text().split()[0]:
            raise ContrastError(f'A1_AMENDMENT_SHA_MISMATCH: {path}')
        amendment = read(path)
        if amendment.get('registration_sha256') != base_sha:
            raise ContrastError(
                f'A1_AMENDMENT_CHAIN_MISMATCH: {path} binds to '
                f'{amendment.get("registration_sha256")}, registration is '
                f'{base_sha}')
        out.append({'path': str(path), 'sha256': sha(path),
                    'amendment': amendment})
        index += 1
    return base_sha, out


#: This file pins every OTHER input, and the amendment pins this file.  It is
#: excluded from its own drift set and checked separately by
#: ``verify_self`` — otherwise the worker could never be edited without a
#: re-amendment in the same breath, and a stale self-hash would make the
#: drift check fail on the one file the operator can see is correct.  The
#: separation also makes the two failures distinguishable in the receipt:
#: "an input moved" is a different fact from "the worker is not the reviewed
#: bytes", and only the second is a provenance problem.
SELF_PATH = Path(__file__).resolve()


def effective_pinned_inputs(registration, amendments):
    """The LAST amendment's input set wins; the registration's is the base.

    The worker's own path is removed — see ``SELF_PATH``/``verify_self``.
    """
    pinned = dict(registration['pinned_inputs_sha256'])
    for entry in amendments:
        pinned.update(entry['amendment'].get('pinned_inputs_sha256', {}))
    pinned.pop(str(SELF_PATH), None)
    return pinned


#: Declaration blocks an amendment may ADD to the registration. Amendment 2
#: introduced all three: the registration originally declared no runtime
#: frame, no flags source and no preconditions, which is why the lead's runs
#: discovered them one lease at a time.
AMENDABLE_BLOCKS = ('runtime_frame', 'c7_r3_registration', 'preconditions')


def effective_registration(registration, amendments):
    """The registration as the LAST amendment leaves it.

    Later amendments override earlier ones block by block; the registration
    is the base. Everything the worker reads (frame, flags source,
    preconditions) goes through here, so a declaration added by amendment is
    indistinguishable from one that was in the registration all along.
    """
    effective = dict(registration)
    for entry in amendments:
        amendment = entry['amendment']
        for block in AMENDABLE_BLOCKS:
            if block in amendment:
                effective[block] = amendment[block]
    return effective


def verify_self(amendments):
    """Report whether this file matches the sha the amendment pinned.

    NOT fatal on its own: the lead may legitimately be running a worker that
    has been edited since the amendment was written (that is what an
    amendment is FOR).  It is recorded in every row receipt so a reader can
    always tell which bytes produced the numbers.
    """
    want = None
    for entry in amendments:
        worker = entry['amendment'].get('worker') or {}
        if worker.get('sha256'):
            want = worker['sha256']
    got = sha(SELF_PATH)
    return {'path': str(SELF_PATH), 'sha256': got,
            'pinned_sha256': want, 'matches_amendment': (want == got)}


def verify_pinned_inputs(pinned):
    """Re-hash every pinned input BEFORE any model load or lease."""
    drift = []
    for path, want in sorted(pinned.items()):
        p = Path(path)
        got = sha(p) if p.is_file() else None
        if got != want:
            drift.append({'path': path, 'want': want, 'got': got})
    if drift:
        raise ContrastError('A1_INPUT_SHA_MISMATCH: '
                            + json.dumps(drift, sort_keys=True))
    return len(pinned)


# ---------------------------------------------------------------- 2. rows

def plan_rows(registration):
    """The 23 registered rows, in a stable order, each with its checkpoint.

    RD2 rows carry ``class='alias'`` and ``answerable=True``: they are the 8
    rows the registered prediction (>= 5/8) is scored on.  The C7 r3 rows add
    10 answerable probes and 5 unanswerable controls; every control that
    moves is a STOP.
    """
    rows = []
    c7_by_probe = {r['probe_id']: r for r in registration['c7_r3_rows']}
    for row in registration['rd2_rows']:
        pinned = c7_by_probe.get(row['probe_id'], {})
        rows.append({
            'row_id': 'rd2::' + row['probe_id'],
            'group': 'rd2',
            'probe_id': row['probe_id'],
            'class': 'alias',
            'answerable': True,
            'scored_for_prediction': True,
            'question': row['question'],
            'expected': row['expected'],
            'recorded_served': row.get('recorded_served'),
            'recorded_mounted_ids': row.get('recorded_mounted_ids'),
            'cell': row.get('cell') or pinned.get('cell'),
            'checkpoint_dir': pinned.get('checkpoint_dir'),
        })
    for row in registration['c7_r3_rows']:
        rows.append({
            'row_id': 'c7::' + row['probe_id'],
            'group': 'c7_r3',
            'probe_id': row['probe_id'],
            'class': 'alias',
            'answerable': bool(row['answerable']),
            'scored_for_prediction': False,
            'question': row['question'],
            'expected': row['expected'],
            'cell': row['cell'],
            'checkpoint_dir': row['checkpoint_dir'],
        })
    return rows


# ------------------------------------- 3. flags, frame and preconditions
#
# ALL THREE of the 2026-09-11 lead-run stops live in this section, and they
# share one cause: the fake arm diverged from the GPU arm AT THE LOADER, so
# the frame read, the native-runtime dependency and `environment(flags)` were
# never exercised on CPU. The fix is not three patches — it is making both
# arms call the SAME `resolve_flags` / `build_environment` /
# `load_runtime_frame`, so a shape error fails on the CPU gate.

def resolve_flags(registration):
    """The frame flags for the pinned C7 r3 checkpoints. READ, never typed.

    Source of truth: the C7 r3 registration's own ``arms.A.flags`` — the
    exact frame those checkpoints were produced under.  Hand-assembling a
    flags dict here (or borrowing the RS1 battery's ``resolved_flags``, which
    is what the first version did) silently measures a DIFFERENT arena than
    the checkpoint lived in.  Measured consequence of getting this wrong:
    ``KeyError: 'ephemeral'`` inside ``grm_c2_cells.environment``, because the
    RS1 frame carries no C2 frame keys at all.

    This one dict satisfies BOTH consumers — ``environment()`` (which needs
    ephemeral / capture_pin / seat_near_live / lsr_fixes / rt1_rule /
    demand_ngh / gqa_cuda_route / graft_storage_bits / route_query_lex /
    probe_ladder / sup_resolve / adm_decisive) and
    ``rs1._load_lived_repo`` (arena_width / topk / live_turns / max_live /
    graft_storage_bits / sup_resolve / adm_decisive) — which is why one
    shared accessor is enough and a second one would be a bug waiting.
    """
    path = Path(registration['c7_r3_registration']['path'])
    if not path.is_file():
        raise ContrastError(f'A1_C7_REGISTRATION_MISSING: {path}')
    want = registration['c7_r3_registration'].get('sha256')
    got = sha(path)
    if want and want != got:
        raise ContrastError(
            f'A1_C7_REGISTRATION_SHA_MISMATCH: {path} want={want} got={got}')
    arm = registration['c7_r3_registration'].get('arm', 'A')
    flags = read(path)['arms'][arm]['flags']
    missing = sorted(ENVIRONMENT_FLAG_KEYS - set(flags))
    if missing:
        raise ContrastError(
            f'A1_FLAGS_INCOMPLETE: {path} arm {arm} lacks {missing}')
    return dict(flags)


#: Exactly the keys ``grm_c2_cells.environment`` indexes. Checked up front so
#: a missing one is a NAMED error here instead of a raw KeyError deep inside
#: a GPU load, after the lease is already held.
ENVIRONMENT_FLAG_KEYS = frozenset((
    'ephemeral', 'capture_pin', 'seat_near_live', 'lsr_fixes', 'rt1_rule',
    'demand_ngh', 'gqa_cuda_route', 'graft_storage_bits', 'route_query_lex',
    'probe_ladder', 'sup_resolve', 'adm_decisive',
))


def build_environment(flags):
    """``grm_c2_cells.environment(flags)`` — called identically by both arms.

    Imported, never reimplemented. The fake arm calls THIS, with the SAME
    flags object the GPU arm uses, so a shape error fails on CPU.
    """
    from scripts.grm_c2_cells import environment

    return environment(flags)


def load_runtime_frame(registration):
    """The frozen runtime frame, sha-pinned by the registration."""
    declared = registration.get('runtime_frame') or {}
    path = Path(declared.get('path') or '')
    if not path.is_file():
        raise ContrastError(f'A1_RUNTIME_FRAME_MISSING: {path}')
    want = declared.get('sha256')
    got = sha(path)
    if want and want != got:
        raise ContrastError(
            f'A1_RUNTIME_FRAME_SHA_MISMATCH: {path} want={want} got={got}')
    return json.loads(path.read_text(encoding='utf-8'))


def check_preconditions(registration, *, fake=False):
    """Every declared input and PRECONDITION, checked BEFORE any lease.

    A precondition is a path the run needs that is NOT sha-pinned because it
    is not evidence — the native runtime and the model snapshot.  They were
    undeclared in the first registration and the run reached the card twice
    before failing on them (``OSError: cpp/build/libgrm_runtime.so``).  This
    reports EVERY missing one at once rather than one per GPU attempt.

    ``fake=True`` skips the GPU-only preconditions, because the CPU doubles
    load neither the native runtime nor the model.
    """
    rows = []

    def note(kind, name, path, *, ok, detail=''):
        rows.append({'kind': kind, 'name': name, 'path': str(path),
                     'ok': bool(ok), 'detail': detail})

    frame = (registration.get('runtime_frame') or {})
    frame_path = Path(frame.get('path') or '')
    frame_ok = frame_path.is_file()
    detail = ''
    if frame_ok and frame.get('sha256') and sha(frame_path) != frame['sha256']:
        frame_ok, detail = False, 'sha mismatch'
    note('input', 'runtime_frame', frame_path, ok=frame_ok, detail=detail)

    c7 = (registration.get('c7_r3_registration') or {})
    c7_path = Path(c7.get('path') or '')
    c7_ok = c7_path.is_file()
    detail = ''
    if c7_ok and c7.get('sha256') and sha(c7_path) != c7['sha256']:
        c7_ok, detail = False, 'sha mismatch'
    note('input', 'c7_r3_registration', c7_path, ok=c7_ok, detail=detail)

    for name, spec in sorted(
            (registration.get('preconditions') or {}).items()):
        if spec.get('gpu_only') and fake:
            note('precondition', name, spec.get('path', ''), ok=True,
                 detail='skipped on --fake')
            continue
        path = Path(spec.get('path') or '')
        note('precondition', name, path,
             ok=(path.is_dir() if spec.get('directory') else path.is_file()),
             detail=spec.get('note', ''))

    for row in registration['c7_r3_rows']:
        path = Path(row['checkpoint_dir'])
        if not path.is_dir():
            note('checkpoint', row['probe_id'], path, ok=False)
    missing = [r for r in rows if not r['ok']]
    return {'checks': rows, 'missing': missing, 'ok': not missing}


# ------------------------------------------------------- 4. repository load

def _load_gpu(checkpoint_dir, registration, cache=None):
    """The lived GPU build over a COPY of the pinned checkpoint.

    The checkpoint is copied to a scratch directory first: the pinned C7 r3
    checkpoints are immutable evidence for another order and this worker must
    not write through to them, even though it only intends to read.
    """
    from scripts import grm_rs1_read_strength_gpu as rs1

    scratch = Path(tempfile.mkdtemp(prefix='grm_a1_'))
    shutil.copytree(Path(checkpoint_dir) / 'repository',
                    scratch / 'repository')
    # THE SAME flags object and THE SAME environment() call the fake arm
    # makes — see resolve_flags(). Divergence here is what let three
    # load-path defects reach the card (lead run, 2026-09-11).
    flags = resolve_flags(registration)
    env = build_environment(flags)
    # _load_lived_repo reads frame["resolved_flags"], so it is handed the
    # SAME flags dict rather than the RS1 frame's own — the RS1 frame was
    # resolved for a different battery and its resolved_flags lacks the C2
    # frame keys (measured: KeyError 'ephemeral').
    frame = dict(load_runtime_frame(registration))
    frame['resolved_flags'] = flags

    # ------------------------------------------------------------------
    # MODEL REUSE. The 20B model is loaded ONCE for the whole invocation and
    # handed to every subsequent row's repository. The first version rebuilt
    # it per row and never freed the previous one; the lead's run completed 7
    # rows and then died in `GptOss20B_TC.from_pretrained` for row 8
    # (`cudaMalloc failed: out of memory`, lease released at 163 s).
    #
    # Prior art: `grm_c7_middle_replay.open_copy(q, session, r, loaded)` (GRM
    # contributors, 2026) — the `loaded=(model, tok, info)` carry that let
    # FIX-8's replay fit 16 of 24 rows from THESE checkpoints in batches
    # F1-F4. Taken verbatim in shape: load once, then construct a fresh
    # GraftRepository per row over the SAME model/tokenizer.
    # ------------------------------------------------------------------
    if cache is not None and cache.get('model') is not None:
        e2e = cache['e2e']
        model, tokenizer = cache['model'], cache['tokenizer']
        model_info = cache['model_info']
        repo = _repo_over(e2e, model, tokenizer, scratch, flags, frame)
    else:
        e2e, model, tokenizer, repo, model_info = rs1._load_lived_repo(
            scratch, frame)
        if cache is not None:
            cache.update({'e2e': e2e, 'model': model, 'tokenizer': tokenizer,
                          'model_info': model_info})
    encode = (lambda text: tokenizer.encode(text, add_special_tokens=False))
    return {
        'repo': repo, 'e2e': e2e, 'encode': encode, 'scratch': scratch,
        'model_info': model_info, 'flags': flags, 'environment': env,
    }


def _repo_over(e2e, model, tokenizer, scratch, flags, frame):
    """A fresh repository over an ALREADY-LOADED model, under a VRAM budget.

    Shape taken from ``grm_c7_middle_replay.open_copy``'s reuse branch (GRM,
    2026), with ONE deliberate difference, named here because it is the
    second half of the OOM repair:

    ``vram_budget_mb`` is SET rather than ``None``.  With it ``None``:
      * ``GraftRepository.load()``'s ``_load_can_materialize_device`` returns
        True unconditionally, so EVERY non-retired node's payload is unpacked
        onto the device at load; and
      * ``_page()`` returns 0 on its first line, so nothing is ever spilled.

    Measured on the pinned checkpoints (artifacts/grm_a1/payload_accounting.
    json, CPU-only, derived with the repository's own ``_node_bytes``
    formula): the d250 cells pin ~659 MB of node payloads while the mount
    plan can seat at most ``arena_width`` 96 tokens = 4.5 MB. With a budget
    set, load() stops materialising past it and ``arena._ensure_h`` pages the
    mounted nodes back in through ``node_loader`` (``_load_node``), which is
    already wired at ``graft_repository.py:354``. No core change is needed —
    the pager was simply never armed.
    """
    return e2e.GraftRepository(
        model,
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        lambda ids: tokenizer.decode(ids, clean_up_tokenization_spaces=False),
        str(scratch / 'repository'),
        autosave=False,
        arena_cls=e2e.GptOssGQAArenaCache,
        native_lib_path=str(frame.get('native_library', {}).get('path')
                            or ROOT / 'cpp/build/libgrm_runtime.so'),
        native_auto=False,
        vram_budget_mb=NODE_VRAM_BUDGET_MB,
        route_layer=int(
            e2e.gpt_oss_grm_dialect_kwargs(model.config)['route_layer']),
        arena_width=int(flags['arena_width']),
        topk=int(flags['topk']),
        live_turns=int(flags['live_turns']),
        max_live=int(flags['max_live']),
        ephemeral=bool(flags['ephemeral']),
        sink_text=e2e.HARMONY_SINK,
        prompt_template=e2e.harmony_turn,
        stop_sequences=e2e.HARMONY_STOPS,
        storage_bits=int(flags['graft_storage_bits']),
        revision_resolution=bool(flags['sup_resolve']),
        decisive_admission=bool(flags['adm_decisive']),
    )


def _load_fake(checkpoint_dir, registration, cache=None):
    """CPU doubles standing in for the checkpoint, same downstream code.

    Seeds the repository from the registration's OWN recorded texts (the
    fixture the seat built from RD2 / C7 r3 receipts), so the fake path has
    real alias edges, a real compound base and a real unanswerable control to
    exercise. It never reads an expected answer.
    """
    from _pytest.monkeypatch import MonkeyPatch
    from scripts import grm_c7_diagnose as diag
    from scripts import grm_e2e_session as e2e

    fixture = read(ROOT / 'artifacts/grm_a1/alias_fixture.json')
    mp = MonkeyPatch()
    scratch = Path(tempfile.mkdtemp(prefix='grm_a1_fake_'))
    repo = diag.repository(scratch / 'repository', mp)

    import re as _re

    class _FoldModel(diag.Model):
        """FIX-5 stimulus over the enumerated source spans only."""

        def __call__(self, ids, kv_caches=None, **kwargs):
            if kv_caches is None:
                prompt = self.codec.decode(ids[0])
                lines = _re.findall(r'^\[source \d+\] (.+)$', prompt, _re.M)
                if lines:
                    self.fold_output = (
                        'The archived record states that '
                        + ' '.join(l.rstrip('.') + '.' for l in lines)
                        + '<|end|>')
                else:
                    self.fold_output = None
            return super().__call__(ids, kv_caches=kv_caches, **kwargs)

    repo.arena.m = _FoldModel(repo.arena.m.codec)

    texts = [fixture['rd2']['base_text']]
    texts += [r['edge_text'] for r in fixture['rd2']['rows'][:2]]
    seen = set(texts)
    for row in fixture['c7_r3']['rows']:
        for src in row['oracle_source_texts']:
            node = ('<|start|>system<|message|>You are ChatGPT. Reasoning: '
                    'low. Valid channel: final.<|end|>'
                    '<|start|>user<|message|>' + src + '<|end|>'
                    '<|start|>assistant<|channel|>final<|message|>'
                    'Recorded.<|end|>')
            if node not in seen:
                seen.add(node)
                texts.append(node)
    for text in texts:
        idx = repo.arena.deposit(text)
        repo.arena.grafts[idx]['kind'] = 'turn'
        repo.arena.grafts[idx]['no_fold'] = True
    repo._sync_lifecycle()
    repo.arena.m.fold_output = None
    # THE SAME flags object and THE SAME environment() call the GPU arm
    # makes. This is the whole point of the 2026-09-11 repair: a shape error
    # in either now fails HERE, on CPU, instead of on the card.
    flags = resolve_flags(registration)
    env = build_environment(flags)
    # The frozen frame is READ on the fake path too, so a missing or drifted
    # frame is a CPU failure. Only the model and tokenizer are stubbed.
    frame = load_runtime_frame(registration)
    return {
        'repo': repo, 'e2e': e2e, 'encode': repo.arena.encode,
        'scratch': scratch, 'monkeypatch': mp,
        'model_info': {'id': 'CPU_DOUBLE_scripts.grm_c7_diagnose',
                       'note': 'NOT a model-quality measurement',
                       'frame_schema': frame.get('schema')},
        'flags': flags, 'environment': env,
    }


# ------------------------------------------------------------- 4. the pins

def pin_flags(environment, *, rule=None):
    """Apply ``environment(flags)`` THEN pin A1's flags, and verify.

    ``grm_c2_cells.environment`` removes every ambient ``GRM_*`` key, so the
    two pins below MUST come after it. The assertions are not decoration:
    pinning before ``environment()`` fails silently, and that exact mistake
    produced a wrong reading earlier in this arc.
    """
    from core.grm_admission import admission_rule
    from core.grm_alias_fold import alias_fold_enabled

    os.environ.clear()
    os.environ.update(environment)
    os.environ[FLAG_ENV] = '1'
    if rule:
        os.environ[RULE_ENV] = str(rule)
    if not alias_fold_enabled():
        raise ContrastError('A1_FLAG_NOT_IN_FORCE_AFTER_PIN')
    if rule and admission_rule() != rule:
        raise ContrastError('A1_ADMISSION_RULE_NOT_IN_FORCE_AFTER_PIN')
    return {FLAG_ENV: os.environ[FLAG_ENV],
            RULE_ENV: os.environ.get(RULE_ENV),
            'alias_fold_enabled': True,
            'admission_rule': admission_rule()}


# -------------------------------------------------- 5. merge + width receipt

def apply_merge(repo, encode):
    """Merge on the LOADED state, then width-receipt every digest.

    The width receipt is taken with the loaded tokenizer (GPT-OSS on the real
    path) BEFORE anything claims a single mount, which is the order's
    requirement. ``_alias_fold_once`` already routes an over-width digest
    through the existing LSR-P2C guard and records ``width_guard_split``; this
    re-measures with the REAL tokenizer and records both numbers so a reader
    can see whether the two tokenizations disagreed.
    """
    from core import grm_alias_fold as af

    repo.alias_fold_merge = True
    decisions = repo.alias_fold_pass()
    width_rows = []
    for decision in decisions:
        digest = decision.get('digest')
        if digest is None:
            width_rows.append({'alias': decision.get('alias'),
                               'reason': decision['reason'], 'digest': None,
                               'width_loaded_tokenizer': None})
            continue
        text = repo.arena.grafts[int(digest)].get('text', '')
        measured = af.width_receipt(encode, text, int(repo.arena.width))
        width_rows.append({
            'alias': decision.get('alias'),
            'base_name': decision.get('base_name'),
            'reason': decision['reason'],
            'lineage': decision.get('lineage'),
            'digest': int(digest),
            'base_shared_with_edges': decision.get('base_shared_with_edges'),
            'base_identifiers_missing': decision.get(
                'base_identifiers_missing'),
            'width_loaded_tokenizer': measured,
            'width_at_merge_time': decision.get('width'),
            'width_guard_split': decision.get('width_guard_split'),
            'tokenizers_agree': (
                measured['digest_tokens']
                == (decision.get('width') or {}).get('digest_tokens')),
        })
    over = [w for w in width_rows
            if w.get('width_loaded_tokenizer')
            and not w['width_loaded_tokenizer']['fits']]
    return {'decisions': decisions, 'widths': width_rows,
            'digests': len([w for w in width_rows
                            if w.get('digest') is not None]),
            'over_width': len(over)}


# --------------------------------------------------- 5b. memory receipts

def payload_residency(repo):
    """How many node payloads are DEVICE-resident, and how many bytes.

    ``g["h"] is not None`` is the device copy; ``host_payload`` is the RAM
    copy the pager can spill to and reload from. Both are counted so a reader
    can see that spilling moved payloads rather than losing them.
    """
    grafts = repo.arena.grafts
    device = [i for i, g in enumerate(grafts) if g.get('h') is not None]
    host = [i for i, g in enumerate(grafts)
            if g.get('host_payload') is not None]
    try:
        device_bytes = sum(repo._node_bytes(grafts[i]) for i in device)
    except Exception:
        device_bytes = None
    return {
        'nodes': len(grafts),
        'payloads_device_resident': len(device),
        'payloads_host_resident': len(host),
        'device_payload_bytes': device_bytes,
        'device_payload_mb': (None if device_bytes is None
                              else round(device_bytes / 2**20, 2)),
        'vram_budget_mb': (None if repo.vram_budget is None
                           else round(repo.vram_budget / 2**20, 2)),
        'page_ins': int(getattr(repo.arena, 'page_ins', 0) or 0),
    }


def device_memory(fake=False):
    """Free/used device MiB, or ``None`` on the CPU gate.

    Read through ``nvidia-smi`` rather than the tensor_cuda module:
    ``tensor_cuda`` exposes ``empty_cache`` and ``synchronize`` but no
    memory-info call (checked), and inventing one would be a fabricated API.
    A read-only query, never a mutation of any other process's state.
    """
    if fake:
        return None
    try:
        out = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=memory.free,memory.total,memory.used',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=30, check=True)
        free, total, used = (int(v.strip())
                             for v in out.stdout.splitlines()[0].split(','))
        return {'free_mb': free, 'total_mb': total, 'used_mb': used}
    except Exception as exc:
        return {'unavailable': f'{type(exc).__name__}: {exc}'}


def release_payloads(repo, keep=()):
    """Drop device payloads for every node except ``keep``.

    The pager's own idiom: ``g["h"] = None`` plus the native device-copy
    eviction ``_page`` uses, so a released node is reloadable through
    ``node_loader`` exactly as a spilled one is. RAM (``host_payload``) is
    deliberately left intact — that is what makes the reload cheap and is
    why ``_page`` calls this "spill", not "free".
    """
    keep = {int(i) for i in keep}
    released = 0
    for i, g in enumerate(repo.arena.grafts):
        if i in keep or g.get('h') is None:
            continue
        if g.get('host_payload') is None:
            try:
                repo._ensure_host_payload(i, g)
            except Exception:
                continue
        g['h'] = None
        released += 1
        try:
            repo._native_evict_device_copy(i)
            repo._ensure_lifecycle(i, g)
        except Exception:
            pass
    return released


def empty_cache(fake=False):
    """``tensor_cuda.empty_cache()`` after a row, when there is a card."""
    if fake:
        return False
    try:
        from core.mistral7b_tc import tc

        tc.empty_cache()
        return True
    except Exception:
        return False


# ------------------------------------------------------------- 6. the serve

def serve(repo, e2e, question, *, fake):
    """One read under the PLAIN production prompt.

    The real path goes through ``_probe_ladder_chat`` exactly as
    ``grm_c7_run`` does.  The fake path calls ``arena.step`` directly because
    the CPU double has no ladder; both return ``(answer, info)`` and both
    leave ``arena.cur_mounts`` set, which is what the receipt records.
    """
    if fake:
        return repo.arena.step(question, ngen=32, deposit=False)
    return e2e._probe_ladder_chat(repo, question, topk=3, ngen=32,
                                  max_trips=1, defer_memory=True)


def effective_question(question):
    """C7 r3 plain-prompt projection, reused rather than retyped."""
    from scripts.grm_c7_run import effective_question as project

    previous = os.environ.get('GRM_C7_REVISION')
    os.environ['GRM_C7_REVISION'] = 'r3'
    try:
        return project(question)
    finally:
        if previous is None:
            os.environ.pop('GRM_C7_REVISION', None)
        else:
            os.environ['GRM_C7_REVISION'] = previous


def score(answer, row):
    """Frozen-style exact scoring; controls are scored as 'must not move'.

    An answerable row is EXACT when the normalized answer equals the expected
    value.  A control (``answerable == False``, expected UNKNOWN) is BROKEN
    when the model asserts a value instead of declining — that is the "0
    controls broken" half of the registered prediction.
    """
    from core.grm_text_norm import normalize_glyphs

    text = normalize_glyphs(str(answer or '')).strip()
    want = normalize_glyphs(str(row['expected'] or '')).strip()
    exact = text.casefold() == want.casefold()
    if row['answerable']:
        return {'exact_correct': bool(exact), 'control_broken': False}
    declined = (not text) or ('unknown' in text.casefold()) or any(
        phrase in text.casefold()
        for phrase in ("can't", 'cannot', 'not in memory', 'no stored record'))
    return {'exact_correct': bool(exact or declined),
            'control_broken': bool(not declined and not exact)}


# --------------------------------------------------------------- 7. the run

def receipt_path(out_dir, row):
    stem = row['row_id'].replace('::', '__').replace('/', '_')
    return Path(out_dir) / 'rows' / f'{stem}.json'


#: Exception types that mean "this row did not FIT on the card", as opposed
#: to "this row is wrong". Matched on the message too, because tensor_cuda
#: surfaces allocation failures as a plain RuntimeError.
_NON_FIT_MARKERS = ('out of memory', 'cudamalloc', 'cuda_error_out_of_memory',
                    'cublas_status_alloc_failed')


def is_non_fit(exc):
    """Is this exception an allocation failure rather than a defect?"""
    if isinstance(exc, MemoryError):
        return True
    text = f'{type(exc).__name__}: {exc}'.casefold()
    return any(marker in text for marker in _NON_FIT_MARKERS)


def run_row(row, registration, out_dir, *, fake, rule, provenance=None,
            cache=None):
    """Execute ONE row end to end. Create-only, so a redo is impossible.

    A row that cannot FIT on the card writes a create-only ``NON_FIT``
    receipt carrying the memory snapshot and RETURNS, so the campaign
    continues to the next row. A NON_FIT row is UNMEASURED: it is excluded
    from the prediction's numerator AND its denominator, and reported on its
    own line — silently shrinking the denominator would turn a capacity
    failure into a better-looking score.
    """
    path = receipt_path(out_dir, row)
    if path.exists():
        return {'row_id': row['row_id'], 'status': 'SKIPPED_ALREADY_DONE'}

    loader = _load_fake if fake else _load_gpu
    checkpoint = row.get('checkpoint_dir')
    if not fake and not (checkpoint and Path(checkpoint).is_dir()):
        raise ContrastError(f'A1_CHECKPOINT_MISSING: {checkpoint}')

    started = time.monotonic()
    memory_before = device_memory(fake=fake)
    loaded = None
    try:
        loaded = loader(checkpoint, registration, cache)
        repo = loaded['repo']
        pins = pin_flags(loaded['environment'], rule=rule)
        residency_after_load = payload_residency(repo)
        merge = apply_merge(repo, loaded['encode'])
        # The fold's sources were paged in to be mounted for the digest
        # capture (the FIX-5 path runs on the model). Release them the moment
        # the digest exists — the mount plan below needs a handful of nodes,
        # not the whole conversation.
        released_after_merge = release_payloads(repo)
        question = effective_question(row['question'])
        answer, info = serve(repo, loaded['e2e'], question, fake=fake)
        mounts = [int(i) for i in (repo.arena.cur_mounts or ())]
        residency_after_mount = payload_residency(repo)
        mounted_text = '\n'.join(
            str(repo.arena.grafts[i].get('text', '')) for i in mounts)
        record = {
            'schema': 'grm.a1.gpu_contrast_row.v1',
            'row_id': row['row_id'], 'group': row['group'],
            'probe_id': row['probe_id'], 'class': row['class'],
            'answerable': row['answerable'],
            'scored_for_prediction': row['scored_for_prediction'],
            'cell': row.get('cell'), 'checkpoint_dir': checkpoint,
            'question_registered': row['question'],
            'question_served': question,
            'expected': row['expected'],
            'answer': str(answer),
            'mounted_ids': mounts,
            'mount_count': len(mounts),
            'single_mount': len(mounts) == 1,
            'mounted_text': mounted_text[:4000],
            'abstained': bool(info.get('abstained')),
            'abstain_reason': info.get('abstain_reason'),
            'admission_policy_branch': info.get('admission_policy_branch'),
            'admission_rank_plan': info.get('admission_rank_plan'),
            'merge': merge, 'pins': pins,
            'worker_provenance': provenance,
            'model': loaded['model_info'],
            'fake': bool(fake),
            'non_fit': False,
            'memory': {
                'device_before': memory_before,
                'device_after': device_memory(fake=fake),
                'residency_after_load': residency_after_load,
                'residency_after_mount': residency_after_mount,
                'payloads_released_after_merge': released_after_merge,
                'node_vram_budget_mb': NODE_VRAM_BUDGET_MB,
            },
            'elapsed_s': round(time.monotonic() - started, 3),
        }
        record.update(score(answer, row))
        create(path, record)
        return record
    except Exception as exc:
        if not is_non_fit(exc):
            raise
        # CAPACITY, not correctness. Receipt it and let the campaign go on.
        record = {
            'schema': 'grm.a1.gpu_contrast_row.v1',
            'row_id': row['row_id'], 'group': row['group'],
            'probe_id': row['probe_id'], 'class': row['class'],
            'answerable': row['answerable'],
            'scored_for_prediction': row['scored_for_prediction'],
            'cell': row.get('cell'), 'checkpoint_dir': checkpoint,
            'question_registered': row['question'],
            'expected': row['expected'],
            'non_fit': True,
            'status': 'NON_FIT',
            'error': f'{type(exc).__name__}: {exc}',
            'traceback': traceback.format_exc()[-4000:],
            'worker_provenance': provenance,
            'fake': bool(fake),
            'memory': {
                'device_before': memory_before,
                'device_at_failure': device_memory(fake=fake),
                'residency_at_failure': (
                    payload_residency(loaded['repo'])
                    if loaded and loaded.get('repo') else None),
                'node_vram_budget_mb': NODE_VRAM_BUDGET_MB,
            },
            'exact_correct': None, 'control_broken': None,
            'elapsed_s': round(time.monotonic() - started, 3),
        }
        create(path, record)
        return record
    finally:
        if loaded is not None:
            try:
                release_payloads(loaded['repo'])
            except Exception:
                pass
            try:
                loaded['repo'].close()
            except Exception:
                pass
            if loaded.get('monkeypatch') is not None:
                loaded['monkeypatch'].undo()
            shutil.rmtree(loaded['scratch'], ignore_errors=True)
        empty_cache(fake=fake)


def summarize(out_dir):
    rows = [read(p) for p in
            sorted((Path(out_dir) / 'rows').glob('*.json'))]
    non_fit = [r for r in rows if r.get('non_fit')]
    measured = [r for r in rows if not r.get('non_fit')]
    # NON_FIT rows are UNMEASURED: out of the numerator AND the denominator.
    # Leaving them in the denominator would score a capacity failure as a
    # wrong answer; dropping them silently would shrink the denominator and
    # flatter the result. They get their own line instead.
    scored = [r for r in measured if r.get('scored_for_prediction')]
    controls = [r for r in measured if not r.get('answerable')]
    hits = sum(bool(r.get('exact_correct')) for r in scored)
    broken = sum(bool(r.get('control_broken')) for r in controls)
    single = sum(bool(r.get('single_mount')) for r in scored)
    over = sum(int((r.get('merge') or {}).get('over_width', 0))
               for r in measured)
    fake = bool(rows) and all(r.get('fake') for r in rows)
    summary = {
        'schema': 'grm.a1.gpu_contrast_summary.v1',
        'rows': len(rows),
        'rows_measured': len(measured),
        'rows_non_fit': len(non_fit),
        'non_fit_row_ids': sorted(r['row_id'] for r in non_fit),
        'non_fit_note': ('NON_FIT rows are UNMEASURED: excluded from the '
                         'prediction numerator AND denominator, never '
                         'counted as wrong answers.'),
        'scored_rows': len(scored),
        'aliases_exact': hits,
        'aliases_single_mount': single,
        'controls': len(controls),
        'controls_broken': broken,
        'digests_over_width': over,
        'prediction': 'aliases >= 5/8 exact AND 0 controls broken',
        # A prediction over a PARTIAL denominator is not the registered
        # prediction. If any scored row went NON_FIT, the verdict is withheld
        # and named as such rather than computed from what survived.
        'prediction_met': (
            None if any(r.get('scored_for_prediction') for r in non_fit)
            else bool(hits >= 5 and broken == 0)),
        'verdict': (
            'INCOMPLETE_NON_FIT'
            if any(r.get('scored_for_prediction') for r in non_fit)
            else ('GREEN' if hits >= 5 and broken == 0 else 'RED')),
        'fake': fake,
    }
    if fake:
        # A fake run EXERCISES the code; it does not TEST the prediction.
        # The CPU reader is entity-blind (it returns the LAST
        # "current X value is Y" it can see) and the RD2 base is one
        # COMPOUND record naming two entities, so the Signal-0 rows miss by
        # construction — pinned by tests/test_grm_a1_alias_fold.py::
        # test_rd2_fake_reader_entity_blind_limit. Reporting a fake
        # aliases_exact as though it were the contrast would be exactly the
        # false-success this arc keeps refusing.
        # A NON_FIT scored row withholds the verdict on EITHER path: it is a
        # stronger statement than "this was a fake run", so it wins.
        if summary['verdict'] != 'INCOMPLETE_NON_FIT':
            summary['verdict'] = 'FAKE_RUN_NOT_A_PREDICTION'
        summary['prediction_met'] = None
        summary['fake_note'] = (
            'CPU doubles. aliases_exact is NOT the registered measurement: '
            'the stub reader is entity-blind on the compound RD2 base. The '
            'meaningful fake-path assertions are structural — rows, '
            'single_mount, controls_broken, digests_over_width.')
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--registration', default=str(REGISTRATION))
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    ap.add_argument('--resume', action='store_true',
                    help='skip rows whose receipt already exists (default)')
    ap.add_argument('--fake', action='store_true',
                    help='CPU doubles; the seat gate. No GPU, no lease.')
    ap.add_argument('--rule', default=None,
                    help='pin GRM_ADMISSION_RULE after environment(flags)')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--dry-run', action='store_true',
                    help='check every declared input and precondition, '
                         'report the missing ones, and exit. Takes NO lease '
                         'and loads nothing.')
    ap.add_argument('--lease-seconds', type=_lease_seconds,
                    default=LEASE_SECONDS,
                    help=f'GPU lease per invocation (default '
                         f'{LEASE_SECONDS}s; house cap 590s)')
    ap.add_argument('--lock-wait-seconds', type=int,
                    default=LOCK_WAIT_SECONDS)
    args = ap.parse_args(argv)

    registration = read(args.registration)
    out_dir = Path(args.out)
    (out_dir / 'rows').mkdir(parents=True, exist_ok=True)

    # (1) DRIFT FIRST — before any load, any lease, any card. The amendment
    # chain is walked first so the input set under test is the CURRENT one.
    base_sha, amendments = load_amendments(args.registration)
    # Declarations an amendment ADDED (frame, flags source, preconditions)
    # must be visible to everything downstream, or the worker would keep
    # reading the registration's original, incomplete view.
    registration = effective_registration(registration, amendments)
    pinned = effective_pinned_inputs(registration, amendments)
    count = verify_pinned_inputs(pinned)
    provenance = verify_self(amendments)
    print(f'registration sha256={base_sha}', flush=True)
    for entry in amendments:
        print(f'  amendment {Path(entry["path"]).name} '
              f'sha256={entry["sha256"]}', flush=True)
    print(f'pinned inputs verified: {count}', flush=True)
    print(f'worker sha256={provenance["sha256"]} '
          f'matches_amendment={provenance["matches_amendment"]}', flush=True)

    # (2) PRECONDITIONS — every declared input and dependency, BEFORE any
    # lease. The first three lead GPU attempts each burned a lease slot
    # discovering one missing dependency at a time; this reports them all at
    # once and never touches the card.
    pre = check_preconditions(registration, fake=bool(args.fake))
    for check in pre['checks']:
        mark = 'ok ' if check['ok'] else 'MISSING'
        print(f"  {mark} {check['kind']:13s} {check['name']:22s} "
              f"{check['path']}"
              + (f"  [{check['detail']}]" if check['detail'] else ''),
              flush=True)
    if args.dry_run:
        print(json.dumps({'preconditions_ok': pre['ok'],
                          'missing': pre['missing']},
                         indent=1, sort_keys=True), flush=True)
        return 0 if pre['ok'] else 2
    if not pre['ok']:
        raise ContrastError(
            'A1_PRECONDITION_MISSING: '
            + json.dumps(pre['missing'], sort_keys=True))

    rows = plan_rows(registration)
    if args.limit:
        rows = rows[:args.limit]
    pending = [r for r in rows if not receipt_path(out_dir, r).exists()]
    print(f'rows total={len(rows)} pending={len(pending)} '
          f'fake={bool(args.fake)} lease_s={args.lease_seconds}', flush=True)

    # One model for the whole invocation (prior art: FIX-8's `open_copy`
    # `loaded` carry). Rebuilding it per row is what OOM'd the lead's run.
    cache = {}

    def execute():
        for row in rows:
            result = run_row(row, registration, out_dir, fake=args.fake,
                             rule=args.rule, provenance=provenance,
                             cache=cache)
            status = result.get('status') or (
                'EXACT' if result.get('exact_correct') else 'MISS')
            memory = (result.get('memory') or {}).get('residency_after_mount')
            detail = ''
            if memory:
                detail = (f"  [payloads device={memory['payloads_device_resident']}"
                          f" {memory['device_payload_mb']}MB"
                          f" page_ins={memory['page_ins']}]")
            print(f"  {row['row_id']:34s} {status}{detail}", flush=True)

    lease_expired = False
    if args.fake:
        # No card, so no lease: taking the GPU flock on the CPU gate would
        # block the operator for nothing.
        execute()
    else:
        from scripts.grm_cmc1_gpu_arms import gpu_lease
        try:
            with gpu_lease(int(args.lease_seconds),
                           int(args.lock_wait_seconds)):
                execute()
        except TimeoutError:
            # EXPECTED under a <=590s house lease: the rows are create-only
            # and resumable, so an expired lease is a pause, not a failure.
            # The operator re-runs the same --resume line until pending=0.
            lease_expired = True
            print('lease expired; finished rows are durable, re-run '
                  '--resume to continue', flush=True)

    summary = summarize(out_dir)
    remaining = [r for r in rows if not receipt_path(out_dir, r).exists()]
    summary['pending'] = len(remaining)
    summary['lease_expired'] = bool(lease_expired)
    summary['complete'] = not remaining
    # The summary is a DERIVED view and is rewritten on each invocation; the
    # per-row receipts underneath it stay create-only.
    (out_dir / 'summary.json').write_text(
        json.dumps(summary, indent=1, sort_keys=True) + '\n')
    # probes.jsonl is what lead_commands.txt's summary block reads.
    with (out_dir / 'probes.jsonl').open('w') as stream:
        for path in sorted((out_dir / 'rows').glob('*.json')):
            stream.write(json.dumps(read(path), sort_keys=True) + '\n')
    print(json.dumps(summary, indent=1, sort_keys=True), flush=True)
    if summary.get('fake'):
        # Exit status on the fake path reflects STRUCTURAL health only.
        structural_ok = (summary['controls_broken'] == 0
                         and summary['rows'] > 0)
        return 0 if structural_ok else 1
    if remaining:
        # Rows still pending: NOT a verdict. Exit 4 so the operator's loop
        # can tell "re-run me" apart from "the prediction failed" (1).
        print(f'pending={len(remaining)} — re-run with --resume', flush=True)
        return 4
    if summary['rows_non_fit']:
        print(f"NON_FIT rows (unmeasured): {summary['rows_non_fit']} "
              f"{summary['non_fit_row_ids']}", flush=True)
    if summary['prediction_met'] is None:
        # A scored row did not fit: the denominator is incomplete, so there
        # is no verdict to report. Exit 5, distinct from a RED result.
        print('INCOMPLETE: a scored row went NON_FIT; no verdict', flush=True)
        return 5
    return 0 if summary['prediction_met'] else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except ContrastError as exc:
        print(f'RED: {exc}', file=sys.stderr)
        raise SystemExit(2)
    except Exception:
        traceback.print_exc()
        raise SystemExit(3)
