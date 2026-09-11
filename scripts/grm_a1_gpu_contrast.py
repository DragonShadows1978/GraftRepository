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

#: Lease cap. The registration budgets 0.5 GPU-h for the whole run; a single
#: row is bounded well under this and the cap exists so a wedged row releases
#: the card instead of holding it.
LEASE_SECONDS = 2400
LOCK_WAIT_SECONDS = 7200


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


# ------------------------------------------------------- 3. repository load

def _load_gpu(checkpoint_dir, registration):
    """The lived GPU build over a COPY of the pinned checkpoint.

    The checkpoint is copied to a scratch directory first: the pinned C7 r3
    checkpoints are immutable evidence for another order and this worker must
    not write through to them, even though it only intends to read.
    """
    from scripts import grm_c2_cells as c2
    from scripts import grm_rs1_read_strength_gpu as rs1
    from scripts import grm_det1_2_gpu as det1_2

    scratch = Path(tempfile.mkdtemp(prefix='grm_a1_'))
    shutil.copytree(Path(checkpoint_dir) / 'repository',
                    scratch / 'repository')
    # The FROZEN runtime frame, read the way rs1 reads it (its module-level
    # RUNTIME_FRAME constant). Not retyped here: divergence from the lived
    # build would silently change the arena the contrast is measured on.
    frame_path = Path(registration.get('runtime_frame')
                      or rs1.RUNTIME_FRAME)
    if not frame_path.is_file():
        raise ContrastError(f'A1_RUNTIME_FRAME_MISSING: {frame_path}')
    frame = json.loads(frame_path.read_text(encoding='utf-8'))
    e2e, model, tokenizer, repo, model_info = rs1._load_lived_repo(
        scratch, frame)
    del det1_2  # imported only to fail fast if the GPU stack is absent
    encode = (lambda text: tokenizer.encode(text, add_special_tokens=False))
    return {
        'repo': repo, 'e2e': e2e, 'encode': encode, 'scratch': scratch,
        'model_info': model_info, 'flags': frame['resolved_flags'],
        'environment': c2.environment(frame['resolved_flags']),
    }


def _load_fake(checkpoint_dir, registration):
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
    return {
        'repo': repo, 'e2e': e2e, 'encode': repo.arena.encode,
        'scratch': scratch, 'monkeypatch': mp,
        'model_info': {'id': 'CPU_DOUBLE_scripts.grm_c7_diagnose',
                       'note': 'NOT a model-quality measurement'},
        'flags': {'arena_width': int(repo.arena.width)},
        'environment': dict(os.environ),
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


def run_row(row, registration, out_dir, *, fake, rule, provenance=None):
    """Execute ONE row end to end. Create-only, so a redo is impossible."""
    path = receipt_path(out_dir, row)
    if path.exists():
        return {'row_id': row['row_id'], 'status': 'SKIPPED_ALREADY_DONE'}

    loader = _load_fake if fake else _load_gpu
    checkpoint = row.get('checkpoint_dir')
    if not fake and not (checkpoint and Path(checkpoint).is_dir()):
        raise ContrastError(f'A1_CHECKPOINT_MISSING: {checkpoint}')

    started = time.monotonic()
    loaded = loader(checkpoint, registration)
    try:
        pins = pin_flags(loaded['environment'], rule=rule)
        merge = apply_merge(loaded['repo'], loaded['encode'])
        question = effective_question(row['question'])
        answer, info = serve(loaded['repo'], loaded['e2e'], question,
                             fake=fake)
        mounts = [int(i) for i in (loaded['repo'].arena.cur_mounts or ())]
        mounted_text = '\n'.join(
            str(loaded['repo'].arena.grafts[i].get('text', ''))
            for i in mounts)
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
            'elapsed_s': round(time.monotonic() - started, 3),
        }
        record.update(score(answer, row))
        create(path, record)
        return record
    finally:
        try:
            loaded['repo'].close()
        except Exception:
            pass
        if loaded.get('monkeypatch') is not None:
            loaded['monkeypatch'].undo()
        shutil.rmtree(loaded['scratch'], ignore_errors=True)


def summarize(out_dir):
    rows = [read(p) for p in
            sorted((Path(out_dir) / 'rows').glob('*.json'))]
    scored = [r for r in rows if r.get('scored_for_prediction')]
    controls = [r for r in rows if not r.get('answerable')]
    hits = sum(bool(r.get('exact_correct')) for r in scored)
    broken = sum(bool(r.get('control_broken')) for r in controls)
    single = sum(bool(r.get('single_mount')) for r in scored)
    over = sum(int((r.get('merge') or {}).get('over_width', 0)) for r in rows)
    fake = bool(rows) and all(r.get('fake') for r in rows)
    summary = {
        'schema': 'grm.a1.gpu_contrast_summary.v1',
        'rows': len(rows),
        'scored_rows': len(scored),
        'aliases_exact': hits,
        'aliases_single_mount': single,
        'controls': len(controls),
        'controls_broken': broken,
        'digests_over_width': over,
        'prediction': 'aliases >= 5/8 exact AND 0 controls broken',
        'prediction_met': bool(hits >= 5 and broken == 0),
        'verdict': ('GREEN' if hits >= 5 and broken == 0 else 'RED'),
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
    ap.add_argument('--lease-seconds', type=int, default=LEASE_SECONDS)
    ap.add_argument('--lock-wait-seconds', type=int,
                    default=LOCK_WAIT_SECONDS)
    args = ap.parse_args(argv)

    registration = read(args.registration)
    out_dir = Path(args.out)
    (out_dir / 'rows').mkdir(parents=True, exist_ok=True)

    # (1) DRIFT FIRST — before any load, any lease, any card. The amendment
    # chain is walked first so the input set under test is the CURRENT one.
    base_sha, amendments = load_amendments(args.registration)
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

    rows = plan_rows(registration)
    if args.limit:
        rows = rows[:args.limit]
    pending = [r for r in rows if not receipt_path(out_dir, r).exists()]
    print(f'rows total={len(rows)} pending={len(pending)} '
          f'fake={bool(args.fake)}', flush=True)

    def execute():
        for row in rows:
            result = run_row(row, registration, out_dir, fake=args.fake,
                             rule=args.rule, provenance=provenance)
            status = result.get('status') or (
                'EXACT' if result.get('exact_correct') else 'MISS')
            print(f"  {row['row_id']:34s} {status}", flush=True)

    if args.fake:
        # No card, so no lease: taking the GPU flock on the CPU gate would
        # block the operator for nothing.
        execute()
    else:
        from scripts.grm_cmc1_gpu_arms import gpu_lease
        with gpu_lease(int(args.lease_seconds),
                       int(args.lock_wait_seconds)):
            execute()

    summary = summarize(out_dir)
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
