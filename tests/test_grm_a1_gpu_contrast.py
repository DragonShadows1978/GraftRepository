"""GRM-A1 — fake-path gate for the GPU contrast worker.

HOUSE NORM (lead, 2026-09-10): the seat writes the GPU worker with a
``--fake`` path that exercises every branch on the CPU doubles; the lead runs
it on the card.  This file IS that gate.

What it proves, and what it deliberately does not.  It proves the worker's
CONTROL FLOW is correct: the drift check fires before anything else, the
amendment chain is verified and its input set wins, the flags are pinned
AFTER ``environment(flags)`` and asserted in force, every registered row runs,
receipts are create-only so an interrupted run resumes without rewriting
history, the width receipt is taken before any single-mount claim, and an
over-width digest takes the existing split path.

It does NOT prove the registered prediction.  The CPU reader is entity-blind
and the RD2 base is one compound record, so ``aliases_exact`` on the fake path
is meaningless by construction — the worker itself refuses to report a fake
run as a prediction (``verdict: FAKE_RUN_NOT_A_PREDICTION``), and
``test_fake_run_refuses_to_report_a_prediction`` pins that refusal.

Prior art: ``tests/test_grm_scout_fix8_replay.py`` / ``…_resume.py`` (GRM
contributors, 2026) — the drift-RED and resume-ledger gate shapes, reused.
The pin-and-read-back contract is ``scripts/grm_r1_replay.py:242``'s
``pin_rule`` (GRM, 2026).  No prior art known to me for this exact
fake-path-gates-its-own-GPU-path composition.
"""

import json
import os
from pathlib import Path

import pytest

from scripts import grm_a1_gpu_contrast as W

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / 'artifacts/grm_a1/gpu_contrast_registration.json'


@pytest.fixture(scope='module')
def registration():
    return W.read(REGISTRATION)


# ------------------------------------------------------------- the chain

def test_amendment_chain_is_sha_bound_to_the_registration():
    base_sha, amendments = W.load_amendments(REGISTRATION)
    assert base_sha.startswith('329d5865'), base_sha
    assert amendments, 'amendment 1 must exist'
    for entry in amendments:
        amendment = entry['amendment']
        assert amendment['registration_sha256'] == base_sha
        assert amendment['schema'] == 'grm.a1.lead-amendment.v1'
        checksum = Path(entry['path']).with_suffix('.sha256')
        assert checksum.read_text().split()[0] == entry['sha256']
    worker = amendments[-1]['amendment']['worker']
    assert worker['path'] == 'scripts/grm_a1_gpu_contrast.py'
    assert worker['sha256'] == W.sha(ROOT / worker['path'])


def test_amendment_input_set_wins_over_the_registration(registration):
    """The amendment's hashes override the registration's, file by file."""
    _, amendments = W.load_amendments(REGISTRATION)
    pinned = W.effective_pinned_inputs(registration, amendments)
    # A file the amendment ADDED is now pinned.
    gate = str(ROOT / 'tests/test_grm_a1_gpu_contrast.py')
    assert gate in pinned and gate not in registration['pinned_inputs_sha256']
    # A file the amendment RE-pinned carries the new hash, not the old one.
    tests = str(ROOT / 'tests/test_grm_a1_alias_fold.py')
    assert pinned[tests] == W.sha(tests)
    assert pinned[tests] != registration['pinned_inputs_sha256'][tests]


def test_worker_is_excluded_from_its_own_drift_set_and_checked_separately(
        registration):
    """A self-pinned file inside its own drift set is uneditable; don't.

    The worker is pinned by the amendment (so the lead knows which bytes were
    reviewed) but removed from the set it verifies (so editing it reports a
    PROVENANCE fact, not a spurious input-drift RED). Both halves asserted.
    """
    _, amendments = W.load_amendments(REGISTRATION)
    pinned = W.effective_pinned_inputs(registration, amendments)
    worker = str(W.SELF_PATH)
    assert worker not in pinned, 'the worker must not verify itself as input'
    # ... but the amendment DOES pin it, and it currently matches.
    assert amendments[-1]['amendment']['pinned_inputs_sha256'][worker]
    provenance = W.verify_self(amendments)
    assert provenance['sha256'] == W.sha(W.SELF_PATH)
    assert provenance['pinned_sha256'] == provenance['sha256']
    assert provenance['matches_amendment'] is True


def test_amendment_records_the_lt1_retraction():
    _, amendments = W.load_amendments(REGISTRATION)
    correction = amendments[-1]['amendment']['lt1_correction']
    assert correction['status'] == 'RETRACTED'
    assert 'GRM_ADMISSION_RULE' in correction['cause_of_the_wrong_reading']
    receipt = ROOT / 'artifacts/grm_a1/lt1_margin_first.json'
    assert W.sha(receipt) == correction['receipt_sha256']


def test_chain_rejects_a_broken_amendment(tmp_path):
    """A tampered amendment is a hard RED, not a silently ignored file."""
    base = tmp_path / 'gpu_contrast_registration.json'
    base.write_text(json.dumps({'pinned_inputs_sha256': {}}) + '\n')
    amendment = tmp_path / 'gpu_contrast_amendment_1.json'
    amendment.write_text(json.dumps(
        {'schema': 'grm.a1.lead-amendment.v1',
         'registration_sha256': 'deadbeef',
         'pinned_inputs_sha256': {}}) + '\n')
    (tmp_path / 'gpu_contrast_amendment_1.sha256').write_text(
        W.sha(amendment) + '  gpu_contrast_amendment_1.json\n')
    with pytest.raises(W.ContrastError, match='A1_AMENDMENT_CHAIN_MISMATCH'):
        W.load_amendments(base)

    amendment.write_text(json.dumps(
        {'schema': 'grm.a1.lead-amendment.v1',
         'registration_sha256': W.sha(base),
         'pinned_inputs_sha256': {}}) + '\n')
    with pytest.raises(W.ContrastError, match='A1_AMENDMENT_SHA_MISMATCH'):
        W.load_amendments(base)


# --------------------------------------------------------------- drift RED

def test_drift_is_red_before_anything_else(tmp_path):
    victim = tmp_path / 'input.txt'
    victim.write_text('original\n')
    pinned = {str(victim): W.sha(victim)}
    assert W.verify_pinned_inputs(pinned) == 1
    victim.write_text('tampered\n')
    with pytest.raises(W.ContrastError, match='A1_INPUT_SHA_MISMATCH'):
        W.verify_pinned_inputs(pinned)


def test_drift_red_names_a_missing_input(tmp_path):
    pinned = {str(tmp_path / 'absent.json'): 'a' * 64}
    with pytest.raises(W.ContrastError, match='A1_INPUT_SHA_MISMATCH'):
        W.verify_pinned_inputs(pinned)


def test_main_refuses_to_run_on_drift(tmp_path, monkeypatch):
    """End to end: main() raises BEFORE planning or loading anything."""
    registration = json.loads(REGISTRATION.read_text())
    registration['pinned_inputs_sha256'] = {
        str(ROOT / 'core/grm_alias_fold.py'): 'b' * 64}
    path = tmp_path / 'gpu_contrast_registration.json'
    path.write_text(json.dumps(registration) + '\n')
    called = []
    monkeypatch.setattr(W, 'plan_rows', lambda *a: called.append(1) or [])
    with pytest.raises(W.ContrastError, match='A1_INPUT_SHA_MISMATCH'):
        W.main(['--registration', str(path), '--out', str(tmp_path / 'o'),
                '--fake'])
    assert called == [], 'drift must stop the run before planning rows'


# ------------------------------------------------------------------ rows

def test_plan_covers_all_23_registered_rows(registration):
    rows = W.plan_rows(registration)
    assert len(rows) == 23
    rd2 = [r for r in rows if r['group'] == 'rd2']
    c7 = [r for r in rows if r['group'] == 'c7_r3']
    assert len(rd2) == 8 and len(c7) == 15
    assert all(r['scored_for_prediction'] for r in rd2)
    assert not any(r['scored_for_prediction'] for r in c7)
    assert sum(1 for r in c7 if r['answerable']) == 10
    assert sum(1 for r in c7 if not r['answerable']) == 5
    for row in rows:
        assert row['checkpoint_dir'], row
        assert Path(row['checkpoint_dir']).is_dir(), row
    assert len({r['row_id'] for r in rows}) == 23


def test_plain_production_prompt_drops_the_abstention_clause():
    registered = ('What is the current value for C7-Signal-0? Reply only '
                  'with the answer; if unspecified, reply UNKNOWN.')
    assert W.effective_question(registered) == (
        'What is the current value for C7-Signal-0? Reply only with the '
        'answer.')
    plain = 'What is the current C7-Plain-0 value? Reply only with the answer.'
    assert W.effective_question(plain) == plain


def test_effective_question_restores_the_revision_env(monkeypatch):
    monkeypatch.delenv('GRM_C7_REVISION', raising=False)
    W.effective_question('anything.')
    assert 'GRM_C7_REVISION' not in os.environ
    monkeypatch.setenv('GRM_C7_REVISION', 'r2')
    W.effective_question('anything.')
    assert os.environ['GRM_C7_REVISION'] == 'r2'


# ----------------------------------------------------------------- scoring

def test_scoring_answerable_and_controls():
    answerable = {'answerable': True, 'expected': 'Jasper-711'}
    assert W.score('Jasper-711', answerable) == {
        'exact_correct': True, 'control_broken': False}
    assert W.score('Jasper-712', answerable) == {
        'exact_correct': False, 'control_broken': False}
    control = {'answerable': False, 'expected': 'UNKNOWN'}
    for declined in ('UNKNOWN', 'unknown',
                     "I'm sorry, but I can't provide that.",
                     'Not in memory: no stored record matches'):
        assert W.score(declined, control)['control_broken'] is False
    broken = W.score('Jasper-711', control)
    assert broken['control_broken'] is True
    assert broken['exact_correct'] is False


# --------------------------------------------------------------- the pins

def test_flags_are_pinned_after_environment_and_asserted(monkeypatch):
    """The pin must SURVIVE environment(flags), which strips every GRM_*."""
    from scripts.grm_c2_cells import environment

    flags = {'ephemeral': True, 'capture_pin': 'live', 'seat_near_live': 0,
             'lsr_fixes': 1, 'rt1_rule': 1, 'demand_ngh': 0,
             'gqa_cuda_route': 0, 'graft_storage_bits': 8,
             'route_query_lex': 1, 'probe_ladder': 1, 'sup_resolve': 1,
             'adm_decisive': 1}
    env = environment(flags)
    # The premise: environment() carries NEITHER of A1's two switches.
    assert W.FLAG_ENV not in env
    assert W.RULE_ENV not in env

    pins = W.pin_flags(env, rule='margin_first')
    assert pins[W.FLAG_ENV] == '1'
    assert pins['alias_fold_enabled'] is True
    assert pins['admission_rule'] == 'margin_first'
    from core.grm_admission import admission_rule
    from core.grm_alias_fold import alias_fold_enabled
    assert alias_fold_enabled() is True
    assert admission_rule() == 'margin_first'


def test_pin_failure_is_red(monkeypatch):
    """If the flag does not take, the worker refuses rather than measuring."""
    monkeypatch.setattr(W, 'FLAG_ENV', 'GRM_A1_FLAG_THAT_NOTHING_READS')
    with pytest.raises(W.ContrastError,
                       match='A1_FLAG_NOT_IN_FORCE_AFTER_PIN'):
        W.pin_flags({})


# ------------------------------------------------------- full fake run

@pytest.fixture(scope='module')
def fake_run(tmp_path_factory):
    out = tmp_path_factory.mktemp('gpu_fake')
    rc = W.main(['--registration', str(REGISTRATION), '--out', str(out),
                 '--fake'])
    return rc, out


def test_fake_run_executes_every_registered_row(fake_run):
    rc, out = fake_run
    assert rc == 0, 'structural health, not the prediction'
    receipts = sorted((out / 'rows').glob('*.json'))
    assert len(receipts) == 23
    rows = [W.read(p) for p in receipts]
    assert all(r['fake'] is True for r in rows)
    assert all(r['schema'] == 'grm.a1.gpu_contrast_row.v1' for r in rows)
    assert all(r['pins'][W.FLAG_ENV] == '1' for r in rows)
    assert all(r['pins']['alias_fold_enabled'] is True for r in rows)
    assert all('if unspecified' not in r['question_served'] for r in rows)
    # Every row names the worker bytes that produced it.
    for row in rows:
        assert row['worker_provenance']['sha256'] == W.sha(W.SELF_PATH)


def test_fake_run_single_mount_and_controls(fake_run):
    _, out = fake_run
    summary = W.read(out / 'summary.json')
    assert summary['rows'] == 23
    assert summary['scored_rows'] == 8
    assert summary['aliases_single_mount'] == 8
    assert summary['controls'] == 5
    assert summary['controls_broken'] == 0


def test_fake_run_refuses_to_report_a_prediction(fake_run):
    """A fake run is not evidence for the registered prediction, and says so."""
    _, out = fake_run
    summary = W.read(out / 'summary.json')
    assert summary['fake'] is True
    assert summary['verdict'] == 'FAKE_RUN_NOT_A_PREDICTION'
    assert summary['prediction_met'] is None
    assert 'entity-blind' in summary['fake_note']


def test_fake_run_width_receipt_precedes_the_single_mount_claim(fake_run):
    """Every digest carries a measured width, and none was over 96."""
    _, out = fake_run
    summary = W.read(out / 'summary.json')
    assert summary['digests_over_width'] == 0
    rows = [W.read(p) for p in sorted((out / 'rows').glob('*.json'))]
    measured = 0
    for row in rows:
        for width in row['merge']['widths']:
            if width.get('digest') is None:
                continue
            measured += 1
            receipt = width['width_loaded_tokenizer']
            assert receipt['arena_width'] == 96
            assert receipt['digest_tokens'] <= 96
            assert receipt['fits'] is True
            assert receipt['single_mount_claimable'] is True
            assert 'width_at_merge_time' in width
            assert width['tokenizers_agree'] is True
    assert measured, 'the fake run must have measured at least one digest'


def test_fake_run_emits_the_probes_jsonl_lead_commands_reads(fake_run):
    _, out = fake_run
    lines = [json.loads(l) for l in
             (out / 'probes.jsonl').read_text().splitlines() if l.strip()]
    assert len(lines) == 23
    assert {'class', 'answerable', 'exact_correct', 'control_broken'} <= set(
        lines[0])


# ------------------------------------------------------ resume / create-only

def test_resume_after_interrupt_skips_done_rows_and_never_rewrites(tmp_path):
    """Interrupt, re-invoke, and the finished rows are untouched."""
    out = tmp_path / 'resume'
    # First pass: only the first 3 rows (stands in for an interrupt).
    assert W.main(['--registration', str(REGISTRATION), '--out', str(out),
                   '--fake', '--limit', '3']) == 0
    first = sorted((out / 'rows').glob('*.json'))
    assert len(first) == 3
    before = {p.name: p.read_bytes() for p in first}

    registration = W.read(REGISTRATION)
    for row in W.plan_rows(registration)[:3]:
        result = W.run_row(row, registration, out, fake=True, rule=None)
        assert result['status'] == 'SKIPPED_ALREADY_DONE', result

    assert W.main(['--registration', str(REGISTRATION), '--out', str(out),
                   '--fake', '--resume']) == 0
    after = sorted((out / 'rows').glob('*.json'))
    assert len(after) == 23
    # Byte-for-byte unchanged: create-only means a resume cannot rewrite.
    for name, payload in before.items():
        assert (out / 'rows' / name).read_bytes() == payload


def test_receipts_are_create_only(tmp_path):
    """A direct second write to an existing receipt raises, never overwrites."""
    from scripts.grm_c7_common import create

    path = tmp_path / 'row.json'
    create(path, {'a': 1})
    with pytest.raises(FileExistsError):
        create(path, {'a': 2})
    assert W.read(path) == {'a': 1}


# ---------------------------------------------------------- over-width path

def test_over_width_digest_takes_the_existing_split_path(tmp_path):
    """A digest longer than the arena width is split, and the receipt says so.

    Driven through ``apply_merge`` — the same function the worker calls — so
    this is the worker's own over-width branch, not a re-implementation.
    """
    import re as _re

    from _pytest.monkeypatch import MonkeyPatch
    from scripts import grm_c7_diagnose as diag

    mp = MonkeyPatch()
    mp.setenv(W.FLAG_ENV, '1')
    repo = diag.repository(tmp_path / 'repo', mp)

    class _Verbose(diag.Model):
        def __call__(self, ids, kv_caches=None, **kwargs):
            if kv_caches is None:
                prompt = self.codec.decode(ids[0])
                lines = _re.findall(r'^\[source \d+\] (.+)$', prompt, _re.M)
                if lines:
                    filler = ' '.join(f'Context note {n} records detail.'
                                      for n in range(140))
                    self.fold_output = (
                        'The archived record states that '
                        + ' '.join(l.rstrip('.') + '.' for l in lines)
                        + ' ' + filler + '<|end|>')
                else:
                    self.fold_output = None
            return super().__call__(ids, kv_caches=kv_caches, **kwargs)

    try:
        repo.arena.m = _Verbose(repo.arena.m.codec)
        system = ('<|start|>system<|message|>You are ChatGPT. Reasoning: '
                  'low. Valid channel: final.<|end|>')

        def node(user):
            return (system + '<|start|>user<|message|>' + user + '<|end|>'
                    + '<|start|>assistant<|channel|>final<|message|>'
                    + 'Recorded.<|end|>')

        for text in ('The current C7-Wide-0 value is Jasper-600.',
                     'C7-WideSig-0 is an alias for C7-Wide-0.'):
            idx = repo.arena.deposit(node(text))
            repo.arena.grafts[idx]['kind'] = 'turn'
        repo._sync_lifecycle()
        repo.arena.m.fold_output = None

        merge = W.apply_merge(repo, repo.arena.encode)
        assert merge['digests'] == 1
        assert merge['over_width'] == 1
        width = merge['widths'][0]
        receipt = width['width_loaded_tokenizer']
        assert receipt['fits'] is False
        assert receipt['single_mount_claimable'] is False
        assert receipt['digest_tokens'] > receipt['arena_width']
        # The EXISTING LSR-P2C guard repaired it into mountable children.
        assert width['width_guard_split']
        for child in width['width_guard_split']:
            assert int(repo.arena.grafts[child]['ntok']) <= int(
                repo.arena.width)
    finally:
        repo.close()
        mp.undo()


# =====================================================================
# AMENDMENT 2 — the three defects the 2026-09-11 lead GPU runs found.
#
# All three reached the card because the fake arm diverged from the GPU arm
# AT THE LOADER. Each test below fails on the PRE-amendment-2 worker.
# =====================================================================

# ------------------------------------------------- defect 1: lease shape

def test_lease_default_is_under_the_house_cap():
    """The first worker defaulted to 2400s; gpu_lease refuses anything >590."""
    from scripts.grm_cmc1_gpu_arms import MAX_LEASE_SECONDS

    assert MAX_LEASE_SECONDS == 590
    assert W.LEASE_SECONDS <= MAX_LEASE_SECONDS
    assert W.LEASE_SECONDS == 560, 'the value C7/LT1 use, ~30s of headroom'


def test_over_cap_lease_is_rejected_at_argparse():
    """--lease-seconds 2400 must die at PARSE time, before any flock."""
    import argparse as _argparse

    with pytest.raises(_argparse.ArgumentTypeError, match='house cap'):
        W._lease_seconds('2400')
    with pytest.raises(_argparse.ArgumentTypeError):
        W._lease_seconds('591')
    with pytest.raises(_argparse.ArgumentTypeError):
        W._lease_seconds('0')
    assert W._lease_seconds('590') == 590
    assert W._lease_seconds('560') == 560


def test_main_rejects_an_over_cap_lease_before_doing_anything(tmp_path,
                                                              capsys):
    """SystemExit(2) from argparse — not a mid-run LiveError on the card."""
    with pytest.raises(SystemExit) as excinfo:
        W.main(['--registration', str(REGISTRATION),
                '--out', str(tmp_path / 'o'), '--fake',
                '--lease-seconds', '2400'])
    assert excinfo.value.code == 2
    assert 'house cap' in capsys.readouterr().err


def test_amendment_2_registers_the_lease_shape():
    _, amendments = W.load_amendments(REGISTRATION)
    lease = amendments[-1]['amendment']['lease']
    assert lease['house_cap_seconds'] == 590
    assert lease['per_invocation_seconds'] == W.LEASE_SECONDS
    assert '--resume' in lease['shape']
    # The BUDGET is unchanged; only the lease shape moved.
    unchanged = amendments[-1]['amendment']['unchanged']
    assert unchanged['budget_gpu_hours_max'] == 0.5
    assert unchanged['rows'].startswith('23 unchanged')
    assert '5/8' in unchanged['registered_prediction']


# --------------------------------- defect 2: undeclared frame / runtime

def test_runtime_frame_and_flags_source_are_declared_and_pinned():
    reg = W.read(REGISTRATION)
    _, amendments = W.load_amendments(REGISTRATION)
    effective = W.effective_registration(reg, amendments)
    # The registration ALONE never declared these — that is the defect.
    assert 'runtime_frame' not in reg
    assert 'c7_r3_registration' not in reg
    assert 'preconditions' not in reg
    for block in ('runtime_frame', 'c7_r3_registration'):
        declared = effective[block]
        assert Path(declared['path']).is_file()
        assert declared['sha256'] == W.sha(declared['path'])
    assert set(effective['preconditions']) == {'native_runtime',
                                               'model_snapshot'}


def test_dry_run_reports_ok_and_takes_no_lease(tmp_path, monkeypatch):
    """--dry-run checks everything and never reaches a lease or a row."""
    taken = []
    monkeypatch.setattr(W, 'plan_rows',
                        lambda *a: taken.append('planned') or [])
    assert W.main(['--registration', str(REGISTRATION),
                   '--out', str(tmp_path / 'o'), '--dry-run']) == 0
    assert taken == [], 'dry-run must not plan or execute rows'


def test_dry_run_is_red_on_a_missing_frame(tmp_path):
    """The exact stop the lead hit, now caught on CPU with no lease."""
    reg = W.read(REGISTRATION)
    _, amendments = W.load_amendments(REGISTRATION)
    effective = W.effective_registration(reg, amendments)
    effective['runtime_frame'] = {
        'path': str(tmp_path / 'absent_frame.json'), 'sha256': 'a' * 64}
    result = W.check_preconditions(effective)
    assert result['ok'] is False
    names = {row['name'] for row in result['missing']}
    assert 'runtime_frame' in names
    # ... and loading it raises the NAMED error, not a bare FileNotFoundError.
    with pytest.raises(W.ContrastError, match='A1_RUNTIME_FRAME_MISSING'):
        W.load_runtime_frame(effective)


def test_dry_run_is_red_on_a_missing_native_runtime(tmp_path):
    """The OSError the lead hit AFTER the lease, now a pre-lease RED."""
    reg = W.read(REGISTRATION)
    _, amendments = W.load_amendments(REGISTRATION)
    effective = W.effective_registration(reg, amendments)
    effective['preconditions'] = dict(effective['preconditions'])
    effective['preconditions']['native_runtime'] = {
        'path': str(tmp_path / 'libgrm_runtime.so'), 'gpu_only': True}
    result = W.check_preconditions(effective)
    assert result['ok'] is False
    assert {row['name'] for row in result['missing']} == {'native_runtime'}
    # On --fake the GPU-only precondition is skipped, and says so.
    fake = W.check_preconditions(effective, fake=True)
    assert fake['ok'] is True
    skipped = [r for r in fake['checks'] if r['name'] == 'native_runtime']
    assert skipped[0]['detail'] == 'skipped on --fake'


def test_run_refuses_to_take_a_lease_when_a_precondition_is_missing(
        tmp_path, monkeypatch):
    """A1_PRECONDITION_MISSING fires BEFORE gpu_lease is ever imported."""
    leased = []
    monkeypatch.setattr(W, 'check_preconditions',
                        lambda *a, **k: {'checks': [], 'ok': False,
                                         'missing': [{'name': 'native_runtime',
                                                      'path': '/nope'}]})
    monkeypatch.setattr(W, 'plan_rows',
                        lambda *a: leased.append('planned') or [])
    with pytest.raises(W.ContrastError, match='A1_PRECONDITION_MISSING'):
        W.main(['--registration', str(REGISTRATION),
                '--out', str(tmp_path / 'o'), '--fake'])
    assert leased == []


def test_frame_sha_drift_is_named(tmp_path):
    reg = W.read(REGISTRATION)
    _, amendments = W.load_amendments(REGISTRATION)
    effective = W.effective_registration(reg, amendments)
    frame = tmp_path / 'frame.json'
    frame.write_text(json.dumps({'schema': 'x', 'resolved_flags': {}}) + '\n')
    effective['runtime_frame'] = {'path': str(frame), 'sha256': 'b' * 64}
    with pytest.raises(W.ContrastError, match='A1_RUNTIME_FRAME_SHA_MISMATCH'):
        W.load_runtime_frame(effective)


# ------------------------------ defect 3: flags shape / environment(flags)

def test_flags_come_from_the_c7_r3_registration_not_the_rs1_frame():
    """The RS1 frame carries NO C2 frame keys — that was the KeyError."""
    reg = W.read(REGISTRATION)
    _, amendments = W.load_amendments(REGISTRATION)
    effective = W.effective_registration(reg, amendments)

    flags = W.resolve_flags(effective)
    assert W.ENVIRONMENT_FLAG_KEYS <= set(flags)
    assert flags['ephemeral'] is True
    assert flags['arena_width'] == 96

    # The flags are READ from the C7 r3 registration's arm, verbatim.
    c7 = W.read(effective['c7_r3_registration']['path'])
    assert flags == c7['arms'][effective['c7_r3_registration']['arm']]['flags']

    # The RS1 frame's own resolved_flags would have raised KeyError.
    rs1_flags = W.load_runtime_frame(effective)['resolved_flags']
    assert not W.ENVIRONMENT_FLAG_KEYS <= set(rs1_flags)
    assert 'ephemeral' not in rs1_flags
    with pytest.raises(KeyError, match='ephemeral'):
        W.build_environment(rs1_flags)


def test_environment_is_built_from_those_flags_without_raising():
    reg = W.read(REGISTRATION)
    _, amendments = W.load_amendments(REGISTRATION)
    effective = W.effective_registration(reg, amendments)
    env = W.build_environment(W.resolve_flags(effective))
    assert env['GRM_PERSISTENT_BOAT'] == '0'   # ephemeral True -> not -> 0
    assert env['GRM_CAPTURE_PIN'] == 'live'
    # Still carries NEITHER A1 switch, so the post-pin contract holds.
    assert W.FLAG_ENV not in env and W.RULE_ENV not in env


def test_incomplete_flags_are_a_named_error_not_a_keyerror(tmp_path):
    """A missing frame key fails HERE, by name, not deep inside a GPU load."""
    reg = W.read(REGISTRATION)
    _, amendments = W.load_amendments(REGISTRATION)
    effective = W.effective_registration(reg, amendments)
    c7 = W.read(effective['c7_r3_registration']['path'])
    del c7['arms']['A']['flags']['ephemeral']
    path = tmp_path / 'c7.json'
    path.write_text(json.dumps(c7) + '\n')
    effective['c7_r3_registration'] = {'path': str(path), 'sha256': W.sha(path),
                                       'arm': 'A'}
    with pytest.raises(W.ContrastError, match='A1_FLAGS_INCOMPLETE'):
        W.resolve_flags(effective)


def test_fake_arm_uses_the_same_flags_and_environment_as_the_gpu_arm(
        tmp_path, monkeypatch):
    """THE regression guard: the fake loader must call the SAME helpers.

    Both defects 2 and 3 survived the first gate because ``_load_fake``
    returned a hand-made flags dict and ``dict(os.environ)``. If it ever does
    that again, the spies below see zero calls and this fails.
    """
    reg = W.read(REGISTRATION)
    _, amendments = W.load_amendments(REGISTRATION)
    effective = W.effective_registration(reg, amendments)

    seen = {'flags': [], 'env': [], 'frame': []}
    real_flags, real_env, real_frame = (
        W.resolve_flags, W.build_environment, W.load_runtime_frame)
    monkeypatch.setattr(W, 'resolve_flags',
                        lambda r: seen['flags'].append(r) or real_flags(r))
    monkeypatch.setattr(W, 'build_environment',
                        lambda f: seen['env'].append(f) or real_env(f))
    monkeypatch.setattr(W, 'load_runtime_frame',
                        lambda r: seen['frame'].append(r) or real_frame(r))

    loaded = W._load_fake(None, effective)
    try:
        assert len(seen['flags']) == 1 and len(seen['env']) == 1
        assert len(seen['frame']) == 1, 'the fake arm must READ the frame'
        # The env handed downstream is environment()'s, not os.environ.
        assert loaded['environment'] is not os.environ
        assert loaded['environment']['GRM_CAPTURE_PIN'] == 'live'
        assert W.ENVIRONMENT_FLAG_KEYS <= set(loaded['flags'])
    finally:
        loaded['repo'].close()
        loaded['monkeypatch'].undo()


def test_fake_rows_record_the_real_frame_and_pins(tmp_path):
    """End to end on --fake: the receipt proves the frame was really read."""
    out = tmp_path / 'frames'
    assert W.main(['--registration', str(REGISTRATION), '--out', str(out),
                   '--fake', '--limit', '2']) == 0
    rows = [W.read(p) for p in sorted((out / 'rows').glob('*.json'))]
    assert rows
    for row in rows:
        assert row['model']['frame_schema'] == 'grm.det1.runtime_frame.v1'
        assert row['pins'][W.FLAG_ENV] == '1'


# ------------------------------------------------- resumable lease loop

def test_summary_reports_pending_and_lease_state(tmp_path):
    out = tmp_path / 'pending'
    rc = W.main(['--registration', str(REGISTRATION), '--out', str(out),
                 '--fake', '--limit', '3'])
    assert rc == 0
    summary = W.read(out / 'summary.json')
    assert summary['pending'] == 0 and summary['complete'] is True
    assert summary['lease_expired'] is False
