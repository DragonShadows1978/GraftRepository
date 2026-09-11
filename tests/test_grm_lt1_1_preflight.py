"""LT1.1 chain preflight — the lead's ruling, gated.

THE RULING (lead, 2026-09-11): LT1's original registration is a frozen receipt
of its day and is NOT re-validated against today's core. LT1.1's preflight
validates LT1.1's OWN chain, using the same verification classes, pointed at
our documents. LT1's identity is carried as parent lineage, not as a gate.

The danger in a ruling like this is that "stop checking X" quietly becomes
"stop checking". So the load-bearing test here is
`test_a_planted_drift_in_our_own_amendment_goes_red`: it plants a drifted core
sha in LT1.1's OWN amendment 1 and requires the preflight to catch it. The
gate still has teeth; it just points them at the right chain.

Prior art: the verification classes are LT1's own (`grm_lt1.verify` /
`preflight`, GRM contributors 2026), reused in kind. Full annotation in
scripts/grm_lt1_1.py:lt1_1_preflight.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_lt1 as lt                # noqa: E402
from scripts import grm_lt1_1 as runner          # noqa: E402


@pytest.fixture
def pinned():
    """The preflight checks the pinned admission rule, as LT1's does."""
    with runner.pinned_arm('A'):
        yield


def _stage(monkeypatch, tmp_path):
    """A writable copy of the LT1.1 document set, for tamper tests."""
    staged = tmp_path / 'lt1_1'
    shutil.copytree(runner.OUT, staged)
    monkeypatch.setattr(runner, 'OUT', staged)
    monkeypatch.setattr(runner, 'REGISTRATION', staged / 'registration.json')
    monkeypatch.setattr(runner, 'AMENDMENT1', staged / 'amendment1.json')
    monkeypatch.setattr(runner, 'FIXTURE', staged / 'dialogue.json')
    return staged


def _rewrite(path, doc):
    payload = (json.dumps(doc, sort_keys=True, indent=2) + '\n').encode()
    path.write_bytes(payload)
    path.with_suffix('.sha256').write_text(
        hashlib.sha256(payload).hexdigest() + '  ' + path.name + '\n')


# --------------------------------------------------------------- it passes

@pytest.mark.campaign_receipt(
    registration='artifacts/grm_d1/lt1_1/registration.json + artifacts/grm_lt1/amendment4/resume_registration.json')
@pytest.mark.parametrize('arm', ['A', 'A+'])
def test_the_chain_preflight_is_ready_on_this_tree(arm):
    with runner.pinned_arm(arm):
        gate = runner.lt1_1_preflight(arm)
    assert gate['status'] == 'READY', gate['reasons']
    assert gate['reasons'] == []
    assert gate['gpu_executed'] is False


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_d1/lt1_1/registration.json + artifacts/grm_lt1/amendment4/resume_registration.json')
def test_no_input_sha_mismatch_anywhere(pinned):
    """The blocker the ruling resolves must be gone, by name."""
    gate = runner.lt1_1_preflight('A')
    assert not any('INPUT_SHA_MISMATCH' in r for r in gate['reasons'])


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_d1/lt1_1/registration.json + artifacts/grm_lt1/amendment4/resume_registration.json')
def test_it_checks_the_classes_the_ruling_names(pinned):
    """sha-bound inputs, fixture sha, cell schedule, budget."""
    gate = runner.lt1_1_preflight('A')
    assert gate['inputs_checked'] >= 6        # 5 rebound + 1 new + runner
    assert gate['fixture_sha256'] == runner.sha(runner.FIXTURE)
    assert gate['cells'] == 26
    assert gate['reservation_seconds'] <= gate['budget_gpu_seconds']
    assert gate['free_bytes'] > gate['minimum_free_bytes']
    # Every amendment on disk is verified, not a hard-coded subset.
    on_disk = {p.name for p in runner.OUT.glob('amendment[0-9]*.json')}
    assert on_disk <= set(gate['document_sha256'])
    assert 'registration.json' in gate['document_sha256']


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_d1/lt1_1/registration.json + artifacts/grm_lt1/amendment4/resume_registration.json')
def test_the_ruling_is_carried_in_the_receipt(pinned):
    gate = runner.lt1_1_preflight('A')
    assert 'frozen receipt' in gate['ruling']
    assert 'not as a gate' in gate['ruling']


# ------------------------------------------------------ it still has teeth

def test_a_planted_drift_in_our_own_amendment_goes_red(monkeypatch, tmp_path):
    """THE teeth test: drift in LT1.1's OWN chain must still be caught.

    The ruling stops re-validating LT1's day-of pins. It does NOT stop
    validating ours. Plant a bad core sha in LT1.1's own amendment 1 and
    require the preflight to report INPUT_SHA_MISMATCH for that input. The
    sidecar is kept consistent, so the failure is the INPUT check and not
    merely a document-sha mismatch.
    """
    staged = _stage(monkeypatch, tmp_path)
    doc = json.loads((staged / 'amendment1.json').read_text())
    victim = 'core/graft_arena.py'
    assert victim in doc['core_rebind']['changed']
    doc['core_rebind']['changed'][victim]['after_sha256'] = 'deadbeef' * 8
    _rewrite(staged / 'amendment1.json', doc)

    with runner.pinned_arm('A'):
        gate = runner.lt1_1_preflight('A')
    assert gate['status'] == 'BLOCKED'
    assert 'INPUT_SHA_MISMATCH: ' + victim in gate['reasons'], gate['reasons']


def test_a_planted_drift_in_a_new_input_goes_red(monkeypatch, tmp_path):
    """The new-input branch needs teeth too, not just the changed one."""
    staged = _stage(monkeypatch, tmp_path)
    doc = json.loads((staged / 'amendment1.json').read_text())
    victim = 'core/grm_alias_fold.py'
    doc['core_rebind']['new_inputs'][victim]['sha256'] = 'deadbeef' * 8
    _rewrite(staged / 'amendment1.json', doc)
    with runner.pinned_arm('A'):
        gate = runner.lt1_1_preflight('A')
    assert 'INPUT_SHA_MISMATCH: ' + victim in gate['reasons']


def test_a_broken_chain_goes_red(monkeypatch, tmp_path):
    staged = _stage(monkeypatch, tmp_path)
    doc = json.loads((staged / 'amendment2.json').read_text())
    doc['previous_amendment_sha256'] = 'deadbeef' * 8
    _rewrite(staged / 'amendment2.json', doc)
    with runner.pinned_arm('A'):
        gate = runner.lt1_1_preflight('A')
    assert gate['status'] == 'BLOCKED'
    assert 'AMENDMENT2_CHAIN_MISMATCH' in gate['reasons']


def test_a_tampered_document_goes_red(monkeypatch, tmp_path):
    staged = _stage(monkeypatch, tmp_path)
    (staged / 'amendment3.json').write_text('{"amendment": 3}\n')
    with runner.pinned_arm('A'):
        gate = runner.lt1_1_preflight('A')
    assert gate['status'] == 'BLOCKED'
    assert 'DOCUMENT_SHA_MISMATCH: amendment3.json' in gate['reasons']


def test_a_tampered_fixture_goes_red(monkeypatch, tmp_path):
    staged = _stage(monkeypatch, tmp_path)
    (staged / 'dialogue.json').write_text('{}\n')
    with runner.pinned_arm('A'):
        gate = runner.lt1_1_preflight('A')
    assert gate['status'] == 'BLOCKED'
    assert 'FIXTURE_SHA_MISMATCH' in gate['reasons']


def test_an_unpinned_admission_rule_goes_red(monkeypatch):
    monkeypatch.delenv(runner.RULE_ENV, raising=False)
    gate = runner.lt1_1_preflight('A')
    assert 'REGISTERED_ADMISSION_RULE_MISMATCH' in gate['reasons']


# ------------------------------------------------------- the parent lineage

@pytest.mark.campaign_receipt(
    registration='artifacts/grm_d1/lt1_1/registration.json + artifacts/grm_lt1/amendment4/resume_registration.json')
def test_lt1_identity_is_recorded_not_gated(pinned):
    gate = runner.lt1_1_preflight('A')
    parent = gate['parent']
    assert parent['lt1_registration_sha256'] == runner.sha(
        ROOT / 'artifacts/grm_lt1/registration.json')
    assert parent['lt1_recorded_core_pins'] == 26
    assert 'not gated' in parent['status']
    inputs = {row['input'] for row in parent['drift_table']}
    assert 'core/graft_arena.py' in inputs
    assert 'core/grm_alias_fold.py' in inputs
    for row in parent['drift_table']:
        assert row['attribution']
        # Our rebound sha is what is actually on the tree ...
        assert row['lt1_1_rebound'] == row['on_tree_now']
        # ... and differs from LT1's frozen record, which is the whole point.
        if row['lt1_recorded'] is not None:
            assert row['lt1_recorded'] != row['on_tree_now']


def test_apply4_protocol_binding_still_passes_untouched():
    """The lead asked. Measured answer: it passes, and we did not touch it."""
    recorded = json.loads(
        (ROOT / 'artifacts/grm_lt1/amendment4/resume_registration.json')
        .read_text())['protocol_binding']
    assert recorded == lt.binding('CPU'), (
        'apply4 protocol-binding check would fail; the ruling would then need '
        'to extend to it as well')


def test_the_old_lt1_gate_is_no_longer_called():
    """A mention in prose is fine; an actual CALL is not.

    The docstrings deliberately explain what `lt.preflight` did and why it was
    replaced, so a substring search would match the explanation. Check the
    parsed AST for a real call instead.
    """
    import ast
    source = (ROOT / 'scripts/grm_lt1_1.py').read_text()
    assert 'def lt1_1_preflight(' in source
    assert 'gate = (lt1_1_preflight(arm)' in source
    calls = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Attribute) and func.attr == 'preflight'
                and isinstance(func.value, ast.Name) and func.value.id == 'lt'):
            calls.append(node.lineno)
    assert calls == [], (
        'lt.preflight() is still called at line(s) %s; the ruling says LT1 '
        'registration is a frozen receipt, not a gate' % calls)


# ---------------------------- the amendment-3 blocker, INVERTED with receipt

@pytest.mark.campaign_receipt(
    registration='artifacts/grm_d1/lt1_1/registration.json + artifacts/grm_lt1/amendment4/resume_registration.json')
def test_the_amendment3_host_blocker_is_resolved():
    """Inverts `test_the_host_blocker_claim_is_true_right_now`.

    Amendment 3 recorded an OPEN blocker: the real `--resume` stopped at
    `INPUT_SHA_MISMATCH: core/graft_arena.py`. Under the ruling that check is
    no longer a gate, so the blocker is resolved. Both halves are asserted:
    the underlying LT1 condition is UNCHANGED (still drifted -- we did not
    paper over it), and the LT1.1 route is now green anyway.
    """
    # 1. The underlying condition is untouched: LT1's own gate still refuses.
    with pytest.raises(ValueError, match='INPUT_SHA_MISMATCH'):
        lt.preflight()
    # 2. And LT1.1's gate, which is the one that governs, is READY.
    with runner.pinned_arm('A+'):
        gate = runner.lt1_1_preflight('A+')
    assert gate['status'] == 'READY', gate['reasons']
    # 3. The amendment records the resolution.
    amendment4 = runner.OUT / 'amendment4.json'
    if amendment4.exists():
        doc = json.loads(amendment4.read_text())
        assert doc['resolves']['status'] == 'RESOLVED by ruling'
        assert 'INPUT_SHA_MISMATCH' in doc['resolves']['blocker']


# --------------------------------------------------- the route, host gate ON

@pytest.mark.campaign_receipt(
    registration='artifacts/grm_d1/lt1_1/registration.json + artifacts/grm_lt1/amendment4/resume_registration.json')
@pytest.mark.parametrize('arm', ['A', 'A+'])
def test_dry_lease_passes_with_the_host_gate_on(arm, tmp_path):
    """The gate the lead asked for: both arms, real route, no escape hatch."""
    value = runner.resume(arm, root=tmp_path / arm.replace('+', 'plus'),
                          dry_lease=True)          # host_gate defaults True
    assert value['status'] == 'PASS'
    assert value['host_preflight'] == 'READY'
    assert value['stopped_at'] == 'worker.run_cell (lease boundary)'
    assert value['next_cell'] == 'A-001-008'
    assert value['pinned']['alias_fold_merge'] is (arm == 'A+')
    assert 'INPUT_SHA_MISMATCH' not in json.dumps(value)


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_d1/lt1_1/registration.json + artifacts/grm_lt1/amendment4/resume_registration.json')
@pytest.mark.parametrize('arm', ['A', 'A+'])
def test_dry_lease_passes_from_the_cli_without_no_host_gate(arm, tmp_path):
    result = subprocess.run(
        [sys.executable, 'scripts/grm_lt1_1.py', '--arm', arm, '--resume',
         '--dry-lease', '--out', str(tmp_path / arm.replace('+', 'plus'))],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload['host_preflight'] == 'READY'
    assert payload['stopped_at'] == 'worker.run_cell (lease boundary)'
