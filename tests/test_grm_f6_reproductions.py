"""GRM-F6: each rebound script reproduces the number its receipt recorded.

A rebind is only honest if the repo-relative root yields the SAME answer the
absolute root yielded on the day. These are the reproductions, each pinned
against a frozen receipt under ``artifacts/``:

==============================  ==========================================
script                          receipt reproduced
==============================  ==========================================
grm_lt1_offline_supplement      artifacts/grm_f5/c2_replay_off.json
  (via grm_f5_c2_replay_gate)     -> rows 132, off_byte_identical 132
grm_scout_fix8_cpu.c2_plans     artifacts/grm_scout_fix8/green.json
                                  -> c2 132, c2_identical 132, hex parity
grm_r1_register.measured_c2_    artifacts/grm_r1/registration.json
  cost                            -> cost_model, float-exact
grm_rd1.census                  artifacts/grm_rd1/registration.json
                                  -> counts {52, 520, 30, 16}, 16 cells,
                                     100 sha-bound inputs identical
grm_d1_cause_table              tests/test_grm_d1_cause_table.py, whose 12
                                  gates were SKIPPED (not vacuous) while
                                  ct.CELLS pointed at the pruned worktree
==============================  ==========================================

These read gitignored campaign artifacts, so they carry the
``campaign_receipt`` marker and are skipped by default like every other
receipt in this tree (see tests/conftest.py). ``-m campaign_receipt`` runs
them.

Prior art: golden/characterization testing -- pin the current output, then
refactor against it (Michael Feathers, *Working Effectively with Legacy
Code*, 2004; "approval tests", Llewellyn Falco, 2011). Taken: the shape --
the receipt is the golden, the rebind is the refactor. Ours: nothing
algorithmic; the receipts and their numbers are the prior campaigns' (GRM
contributors, 2026), reproduced here rather than re-derived.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RECEIPT = 'artifacts/grm_f6/ (GRM-F6 rebind reproductions)'


def _read(relative):
    return json.loads((ROOT / relative).read_text())


@pytest.mark.campaign_receipt(registration=RECEIPT)
def test_f5_c2_replay_gate_reproduces_132_of_132():
    """artifacts/grm_f5/c2_replay_off.json: rows 132, off_byte_identical 132."""
    want = _read('artifacts/grm_f5/c2_replay_off.json')
    proc = subprocess.run(
        [sys.executable, 'scripts/grm_f5_c2_replay_gate.py'],
        cwd=str(ROOT), capture_output=True, text=True, timeout=560)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)
    assert got['rows'] == want['rows'] == 132
    assert got['off_byte_identical'] == want['off_byte_identical'] == 132
    assert got['verdict'] == want['verdict'] == 'PASS'


@pytest.mark.campaign_receipt(registration=RECEIPT)
def test_scout_fix8_c2_plans_reproduce_green_json():
    """artifacts/grm_scout_fix8/green.json: c2 132, c2_identical 132."""
    from scripts import grm_scout_fix8_cpu as f8
    green = _read('artifacts/grm_scout_fix8/green.json')
    identity = _read('artifacts/grm_scout_fix8/c2_identity.json')
    frozen_rows = green['c2']          # the 132 per-execution receipt rows
    rows = f8.c2_plans()

    assert len(rows) == len(frozen_rows) == green['c2_identical'] == 132
    assert identity['count'] == 132 and identity['byte_identical'] is True
    assert sum(1 for r in rows if r['identical']) == green['c2_identical'] == 132

    # Byte parity, not just the count: the recorded/replayed rank_plan hex of
    # every execution must equal the frozen receipt's, in the same order.
    for got, frozen in zip(rows, frozen_rows):
        assert got['execution_id'] == frozen['execution_id']
        assert got['recorded_hex'] == frozen['recorded_hex']
        assert got['replayed_hex'] == frozen['replayed_hex']
        # sha-bound receipts are untouched; only the ROOT prefix moved.
        assert got['checkpoint_sha256'] == frozen['checkpoint_sha256']
        assert got['manifest_sha256'] == frozen['manifest_sha256']
        assert got['source_sha256'] == frozen['source_sha256']
        assert (got['checkpoint'].split('/artifacts/grm_c2/', 1)[1]
                == frozen['checkpoint'].split('/artifacts/grm_c2/', 1)[1])


@pytest.mark.campaign_receipt(registration=RECEIPT)
def test_r1_measured_c2_cost_reproduces_registration_float_exact():
    """artifacts/grm_r1/registration.json cost_model, to the last float bit."""
    from scripts import grm_r1_register as reg
    want = _read('artifacts/grm_r1/registration.json')['cost_model']
    got = reg.measured_c2_cost()
    for side in ('defaults', 'profile'):
        assert got[side] == want[side], side
    assert got['evidence_class'] == want['evidence_class']
    assert len(got['rows']) == 20         # the C2 restart controllers


@pytest.mark.campaign_receipt(registration=RECEIPT)
def test_rd1_census_reproduces_registration_counts_and_shas():
    """artifacts/grm_rd1/registration.json: counts, cells, and input shas.

    The registration's absolute ``/mnt/ForgeRealm/wt/grm-c7/`` keys are NOT
    rewritten -- they are the receipt of what was hashed on the day. The
    check strips each side's own root prefix and compares the shas.
    """
    from scripts import grm_rd1 as rd
    reg = _read('artifacts/grm_rd1/registration.json')
    fixture, rows, cells, inputs = rd.census(rd.SOURCE)

    requests, memory_failures, oracle_failures = [], 0, 0
    for row in rows:
        for side in rd.SIDES:
            for arm in rd.ARMS:
                requests.append(rd.request(row, side, arm))
        if row['probe']['answerable']:
            memory_failures += int(
                bool(row['historical']['memory']['score']['abstention_error']))
            oracle_failures += int(
                bool(row['historical']['oracle']['score']['abstention_error']))
    assert {'probes': len(rows), 'requests': len(requests),
            'memory_failure_sides': memory_failures,
            'oracle_failure_sides': oracle_failures} == reg['counts']
    assert [c['id'] for c in cells] == [c['id'] for c in reg['cells']]

    old_prefix = '/mnt/ForgeRealm/wt/grm-c7/'
    new_prefix = str(rd.SOURCE) + '/'
    got = {k.replace(new_prefix, ''): v for k, v in inputs.items()}
    want = {k.replace(old_prefix, ''): v for k, v in reg['inputs'].items()}
    common = set(got) & set(want)
    assert len(common) == len(got) == 100, sorted(set(got) - set(want))
    assert all(got[k] == want[k] for k in common), (
        'sha mismatch: %s' % sorted(k for k in common if got[k] != want[k]))


@pytest.mark.campaign_receipt(registration=RECEIPT)
def test_rd1_requests_identical_modulo_the_source_root_prefix():
    """All 520 arm requests match artifacts/grm_rd1/requests.json exactly.

    The ONLY field that may differ is ``checkpoint``, and only by its root
    prefix -- that is precisely the rebind, and nothing else moved.
    """
    from scripts import grm_rd1 as rd
    frozen = _read('artifacts/grm_rd1/requests.json')
    fixture, rows, cells, inputs = rd.census(rd.SOURCE)
    got = []
    for row in rows:
        for side in rd.SIDES:
            for arm in rd.ARMS:
                got.append({'arm': arm, **rd.request(row, side, arm)})
    assert len(got) == len(frozen) == 520

    old_prefix, new_prefix = '/mnt/ForgeRealm/wt/grm-c7/', str(rd.SOURCE) + '/'

    def strip(record, prefix):
        return {k: (v.replace(prefix, '') if isinstance(v, str) else v)
                for k, v in record.items()}

    differing = set()
    for mine, theirs in zip(got, frozen):
        a, b = strip(mine, new_prefix), strip(theirs, old_prefix)
        differing |= {k for k in a if a[k] != b[k]}
    assert differing == set(), differing


@pytest.mark.campaign_receipt(registration=RECEIPT)
def test_d1_cause_table_gates_are_live_again():
    """The 12 gates that were SKIPPED while ct.CELLS pointed at grm-lt1.

    This dead path did not pass vacuously, it DESELECTED: the module carries
    ``skipif(not ct.CELLS.exists())``. The rebind makes the gates run.
    """
    from scripts import grm_d1_cause_table as ct
    assert ct.CELLS.exists() and ct.FIXTURE.exists()
    assert ct.CELLS.is_relative_to(ROOT) and ct.FIXTURE.is_relative_to(ROOT)
    proc = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', '--no-header', '-p', 'no:cacheprovider',
         '--basetemp', str(ROOT / 'artifacts/grm_f6/tmp_d1'),
         'tests/test_grm_d1_cause_table.py'],
        cwd=str(ROOT), capture_output=True, text=True, timeout=560)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert '12 passed' in proc.stdout, proc.stdout
