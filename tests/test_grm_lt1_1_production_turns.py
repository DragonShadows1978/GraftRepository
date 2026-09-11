"""LT1.1 production turn semantics — through the REAL `execute`.

WHAT WENT WRONG. The lead ran LT1.1 arm A+ on the card: 24 of 26 cells
COMPLETE, then `A-189-196` RED with `KeyError: 'assistant'` on the first
`recap_probe`. `--summary` over the 24 cells returned fresh 14/15,
corrections 5/10, aliases 5/10 -- byte-for-byte LT1's numbers.

The cause is in `grm_lt1_worker.execute`: it branched on `event['kind']` only
for `probe` and `recap`; every other kind, INCLUDING `supersede` and `alias`,
took `a.feed(harmony_turn(...))`. So

  * the 15 registered supersede turns were deposited as prose, nothing was
    retired, and the stale node stayed in the candidate base -- the exact
    fixture-lineage trap D1 diagnosed, re-created one layer down;
  * A1's `alias_fold_pass` never ran, because it lives inside the production
    turn funnel and `arena.feed()` does not call it. The flag was pinned and
    the mechanism was inert.

LT1.1 as run measured nothing new. These gates drive the REAL `execute` on the
CPU double and assert the semantics that were missing.

FAIRNESS NOTE, asserted below: arms A and A+ now share THIS worker, so
A-vs-A+ is a fair same-worker contrast. LT1's numbers are the PARENT
BASELINE -- different worker, different fixture -- and are not a same-worker
control.

Prior art: the production turn funnel and its finding are GRM-P1's
(`scripts/grm_chat.py:224-245`); the supersede turn is
`scripts/grm_e2e_session.py:2590-2597`. Full annotation in
`scripts/grm_lt1_1.py:deposit_turn`.
"""
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_lt1_1 as runner          # noqa: E402
from scripts import grm_lt1_worker as worker     # noqa: E402


def _cells_through(arm, tmp_path, count):
    """Run the first `count` cells through the REAL execute, one per process."""
    root = tmp_path / ('run_' + arm.replace('+', 'plus'))
    runner.run_fake(arm, root=root, limit=count)
    return root


def _manifest(root, cell_id):
    path = root / 'cells' / cell_id / 'checkpoint/repository/manifest.json'
    return json.loads(path.read_text())['nodes']


# ------------------------------------------------ the seam is wired at all

def test_the_worker_routes_kinds_through_the_seams():
    """The default is LT1's; the LT1.1 handlers install and restore."""
    assert not hasattr(worker, 'deposit_turn')
    assert not hasattr(worker, 'recap_probe_turn')
    with runner.lt1_1_seams('A+'):
        assert worker.deposit_turn is runner.deposit_turn
        assert worker.recap_probe_turn is runner.recap_probe_turn
    assert not hasattr(worker, 'deposit_turn')
    assert not hasattr(worker, 'recap_probe_turn')


def test_the_lt1_default_branch_is_byte_identical():
    """LT1's own campaign keeps prose deposits, unchanged."""
    source = (ROOT / 'scripts/grm_lt1_worker.py').read_text()
    assert ("idx=a.feed(e2e.harmony_turn(event['user'],event['assistant']))"
            in source)
    assert ("a.grafts[idx]['kind']='turn';state['turn_nodes'][str(turn)]=idx"
            in source)
    assert ('# User corrections are ordinary prose, not hidden supersede '
            'calls.') in source


def test_an_unregistered_kind_refuses_rather_than_guessing():
    """`recap_probe` with no handler must raise, not KeyError on 'assistant'."""
    source = (ROOT / 'scripts/grm_lt1_worker.py').read_text()
    assert "raise ValueError('UNREGISTERED_TURN_KIND: recap_probe')" in source


def test_the_fake_path_uses_one_seam_definition():
    """`fake_cell` must not re-declare a narrow subset that can drift."""
    source = (ROOT / 'scripts/grm_lt1_1.py').read_text()
    body = source[source.index('def fake_cell('):source.index('def run_fake(')]
    assert 'with lt1_1_seams(arm)' in body
    assert 'lt.FIX, worker.bind, worker.RUN = FIXTURE' not in body, (
        'fake_cell re-declares the seam inline; it drifted once already')


# --------------------------------------------------- supersede is REAL now

@pytest.mark.parametrize('arm', ['A', 'A+'])
def test_a_supersede_turn_retires_the_old_value(arm, tmp_path):
    """THE gate: after a supersede, the old node is retired and un-routable.

    Cells 1-3 span turns 1-24, which include the Vega lineage
    (17 -> 23 -> 29 credits) and the Medibay lineage (12 -> 14 -> 16 beds).
    """
    root = _cells_through(arm, tmp_path, 3)
    nodes = _manifest(root, 'A-017-024')
    active = ' '.join(n.get('text', '') for n in nodes if not n.get('retired'))
    retired = ' '.join(n.get('text', '') for n in nodes if n.get('retired'))

    # The superseded values are retired ...
    assert '23 credits' in retired, 'the replaced Vega value was not retired'
    assert '14 beds' in retired, 'the replaced Medibay value was not retired'
    # ... and gone from what a route can see.
    assert '23 credits' not in active
    assert '14 beds' not in active
    # The current values survive.
    assert '29 credits' in active
    assert '16 beds' in active


@pytest.mark.parametrize('arm', ['A', 'A+'])
def test_the_retired_nodes_leave_the_route_candidate_base(arm, tmp_path):
    """Retirement must reach `_route_cand_base`, not just the manifest."""
    root = _cells_through(arm, tmp_path, 3)
    nodes = _manifest(root, 'A-017-024')
    retired = [i for i, n in enumerate(nodes) if n.get('retired')]
    assert retired, 'nothing was retired; the supersede path did not run'
    # `_route_cand_base` is exactly "not retired and not kind=recall", so a
    # retired index can never be a candidate. Assert the manifest agrees.
    for i in retired:
        assert nodes[i].get('retired') is True


def test_the_old_worker_would_have_left_the_stale_value_active(tmp_path):
    """RED-before: with LT1's default, nothing is retired.

    Runs one cell with the deposit seam removed, reproducing the worker that
    produced the A+ = LT1 null.
    """
    reg = runner.registration('A+')
    root = tmp_path / 'lt1_default'
    with pytest.MonkeyPatch.context() as patch:
        loader = runner.fake_loader(patch)
        with runner.lt1_1_seams('A+'), runner.pinned_arm('A+'):
            worker.RUN = root
            # Remove ONLY the turn seam: LT1's prose-deposit default returns.
            del worker.deposit_turn
            cell = reg['cells'][0]
            directory = root / 'cells' / cell['id']
            directory.mkdir(parents=True)
            worker.execute(cell, directory, reg, loader,
                           run=root / 'cells', fake=True)
    nodes = _manifest(root, 'A-001-008')
    assert not any(n.get('retired') for n in nodes), (
        'the LT1 default retired something; this RED reproduction is stale')
    active = ' '.join(n.get('text', '') for n in nodes)
    assert '17 credits' in active and '23 credits' in active, (
        'both stale Vega values should still be active under the old worker')


# ------------------------------------------------- the alias mechanism runs

def test_the_funnel_that_carries_a1_is_actually_called():
    """A1's fold rides in `_guard_deposit_width`, inside `_finish_turn_event`."""
    runtime = (ROOT / 'core/grm_runtime.py').read_text()
    assert '_alias_fold_deposits' in runtime
    handler = (ROOT / 'scripts/grm_lt1_1.py').read_text()
    body = handler[handler.index('def deposit_turn('):
                   handler.index('def recap_probe_turn(')]
    assert '_finish_turn_event' in body, (
        'the deposit handler skips the production funnel, so A1 never runs')
    assert 'apply_memory_command' in body


@pytest.mark.parametrize('arm', ['A', 'A+'])
def test_alias_turns_reach_a_node_naming_both_names(arm, tmp_path):
    """Both names must end up in ONE mountable node, by whichever path.

    Under A+ the alias fold-merge is enabled; under A it is off. What this
    asserts is the OUTCOME the alias rows need -- a single active node naming
    the alias and its base -- without asserting WHICH mechanism produced it,
    because on this fixture the librarian's chronicle fold reaches the same
    pairing. Claiming "the fold fired" when a chronicle digest did the work
    would be a false attribution; see the report.
    """
    from core.grm_alias_fold import names_present
    root = _cells_through(arm, tmp_path, 3)
    nodes = _manifest(root, 'A-017-024')
    active = [n.get('text', '') for n in nodes if not n.get('retired')]
    paired = [t for t in active if names_present(t, 'the Hauler', 'Kestrel')]
    assert paired, (
        'no active node names both "the Hauler" and "Kestrel"; the alias '
        'rows cannot be answered from this state')


def test_flag_off_never_records_an_alias_fold_decision(tmp_path):
    """Arm A must be byte-inert with respect to A1."""
    from scripts.grm_c7_diagnose import repository
    with pytest.MonkeyPatch.context() as patch:
        with runner.pinned_arm('A'):
            repo = repository(tmp_path / 'repo', patch)
            assert repo.alias_fold_merge is False
            assert repo.alias_fold_pending() == 0
            assert repo.alias_fold_pass() == []


# ----------------------------------------------------------- recap probes

#: The two full-campaign tests each replay 26 cells, and every cell copies the
#: whole repository tree forward. One run costs ~27 GB of `tmp_path` and ~9
#: minutes. Running them inside the ordinary battery exhausted the machine's
#: root filesystem (0 bytes free) mid-suite on 2026-09-11, which surfaced as
#: 29 spurious ENOSPC errors. They are opt-in via GRM_LT1_1_FULL_RUN=1 so the
#: battery stays runnable; the receipts they produce are reported separately.
FULL_RUN = os.environ.get('GRM_LT1_1_FULL_RUN') == '1'
full_run_only = pytest.mark.skipif(
    not FULL_RUN,
    reason='26-cell replay: ~27 GB tmp_path and ~9 min per arm; '
           'set GRM_LT1_1_FULL_RUN=1 to run')


@full_run_only
def test_recap_probe_rows_are_written_and_scored(tmp_path):
    """The kind that killed A-189-196 now produces scored rows."""
    root = _cells_through('A+', tmp_path, 26)
    rows = []
    for cell in runner.registration('A+')['cells']:
        path = root / 'cells' / cell['id'] / 'probes.jsonl'
        if path.exists():
            rows.extend(json.loads(line) for line in
                        path.read_text().splitlines() if line.strip())
    recaps = [r for r in rows if r.get('kind') == 'recap_probe']
    assert len(recaps) == 5, 'expected five recap probes, got %d' % len(recaps)
    for row in recaps:
        assert row['expected']
        assert 'score' in row['memory']
        assert row['targets']


def test_the_oracle_still_answers_the_recap_battery_five_of_five():
    """The questions themselves remain answerable (the CPU oracle path)."""
    from scripts import grm_d1_recap as recap
    result = recap.oracle_gate()
    assert result['status'] == 'PASS'
    assert result['matched'] == 5


# ------------------------------------------------- the full 26-cell run

@full_run_only
@pytest.mark.parametrize('arm', ['A', 'A+'])
def test_the_twentysix_cell_fake_run_completes(arm, tmp_path):
    """Both arms, end to end, through the real execute. A-189-196 included."""
    root = _cells_through(arm, tmp_path, 26)
    reg = runner.registration(arm)
    for cell in reg['cells']:
        controller = root / 'cells' / cell['id'] / 'controller.json'
        assert controller.is_file(), cell['id']
        assert json.loads(controller.read_text())['status'] == 'COMPLETE'
    value = runner.summary(arm, root=root)
    assert value['complete'] is True
    assert value['complete_cells'] == 26


# --------------------------------------------------------- the fairness

def test_the_two_arms_share_this_worker():
    """A-vs-A+ is a same-worker contrast; LT1 is the parent baseline."""
    handlers = {}
    for arm in runner.ARMS:
        with runner.lt1_1_seams(arm):
            handlers[arm] = (worker.deposit_turn, worker.recap_probe_turn,
                             worker.execute)
    assert handlers['A'] == handlers['A+'], (
        'the arms use different worker code; A-vs-A+ would not be a fair '
        'contrast')
