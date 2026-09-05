import pytest

from scripts.grm_wc1_sweep import BASELINE, WIDTHS, assemble, evaluate_predictions, plan, regressions


def test_plan_runs_reproduction_before_all_other_widths():
    rows = plan()
    assert [r['battery'] for r in rows if r['width'] == BASELINE] == ['sup', 'census', 'longhorizon']
    assert {r['width'] for r in rows[:3]} == {BASELINE}
    assert len(rows) == len(WIDTHS) * 3
    assert all(r['gate'] == 'G3' for r in rows[3:])


@pytest.mark.parametrize('widths', [(64, 96), (64, 96, 128, 192, 384), (64, 96, 128, 192, 256, 96)])
def test_plan_rejects_unregistered_or_duplicate_widths(widths):
    with pytest.raises(ValueError):
        plan(widths)


def test_regressions_are_paired_by_probe_id_not_count_or_position():
    base = [{'probe_id': 'a', 'correct': True}, {'probe_id': 'b', 'correct': False}]
    current = [{'probe_id': 'b', 'correct': True}, {'probe_id': 'a', 'correct': False}]
    assert regressions(current, base) == ['a']
    assert regressions(base, base) == []


def test_incomplete_or_duplicate_probes_cannot_be_green():
    base = [{'probe_id': 'a', 'correct': True}]
    with pytest.raises(ValueError):
        regressions([], base)
    with pytest.raises(ValueError):
        regressions(base + base, base)


def test_table_distinguishes_not_run_and_loss_and_weights_turns():
    rows = [{'width': BASELINE, 'battery': 'sup',
             'probes': [{'probe_id': 'a', 'correct': True}],
             'turns': [{'resident': 40, 'wall_ms': 100}, {'resident': 80, 'wall_ms': 300}],
             'split_parents': 2, 'split_children': 5},
            {'width': 64, 'battery': 'sup',
             'probes': [{'probe_id': 'a', 'correct': False}], 'turns': []}]
    table = {(r['width'], r['battery']): r for r in assemble(rows)}
    assert table[(BASELINE, 'sup')]['mean_wall_ms_per_turn'] == 200
    assert table[(BASELINE, 'sup')]['mean_resident_seats_per_turn'] == 60
    assert table[(BASELINE, 'sup')]['split_children'] == 5
    assert table[(64, 'sup')]['regressions_vs_96'] == ['a']
    assert table[(128, 'sup')]['correct'] is None
    assert table[(128, 'sup')]['status'] == 'NOT_RUN'


def test_table_rejects_duplicate_cells():
    row = {'width': BASELINE, 'battery': 'sup', 'probes': [], 'turns': []}
    with pytest.raises(ValueError):
        assemble([row, row])


def test_unrun_predictions_cannot_be_hits_or_misses():
    assert {r['outcome'] for r in evaluate_predictions(assemble([]))} == {'NOT_TESTED'}


def test_prediction_losses_are_paired_and_latency_needs_both_clauses():
    results = []
    for cell in plan():
        probes = [{'probe_id': str(i), 'correct': cell['width'] != 64 or i >= 2} for i in range(4)]
        results.append({**cell, 'probes': probes,
                        'turns': [{'resident': 1, 'wall_ms': cell['width']}]})
    predictions = evaluate_predictions(assemble(results))
    assert [r['outcome'] for r in predictions] == ['HIT', 'HIT', 'MISS', 'HIT']


def test_guard_resolves_dir_fd_relative_removals(tmp_path):
    import os
    from tests.grm_wc1_guard import _writable
    fd = os.open(tmp_path, os.O_RDONLY)
    try:
        _writable('manifest.json', fd)
        with pytest.raises(PermissionError):
            _writable('/mnt/ForgeRealm/GraftRepository/core/graft_arena.py')
    finally:
        os.close(fd)
