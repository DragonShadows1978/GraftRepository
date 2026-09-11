"""CPU author baseline for lead cap amendment, never a GPU/blind gate.

Prior art: local C4/A4/A5 adversarial manifest and reserve-boundary tests
(house, 2026). Reuses temp artifacts and fake time; adds cap overlay attacks.
"""
import copy

import pytest

from scripts import grm_c4_campaign as c
from scripts import grm_c4_resume_a2 as a2
from scripts import grm_c4_remaining_a3 as a3
from scripts import grm_c4_cap_a4 as cap


@pytest.mark.campaign_receipt(registration='artifacts/grm_c4/registration.json + amendment_a1..a3.json (pin absolute /mnt/ForgeRealm/wt/grm-c4/ paths)')
@pytest.mark.parametrize('runner', [a2, a3], ids=['A2', 'A3'])
def test_runtime_cap_and_unchanged_scope(runner):
    reg = runner.binding()
    amendment = cap.binding()
    assert reg['budget']['gpu_seconds'] == 5400
    assert reg['budget_amendment'] == c.record(cap.AMENDMENT)
    assert {k: v for k, v in reg['budget'].items() if k != 'gpu_seconds'} == {
        k: v for k, v in amendment['previous_budget'].items() if k != 'gpu_seconds'}
    assert reg['amendment'] == c.record(runner.AMENDMENT)
    assert reg['sources'] == c.read(runner.AMENDMENT)['sources']
    baseline = c.binding()
    for key in ('units_per_new_cell', 'fixtures', 'counts', 'geometry',
                'prediction', 'acceptance', 'rejection', 'execution_order'):
        assert reg[key] == baseline[key]


@pytest.mark.parametrize('runner', [a2, a3], ids=['A2', 'A3'])
@pytest.mark.parametrize('attack', [
    'stale_sha', 'cap', 'float_cap', 'old_cap', 'worker_rail', 'extra_field',
    'order', 'previous_amendment', 'registration', 'previous_budget',
    'missing_replacement', 'forged_before', 'forged_after', 'extra_replacement',
    'missing_source', 'forged_source', 'schema', 'sequence', 'stale_a5',
])
def test_forged_or_stale_cap_refused(tmp_path, monkeypatch, runner, attack):
    row = c.read(cap.AMENDMENT)
    if attack == 'stale_sha': row['immutable'] = False
    elif attack == 'cap': row['overrides']['budget']['gpu_seconds'] = 5401
    elif attack == 'float_cap': row['overrides']['budget']['gpu_seconds'] = 5400.0
    elif attack == 'old_cap': row['overrides']['budget']['gpu_seconds'] = 4800
    elif attack == 'worker_rail': row['overrides']['budget']['worker_seconds'] = 286
    elif attack == 'extra_field': row['scheduled_units'] = []
    elif attack in ('order', 'previous_amendment', 'registration'):
        row[attack]['sha256'] = 'forged'
    elif attack == 'previous_budget': row['previous_budget']['gpu_seconds'] = 5399
    elif attack == 'missing_replacement': row['source_replacements'].pop()
    elif attack == 'forged_before': row['source_replacements'][0]['before']['sha256'] = 'forged'
    elif attack == 'forged_after': row['source_replacements'][0]['after']['sha256'] = 'forged'
    elif attack == 'extra_replacement': row['source_replacements'].append(copy.deepcopy(row['source_replacements'][0]))
    elif attack == 'missing_source': row['added_sources'].pop()
    elif attack == 'forged_source': row['added_sources'][0]['sha256'] = 'forged'
    elif attack == 'schema': row['schema'] = 'forged'
    elif attack == 'sequence': row['artifact_sequence'] = 5
    elif attack == 'stale_a5': row = c.read(cap.PREVIOUS)
    path = tmp_path/'amendment_a6.json'
    c.write_once(path, row)
    # Attacker can recompute the sidecar. Semantic/order/chain validation must
    # still refuse; this is stronger than a checksum-only corruption test.
    path.with_suffix('.sha256').write_text('stale' if attack == 'stale_sha' else c.record(path)['sha256'])
    monkeypatch.setattr(cap, 'AMENDMENT', path)
    with pytest.raises(AssertionError):
        runner.binding()


@pytest.mark.parametrize('runner', [a2, a3], ids=['A2', 'A3'])
@pytest.mark.parametrize('target', ['order', 'predecessor', 'replacement', 'helper', 'commands'])
def test_current_source_or_anchor_drift_refused(monkeypatch, runner, target):
    path = {'order': cap.ORDER, 'predecessor': cap.PREVIOUS,
            'replacement': c.ROOT/cap.REPLACEMENTS[0],
            'helper': c.ROOT/cap.ADDITIONS[0],
            'commands': c.OUT/'lead_commands_a2.txt'}[target]
    original = c.record
    def drift(p):
        row = original(p)
        if row['path'] == str(path):
            row['sha256'] = 'drift'
        return row
    monkeypatch.setattr(c, 'record', drift)
    with pytest.raises(AssertionError):
        runner.binding()


@pytest.mark.campaign_receipt(registration='artifacts/grm_c4/registration.json + amendment_a1..a3.json (pin absolute /mnt/ForgeRealm/wt/grm-c4/ paths)')
@pytest.mark.parametrize('runner', [a2, a3], ids=['A2', 'A3'])
def test_missing_cap_never_falls_back(tmp_path, monkeypatch, runner):
    monkeypatch.setattr(cap, 'AMENDMENT', tmp_path/'absent.json')
    with pytest.raises(FileNotFoundError):
        runner.binding()


@pytest.mark.campaign_receipt(registration='artifacts/grm_c4/registration.json + amendment_a1..a3.json (pin absolute /mnt/ForgeRealm/wt/grm-c4/ paths)')
@pytest.mark.parametrize('runner', [a2, a3], ids=['A2', 'A3'])
@pytest.mark.parametrize('used,allowed', [(4516, True), (4924.074634324294, True),
                                         (5115, True), (5115.001, False)])
def test_reserve_boundary_and_original_red_charged(tmp_path, monkeypatch, runner, used, allowed):
    reg = runner.binding()
    old = reg['a4_context'] if runner is a3 else reg
    monkeypatch.setattr(c, 'OUT', tmp_path/'original')
    monkeypatch.setattr(a2, 'OUT', tmp_path/'a2')
    monkeypatch.setattr(a3, 'OUT', tmp_path/'a3')
    monkeypatch.setattr(a2.time, 'time', lambda: 1000)
    monkeypatch.setattr(a2, 'eligible_suffix', lambda *args: ['c4-lh-06'])
    for amended, seconds, status in ((False, 73.72084314282984, 'RED'),
                                     (True, used - 73.72084314282984, 'PASS')):
        c.write_once(a2.receipt_path('c64_w96', 'longhorizon', 'c4-lh-06', amended=amended), {
            'cell': 'c64_w96', 'battery': 'longhorizon', 'spec': 'c4-lh-06',
            'registration': c.record(c.REG), 'status': status, 'error': a2.ROOT_ERROR,
            'amendment': old['amendment' if amended else 'legacy_amendment'],
            'sources': old['sources' if amended else 'legacy_sources'],
            'gpu_seconds': seconds, 'finished_unix': 0})
    if allowed:
        assert runner.accounting(reg) == pytest.approx(used)
    else:
        with pytest.raises(AssertionError, match='GPU budget non-fit'):
            runner.accounting(reg)


@pytest.mark.campaign_receipt(registration='artifacts/grm_c4/registration.json + amendment_a1..a3.json (pin absolute /mnt/ForgeRealm/wt/grm-c4/ paths)')
def test_commands_receipts_and_prior_amendments_byte_unchanged():
    rows = c.read(cap.OUT/'before.json')['files']
    assert len(rows) >= 29
    assert all(c.record(r['path']) == r for r in rows)
    commands = [r for r in rows if r['path'].endswith(('lead_commands_a2.txt', 'lead_commands_a3.txt'))]
    assert len(commands) == 2
    assert c.record(c.OUT/'lead_commands_a2.txt')['sha256'] == 'fbaafca244cd0fdc73d28eb1effe5dfd67222980af9ba874e67d0808481face4'
    assert c.record(c.OUT/'lead_commands_a3.txt')['sha256'] == '01266f9831cffb5676313667a9bd841d55dcd0ad5dc46424910b39a2f6f39b25'
