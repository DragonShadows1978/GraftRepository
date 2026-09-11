"""Prior art: EB1 CPU fake (GRM, 2026), hand-computed accounting examples,
exception and boundary testing (established practice; no new algorithm).
Author baseline only; independent blind verification belongs to the lead.
"""
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import pytest
from scripts import grm_c8_profiler as p
from scripts import grm_c8_cells as c


def spec(fn, stage, phases=None):
    return {(fn.__code__.co_filename, fn.__code__.co_firstlineno, fn.__name__): (stage, phases)}


class ForbiddenMeter:
    gpu = False
    def begin(self):
        raise AssertionError('OFF touched instrumentation')
    end = begin


def test_default_off_exact_object_no_instrumentation():
    value = object(); before = dict(os.environ)
    profiler = p.TurnProfiler(meter=ForbiddenMeter())
    assert profiler.run(lambda: value) is value
    assert profiler.receipt is None and dict(os.environ) == before


def test_fake_complete_pipeline_utf8_identity_and_state():
    # Known staged fake; unlike comparing two empty answers this exercises
    # routing, nested harvest, generation, deposit and fold state transitions.
    def lexical(state):
        state.append('lex')
    def route(state):
        lexical(state); state.append('route')
    def admission(state):
        state.append('admission')
    def fetch(state):
        state.append('fetch')
    def seat(state):
        state.append('seat')
    def prefill(state):
        state.append('prefill')
    def decode(state):
        state.append('decode'); return 'Auric-4-Alpha — 確認\n'
    def deposit(state):
        prefill(state); state.append('deposit')
    def fold(state):
        decode(state); state.append('fold')
    def turn(state):
        route(state); admission(state); fetch(state); seat(state); prefill(state)
        answer = decode(state); deposit(state); fold(state); return answer
    mapping = {}
    for fn, label in ((lexical, 'lexical_scan'), (route, 'route'), (admission, 'admission'),
        (fetch, 'cold_fetch'), (seat, 'seat_mount'), (prefill, 'prefill'),
        (decode, 'decode'), (deposit, 'deposit_harvest'), (fold, 'fold')):
        mapping.update(spec(fn, label))
    off_state, on_state = [], []
    off = p.TurnProfiler().run(turn, off_state)
    profiler = p.TurnProfiler(enabled=True, mapping=mapping)
    on = profiler.run(turn, on_state)
    assert off.encode('utf-8') == on.encode('utf-8') and off_state == on_state
    row = profiler.receipt
    assert row['status'] == 'COMPLETE'
    assert all(row['stages'][s]['wall_ms'] > 0 for s in p.STAGES)
    assert sum(x['wall_ms'] for x in row['stages'].values()) == pytest.approx(row['turn_wall_ms'])
    assert all(x['gpu_ms'] is None for x in row['stages'].values())


def test_real_eb1_step_cpu_fake_byte_identity():
    # Imported test helper drives real ArenaCache.step; only model-dependent
    # seams are faked. This complements, rather than replaces, complete fake.
    import importlib.util
    module_spec = importlib.util.spec_from_file_location('c8_eb1_fake',
        p.ROOT / 'tests/test_grm_eb1_ephemeral_frame.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    off = module._drive(module._step_probe(ephemeral=True))
    profiler = p.TurnProfiler(enabled=True)
    on = profiler.run(module._drive, module._step_probe(ephemeral=True))
    assert json.dumps(off, sort_keys=True).encode() == json.dumps(on, sort_keys=True).encode()
    assert len(on) == 4 and all(x['answer'] for x in on)


def test_nested_work_not_double_counted_or_mislabeled():
    def decode():
        time.sleep(.002)
    def deposit():
        decode()
    def fold():
        deposit()
    mapping = {**spec(decode, 'decode'), **spec(deposit, 'deposit_harvest'), **spec(fold, 'fold')}
    profiler = p.TurnProfiler(enabled=True, mapping=mapping)
    profiler.run(fold)
    assert profiler.receipt['stages']['decode']['wall_ms'] == 0
    assert profiler.receipt['stages']['deposit_harvest']['wall_ms'] == 0
    assert profiler.receipt['stages']['fold']['wall_ms'] >= 2


def test_exception_preserved_and_trace_restored():
    error = ValueError('sentinel')
    def route():
        raise error
    profiler = p.TurnProfiler(enabled=True, mapping=spec(route, 'route'))
    with pytest.raises(ValueError) as caught:
        profiler.run(route)
    assert caught.value is error and sys.gettrace() is None
    assert profiler.receipt['status'] == 'RED_EXCEPTION'


def test_existing_trace_refused_without_replacing():
    def trace(*args):
        return trace
    sys.settrace(trace)
    try:
        with pytest.raises(RuntimeError, match='existing trace'):
            p.TurnProfiler(enabled=True).run(lambda: None)
        assert sys.gettrace() is trace
    finally:
        sys.settrace(None)


def test_attempt_phase_map_matches_real_source():
    mapping = p.source_map()
    key = next(k for k in mapping if k[2] == '_attempt')
    label, (prefill, decode) = mapping[key]
    lines = Path(key[0]).read_text().splitlines()
    assert label == 'seat_mount'
    assert lines[prefill - 1].strip().startswith('prompt_ids =')
    assert lines[decode - 1].strip() == 'out = [int(row.argmax())]'
    assert prefill < decode
    assert set(p.STAGES) - {'other'} <= {s for s, _ in mapping.values()}


def test_line_phases_cover_decode_loop_and_single_token_prompt():
    def attempt():
        x = 'mount'
        ids = [7]
        x = 'prefill'
        out = [1]
        time.sleep(.002)
        return out
    first = attempt.__code__.co_firstlineno
    mapping = spec(attempt, 'seat_mount', (first + 2, first + 4))
    profiler = p.TurnProfiler(enabled=True, mapping=mapping)
    assert profiler.run(attempt) == [1]
    assert profiler.receipt['stages']['decode']['wall_ms'] >= 2
    assert profiler.receipt['stages']['prefill']['wall_ms'] > 0


def test_metrics_use_sum_gpu_max_memory():
    class Meter:
        gpu = True
        def begin(self):
            pass
        def end(self):
            return {'gpu_ms': 2., 'peak_pool_used_bytes': 100,
                    'device_used_boundary_peak_bytes': 300}
    def decode():
        return 1
    profiler = p.TurnProfiler(enabled=True, meter=Meter(), mapping=spec(decode, 'decode'))
    profiler.run(lambda: (decode(), decode()))
    row = profiler.receipt['stages']['decode']
    assert row['gpu_ms'] == 4 and row['peak_pool_used_bytes'] == 100
    assert profiler.receipt['full_device_peak_bytes'] is None


def rows_with_fraction(fraction, count=30):
    rows = []
    for _ in range(count):
        stages = {s: dict(wall_ms=0., gpu_ms=0., peak_pool_used_bytes=1,
                         device_used_boundary_peak_bytes=2) for s in p.STAGES}
        stages['decode']['wall_ms'] = 100 * fraction
        stages['other']['wall_ms'] = 100 * (1 - fraction)
        rows.append({'status': 'COMPLETE', 'turn_wall_ms': 100., 'stages': stages})
    return rows


@pytest.mark.parametrize('fraction,expected', [(.499, 'session'), (.5, 'APA'), (.501, 'APA')])
def test_registered_decision_boundary(fraction, expected):
    report = p.summarize(rows_with_fraction(fraction))
    assert report['decision'].startswith(expected)
    assert report['decode_fraction'] == pytest.approx(fraction)
    assert report['nondecode_fraction'] == pytest.approx(1 - fraction)
    assert report['apa_sp_counterfactual_turn_saving_fraction'] < 0


def test_quantiles_hand_computed():
    assert p.stats(list(range(1, 31))) == {'mean': 15.5, 'p50': 15, 'p95': 29}
    assert p.stats([]) == {'mean': None, 'p50': None, 'p95': None}


def test_ratio_of_sums_not_mean_ratios():
    rows = rows_with_fraction(.1)
    rows[0]['turn_wall_ms'] = 1000.
    rows[0]['stages']['decode']['wall_ms'] = 910.
    result = p.summarize(rows)
    assert result['decode_fraction'] == pytest.approx((29 * 10 + 910) / 3900)


@pytest.mark.parametrize('count', [0, 1, 29])
def test_missing_turns_never_pass(count):
    assert p.summarize(rows_with_fraction(.1, count))['decision'] is None


@pytest.mark.parametrize('defect', ['negative', 'nan', 'conservation', 'failed'])
def test_corrupt_timing_rejected(defect):
    rows = rows_with_fraction(.1)
    if defect == 'failed':
        rows[0]['status'] = 'RED_EXCEPTION'
        assert p.summarize(rows)['decision'] is None
    else:
        rows[0]['stages']['decode']['wall_ms'] = {'negative': -1, 'nan': float('nan'), 'conservation': 2}[defect]
        with pytest.raises(ValueError):
            p.summarize(rows)


@pytest.mark.campaign_receipt(registration='artifacts/grm_c8/registration.json (C8 core pins)')
def test_registration_complete_counts_dependencies_flags_budget():
    r = c.registration(); seen = set(); counts = {}
    for cell in r['cells']:
        assert set(cell['depends']) <= seen
        seen.add(cell['id'])
        counts[cell['battery']] = counts.get(cell['battery'], 0) + cell['turns']
        assert cell['estimate_seconds'] <= 285
        assert cell['demand_on'] == (cell['battery'] == 'demand')
    assert counts['sup'] == 36 and counts['census'] == 34 and counts['longhistory'] == 104
    assert counts['demand'] == 1 and counts['identity'] == 1
    assert r['budget_seconds'] == 1800 < r['estimated_seconds']
    with pytest.raises(ValueError, match='NON_FIT_BUDGET'):
        c.fit(r)


@pytest.mark.campaign_receipt(registration='artifacts/grm_c8/registration.json (C8 core pins)')
def test_nonfit_refused_before_any_lease_or_reservation(tmp_path, monkeypatch):
    r = c.registration()
    monkeypatch.setattr(c, 'OUT', tmp_path / 'never_created')
    with pytest.raises(ValueError, match='NON_FIT_BUDGET'):
        c.run_cell(r, r['cells'][0])
    assert not c.OUT.exists()


def test_worker_requires_leased_parent_before_imports(monkeypatch):
    monkeypatch.delenv('GRM_C8_LEASE_PARENT', raising=False)
    with pytest.raises(RuntimeError, match='foreground leased parent'):
        c.worker({})


@pytest.mark.campaign_receipt(registration='artifacts/grm_c8/registration.json (C8 core pins)')
def test_forged_registration_rejected(tmp_path, monkeypatch):
    r = c.registration(); r['budget_seconds'] = 99999
    target = tmp_path / 'forged.json'; target.write_text(json.dumps(r))
    monkeypatch.setattr(c, 'REG', target)
    with pytest.raises(ValueError, match='code anchor'):
        c.registration()


def test_side_b_does_not_relabel_context_or_claim_saving():
    value = c.side_b()
    assert value['receipt_context_tokens'] == 2048
    assert value['order_claimed_context_tokens'] == 12288
    assert value['saving_at_12288_or_width96'] is None
    assert value['single_pass_saving_ms_tok'] == pytest.approx(-1.6)
    assert value['two_pass_saving_ms_tok'] == pytest.approx(-1.4)


@pytest.mark.campaign_receipt(registration='artifacts/grm_c8/registration.json (C8 core pins)')
def test_missing_gpu_summary_not_a_decision():
    report = c.summary()
    assert report['status'] == 'BLOCKED_NOT_MEASURED'
    assert report['allocation_decision'] is None
    assert all(b['decision'] is None for b in report['batteries'].values())
