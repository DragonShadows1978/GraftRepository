"""C4 CPU gates. Prior art: house P2C stub repository (2026), reused by import;
C4 assertions target crossed controls, OFF parity, and measured token seats.
Author-run unit evidence only, not a blind red-team or GPU acceptance gate.
"""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from core.graft_arena import ArenaCache
from core.graft_repository import GraftRepository
from scripts import grm_c4_campaign as campaign
from scripts.grm_c4_adapter import (
    CHUNK_ENV, chunk_setting, deposit_chunking, fixed_geometry, residency,
)

spec = importlib.util.spec_from_file_location('c4_existing_p2c_tests',
    Path(__file__).with_name('test_grm_lsr_p2c_split_descent.py'))
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


def encoded(nodes):
    return json.dumps(nodes, sort_keys=True, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else sorted(v)).encode()


@pytest.mark.parametrize('chunk,width', [(64,64), (64,96), (96,64), (96,96)])
def test_crossed_deposit_parent_preserved(chunk, width):
    text = ' '.join(f'word{i}' for i in range(150))
    repo = legacy._guard_repo(width=width, text=text, ntok=150)
    events = []
    with deposit_chunking(chunk, events):
        result = repo._guard_deposit_width(0)
    sizes = [repo.arena.grafts[i]['ntok'] for i in result['children']]
    assert result['budget'] == chunk
    assert sizes[0] == chunk and max(sizes) <= chunk and sum(sizes) == 150
    assert repo.arena.grafts[0]['text'] == text
    assert events[0]['parent_preserved']
    if chunk == 96 and width == 64:
        assert max(sizes) > width  # catches clamping the intended intervention


def test_explicit_fit_budget_survives_chunk_control():
    repo = legacy._guard_repo(width=64, text=' '.join(['word']*150), ntok=150)
    with deposit_chunking(96):
        result = repo._guard_deposit_width(0, budget=32)
    assert result['budget'] == 32
    assert max(repo.arena.grafts[i]['ntok'] for i in result['children']) <= 32


@pytest.mark.parametrize('token', [None, '0', 'off', ''])
def test_off_existing_manifest_byte_parity(monkeypatch, token):
    if token is None:
        monkeypatch.delenv(CHUNK_ENV, raising=False)
    else:
        monkeypatch.setenv(CHUNK_ENV, token)
    raw = campaign.MANIFEST.read_bytes()
    manifest = json.loads(raw)
    original = GraftRepository._guard_deposit_width
    checked = 0
    for node in manifest['nodes']:
        a = legacy._guard_repo(width=96, text=node['text'], ntok=node['ntok'],
                               kind=node['kind'], metadata=copy.deepcopy(node.get('metadata', {})))
        b = copy.deepcopy(a)
        expected = original(a, 0)
        with deposit_chunking():
            assert GraftRepository._guard_deposit_width is original
            actual = b._guard_deposit_width(0)
        assert encoded(actual) == encoded(expected)
        assert encoded(a.arena.grafts) == encoded(b.arena.grafts)
        checked += 1
    assert checked == len(manifest['nodes']) and checked > 0
    assert campaign.MANIFEST.read_bytes() == raw


def test_bad_flag_refuses_and_context_restores_on_error():
    with pytest.raises(ValueError):
        chunk_setting({CHUNK_ENV: '65'})
    before = GraftRepository._guard_deposit_width
    with pytest.raises(RuntimeError), deposit_chunking(64):
        raise RuntimeError('test cleanup')
    assert GraftRepository._guard_deposit_width is before


@pytest.mark.parametrize('width', [64,96])
def test_numeric_geometry_fixed_and_rs3_restored(width):
    events = []
    layer = SimpleNamespace(self_attn=SimpleNamespace(live_shift=None))
    def init(arena, *args, **kw):
        arena.width = kw['arena_width']
        arena.n_sink = 5
        arena.live_shift = 5 + kw['arena_width']
        arena.live_turns = 2
        arena.ephemeral = kw['ephemeral']
        arena.m = SimpleNamespace(layers=[layer])
        arena.grafts = [{'ntok': 30}, {'ntok': 20}]
    with patch.object(ArenaCache, '__init__', init), fixed_geometry(width, events):
        arena = ArenaCache()
        assert arena.width == width and arena.live_shift == 101
        with arena._capture_geometry() as receipt:
            assert receipt['capture_shift_observed'] == 101
            assert layer.self_attn.live_shift == 101
        assert layer.self_attn.live_shift is None
        order, info = arena._rs3_seat_plan([0,1])
        assert order == [1,0] and info['plan_head_last_position'] == 100
        assert info['mount_pos0'] == 51
    assert any(e['event'] == 'capture' for e in events)


def test_residency_uses_tokens_and_rejects_wrong_counter():
    arena = SimpleNamespace(cur_mounts=[0,1], grafts=[{'ntok': 31},{'ntok': 17}],
        cur_mount_n=48, width=64, live_shift=101, n_sink=5,
        PAYLOAD=(('k',1),), caches=[(np.zeros((1,60,1)),)])
    row = residency(arena)
    assert row['mounted_token_seats'] == 48 and row['live_token_seats'] == 7
    assert row['physical_cache_token_seats'] == 60
    arena.cur_mount_n = 2
    with pytest.raises(AssertionError):
        residency(arena)


def test_registered_probe_sets_and_segment_coverage():
    r = campaign.binding()
    assert sum(len(v) for v in r['fixtures']['sup'].values()) == 9
    assert len(r['fixtures']['census']) == 10
    assert len(r['fixtures']['longhorizon']['probes']) == 14
    assert r['fixtures']['longhorizon']['total_turns'] == 104
    assert list(campaign.LH_STOPS.values()) == list(range(8,105,8))
    assert max(u['estimate_seconds'] for u in campaign.units()) < 285
    assert r['budget']['new_cells_estimate_seconds'] == 4800
    assert r['budget']['full_fixed_geometry_estimate_seconds'] == 4800
    assert r['budget']['gpu_seconds'] == 4800


def test_registration_and_source_drift_refused(tmp_path, monkeypatch):
    f = tmp_path/'source.py'
    f.write_text('before')
    r = tmp_path/'registration.json'
    campaign.write_once(r, {'sources': [campaign.record(f)]})
    (tmp_path/'registration.sha256').write_text(campaign.record(r)['sha256']+'  registration.json\n')
    monkeypatch.setattr(campaign, 'OUT', tmp_path)
    monkeypatch.setattr(campaign, 'REG', r)
    campaign.binding()
    f.write_text('after')
    with pytest.raises(AssertionError, match='source drift'):
        campaign.binding()
    with pytest.raises(FileExistsError):
        campaign.write_once(r, {})


@pytest.mark.parametrize('cell', ['c64_w96', 'c96_w64', 'c64_w64'])
def test_missing_results_cannot_score(tmp_path, monkeypatch, cell):
    r = campaign.binding()
    monkeypatch.setattr(campaign, 'OUT', tmp_path)
    monkeypatch.setattr(campaign, 'binding', lambda: r)
    with pytest.raises(FileNotFoundError):
        campaign.score(cell)


def test_imported_loader_paths_pinned_without_gpu(tmp_path):
    from scripts import grm_det1_2_gpu as loader
    original_native = loader.NATIVE_LIB
    with campaign.harness(64, 96, tmp_path, []):
        assert loader.NATIVE_LIB == campaign.SHARED / 'cpp/build/libgrm_runtime.so'
        assert loader.MODEL_DIR == Path(campaign.read(campaign.FRAME)['model']['path'])
    assert loader.NATIVE_LIB == original_native


@pytest.mark.parametrize('cell', ['c64_w96', 'c64_w64'])
def test_timeout_receipted_and_same_worker_cannot_retry(tmp_path, monkeypatch, cell):
    from contextlib import contextmanager
    from scripts import grm_cmc1_gpu_arms as leases
    reg = campaign.binding()
    monkeypatch.setattr(campaign, 'binding', lambda: reg)
    monkeypatch.setattr(campaign, 'OUT', tmp_path)
    @contextmanager
    def lease(seconds, wait):
        assert seconds == 285 and wait == 0
        yield
    @contextmanager
    def harness(*args):
        def serve(*args, **kwargs):
            raise TimeoutError('synthetic rail')
        yield SimpleNamespace(serve_fixture=serve), None, None
    monkeypatch.setattr(leases, 'gpu_lease', lease)
    monkeypatch.setattr(campaign, 'harness', harness)
    with pytest.raises(TimeoutError, match='synthetic rail'):
        campaign.worker(cell, 'sup', campaign.SUP[0])
    r = campaign.read(tmp_path/'runs'/cell/f'sup_{campaign.SUP[0]}.json')
    assert r['status'] == 'RED' and r['error'] == 'TimeoutError: synthetic rail'
    with pytest.raises(AssertionError, match='No retries'):
        campaign.worker(cell, 'sup', campaign.SUP[0])


def test_imported_lh_segments_copy_saved_state(tmp_path, monkeypatch):
    from scripts import grm_e2e_session as e2e
    from scripts import grm_det1_e2e as det
    stops = []
    def main(argv):
        dest = Path(argv[argv.index('--session-dir')+1])
        dest.mkdir(parents=True, exist_ok=True)
        index = len(stops)
        if index:
            assert '--resume' in argv
            assert (dest/'sentinel.txt').read_text() == str(stops[-1])
        else:
            assert '--resume' not in argv
        stop = list(campaign.LH_STOPS.values())[index]
        stops.append(stop)
        (dest/'sentinel.txt').write_text(str(stop))
        campaign.write_once(dest/'probe_scorecard.json', {'probes': []}) if not (dest/'probe_scorecard.json').exists() else None
        raise det._DETStageStop()
    monkeypatch.setattr(e2e, 'main', main)
    monkeypatch.setattr(e2e, 'run_turn', lambda *a, **kw: None)
    with campaign.harness(64, 96, tmp_path, []) as (_, _, lh):
        for spec in campaign.LH_STOPS:
            result = lh.run_shard(spec, run_dir=tmp_path/'sessions')
            assert result['stop_after_turns'] == stops[-1]
    assert stops == list(range(8,105,8))


def test_abandoned_worker_blocks_other_cell_without_gpu(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from scripts import grm_cmc1_gpu_arms as leases
    reg = campaign.binding()
    monkeypatch.setattr(campaign, 'binding', lambda: reg)
    monkeypatch.setattr(campaign, 'OUT', tmp_path)
    campaign.write_once(tmp_path/'runs/c96_w64/sup_abandoned.attempt.json', {'started': True})
    @contextmanager
    def lease(*args):
        yield
    monkeypatch.setattr(leases, 'gpu_lease', lease)
    with pytest.raises(AssertionError, match='Unfinished worker claim'):
        campaign.worker('c64_w96', 'sup', campaign.SUP[0])


def test_lead_amendment_preserves_prediction_geometry_and_rails():
    # Prior art: C4/RS3 registration invariants (house, 2026). Check the
    # lead's exact permitted delta against the immutable base, not a new bar.
    base = campaign.read(campaign.REG)
    reg = campaign.binding()
    assert campaign.executable_cells(reg) == ('c64_w96', 'c96_w64', 'c64_w64')
    for key in ('geometry', 'fixtures', 'counts', 'acceptance', 'prediction', 'rejection', 'units_per_new_cell'):
        assert reg[key] == base[key]
    for key in ('worker_seconds', 'outer_seconds', 'cooldown_seconds', 'stop'):
        assert reg['budget'][key] == base['budget'][key]
    cell = next(c for c in reg['cells'] if c['id'] == 'c64_w64')
    assert (cell['chunk'], cell['width'], cell['estimate_seconds']) == (64, 64, 1600)
    assert sum(c['estimate_seconds'] for c in reg['cells'] if c['status'] == 'NEW') == 4800
    assert reg['diagonal_comparison']['historical_scores'] == dict(sup=8, census=10, longhorizon=14)
    pin = reg['diagonal_ruling']
    assert pin['decision'] == 'CITE_WC1'
    assert (pin['n_sink'], pin['capture_shift'], pin['live_shift'], pin['plan_head_last_position']) == (19, 115, 115, 114)
    assert len(pin['worker_pins']) == 13 and all(r['match'] for r in pin['historical_fingerprints'])


def test_amendment_sha_and_order_drift_refused(tmp_path, monkeypatch):
    # Prior art: C4 source-drift refusal (house, 2026), extended to the order
    # and amendment itself. All corruptions are temporary CPU fixtures.
    order = tmp_path/'order.md'
    order.write_text('original lead decision')
    reg_path = tmp_path/'registration.json'
    campaign.write_once(reg_path, {'sources': []})
    (tmp_path/'registration.sha256').write_text(campaign.record(reg_path)['sha256'])
    amend = tmp_path/'amendment_a1.json'
    campaign.write_once(amend, {'registration': campaign.record(reg_path),
        'sources': [], 'order': campaign.record(order), 'overrides': {'budget': {'gpu_seconds': 4800}}})
    amend.with_suffix('.sha256').write_text(campaign.record(amend)['sha256'])
    monkeypatch.setattr(campaign, 'OUT', tmp_path)
    monkeypatch.setattr(campaign, 'REG', reg_path)
    assert campaign.binding()['budget']['gpu_seconds'] == 4800
    order.write_text('changed lead decision')
    with pytest.raises(AssertionError, match='amendment order drift'):
        campaign.binding()
    order.write_text('original lead decision')
    with amend.open('a') as f:
        f.write(' ')
    with pytest.raises(AssertionError, match='amendment SHA mismatch'):
        campaign.binding()


def test_lead_commands_complete_each_cell_before_score():
    # Prior art: WC1 dependency-ordered commands (house, 2026). Ensure the
    # new diagonal is executable and each score needs only completed units.
    import shlex
    commands = (campaign.OUT/'lead_commands.txt').read_text().splitlines()
    reg = campaign.binding()
    observed = []
    pending = []
    for i, line in enumerate(commands):
        if ' worker --cell ' in line:
            args = shlex.split(line)
            assert args[:3] == ['timeout', '--signal=KILL', '590s']
            cell = args[args.index('--cell')+1]
            pending.append((cell, args[args.index('--battery')+1], args[args.index('--spec')+1]))
            assert commands[i+1] == 'sleep 30'
        elif ' score --cell ' in line:
            cell = shlex.split(line)[-1]
            assert pending == [(cell, u['battery'], u['spec']) for u in reg['units_per_new_cell']]
            pending = []
            observed.append(cell)
    assert not pending and tuple(observed) == campaign.executable_cells(reg)


def test_historical_diagonal_cannot_launch_worker():
    with pytest.raises(AssertionError, match='Only registered new cells'):
        campaign.worker('c96_w96', 'sup', campaign.SUP[0])
