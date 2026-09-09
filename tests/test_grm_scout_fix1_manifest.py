"""Prior art: RS3 capture receipt and repository lifecycle CPU fake (2026).
New coverage: ordinary manifest persistence, no payload recapture or GPU.
"""
import json
import pytest
from core.graft_repository import GraftRepository
from tests.test_grm_runtime_lifecycle import FakeArena, FakeModel


def repository(path):
    return GraftRepository(FakeModel(), lambda s: s.split(), lambda ids: ' '.join(ids),
                           str(path), autosave=False, arena_cls=FakeArena)


@pytest.mark.parametrize('mode', ['live', 'off', 'cache'])
def test_capture_save_reload_round_trip(tmp_path, mode):
    repo = repository(tmp_path)
    idx = repo.add_document('capture fixture 1234')
    capture = dict(capture_pin='off' if mode == 'off' else 'live',
                   capture_shift=None if mode == 'off' else 100,
                   capture_shift_observed=None if mode == 'off' else 100,
                   capture_shift_derived_from='fixture geometry',
                   n_sink=4, arena_width=96, live_shift=100)
    if mode == 'cache':
        capture.update(capture_payload_source='live_cache_slice',
                       capture_span_pos0=104, capture_pin_moves_payload=False)
    repo.arena.grafts[idx].update(capture)
    repo.save()
    manifest = json.loads((tmp_path/'manifest.json').read_text())
    assert manifest['nodes'][idx]['capture'] == capture
    reloaded = repository(tmp_path)
    node = reloaded.arena.grafts[idx]
    assert {key: node[key] for key in capture} == capture
    assert node['text'] == repo.arena.grafts[idx]['text']
    assert node['ntok'] == repo.arena.grafts[idx]['ntok']
    reloaded.arena._ensure_h([idx])
    assert node['h'] == repo.arena.grafts[idx]['h']
    reloaded.save()
    assert json.loads((tmp_path/'manifest.json').read_text())['nodes'][idx]['capture'] == capture


def test_old_manifest_without_capture_loads_unchanged(tmp_path):
    repo = repository(tmp_path)
    idx = repo.add_document('old capture fixture 9876')
    repo.save()
    path = tmp_path/'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['nodes'][idx].pop('capture', None)
    path.write_text(json.dumps(manifest))
    reloaded = repository(tmp_path)
    node = reloaded.arena.grafts[idx]
    assert not any(k.startswith('capture_') for k in node)
    assert not {'n_sink', 'arena_width', 'live_shift'} & node.keys()
    reloaded.arena._ensure_h([idx])
    assert node['h'] == repo.arena.grafts[idx]['h']
    assert node['text'] == repo.arena.grafts[idx]['text']
