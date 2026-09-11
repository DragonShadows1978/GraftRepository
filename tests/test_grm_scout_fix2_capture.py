"""Prior art: B3 manifest fixtures and LSR-P2C CPU slice arena (project, 2026).
Taken: deterministic CPU payloads; new: inherited capture identity/isolation gates.
No prior art known to me for this regression beyond those local systems.
"""
import copy
import json
import numpy as np
import pytest
from core.graft_repository import GraftRepository
from tests.test_grm_runtime_lifecycle import FakeModel, FakeSliceArena, enc, dec
from scripts.grm_c2_cells import strict_capture_grade

CAPTURE = dict(capture_pin='live', capture_shift=115,
    capture_shift_observed=115, capture_shift_derived_from='recorded parent geometry',
    n_sink=19, arena_width=96, live_shift=115,
    capture_provenance_hashes={'source': ['a' * 64]})


class SliceArena(FakeSliceArena):
    def deposit(self, text):
        idx = super().deposit(text)
        self.grafts[idx]["rare"] = self._rare_tokens(text)
        return idx


class CapturedArena(SliceArena):
    def deposit(self, text):
        idx = super().deposit(text)
        self.grafts[idx].update(copy.deepcopy(CAPTURE))
        return idx


def repository(path, captured=True, width=None):
    repo = GraftRepository(FakeModel(), enc, dec, str(path), autosave=False,
                          arena_cls=CapturedArena if captured else SliceArena)
    if width is not None:
        repo.arena.width = width
    return repo


def expected(parent, identity):
    return dict(parent, capture_inherited_from_parent=True,
                capture_parent_graft_id=identity)


@pytest.mark.parametrize('route', ['deposit', 'cull', 'fit', 'repair'])
def test_split_children_inherit_exact_capture(tmp_path, route):
    repo = repository(tmp_path, width=3 if route == 'deposit' else None)
    parent = repo.add_document('A1 B2 C3 D4 E5 F6')
    if route == 'deposit':
        children = repo.arena.grafts[parent]['sources']
    elif route == 'cull':
        children = repo.cull_graft(parent, max_tokens=3)['children']
    elif route == 'fit':
        children = repo._split_for_fit(parent, 3)
    else:
        repo.arena.width = 3
        children = repo.split_oversized(budget=3)['children']
    assert len(children) == 2
    assert repo._node_manifest(repo.arena.grafts[parent])['capture'] == CAPTURE
    for i, child_id in enumerate(children):
        child = repo.arena.grafts[child_id]
        assert repo._node_manifest(child).get('capture') == expected(CAPTURE, parent)
        assert child['host_payload']['tok'].tolist() == list(range(i * 3, (i + 1) * 3))
    nodes = [repo._node_manifest(g) for g in repo.arena.grafts]
    assert strict_capture_grade(nodes, 'profile', 96, 19)['valid']
    repo.arena.grafts[children[0]]['capture_provenance_hashes']['source'].append('child only')
    assert repo.arena.grafts[parent]['capture_provenance_hashes'] == CAPTURE['capture_provenance_hashes']
    assert repo.arena.grafts[children[1]]['capture_provenance_hashes'] == CAPTURE['capture_provenance_hashes']


@pytest.mark.parametrize('mode', ['live', 'off', 'cache'])
def test_split_capture_save_reload_round_trip(tmp_path, mode):
    repo = repository(tmp_path)
    parent = repo.add_document('A1 B2 C3 D4 E5 F6')
    capture = copy.deepcopy(CAPTURE)
    if mode == 'off':
        capture.update(capture_pin='off', capture_shift=None, capture_shift_observed=None)
    elif mode == 'cache':
        capture.update(capture_payload_source='live_cache_slice',
                       capture_span_pos0=120, capture_pin_moves_payload=False)
    repo.arena.grafts[parent].update(capture)
    children = repo.cull_graft(parent, max_tokens=3)['children']
    repo.save()
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    for child_id in children:
        assert manifest['nodes'][child_id].get('capture') == expected(capture, parent)
    loaded = repository(tmp_path)
    for child_id in children:
        node = loaded.arena.grafts[child_id]
        assert loaded._node_manifest(node)['capture'] == expected(capture, parent)
        np.testing.assert_array_equal(node['host_payload']['tok'], repo.arena.grafts[child_id]['host_payload']['tok'])
    loaded.save()
    assert [n.get('capture') for n in json.loads((tmp_path / 'manifest.json').read_text())['nodes']] == [n.get('capture') for n in manifest['nodes']]


def test_nested_split_names_immediate_parent(tmp_path):
    repo = repository(tmp_path)
    parent = repo.add_document('A1 B2 C3 D4 E5 F6')
    child = repo.cull_graft(parent, max_tokens=3)['children'][0]
    grandchild = repo.cull_graft(child, max_tokens=2)['children'][0]
    assert repo._node_manifest(repo.arena.grafts[grandchild]).get('capture') == expected(CAPTURE, child)


def test_split_without_capture_does_not_invent_it(tmp_path):
    repo = repository(tmp_path, captured=False)
    parent = repo.add_document('A1 B2 C3 D4')
    children = repo.cull_graft(parent, max_tokens=2)['children']
    assert all('capture' not in repo._node_manifest(repo.arena.grafts[i]) for i in [parent] + children)


def test_non_split_capture_is_unchanged(tmp_path):
    repo = repository(tmp_path, width=10)
    parent = repo.add_document('A1 B2')
    assert len(repo.arena.grafts) == 1
    assert repo._node_manifest(repo.arena.grafts[parent])['capture'] == CAPTURE
