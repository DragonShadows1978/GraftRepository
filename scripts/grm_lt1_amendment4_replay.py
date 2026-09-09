"""CPU checkpoint/publication diagnostic, not GPT-OSS inference.
Prior art: GRM contributors (2026), C7 numerical doubles and production
load/feed/_native_sync_node reused. No prior art known to me for this exact
checkpoint replay composition. No new retrieval algorithm.
"""
from pathlib import Path
from types import SimpleNamespace
import dataclasses
import json
import shutil
import tempfile
import numpy as np
import pytest
from scripts import grm_lt1 as lt
from scripts import grm_e2e_session as e2e
from scripts.grm_c7_diagnose import repository, CPUArena
from core.graft_arena import ArenaCache
from core.graft_repository import DialectDescriptor

OUT = lt.OUT/'amendment4'
SOURCE = lt.RUN/'cells/A-025-032/checkpoint'
LIB = Path('/mnt/ForgeRealm/GraftRepository/cpp/build/libgrm_runtime.so')


def restored(path, patch):
    """Real manifest/index/WAL/native load; only tensor/model boundaries doubled."""
    repo = repository(path/'empty', patch)
    man = lt.read(SOURCE/'repository/manifest.json')
    target = path/'repository'
    shutil.copytree(SOURCE/'repository', target)
    repo.path = str(target)
    repo.dialect = man['dialect']
    repo.dialect_desc = DialectDescriptor(**man['dialect_descriptor'])
    a = repo.arena
    a.route_layer = man['route_layer']
    # Real checkpoint index values are loaded unchanged. Payloads are text-token
    # doubles so no real K/V tensor reaches a GPU or decoder.
    a.unpack_index = lambda z, i: z[f'rkey_{i:04d}'].astype(np.float32)
    repo._read_payload_file = lambda i: {'tok': np.asarray(a.encode(man['nodes'][i]['text']), dtype=np.int64)}
    repo.native_store = repo._open_native_store(str(LIB))
    repo._own_native_store = True
    a.native_store = repo.native_store
    repo._native_configure_arena()
    repo.load()
    # Execute actual ArenaCache.deposit and feed, replacing only tensor alloc,
    # harvest and key forward with C7 CPU numerical seams.
    import core.graft_arena as ga
    patch.setattr(ga.tc, 'tensor', lambda v: np.asarray(v))
    a.dt = np.float32
    a.deposit = ArenaCache.deposit.__get__(a)
    a._rs3_seat_explicit = False
    return repo


def feed_cell_prefix(repo):
    fed = {}
    for e in lt.read(lt.FIX)['turns'][32:36]:
        i = repo.arena.feed(e2e.harmony_turn(e['user'], e['assistant']))
        repo.arena.grafts[i]['kind'] = 'turn'
        fed[i] = repo.arena.grafts[i]
    return fed


def run():
    r = lt.verify()
    original = {}
    complete = []
    for c in sorted(r['cells'], key=lambda c:(c['start'], c['arm'])):
        if c['stop'] > 32:
            continue
        d = lt.RUN/'cells'/c['id']
        w, ctl = lt.read(d/'worker.json'), lt.read(d/'controller.json')
        assert ctl['status'] == 'COMPLETE'
        assert w['binding'] == ctl['binding'] == lt.binding(c['arm'])
        lt.validate_checkpoint(d/'checkpoint', c['stop']+1, lt.binding(c['arm']))
        assert w['checkpoint_sha256'] == lt.sha(d/'checkpoint/checkpoint.json')
        complete.append(c['id'])
    # Bind EVERY original receipt/checkpoint file, including ignored npz/bin.
    for p in sorted(lt.RUN.rglob('*')):
        if p.is_file():
            original[str(p.relative_to(lt.ROOT))] = lt.sha(p)
    lt.create(OUT/'original_run_shas.json', original)
    with tempfile.TemporaryDirectory(prefix='lt1-a4-replay-') as temp, pytest.MonkeyPatch.context() as patch:
        repo = restored(Path(temp), patch)
        try:
            a = repo.arena
            result = {'evidence_class':'CPU numerical-boundary replay with actual native C ABI and checkpoint',
                      'limits':'Forced failing mount [25] from traceback; no claim to reconstruct missing GPT-OSS turn-39 scores or generation.',
                      'completed_cells_validated':complete,
                      'load':{'nodes':len(a.grafts),'native_stats':dataclasses.asdict(repo.native_store.stats()),
                              'ids':[g.get('native_node_id') for g in a.grafts],
                              'native_checkpoint_loaded':repo._native_checkpoint_loaded}}
            assert len(a.grafts) == 25 and result['load']['ids'] == list(range(25))
            fed = feed_cell_prefix(repo)
            result['fed'] = [{'id':i,'turn':33+i-25,'text':g['text'],'native_node_id':g.get('native_node_id'),
                              'sources':g.get('sources',[]),'metadata':g.get('metadata',{})} for i,g in fed.items()]
            result['after_feed_native_stats'] = dataclasses.asdict(repo.native_store.stats())
            assert list(fed) == [25,26,27,28] and all(g.get('native_node_id') is None for g in fed.values())
            try:
                a._attempt("What did we settle on for Breakwater's map position?", [25], 1, False, (), defer_memory=True)
            except RuntimeError as exc:
                result['RED_before'] = str(exc)
            assert result.get('RED_before') == 'graft 25 has no native_node_id'
            # Existing repository publication is the treatment/control. If this
            # fails, the registered core STOP rail applies; do not repair core.
            native_id = repo._native_sync_node(25)
            a.caches = None
            answer, info = a._attempt("What did we settle on for Breakwater's map position?", [25], 1, False, (), defer_memory=True)
            result['existing_core_publication_control'] = {'native_id':native_id, 'mounts':a.cur_mounts,
                'native_stats':dataclasses.asdict(repo.native_store.stats()),'status':'PASS'}
            lt.create(OUT/'diagnosis.json', result)
            print(json.dumps(result, indent=2))
        finally:
            repo.close()

if __name__ == '__main__':
    run()
