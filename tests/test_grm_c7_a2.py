"""C7 A2 forced pager protocol. Prior art: S4 loader/pager (GRM,2026),
Denning1968 working sets. CPU I/O doubles test the harness, not model quality.
"""
from types import SimpleNamespace
from scripts import grm_c7_common as c
from scripts import grm_c7_run as run


def test_pressure_forces_old_hot_to_disk_and_back_without_claiming_retrieval(tmp_path):
    payload=tmp_path/'repository/nodes/0000.npz'
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b'old cold source')
    g={'h':None,'host_payload':None,'sources':[]}
    arena=SimpleNamespace(grafts=[g],consolidate=lambda *_:None,_attempt=lambda *_:None)
    context={'turn':240,'phase':'post_turn','arm':'A'}
    state={'turn_records':{'6':{'chat_node_id':0}},'cold_nodes':[]}
    def load(i):
        value=g['host_payload'] if g['host_payload'] is not None else payload.read_bytes()
        g.update(h=value,host_payload=value)
        return value
    def ensure(ids):
        for i in ids:
            if arena.grafts[i]['h'] is None:
                arena.node_loader(i)
    arena._ensure_h=ensure
    def page():
        assert repo.vram_budget==0
        assert g['h']==b'old cold source'
        g['h']=None
        return 1
    repo=SimpleNamespace(path=payload.parents[1],arena=arena,_load_node=load,
                         _page=page,flush_now=lambda:None,vram_budget=None)
    run.install_observers(repo,tmp_path,context,state)
    row=run.pressure(repo,state,240,tmp_path,context)
    assert row['old_hot_before']==[0] and row['evicted_total']==1
    assert row['cold'][0]['was_hot'] and row['cold'][0]['controlled_return']
    assert g['h']==b'old cold source' and repo.vram_budget is None
    assert not row['controlled_return_is_retrieval_success']
    returns=[x for x in run.lines(tmp_path/'page_ins.jsonl') if x['phase']=='pressure_return']
    assert len(returns)==1 and returns[0]['source']=='nvme' and returns[0]['success']
