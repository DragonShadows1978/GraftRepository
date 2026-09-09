"""C7 A1 CPU regression: fault state survives reload, then a real loader seam
consumes it. Prior art: C2/S4 (GRM, 2026); synthetic I/O, no model/GPU proof.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import grm_c7_common as c
from scripts import grm_c7_run as run


def test_cold_resume_discards_reloaded_ram_and_consumes_state_on_page_in(tmp_path):
    payload=tmp_path/'repository/nodes/0000.npz'
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b'synthetic durable payload')
    state={'cold_nodes':[{'node':0,'payload_sha256':c.sha(payload),'pressure_turn':60}]}
    evicted=[]
    arena=SimpleNamespace(grafts=[{'h':'device restored by constructor','host_payload':'RAM restored by constructor'}],
                          consolidate=lambda *_: None, _attempt=lambda *_: None)
    def load(i):
        # A real file read through the production loader interface, using a
        # CPU double for its model payload. It would fail if RAM survived.
        assert arena.grafts[i]['host_payload'] is None
        data=payload.read_bytes()
        arena.grafts[i].update(h=data,host_payload=data)
        return data
    repo=SimpleNamespace(path=payload.parents[1],arena=arena,_load_node=load,
                         _native_evict_device_copy=evicted.append)
    run.restore_cold_state(repo,state,tmp_path,63)
    assert arena.grafts[0]['h'] is None and arena.grafts[0]['host_payload'] is None
    assert evicted==[0] and len(state['cold_nodes'])==1
    run.install_observers(repo,tmp_path,{'turn':66,'phase':'session','arm':'A'},state)
    assert arena.node_loader(0)==b'synthetic durable payload'
    assert state['cold_nodes']==[]
    record=run.lines(tmp_path/'page_ins.jsonl')[-1]
    assert record=={'turn':66,'node':0,'source':'nvme','success':True,'phase':'session'}


def test_cold_resume_missing_or_changed_payload_is_red(tmp_path):
    payload=tmp_path/'nodes/0000.npz'
    payload.parent.mkdir()
    payload.write_bytes(b'before')
    state={'cold_nodes':[{'node':0,'payload_sha256':c.sha(payload)}]}
    repo=SimpleNamespace(path=tmp_path)
    payload.write_bytes(b'after')
    with pytest.raises(ValueError,match='COLD_RESUME_PAYLOAD_CHANGED'):
        run.restore_cold_state(repo,state,tmp_path,63)
    payload.unlink()
    with pytest.raises(ValueError,match='COLD_RESUME_PAYLOAD_CHANGED'):
        run.restore_cold_state(repo,state,tmp_path,63)


def test_amendment_is_bound_to_registration_and_archive(tmp_path,monkeypatch):
    a=c.read(c.OUT/'amendment_A1.json')
    reg_sha=(c.OUT/'registration.sha256').read_text()
    monkeypatch.setattr(c,'OUT',tmp_path)
    (tmp_path/'registration.sha256').write_text(reg_sha)
    a['registration_sha256']='0'*64
    c.create(tmp_path/'amendment_A1.json',a)
    (tmp_path/'amendment_A1.sha256').write_text(c.sha(tmp_path/'amendment_A1.json'))
    with pytest.raises(ValueError,match='AMENDMENT_REGISTRATION_MISMATCH'):
        c.verify()
    (tmp_path/'amendment_A1.json').write_text('{}')
    with pytest.raises(ValueError,match='AMENDMENT_SHA_MISMATCH'):
        c.verify()
