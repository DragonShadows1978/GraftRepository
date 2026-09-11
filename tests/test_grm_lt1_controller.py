"""Author CPU controller contracts. Prior art: C2/C7 reservation and
checkpoint corruption tests (GRM, 2026). New combined-arm boundary cases;
no prior art known to me for this exact composition. No processes launched.
"""
import ast
from pathlib import Path
import pytest
from scripts import grm_lt1 as lt
from scripts import grm_lt1_worker as worker


def test_combined_budget_reserves_full_lease_before_start(monkeypatch):
    r=lt.verify();cell=r['cells'][0]
    monkeypatch.setattr(worker,'accounting',lambda:10800-284)
    with pytest.raises(ValueError,match='COMBINED_GPU_BUDGET_RAIL'):
        worker.run_cell(cell,r)


def test_resume_validates_completed_checkpoint_then_interleaves(tmp_path):
    r=lt.verify();cell=r['cells'][0]
    assert worker.pending(r,tmp_path)['id']=='A-001-008'
    d=tmp_path/'cells'/cell['id'];cp=d/'checkpoint';cp.mkdir(parents=True);(cp/'repository').mkdir()
    lt.checkpoint(cp,dict(next_turn=9,process_id='completed-A'),lt.binding('A'))
    lt.create(d/'reservation.json',dict(seconds=285))
    lt.create(d/'controller.json',dict(status='COMPLETE',charged_seconds=200,binding=lt.binding('A')))
    lt.create(d/'worker.json',dict(binding=lt.binding('A'),checkpoint_sha256=lt.sha(cp/'checkpoint.json')))
    assert worker.pending(r,tmp_path)['id']=='B-001-008'
    (cp/'repository/corrupted').write_text('unexpected payload')
    with pytest.raises(ValueError,match='CHECKPOINT_INTEGRITY'):worker.pending(r,tmp_path)


def test_controller_has_no_kill_or_subprocess_timeout_path():
    tree=ast.parse(Path(worker.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            assert not (isinstance(node.func,ast.Attribute) and node.func.attr in ('kill','terminate'))
            if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='subprocess':
                assert not any(k.arg=='timeout' for k in node.keywords)
