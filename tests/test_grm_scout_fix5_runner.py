"""Prior art: C7 CPU receipt/double and SHA checks (GRM contributors, 2026).
New replay restoration and controller rail stimuli; no prior art known to me
for this exact composition. CPU unpacking is not device attention validation.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer

from core.graft_quant import unpack_kv_arrays
from scripts import grm_scout_fix5 as run

REG = run.read(run.REG)


@pytest.mark.parametrize('fold', REG['folds'], ids=lambda f: str(f['turn']))
def test_bound_source_restore_cpu(fold):
    tok = Tokenizer.from_file(str(Path(REG['model']['path'])/'tokenizer.json'))
    def unpack(payload):
        kv = unpack_kv_arrays(payload, ['k','v'])
        return [{key:kv[key][i][None] for key in ('k','v')} for i in range(24)]
    a = SimpleNamespace(grafts=[], encode=lambda s:tok.encode(s, add_special_tokens=False).ids,
            unpack_index=lambda z,i:z[f'rkey_{i:04d}'].astype(np.float32), unpack_node=unpack)
    mapping = run.restore_sources(a, fold)
    start = run.read(run.ROOT/fold['start'])
    assert [g['text'] for g in a.grafts] == [s['text'] for s in start['sources']]
    assert [m['original_id'] for m in mapping] == [s['id'] for s in start['sources']]
    assert [m['replay_id'] for m in mapping] == list(range(len(start['sources'])))
    assert all(g['kind'] == (s['kind'] or 'turn') for g,s in zip(a.grafts,start['sources']))
    assert all(len(g['h']) == 24 for g in a.grafts)
    assert all(layer['k'].shape[1:] == (8,g['ntok'],64) for g in a.grafts for layer in g['h'])


def test_gpu_registration_bindings():
    assert len(run.verify()['folds']) == 13


def test_verify_rejects_source_hash_change(monkeypatch):
    original = run.sha
    monkeypatch.setattr(run, 'sha', lambda p:'bad' if Path(p)==run.ROOT/'core/graft_arena.py' else original(p))
    with pytest.raises(ValueError, match='INPUT_SHA_MISMATCH'):
        run.verify()


def test_direct_worker_refused_before_model_or_lease(monkeypatch):
    monkeypatch.delenv('GRM_FIX5_PARENT', raising=False)
    with pytest.raises(ValueError, match='WORKER_REQUIRES_FOREGROUND_PARENT'):
        run.worker(8)


@pytest.mark.parametrize('fail_at', [None, 4])
def test_controller_foreground_no_retry_accounting(tmp_path, monkeypatch, fail_at):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    monkeypatch.setattr(run, 'verify', lambda:REG)
    calls = []
    def fake_child(command, **kwargs):
        assert 'timeout' not in kwargs
        assert kwargs['env']['GRM_FIX5_PARENT'] == str(run.os.getpid())
        turn = int(command[-1])
        calls.append(turn)
        if len(calls) == fail_at:
            return SimpleNamespace(returncode=1)
        run.create(tmp_path/'gpu'/f'{turn:03d}'/'receipt.json', {
            'turn':turn,'accepted':False,'digest_text':None,'result':{'best_cov':.25}})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(run.subprocess, 'run', fake_child)
    assert run.run() == (1 if fail_at else 0)
    assert len(calls) == (fail_at or 13)
    summary = run.read(tmp_path/'gpu/summary.json')
    assert summary['charged_seconds'] == 80*len(calls) <= 1080
    with pytest.raises(FileExistsError):
        run.run()
    assert len(calls) == (fail_at or 13)


def test_core_delta_scope_and_unchanged_qc_coverage():
    before = ast.parse((run.OUT/'before/core/graft_arena.py').read_text())
    after = ast.parse((run.ROOT/'core/graft_arena.py').read_text())
    changed = set()
    old = next(n for n in before.body if isinstance(n, ast.ClassDef) and n.name=='ArenaCache')
    new = next(n for n in after.body if isinstance(n, ast.ClassDef) and n.name=='ArenaCache')
    assert len(old.body) == len(new.body)
    for a,b in zip(old.body,new.body):
        if ast.dump(a) != ast.dump(b):
            assert isinstance(a, ast.FunctionDef) and a.name==b.name
            changed.add(a.name)
    assert changed == {'consolidate','_consolidation_prompts'}
    old.body = new.body = []
    assert ast.dump(before) == ast.dump(after)
