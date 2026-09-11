"""A7 CPU author baseline; no GPU or blind-review claim.

Prior art: C4/A5/A6 synthetic accounting and lease guards (house,2026), reused.
A7 adds an exact on-disk filename/byte layout and original-function comparison.
No novel algorithm claimed.
"""
import ast
import copy
import json
from pathlib import Path
import shutil

import pytest
from scripts import grm_c4_campaign as c
from scripts import grm_c4_resume_a2 as a2
from scripts import grm_c4_remaining_a3 as a3
from scripts import grm_c4_split_a5 as a5
from scripts import grm_c4_skip_a6 as a6
from scripts import grm_c4_accounting_a7 as a7
from test_grm_c4_skip_a6 import a6_layout
from test_grm_c4_split_a5 import layout, run


@pytest.fixture
def crashed_disk_layout(tmp_path, monkeypatch):
    # Exact top-level scan layout and bytes, including runtime_frame.json in
    # BOTH roots, real paired attempts, REDs and the five lead A6 CPU outputs.
    # No model/session data is copied: accounting never descends into it.
    reg = copy.deepcopy(a6.binding())
    originals = (c.OUT, a2.OUT, a3.OUT, a5.OUT)
    destinations = tuple(tmp_path/root.relative_to(c.OUT) for root in originals)
    pins = []
    for root, dest in zip(originals, destinations):
        for source in (root/'runs').glob('*/*.json'):
            target = dest/source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            assert source.read_bytes() == target.read_bytes()
            if root != a5.OUT:
                pins.append(c.record(target))
    monkeypatch.setattr(c, 'OUT', destinations[0])
    monkeypatch.setattr(a2, 'OUT', destinations[1])
    monkeypatch.setattr(a3, 'OUT', destinations[2])
    monkeypatch.setattr(a5, 'OUT', destinations[3])
    monkeypatch.setattr(a6, 'binding', lambda: reg)
    c.write_once(a5.OUT/'before.json', {'files': pins})
    with a6.executor():
        yield reg


def original_accounting():
    tree = ast.parse((a7.OUT/'grm_c4_split_a5.py.before').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'accounting')
    namespace = dict(vars(a5))
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'original_a5_accounting', 'exec'), namespace)
    return namespace['accounting']


@pytest.mark.campaign_receipt(registration='artifacts/grm_c4/registration.json + amendment_a1..a3.json (pin absolute /mnt/ForgeRealm/wt/grm-c4/ paths)')
def test_exact_crashed_disk_layout_keyerror_then_completed_receipts(crashed_disk_layout, capsys, monkeypatch):
    reg = crashed_disk_layout
    with pytest.raises(KeyError, match='status') as caught:
        original_accounting()(reg, reserve=False)
    frame = caught.traceback[-1].frame
    assert frame.f_locals['path'] == c.OUT/'runs/c64_w96/runtime_frame.json'
    assert frame.f_locals['row']['schema'] == 'grm.det1.runtime_frame.v1'
    assert a5.accounting(reg, reserve=False) == pytest.approx(2157.6785489856265)
    diagnostics = [json.loads(s) for s in capsys.readouterr().err.splitlines()]
    skipped = {r['accounting_skip']: r['reason'] for r in diagnostics}
    assert 'metadata' in skipped[str(c.OUT/'runs/c64_w96/runtime_frame.json')]
    assert 'metadata' in skipped[str(a2.OUT/'runs/c64_w96/runtime_frame.json')]
    assert 'score summary' in skipped[str(a5.OUT/'runs/c64_w96/score.json')]
    for root in (c.OUT, a2.OUT, a3.OUT, a5.OUT):
        for p in (root/'runs').glob('*/*.attempt.json'):
            assert 'attempt marker' in skipped[str(p)]
    assert sum('NON_FIT' in reason for reason in skipped.values()) == 4
    # Treatment effect at the exact first unstarted worker: it now reaches
    # readiness after admission. Stop BEFORE any lease import/acquisition.
    class ReachedReadiness(Exception):
        pass
    def ready(*args):
        raise ReachedReadiness
    monkeypatch.setattr(a5, 'ready', ready)
    with pytest.raises(ReachedReadiness):
        a5.worker('c96_w64', 'sup', 'correction_then_restatement')


@pytest.mark.campaign_receipt(registration='artifacts/grm_c4/registration.json + amendment_a1..a3.json (pin absolute /mnt/ForgeRealm/wt/grm-c4/ paths)')
def test_metadata_attempt_and_score_are_excluded_before_read(crashed_disk_layout, monkeypatch, capsys):
    original = c.read
    def read(path):
        if Path(path).name in ('runtime_frame.json','score.json') or str(path).endswith('.attempt.json'):
            pytest.fail(f'Accounting parsed non-receipt: {path}')
        return original(path)
    monkeypatch.setattr(c, 'read', read)
    assert a5.accounting(crashed_disk_layout, reserve=False) == pytest.approx(2157.6785489856265)
    assert 'runtime_frame.json' in capsys.readouterr().err


@pytest.mark.campaign_receipt(registration='artifacts/grm_c4/registration.json + amendment_a1..a3.json (pin absolute /mnt/ForgeRealm/wt/grm-c4/ paths)')
@pytest.mark.parametrize('contents', ['{', '[]', '{}', '{"status":"PASS"}',
    '{"status":"PASS","cell":"c96_w64","battery":"sup","spec":"partial"}'])
@pytest.mark.parametrize('claim', [False, True])
def test_partial_worker_diagnosed_and_blocks_admission(a6_layout, capsys, contents, claim):
    path = a5.OUT/'runs/c96_w64/sup_partial.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    if claim:
        path.with_suffix('.attempt.json').write_text('not JSON; marker only')
    with pytest.raises(AssertionError, match='campaign blocked'):
        a5.accounting(a6_layout.reg, reserve=False)
    err = capsys.readouterr().err
    assert str(path) in err and ('incomplete' in err or 'unreadable' in err)
    assert a6_layout.controls['lease_calls'] == 0


@pytest.mark.campaign_receipt(registration='artifacts/grm_c4/registration.json + amendment_a1..a3.json (pin absolute /mnt/ForgeRealm/wt/grm-c4/ paths)')
@pytest.mark.parametrize('field,value', [('gpu_seconds',float('nan')), ('gpu_seconds',float('inf')),
    ('gpu_seconds',-1), ('gpu_seconds',True), ('gpu_seconds','1'),
    ('finished_unix',float('nan')), ('finished_unix',-1)])
def test_invalid_completed_wall_refused(a6_layout, field, value):
    run(a6_layout,'c96_w64',a6_layout.reg['units_per_new_cell'][:1])
    path=a5.receipt_path('c96_w64','sup',c.SUP[0]);row=c.read(path)
    row[field]=value;path.write_text(json.dumps(row))
    with pytest.raises(AssertionError, match='Invalid wall accounting'):
        a5.accounting(a6_layout.reg, reserve=False)


@pytest.mark.campaign_receipt(registration='artifacts/grm_c4/registration.json + amendment_a1..a3.json (pin absolute /mnt/ForgeRealm/wt/grm-c4/ paths)')
def test_a6_plan_budget_commands_and_historical_artifacts_unchanged():
    reg=a6.binding();old=c.read(a6.AMENDMENT)
    assert reg['a5_plan']==old['plan'] and reg['budget']==old['budget']
    assert reg['budget']['gpu_seconds']==6600
    assert a7.COMMANDS.read_bytes()==a6.COMMANDS.read_bytes()
    replacements={str(c.ROOT/p) for p in a7.REPLACEMENTS}
    before=c.read(a7.OUT/'before.json')['files']
    assert all(c.record(p['path'])==p for p in before
               if p['path'] not in replacements and p['path']!=str(c.OUT/'NARRATIVE_SYNTHESIS.md'))
    assert len(list((a5.OUT/'runs').glob('*/*.json')))==5
    assert not list((a5.OUT/'runs').glob('*/*.attempt.json'))


@pytest.mark.parametrize('field', ['sha','source_replacements','added_sources','policy','extra'])
def test_a7_forged_manifest_refused(tmp_path, monkeypatch, field):
    row=c.read(a7.AMENDMENT)
    if field=='sha':row['immutable']=False
    elif field in ('source_replacements','added_sources'):row[field]=[]
    else:row[field]='forged'
    path=tmp_path/'amendment.json';c.write_once(path,row)
    path.with_suffix('.sha256').write_text('stale' if field=='sha' else c.record(path)['sha256'])
    monkeypatch.setattr(a7,'AMENDMENT',path)
    with pytest.raises(AssertionError):a6.binding()


@pytest.mark.parametrize('target', ['executor','entrypoint','helper','test','commands','order','before','predecessor','snapshot'])
def test_a7_stale_bound_bytes_refused(monkeypatch,target):
    paths={'executor':Path(a5.__file__),'entrypoint':Path(a6.__file__),
        'helper':Path(a7.__file__),'test':Path(__file__), 'commands':a7.COMMANDS,
        'order':a7.ORDER,'before':a7.OUT/'before.json','predecessor':a7.PREVIOUS,
        'snapshot':a7.OUT/'grm_c4_split_a5.py.before'}
    original=c.record
    def record(path):
        pin=original(path)
        if str(path)==str(paths[target]):pin['sha256']='stale'
        return pin
    monkeypatch.setattr(c,'record',record)
    with pytest.raises(AssertionError):a6.binding()
