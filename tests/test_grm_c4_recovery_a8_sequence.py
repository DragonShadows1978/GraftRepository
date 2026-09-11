"""A8 supplementary CPU full-sequence check; fake leases only.
Prior art: C4/A5/A6 full-sequence synthetic fixtures (house, 2026), reused
under A8's isolated root to check dependency and partial-summary integration.
"""
from pathlib import Path
from scripts import grm_c4_campaign as c
from scripts import grm_c4_split_a5 as a5
from scripts import grm_c4_recovery_a8 as a8
from test_grm_c4_split_a5 import layout, run, seed_legacy
from test_grm_c4_skip_a6 import a6_layout
from test_grm_c4_recovery_a8 import recovery, marker, TRUNCATED


def test_entire_remaining_sequence_retains_skips_dependencies_and_partial_summary(recovery):
    r=recovery
    seed_legacy(r.base)
    run(r.base,'c96_w64',r.previous['units_per_new_cell'][:5])
    path=r.old/'runs/c96_w64/census_e2e-2.json'
    marker(path,r.previous);path.write_bytes(TRUNCATED.read_bytes())
    r.reg['a8_incidents']=[a8.classify(path,path.with_suffix('.attempt.json'),r.previous)]
    pins=[c.record(p) for p in r.old.rglob('*') if p.is_file()]
    for u in a8.remaining(r.previous,r.reg['a8_incidents']):
        r.base.clock[0]+=1000
        a8.worker(u['cell'],u['battery'],u['spec'])
    with a8.executor():
        for cell in a5.CELLS:
            row=a5.score(cell)
            assert row['status']=='NON_FIT' and len(row['completed_segments'])==19
            assert len(row['failed_segments'])==4 and not row['missing_segments']
            assert 'scores' not in row
        row=a5.cross_summary()
        assert row['factorial_verdict']=='INCONCLUSIVE' and row['prediction_met'] is None
        assert a5.format_summary(row).count('NON_FIT (completed:')==3
        a8.accounting(r.reg,reserve=False)
    assert r.base.controls['lease_calls']==38
    for cell in a5.CELLS:
        for spec in ('c4-lh-12a','c4-lh-13b','c4-lh-12b','c4-lh-13a'):
            p=a8.OUT/'runs'/cell/f'longhorizon_{spec}.json'
            assert c.read(p)['status']=='NON_FIT'
            assert not p.with_suffix('.attempt.json').exists()
    assert all(c.record(p['path'])==p for p in pins)
