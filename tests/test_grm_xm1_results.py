"""XM1 completeness and evidence-class tests. Prior art: RS4 joins, GRM 2026."""
from scripts import grm_xm1_parity as xm
from scripts.grm_xm1_registration import read
from scripts.grm_xm1_results import summarize


def test_empty_results_are_not_parity(tmp_path):
    reg=read(xm.REG)
    r=summarize(reg,tmp_path,'gpu')
    assert r['reference_barrier']['status']=='BLOCKED'
    assert all(m['verdict']=='BLOCKED_REFERENCE' and m['pairs']==0 for m in r['models'])
    assert len(r['rows'])==30 and all(p['fed_minus_mount'] is None for p in r['rows'])


def test_cpu_summary_never_claims_model_parity(tmp_path):
    r=summarize(read(xm.REG),tmp_path,'cpu')
    assert all(m['verdict']=='CPU_DOUBLE_ONLY' for m in r['models'])
