#!/usr/bin/env python3
"""Create-only XM2 amendment 1; run before CPU gates.

Prior art: GRM XM1/LT1 (2026), immutable hash overlays and pessimistic
lost-worker accounting, reused. Ours: receipt-derived cold/warm caps;
no new loading or attention algorithm.
"""
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.grm_xm1_registration import read,sha,create

OUT=ROOT/'artifacts/grm_xm2'
BASE='7d537ab25ba485094b0235c9b1407d75b542dcbba21ebdd587a6534d16d73c3d'


def build():
    assert sha(OUT/'registration.json')==BASE
    original=read(OUT/'registration.json')
    evidence=read(OUT/'amendment_1/evidence_before.json')
    timings=evidence['historical_elapsed']
    cold_id='C3l__sup_harbor_restatement'
    cold=next(r for r in timings if Path(r['path']).stem==cold_id)
    warm=max(r['elapsed_s'] for r in timings if r is not cold)
    # Prior art: XM1 fixed lease caps (GRM, 2026). Our preregistered policy:
    # cold = ceil historical seconds; warm = 20 s margin rounded up to 10 s.
    # Warm classification is historical, not a promise of future cache state.
    cold_cap=math.ceil(cold['elapsed_s'])
    warm_cap=10*math.ceil((warm+20)/10)
    concurrent=read(OUT/'amendment_1/evidence_concurrent.json')['receipts']
    meridian_id='C5__sup_reserve_meridian_docket'
    meridian=next(r for r in concurrent if Path(r['path']).stem==meridian_id)
    meridian_cap=10*math.ceil(meridian['elapsed_s']/10)
    cells=[{**c,'reservation_s':cold_cap if c['id']==cold_id else
            meridian_cap if c['id']==meridian_id else warm_cap}
           for c in original['cells']]
    attempts={str(p.relative_to(ROOT)):sha(p) for p in sorted((OUT/'run/attempts').glob('*.json'))}
    prior_reserved=sum(read(ROOT/p)['reservation_s'] for p in attempts)
    total=sum(c['reservation_s'] for c in cells)
    assert cold_cap==130 and warm_cap==90 and meridian_cap==80 and prior_reserved==330
    assert total+prior_reserved<=original['gpu_budget_s']==1800
    pins={
        'orders/GRM_XM1_X2_AMENDMENT_1.md',
        'scripts/grm_xm1_parity.py','scripts/grm_xm1_x2_run.py',
        'scripts/grm_xm1_x2_amendment_1.py','scripts/grm_cmc1_gpu_arms.py',
        'tests/test_grm_xm1_amendment_1.py','tests/test_grm_xm2_final.py',
        'tests/test_grm_xm2_loader.py','artifacts/grm_xm2/lead_commands.txt',
        'artifacts/grm_xm2/amendment_1/evidence_before.json',
        'artifacts/grm_xm2/amendment_1/evidence_concurrent.json',
        'core/qwen35_tc.py','core/mistral7b_tc.py'}
    pins.update(attempts)
    pins.update(str(p.relative_to(ROOT)) for p in (OUT/'run/cells').glob('*.json'))
    pins.update(str(p.relative_to(ROOT)) for p in (OUT/'amendment_1/sources').rglob('*') if p.is_file())
    pins.update(str(p.relative_to(ROOT)) for p in OUT.glob('lead_*.log'))
    return dict(schema='grm.xm2.registration_amendment.v1',amendment=1,immutable=True,
        registration_sha256=BASE,seat_model='gpt-6-astra',seat_effort='high',
        evidence_class='preregistration and source comparison; GPU failure not claimed fixed',
        prior_art=original['prior_art'],
        loader_finding=evidence['finding'],
        loader_contract=dict(callable='scripts.grm_xm1_parity.gpu_loader',
            adapter='core.qwen35_tc.Qwen35_TC.from_pretrained',adapter_kwargs={},
            lm_head='whole-matrix QuantLinearTC',weight_bits=4,
            compute_dtype='bfloat16',linear_dtype='bfloat16',fused_decode=True,fused_norm=True,
            host_embedding=True,host_lm_head=False,chunked_lm_head=False),
        timing_evidence=[{k:r[k] for k in ('path','sha256','elapsed_s')}
            for r in timings+concurrent],
        cap_policy=dict(cold_cell=cold_id,cold_observed_s=cold['elapsed_s'],cold_cap_s=cold_cap,
            warm_max_observed_s=warm,warm_cap_s=warm_cap,
            meridian_c5_observed_s=meridian['elapsed_s'],meridian_c5_cap_s=meridian_cap,
            rule='cold ceil(elapsed_s); Meridian C5 ceil(newer elapsed_s/10)*10; other cells ceil((XM1 max warm elapsed_s + 20)/10)*10. Explicit pre-run allocation under unchanged 1800 s including concurrent prior attempts.',
            residual='A later cold process may exceed 90 s. Stop RED at its registered cap; no retry or cap change. 130 s covers historical cold elapsed, not future GPU fit.'),
        overrides=dict(cells=cells,total_reserved_s=total,prior_reserved_s=prior_reserved,
            prior_attempt_pins=attempts,
            execution='New amendment_1_run directory; one attempt per cell, exact amendment binding. 130 s Harbor C3l, 80 s Meridian C5, 90 s each other cell; 1470 s new plus 330 s prior reservations = 1800 s. Worker leases itself fail-fast, no outer flock, 30 s gap including old attempts, no waiting or automatic retry. Original failed receipts remain immutable.'),
        original_source_pins={str(p.relative_to(OUT/'amendment_1/sources')):sha(p)
            for p in (OUT/'amendment_1/sources').rglob('*') if p.is_file()},
        gates=[
            'Both actual worker entry points use the identical XM1 callable; CPU replay actual adapter load methods and compare lm_head matrix branch, INT4, BF16, fused settings, tokenizer kwargs, and GRM environment.',
            'Historical source identity, original registration/snapshots, amendment parent/source drift fail closed, copied receipt timings and total budget, wrong weight mode rejection.',
            'All 16 refreshed commands --dry-run; original XM1/XM2 suite verbatim LAST; CPU only; scratch cleanup.'],
        deviations=[
            'Different lm_head load path premise contradicted by byte-identical adapters/QuantLinear and identical loader AST. No chunked/host lm_head exists in cited current paths; none invented.',
            'Observed GPU allocation failure root cause remains unresolved; no GPU allowed, not claimed fixed.',
            'Per-cell cold/warm caps preserve 1800 s total including prior attempts. Uniform 130 s would require 2410 s including prior attempts, outside authority.',
            'Procedural RED: first preflight ran after registration creation failed on concurrent attempt accounting; missing-registration/scratch errors, no GPU. Logged before successful registration; no retrospective threshold change.',
            'New amendment authorizes a separate attempt namespace; prior failed/missing receipts retained and charged.'],
        pins={p:sha(ROOT/p) for p in sorted(pins)})


if __name__=='__main__':
    path=OUT/'registration_amendment_1.json'
    create(path,build())
    with (OUT/'registration_amendment_1.sha256').open('x') as f:
        f.write(sha(path)+'  '+str(path.relative_to(ROOT))+'\n')
    print(sha(path))
