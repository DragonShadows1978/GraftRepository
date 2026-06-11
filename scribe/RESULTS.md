# SCRIBE — results ledger (house format: every gate, dated, raw numbers)

| Gate | Date | Verdict | Measurement |
|---|---|---|---|
| G0 instrument | 2026-06-11 | **PASS** | Minter (scribe/mint.py) pipeline Δ = 0.00000 vs direct harvest (3 docs, 3 domains); graft-vs-in-context floor 0.3125 max\|Δlogit\|; flips only at ≤1-ULP tie margins. Harness: tests/test_scribe_g0.py |
| Fit probe | 2026-06-11 | **co-resident OK** | target+student+Adam peak 4,223 MB / 8,192 (S=224); functional loss needs NO swap mode. ARM-L 45.7M / ARM-S 31.5M params |
| Throughput (§5 ratio) | 2026-06-11 | measured | **C_student/C_target: ARM-L 0.008, ARM-S 0.022** (student 0.03/0.10 ms/tok vs teacher prefill 4.2/4.6 ms/tok) — minting at 1-2%% of teacher cost; cost is architecture-set, training buys fidelity |
| Resume-from-kill | 2026-06-11 | **PASS** | warm trainer: ckpt -> hard-kill -> fresh-process resume, steps 4-7 BIT-IDENTICAL to unbroken run; cursor + per-layer error EMA continuous |

**PHASE 0 EXIT: 2026-06-11 — all criteria met** (G0 green; fit mode
chosen+verified co-resident; thresholds registered; resume-from-kill
green). Phase 1 (ARM-L floor) is next: needs the minted seed set.
| G1 (ARM-L) | 2026-06-11 | recorded | per-layer error: L0 0.002 (solved), BULGE peak L36 ~0.20, decline to ~0.17 at L61 — **H3's monotone form falsified**: contextualization hump, not growth. Full trajectory in arm_l ckpts |
| G2 (ARM-L floor) | 2026-06-11 | FAIL (floor recorded) | top-1 51-53%%, KL 0.85-1.02 vs thresholds 90%%/0.5 |
| G3 (ARM-L floor) | 2026-06-11 | FAIL (floor recorded) | predicted 0/10 vs exact 10/10 in-context 10/10 — **H2 strong form at the floor: content collapses, logits half-right** |
| G4 (ARM-L floor) | 2026-06-11 | INSTRUMENT FLAW | same-template sibling docs broke centroid routing for EXACT too (5/20@1) — corpus-100 regime reproduced; predicted trailed by only 3/20. Gate re-instrumented with diverse-topic docs BEFORE any ARM-S run |

**Phase 1 verdict: floor recorded. The problem is NOT easier than
believed — linear maps solve routing-adjacent layers, cannot write
readable content.** ARM-S (Phase 2) is the real test.
