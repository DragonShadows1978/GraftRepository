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
