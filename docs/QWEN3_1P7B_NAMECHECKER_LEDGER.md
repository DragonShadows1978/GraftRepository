# QWEN3-1.7B NAME-CHECKER — LEDGER

Receipts only. Plan: `QWEN3_1P7B_NAMECHECKER_PLAN.md` (immutable).

## 2026-07-20 — Track opened
- Plan committed this commit. Model decided in [[project-frontier-npc-llm]]
  session: Qwen/Qwen3-1.7B (config verified from HF: 28L, 16Q/8KV,
  hd128, RoPE theta 1e6, tied embeds, vocab 151936, max_position 40960).
- Seats per David: kernels = Sol max (Project-Tensor fork
  `int6-weights`); all other code = Opus 4.8 at MAX effort.
- Phases P0 (HF baseline) → P1 (adapter+APA r0.15) → P2 (INT4) →
  P3 (INT6, conditional) → P4 (GRM proof). Receipts append below.

## 2026-07-20 ~05:40 — P3 kernel half DELIVERED (Sol, one dispatch)
- INT6 weight path landed on Project-Tensor fork branch `int6-weights`
  at 4d10951: g128 symmetric (z=-32s), 4-codes-per-3-bytes packing,
  CUDA dequant + packed GEMV + fused tile path (no fp16 expansion),
  INT4-mirrored Python surface (int6_dequant/int6_linear/
  int6_linear_fused). All order gates numbered PASS in receipt
  (dequant bit-exact 0.0; GEMV parity ~3e-4 rel-fro at real shapes
  incl. 151936-row chunked head; INT4 regression untouched).
- Full-suite gate honestly FAIL: pre-existing GroupNorm
  test_norms_and_conv1d failure — lead REPRODUCED on canonical
  (1 failed, 0.33s) — predates INT6, not this branch's problem.
- Lead spot-checks: import surface verified; canonical-failure claim
  verified; acceptance gate re-run under lead harness queued on flock
  (receipt appends when it completes).
- Deviations disclosed by seat: build.sh FetchContent git-clones
  blocked by sandbox (built from cached pybind11 v2.12.0); sandbox
  had no CUDA — GPU gate runs executed outside that boundary with
  lock+timeout retained (lead re-run is the authoritative receipt).
- Merge to canonical = DAVID'S CALL (plan-registered), pending his
  morning review + the P3 measurement half (INT6-quantized 1.7B
  battery, needs P2's adapter INT4 path as template).
