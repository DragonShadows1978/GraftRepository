# QWEN3-1.7B NAME-CHECKER — SYNTHESIS (overnight run, 2026-07-20)

Plan `QWEN3_1P7B_NAMECHECKER_PLAN.md` (immutable, 2db1c58) executed to
completion P0→P4 in one overnight autonomous session. Receipts:
`QWEN3_1P7B_NAMECHECKER_LEDGER.md` + `logs/nc17/`. Evidence classes as
registered: memory shape, ppl measurement, parity margins, GRM gates —
no capability claims.

## Cross-stage table (the deliverable)

| | P0 HF bf16 | P1 tc bf16 | P2 tc INT4¹ | P3 tc INT6² |
|---|---|---|---|---|
| resident weights (MiB) | 3890 | 3875.5 | **1901.5** | 2216.5 |
| ppl @2K / @4K³ | 14.96 / 14.70 | 16.47 / 16.91 | 22.30 / 22.49 | **17.07 / 17.29** |
| real quant flips (adjudicated) | — | 0 (port sound) | **92** | **7** |
| fresh final-prefill agreement | ref | 11/11 | 7/11 | **11/11** |
| prefill ceiling (std) | **26880** | 7616 | 8448 | 8320 |
| decode ceiling (std) | 18176 | 30592 | **37888** | 36864 |
| APA r0.15 ppl cost | — | +0.15 | +0.03 | +0.13 |
| APA ceilings (prefill/decode) | — | 5120 / 20224 | 5824 / 23552 | 5632 / 22528 |

¹ asymmetric g128 (house default path). ² symmetric g128 (fork engine,
z=−32·s). ³ P0 = full 299k-token corpus; tc columns = 6–8-window
subsets — tc-vs-P0 absolutes are coverage-confounded; within-engine
deltas are clean. All INT6 ceilings re-measured to true cudaMalloc OOM (P3x,
2026-07-20 morning, David-directed cap removal). 32K product target:
INT6 std-decode session @32768 = 4831 MiB steady / 9001 peak.

## Findings

1. **INT6 is the operating point.** INT4-asym-g128 loses real quality
   on this 1.7B (+5.6 ppl, 92 adjudicated quant flips, 4/11 final
   flips); INT6-sym recovers essentially all of it (+0.6 ppl vs bf16,
   7 flips, finals clean) for +315 MiB. David's INT6 instinct is
   confirmed by measurement.
2. **APA r0.15 is ppl-neutral at every weight format** (≤0.15,
   engagement asserted ~29% per run) but is a net *ceiling* cost at
   these shapes — consistent with the A0 net-cost law. For the
   name-checker's ≤2K prompts this is irrelevant; APA stays available
   without a quality argument against it.
3. **The parity spec fork** (registered law): cached-chain parity
   measures drift tolerance, not correctness — P1's RED was proven
   pure bf16 cache-drift (17/17 fresh-refeed reversions); port
   certification = fresh-prefill protocol. P2/P3 receipts carried
   built-in adjudication thereafter.
4. **GRM is proven on the product config** (INT6/fork): graft
   equivalence at the noise floor with zero flips; save/restore
   bit-identical (the session-multiplexing mechanism); E4 arena 6/6
   at 209 resident seats vs 593 in-context. The 1.7B graft surface is
   pure inheritance — zero adapter code was needed.
5. **Engine prefill-memory gap:** tc bf16 prefill ceiling (7.6K) is
   ~3.5× below HF-SDPA (26.9K) — the O(L²) fp32 score materialization.
   Decode is the mirror image (tc 30.6K beats HF 18.2K). For the
   name-checker (short prompts, session decode) tc's shape wins.

## Product readout (name-checker)

INT6 weights (2.2 GiB resident) + tied head + arena-based session
multiplexing: gates green end-to-end at ≤2.5 GiB peaks. 32K "fit"
question is moot at product prompt sizes; the measured 8K+ walls are
scoring-protocol artifacts (fp32 full-window logits) and time-capped
probes, not product constraints.

## Registered successors (not run)

- INT4 recovery variants: symmetric-8 and/or g32 (was the asym-g128
  loss format-specific?).
- Full-corpus tc ppl (unconfounded absolutes) via chunked scoring.
- INT6 fork → canonical merge (DAVID'S CALL; kernel gates
  lead-verified, branch `int6-weights` at 4d10951).
- TIME-walled APA ladder rungs re-probed with longer caps if long-ctx
  ever matters here.
- Cached-chain drift characterization as its own question (P1b's
  method note).
- Pre-existing canonical GroupNorm failure (test_norms_and_conv1d) —
  found during INT6 gates, predates the branch, unowned.
