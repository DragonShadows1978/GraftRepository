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
- 05:5x lead re-run receipt: acceptance gate PASS under lead harness,
  all numbers identical to seat receipt (GEMV rel-fro 3.0-3.2e-4 all
  four shapes; dequant 0.0). INT6 kernel half VERIFIED-GREEN.

## 2026-07-20 ~05:5x — P0 COMPLETE (Opus seat, clean)
- Revision 70d244cc, 3.80 GiB bf16. Config confirmed (28L/16Q/8KV/
  hd128, tied embeds).
- CEILINGS stock-HF bf16 SDPA: prefill last-solid 26880 / OOM 27392
  (fp32 score-matrix alloc); decode+64 last-solid 18176 / OOM 18432.
  Poller peaks 11.77/11.86 GiB.
- PPL (wikitext-2 stride 512, 299,077 tok): 2048=14.96245,
  4096=14.70080; 8192+16384 OOM AT SCORING-FORWARD (fp32 SDPA scores
  4.64/9.27 GiB — not fragmentation, verified as 8-window slice);
  32768 above ceiling, skipped. Stock HF cannot SCORE >=8K on this
  card — the baseline the engine battery is measured against.
- Deviations (accepted): memory-frugal lm_head-on-tail loss path,
  proven bit-identical @2048 before use; sibling-P1 concurrency
  disclosed, flock held.
- GT: p0_gt.npz (11 prompts incl. 8 coverage languages + 2
  name-verdict shapes, final+64-step logits), ppl corpus sha
  696cca6b…; sentinel written last.

## 2026-07-20 ~06:5x — P1 COMPLETE (Opus seat); parity gate RED-as-registered, adjudication dispatched
- Adapter core/qwen3_1p7b_tc.py, bf16. Tied head RESIDENT at 0 bytes
  (bit-identical tensors verified; separate load would waste 593.5MiB).
  Resident bf16 3875.5 MB.
- PARITY vs GT (sha 0fc4099b…): 682/715 top-1 (95.4%); final-prefill
  position 11/11 CLEAN; 33 flips = 16 near-tie + 17 HARD (> registered
  0.25 margin) → GATE RED. Seat hypothesis: bf16 cache-chain drift over
  64-step teacher-forced decode (15/17 HARD at step ≥10, median 34).
  HYPOTHESIS UNTESTED → P1b adjudication order dispatched (fresh-refeed
  at flip positions). P2 HELD until verdict.
- CEILINGS tc-bf16 (one-proc-per-probe): std prefill 7616 (engine
  O(L²) fp32 scores — far below HF-SDPA 26880); std decode 30592
  (beats HF 18176); APA r0.15 prefill 5120 / decode 20224 — APA COSTS
  ceiling at bf16 (−33%), consistent with A0 net-cost finding.
- PPL (matched-window clean delta): APA r0.15 costs +0.15 ppl @2048,
  +0.14 @4096, engagement asserted (mean engaged fraction 0.292,
  56/56 calls). ≥8K full-window scoring OOMs on BOTH stacks (inherent
  fp32 logits+scores wall). tc-vs-HF absolute ppl coverage-confounded
  (6-8 windows vs full corpus) — stated, not compared.
- DEVIATION accepted after lead diff review: one-line bulk_bits fix in
  core/mistral7b_tc.py pure-apa path (hardcoded 2 → self.bulk_bits;
  default preserved; qwen3 family sets 8, qwen35 sets 4; fix took 1.7B
  APA ppl 132 → 16.6). Other ports unaffected by inspection.

## 2026-07-20 ~07:1x — P1b ADJUDICATION: PORT SOUND (17/17 explained, control 10/10)
- Fresh-prefill refeed at every HARD flip position: all 17 revert to GT
  top-1 on the merits (none needed the near-tie clause); 10-position
  agreeing control all clean. Cache-chain bf16 drift CONFIRMED as the
  P1 RED's mechanism. Registered threshold ≥15/17; result 17/17.
- P1 parity gate REMAINS RED as written (its cached-chain protocol
  measures drift tolerance, not port correctness). SPEC NOTE REGISTERED
  (lead): port-certification parity = fresh-prefill protocol (P1b
  method); cached-chain parity = drift characterization, separate
  question. P2's clean quant isolation = INT4-vs-P1-bf16 same-engine
  cached-chain delta (both chains drift identically).
- Artifacts: tests/nc17/p1b_adjudicate.py, logs/nc17/p1b_verdict.{json,log}.
  Peak 3721 MiB, rc=0. PIPELINE UNBLOCKED → P2 dispatched.

## 2026-07-20 ~09:0x — P2 COMPLETE (Opus seat): INT4 MARGINAL — P3 triggered
- Quant: house INT4 asymmetric g128 (196 proj matrices; embeds/norms/
  qk-norm exempt; tied head 0-cost). Artifact weights_nc17/
  qwen3_1p7b_int4/int4_weights.npz sha 20a58460… 714.2 MiB on disk
  (3.76× vs bf16 proj); RESIDENT 1901.5 MiB (vs bf16 3875.5).
- PARITY: cached-chain vs GT 75.4% (GATE RED, expected class);
  adjudication BUILT-IN: 13 flips shared-with-bf16 (drift), 44
  drift-explained, **92 REAL INT4 quant flips** (>0.25 GT margin,
  persist under fresh refeed); fresh final-prefill 7/11 clean vs P1's
  11/11 (4 HARD flips, margins to 4.125). Port SOUND (control 10/10);
  loss is real INT4 accuracy loss. Same-engine Δ-vs-bf16: mean 1.63,
  p99 6.14, max 19.03.
- PPL matched-window: INT4 +5.6 to +5.8 vs bf16 (22.30/22.49 vs
  16.47/16.91 std @2K/4K); APA r0.15 delta stays negligible on INT4
  (+0.03/−0.12, ENGAGED, frac 0.293). INT4 scored ONE 8K window
  (22.79) before OOM — partial lift of the 8K scoring wall.
- CEILINGS: INT4 lifts all tc rungs — std prefill 8448 / decode 37888
  (bf16: 7616/30592); APA 5824/23552 (APA still net ceiling cost).
- DECISION (lead, per goal authority; plan gate = David's, exercised
  in his sleep per "best judgement" directive): INT4 MARGINAL →
  **P3m DISPATCHED** (INT6 battery on fork engine). Successor
  registered, NOT run tonight: INT4 variants (g32 / symmetric-8) as
  possible recovery of the asymmetric-g128 loss.
- Deviation accepted: adapter extended with int4 load path (bf16 path
  verified untouched). Seat process-discipline corrected mid-run
  (waiter churn → foreground) — receipts unaffected.

## 2026-07-20 ~10:1x — P3m COMPLETE (Opus seat, fork engine): INT6 = THE OPERATING POINT
- Fork engine asserted per run (ENGINE=int6-fork on numeric lines);
  sanity gate PASS (dequant 0.0, fused rel-fro 3.1e-7). Engine-path
  shadowing hazard (mistral7b_tc sys.path insert) neutralized via
  import order, no engine edits.
- Quant: INT6 g128 SYMMETRIC (z=-32s, empty zeros), 196 matrices,
  artifact sha 2bde6944… 1029.1 MiB disk; RESIDENT 2216.5 MiB
  (INT4 1901.5 / bf16 3875.5).
- QUALITY: ppl 17.07/17.29 @2K/4K (bf16 16.47/16.91; INT4
  22.30/22.49) → +0.6 vs bf16, recovers ~5 ppl of INT4's loss. One 8K
  window 17.15. Parity: cached-chain 93.4% (INT4 75.4%, bf16 95.4%);
  FINALS 11/11 CLEAN; adjudicated real quant flips 7 (INT4: 92) —
  ~13× cleaner. Same-engine Δ-vs-bf16 mean 0.346 / p99 1.42.
- APA r0.15 on INT6: negligible ppl cost (+0.13/+0.05), engaged 0.292.
- CEILINGS: std prefill 8320; std decode ≥25600 (bracket-exhausted,
  not wall — deviation accepted); apa prefill 5056 (7168 = true OOM);
  apa decode 16384 = TIME wall not VRAM (probe caps on serial chunked
  prefill; VRAM 3.4GiB with 8+ free) — wall_type recorded in JSONs.
- VERDICT (per P2's MARGINAL trigger): INT6 symmetric is the
  clean-quality operating point for the name-checker; INT4's loss was
  real and INT6 recovers it for +315 MiB resident.
- P4 DISPATCHED on INT6/fork config (product-realistic), with
  built-in bf16 re-run rule if any GRM gate fails (separate machinery
  from quant).

## 2026-07-20 ~11:2x — P4 COMPLETE (Opus seat): ALL GATES PASS ON INT6
- Gate A graft-vs-in-context @1.0: 0 top-1 flips, max|Δ| 1.0 =
  bf16-cache noise floor; cached-stream at engine floor; graft_seats
  persists cached decode.
- Gate B STATE save/restore: bit-identical (max|Δ|=0) over 23 steps
  with graft resident across boundary. Deviation fixed+disclosed:
  bf16 dtype recorded on save, re-cast on restore (numpy has no bf16).
- Gate C E4 20-turn: baseline 6/6 @ ~590 ctx; ARENA 6/6 @ max 209
  seats (65% residency saving), 6 routed swaps, evictions 20-41/probe;
  amnesia control 0/6. Router layer-0 |q·k| per GQA law.
- bf16 adjudication rule not triggered (nothing failed). Peaks
  2219-2427 MiB. FINDING: 1.7B graft surface = pure inheritance from
  shared GQAAttentionTC — zero adapter graft code needed; "the 4B
  pattern" is the attention module itself.
- PLAN 2db1c58 FULLY EXECUTED: P0/P1(+P1b)/P2/P3(kernel+measurement)/
  P4 all complete with receipts. Synthesis follows.
