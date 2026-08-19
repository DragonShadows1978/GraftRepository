# ORDER QWEN38-LC1-F2 — G-LC0 parity: legacy attention stays the default path

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` — edits/builds/test
runs AUTHORIZED. Continuation of QWEN38_LC1_LONG_CONTEXT.md + F1 (read both;
boundaries unchanged). Lead runs GPU gates.

## What happened

G-LC0 rerun after F1: no crash, but 2/3 demo generations diverge from the
2026-08-19 baseline (code @ tok 33, factual exact, chat @ tok 15 — receipts
in `logs/lc1_g0_baseline_r2.log`). Single greedy flips, consistent with the
LC1 online-softmax tiling changing reduction order — OR with a genuine
tiling defect. G-LC0's bar is registered and immutable: the DEFAULT path
must reproduce baseline token_ids exactly.

## Required design

1. **Restore the pre-LC1 single-shot attention as the default path**, taken
   whenever the device bf16 KV mode is active and the sequence fits without
   tiling. It must be the ORIGINAL code path — bit-identical arithmetic, not
   a reimplementation. The LC1 tiled/online-softmax path engages ONLY for
   configurations the legacy path cannot serve (kv-int8, kv-host, or
   sequences beyond the single-shot transient budget), and the LOAD_RESULT
   line must state which attention path is active (`attn_path: legacy|tiled`).
2. **CPU exactness proof for the tiled math**: a numpy reference test —
   same random inputs, fp32: online-softmax-tiled result vs standard softmax
   result, agreement ≤1e-6 rel, multiple tile counts (1, 2, 7 tiles) and
   shapes including tile-boundary remainders. If the current implementation
   has a real defect (e.g. rescale error), this test finds it; fix it.
3. **Teacher-forced margin diagnostic** (`--tf-margin` mode in the gate
   harness): run the tiled path teacher-forced over the three baseline
   token sequences; at every step report top1-agreement with baseline and
   the logit margin (top1−top2) wherever they disagree. REPORT-ONLY output
   (JSON lines) — this quantifies whether tiled divergence is near-tie
   noise or large-margin disagreement. No threshold; the lead adjudicates
   with the recall/coherence gates as the arbiters for the tiled path.

## Gates (unchanged + clarified scope)

- G-LC0 governs the DEFAULT path: with no flags, demo token_ids must be
  IDENTICAL to baseline. This is now achievable by construction (item 1).
- The tiled path is gated by G-LC2 recall + G-LC4 coherence (+ the margin
  diagnostic as evidence). No bit-parity claim is made for it — different
  reduction order is a different arithmetic mechanism; task gates arbitrate.

## Boundaries

Same as LC1/F1: no git, no subagents, no network, READ-ONLY engine/models/
cache, RED honesty, run CPU tests synchronously.

## Done — final message MUST contain verbatim

1. Diff hunks (or file:line list) restoring the legacy default + the
   dispatch condition between paths.
2. CPU exactness test output (item 2), including any defect it caught.
3. Updated full CPU test suite output.
4. Exact lead commands: G-LC0 rerun, tf-margin diagnostic, and unchanged
   descent/recall commands.
5. Residuals stated plainly.
