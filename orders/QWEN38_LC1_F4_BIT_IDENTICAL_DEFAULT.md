# ORDER QWEN38-LC1-F4 — make the default path BIT-IDENTICAL (evidence attached)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` — edits/builds/test
runs AUTHORIZED. Continuation of LC1+F1+F2+F3 (boundaries unchanged). Lead
runs GPU gates.

## Evidence you must explain (all receipts in logs/)

1. `lc1_determinism_control.log`: pre-LC1 code (LC1 stack stashed) reproduces
   the 2026-08-19 baseline token_ids EXACTLY, all 3 prompts. The engine is
   deterministic. Any G-LC0 divergence is caused by an LC1-stack hunk.
2. `lc1_g0_baseline_r2.log` (tiled) and `lc1_g0_baseline_r3.log`
   (attn_path=legacy): IDENTICAL divergences (chat flips at tok 15 → 279,
   code at tok 33 → 11, factual exact). The F2 "legacy" path did NOT restore
   original behavior — the perturbation is in machinery SHARED by both
   attention paths.
3. `lc1_g0_ab_lmhead.log`: --lm-head-chunk-rows 31040 A/B was INCONCLUSIVE
   (cudaMalloc OOM mid-run under the new KV prealloc). lm_head chunking is
   not yet exonerated.

Prime suspects: KV preallocation (in-place writes / strided or blocked reads
vs the old contiguous concat-grown cache — different engine kernel dispatch
= different reduction order), lm_head chunk-rows default change, any other
decode-path change shared by both attention paths. The F1 zero-allocator
change is a lesser suspect (zeros are dtype-exact) but is not exempt.

## Requirement

**Default configuration (no flags) must execute the ORIGINAL mechanisms
end-to-end**: original concat-style KV handling, original lm_head chunk rows
(31040), original attention call — the pre-LC1 code path, byte-for-byte
behavior. All LC1 machinery (prealloc KV, small-chunk lm_head, tiled
attention, kv-int8, kv-host, force-alloc) engages ONLY when a long-context
flag/config demands it. LOAD_RESULT must name every mechanism active
(`kv_cache: concat|prealloc`, `attn_path`, chunk values).

Do not try to make the NEW machinery bit-identical — scope it out of the
default path instead. Bit-parity for LC modes is explicitly not claimed;
G-LC2/G-LC4 arbitrate those.

## Also

- Root-cause honestly: state WHICH hunk caused the flips and the mechanism,
  with file:line receipts, if your restoration confirms it. If you cannot
  attribute it definitively, say so — the restoration must still make
  G-LC0 pass by construction.
- CPU tests: assert default-config mechanism selection (concat KV, 31040,
  legacy attention) and flag-driven selection of each LC mechanism.

## Done — final message MUST contain verbatim
1. Diff hunks / file:line list of the restoration + dispatch conditions.
2. Root-cause statement (or explicit non-attribution).
3. Full CPU test suite output.
4. Exact lead command for the G-LC0 rerun.
5. Residuals stated plainly.
