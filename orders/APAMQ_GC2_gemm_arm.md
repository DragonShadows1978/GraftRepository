# APAMQ-GC2 — Gemma gemm_apa Mode Wiring + FC Perplexity Arm

YOUR WRITABLE TARGET is this git worktree (GraftRepository branch
`apamq-gc2`, base 71abd15 = merged main) — `core/`, `scripts/`,
`tests/` AUTHORIZED. Run first, report after; a registered order IS
the permission.

HARD BOUNDARIES: canonical repos READ-ONLY. The CANONICAL
Project-Tensor engine (already rebuilt at e1e52ba) now exports
`tc.apa_gemm_selective_attention(q, k, v, scale, zthr, is_causal, *,
Lq=0, row0=0, window=0, k_codes=None, k_scales=None)` and
`tc.apa_gemm_selective_quantize_k(k)` — use the canonical import, no
TENSOR_CUDA_ROOT juggling. No git, no subagents, no network, models
read-only. Sandbox has NO GPU: CPU checks only; the lead runs the arm.
RED honesty.

## Context (pre-nailed)

APAMQ ledger (Project-Tensor docs/APA_MQA_ROOTCAUSE_LEDGER.md,
2026-08-14 entries): gemm_apa is the merged SPEED mode (15/15 engine
gates, 1.30–1.65× cuBLAS, decode 1.41 ms, fp32 scores, int8 internal
quantizers). Pre-registered quality gate (SB1 order + ledger G-C):
**the gemm arm's relative ppl must be ≤ the apa_blend arm + 0.25%.**
Existing receipts in artifacts/apamq_fc/ (standard / apa_blend /
apa_fused / apa_int4) were produced by scripts/apamq_fc_ppl.py at
protocol fingerprint 2b8f4d2d…; the summary REJECTS mismatched
fingerprints.

## Tasks

1. **Port wiring** (`core/gemma4_tc.py`): mode flag `GEMMA4_APA_GEMM=1`
   (default OFF, capability-guarded on
   `hasattr(tc, "apa_gemm_selective_attention")`). When active and APA
   is engaged on a global layer: call the gemm entry at BOTH the
   prefill chunk site and the decode site, with correct bottom-right
   parameters (`is_causal=True, Lq=<full L>, row0=<chunk start>` at
   prefill; decode L=1 mirrors the existing fused-decode call's
   causal convention) on the UNEXPANDED MQA cache. When active: no
   kqb ring is allocated/grown/populated and no _quantize_keys calls
   happen (the engine quantizes internally). Optional but preferred:
   cache K codes per ring generation via
   apa_gemm_selective_quantize_k + the `k_codes/k_scales` kwargs,
   invalidating on append (state the invalidation rule you implement);
   if that is not safely doable this round, per-call quantize is
   acceptable — say which you shipped.
2. **FC harness arm** (`scripts/apamq_fc_ppl.py`): add arm `apa_gemm`
   — env GEMMA4_APA_GEMM=1, engagement receipts mirroring the other
   arms: 409,600 global-layer L=1 decode calls censused as gemm (add
   the census counter for the gemm branch), blend/fused/int4 counts
   zero, ring-ABSENCE assertion (like the int4 arm), same 25-doc
   protocol. **The protocol fingerprint must remain 2b8f4d2d… so the
   four existing receipts stay valid for the summary** — if your arm
   addition unavoidably changes the fingerprint, STOP and report why
   rather than silently invalidating last night's receipts.
3. **Gates/CPU checks:** py_compile all touched files; extend the
   cpu-check in tests/gemma4_apa_decode_fused.py or a sibling with a
   gemm flag-matrix row (capability-guarded skip); pytest skip-safe.

## Done

Final message verbatim: files changed with the wiring diff
summarized (both call sites), the K-code caching decision +
invalidation rule (or per-call fallback rationale), fingerprint
status (unchanged / changed+why), CPU check outputs, the exact lead
commands (arm run + summary), deviations. No GPU numbers.
