# ORDER QWEN38-A2 — lm_head OOM fix + generation gates rerun

YOUR WRITABLE TARGET is /mnt/ForgeRealm/GraftRepository — AUTHORIZED.
Also writable: /tmp. READ-ONLY: /mnt/ForgeRealm/Project-Tensor,
/mnt/ForgeRealm/models/Qwen3.8-27B, installed transformers. No git
writes (read-only git fine). No subagents.

## Context (receipts from A1's lead gate run)

Your A1 delivery: G0 PASS 5.9e-7; G1 PASS on the real GPU — full 27B
INT3 resident at 9.94 GiB computed. G3 CRASHED before generating:

    core/mistral7b_tc.py:144 tc.intn_linear(...)  # lm_head
    RuntimeError: cudaMalloc failed: out of memory

Cause: two-stage `intn_linear` dequantizes the FULL lm_head
(248320×5120 fp16 = 2.54 GiB transient) on top of 9.94 GiB resident on
a 12.28 GiB card. Your fused-decode fallback (correct call — canonical
fused-INT3 parity defect rel 2.25e-3 stands confirmed) is what routes
lm_head through two-stage. The lm_head is the ONLY whale: the largest
body transient is ~214 MiB (6144×17408).

Lead gate bookkeeping, for your receipt only (no action): G2 as
registered (rel_l2 1e-3) FAILED at 5.72e-3 and remains FAILED — the
threshold was mis-derived. Successor gate G2b is registered BELOW,
before any rerun, with provenance: the proven 9B port's parity envelope
was layer cosines 0.90–0.997 with exact teacher-forced greedy match as
the primary gate (docs/QWEN35_PORT_LEDGER.md).

## Task

1. **Fix the lm_head path** (in THIS repo, engine untouched):
   a. Greedy decode and prefill need logits for the LAST position
      only — slice the final hidden state to (B, 1, H) before lm_head.
   b. Row-chunk the lm_head matmul: run the packed lm_head in R-row
      chunks (default ~31040 rows = 8 chunks, configurable), each
      chunk's dequant transient ~318 MiB, concatenating logits. Chunk
      the PACKED representation — never materialize full fp16 rows for
      more than one chunk at a time.
   Both (a) and (b): (a) makes decode cheap, (b) keeps any
   full-logits use (perplexity later) viable.
2. **Peak-VRAM receipt**: sample `nvidia-smi` (or
   cudaMemGetInfo if exposed) at load, after prefill, and during
   decode; include peaks in the receipt.
3. **Rerun gates** (GPU is expected available to you this run; if not,
   PENDING-LEAD with exact commands):
   - G1 load-only (should be unchanged).
   - G2b (REGISTERED HERE): per-layer on-device vs CPU-reference
     cosine ≥ 0.999 for layers {0, 3}; report rel_l2 alongside.
   - G3: greedy, 3 prompts (code, factual, chat), 64 tokens each,
     chat template applied; print raw outputs VERBATIM. No NaN, no
     collapse. Coherence is the primary arch verdict.
   - G3-DIAG (run ONLY if G3 output is degraded/incoherent): flip the
     attention output gate to sigmoid (config override) and regenerate
     prompt 1; print both outputs side by side. This adjudicates the
     swish-vs-sigmoid judgment against the stale transformers source.
   - G4: decode tok/s + prefill seconds for a 128-token prompt.
4. GPU discipline: every GPU-touching command under
   `flock -w 3600 /tmp/forge-gpu.lock`, `timeout 3000`.

## Rails

RED honesty; verbatim errors. INT3 output quality is characterization,
never a claim. Keep RAM < 50 GiB. No monitor-idling.

## Done — final message must contain, verbatim

- Diff summary of the lm_head fix (files, approach, chunk size).
- Peak VRAM numbers at load / prefill / decode.
- G1, G2b, G3 (raw outputs verbatim), G4 results; G3-DIAG if run.
- Any deviation.
