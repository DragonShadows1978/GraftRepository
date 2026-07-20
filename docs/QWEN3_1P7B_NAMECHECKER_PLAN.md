# QWEN3-1.7B NAME-CHECKER — IMPLEMENTATION PLAN

**Status: PLAN — IMMUTABLE after initial commit** (house rules). Receipts
in `QWEN3_1P7B_NAMECHECKER_LEDGER.md`; meaning in the synthesis, written
at close. Scope: the Qwen3-1.7B model track ONLY (per David, 2026-07-20)
— the MiniCPM3-4B dialog track, serving shim, and Frontier integration
are separate future plans.

Context: [[project-frontier-npc-llm]] — Qwen3-1.7B is the decided
name-checker model (dense GQA 16q/8kv hd128, RoPE θ1e6, qk-norm, tied
embeddings, vocab 151936, 28 layers, config max_position 40960, Apache
2.0). Target machine: MilleniumFalcon, RTX 4070 SUPER 12GB.

## Roles (David's directive, 2026-07-20)

- **Lead (Fable):** plan, orders, verification, ledger, commits.
- **Kernel work (INT6, any new CUDA): Codex Sol** — raw Bash shim,
  `-m gpt-5.6-sol`, `-c model_reasoning_effort=max`, dispatched from
  inside the target repo. Works in a dedicated **fork checkout of
  Project-Tensor** (branch `int6-weights`, separate worktree dir) —
  merge to canonical later, after lead verification.
- **All other code (adapter, harnesses, GRM): Opus 4.8 seats at MAX
  effort** (Agent tool, order-file prompts, effort max, no-subagents
  rail).

## Measurement protocol (fixed now, applies to every phase)

- **OOM / context ceiling:** per-rung ladder, ONE PROCESS PER PROBE
  (fragmentation fakes OOMs across contexts — Gemma law). Separate
  prefill-ceiling and decode-ceiling probes. Binary-search context
  length to the wall; report last-solid and first-OOM.
- **Peak memory:** 1s `nvidia-smi` poller per run (A0 law: poller
  peaks, not framework counters) + framework-reported allocator peak
  where available. Report BOTH total and peak fill at every stage.
- **Perplexity:** wikitext-2-raw test split, sliding-window protocol,
  stride 512, identical tokenization across all stages. Windows: 2K
  control + 4K / 8K / 16K / 32K (as fits). **APA runs must ENGAGE APA
  during scoring** (June lesson: a 2048-window sweep never engaged APA
  and falsely read "within noise") — scoring windows must exceed
  apa_min_context and engagement must be asserted/logged per run.
- **Matched-reference law (Trinity):** every parity comparison names
  its exact reference capture; no cross-reference drift.
- **Evidence classes:** OOM/peak = memory shape; ppl = model-bound
  measurement; parity margins = port correctness. No capability claims
  from any of these.
- Runs bounded ≤10 min each (power-safety program rule); single-GPU.

## Phases

### P0 — Baseline: stock HF transformers
1. Download `Qwen/Qwen3-1.7B` (bf16 safetensors) to the HF cache.
2. HF transformers, bf16, SDPA default: OOM ladder (prefill + decode),
   peak-fill trace, out-of-the-box context ceiling on the 12GB card.
3. Perplexity per protocol at every window that fits.
4. Deliverable: baseline table (ctx ceiling, ppl×window, peak×ctx).
   This run also mints the **GT captures** (logits on fixed prompts +
   ppl corpus tokenization) that all later parity gates reference.

### P1 — tensor_cuda adapter + APA full-precision
1. Adapter `core/qwen3_1p7 support` via config-swap of `core/qwen3_tc.py`
   (Qwen3-4B). Known deltas to handle, not discover: **tied embeddings**
   (lm_head = embedding; the 4B adapter path is untied — head handling
   must be explicit, host-or-resident decision recorded), vocab 151936,
   28L/16Q/8KV/hd128, qk-norm per-head (same as 4B).
2. Parity gate vs P0 GT: margin protocol (top-1 flips only at near-tie
   margins; exact-greedy is the wrong bar across stacks).
3. APA ON at **refine r0.15** (the registered full-refine operating
   point for this plan), bf16 weights: OOM ladder, ppl per protocol
   (engagement asserted), peak fill. APA-off control at the same
   windows for the delta.
4. Deliverable: P1 table beside P0 (ceiling / ppl / peak, APA on+off).

### P2 — INT4 weights
1. Quantize with the house INT4 stack (David's; group/symmetric config
   recorded in ledger at execution). No GGUF import — self-quantized.
2. Parity margin gate vs P0 GT (INT4 flips only near-ties — the
   qwen3-4B PARITY precedent).
3. Full measurement battery, APA r0.15 engaged: OOM ladder, ppl, peak.
4. Deliverable: P2 table beside P0/P1. **Decision point (David):**
   INT4 ppl delta acceptable → P3 optional; INT4 marginal → P3 runs.

### P3 — INT6 weights (conditional, David's call at the P2 gate)
1. tensor_cuda has no INT6 weight path today → **Sol seat implements
   INT6** (pack/unpack, dequant, GEMV/tile paths as needed) in the
   Project-Tensor fork (branch `int6-weights`, own worktree).
   Gates before use: dequant bit-exactness on synthetic grids, GEMV
   parity vs dequant-reference matmul (~1e-3 class rel), existing
   engine test suite green in the fork.
2. Quantize 1.7B to INT6; parity margin gate vs GT; full battery, APA
   r0.15: OOM ladder, ppl, peak.
3. Deliverable: P3 table beside P0-P2. Merge decision on the fork =
   David's, after lead verification of the kernel receipts.

### P4 — GRM on the 1.7B (proof, not production)
1. GQA-dialect arena + repository on the 1.7B (the current, hardened
   GQA stack — adapter hooks per the Qwen3-4B pattern: `_capture`,
   prefill-only injection, `graft_seats`/`live_shift`).
2. Multi-turn battery: E4-class conversation gate (20 turns, planted
   facts, routed mounts, session save/restore bit-identical), on the
   best weight format standing after P2/P3.
3. Explicitly out of scope: production name-checker integration,
   serving shim, moderation-ladder code. This phase proves the
   substrate; the product plan comes later.

## Memory accounting (every phase, same table)

| stage | weights resident | KV @ctx | activations/overhead | peak fill | ceiling |

reported for: HF-bf16 / tc-bf16 / tc-bf16+APA / tc-INT4+APA
(/ tc-INT6+APA if P3 runs), at each ladder rung.

## Registered decision points (David's, not the lead's)

- P2→P3: whether INT4 results warrant trying INT6.
- P3 merge: fork → canonical Project-Tensor.
- Post-P4: whether GRM ships in the name-checker at all ("won't be
  needed" is the working assumption).

## Deliverables at close

Cross-stage comparison table (context ceiling, ppl per window, peak
fill, weights-resident), parity receipts per stage, ledger complete,
synthesis with the INT4-vs-INT6-vs-bf16 recommendation grounded in the
measured deltas.
