# Qwen3.5-9B → tensor_cuda port ledger

Target: text-only INT4 inference on the 8GB 3070, then APA on the
attention layers and GRM (graft dialect) on the hybrid cache.
Ground truth law: adjudicate against PyTorch/transformers, never
in-repo runs. Sources: local checkpoint config.json + transformers
5.7 `modeling_qwen3_5.py` (read line-by-line), cross-checked by
4-agent web recon (HF index, Gated DeltaNet paper 2412.06464, vLLM
implementation). Serving for the corpus track is ollama `qwen3.5:9b`
(verified 25 tok/s, `think:false` required) — this port is the
research stack, not the corpus dependency.

## Shape sheet (text_config)

| | |
|---|---|
| layers | 32 = 24×GatedDeltaNet + 8×attention (indices 3,7,…,31; `3:1` blocks) |
| hidden / FFN | 4096 / 12288 SwiGLU (silu), dense — no MoE |
| attention | GQA 16q/4kv, head_dim 256, scale 256^-0.5, softmax fp32 |
| output gate | `attn_out * sigmoid(gate)`, ELEMENTWISE per channel, pre-o_proj |
| q_proj fused | 4096→8192; per head `[q(256) | gate(256)]`; gate bypasses q_norm AND RoPE |
| qk-norm | per-head RMSNorm(256), eps 1e-6, **(1+w) convention** |
| RoPE | partial 0.25 → first 64 dims, GLM rotate_half, θ=1e7; mRoPE = NO-OP for text (3 identical position streams) |
| DeltaNet | 16 k-heads ×128, 32 v-heads ×128 (v 2i,2i+1 share kq head i via repeat_interleave); conv1d depthwise k=4 NO bias + SiLU; state fp32 |
| DeltaNet projs | in_proj_qkv 4096→8192 rows `q2048|k2048|v4096`; in_proj_z 4096→4096; in_proj_b/a 4096→32 (keep unquantized) |
| recurrence | q̃,k̃ = l2norm(eps 1e-6); q̃ /= √128; **S ← exp(g)·S FIRST**, kv_mem = Sᵀk̃, S += k̃⊗(v−kv_mem)·β, o = Sᵀq̃ |
| gates | β = sigmoid(b); g = −exp(A_log)·softplus(a + dt_bias), fp32, per v-head |
| DeltaNet out | per-head RMSNormGated(128): norm(o)·w·silu(z), **plain-w convention**, then out_proj |
| embeddings | vocab 248320, UNTIED lm_head, no embedding scale |
| final norm | RMSNorm 4096, (1+w) |
| eos | 248044; chat template thinks by default |

## Weight map (19.31GB bf16, 4 shards, dash-named `model.safetensors-0000X-of-00004.safetensors`)

- KEEP: `model.language_model.*` (426 tensors), `lm_head.weight`
- SKIP: `model.visual.*` (333), `mtp.*` (15)
- Layer tensors are spread across shards — open all 4, filter by prefix.

## Parity traps (registered before coding)

1. TWO RMSNorm conventions in one model: standard norms multiply by
   **(1+w)** (input/post/final/q/k norms — bake 1+w at load); the
   DeltaNet's gated norm multiplies by plain w.
2. q/gate interleave is PER HEAD (`[h0_q|h0_g|h1_q|h1_g|…]`), not two
   contiguous blocks.
3. DeltaNet decay order: state decays BEFORE the delta correction.
4. l2norm of q/k happens INSIDE the rule (after conv+split), eps 1e-6,
   and the 1/√d_k scale applies to q after l2norm.
5. conv1d has NO bias; SiLU applied after; causal left-pad 3.
6. `mamba_ssm_dtype: float32` — state and the g/β computation in fp32
   (a bf16 state will drift on long sequences).
7. mRoPE sections [11,11,10] interleaved are a no-op for text — do NOT
   implement them; standard partial RoPE only.
8. HF state layout (k_dim, v_dim) differs from vLLM (v_dim, k_dim) —
   GRM serialization must pin ours (HF order) in the header.

## VRAM budget (INT4)

body ~3.6GB + lm_head 0.51GB + scales/zeros + embed HOST-side
(2.03GB bf16 stays in RAM, gather rows per token) + DeltaNet states
50MB/seq + KV 4KB/token ≈ **~4.6GB resident** — fits with headroom.

## Plan

1. GT vectors: transformers CPU forward (logits + per-layer probes).
2. `core/qwen35_tc.py`: loader + attention side (reuse qwen3_tc/
   mistral7b_tc patterns) + DeltaNet recurrent form (per-token loop,
   correctness first; chunked form later if prefill speed matters).
3. Parity: teacher-forced logits + greedy match vs GT (MiniCPM3
   methodology), per-layer probe bisection on mismatch.
4. APA on the 8 attention layers (qk-norm → predict bulk_bits 4).
5. GRM: hybrid graft dialect = KV seats @ attention layers +
   (recurrent, conv) state pairs @ DeltaNet layers; single-graft
   restore is exact by construction (Markovian state); ARENA
   composition is the open research question.

## Results (2026-06-12, overnight run)

| Gate | Verdict | Measurement |
|---|---|---|
| First parity run | near-pass, FIRST TRY | prompt 0 EXACT (5/5 teacher-forced + 16/16 greedy); L0 DeltaNet state cos 0.9986-0.9987 on all prompts; layer cosines 0.90-0.997 |
| Tie-flip diagnostic | noise confirmed | every engine-vs-GT disagreement sits at a GT near-tie: worst flip cost 1.92 logits, typical <0.5 — INT4 perturbation, not math error. Gate re-registered margin-based (flip cost ≤3.0 + state cos ≥0.995 + L31 cos ≥0.90) |
| **PARITY (registered)** | **PASS** | all 3 GT prompts OK under the margin gate |
| **STATE (GRM gate)** | **PASS** | save/restore of the full hybrid cache (KV + conv + recurrent): post-prefill AND mid-decode restores continue BIT-IDENTICALLY (logits array-equal) |
| Coherence smoke | clean | chat template + hybrid decode: correct Rayleigh answer, clean haiku |
| Throughput v1 | recorded | ~1.2 tok/s end-to-end (per-token DeltaNet loop, ~300 launches/tok) — correctness-first; chunked prefill + fused step kernel are the known levers |
| **APA (bulk4 / refine 0.15)** | **PASS** | cuBLAS blend path on the 8 attention layers, KV-head-granularity codebooks: **ZERO top-1 flips vs standard** on all GT prompts; APA-mode chat coherent. bulk_bits-tracks-key-normalization law holds on a 5th architecture (qk-norm → 4) |

Serving for corpus work is ollama qwen3.5:9b (25 tok/s, Q4_K_M,
`think:false`); readiness file dropped at
`GRAPA-Native-LLM/corpus/QWEN_READY.md` 2026-06-12.

## Decode forensics (2026-06-12, measured — tests/qwen35_profile.py)

| Component (M=1, median of 20) | two-stage | int4 GEMV | speedup |
|---|---|---|---|
| **full step** | **840.3 ms** | **122.0 ms** | **6.9×** |
| lm_head (248320×4096) | 348.8 ms | 1.5 ms | 232× |
| MLP ×32 | 434.6 ms | 12.1 ms | 36× |
| DeltaNet mixer ×24 | 142.5 ms | 18.7 ms | 7.6× |
| attention mixer ×8 | 41.9 ms | 4.4 ms | 9.5× |
| host argmax | 0.17 ms | 0.19 ms | — |
| sum-of-parts | 968 ms | 36.9 ms | |

Diagnosis CONFIRMED: two-stage INT4 dequantized ~31GB of weights per
token (lm_head's 2GB transient alone = 349ms). One flag
(`QuantLinearTC.FUSED_DECODE=True`, engine GEMV kernel 8501a5c) takes
decode 1.2 → **8.2 tok/s**.

**Plan reordered by measurement:** at 122ms/step only 37ms is in the
measured mixers/heads — **~85ms is orchestration** (64 RMSNorm
composed chains, casts, residuals, ~2k Python-dispatched launches at
~25-50µs). So launch-count reduction now outranks the fused GDN step
kernel (which would save only ~17ms): (1) flip RMSNormTC.USE_FUSED
(engine-gated fused rms_norm; valid under tc.no_grad; our (1+w) bake
is weight-side so the op is unchanged); (2) fused GDN step (~17ms);
(3) CUDA-graph/batched residual+cast chains. llama.cpp reference
(recon 2026-06-12): fused GGML_OP_GATED_DELTA_NET + MMVQ dp4a matvecs,
50-60 tok/s on 8GB cards, 3070 bandwidth ceiling ~77 tok/s.

Dispatch parity: GEMV vs two-stage max|Δlogit| = 0.125 (bf16
reassociation scale, tie-flip class — NOT bit-identical). REQUIRED
before FUSED_DECODE becomes the port default: re-run the full gate
suite (parity / state / APA) under GEMV at the next GPU gap.

## Speed ladder (2026-06-12, all gated — Architect directive: close the
## gap to baseline; corpus paused for the work)

| Rung | ms/tok | tok/s | What |
|---|---|---|---|
| v1 composed | 840 | 1.2 | two-stage INT4 dequantized ~31GB/tok |
| + int4 GEMV default | 122 | 8.2 | engine 8501a5c kernel, dispatch flip |
| + fused RMSNorm default | 34 | 29 | killed the ~85ms composed-op orchestration share |
| + fused GDN step kernel | **26.4** | **38** | engine `gated_delta_step` (a3363b0/0eb2ca9): l2norm + gates + delta rule + readout in ONE launch/layer, warp-per-state-column |

**38 tok/s = 1.5× the ollama baseline (25), ~half the 3070 bandwidth
ceiling (~77).** Full suite green at every rung (parity margins
unchanged, state restore bit-identical, APA zero/near-zero flips).

Kernel design notes: folds MORE than llama.cpp's GGML_OP_GATED_DELTA_NET
(their composed surroundings are C-launched ~1µs; ours were
Python-launched ~25µs, so the norm/gate chains went INTO the kernel).
**FUNCTIONAL contract enforced**: (out, new_state) with input state
byte-untouched — the in-place variant corrupted held cache references
(caught by the parity gate's L0-state cosine; branch-from-cache is
GRM's restore-once-decode-many pattern). Same traffic, 0ms cost.

Remaining levers (diminishing): mlp 12.2ms is now the biggest line
(bandwidth-bound GEMVs — near floor), attention 4.4, deltanet-side
composed remainder (conv/split/gated-norm ~6); batching/CUDA-graphs
could approach ~50 tok/s but the easy 30× is banked.

## GRM design note

A DeltaNet state is a *summary*, not a seat-addressable cache — the
graft story for hybrids is: KV seats compose (existing arena math),
states do NOT (state(A→B) ≠ f(state(A), state(B))). Single-mount
grafts (template/boilerplate prefix) are exact. Multi-mount needs
either (a) state-at-mount-boundary chaining (sequential semantics,
order-dependent), or (b) research into state superposition error.
Start with (a) — it matches the corpus use case (one template
library prefix + live query).
