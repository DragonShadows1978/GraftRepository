# Gemma 4 12B Unified — tensor_cuda port ledger

**Started:** 2026-06-12 · **Hardware:** RTX 3070 8GB ·
**Weights:** `/mnt/ForgeRealm/models/gemma-4-12B-it` (bf16, 22.3GB, Apache 2.0)
**Directive:** straight to APA+GRM — no baseline benchmarking. PyTorch
fp32 stays as the port-correctness oracle only (the established law).

## Why this model

12B dense, 48L = 40× sliding-window(1024) + 8× global attention — and the
8 growing layers are **MQA with a single 512-dim head where K and V share
one projection**. Growing cache ≈ 16KB/token raw (2KB/token/global-layer
× 8), sliding layers hard-cap at 320MB. **Pure-KV model: no recurrent
state — GRM seats compose (the thing DeltaNet states can't do); full
KV-Graft arena semantics apply.**

## Shape sheet (config + checkpoint verified)

| | |
|---|---|
| hidden / FFN | 3840 / 15360, gelu_pytorch_tanh gated MLP |
| layers | 48; global at i%6==5 (5,11,17,23,29,35,41,47), final global |
| sliding attn | 16q/8kv, head_dim 256, window 1024, RoPE θ=1e4 full-dim |
| global attn | 16q/**1kv**, head_dim **512**, K=V shared proj, p-RoPE θ=1e6 factor 0.25 |
| vocab | 262144, **tied** embeddings (no lm_head in ckpt), pad 0 bos 2 eos {1,106,50} |
| weights | `model.language_model.*`, 666 tensors; vision_embedder/embed_audio/embed_vision skipped (text-only) |
| params | 11.96B total; INT4 resident est. ~6.3GB (incl. tied lm_head); embed gather host-side fp32 (~4GB RAM) |

## REGISTERED TRAPS — verified in modeling_gemma4_unified.py (transformers 5.12)

1. **RMSNorm is PLAIN w** — `normed * weight`, init ones. NOT Gemma 3's
   (1+w). **No bake anywhere.** Applies to all norms incl. q_norm/k_norm.
2. **`layer_scalar` is load-bearing**: per-layer ckpt scalar multiplies the
   ENTIRE post-block hidden state (residual included) at every block exit.
   Values 0.0045 (L11!) … 0.886. Omit it → garbage. fp32 multiply.
3. **Attention scaling = 1.0.** No 1/√d — qk-norm does the conditioning.
4. **Embed scale** = √3840 ≈ 61.9677 multiplies embedding output. HF
   casts it to weight dtype (bf16 → 62.0); fp32 GT uses 61.9677 — use the
   fp32 value in our host-side gather (diff is sub-INT4-noise).
5. **Global K=V**: one k_proj (512 out), v_proj absent in ckpt.
   K path: k_norm (scaled RMSNorm) → p-RoPE. V path: v_norm
   (**scale-free** RMSNorm, no weight in ckpt), NO RoPE. Cache stores both
   processed tensors (simple route, 2KB/tok/layer).
6. **p-RoPE**: inv_freq = [θ^(−2i/512) for i<64] ++ zeros(192) → 256
   freqs, full-width 512 rotary apply; zero-freq dims are identity
   (cos 1, sin 0). rotate_half convention, fp32 tables.
7. Sliding RoPE: standard full 256-dim, θ=1e4.
8. **Sliding mask**: kv ∈ (q−1024, q] — includes self, 1024 keys max.
   Cache ring keeps last 1024 per sliding layer.
9. **Sandwich norms** (Gemma-2 style): input_ln → attn → post_attn_ln →
   +residual; pre_ffw_ln → mlp → post_ffw_ln → +residual.
10. **Final logit softcap 30.0**: logits = 30·tanh(logits/30), after tied
    lm_head. No attention softcap (param exists, never passed).
11. RMSNorm formula: x·(mean(x²)+eps)^−0.5 in fp32 then ·w, cast back;
    eps=1e-6 INSIDE the rsqrt.
12. RoPE applied in (B,S,H,D) layout pre-transpose (unsqueeze_dim=2) —
    same math as our post-transpose path, keep our layout.
13. qk-norm present → **bulk-bits law predicts bulk_bits=4** (6th
    architecture test). q_norm/k_norm per-head-dim (512 global / 256
    sliding), scaled, plain-w.
14. Inert machinery (config-disabled): num_kv_shared_layers=0, MoE off,
    double-wide MLP off, use_bidirectional_attention="vision" (text =
    causal everywhere). Don't port.
15. Chat template (thinking OFF is the default render):
    `<bos><|turn>user\n…<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`
    — generation continues directly after the pre-filled empty thought
    block. `<|think|>` (id 98) in system prompt enables thinking.
    eos set {1, 106, 50}; suppress {258882, 258883}.
16. Engine has exact `gelu` (tanh approx, √(2/π), 0.044715) — matches
    gelu_pytorch_tanh. No new kernels required for this port.

## Gate plan (same law as Qwen3.5)

- **GT oracle**: tests/gemma4_gt.py — fp32 CPU via transformers 5.12,
  3 prompts: full per-layer hiddens, final logits, greedy 16, k/v probes
  at L0 (sliding) and L5 (global). Adjudication is margin-based
  (INT4-vs-fp32: flips must sit at GT near-ties).
- **State gate**: save/restore hybrid… no, *pure-KV* caches —
  bit-identical continuation post-prefill and mid-decode.
- **APA gate**: apa_selective on the 8 global layers (cuBLAS blend,
  head_dim 512 > fused cap 128; single KV head → trivial codebook
  granularity), bulk4/refine 0.15 predicted; flips vs standard ≈ 0.
- **Ready gate**: the consumer's own loop through the serving shim — the
  only valid done signal.

## Results

### Forensics: bf16-origin INT4-g128 FAILS on this model (2026-06-12)

First parity run vs fp32 GT: late-layer collapse (trail: L41 cos 0.71,
L47 0.28, final-norm 0.67; worst flip costs 8.7-9.2 logits). Bisection
chain, each step decisive:

1. **GT hidden semantics adjudicated from checkpoint math**: hidden[0]
   = scaled embeds; hidden[48] reproduces GT logits through the tied
   head with NO extra norm → it is POST-final-norm. The naive L47 probe
   compared pre-norm vs post-norm (artifact); entries 1..47 are layer
   outputs as assumed.
2. **Graph definition EXACT**: GT's own L46 output pushed through this
   port's L47 graph in fp64 numpy with exact weights reproduces GT's
   hidden[48] at cos 1.000000 (gemma4_l47_probe.py).
3. **Engine execution EXACT**: stage-by-stage L47 attention on the
   engine vs an fp64 reference with the same INT4-dequantized weights:
   1.00000 at every stage (gemma4_l47_engine_bisect.py).
4. **bf16 exonerated**: HF-bf16 vs HF-fp32 stays ≥0.995 cos all 48
   layers, zero top-1 flips, last|d| 1.45 (gemma4_gt_bf16.py).
5. **SMOKING GUN**: full 48-layer host simulation in fp32 with
   quantize→dequantized g128 weights reproduces the engine trail
   layer-for-layer (L41 0.707/0.706, L47 0.284/0.282, final
   0.676/0.675; gemma4_int4_sim.py). **The entire divergence is
   post-hoc INT4-g128 quantization noise on Gemma 4's weights.**

Notes: damage concentrates at GLOBAL layers (MQA 1-KV-head — all 16
query heads share the noisy key/value, no cross-head averaging) and is
amplified by massive-activation outlier dims (hidden[24] has a 220:1
outlier; cosine on raw hiddens is dominated by them — the 0.28→0.67
"recovery" through the final norm is those dims being down-weighted).
**This is why Google ships a QAT release for Gemma 4** — the Architect
flagged the QAT repo before the evidence landed.

Requant check (gemma4_qat_requant_sim.py): QAT weights re-quantized at
g128 wander less (final cos 0.84, top1 held on prompt 0) but lose much
of the QAT benefit → exact import required.

### QAT pivot: exact q4_0 import (the production path)

- Engine: **symmetric-8 INT4 variant** (Project-Tensor, commit after
  0eb2ca9): empty zeros tensor → kernels derive z = −8·s in-register
  across all three int4 paths. The q4_0 grid w = s·(q−8) is
  represented BIT-EXACTLY at group_size 32 with no zeros tensor
  (~0.75GB VRAM saved). Engine gate: dequant max|d| 0.0 on the grid,
  GEMV/tile/two-stage rel err ~3e-7, asymmetric regression max|d| 0.0;
  full suite 62/62 green.
- Port: `Q40LinearTC` + `_q40_repack` (GGUF j/j+16 nibble interleave →
  engine even/odd pairs; verified byte-exact vs gguf dequantize on
  attn/ffn shapes). `load_weights_qat` = default. Only post-hoc tensor:
  the tied head (token_embd ships Q6_K) requantized at g32 asym.
- GGUF facts: norms PLAIN w (means match safetensors — no +1 shift);
  `rope_freqs` = freq_factors [1.0]×64 ++ [1e30]×192 — llama.cpp's
  encoding of p-RoPE, independently confirming ours; layer_output_scale
  values differ from the bf16 release (QAT is a distinct training).
- GT: QAT GGUF dequantized → fp32 HF checkpoint (2 shards — a single
  48GB save_file OOMs 62GB RAM) → transformers is the oracle for the
  SAME weights the engine serves (tests/gemma4_qat_to_hf.py).

### Gates vs QAT GT (engine = exact q4_0 import, 2026-06-12)

| Gate | Result |
|---|---|
| **PARITY** | **PASS** — top-1 75/80; worst flip cost 1.081 (true near-ties; was 8.7-9.2 pre-QAT); L0/L5 caches 0.9999-1.0000; final-norm cos 0.965-0.9999. Engine-vs-QAT-GT distance = bf16 compute noise, as predicted. |
| **STATE** | **PASS** — pure-KV save/restore bit-identical, post-prefill and mid-decode. |
| **APA** | **PASS** — apa_selective bulk4/refine 0.15 on the 8 globals: 4/80 flips, worst 0.25 logits. **Bulk-bits law (qk-norm → 4): 6th architecture.** |
| LONGCTX | (running — band mask + ring trim vs HF fp32 @ 1200 tok) |

Fixes en route (both registered): (a) the parity gate's last-layer
probe now compares final-normed output vs GT hidden[48] (pre-vs-post
norm comparison collapses by construction under massive activations);
(b) **auto-chunked prefill** (`PREFILL_CHUNK=512` in `__call__`) — the
QAT body is ~6.8GB resident and single-shot long prefills OOM on
transients; chunking bounds them. Trap within the fix:
`last_token_only` must pass through to EVERY chunk — computing full
chunk logits to discard them is a 0.8GB transient per chunk (OOMed
once before the probe was caught). Operational lesson re-learned: ONE
GPU job at a time — a second gate chain launched while the first still
held the card produced two mutual-OOM casualties and one survivor.
