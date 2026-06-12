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
| **LONGCTX** | **PASS** — 1200-tok prefill vs HF fp32: top-1 match (the external adjudication of band mask + ring trim); cross-boundary chunking (512 vs 384) top-1 consistent; greedy continuations identical. |
| **READY-TO-WORK** | **PASS** — corpus driver's own run_shard through the shim: **10/10 validated STORED_IN templates in 44s** (one request, cold). APA explicitly ON and logged. |

### Speed ladder (begins 2026-06-12; measured at the ready gate)

| Configuration | decode | cold prefill | warm (GRM mount) |
|---|---|---|---|
| v1: fused-tile/GEMV only, chunk 512 | 4.6 tok/s | 22 ms/tok (15.2s @ 691) | — |
| + GEMV 96KB shmem (engine `2d628d6`; ffn_down K=15360 onto GEMV) | 8.6 tok/s | 15.2s (unchanged, tile) | — |
| + free prefix minting (slice the request's own caches) + len-1 cap | 8.6 tok/s | 15.2s | **prefill 0.2s, mounted 690/691** |
| + concat projections (qkv / gate\|up: 7→4 GEMV launches), fused head norms, no-repeat_kv decode attention | 13.5 tok/s | — | — |
| + APA context threshold (blend only past 2048 tok; the per-token whole-cache requant at S=700 was pure overhead — gate forces threshold 0 to keep testing the machinery) | 15.0 tok/s | — | — |
| + native-dtype GEMV staging + fused `rope_apply` (engine) | 15.4 tok/s | — | — |
| + **fused causal_softmax at decode** (the composed softmax chain measured **580µs/layer** on an 11K-element tensor — 23ms/token!) + no-op ring-trim guard | **31.0 tok/s (32.3 ms/tok)** | — | — |

Forensic method note: the rope rung "should" have saved 14ms and saved
1.5 — the launch-cost model was wrong (dispatch measured 5.6µs, not
25µs). Micro-benchmarks of every decode op at REAL shapes then put the
missing 0.8ms/layer in softmax (580µs) and full-cache "trim" copies
(91µs) — both confirmed by the post-fix step time landing within 1ms
of the micro-bench prediction. Measure, don't model.

Remaining levers (diminishing): `down` GEMV at 179 GB/s (others
320-355); cat-append cache copies (~95µs/layer — needs an engine ring
op or capacity-doubling); cold prefill on the tile path (column-
blocked two-stage cuBLAS); CUDA graphs.

### Softmax investigation (Architect-directed, 2026-06-12)

Two latent bugs found, both invisible to every prior gate on every
model:

1. **Engine trailing-axis reductions ran one thread per OUTPUT**
   (533µs for an 11K-element max — 16 working threads on the GPU).
   Block-per-row fast path: 57×/51× (engine `eb20615`). Explains the
   580µs/layer composed softmax completely. All composed paths (qwen
   stack, training, APA internals) inherit the fix.
2. **APA blend causal mask was TOP-LEFT aligned for cached chunks** —
   and this bug had ALREADY BEEN FIXED ONCE (Architect's catch):
   `functional._causal_mask` carries the bottom-right fix with the
   measured anecdote from the KV-Graft era. The blend path kept its
   OWN private mask construction (`_CAUSAL_BLOCK_CACHE`,
   `triu(k=i+1)`, assumes L==S) and the fix never propagated. Every
   rectangular-with-cache APA call silently blinded queries to the
   most recent S−L keys; survived because ALL APA gates ran square
   cacheless shapes. Found by the refine ppl sweep (ppl 121 →
   11,000,000). Fix: the blend now SLICES the canonical
   `functional._causal_mask` (single implementation, equivalence
   verified incl. square shapes; private cache deleted); APA gate
   carries a permanent rect-with-cache check. TWO LESSONS
   (registered): (a) gate the regime the feature EXISTS FOR, not the
   shapes that are convenient; (b) a fixed bug isn't fixed while a
   duplicate of the buggy logic survives — fixes must hunt their
   copies.

**Refine-percentile ppl sweep** (wikitext raw, 6×2048-token windows,
1024 scored each, blend forced active, post-fix):

| setting | ppl | vs standard |
|---|---|---|
| standard | 121.74 | — |
| apa r0.15 | 119.53 | −1.8% (noise) |
| apa r0.10 | 117.30 | −3.6% (noise) |
| apa r0.05 | 121.29 | −0.4% (noise) |

The Architect's r0.10/r0.05 prior transfers: **refine cost is
unmeasurable at this sample size, down to r0.05.** (Absolute ppl ~121
on raw untemplated text is consistent with the -it model's hard
template binding; HF control pending. The APA conclusion is relative —
same harness, same data, only the attention path varies.)

All attention softmaxes also now route through the fused row kernel at
inference (`functional.py`): masks fold into scores, then
causal_softmax over L=1 rows — 8× at band-mask prefill shapes.

### OOM ceiling probe (Architect-directed, 2026-06-12) — tests/gemma4_oom_ceiling.py

Predictions registered BEFORE measurement; per-process ladders (an
OOM episode leaves the allocator fragmentation-pinned — ~1GB of
reserved segments survive gc + dual-pool flush; ONE MODE PER PROCESS
is the only rigorous ladder). 1K stages, PREFILL_CHUNK=256.

| mode | ceiling | decode at top rung |
|---|---|---|
| APA r0.10 (blend >2048) | **3,072 tok** | ~757 ms/tok |
| standard | **6,144 tok** | ~791 ms/tok |

**Standard runs 2× further — the registered MQA-expansion prediction
confirmed, the Architect's design expectation ("APA should run
longer") violated BY THE IMPLEMENTATION:** `_cublas_blend_attention`
opens with `_repeat_kv(k, 16)` (+ expanded kq and V) — written in the
GQA era (2-4× expansion), it materializes 16 copies of the global
cache on this MQA architecture. Registered fix: grouped-batch GEMMs
against UNEXPANDED K/V (the decode fast path's trick) — after which
APA's bounded score blocks beat standard's L×S prefill scores by
construction and the design contract holds.

Both modes share the bigger disease: decode is ~770-790 ms/tok at
EVERY rung past 1K (flat with S; 24× the 32ms at S=700) — the
cat-churn signature: at full rings every token allocates+copies
~670MB of cache tensors (cat append + trim slice ×40 layers) against
a near-full card. **Registered fix (THE long-context enabler): ring
buffers** — sliding layers get fixed (KV,1024,D) buffers + one
in-place row write/token + bias-masked invalid rows (zero-init keeps
softmax NaN-safe); globals get capacity-doubling append. Steady-state
decode cache copies → ZERO. Contract change: in-place writes end
pure-functional cache sharing — live caches are exclusively owned by
their decode loop; sharing requires explicit copy (GRM already copies
via the host round-trip; state gate re-registers the semantics).

Serving today is unaffected (corpus prompts ~700 tok sit below all of
this); these are the long-context tickets, in order: ring buffers →
blend de-expansion → re-probe (expect APA > standard) → V-side APA.

**Job economics at the ready gate: 10/10 validated templates in 16s
warm** (vs 38s on the Qwen3.5 stack — fewer output tokens per accepted
template; per-job this is already the fastest stack on the machine).
GRM mint discovery: pure-KV means the state-at-cp is EXACTLY a slice
of the request's own post-prefill caches (below the sliding window) —
minting is FREE; the re-prefill mint (which doubled request latency)
is only needed past 1023 tokens. Identical prompts mount via the
len(ids)-1 cap. Prefix states stored in HOST RAM (8 GPU-resident sets
would eat half the card), uploaded on mount (~30ms, invisible).

Remaining decode rungs (8.6 tok/s = 116ms/tok vs ~25ms bandwidth-ish
floor): profile first (qwen pattern — orchestration vs kernel split);
GEMV occupancy at 60KB shmem (1 block/SM); cold prefill via
column-blocked two-stage cuBLAS; CUDA graphs.

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
