# Gemma 4 12B Unified — tensor_cuda port ledger

## K4+V4 RESIDENT KV-QUANT (2026-06-13) — the wall moved 14.3K → 16K+, MEASURED

The tail-law arc completed: real 4-bit RESIDENT storage of BOTH K and V on
the global layers (not a round-trip that stays bf16 — actual uint8 packed
cache, ~4x cut). New engine primitive `tc.kv_int4_pack`/`kv_int4_unpack`
(D-grouped symmetric-8, distinct from the K-grouped weight `int4_dequant`;
the segfault-prone uint8 strided-copy avoided by writing raw uint8 in pack
and st<float> in unpack). Bit-identical to the rt_int4 ppl math (max|d|=0).
Wired into KVRing behind `GEMMA4_QUANT_KV4` (global-only — 4-bit V is fatal
on the sliding window, +1696%). Config from the bulk-bits law (Gemma qk-norm
=> 4): `int4_kv_glob` (K4+V4) = **+5.55% ppl** (baseline 119.85), survivable.

**MEASURED CEILING LADDER — prefill-512 + sequential decode, 3070 8GB:**

| | bf16 baseline | **K4+V4** |
|---|---|---|
| decode ceiling | **OOM at ~14,342** | **OOM at ~20,146** |
| context gain | — | **+5,800 tok / +40%** |
| @13,312 tok | 7835 MiB | 7609 MiB (−226) |
| @16,384 tok | (dead) | 7767 MiB SURVIVED |
| @19,456 tok | (dead) | 7767 MiB (cache PLATEAUED) |

bf16 walls at ~14.3K; K4+V4 reaches ~20K — 40% more context on the same
card. Key finding: past ~16K the q4 cache footprint PLATEAUS (7767 flat
16K→19K — 4-bit per-token KV growth is negligible). The 20K wall is NOT
the cache — it's the DECODE ATTENTION TRANSIENT (score/softmax buffers in
the blend path grow with context). So the next memory lever shifted from
storage-quant to the fused O(L) decode path. Lever identified, not yet
pulled. NO REGRESSION: Qwen3.5 parity PASS, Gemma parity
(q4 off) PASS w/ KV cosines 1.0000; r0.10 refine still −4.1% below standard
(the APA path untouched). gemma4_decode_ceiling.py is the probe.

## OVERNIGHT BUILD SUMMARY (2026-06-13) — KV-memory extension, 5 milestones shipped

Goal: extend Gemma's context on 8GB beyond where bf16 walls (~8K). The
APA was missing the fused O(L) path every other architecture uses; the
tail-law storage arc was the load-bearing complement. Both built.

| # | Milestone | Commit | Result |
|---|---|---|---|
| 1 | engine `write_rows` uint8 | `6b5e9b5` | storage-ring enabler, gated |
| 2 | INT8 V-storage in KVRing | `a5c04b9` | all gates green; 8K survives 64MB lighter |
| 3 | asymmetric K8+V4 measured | `a9f8373` | **−3.56% ppl, BEST mode** (anti-additive) |
| 4 | D=512 fused kernel + bottom-right causal | `d57b76e` | engine 66/66; the causal trap caught proactively |
| 5 | fused dispatch + chunked prefill quantize | `b267144` | **12K prefill survives at 7805 MiB — wall moved** |

MEASURED CEILINGS (all rungs, one-process-each): **PREFILL** — bf16:
solid ~10-11K, 12K RAGGED-EDGE (allocator nondeterminism at ~5%
headroom — survived bare/gate at 7805 MiB, OOM'd in the ladder run),
16K/20K hard OOM. qv (INT8-V): **12K SOLID at 7802 MiB** (firms up the
ragged bf16 edge), 16K OOM. So the real prefill ceiling is **~12K**,
bf16-marginal / qv-solid; INT8-V's ~64MB saving firmed the 12K floor
but did NOT reach 16K (16K's footprint is well beyond the saving).
**DECODE** — bf16 8K solid (12K OOMs in prefill), qv 8K at 7545 MiB
(64MB lighter, ~6ms/tok dequant tax). NET: the stack took Gemma from
walling at ~8K to a SOLID ~12K (prefill, qv) — a real ~1.5× extension
toward the trained window, achieved as a managed rising wall, NOT
MLA-flat (Gemma's full-width cache grows structurally). bf16 alone
can't hold the trained 32K (~25K wall); these pieces let it climb. ·

DISCIPLINE HELD: every quant test ACTIVE-guarded (no dead patches —
one was caught: KVRing.append hook was a no-op, +0.000% across 7 modes
flagged it); the bottom-right causal bug caught BEFORE shipping (vs the
121→11M blowup it caused once); the −3.56% surprise flagged for
confirmation not trusted; three fake OOMs correctly traced to
allocator fragmentation, now fixed in test DESIGN (one-context-per-
process). Open items: (a) confirm the −3.56% asymmetric; (b) asymmetric
K8+V4 resident storage as a second opt-in (deepest-context config);
(c) _v_get read-path opt (full-cap astype wasteful); (d) native uint8
engine slice (nvcc segfaults on ld/st<uint8> — astype-then-slice
workaround shipped).

---

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

### Ring rework (2026-06-12, "build the ring buffers — APA gets its rematch")

Engine: `write_rows` (in-place ring block write, grad-guarded, gated
exact incl. wrap). Port: `KVRing` — sliding layers ring at window
1024, globals capacity-double; decode appends are in-place; invalid
rows bias-masked (zero-init keeps softmax finite); lazy per-layer
conversion from prefill tuples at first decode; prefill-after-decode
unwraps via `ordered()` (two-slice unroll on wrap — band masks need
temporal order; attention itself doesn't). OWNERSHIP CONTRACT: the
decode loop owns ring buffers; sharing copies (GRM host round-trip
already does).

**Measured (per-step, synchronized): steady decode 28.8 ms/tok at
S=1100+ — flat with S, faster than the 32.3ms tuple-path at S=700,
27× the churn-era 773ms.** Ring decode is exactly deterministic
through the wrap boundary (C1 gate).

Iteration lessons (each OOM-proven, each fixed):
1. Saving rings as FULL BUFFERS made a 48-token state file 670MB and
   OOMed its own restore → save VALID ROWS + ring flag; wrapped rings
   restore with count==cap so the next append hits the logical-oldest.
2. FIXED-size rings cost ~370MB per cache set at ANY context (state
   gate holds 3 sets) → capacity-double from 64, cap at window, wrap
   only at window.
3. Per-chunk prefill score transients (16·step·S_all·2B ×2) OOM past
   ~3-4K → ADAPTIVE chunking holds the transient ≤32MB (step shrinks
   with context, floor 64).
4. KVRing guards its own in-place writes (load_caches runs outside
   caller no_grad).
5. Probe methodology: the v1 ladder (interleaved prefill/decode
   stages) produced decode numbers 27× off the clean per-step
   measurement and cycled caches through unwrap/reconvert each rung —
   superseded by gemma4_decode_probe.py (one prefill, long decode,
   one process per (mode, S)). And — third strike — NEVER grep-filter
   a failing gate's output; full logs to /tmp, always.

### The rematch (blend de-expansion + maskless prefill, 2026-06-12/13)

- **MQA blend de-expansion** (`_cublas_blend_attention` MQA branch +
  unexpanded call site): q heads fold into rows against the unexpanded
  K/quantized-K/V — the expansion path materialized 3× ~200MB per
  chunk at 12K. **Proven numerically exact**: synthetic diff vs the
  expansion path max|d| 0.0156 (bf16 ulp), isolated-real rect check
  worst flip 0.361 (cleaner than the expansion path's 1.18).
- **Maskless global prefill**: causal_softmax's built-in bottom-right
  causal replaces materialized masks (adaptive chunking gave every
  chunk a unique (L,S) → ~200MB of mask-cache churn at 12K;
  rectangular semantics verified exact at all shapes incl. L=1).
- **CASCADE SENSITIVITY (registered phenomenon, not a bug)**: two
  equally-valid pipelines (blend vs standard, or different chunkings)
  compound benign per-chunk rounding chaotically over 48 layers ×
  hundreds of tokens — isolated blend worst 0.361 vs
  blend-through-prefill 5.9 ON THE SAME INPUTS. Gates must test the
  MACHINERY (isolated) and treat cross-pipeline cascades as
  informational; the APA rect check now isolates (apa_min_context >
  prefill length).

**Ring-era scoreboard (decode probes, real serving shape):**

| S | standard | APA r0.10 (de-expanded) |
|---|---|---|
| 1.1K | 28.8 ms/tok | — |
| 4K | **47.0 ms/tok, SURVIVED** (7.39GB) | **56.1 ms/tok, SURVIVED** (7.54GB) |
| 8K | **69.7 ms/tok, SURVIVED** (7.64GB) | OOM — `_quantize_keys` |
| 12K | OOM — prefill budget (weights 6.77 + rings 0.34 + KV 0.2 leave ~340MB for transients+fragmentation) | OOM — `_quantize_keys` re-quantizes the WHOLE cache per call and materializes a reconstruction |

**Verdict on the rematch:** standard reaches 8K, APA dies at 8K — APA
runs SHORTER, the OPPOSITE of the design intent ("APA should run
longer"), confirmed measured. BUT the cause is now pinpointed and it
is NOT the algorithm: `_quantize_keys` (tensor_cuda/quant.py) rebuilds
the entire quantized key set AND a full-precision reconstruction
(`tc.matmul(centroids, Rb) * norms`) EVERY decode step — O(S) work and
O(S) transient per token, on the whole growing cache. The
`cache_apa_kq` lever (incremental: quantize only the appended key,
reuse cached quantized prefix) already exists in GQAAttentionTC and
was simply never wired into the Gemma port. Wiring it makes APA's
per-step cost flat and its transient bounded — exactly the design
contract — and is the gate to APA outrunning standard. Registered as
the next ticket.

Where the pre-ring stack decoded at 0.8s/tok and died at 5K, both
modes now serve real decode at 4K+. Remaining long-context tickets:
incremental kq caching (the `cache_apa_kq` lever that ALREADY EXISTS
in GQAAttentionTC, never wired here), prefill-into-rings, and the arc
that actually moves the 12K wall for BOTH modes: **quantized KV
STORAGE** (INT8 → 4-bit bulk + hot set per the Architect's tail law —
where APA stops being scoring-only and starts extending memory).

### Root-cause: WHY APA runs shorter on Gemma but longer elsewhere (multi-agent investigation, conf 0.88)

The contradiction has TWO layers; a single mechanism explains the
proximate OOM, three stacked deficits explain the sign flip.

**Proximate cause (the OOM):** Gemma APA decode re-quantizes the
ENTIRE growing global cache every token — `gemma4_tc.py:401` calls
`_quantize_keys(kv_cache.ordered())` on the whole `(1,1,S,512)` cache;
`new_kv = ring_cache` (`:394`) stashes NO kq, so it recomputes from
scratch each step. `_quantize_keys` (`quant.py:120-127`) materializes
5-6 simultaneous fp32 `(1,1,S,512)` tensors (kd, kd*kd, unit, rotated,
centroids, recon). At S=8192 that is ~80-96 MiB transient; the OOM
trace lands EXACTLY on `recon = matmul(centroids,Rb)*norms`
(`quant.py:127`), on APA's FIRST decode step (transient, not a leak).
Standard's competing transient is the ~0.5MB `(1,1,16,cap)` score
pair — two orders smaller — so standard survives 8K (7641 MiB) where
APA OOMs.

**Why the SIGN FLIPS (three stacked deficits, none present elsewhere):**
1. `cache_apa_kq` (incremental decode: quantize only the new key,
   concat onto cached prefix — `mistral7b_tc.py:418-420,459-463`) is
   WIRED on the GQA ports, never ported to Gemma. Makes APA's per-step
   quantize transient O(1) vs Gemma's O(S).
2. The fused O(L)-memory kernel `apa_selective_attention` — the real
   source of "APA longer" elsewhere (MiniCPM3: std 3,072 vs APA
   **32,768**) — is ARCHITECTURALLY FORBIDDEN on Gemma:
   `kernels.cu TC_APA_MAXD=256`, hard-throws for head_dim>256, Gemma
   global D=**512**. Gemma can ONLY run the transient-heavy cuBLAS
   blend fallback.
3. No INT8 KV storage (`quant_kv_cache=True` elsewhere) + ~6.8GB QAT
   body on 8GB: standard's own 8K decode high-water is only ~128 MiB,
   so any tens-of-MB transient is fatal.

**Adversarially RULED OUT:** the 16x MQA blend (de-expansion already
fixed it; OOM is in `_quantize_keys`, not the blend); head_dim-512 as
prime cause (MQA makes Gemma's KV×D=512 the SMALLEST of all APA ports —
if head_dim drove it MiniCPM3's 40×96=3840 would die first; it's a 2x
multiplier riding the true cause); fixed-overhead-shifts-the-wall (the
transient is provably O(S·D) by structure); ordered()-copy (globals
return slice VIEWS, zero copy); resident leak (no kq stashed back).

**Fix + honest scope:** wiring `cache_apa_kq` collapses the decode
transient ~80-96 MiB → ~16-32 KiB and lets APA CATCH UP to standard's
8K decode ceiling. It does NOT restore "APA longer" on Gemma: (a) the
12K APA OOM is in PREFILL, where the incremental lever is inert
(whole-span quantize), and (b) outrunning standard needs raising
`TC_APA_MAXD` to admit head_dim 512 into the fused kernel AND/OR INT8/
4-bit KV STORAGE (the tail-law arc). Decisive test: allocator
high-water for one S=8192 APA decode step, current path (~80-96 MiB
spike at quant.py:125-127) vs a one-key incremental patch (~16-32 KiB),
then re-run the 8K decode ceiling probe with cache_apa_kq wired.

**Open:** an unexplained ~110 MiB of allocator high-water at S=4096
(derived quantize transient is only ~44 MiB there) — wants a live
allocator trace to isolate fragmentation vs a near-fixed cuBLAS
workspace. The cross-model "APA longer" baselines are from the
MiniCPM3 results doc + code levers, not a freshly measured GQA
scoreboard this session.

### THE FIX: incremental kq cache wired into Gemma APA decode (2026-06-13)

Implemented the cheap lever the two investigations pointed at. `KVRing`
gains a parallel `kqb` quantized-key ring + `kq_count`; `quantized_keys
(quantize_fn)` quantizes ONLY newly-appended rows `[kq_count:count)`
(O(D)/step) and caches them — a key's quantization is fixed once it is
roped+written, so this is exact (verified row-independent: incremental
== whole-span, bit-identical at the same batch shape; the cross-batch
fp32 reassociation diff is 2.6e-6, bf16-noise class). The decode APA
branch reads cached kq + the unexpanded MQA cache directly into the
de-expanded blend — collapsing the per-step quantize transient from
~80-96 MiB (the measured 8K-decode OOM) to ~16-32 KiB.

kq is DERIVED state, never persisted (save/load round-trips k/v only;
kqb re-derives lazily on the first APA call after restore). Gate:
`tests/gemma4_apa_incremental.py` — all 8 global layers engage
(kqb populated, kq_count==count, none re-quantized), greedy
deterministic. Unit test: `quantized_keys` is bit-exact vs whole-span
at matched shapes.

**RESULT — APA SURVIVES 8K DECODE (the deliverable):** the rung where
APA OOM'd in `_quantize_keys` before now serves at 73.9 ms/tok,
7.61GB — *lighter than standard's own 8K high-water (7.64GB)*. APA has
CAUGHT UP to standard's decode ceiling (both ~8K) — exactly the
kernel-investigation prediction ("cache_apa_kq makes APA catch up, not
outrun"). APA decode 4K also improved: 50.1 ms/tok / 7.51GB (was
56.1 / 7.54 pre-fix — faster AND lighter, no per-step requant).

Wiring the incremental cache SURFACED three latent memory issues that
nothing had stressed because no run reached APA this deep before —
each a real fix helping BOTH modes:
1. **bf16-direct ring allocation** (`_zeros` via `tc.zeros(dtype=...)`):
   the old `_cast(tc.tensor(np.zeros(fp32)))` transiently needed 2× the
   final bf16 size — OOMed kqb alloc at 8K.
2. **chunked cold-start quantize**: the first APA decode quantizes the
   whole prefill span — `quantized_keys` does it in 512-row slices
   (row-independent → bit-exact), capping the one-time transient.
3. **bounded capacity growth** (`_grow_cap`): pure power-of-two
   doubling wasted up to 2× — next_pow2(8193)=16384 held 8192 tokens
   at 8K (+100%). Now: double while small, +2048 blocks once large →
   8K cap 16384→10240 (+25%), ~150MB recovered at the pressure point.
   Helps STANDARD too (it paid the same overallocation).

**Honest ceiling:** APA 12K still OOMs — but in PREFILL (the prefill
APA branch's whole-span `_quantize_keys`, ~140MB transient), NOT
decode. Decode is fixed. The 12K prefill wall is the same chunking
idea applied to the prefill branch, a distinct ticket; and 12K may
not fit 8GB regardless given resident KV. The dramatic ceiling
extension (APA OUTRUNNING standard) still needs quantized KV STORAGE
(the tail-law arc) — incremental kq carries +50% resident (the kqb
ring), so it equals standard's ceiling, doesn't exceed it.

### IMPLEMENTED: INT8 V-storage (the ceiling lever, 2026-06-13 overnight)

Built the measured-safe storage quantization. Engine: `write_rows`
uint8 path (commit 6b5e9b5 — strided raw copy, dtype-agnostic; the
slice-uint8 companion was attempted but nvcc SEGFAULTS instantiating
dimcopy's ld/st<uint8>, so a native uint8 slice is a future ticket).
Port (KVRing): `QUANT_V` flag (env GEMMA4_QUANT_V, default off) makes
`vb` uint8 + `vsb` per-row scale; `_v_put` quantizes on write,
`_v_get` dequantizes on read (astype-then-slice, since the engine
slice is float-only — resident stays uint8 so the ceiling win holds;
the read transient is the full cap buffer, registered as a read-path
optimization follow-up). All sites routed: init/append/_grow1/ordered/
the two attention reads/save_caches(via ordered)/load_caches.
WHY V-only INT8 and K bf16: measured int8_v −1.5% (a mild regularizer)
vs int8_glob +6.2% (the global K score-path is precision-sensitive —
head_dim 512 with attention scale 1.0, no 1/√d damping). Smoke: 48
rings uint8+scale, coherent decode, save/load round-trips.

**GATES (GEMMA4_QUANT_V=1): PARITY / APA / LONGCTX PASS; STATE PASS
as-corrected** — the "lossless bit-identical restore" contract became
"round-trip-idempotent" (tokens match exactly, bf16-exact logit
identity gone by design); the gate now demands bit-identity only at
bf16, token-match at QUANT_V. Re-baselined honestly, not forced.

**MEASURED CEILING (decode probe, qv):** 8K SURVIVES at **7545 MiB —
64 MiB lighter than bf16-V's 7609 MiB** (the resident saving is real
and shows up); decode 84 ms/tok (vs 74 bf16 — the dequant-on-read tax,
recoverable by the registered _v_get read-path opt). **12K still OOMs
— but in PREFILL (`_quantize_keys` whole-span fp32 transient ~140MB),
NOT the V cache.** This is the predicted boundary: storage quant bends
the RESIDENT slope (confirmed — 8K lighter), but the 12K wall is the
prefill TRANSIENT, which only the fused D=512 kernel removes. The two
pieces are complementary exactly as designed — storage + fused kernel,
not storage alone. INT8 V-storage shipped (commit a5c04b9).

CEILING (workflow-verified byte math, corrects the earlier "~32K
modest" framing): bf16 KV can't even hold trained 32K on 8GB (~25K
wall); uniform INT8 (12.3KB/tok global vs 24KB bf16, ~2×) reaches
~38-51K realistic / ~51-65K optimistic — EXCEEDS trained 32K with
margin. So INT8 is what lets Gemma reach its own designed context.
Asymmetric (K8+V4+kqb4, ~1.4× further to ~53-92K) was a speculative
reach gated behind the combined mode — **now MEASURED, and the result
flips the decision:**

**int8k_int4v_glob (K-INT8 + V-4bit on global) = 115.58 ppl, −3.56%
vs baseline — the BEST mode in the entire table**, better than either
component alone (int8_glob +6.2%, int4v_glob +5.1%) AND better than
int8_v (−1.5%). The combination is ANTI-additive: the two perturbations
partially cancel into net benefit (a regularizer effect), not compound.
So asymmetric global K8+V4 is BOTH the best quality AND the bigger
memory cut (~1.4× past uniform INT8) — genuinely worth building, not
just survivable. (CAVEAT: a −3.56% "too good" result gets the same
skepticism as the dead-patch +0.000% — ACTIVE-guard confirms it's real
and the value is in-band/plausible, but a confirmation run is
registered before it becomes the shipped default. The current shipped
default is V-only INT8, conservatively.)

### IMPLEMENTED: fused D=512 APA kernel wired — 12K PREFILL WALL MOVED (2026-06-13 overnight)

The fused O(L)-memory `apa_selective_attention` path, the piece that
was architecturally missing from Gemma's APA (it ran the memory-heavy
cuBLAS blend at every context). Engine (commit d57b76e): TC_APA_MAXD
256→512 + dispatch arm + the BOTTOM-RIGHT causal fix (s_max=(S-L)+i+1,
was top-left — the 121→11M bug class, caught BEFORE shipping this time;
regression tests cover non-square S>L + MQA-D512). Port: fast_max_seq
=4096 switch at the global PREFILL APA site (decode stays on the blend
— at L=1 the blend transient is ~1MB and the fused kernel loses 30× on
the 16-block launch), PLUS chunked whole-span `_quantize_keys` (2048-
row slices, bit-exact, caps the ~140MB fp32 recon transient the fused
kernel doesn't touch).

**MEASURED: 12K PREFILL NOW SURVIVES at 7805 MiB** (load 6759 + ~1046
of cache/transients, ~390 MiB headroom) — where the blend OOM'd. The
prefill wall moved from <12K to ≥12K. Fused≡blend equivalence: top1
match, max|d| 1.25 (near-tie) at a shared context. NOTE: the gate must
drain the allocator pool between contexts — running 1024 then 12K in
one process fragmented the pool and falsely OOMed (the recurring
one-process-per-context lesson; gate now flushes).

So the complementary picture is now BUILT and MEASURED: incremental-kq
(decode O(D)) + INT8 V-storage (resident ~2×) + fused kernel (prefill
transient) together take Gemma from walling at ~8K to **12K prefill +
deeper decode**, reaching toward its trained window. Asymmetric K8+V4
storage (measured −3.56%, best mode) is the next resident lever for
beyond 12K.

### THE MISSING PIECE + the honest MLA-flatness limit (Architect's catch, 2026-06-13)

The Architect sensed Gemma's APA was "missing something" — every other
architecture saw massive memory gains, Gemma didn't. CONFIRMED in
source: **Gemma's APA has NO fused-kernel branch.** MLA/GQA dispatch
(minicpm3_tc.py:231-243, mistral7b_tc.py:432-441) switch to the fused
O(L)-memory `apa_selective_attention` above fast_max_seq=4096 — that
fused path is what ran MiniCPM3's 2,048→32,768 climb at ~10MB resident
drift. Gemma runs `_cublas_blend_attention` at EVERY length (no
fast_max_seq, no switch) because the fused kernel throws at head_dim>256
and Gemma globals are 512. The prior "DON'T build D=512" verdict was
correct ON THE SPEED axis and WRONG on the memory axis.

**Two corrections to the framing (memory-feasibility investigation,
conf 0.86):**
1. The fused kernel saves memory at PREFILL, not DECODE (at L=1 the
   blend transients collapse to ~1MB). The decode 8K wall was already
   broken by incremental-kq. So the fused kernel is the **12K+ PREFILL
   ceiling lever**, not the decode fix.
2. **Gemma CANNOT get MLA-flat residency — it's a HARD architectural
   difference.** MLA flattened because its cache is a tiny fixed latent
   (~288 vals/token). Gemma globals are full-width MQA: kb+vb+kqb =
   24KB/token, growing linearly, ~85× heavier per token, structural
   (from the trained 512-wide single-head global attention). No kernel
   flattens it. Honest ceiling: fused kernel converts an early ~8-12K
   TRANSIENT wall into a later ~24-32K RESIDENT wall (a rising line
   192→768MB across 8K→32K, NOT MLA's flat shelf); +INT8/4-bit storage
   bends the slope ~1.5-2.7× shallower (clears 32K with margin). The
   achievable goal is "reaches the trained 32K window as a managed
   rising wall," not "memory never walls."

**Build plan: BUILD-WITH-STORAGE.** Piece 1 (transient killer): D=512
dispatch (TC_APA_MAXD 256→512, launch arm, delete throw) + fast_max_seq
switch in the Gemma APA branch. TRAP (the same class that's bitten 3×):
the fused kernel is TOP-LEFT causal (s_max=i+1) while the whole Gemma
stack is BOTTOM-RIGHT — they agree ONLY at S==L, and Gemma always
chunks prefill onto a cache (S>L), so a naive wire is the 121→11M-ppl
bug AGAIN, and the existing kernel test runs only S==L so it passes
while wrong. Needs a bottom-right/q_offset kernel mode + a NON-SQUARE
regression test. Piece 2 (the load-bearing half — bends the resident
slope, the only durable-ceiling lever): INT8/4-bit kb/vb storage in
KVRing with dequant-on-read, gated by the ppl experiment below.

**KV-STORAGE PPL (tests/gemma4_kv_quant_ppl.py, real wikitext 6×2048).**
CORRECTION: an earlier version of this entry claimed "INT8 perceptually
free, +0.000%, bit-identical" — that was a DEAD PATCH (it hooked
KVRing.append, which only runs at first decode; the scored tokens flow
through the prefill/cat path it never touched, so all 7 modes returned
bit-identical ppl — the physical impossibility was the tell). The test
now hooks `Gemma4AttentionTC.KV_STORE_HOOK` (every scored token's K/V,
every path) with a NO-OP GUARD asserting each mode actually perturbs the
logits. Real numbers (all ACTIVE, baseline 119.85):

| mode | ppl | Δ | |
|---|---|---|---|
| int8_v | 118.02 | **−1.5%** | best — V-only INT8 safe |
| int8_slide | 121.51 | +1.4% | safe |
| int8_kv | 123.11 | +2.7% | borderline |
| int4v_glob | 125.94 | +5.1% | marginal |
| int8_glob | 127.25 | +6.2% | marginal |
| int4v_slide | 2152 | **+1696%** | CATASTROPHIC |
| int4v_all | 6559 | **+5373%** | CATASTROPHIC |

**FINDINGS, against prediction:** (1) The "scale-free v_norm ⇒ V
tolerates 4-bit like K" hypothesis is REFUTED — 4-bit V is fine on
GLOBAL (+5%) but ANNIHILATES SLIDING (+1696%). Same quant, opposite
outcome by layer class. (2) Likely mechanism: sliding attention sees
only 1024 keys, so each V carries much higher weight (less softmax
averaging) — a 4-bit V error on a high-weight value is NOT crushed (the
un-crushed-hot-value risk the kernel workflow flagged). (3) INT8 is
GOOD not free: int8_v (V-only) is the sweet spot at −1.5%; INT8 on K
(int8_glob/kv) costs more (the score path is more precision-sensitive
than the value path). (4) The tail-law arc's realistic operating point
is **INT8 V-storage** (or INT8 sliding + INT8 V global), NOT blanket
4-bit. 4-bit is reserved for the GLOBAL V only, and even there +5% needs
a serving-quality decision. The 2.7× slope cut from blanket 4-bit is
OFF THE TABLE; the achievable cut is ~2× (INT8) with a V-only refinement.

### CAN the fused kernel be fixed for D=512? (kernel-feasibility investigation, conf 0.86)

VERDICT: mechanically YES, but it's the WRONG LEVER — DON'T build it;
wire `cache_apa_kq` instead. Three findings (an agent RE-COMPILED the
kernel with nvcc 12.0 -arch=sm_86 -Xptxas -v, numbers measured not
estimated):
1. The `TC_APA_MAXD=256` throw guards a register cliff THAT DOESN'T
   EXIST. `acc[DMAX]` is dynamically indexed → ptxas places it in
   LOCAL memory at EVERY head_dim (stack frame 512/1024/2048 B at
   D=128/256/512); the comment claiming register residency is false.
   D=512 compiles clean (42 regs, 13312 B smem, 0 spills), launches
   legally (~7 blocks/SM, smem-bound), correct output. Removing the
   throw is a 3-line change.
2. But the fused kernel is a DECODE LOSER BY STRUCTURE: one block per
   (b,h,row) → at decode L=1,B=1 it launches B*H*L=16 blocks on a
   46-SM card, 30 SMs idle every token. Measured ~5.0ms (fused) vs
   ~0.16ms (cuBLAS blend) at S=8192 — 30× slower. cuBLAS GEMV
   saturates the device; one-block-per-row cannot. Warp-scoping
   (the only correct non-naive restructure) makes it WORSE.
3. And it's UPSTREAM of the wrong problem: the 8K OOM is in
   `_quantize_keys` (the CALLER, before attention dispatches) — no
   attention backend can fix it. The fused kernel's only real win is
   prefill memory, already delivered by the blend de-expansion.

So: `cache_apa_kq` (≈1-2 days, machinery exists in GQAAttentionTC,
removes the measured O(S) requant transient) gets essentially ALL the
win and is the ONLY thing touching the real allocation. A D=512 fused
kernel would matter only for a future PREFILL-ceiling push (~2× slower
prefill-memory win, gate above fast_max_seq); a decode-fast fused
kernel needs a FlashDecoding split-over-S redesign (1-2 weeks), none
of the explored approaches. Expected outcome of the cheap fix: APA
reaches a HIGHER context ceiling than standard ("runs longer"), while
staying a few ms/tok SLOWER per token (more quant work) — longer, not
faster. That distinction is the whole claim.

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
