# MiniCPM3-4B on tensor_cuda — first MLA model, validated against PyTorch ground truth

**2026-06-09. One day: architecture port, two-tier PyTorch validation, APA enablement.**
Companion to `APA-Quant_CrossModel_Results.md` (the GQA-family results). Every number
measured on the same 8GB RTX 3070, same guide-corpus protocol (ppl of the last 512
tokens given the prefix), fresh processes.

## Why this model
MLA (Multi-head Latent Attention) is the attention the frontier models run
(DeepSeek V2/V3/R1, Kimi K2): instead of caching per-head K/V, it caches one small
latent per token (here 256+32 values/layer vs GQA-Qwen3's 2,048) — ~¼ the KV
footprint per token at 62-vs-36 layers. MiniCPM3-4B is the only open MLA model
that fits an 8GB card (INT4 ≈ 2.2GB weights). It is also architecturally the
hardest port so far: MLA's latent down/up-projection dataflow, composite 96-dim
keys (64 no-RoPE + 32 RoPE'd, the RoPE key shared MQA-style across all 40 heads),
**asymmetric dims (D_qk=96, D_v=64)**, three μP scaling factors that silently
destroy output if missed (embeddings ×12, residuals ×1.4/√62, hidden ÷10 before
the tied lm_head), a longrope frequency table, and 62 layers. Ships as
`pytorch_model.bin` (no safetensors).

## Adapter validation — correct on first build
`core/minicpm3_tc.py` (new `MLAAttentionTC` class). Smoke: *"The capital of France
is **Paris**. — Paris in History…"*, *"Water is made of two elements: **hydrogen
and oxygen**."* Coherent, correct, structured, at 2,783MB loaded.

## Ground truth vs PyTorch (two tiers, same card, same protocol)

| | PyTorch CPU bf16 (their code, unquantized) | **tensor_cuda INT4** | PyTorch GPU bnb-4bit (default stack) |
|---|---|---|---|
| behavior (greedy) | Paris / H+O ✓ | Paris / H+O ✓ | Paris / H+O ✓ |
| ppl@1024 last-512 | **17.357** (ground truth) | **20.065** (+15.6%) | 22.507 (+29.7%) |
| prefill@1024 | — (CPU) | 2,820 ms | **441 ms** |
| load (GPU) | — | 2,783 MB | 2,948 MB |
| context ceiling | — | **3,072** (standard attn) | **2,048** (OOM @4096) |

**The honest sentence: at matched 4-bit on the same card, tensor_cuda is ~11%
more accurate than the default PyTorch stack and holds 1.5× the context, while
being 6.4× slower at prefill.**

- The ppl ordering (GT < ours < bnb) validates the adapter end-to-end. Our +15.6%
  INT4 gap is larger than the Mistral pattern (+5%) — but bnb's +29.7% on the same
  model independently confirms MiniCPM3 itself is quantization-sensitive
  (plausibly the μP scalings amplifying quant noise), not an adapter leak.
- The 6.4× prefill gap is NOT the INT4 two-pass (measured single-digit % in the
  engine audit). It is launch-bound: 62 layers × ~15 Python-driven kernel launches,
  on MLA's tiny latent matrices (2560→768, 2560→288) that finish faster than their
  launch overhead, vs PyTorch's fused kernels + flash attention. Engineering debt,
  not architecture.
- Decode tok/s is NOT comparable in this table (our bench re-forwards the full
  prompt per token; PyTorch used KV caching). Prefill is the honest speed number.
- trust_remote_code note: their modeling file targets transformers 4.4x; on 5.7 it
  crashes (`is_torch_fx_available` removed). Reference ran in a pinned 4.49 venv.

## APA on MLA — works, and 4-bit is the measured operating point

**The structural problem:** MLA breaks an assumption baked into the fused selective
kernel — it has ONE head-dim `D` for both the key dots and the value accumulation,
but MLA has D_qk=96 ≠ D_v=64. Two-path solution, no kernel changes:
- **S ≤ 4096:** the cuBLAS-blend path (separate matmuls — dim-agnostic).
- **S > 4096:** the fused O(L) kernel with **V zero-padded 64→96** and the output
  sliced back (wastes ~33% of the V transient; correctness verified live).

**Bulk-bits sweep on the composite 96-dim keys** (ppl@1024, refine 0.10; standard
reproduced exactly at 20.065 in-process — APA code path adds zero drift):

| config | ppl@1024 |
|---|---|
| standard | 20.065 |
| apa 8-bit | 20.080 (+0.015, free) |
| **apa 4-bit** | **19.817 (−0.25 — noise-level free)** |
| apa 2-bit | 29.145 (+9.1, broken) |

**The key-distribution finding gains its fourth architecture — and holds:** the
MLA composite key is half-normalized (the k_nope half passes through the RMSNorm'd
latent) and half-raw (the RoPE'd k_pe slice). Prediction: between OLMoE's
qk-norm-4-bit and Qwen's raw-8-bit. Measured: **4-bit suffices.** Table now:
Llama-natural **2**, OLMoE-qknorm **4**, **MLA-latent 4**, Qwen-raw **8** —
normalization tracks bulk bits across four architectures including the frontier
attention design.

## APA context ceiling — the model's full trained window. Memory never walled.
At bulk 4 / refine 0.10, diverse-token validated protocol, grid capped at the
model's trained 32,768 (beyond = extrapolation, not capability). **Every rung
from 2,048 to 32,768 validated** (cap logits peak 0.186), fused padded-V path
live above 6144. Resident VRAM across the entire climb: **2,846-2,856MB — a 16×
depth increase moved residency ~10MB.** Transients peaked ~5.9GB in the deepest
prefills (the O(S²) bulk pass), which is also why the deep rungs are slow
(28,672 took ~61 min, 32,768 ~100 min on the 3070 — capability, not comfort).

**Final three-way: PyTorch default 2,048 / engine standard 3,072 / engine APA
32,768 — the ceiling is the model's own trained window, not the 8GB card.**
On this architecture class (MLA), APA converts a memory-bound context limit
into a model-bound one.

## Speed pass (2026-06-10, overnight) — 6.4× gap closed to 2.8× opt-in / 5.9× default

nsys on the 2,820ms prefill found the engine's #1 cost was **not compute**:
~6,700 cudaMalloc/cudaFree pairs per forward, 8.6s of blocking API time across
3 forwards vs 0.14s of kernel launches (cudaFree implicitly syncs the device,
so CPU and GPU ran lock-step). Tensor cores were already engaged (cutlass
HMMA kernels in the trace) — the GEMMs were never the problem.

| config | prefill@1024 | vs PyTorch-bnb 441ms | ppl@1024 | walls |
|---|---|---|---|---|
| baseline (63960a4) | 2,820ms | 6.4× | 20.065 | all intact |
| + matmul alpha/trans_b | 2,611ms | 5.9× | 20.091 | all intact (5-way gated) |
| + fused RMSNorm | 2,122ms | 4.8× | 20.050 — better than baseline | both walls PASS |
| + fused causal softmax (**default candidate**, commit 0223c31) | **1,537ms** | **3.48×** | 20.102 | parity at every rectangle; std-attn wall PASS |
| + `tc.set_alloc_pooling(True)` (opt-in) | **675ms** | **1.53×** | 20.102 | MiniCPM3 intact; Mistral exact-margin −1 rung |

*(The fused softmax and the pool COMPOUND: the eager softmax chain was the
pool's largest transient churn — ~6 passes over an 84MB score matrix per
layer. Fusing it took the pooled path from 1,046ms straight through the 2×
target to 1.53×.)*

What landed:
- **matmul `alpha` + `trans_b`** — attention scale folded into the GEMM's fp32
  accumulator (the +0.026 ppl is one FEWER 16-bit rounding), K consumed via
  cuBLAS OP_T with no materialized transpose. Gated: parity (fwd+bwd, 3 dtypes),
  59/59 tests, ppl, and BOTH ceiling spot-checks.
- **Fused RMSNorm** (`tc.rms_norm`, `RMSNormTC.USE_FUSED`) — replaces the
  9-launch / 9-alloc chain at 249 sites/forward with one kernel and one alloc.
  Sum-of-squares accumulates in **fp64** (free — the kernel is bandwidth-bound;
  fp32 tree-reduce drifted ppl to 20.119, 0.004 past the gate, and was
  REJECTED). Result: ppl 20.050 — *better* than the unfused baseline, because
  the fp64 mean is more accurate than the old fp32 serial reduce. Inference-
  only (backward throws; training falls back to the chain automatically).
- **Transients-only allocation pool, opt-in flag** — persistents (weights/
  scales/norms) always raw; forward transients use the driver's stream-ordered
  pool when enabled after load. Numerically bit-identical; the win is pure
  allocator serialization removed.

**The allocator-vs-walls finding (measured the hard way, 4 variants):** every
pool design loses an EXACT-MARGIN context wall — all-pool, pool+synced-trim-
retry, ≤16MB hybrid all OOM'd Mistral r0.05@16384 while pure-raw and ≤1MB-
hybrid (which pools ~nothing at S=1024) hold it; a full bisect exonerated the
matmul change (raw+step-1 PASSES). Mechanism: the pool expands in coarse
reserved chunks with alignment slack, so at zero margin it needs MORE driver
memory than the exact-size raw malloc it replaced; live blocks pin chunks
against trimming. Hence the dial: default = every published ceiling intact;
flag-on = 2.8×-of-PyTorch prefill, ceilings only affected where the margin was
already zero (MiniCPM3's 16k spot passes flag-on; Mistral r0.05's wall moves
exactly one rung, 16,384 → 14,336 — a measured 12.5% ceiling cost on that one
exact-margin config). Models with headroom pay nothing.

Next levers (benefit the DEFAULT path): fused RMSNorm and fused causal-softmax
remove their intermediate tensors entirely — each fused chain deletes both
kernel time AND its raw alloc/free pairs (~2,000 pairs/forward for norms
alone), which is the ceiling-safe road from 2,611ms toward the 2× band.

## Doors (what could be wrong / what's unmeasured)
- ppl protocol is one 512-token window of one corpus; cross-length and
  cross-corpus generalization unmeasured on this model.
- The +15.6% INT4 gap: attributed to model quant-sensitivity via the bnb control;
  a per-subsystem ablation (which projections hurt most under INT4 — the tiny
  latent mats are suspects) would close it fully.
- Padded-V fused path verified by output validity, not yet by logit-equivalence
  against the blend path at the same S.
- Speed pass (above) measured at S=1024 on MiniCPM3 only; flag-on behavior at
  other depths/models spot-checked, not swept.
- ~~The MLA *latent-caching* memory win is NOT yet implemented~~ DONE: the
  cache holds (post-norm 256-d latent, RoPE'd 32-d shared key) = 288
  values/token/layer; cached greedy decode is token-IDENTICAL to re-forward.

## Decode speed pass (2026-06-10): 675 → 21.6 ms/token (31×)

Decode @S≈360, RTX 3070, greedy, full 62-layer INT4 model. Five compounding
levers, each gated (greedy token parity vs the all-off reference: IDENTICAL,
plain + graft-arena conditions; teacher-forced max logit diff 0.41 = the
plain cache-vs-prefill noise floor; engine suite 59/59):

| Stack | ms/tok | tok/s |
|---|---|---|
| baseline (eager, two-stage INT4) | 675 | 1.5 |
| + absorbed MLA decode | 563 | 1.8 |
| + transients pool | 383 | 2.6 |
| + fused INT4 (old tiled kernel) + lm_head trans_b | 283 | 3.5 |
| + tc.no_grad() + fused rms_norm | 158 | 6.3 |
| + **int4 GEMV kernel** (engine `8501a5c`) | **21.6** | **46.3** |

What each fixed:
- **Absorbed decode** (`absorbed_decode` flag, DeepSeek-V2 style): score the
  latent directly — (W_uk^T q_nope)·c_n + q_pe·k_pe, W_uv after the weighted
  latent sum; kv_b never re-expands the span. W_uk/W_uv recovered
  kernel-exactly via `kv_b(I)`. Prefill keeps the expanded path.
- **lm_head trans_b**: the tied head materialized a 188M-element transpose
  every forward (38.6ms/step, 10%) — `matmul(x, w, trans_b=True)` is free.
- **tc.no_grad()**: `tc.is_grad_enabled()` defaults TRUE — every generation
  loop in the project had been paying autograd-tape overhead, and the
  fused-norm guard silently fell back to the eager chain. ~2× alone.
- **int4 GEMV kernel**: nsys showed the two-stage path dequantizing ~3GB of
  weights/token (the fused tiled GEMM wastes 15/16 of each tile at M=1).
  New warp-per-row GEMV streams packed INT4 once; dispatched inside
  `int4_linear_fused` when M==1. Mission-side pick: `QuantLinearTC.
  FUSED_DECODE` routes M≤8 calls there; prefill stays two-stage cuBLAS
  (the tiled fused kernel LOSES at large M — 11ms vs cuBLAS's tensor-core
  path; the flag must be shape-aware, never global).

Arena impact (Graft Repository E4-arena, one persistent cache): probe turns
33s → **1.5s**, scripted feeds 1.3s → 0.5s, recall 6/6 unchanged.

All flags ship default-off (validated two-stage/eager remain the defaults);
the fast stack = `tc.no_grad()` + `set_alloc_pooling(True)` +
`QuantLinearTC.FUSED_DECODE` + `RMSNormTC.USE_FUSED` + `absorbed_decode`.

Files: `core/minicpm3_tc.py` (adapter), `/tmp/minicpm3_{smoke,reference,
engine_bench,ceiling}.py` + APA sweep inline (protocols), reference venv:
transformers 4.49 + system torch. Decode gates:
mission_b74b7906/mla_fullstack_gate.py, gemv_parity.py.
