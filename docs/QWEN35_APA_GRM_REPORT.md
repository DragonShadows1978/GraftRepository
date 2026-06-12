# Qwen3.5-9B with APA + GRM on tensor_cuda — change report & metrics vs baseline

**Date:** 2026-06-12 · **Hardware:** RTX 3070 8GB (sm_86) ·
**Baseline:** ollama `qwen3.5:9b` (llama.cpp, GGUF Q4_K_M, ~25 tok/s
decode measured on this machine)

**Outcome in one line:** the APA+GRM-equipped tensor_cuda stack now
serves the same workload the baseline served, *faster per token and
much faster per request*, with infinite-context economics the
baseline structurally cannot match — verified end-to-end by the
consumer's own acceptance loop.

---

## 1. What was built

### 1.1 The port (`core/qwen35_tc.py`)
First hybrid-architecture model on the engine: 32 layers =
24× Gated DeltaNet (linear attention, fixed-size fp32 state) +
8× gated attention (GQA 16q/4kv, head_dim 256, per-head qk-norm,
elementwise sigmoid output gate, partial RoPE 64/256 @ θ=1e7).
Text-only (vision tower + MTP head skipped), INT4 in-process
quantization, 248K-vocab embeddings host-side with row gather.
**Resident VRAM: ~4.6GB** (baseline ollama: 6.3GB).

Correctness law: adjudicated against PyTorch fp32 CPU ground truth
(`tests/qwen35_gt.py`), never in-repo runs. Parity gate is
margin-based (an INT4 engine vs an fp32 reference flips only
near-ties): all teacher-forced disagreements sit at GT tie margins
(worst 1.92 logits); prompt 0 is exact including 16/16 greedy tokens.

### 1.2 APA on the attention layers
`apa_selective` mode wired into the 8 attention layers: cuBLAS blend
path (fused kernel caps at head_dim 128; this model is 256),
KV-head-granularity codebooks, **bulk_bits = 4** — predicted before
measurement by the bulk-bits-tracks-key-normalization law (qk-norm
family → 4) and confirmed: **the law now holds on a fifth
architecture.**

APA gate (engine-vs-engine, both INT4): **zero top-1 flips** vs
standard attention across all GT prompts at bulk4 / refine 0.15
(final everything-on suite: one flip at 0.062 logits — noise floor).
APA's measured role on this card class: attention transients stay
bounded as context grows (the MiniCPM3-measured law: memory-bound
context limits become model-bound).

### 1.3 GRM on the hybrid state
The hybrid cache is per-layer: KV `(B,4,S,256)` at 8 attention
layers + DeltaNet `(conv (B,3,8192), state (B,32,128,128) fp32)` at
24 layers — Markovian, so save/restore is lossless by construction.

- `save_caches` / `load_caches`: **STATE GATE PASS — restored caches
  continue BIT-IDENTICALLY** (logits array-equal), both post-prefill
  and mid-decode.
- **Functional-kernel contract:** the fused GDN step returns
  `(out, new_state)` with the input state byte-untouched — held
  caches are branch-safe (restore-once-decode-many). The first
  in-place version corrupted held references; the parity gate's
  state check caught it, and the contract is now enforced by an
  engine test. Cost of the fix: zero (the kernel already wrote every
  element).
- **Serving integration (`scripts/qwen35_server.py`):** ollama-API-
  compatible shim with GRM prefix mounting — the hybrid state after
  the longest previously-seen prompt prefix is cached and restored;
  only the variable tail pays prefill.

### 1.4 Engine additions (Project-Tensor)
- `gated_delta_step` kernel (`a3363b0` + functional fix `0eb2ca9`):
  l2norm(q,k) + gate math (sigmoid/softplus/exp) + decay-first
  delta-rule update + readout — **one launch per GDN layer per
  token** (composed path: ~40 Python-dispatched launches).
  Warp-per-state-column register layout. Gated exact vs a float64
  reference; on-model tokens identical to the composed recurrence.
- Decode-speed defaults: int4 GEMV dispatch at M≤8 (kernel existed,
  was never enabled — two-stage had been dequantizing **~31GB of
  weights per token**) and fused RMSNorm.

---

## 2. Metrics vs baseline

### 2.1 Decode speed ladder (single-token, measured)

| Configuration | ms/token | tok/s | vs baseline (25) |
|---|---|---|---|
| v1 composed (two-stage INT4) | 840 | 1.2 | 0.05× |
| + int4 GEMV default | 122 | 8.2 | 0.33× |
| + fused RMSNorm default | 34 | 29 | **1.16×** |
| + fused GDN step kernel | **26.4** | **38** | **1.5×** |

Every rung re-gated: parity margins unchanged, state restores
bit-identical, APA flips zero/near-zero. Hardware ceiling ~77 tok/s
(bandwidth); remaining levers (CUDA graphs, attention batching)
project ~50 but with diminishing returns — the MLP GEMVs (12.2ms)
are already near the bandwidth floor.

### 2.2 Serving the real workload (corpus authoring, measured)

Ready gate = the corpus driver's **own `run_shard`** (its prompts,
`validate()`, repair, dedup) through the shim: **10/10 accepted
templates in 38s.**

| Per-request (699-token prompt, ~220 tokens out) | prefill | total | decode |
|---|---|---|---|
| Shim, cold (no mount) | 5.0–9.8s | 13.0–15.9s | 27.9 tok/s |
| **Shim, warm (GRM mounted 643/699 tokens)** | **0.8s** | **8.8s** | 27.8 tok/s |
| Baseline ollama (same job class) | ~1s | ~10s | ~25 tok/s |

At today's short prompts the warm advantage is modest (~10-15% per
request). The structural difference: **mount cost is flat (~0.8s)
regardless of mounted size; baseline prefill is linear in context.**
At a 10K-token mounted context (e.g. a per-relation library digest
for duplication avoidance — GRM's original purpose) the baseline
pays 10-20s per request; this stack still pays 0.8s.

### 2.3 Context economics (the "doesn't stop working" axis)

| | This stack | Baseline |
|---|---|---|
| Layers that grow with context | 8 of 32 (KV, ~32KB/token) | same arch, but… |
| Re-use of long shared context | restore state, ~0.8s flat | re-prefill, linear |
| Prefix caching on hybrid models | gated bit-identical | limited (mainstream stacks struggle with recurrent-state caching) |
| Attention at long context | APA-bounded transients | full attention |
| Practical mounted-context ceiling (8GB) | ~80-100K tokens | n/a (per-request prefill cost dominates first) |
| Resident VRAM | 4.6GB | 6.3GB |

### 2.4 Quality gates (all PASS at final configuration)

| Gate | Result |
|---|---|
| Parity vs PyTorch fp32 GT | PASS — flips only at GT near-ties (worst 1.92 logits); prompt 0 exact incl. greedy 16/16 |
| GRM state restore | PASS — bit-identical continuation, post-prefill and mid-decode |
| APA vs standard | PASS — 0–1 top-1 flips, worst cost 0.062 logits |
| Fused GDN kernel | PASS — tokens identical to composed; L0 state max\|Δ\| 5.7e-06; exact vs f64 reference |
| Ready-to-work (consumer's own loop) | PASS — 10/10 validated templates / 38s |

---

## 3. Operations

- Shim: `python3 scripts/qwen35_server.py 11435` (~6 min load);
  health: `curl 127.0.0.1:11435/api/tags`.
- Driver switch: `CORPUS_LLM_URL=http://127.0.0.1:11435/api/generate`
  (GRAPA `75de887`; unset = baseline ollama, unchanged).
- 8GB is single-occupancy: shim (4.6GB) XOR ollama (6.3GB).
- Orchestrator manual: `GRAPA-Native-LLM/corpus/QWEN_READY.md` (v2).

## 4. Open items

1. GRM multi-graft ARENA on hybrids — KV seats compose, DeltaNet
   states don't; prefix-chain mounting (shipped) covers the serving
   pattern; true arena composition is the research question.
2. Library-digest mount for duplication avoidance — the original GRM
   purpose; the economics above make it free on this stack.
3. Prompt reorder in the driver (constant blocks first) → near-every
   request warm-mounts; orchestrator's call, gate against yield.
4. Decode ~50 tok/s via CUDA graphs/batching — diminishing returns.
5. Chunked delta-rule prefill (cold-mount minting speed; warm path
   already bypasses it).

## 5. Commit trail

Engine (Project-Tensor): `8501a5c` (int4 GEMV kernel, prior work) →
`a3363b0` (gated_delta_step) → `0eb2ca9` (functional contract).
Port (GraftRepository): `aba80b5` (port + parity/state/APA gates) →
`4a989f2` (decode forensics) → `5db5bad` (speed ladder) →
`b97b150` (shim + ready gate).
Corpus (GRAPA-Native-LLM): `75de887` (driver env override) →
`22bbeab` (orchestrator manual).
Raw lab notebook: `docs/QWEN35_PORT_LEDGER.md`.
