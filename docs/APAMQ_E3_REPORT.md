# APAMQ-E3 — Gemma-4 Local Peak Attribution + Perfect-APA Upper Bound

Date: 2026-08-13. Evidence class: **INSTRUMENTED PORT MEASUREMENT —
memory and speed attribution only.** No quality or perplexity claims are made.

## Execution status

The read-only live-source audit and the instrumentation harness completed. All
six required GPU cells were launched as separate, lock-serialized processes,
but every cell failed before model load with:

```text
cudaMallocAsync failed: no CUDA-capable device is detected
```

The execution environment exposed no `/dev/nvidia*`. The required 1 s NVML
side-thread poller also received `nvidia-smi` return code 9: it could not
communicate with the NVIDIA driver. Consequently no cell reached prefill,
`L=1` decode, CUDA-pool event tracing, or the in-process `kqb` dtype assertion.
The 4K–16K cells below are therefore factual **FAILED / NOT MEASURED** results,
not estimates filled into measurement columns. The optional 32K cells were
skipped because the required 4K–16K matrix did not complete.

Raw receipts are under `artifacts/apamq_e3/`; the exact command/status index is
`artifacts/apamq_e3/matrix_manifest.json`. Every receipt records matching
before/after SHA-256 values for `core/gemma4_tc.py`,
`core/mistral7b_tc.py`, and the shared engine `kernels.cu`.

## Path audit

The live source has 48 layers, with globals at `i % 6 == 5`
(`core/gemma4_tc.py:61-63`): 40 sliding layers and 8 global layers.

- `apa_min_context = 2048` and `fast_max_seq = 4096` are assigned at
  `core/gemma4_tc.py:476-488`. Both comparisons are strict `>` comparisons.
- `L == 1 and kv_cache is not None` enters the decode ring branch at
  `core/gemma4_tc.py:537-545`. APA is active only for a global layer in
  `apa_selective` mode with `S_all > 2048` (`core/gemma4_tc.py:551-555`).
  Once active, decode unconditionally calls `_cublas_blend_attention`
  (`core/gemma4_tc.py:562-579`). There is no fused-dispatch check in this
  branch.
- Non-decode global APA is selected at `core/gemma4_tc.py:616-622`. It uses
  the cuBLAS blend through `S_all <= 4096` and
  `tc.apa_selective_attention` only when `S_all > 4096`
  (`core/gemma4_tc.py:646-662`). Standard global prefill uses the de-expanded
  MQA score/softmax path at `core/gemma4_tc.py:663-683`.
- The blend's MQA arm folds the 16 query heads into rows against unexpanded
  KV=1 tensors at `core/mistral7b_tc.py:264-287`.
- Sliding prefill always stays standard: composed causal SDPA while
  `S_all <= 1024`, otherwise composed band-mask SDPA
  (`core/gemma4_tc.py:684-701`).
- The top-level prefill is adaptively chunked at
  `core/gemma4_tc.py:783-831`. The static audit mirrors that arithmetic rather
  than assuming 512-token chunks throughout.
- The live engine cap is `TC_APA_MAXD = 512`
  (`/mnt/ForgeRealm/Project-Tensor/tensor_cuda/src/kernels.cu:1053`). The
  dispatcher has a 512 arm (`kernels.cu:1847-1850`) and rejects only
  `cap > 512` (`kernels.cu:1868-1869`). Gemma global D=512 is therefore
  **kernel-legal**.

Thus H-D's wiring observation is explicit: **the fused kernel is legal for
Gemma global decode at D=512, but the current Gemma decode branch does not wire
or call it.** This is a static path finding; the unavailable GPU prevented a
direct D=512 decode launch receipt.

### Branches by requested total context

| Total S | Actual prefill chunks | Standard global prefill | APA global prefill | Sliding prefill | Standard decode | APA decode |
|---:|---:|---|---|---|---|---|
| 4,096 | 12 | 12 standard de-expanded | 4 standard (`S_all <= 2048`), 8 blend (`2432..4096`), 0 fused | 2 causal, 10 band-mask | standard ring branch | blend for all 64 steps |
| 8,192 | 44 | 44 standard de-expanded | 4 standard, 7 blend (`2432..4032`), 33 fused (`4224..8192`) | 2 causal, 42 band-mask | standard ring branch | blend for all 64 steps |
| 16,384 | 172 | 172 standard de-expanded | 4 standard, 7 blend, 161 fused | 2 causal, 170 band-mask | standard ring branch | blend for all 64 steps |
| 32,768 | 428 | 428 standard de-expanded | 4 standard, 7 blend, 417 fused | 2 causal, 426 band-mask | standard ring branch | blend for all 64 steps |

The full per-chunk branch map is in
`artifacts/apamq_e3/path_audit.json`.

## Attribution tables

### Matrix status

| S | Standard | APA | Prefill pool / NVML peaks | Decode pool / NVML peaks | 64-step speed |
|---:|---|---|---|---|---|
| 4,096 | FAILED before load | FAILED before load | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| 8,192 | FAILED before load | FAILED before load | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| 16,384 | FAILED before load | FAILED before load | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| 32,768 | SKIPPED | SKIPPED | prerequisite matrix incomplete | prerequisite matrix incomplete | prerequisite matrix incomplete |

### Required S=16K attribution

| S=16,384 bucket | Standard | APA |
|---|---:|---:|
| Prefill CUDA-pool used high-water | NOT MEASURED — no CUDA device | NOT MEASURED — no CUDA device |
| Prefill NVML peak | NOT MEASURED — NVML rc=9 | NOT MEASURED — NVML rc=9 |
| Sliding-attention transient envelope | NOT MEASURED | NOT MEASURED |
| Global-attention transient envelope | NOT MEASURED | NOT MEASURED |
| Weight/body residency Z | NOT MEASURED | NOT MEASURED |
| Mask-cache residency W | NOT MEASURED | NOT MEASURED |
| Other/residual U | NOT COMPUTABLE | NOT COMPUTABLE |
| Decode CUDA-pool used high-water | NOT MEASURED | NOT MEASURED |
| Decode NVML peak | NOT MEASURED | NOT MEASURED |
| `kqb` resident | N/A by path | NOT MEASURED; dtype assertion not reached |
| Quantize/refine transients | N/A | NOT MEASURED |

The runner is prepared to record both true phase envelopes and per-call
attribution. It reads/reset CUDA default-mempool `UsedMemHigh` and
`ReservedMemHigh` directly through `libcudart`, preserves the outer phase peak
across nested resets, wraps sliding/global attention separately, and records
quantize, blend, refine-softmax, fused, causal-softmax, and MLP envelopes. The
1 s NVML poller is an independent side thread. These facilities were not
claimed as executed measurements after device initialization failed.

## Component sizes

### `kqb` ring

No `kqb` allocation occurred in the failed cells. The following are **exact
source/shape byte identities, not live measurements**. They use 8 global
layers × 1 KV head × 512 elements × 2 B/bfloat16. The as-built capacity is
`_grow_cap(S+1)` on the first decode step (`core/gemma4_tc.py:128-139,
545-550`); the perfect zero-capacity-overhead column stores exactly S rows.

| Prefill S | First-decode cap | As-built `kqb` total | Exact-S `kqb` total | Capacity overhead |
|---:|---:|---:|---:|---:|
| 4,096 | 6,144 | 48 MiB | 32 MiB | 16 MiB |
| 8,192 | 10,240 | 80 MiB | 64 MiB | 16 MiB |
| 16,384 | 18,432 | 144 MiB | 128 MiB | 16 MiB |
| 32,768 | 34,816 | 272 MiB | 256 MiB | 16 MiB |

The required in-process assertion is implemented after the first `L=1` APA
decode step: all eight global caches must have non-null `kqb`, dtype
`bfloat16`, and `kq_count == count`. It was **not reached**, so the stored dtype
is not reported as measured.

### Quantization and refine/tail rescore

Measured pool deltas are unavailable. Static source scaling from
`tensor_cuda/quant.py:111-132` is retained only to make the missing quantities
explicit:

| Quantize case | Input rows | One fp32 `(1,1,rows,512)` slab | Source live-set scale (~5–6 slabs) | Status |
|---|---:|---:|---:|---|
| 4K unchunked prefill | 4,096 | 8 MiB | about 40–48 MiB plus ~1 MiB rotation/table storage; A1 expected ~44–50 MiB | NOT MEASURED |
| >4K prefill chunk | 2,048 | 4 MiB | about 20–24 MiB | NOT MEASURED |
| first APA decode, cold chunk | 512 | 1 MiB | about 5–6 MiB | NOT MEASURED |
| steady incremental decode | 1 | 2 KiB | about 10–12 KiB | NOT MEASURED |

The refine/tail rescore is the blend envelope at
`core/mistral7b_tc.py:275-287`: materialized `bulk`, `rank`, and blended
weights. Its pool delta was **not measured**. The fused prefill branch streams
scores and does not materialize those S-wide matrices.

The only persistent APA table implied by the live geometry is one shared
fp32 512×512 rotation plus 16 centroids and 15 boundaries:
1,048,700 B = 1.000118 MiB. This is again a source byte identity; a live table
inventory was not reached.

## 110MB trace

The requested live 4K pool trace did not execute. The failed cells exclude no
allocator site dynamically. Static path narrowing does exclude the following
from the 4K prefill event:

- fused APA is not active at `S_all == 4096` because the gate is strict `>`;
- `kqb` is not allocated until the first `L=1` decode;
- the global blend builds no causal-mask tensor—causal bounds are passed to
  `apa_blend_softmax` (`core/mistral7b_tc.py:249-262`);
- model weights are a fixed baseline, not a per-attention allocator transient.

The strongest remaining static suspect is the APA blend live set. At the 4K
adaptive chunks, its three simultaneous bf16 score-shaped tensors
(`bulk`, `rank`, blended weights) reach 90 MiB at `(L,S_all)=(320,3072)` and
again at `(256,3840)`:

```text
3 × 16 heads × 320 queries × 3072 keys × 2 B = 94,371,840 B = 90 MiB
```

The live `q`, reconstructed `kq`, output, and projection tensors can occupy the
remaining scale toward ~110 MiB. This is a **narrowed suspect, not an
attribution**. A completed `UsedMemHigh` event trace is still required to
separate simultaneous live bytes from allocator reservation high-water and to
close the A1 item.

## Bound arithmetic

The registered T-bound requires measured inputs. Those inputs do not exist in
this run, so **no measurement-derived perfect-APA bound is reported** and no T2
adjudication is made.

For auditability, the following is a static byte-identity cross-check showing
the arithmetic the completed runner will replace with measured transient
envelopes. At S=16,384 the adaptive final chunk is L=64. Standard global
attention holds one bf16 score tensor and one bf16 softmax tensor; perfect
zero-ring-overhead APA is charged exact-S `kqb` rather than the as-built
capacity overhead:

```text
standard score       = 1×16×64×16384×2 B = 33,554,432 B = 32 MiB
standard softmax     = 1×16×64×16384×2 B = 33,554,432 B = 32 MiB
scheduled eliminable score/softmax                         = 64 MiB
irreducible exact-S kqb = 8×1×16384×512×2 B              = 128 MiB
shared APA tables     = 512×512×4 + 16×4 + 15×4 B   = 1.000118 MiB
static raw net input  = 64 - 128 - 1.000118         = -65.000118 MiB
```

The as-built S=16K ring is 144 MiB, but its 16 MiB capacity overhead is omitted
from that perfect/zero-ring-overhead cross-check. At optional S=32K, using the
same L=64 final-chunk identity:

```text
scheduled eliminable score/softmax = 2×16×64×32768×2 B = 128 MiB
irreducible exact-S kqb             = 8×32768×512×2 B   = 256 MiB
shared APA tables                                           = 1.000118 MiB
static raw net input                = 128 - 256 - 1.000118 = -129.000118 MiB
```

These two lines are **not substitutes for T-bound measurements**: the live
per-layer envelopes, weights, masks, cache residency, and residual are missing.
They are included only to freeze the formula and prevent a failed GPU cell from
being silently converted into a numerical verdict.

## Anomalies

- `/dev/nvidia*` was absent while NVIDIA kernel modules were listed by the
  host. CUDA allocation reported no device; NVML reported driver communication
  failure. This is an execution-environment wall, not a Gemma/APA OOM result.
- Every GPU receipt has zero NVML samples and records the poller error inline.
- No speed number is reported: instrumentation never reached a model token.
- No 32K cell was attempted because 4K–16K did not complete, satisfying the
  order's priority and timebox rail.
- `core/`, all existing documentation, model weights, and Project-Tensor were
  not modified. No temporary core probe was used, so no probe revert was
  necessary.

## Exact measurement commands run

```bash
python3 scripts/apamq_e3_attrib.py --audit-only --output artifacts/apamq_e3/path_audit.json
flock -w 7200 /tmp/forge-gpu.lock python3 scripts/apamq_e3_attrib.py --mode standard --seq-len 4096 --decode-steps 64 --refine 0.15 --output artifacts/apamq_e3/standard_s4096.json
flock -w 7200 /tmp/forge-gpu.lock python3 scripts/apamq_e3_attrib.py --mode apa --seq-len 4096 --decode-steps 64 --refine 0.15 --output artifacts/apamq_e3/apa_s4096.json
flock -w 7200 /tmp/forge-gpu.lock python3 scripts/apamq_e3_attrib.py --mode standard --seq-len 8192 --decode-steps 64 --refine 0.15 --output artifacts/apamq_e3/standard_s8192.json
flock -w 7200 /tmp/forge-gpu.lock python3 scripts/apamq_e3_attrib.py --mode apa --seq-len 8192 --decode-steps 64 --refine 0.15 --output artifacts/apamq_e3/apa_s8192.json
flock -w 7200 /tmp/forge-gpu.lock python3 scripts/apamq_e3_attrib.py --mode standard --seq-len 16384 --decode-steps 64 --refine 0.15 --output artifacts/apamq_e3/standard_s16384.json
flock -w 7200 /tmp/forge-gpu.lock python3 scripts/apamq_e3_attrib.py --mode apa --seq-len 16384 --decode-steps 64 --refine 0.15 --output artifacts/apamq_e3/apa_s16384.json
```
