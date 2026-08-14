# APAMQ-FB-D1 — Blend-vs-Fused Decode Divergence

Date: 2026-08-13. Evidence class: **INSTRUMENTED PORT MEASUREMENT**.
This report contains the registered facts, the implemented receipt protocol,
and adjudication material. It does not contain a D1/D2 verdict because this
worktree has no GPU and none of the new GPU legs was executed here.

## Registered lead facts

The following are inputs supplied by the lead and were not re-derived in this
worktree:

- On the canonical engine, the 64-step S=8192 greedy token streams matched,
  while max absolute logit delta was 12.852391 against the registered 0.5
  tolerance, so the raw-logit gate failed.
- The branch census was 512 blend and 512 fused decode calls: eight global
  layers over 64 steps on each path.
- The prior INT4 leg did not load the FA engine because the gate inserted the
  canonical TensorCUDA path at `sys.path[0]`.

No new GPU value has been added to those facts.

## D1 observation protocol

`scripts/apamq_fbd1_diag.py --leg bits4` runs blend decode as the D1 state
driver. At every global-layer call it shadow-runs fused attention on the exact
same q/k/kq/v tensor objects, records the comparison, and returns only the
blend output to the model. Thus the shadow probe cannot perturb the state of
the next layer or step. A separate fresh prefill replays the blend token-input
stream through fused decode for the end-to-end D2 logit comparison; a greedy
prediction mismatch is reported rather than allowed to change later inputs.

For each `(decode step, global layer)` pair, the receipt records:

- blend and fused refined-key counts;
- intersection, union, Jaccard, flip count, and flip fraction;
- max and mean absolute attention-output delta before `o_proj`;
- max and mean difference between the blend bulk score and the fused bulk
  score, plus the threshold difference;
- fused saved-threshold versus exact reduction-replay difference;
- the number of blend keys within four fp32 ULPs of the replayed threshold;
- an explicit same-state/same-tensor-input assertion;
- valid-key count, scale, z threshold, and blend-softmax call count.

The selection masks are not inferred from a generic cuBLAS fp32 score. The
blend hook captures the actual bf16 `bulk` matrix passed to
`apa_blend_softmax` and replays `apa_blend_softmax_kernel2`'s 256-thread
population-stat reduction. The fused hook obtains the actual saved threshold
from `apa_selective_fwd_train` and replays the D=512 warp-cooperative fp32 dot
on the exact bf16 q/kq inputs. At D=512, bf16 products are exactly
representable in fp32; the replay preserves the kernel's lane accumulation and
shuffle tree. The normal inference output remains the output of
`apa_selective_attention`; training-forward output parity is a separate
receipt field. The four-ULP blend-boundary count makes any host/device
division or square-root edge explicit instead of silently treating a
threshold-tie key as certain.

The aggregate correlates per-call flip count with max attention-output delta.
It also isolates the maximum output delta among calls whose masks are
identical. A nonzero identical-mask residual is evidence for score/softmax or
output-reduction effects rather than selection flips.

## Force-select-all control

The `force_all` leg sets `refine_percentile=1.0`, which resolves to `z=-inf`.
Both paths therefore select every valid key and use exact K scores. The
receipt requires an all-key mask on both sides and reports the remaining
attention/logit delta. This residual is the registered non-selection control;
it must not be silently attributed to key flips.

## D2 bits ladder

The ladder freezes S=8192, 64 replayed decode inputs, seed 20260813, and
`refine_percentile=0.15` while changing only `bulk_bits`:

| Leg | Bulk bits | Max/mean logit delta | Selection overlap | Refined fraction | Status |
|---|---:|---|---|---|---|
| `bits4` | 4 | lead receipt pending | pending | pending | NOT RUN HERE |
| `bits6` | 6 | lead receipt pending | pending | pending | NOT RUN HERE |
| `bits8` | 8 | lead receipt pending | pending | pending | NOT RUN HERE |

The CPU `summary` leg consumes those three receipts plus `force_all.json`. Its
registered “strong improvement” rule requires strict improvement in both max
and mean logit delta, nondecreasing overlap, nonincreasing flip fraction, and
the bits=8 max no greater than half the bits=4 max. A max-logit range within
10% of the bits=4 value is labeled flat. Other outcomes are labeled mixed or
weak rather than forced into support/refutation.

If the curve is flat, the same receipts bisect the registered semantic terms
in this order: valid-key population, scale, threshold parameter, same-state
tensor identity, fused threshold/reduction replay, bulk score precision, then the
identical-selection residual. `summary.json` names the first divergent term
supported by those receipts; if none is localized, it says so explicitly.

## D3 root and INT4 mechanics

The gate and E3 attribution script resolve TensorCUDA as:

```text
TENSOR_CUDA_ROOT when set; otherwise
/mnt/ForgeRealm/Project-Tensor/tensor_cuda
```

Resolution occurs before `tensor_cuda` import. The gate prints the resolved
root in its CPU check. `--int4` now compares blend decode with FA INT4-fused
decode using a common non-INT4 prefill and the blend input-token stream. It
also requires 512 INT4 calls and verifies all eight global rings retain
`kqb=None, kq_count=0`. Engines without the INT4 entry produce an explicit
capability SKIP.

The D1 harness has the same environment override. Its non-INT4 legs default
canonical, while its `int4` leg defaults to the FA engine so the required
one-command leg is self-contained. Explicit `TENSOR_CUDA_ROOT` always wins.

| `TENSOR_CUDA_ROOT` | Engine resolution | Default fused gate | Gate `--int4` | E3 `--int4` | D1 `int4` |
|---|---|---|---|---|---|
| unset | canonical TensorCUDA, except D1 `int4` | runnable | capability SKIP if entry absent | capability recorded/inert if absent | FA engine selected, runnable |
| `/mnt/ForgeRealm/wt/apamq-fa/tensor_cuda` | FA worktree engine | runnable | INT4 comparison runnable | FA INT4 entry is importable | FA INT4 entry is importable |
| invalid path | requested path precedes repo imports | import fails loudly | import fails loudly | import or protected-source check fails loudly | import fails loudly |

The INT4 ABI packs raw K internally and does not expose its selection mask.
The INT4 receipt therefore reports decode-logit parity, pre-`o_proj`
attention delta, branch census, and ring drop, while marking selection overlap
`not_exposed_by_int4_abi` rather than inventing it.

## D4 adjudication material (no choice made)

The fused kernel is the implementation that mirrors the selective Rust
reference semantics: bulk scores and population statistics stay fp32 in the
kernel, selection depends only on `abs(bulk)`, selected keys use exact scores,
and unselected keys retain bulk scores. The blend is the composed
approximation: cuBLAS materializes bf16 bulk/rank matrices before the fused
blend-softmax operation.

The existing raw-logit tolerance is not eligible for silent respec. An honest
G-B1 tolerance must be attached to an explicit tuple
`(engine build, S, bulk_bits, refine_percentile, steps, token-input policy)` and
must cover repeated measured maxima plus an adjudicated repeatability margin.
Until the lead receipts exist, there is no honest new numeric tolerance or
operating point to write here. A token-match result alone does not establish a
raw-logit tolerance.

Options for the lead and David, intentionally not selected here:

1. Stabilize selection across composed and fused precision domains.
2. Raise bulk bits and register the chosen bits/refine operating point.
3. Gate task-level behavior or token decisions instead of raw logits, while
   retaining raw-logit drift as a reported diagnostic.

## Exact lead commands

Run one command per leg. The first five are GPU legs; `summary` is a CPU
receipt reducer and must run after the four non-INT4 measurement receipts.

```bash
python3 scripts/apamq_fbd1_diag.py --leg bits4 --output artifacts/apamq_fbd1/bits4.json
python3 scripts/apamq_fbd1_diag.py --leg bits6 --output artifacts/apamq_fbd1/bits6.json
python3 scripts/apamq_fbd1_diag.py --leg bits8 --output artifacts/apamq_fbd1/bits8.json
python3 scripts/apamq_fbd1_diag.py --leg force_all --output artifacts/apamq_fbd1/force_all.json
python3 scripts/apamq_fbd1_diag.py --leg int4 --output artifacts/apamq_fbd1/int4.json
python3 scripts/apamq_fbd1_diag.py --leg summary --output artifacts/apamq_fbd1/summary.json
```

Each measurement command prints a measurement PASS/FAIL line, token/logit
summary, selection summary when the ABI permits it, raw-logit gate PASS/FAIL,
and the receipt path. Measurement completion and raw-logit parity are separate
statuses so a successfully captured divergence remains a valid receipt.

## In-sandbox checks

Only CPU/static checks are authorized here. Expected commands are:

```bash
python3 -m py_compile scripts/apamq_fbd1_diag.py scripts/apamq_e3_attrib.py tests/gemma4_apa_decode_fused.py
python3 scripts/apamq_fbd1_diag.py --leg cpu-check --output /tmp/apamq_fbd1_cpu.json
python3 tests/gemma4_apa_decode_fused.py --cpu-check
TENSOR_CUDA_ROOT=/mnt/ForgeRealm/wt/apamq-fa/tensor_cuda python3 tests/gemma4_apa_decode_fused.py --cpu-check
```

GPU measurements remain **NOT RUN HERE**.

Observed CPU/static results:

```text
py_compile: PASS
CPU REDUCTION REPLAYS: PASS
CPU FORCE-ALL CONTROL: PASS
canonical gate CPU FLAG MATRIX: PASS
canonical gate CPU ENGINE ROOT RESOLUTION: PASS
canonical INT4 ENGINE ENTRY: ABSENT (GPU ring leg SKIP)
FA gate CPU FLAG MATRIX: PASS
FA gate CPU ENGINE ROOT RESOLUTION: PASS
FA INT4 ENGINE ENTRY: PRESENT
E3 canonical audit-only: audit_only_complete, protected files unchanged
E3 FA audit-only: audit_only_complete, protected files unchanged
pytest targeted: 1 passed, 1 skipped, 2 import deprecation warnings
D1 INT4 self-contained FA-root resolution: PASS
D1 explicit TENSOR_CUDA_ROOT precedence: PASS
```

`ruff` was not installed, so no ruff result is claimed. Python compilation,
CLI parsing, the targeted pytest, both engine-root imports, both E3 static
audits, and the reduction/force-all CPU checks passed.
