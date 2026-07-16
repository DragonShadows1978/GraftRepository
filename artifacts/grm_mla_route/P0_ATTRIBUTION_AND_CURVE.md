# GRM MLA CUDA Route — P0 Receipts

Work order: `docs/GRM_MLA_CUDA_ROUTE_PLAN.md` (immutable). Branch
`grm-cuda-bridge-overhead` @ `26cbda3`. Host: MilleniumFalcon, 12 vCPU,
62GB RAM, RTX 4070 SUPER idle throughout (0% util, 287MiB baseline desktop
usage only — P0 is host routing, no GPU path touched). Ambient loadavg
1.7-4.1 across runs (never a quiet machine; noted per run below).

New scripts (read-only w.r.t. product code): `scripts/grm_mla_route_profile.py`,
`scripts/grm_mla_route_wall_curve.py`.

## (a) Correctness re-stamp

```
python3 -m pytest tests/test_grm_router_baseline.py -v   -> 21 passed in 83.58s
python3 -m pytest tests/test_grm_native_runtime.py -q    -> 95 passed, 2 warnings in 243.39s
```

Combined: **116 passed, 0 failed** (21 + 95). No stash-check needed — working
tree at run time carried only unrelated doc edits
(`docs/GPT_OSS_20B_APA_GRM_LEDGER.md`, `docs/GPT_OSS_20B_APA_GRM_SYNTHESIS.md`,
from another track) plus one untracked scratch file
(`scripts/qwen35_graph_parity.py`); no product-code diff on this branch tip.

Deviation from the plan's "~15 and ~95" expectation: router-baseline suite is
**21**, not ~15. This is real growth, not a regression — `git log` on the file
shows GQA CUDA bridge/sidecar tests (smoke, probe, device-entry parity,
capture compaction, batched/exhaustive parity benchmarks) added after the
2026-07-03 receipt that set the "~15" figure
(`docs/GRM_GEMV_ROUTER_PLAN.md`: `109 passed` combined on 2026-07-03, which
implies baseline was 109-95=14 then — matches "~15"). Native-runtime suite is
exactly the expected **95**. Zero failures either way; correctness stands.

## (b) Production-path profile — attribution

**The suspicion is confirmed, not unfounded.** `scripts/grm_router_baseline.py`'s
`NativeRouter.route()` is a bare ctypes call straight into the C ABI
(`grm_store_route`) — it never touches `ArenaCache.route()`. Real turn
execution calls `ArenaCache.route()` (`core/graft_arena.py:1224`, inside
`step()`). That wrapper does real O(N) Python work on every call, and a
second, sharper defect makes it worse: `_native_route_order`
(`core/graft_arena.py:451-453`) calls the native store with
**`topk=len(self.grafts)`** — the full node count — regardless of the
caller's actual `limit`. Confirmed directly (100-node/500-node spy harness):
a `route(..., limit=3)` call internally issues `store.route(..., topk=500)`.
Consequences, all O(N) per call:
  - the native call itself scores/sorts and marshals back **all N** node ids
    across the ctypes boundary (`ctypes.c_uint64 * N` allocated fresh every
    call), not just the caller's top-3;
  - `_native_route_order`'s post-route remap loop then walks all N returned
    ids through a Python dict (`native_to_idx.get(...)`);
  - `route()`'s own `cand = [i for i in range(len(self.grafts)) if ...]`
    candidate list, and `_native_route_order`'s `native_to_idx` build loop,
    are separately O(N).

### Attribution table (ms/call, pstats own-tottime, additive)

cProfile region: 8 reused steady-state `arena.route(probe, limit=3)` calls
after 1 cold call, dim=128, harvested corpus (`--max-files 2048`,
`max_vectors=8192` bank). fp32 = flat native scan; `int4_bounded_staging` =
`GRM_ROUTER_INT4=1 GRM_ROUTER_INT4_REFINE_M=128` (the receipted operating
point). Uninstrumented control column is the clean (no cProfile) median wall
over 8 reused calls, for cross-check against the profiled-region median
(profiled ~2.7-3x control from cProfile's own per-call dispatch cost — a
constant multiplier, doesn't change the attribution ratios below).

| nodes | mode | native_call | route() cand build | native_to_idx build | remap dict.get | remap list.append | named total | control median wall | native/(control) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100k | fp32 | 34.5 | 32.2 | 85.7 | 75.1 | 12.1 | 239.9 | 70.6 ms | 49% |
| 100k | int4 M=128 | 44.0 | 34.0 | 87.7 | 79.0 | 12.7 | 257.5 | 82.9 ms | 53% |
| 1M | fp32 | 326.5 | 346.0 | 894.5 | 897.6 | 125.6 | 2590.4 | 919.7 ms | 35% |
| 1M | int4 M=128 | 415.8 | 346.0 | 883.8 | 905.9 | 127.1 | 2678.7 | 974.9 ms | 43% |

(all other named buckets — `_rare_tokens`, ctypes marshal helpers,
python-fallback scoring — sum to <0.1ms/call at every point; `other`
(unattributed) is 0.05-0.08ms/call, i.e. <0.03% of route time.)

Raw receipts: `profile_100000n_fp32.pstats` /
`profile_int4_100k_100000n_{fp32,int4_bounded_staging}.pstats` /
`profile_1000000n_fp32.pstats` /
`profile_int4_1m_1000000n_{fp32,int4_bounded_staging}.pstats`, plus matching
`*_cumulative.txt` pstats text tables and `*_summary.json` full dumps
(includes per-run harvest stats and per-call control/profiled distributions).

**E1 verdict: MET, decisively.** ≥99.9% of the production-vs-native gap is
named at every node count/mode (100.0% - "other" ∈ [0.02%, 0.03%] of total
named tottime). The two dominant named buckets are the O(N) `native_to_idx`
dict build (`_native_route_order`'s own frame) and the O(N) `dict.get` remap
of the full N-length native result — both directly caused by the
`topk=len(self.grafts)` call, not by anything algorithmically necessary for a
`limit=3` route.

**Material-or-not verdict: MATERIAL, not close to the <10% skip threshold.**
The wrapper gap (control wall minus native_call) is **47-65%** of total route
time at every point measured: 100k fp32 (70.6-34.5)/70.6=51.1%, 100k int4
(82.9-44.0)/82.9=46.9%, 1M fp32 (919.7-326.5)/919.7=64.5%, 1M int4
(974.9-415.8)/974.9=57.3%. P1 is squarely in scope per the plan's own clause.

## (c) Production wall curve vs historic native-only

Historic native-only receipts (`docs/GRM_GEMV_ROUTER_PLAN.md`, P2/P3 notes,
harvested dim128 corpus, `NativeRouter.route()` bare ctypes call, 8-10
measured queries after warmup):

| nodes | native-only fp32 p50 | native-only INT4 M=128 p50 / p95 |
|---:|---:|---:|
| 100k | 22.8177 ms | 9.0242 ms (M=128, one of the small-M sweep points; bounded-staging variant not separately receipted at 100k) |
| 1M | 211.3071 ms | **23.8805 ms p50 / 25.4883 ms p95** (bounded-staging, the receipted operating point this plan cites) |

Through-Python production curve, this run (`ArenaCache.route()`,
uninstrumented, `limit=3`, dim128, same harvested-corpus generation as the
historic runs; queries=20-24 measured after warmup unless noted):

| nodes | mode | p50 | p95 | backend |
|---:|---|---:|---:|---|
| 1k | fp32 | 0.627 ms | 0.637 ms | native |
| 1k | int4 M=128 | 0.680 ms | 0.697 ms | native |
| 10k | fp32 | 6.453-6.448 ms | 7.193-7.259 ms | native |
| 10k | int4 M=128 | 6.676 ms | 6.990 ms | native |
| 100k | fp32 | 67.5-72.7 ms | 74.3-78.2 ms | native |
| 100k | int4 M=128 | 78.5 ms | 92.4 ms | native |
| 1M | fp32 | 849.9-873.4 ms | 868.0-889.2 ms | native |
| 1M | int4 M=128 | **925.6 ms** | **953.6 ms** | native |

(1k/10k/100k fp32 ranges reflect two independent measured runs at ambient
load 1.7-3.4; consistent within ~7%. 1M points are single runs each, given
per-run cost — populate loop alone is the dominant wall-clock cost of each
invocation at 1M, not the routed queries.)

**Bottom line: production 1M route (INT4 bounded-staging M=128, the
receipted operating point) is 925.6ms p50 / 953.6ms p95 through the real
`ArenaCache.route()` call path — ~38.8x the historic 23.88ms p50 native-only
number, ~37.4x the 25.49ms p95 number.** The historic receipt is not wrong on
its own terms (it faithfully times the native call), but it is not what a
real turn pays. David's suspicion is confirmed: production 1M routing is far
worse than 24ms, and the multiplier is large enough that "far worse" is an
understatement — it is two orders of magnitude closer to 1 second than to 25ms.

Raw receipts: `wall_curve_1k_10k_100k.json`, `wall_curve_1m.json`,
`wall_curve_1m_int4_only.json`, `wall_curve_10k_100k_int4.json`,
`wall_curve_1k_int4.json`.

## Host state per run

- 100k fp32 (first): loadavg 2.11/2.64/2.97 pre-run.
- 100k fp32+int4 sweep: concurrent with 1M fp32 profile bg run, loadavg ~2.4-2.9.
- 1M fp32 profile: loadavg 2.38/2.52/2.88 pre-run, ran alone.
- 1M fp32+int4 profile: loadavg 2.52/2.57/2.86 pre-run, ran alone (prior 1M fp32 profile had completed).
- 1k/10k/100k fp32 wall curve: loadavg 2.96/2.70/2.88, concurrent with 1M int4 profile bg run.
- 1M fp32 wall curve: concurrent with 1M int4 profile bg run, loadavg climbed to ~3.4-4.1 (two 1M-scale jobs at once).
- 1M fp32+int4 wall curve (final, both modes together): loadavg 4.12/3.22/3.05 pre-run, ran alone.
- 10k/100k fp32+int4 wall curve, 1k fp32+int4 wall curve: loadavg ~1.7-2.6, ran alone.
- GPU: idle (0% util, 287MiB baseline) at every check — no GPU-touching code ran; P0 is host routing only, as the plan specifies.

## Summary verdicts

- **E1 (≥80% of gap named): MET**, ≥99.9% named at every point.
- **Material-or-not: MATERIAL.** Wrapper gap is 47-65% of total route wall
  time at every measured point — nowhere near the plan's <10% skip threshold.
  P1 (wrapper fixes) is in scope and should proceed.
- **Production path is NOT identical to what the harness times.** The
  suspicion was correct: `grm_router_baseline.py` never exercised
  `ArenaCache.route()`; the 23.88ms/25.49ms receipt is native-call-only.
  Through the real wrapper, 1M INT4 bounded-staging production routing
  measures ~925.6ms p50 / ~953.6ms p95 — a ~38x gap, dominated by two
  wrapper defects: (1) genuinely O(N) Python bookkeeping (`cand` build,
  `native_to_idx` build) that would exist even at correct topk, and (2) the
  sharper, avoidable defect that `_native_route_order` requests
  `topk=len(self.grafts)` from the native call instead of the caller's
  actual `limit`, forcing both the native call and the Python remap to move
  all N results instead of top-`limit`. (2) is the more obviously fixable
  target for P1 — profile-guided, per the plan's P1 scope (cache what is
  re-derived, epoch-gate what is re-checked; this adds "don't ask the native
  layer to rank more than the caller wants").
