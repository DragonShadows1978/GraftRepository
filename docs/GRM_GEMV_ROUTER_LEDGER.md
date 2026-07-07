# GRM GEMV Router Ledger

This ledger records the operational trail for the GRM GEMV/GQA router scaling
track. The implementation plan remains `docs/GRM_GEMV_ROUTER_PLAN.md`; the
measured scaling report remains `docs/ROUTER_SCALING_REPORT.md`.

## 2026-07-07

Action: Closed the stale CUDA route-bank lifecycle gap in the opt-in GQA CUDA
arena bridge.

Repo state:
- Repository: `/mnt/ForgeRealm/GraftRepository`
- Branch: `codex/intn-model-ppl-sweep`
- Pre-existing dirty files outside this slice included GPT-OSS implementation,
  GPT-OSS docs, GPT-OSS scripts, and GPT-OSS scaffold tests. They were not
  changed for this router slice.

Implementation:
- Added `gqa_route_bank_signature()` in `core/grm_cuda_router.py` for explicit
  dense-bank content signatures.
- `CudaGQARouteBank` now carries the direct bank signature.
- `NativeGraftStore` now tracks `_cuda_gqa_bank_signature` and clears it with
  the attached CUDA bank.
- `GQAArenaCache` now binds auto-attached CUDA route banks to a cheap
  route-snapshot signature over native node ids, key shape, dtype, and route-key
  object identity.
- Route calls reuse the GPU bank only when that signature still matches.
  Appended or replaced dense GQA route rows force a fresh
  `configure_cuda_gqa_route_bank()` attachment instead of reusing stale GPU row
  state.

Regression added:
- `test_gqa_arena_rebuilds_cuda_route_bank_when_rows_change`

Focused validation:

```bash
PYTEST_ADDOPTS='-p no:cacheprovider' python3 -m pytest \
  tests/test_grm_native_runtime.py::test_native_gqa_cuda_route_requires_explicit_bank \
  tests/test_grm_native_runtime.py::test_native_gqa_route_mutation_clears_cuda_bank \
  tests/test_grm_native_runtime.py::test_native_gqa_eligibility_mutation_clears_cuda_bank \
  tests/test_grm_native_runtime.py::test_gqa_arena_uses_opt_in_cuda_route_bank \
  tests/test_grm_native_runtime.py::test_gqa_arena_rebuilds_cuda_route_bank_when_rows_change \
  tests/test_grm_native_runtime.py::test_gqa_arena_skips_cuda_route_for_lexical_queries
```

Result: `6 passed, 2 warnings in 9.65s`.

Broader validation:

```bash
PYTEST_ADDOPTS='-p no:cacheprovider' python3 -m pytest tests/test_grm_native_runtime.py
```

Result: `90 passed, 2 warnings in 248.88s`.

Syntax check:

```bash
PYTHONPYCACHEPREFIX=/tmp/grm_pycache python3 -m py_compile \
  core/grm_cuda_router.py core/grm_native.py core/graft_arena.py \
  tests/test_grm_native_runtime.py
```

Result: passed.

Tracking updates:
- Updated `docs/GRM_GEMV_ROUTER_PLAN.md` with the P4 CUDA route-bank lifecycle
  note and validation receipts.
- Updated `docs/ROUTER_SCALING_REPORT.md` follow-up state.
- Added `docs/GRM_GEMV_ROUTER_SYNTHESIS.md`.
- Updated `/mnt/ForgeRealm/AI_Research_Board.md`.

Remaining work after this slice:
- Decide whether CUDA should remain a limited mount-window path or grow a
  larger-topk/full-rank `arena.route()` replacement contract.
- If sub-10ms full-bank GQA routing is required at larger real-capture points,
  implement a lower-level GEMM/BLAS-style layout or packed microkernel. The
  measured scalar/transposed host experiments are rejected as defaults.

## Backfilled Closure State From Existing Plan/Report

The GEMV router implementation is already closed for the host/non-GPU boundary:
P0-P6 shipped with measured MLA/GQA routing receipts, INT4 bulk/refine host
routing, prepared epoch snapshots, opt-in CUDA GQA route attachment, and the
limited `GRM_GQA_CUDA_ROUTE=1` arena bridge. The prior final non-GPU closure
gate in the plan/report was:

```bash
PYTEST_ADDOPTS='-p no:cacheprovider' python3 -m pytest \
  tests/test_grm_router_baseline.py tests/test_grm_native_runtime.py -q
```

Result: `109 passed, 2 warnings in 344.94s` on 2026-07-03.

Action: Ran live CUDA bridge validation against real Qwen3.5-2B GQA capture
banks on the visible 4070 Super process.

Commands:

```bash
env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache \
  python3 scripts/grm_gqa_cuda_bridge_smoke.py \
  --capture-dir /mnt/ForgeRealm/qwen35_graft_translation_poc/captures \
  --capture-role source --capture-layer 3 --capture-limit 128 \
  --node-count 32 --queries 2 --query-tokens 4 --topk 5 \
  --route-repeats 2 \
  --out artifacts/grm_gqa_cuda_bridge/qwen35_2b_layer3_32n_smoke.json \
  --markdown-out artifacts/grm_gqa_cuda_bridge/qwen35_2b_layer3_32n_smoke.md
```

```bash
env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache \
  python3 scripts/grm_gqa_cuda_bridge_smoke.py \
  --capture-dir /mnt/ForgeRealm/qwen35_graft_translation_poc/captures \
  --capture-role source --capture-layer 3 --capture-limit 192 \
  --node-count 128 --queries 3 --query-tokens 4 --topk 5 \
  --route-repeats 3 \
  --out artifacts/grm_gqa_cuda_bridge/qwen35_2b_layer3_128n_smoke.json \
  --markdown-out artifacts/grm_gqa_cuda_bridge/qwen35_2b_layer3_128n_smoke.md
```

```bash
env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache \
  python3 scripts/grm_gqa_cuda_bridge_smoke.py \
  --capture-dir /mnt/ForgeRealm/qwen35_graft_translation_poc/captures \
  --capture-role source --capture-layer 3 --capture-limit 600 \
  --node-count 512 --queries 3 --query-tokens 4 --topk 5 \
  --route-repeats 3 \
  --out artifacts/grm_gqa_cuda_bridge/qwen35_2b_layer3_512n_smoke.json \
  --markdown-out artifacts/grm_gqa_cuda_bridge/qwen35_2b_layer3_512n_smoke.md
```

Live CUDA bridge results:

| Nodes | Queries | Parity | Reused bridge min wall | Direct CUDA route wall | Direct CUDA device/query |
| ---: | ---: | --- | ---: | ---: | ---: |
| 32 | 2 | true | `3.15747ms` | `0.128642ms` | `0.098496ms` |
| 128 | 3 | true | `11.217836ms` | `0.255081ms` | `0.226304ms` |
| 512 | 3 | true | `38.960699ms` | `0.770151ms` | `0.740352ms` |

Interpretation:
- The opt-in `GRM_GQA_CUDA_ROUTE=1` arena bridge is live on real Qwen3.5-2B
  layer-3 capture banks, not just on the standalone CUDA probe.
- Bridge results matched the batched Python raw `|q.k|` reference and direct
  `NativeGraftStore.route_gqa_cuda()` results on all tested node counts.
- The first bridge route includes bank setup and is intentionally not the
  steady-state latency number. The reused bridge route is the relevant runtime
  path after the bank is resident.
- At 512 full 256-token K-bank nodes, the bridge is parity-green but not yet a
  sub-10ms full-bank path. That reinforces the current contract: CUDA is a
  limited mount-window accelerator unless a later work order builds a lower
  level full-rank routing kernel.
