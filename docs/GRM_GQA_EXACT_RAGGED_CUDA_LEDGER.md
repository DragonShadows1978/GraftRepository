# GRM GQA Exact Ragged CUDA Router Ledger

This ledger is the execution record for
`docs/GRM_GQA_EXACT_RAGGED_CUDA_PLAN.md`. The wing narrative continues in
`docs/GRM_GEMV_ROUTER_SYNTHESIS.md`.

## 2026-07-11 — Work Order Opened

Repository state:

- Source repository: `/mnt/ForgeRealm/GraftRepository`
- Source revision: `fdc478ca28a6f74da6570748b80ba67410dc45fb`
- Isolated worktree: `/home/vader/GraftRepository-gqa-ragged-cuda-router`
- Branch: `codex/gqa-ragged-cuda-router`
- Opening tree: clean
- Commits/pushes: none authorized

Predecessor disposition:

- The stopped fixed-card branch is archived as stash commit
  `68c9f0a7945e1872aeb9aba5439a0ee09fc806f2` with implementation, docs, and
  negative Qwen hybrid receipts.
- Development-visible source recall was 18/20 for full ragged raw keys versus
  2/3/4 for source cards and 3/2/3 for hybrid cards at 2/4/8 slots.
- Interpretation frozen from that result: the model-native route signal is too
  distributed for the tested fixed-slot representation; generated aliases did
  not repair the loss. The successor must preserve raw keys.

P0 source inspection:

- `GQAArenaCache._cuda_route_bank_signature()` currently rejects a bank when
  any eligible raw key row differs in full `[kv_heads,tokens,head_dim]` shape.
- `_cuda_route_bank_inputs()` then stacks only identical shapes and attaches
  the result to the already-cached native CUDA sidecar.
- The CUDA score kernel consumes a dense fixed token extent and uses the same
  non-negative absolute-dot maximum law as the CPU scorer.
- Therefore unequal token lengths are a localized representation blocker. For
  non-empty rows, zero tails are score-neutral and preserve every source key.
- `child_cents` remains excluded: concatenating multiple route rows before the
  per-head mean would change the established reduction order.

Frozen initial action:

1. Add focused mixed-length/padding tests.
2. Implement exact padded leaf-bank construction behind the existing opt-in.
3. Run focused lifecycle and native-runtime gates.
4. Run the registered real-Qwen 32/128/512 development gate.

No performance or quality result is claimed at work-order opening.

## 2026-07-11 — P1/P2 Focused Implementation Gate

Implementation:

- Extended the arena's existing immutable CUDA-bank builder to accept leaf
  rows with a common KV-head/head-dim geometry and different positive token
  extents.
- Equal-shaped rows retain the existing stack path. Mixed rows are copied
  byte-for-byte into zero-filled prefixes of one fp32 max-token bank.
- Added a layout receipt with row lengths, raw/padded values and bytes, and
  padding ratio. Non-finite, empty, incompatible, missing-ID, and hierarchical
  rows fail closed before attachment.
- The existing epoch cache, native sidecar, stable node mapping, CPU fallback,
  and `GRM_GQA_CUDA_ROUTE=1` opt-in remain in force.

Focused receipts:

- `python3 -m pytest -q tests/test_gqa_ragged_cuda_bank.py` ->
  `11 passed, 2 warnings in 0.20s`.
- `python3 -m pytest -q tests/test_grm_native_runtime.py -k 'gqa and cuda'`
  -> `11 passed, 110 deselected, 2 warnings in 9.70s`.
- A plain `pytest` invocation failed during collection because that entrypoint
  omitted the repository root from `sys.path`; the module-style project
  invocation above is the executed test result.

## 2026-07-11 — Real-Qwen 32-Node CUDA Smoke

Command:

```text
python3 scripts/grm_gqa_exact_ragged_cuda_gate.py \
  --node-counts 32 --query-counts 4 --route-repeats 1 \
  --out /tmp/grm_gqa_ragged_cuda_smoke.json
```

The first sandboxed attempt reached CUDA arena creation and returned code
`1100` (`cudaErrorNoDevice`): the managed filesystem sandbox exposed no
`/dev/nvidia*`. Re-running the identical command with GPU-device access is the
receipt below; the sandbox failure is environmental, not a product result.

Result (stored model-native Q and K, Qwen3.5-9B target capture layer 3,
length schedule `{32,48,64,96,128,192,256}`):

- Exact top-k parity: 4/4 queries at k `{1,3,5,16}` across ragged NumPy,
  native CPU, direct padded CUDA, and the arena bridge; zero tie or non-tie
  mismatches.
- CUDA backend engagement: 4/4.
- Padding: 14,352,384 raw bytes -> 33,554,432 padded bytes, ratio
  `2.3378995434x` (memory rail green).
- Resident bridge: p50 `0.5880 ms`, p95 `0.6660 ms`.
- Sampled process VRAM: `226 MiB` after attach, direct route, and every bridge
  sample; no monotonic growth. This is sampled residency, not allocator peak.
- Cold host bank build: `5.376 ms`.
- First attach/upload: `251.900 ms`; total cold `257.276 ms` -> seven
  milliseconds over the 250 ms rail.

Verdict: correctness green; preliminary `QUALITY-GREEN / COMPUTE-RED` on the
cold rail. Do not retune the rail. Proceed to the registered 32/128/512 gate;
the full result decides whether exact global padding survives or hands off to
length buckets/segmented offsets.

## 2026-07-11 — P3 Sealed Development Gate, Harness Correction

The first full seed-`20260712` run was retained at
`artifacts/grm_gqa_exact_ragged_cuda/dev_gate_seed20260712.json` and reported
one 128-node top-16 mismatch. Attribution showed it was a harness defect, not
a padded-route defect:

- Native CPU over the original ragged rows, direct padded CUDA, and the arena
  bridge all returned the same order.
- The harness's algebraically grouped-head NumPy einsum placed nodes 117/2 in
  the opposite order. Their grouped scores differed by `9.54e-7`; the
  production `_key_score` scores differed by `-3.87e-7` and agreed with all
  three runtime paths.
- Regrouping changed fp32 accumulation order. The gate now calls
  `GQAArenaCache._key_score` verbatim and has a regression preventing another
  "optimized" reference from replacing it.
- The seed, selected captures, length schedule, policies, and product code
  were not retuned.

The corrected-reference v2 receipt
(`artifacts/grm_gqa_exact_ragged_cuda/dev_gate_seed20260712_v2.json`) was
parity-green at 32/128/512 but remained `QUALITY-GREEN / COMPUTE-RED`:

| Nodes | Queries | exact parity | CUDA engaged | padding ratio | bridge p50 | bridge p95 | cold build |
| ---: | ---: | :---: | :---: | ---: | ---: | ---: | ---: |
| 32 | 25 | yes | 25/25 | 2.3379x | 0.289 ms | 0.572 ms | 249 ms |
| 128 | 50 | yes | 50/50 | 2.2189x | 0.578 ms | 0.862 ms | 225 ms |
| 512 | 100 | yes | 100/100 | 2.1992x | 1.605 ms | 1.972 ms | 896 ms |

All 175 queries matched at top-k `{1,3,5,16}` across the production NumPy
law, ragged native CPU, direct CUDA, and the arena bridge. The only failed rail
was the 512-node cold build.

## 2026-07-11 — Cold Attribution And Exact Attach Fix

The one-query 512-node attribution receipt is
`artifacts/grm_gqa_exact_ragged_cuda/cold_attribution_512_baseline.json`:

- host zero-padding build: `95.59 ms`;
- attach host overhead: `719.72 ms`;
- CUDA arena create: `253.80 ms`;
- total: `1069.11 ms`.

Source inspection matched the timing. The arena built an immutable bank behind
an epoch/signature token, then `CudaGQARouteSidecar.create_bank()` hashed all
512 MiB again. CUDA then allocated a 512 MiB temporary `d_keys`, uploaded the
bank, allocated the final 512 MiB head-major matrix, repacked on device,
synchronized, and freed the temporary.

The correction preserves every value and both public contracts:

1. The epoch-governed arena attach passes its already-authoritative signature
   through `NativeGraftStore`; direct/public sidecar callers still receive the
   deterministic content hash by default.
2. CUDA copies `[node,kv_head,token,dim]` host storage directly into the final
   `[kv_head,node*token,dim]` matrices with one pitched 2-D copy per KV head.
   This is the same index mapping as the removed pack kernel, without the
   bank-sized temporary allocation or repack.

A new capability guard also closes an inherited `main` regression discovered
by the full lifecycle suite: `GraftRepository` accepted custom `arena_cls`
implementations but unconditionally called their CUDA epoch hook. Clean source
revision `fdc478ca` reproduces the failure. Repository mutations now invoke the
hook only when the arena exposes it; production ArenaCache dialects still bump
on every registered mutation, and both GQA mutation batteries remain green.

## 2026-07-11 — P3/P4 Final Development Verdict

Final sealed receipt:
`artifacts/grm_gqa_exact_ragged_cuda/dev_gate_seed20260712_v3.json`.

| Nodes | Queries | exact parity | CUDA engaged | padded bytes | ratio | bridge p50 | bridge p95 | cold build |
| ---: | ---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 25 | yes | 25/25 | 32 MiB | 2.3379x | 0.308 ms | 0.630 ms | 211.71 ms |
| 128 | 50 | yes | 50/50 | 128 MiB | 2.2189x | 0.581 ms | 0.833 ms | 38.62 ms |
| 512 | 100 | yes | 100/100 | 512 MiB | 2.1992x | 1.591 ms | 2.007 ms | 147.09 ms |

Required gates:

- Production NumPy / ragged native CPU / direct padded CUDA / arena bridge:
  175/175 query parity at k `{1,3,5,16}`, zero exact-tie or non-tie misses.
- CUDA backend: 175/175 semantic bridge routes.
- 512-node resident latency: p50 `1.591 ms`, p95 `2.007 ms` against rails
  `5/8 ms`.
- Padding: `244,121,600` raw bytes -> `536,870,912` padded bytes,
  `2.1991946309x` against the `2.5x` rail.
- Cold build: `94.75 ms` host build + `52.34 ms` attach = `147.09 ms`
  against the `250 ms` rail.
- Sampled process VRAM at 512: `706 MiB` after attach, `708 MiB` after route
  scratch, then `708 MiB` at all five steady samples. This is sampled
  residency, not allocator peak.

Verdict: `QUALITY-GREEN` for the registered development envelope. Retain exact
global padding as the development winner; do not start length buckets or a
segmented-offset kernel without evidence outside the registered ratio/latency
envelope. Hierarchical `child_cents` remain CPU-only because their exact
row-to-node reduction is still separate work.

Regression receipts on the final tree:

- New exact/property/cache/signature tests plus existing GQA CUDA selector:
  `25 passed, 110 deselected, 2 warnings in 9.08s`.
- Full native runtime: `121 passed, 2 warnings in 277.60s` (before the later
  capability guard); both affected GQA epoch batteries rerun green afterward.
- Full runtime lifecycle after the inherited capability fix:
  `101 passed, 2 warnings in 127.48s`.
- Router baseline: `21 passed in 80.51s`.
- Existing same-shape Qwen3.5-2B bridge compatibility smoke: parity true on
  4/4 queries; reused bridge minimum `0.446 ms`, direct CUDA
  `0.0996 ms/query`.

One attempted combined pytest invocation of the legacy model files was
interrupted after 7m52s. Those files execute full model experiments at import
and branch on `sys.argv`; they are not a composable pytest suite. The partial
run reproduced its existing feature-A receipt (`5/6` at trips=0, `6/6` at
trips=2) before entering consolidation, but it is not counted as a completed
gate.

No commits or pushes were made. The feature remains disabled by default;
fresh dual-GQA adoption evidence and an operator decision remain a separate
work order.
