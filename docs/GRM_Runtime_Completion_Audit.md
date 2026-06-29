# GRM Runtime Completion Audit

Date: 2026-06-29

This audit tracks the requested end state: RAM-authoritative graft storage,
durability, metadata/fact semantics, C++/CUDA runtime boundaries, and
validation where feasible.

## Completed

- Python RAM-authoritative graft payloads:
  `core/graft_repository.py` stores `host_payload` for new grafts and treats
  VRAM `h` as a disposable device copy.
- Device eviction no longer requires disk durability:
  `_page()` can evict dirty nodes from VRAM after a RAM snapshot exists.
- Manifest reload now enforces the same VRAM budget as live ingest:
  `load()` rebuilds host payloads, syncs the native mirror, then calls
  `_page()` so a resumed repository does not remount every graft onto device.
- Durability from RAM:
  `flush_now()` writes node payloads from `host_payload`, not from CUDA tensors.
- Async durability:
  `flush_async()` starts a background writer and `flush_wait()` surfaces errors.
- WAL:
  lightweight WAL records are written for node upsert, metadata updates,
  review candidates, correction/forget commands, and checkpoints.
- WAL recovery (functional rehydration):
  fresh repositories without a manifest now REHYDRATE the live repository from
  WAL, not just a report. `_rehydrate_from_wal()` rebuilds `arena.grafts` with
  text, kind, metadata, and active state; forgotten nodes recover as retired
  (inactive). Recovered nodes are text-authoritative but not yet routable: they
  carry a dialect-width zero centroid (cosine ~0, never wins routing by
  accident), no `host_payload`, no device tensor, and `payload_pending=True`
  until re-harvested. `flush_now()` skips the missing-payload write for such
  nodes (manifest keeps them as `payload_pending`, not falsely durable), and
  `load()` preserves the pending flag through a manifest round-trip.
  Validated by `test_wal_recovers_text_metadata_without_manifest` and
  `test_wal_recovery_keeps_forgotten_nodes_inert`.
- Metadata/fact semantics:
  nodes carry durability, mutability, scope, write intent, confidence,
  active state, supersession fields, source grafts, and provenance.
- Explicit memory commands:
  `remember`, `forget`, `correct/update`, and `flush memory now` are supported.
- Review buffer:
  uncertain candidates can be recorded and later approved into memory.
- Conservative extractor candidate interface:
  `GraftRepository.apply_extraction_candidate(s)` accepts classifier-style
  candidate dictionaries, writes high-confidence non-conflicting candidates,
  sends low-confidence or inferred conflicting claims to the review buffer, and
  lets authoritative user/system assertions supersede active semantic memory
  with revision metadata. This implements the no-GPU extractor boundary; it is
  not yet an automatic model-based extractor.
- C++ host runtime scaffold:
  `cpp/` contains `DialectDescriptor`, `HostGraftStore`, `RouterIndex`,
  `DirtyQueue`, `DurabilityWriter`, swap/evict-planning `DeviceArena`, and
  CMake build.
- Native Python boundary:
  `cpp/grm_runtime_c.h` exposes a dependency-free C ABI and
  `core/grm_native.py` drives it with `ctypes`.
- Opt-in native host mirror:
  `GraftRepository(..., native_lib_path=...)` mirrors RAM payload bytes into
  `NativeGraftStore`, marks nodes durable on `flush_now()`, and rebuilds the
  native mirror on manifest reload.
- Native page-out cleanliness:
  repository-driven native device-copy eviction now uses an already-synced
  native node id when available, avoiding redundant route/metadata rewrites
  and keeping durable native host payloads clean after budgeted reload paging.
- Structured native payload tensors:
  the native C++ store now accepts named tensors with shape/dtype metadata,
  reports per-node tensor counts/bytes, and exposes tensor shape plus byte
  readback through `core/grm_native.py` so Python can reconstruct numpy arrays.
- Native host-store checkpointing:
  `HostGraftStore` can save/load a dependency-free binary checkpoint containing
  node ids, text, token counts, lifecycle state, and every structured tensor's
  name, dtype, shape, and bytes. The C ABI and Python wrapper expose
  `save_checkpoint()` and `load_checkpoint()`.
- Native semantic metadata persistence:
  native nodes now store the repository metadata JSON blob, mirror semantic
  metadata from `GraftRepository`, expose it through the C ABI/Python wrapper,
  and preserve it through native checkpoints.
- Native graph-edge metadata:
  source turns, source grafts, supersedes, and superseded-by edges are mirrored
  into structured native state through `grm_store_set_graph_edges()`, exposed
  through `NativeGraftStore.graph_edges()`, and preserved in `GRMSTORE4`
  checkpoints.
- Native revision operation:
  `grm_store_apply_revision()` / `NativeGraftStore.apply_revision()` retires
  superseded nodes, links the replacement node, and updates native route
  activity in one native operation after Python decides the correction policy.
- Native memory-command parser boundary:
  `grm_store_parse_memory_command()` parses the deterministic explicit memory
  command grammar (`remember`, `forget`, `correct/update`, review fallback,
  ignore, flush) into a JSON operation plan. `NativeGraftStore` exposes that
  parser and native-backed `GraftRepository.apply_memory_command()` consumes it
  before applying the Python memory policy.
- Native routing/indexing:
  the C ABI now exposes route-key upsert and top-k route lookup through the
  C++ `RouterIndex`; `GraftRepository.native_route()` translates native route
  results back to Python graft indices.
- Native lifecycle-aware routing:
  `HostGraftStore` and `RouterIndex` now carry active/inactive state through
  `grm_store_set_active()`. Native route lookups skip inactive entries,
  repository `forget()` and `correct_memory()` push `metadata.active` into the
  native router, and the active bit is preserved through native checkpoints.
- Native route policy filters:
  `grm_store_set_route_metadata()` mirrors kind, scope, durability, and
  mutability into the C++ route index. `grm_store_route_filtered()` and
  `NativeGraftStore.route(..., kinds=..., scopes=..., durabilities=...,
  mutabilities=...)` can now filter route candidates in native code while
  preserving the existing unfiltered ABI.
- Native route-scan arena path:
  `ArenaCache.route()` now uses the native C++ route index when the native
  mirror is enabled and active MLA candidates have native route entries. The
  C++ lexical score matches Python's fractional identifier bonus, and the
  native index stores multiple route keys per node so digest/era
  child-centroid routing uses the same max-over-keys law as Python. Python
  fallback remains for unsupported dialect score laws or incomplete native
  coverage.
- Native DeviceArena swap-plan boundary:
  `DeviceArena` now exposes the cache movement plan for replacing
  `[sink | old mounts | live tail]` with `[sink | new mounts | live tail]`,
  including live-tail carry offsets, output length, and arena overflow status.
- Native DeviceArena host tensor swap reference:
  the C++ runtime can apply that plan to contiguous host tensors, preserving
  sink rows, inserting the new mount block, and carrying the live tail. The
  Python wrapper exposes this through `NativeGraftStore.apply_swap_tensor()`
  and `apply_swap_payload()` for dialect-shaped graft payload dictionaries.
- Native DeviceArena host tensor evict reference:
  the C++ runtime can drop stale live spans while preserving `[sink | mounts]`
  and carrying the remaining live tail. The Python wrapper exposes this through
  `NativeGraftStore.apply_evict_tensor()` and `apply_evict_payload()`.
- Native arena state alignment:
  native-enabled repositories attach the C++ store to `ArenaCache`, and arena
  bootstrap/swap commits the currently mounted native node ids plus mount-token
  count so native swap plans reflect Python's current cache seating.
- TensorCUDA arena splice/evict kernels:
  Project-Tensor branch `codex/grm-arena-cache-surgery-20260627` exposes fused
  CUDA `splice_rows` and `evict_rows` ops for functional cache surgery, and
  `ArenaCache.swap()` / `evict()` use them when available with the original
  slice/cat implementation as fallback.
- TensorCUDA multi-layer arena swap/re-seat boundary:
  `tc.swap_row_pairs_with_rope()` batches raw payload splice plus positional
  payload re-RoPE and splice across the layer stack behind one C++ binding call.
  `tc.evict_row_pairs()` batches paired raw/positional evictions across the
  layer stack. `tc.arena_row_pair_transaction()` wraps those mechanics into a
  functional cache transaction that selects swap versus empty-mount eviction,
  validates arena width and layer token counts, and returns the new mount-token
  state with the new cache stack. `ArenaCache.swap()` uses this for compatible
  two-payload dialects when available, with the previous per-layer helpers
  retained as fallback.
- TensorCUDA fused RoPE re-seat/export primitive:
  `tc.rope_apply(x, cos, sin, pos0, inverse=False, pair_swap=False)` now
  supports inverse rotation and DeepSeek's pair-swapped positional layout in
  the same one-launch CUDA kernel. `ArenaCache` uses it for mount re-seat when
  available, with the old composed `F.apply_rotary` path as fallback.
- TensorCUDA fused cache-span export primitives:
  `tc.export_rows()` slices raw cache payload spans in one CUDA launch, and
  `tc.export_rope_rows()` slices positional payload spans while applying
  forward or inverse RoPE in the same CUDA launch. `tc.export_row_pair()`
  groups one raw span plus one positional span behind a single C++ binding call.
  `tc.export_row_pairs()` batches that raw+positional pair export across the
  layer stack behind one C++ binding call. `ArenaCache.deposit_from_cache()`
  uses that multi-layer boundary for two-payload dialects when available,
  including DeepSeek MLA `kpe` pair-swap handling, with the pair and lower-level
  raw/RoPE exports retained as fallbacks.
- Arena cache accounting and cleanup:
  `ArenaCache` now supports `live_turns=0`, explicit live-cache reset for
  independent probes, transient CUDA cleanup after route/swap/evict, and
  corrected greedy-decode segment accounting so deposited/evicted spans match
  the tokens actually committed to KV cache.
- DeepSeek GRM hook surface:
  `DeepSeekMLATC` now exposes `_capture`, `_capture_q`, `inject_kv`,
  `graft_seats`, and `live_shift`; `DeepSeekMLAArenaCache` declares the
  512+64 MLA payload geometry and DeepSeek-specific k_pe re-RoPE behavior.
- DeepSeek absorbed decode warmup:
  `DeepSeekV2Lite_TC.warm_absorbed_decode()` materializes the absorbed MLA
  decode projections before repository payloads attach during fresh resume,
  avoiding a lazy projection allocation spike after graft reload.

## Validated

- Non-GPU Python runtime tests:
  `tests/test_grm_runtime_lifecycle.py`
- Native C++ ABI test:
  `tests/test_grm_native_runtime.py`, including structured tensor payloads,
  checkpoint round-trip, metadata round-trip, route-key upsert, route lookup,
  fractional lexical route calibration, multi-key child-centroid route scoring,
  native active/inactive route filtering, active-state checkpoint round-trip,
  native route policy filtering by kind/scope/durability/mutability,
  structured source/supersession graph-edge readback and checkpoint round-trip,
  native `apply_revision()` route-retirement and graph-link behavior,
  `GraftRepository` native route sync, `ArenaCache.route()` native backend use
  and unsupported-store fallback,
  DeviceArena swap/evict plan coverage, and host tensor swap/evict byte
  equivalence against NumPy, including DeepSeek-style `c` plus `kpe` payload
  dictionaries.
- DeepSeek non-GPU hook contract:
  `tests/test_deepseek_grm_hooks_static.py`
- DeepSeek live CUDA smoke gate:
  `tests/deepseek_grm_smoke_gate.py` loads DeepSeek-V2-Lite INT4, checks
  last-token graft-vs-in-context logit parity, runs a greedy graft recall,
  deposits through `DeepSeekMLAArenaCache`, flushes RAM payloads, and reloads
  the repository. It accepts `--native-lib` to include the C++ host mirror and
  assert native payload, durability, and route-index stats.
- DeepSeek routed arena CUDA gate:
  `tests/deepseek_grm_arena_gate.py` loads DeepSeek-V2-Lite INT4, deposits
  three document grafts, feeds three live turns, forces live eviction, answers
  two identifier-keyed greedy probes through routed arena swaps, flushes RAM
  payloads, reloads, and checks native mirror consistency. It supports
  `--mode build` and `--mode resume` so fresh-process resume can be tested
  separately. By default read-only probes use independent live caches and reset
  after each probe; `--keep-probe-cache` keeps the older retained-cache stress
  behavior.
- DeepSeek full paging/open-ended recall CUDA gate:
  `tests/deepseek_grm_full_gate.py` loads DeepSeek-V2-Lite INT4, deposits five
  document grafts and eight live turns, forces a 2 MB graft-device budget,
  flushes/reloads the RAM/native repository, and in resume mode answers four
  open-ended greedy exact-fact probes through the native route backend while
  exercising RAM page-ins. The gate asserts durable clean state, native mirror
  consistency, and budgeted reload behavior (`active_device < nodes`). It now
  passes both the independent-probe default and retained-probe-cache stress
  mode on the 12GB card.
- DeepSeek turn-50 ephemeral-boat CUDA gate:
  `tests/deepseek_grm_turn50_gate.py` uses the original GRM ephemeral boat
  (`ephemeral=True`, recency as mounts, live context cleared between turns),
  stores 50 raw turn grafts with an early needle, flushes/reloads, and recalls
  the turn-1 code from a fresh context through native routing and RAM page-in.
  Folding is disabled by default in this gate to isolate raw off-context turn
  recall. Enabling DeepSeek folding/consolidation hit an OOM around turn 10 and
  remains separate work.
- C++ build:
  `cmake -S cpp -B /tmp/grm_runtime_build && cmake --build /tmp/grm_runtime_build`
- TensorCUDA cache-surgery gates:
  `tensor_cuda/tests/test_splice_rows.py` validates fused CUDA splice/evict
  against NumPy for float and uint8 cache rows, plus multi-layer
  `swap_row_pairs_with_rope()` re-seat/splice and `evict_row_pairs()` paired
  evictions. It also validates `arena_row_pair_transaction()` swap selection,
  empty-mount eviction, state return, arena-width rejection, and inference-only
  grad guards.
  `tests/grm_arena_splice_gate.py` validates the GRM `ArenaCache` helper path,
  including raw+positional pair export through `ArenaCache._export_cache_payload()`
  multi-layer pair export through `ArenaCache._export_cache_payloads()`,
  multi-layer swap/re-seat through `ArenaCache._swap_cache_payloads()`, and
  paired eviction through `ArenaCache._evict_cache_payloads()`, including the
  transaction width guard.
- TensorCUDA fused RoPE gate:
  `tensor_cuda/tests/test_rope_apply.py` validates forward RoPE parity,
  inverse round-trip, and p-RoPE identity dimensions on GPU.
- TensorCUDA fused export gate:
  `tensor_cuda/tests/test_export_rows.py` validates raw float and uint8 cache
  export, fused RoPE forward/export parity, inverse cache-span un-RoPE, the
  DeepSeek pair-swapped positional layout, raw+positional `export_row_pair()`,
  multi-layer `export_row_pairs()`, and inference-only grad guards.

Latest checked result:

```text
PYTHONPATH=/mnt/ForgeRealm/GraftRepository \
pytest -p no:cacheprovider -q \
  tests/test_grm_runtime_lifecycle.py \
  tests/test_grm_native_runtime.py \
  tests/test_deepseek_grm_hooks_static.py
43 passed, 2 warnings

cmake --build /tmp/grm_runtime_build
Built target grm_runtime
Built target grm_runtime_shared

./build.sh 89
Built target _tensor_cuda

PYTHONPATH=/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
python3 tensor_cuda/tests/test_export_rows.py
EXPORT_ROWS GATE: PASS

PYTHONPATH=/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
python3 tensor_cuda/tests/test_rope_apply.py
ROPE_APPLY GATE: PASS

PYTHONPATH=/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
python3 tensor_cuda/tests/test_splice_rows.py
SPLICE_ROWS GATE: PASS

PYTHONPATH=/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
python3 tensor_cuda/tests/test_write_rows.py
WRITE_ROWS GATE: PASS

PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
python3 tests/grm_arena_splice_gate.py
GRM ARENA SPLICE GATE: PASS

PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
python3 tests/deepseek_grm_smoke_gate.py \
  --repo /tmp/deepseek_grm_smoke_native \
  --native-lib /tmp/grm_runtime_build/libgrm_runtime.so
loaded: {'loaded': 5291, 'framework': 'tensor_cuda DeepSeek-V2-Lite INT4', ...}
G1 graft-vs-incontext max|logit diff|=0.1953 top1_flips=0/1
G1b greedy recall: HIT | '73-4412.\nUser: Repeat the exact access code from the briefing.\nAssistant: The access code is 73-'
G2 repository stats: nodes=1 durable_nodes=1 dirty_nodes=0 native.nodes=1 native.durable_nodes=1 native.host_payload_tensors=2 native.route_entries=1
G3 reload stats: nodes=1 durable_nodes=1 dirty_nodes=0 native.nodes=1 native.durable_nodes=1 native.host_payload_tensors=2 native.route_entries=1
DONE

PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
python3 tests/deepseek_grm_arena_gate.py \
  --mode build \
  --repo /tmp/deepseek_grm_arena_fixed \
  --native-lib /tmp/grm_runtime_build/libgrm_runtime.so
doc 0: ntok=36
doc 1: ntok=40
doc 2: ntok=40
fed turn: resident=4 live=0
fed turn: resident=4 live=0
fed turn: resident=4 live=0
stats: nodes=6 durable_nodes=6 dirty_nodes=0 native.nodes=6 native.durable_nodes=6 native.host_payload_tensors=12 native.route_entries=6
DEEPSEEK ARENA BUILD: PASS

PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
python3 tests/deepseek_grm_arena_gate.py \
  --mode resume \
  --repo /tmp/deepseek_grm_arena_fixed \
  --native-lib /tmp/grm_runtime_build/libgrm_runtime.so
absorbed decode weights: warm
probe HIT mounts=[1] backend=native resident=40 evicted=40 | 'The exact access code for ROUTE-ALPHA is A17-9402.'
probe HIT mounts=[2] backend=native resident=44 evicted=42 | 'The exact access code for ROUTE-BRAVO is B88-1120.'
stats: nodes=6 durable_nodes=6 dirty_nodes=0 native.nodes=6 native.durable_nodes=6 native.host_payload_tensors=12 native.route_entries=6
reload stats: nodes=6 durable_nodes=6 native.nodes=6 native.durable_nodes=6 native.host_payload_tensors=12 native.route_entries=6
DEEPSEEK ARENA RESUME GATE: PASS

PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
python3 tests/deepseek_grm_full_gate.py \
  --mode build \
  --repo /tmp/deepseek_grm_full_final \
  --native-lib /tmp/grm_runtime_build/libgrm_runtime.so
loaded: {'loaded': 5291, 'framework': 'tensor_cuda DeepSeek-V2-Lite INT4', ...}
absorbed decode weights: warm
stats: nodes=13 active_device=2 device_mb=2 ram_payload_mb=14 dirty_nodes=0 durable_nodes=13 page_ins=4 native.nodes=13 native.dirty_nodes=0 native.durable_nodes=13 native.host_payload_tensors=26 native.route_entries=13
reload stats: nodes=13 active_device=2 device_mb=2 ram_payload_mb=14 dirty_nodes=0 durable_nodes=13 native.nodes=13 native.dirty_nodes=0 native.durable_nodes=13 native.host_payload_tensors=26 native.route_entries=13
DEEPSEEK FULL BUILD: PASS

PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
python3 tests/deepseek_grm_full_gate.py \
  --mode resume \
  --repo /tmp/deepseek_grm_full_final \
  --native-lib /tmp/grm_runtime_build/libgrm_runtime.so
absorbed decode weights: warm
probe HIT backend=native mounts=[1] | 'The clearance code for ORION is O17-4821.'
probe HIT backend=native mounts=[2] | 'The VEGA clearance code is V22-9140.'
probe HIT backend=native mounts=[3] | 'Bay-05.'
probe HIT backend=native mounts=[5] | 'H44-2088.'
stats: nodes=17 active_device=1 device_mb=1 ram_payload_mb=17 dirty_nodes=0 durable_nodes=17 page_ins=4 native.nodes=17 native.dirty_nodes=0 native.durable_nodes=17 native.host_payload_tensors=34 native.route_entries=17
reload stats: nodes=17 active_device=2 device_mb=1 ram_payload_mb=17 dirty_nodes=0 durable_nodes=17 native.nodes=17 native.dirty_nodes=0 native.durable_nodes=17 native.host_payload_tensors=34 native.route_entries=17
DEEPSEEK FULL RESUME GATE: PASS

PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
python3 tests/deepseek_grm_full_gate.py \
  --mode resume \
  --repo /tmp/deepseek_grm_full_retained \
  --native-lib /tmp/grm_runtime_build/libgrm_runtime.so \
  --keep-probe-cache
probe HIT backend=native mounts=[1] | 'The clearance code for ORION is O17-4821.'
probe HIT backend=native mounts=[2] | 'The VEGA clearance code is V22-9140.'
probe HIT backend=native mounts=[3] | 'The launch bay for LYRA is Bay-05.'
probe HIT backend=native mounts=[5] | 'The exact clearance code for HELIX is H44-2088.'
stats: nodes=17 active_device=1 device_mb=1 ram_payload_mb=18 dirty_nodes=0 durable_nodes=17 page_ins=4 native.nodes=17 native.dirty_nodes=0 native.durable_nodes=17 native.host_payload_tensors=34 native.route_entries=17
reload stats: nodes=17 active_device=2 device_mb=2 ram_payload_mb=18 dirty_nodes=0 durable_nodes=17 native.nodes=17 native.dirty_nodes=0 native.durable_nodes=17 native.host_payload_tensors=34 native.route_entries=17
DEEPSEEK FULL RESUME GATE: PASS

PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
python3 tests/deepseek_grm_turn50_gate.py \
  --mode build \
  --repo /tmp/deepseek_grm_turn50_ephemeral_final \
  --native-lib /tmp/grm_runtime_build/libgrm_runtime.so
turn 1: nodes=1 active_device=0 live_tokens=0
turn 10: nodes=10 active_device=1 live_tokens=0
turn 20: nodes=20 active_device=1 live_tokens=0
turn 30: nodes=30 active_device=1 live_tokens=0
turn 40: nodes=40 active_device=1 live_tokens=0
turn 50: nodes=50 active_device=1 live_tokens=0
stats: nodes=50 active_device=1 ram_payload_mb=103 dirty_nodes=0 durable_nodes=50 native.nodes=50 native.dirty_nodes=0 native.durable_nodes=50 native.host_payload_tensors=100 native.route_entries=50
DEEPSEEK TURN50 BUILD: PASS

PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
python3 tests/deepseek_grm_turn50_gate.py \
  --mode resume \
  --repo /tmp/deepseek_grm_turn50_ephemeral_final \
  --native-lib /tmp/grm_runtime_build/libgrm_runtime.so
turn50 probe HIT backend=native mounts=[1] resident=79 evicted=39 | 'ASTRA-NODE clearance code is T50-7391.'
stats: nodes=51 active_device=0 ram_payload_mb=105 dirty_nodes=0 durable_nodes=51 page_ins=1 native.nodes=51 native.dirty_nodes=0 native.durable_nodes=51 native.host_payload_tensors=102 native.route_entries=51
reload stats: nodes=51 active_device=1 ram_payload_mb=105 dirty_nodes=0 durable_nodes=51 native.nodes=51 native.dirty_nodes=0 native.durable_nodes=51 native.host_payload_tensors=102 native.route_entries=51
DEEPSEEK TURN50 RESUME GATE: PASS
```

## Not Complete

- C++ `DeviceArena` owns the host-reference byte movement for contiguous tensor
  swap and eviction, and TensorCUDA now owns fused CUDA splice/evict movement,
  fused RoPE re-seat movement, fused cache-sliced raw/RoPE export movement, and
  the functional multi-layer raw+positional arena cache transaction on the
  Project-Tensor branch. GRM still keeps the broader routing policy,
  memory-command policy, revision policy, and runtime orchestration in Python
  rather than in one cohesive C++/CUDA runtime object.
- C++ host route-scan acceleration exists for native-backed MLA arena routes,
  including child-centroid digest/era keys. CUDA/GPU route-scan acceleration is
  not implemented.
- Python `GraftRepository` mirrors into `NativeGraftStore` only by opt-in;
  the C++ store owns mirrored payload lifecycle, tensor boundaries, tensor
  shapes/dtypes, payload byte reconstruction, host payload checkpointing, and
  lifecycle-aware route indexing with native kind/scope/durability/mutability
  filters. Semantic metadata is persisted natively as JSON, and source/
  supersession graph edges are now structured native state. Native can apply
  the final revision state once Python decides the correction. The explicit
  memory-command grammar now has a native parse-plan boundary, while conflict
  policy, review approval, conservative extractor-candidate application, and
  final command execution still live in Python.
- DeepSeek-specific GRM attention hooks have passed live CUDA parity, greedy
  recall, repository lifecycle smoke, routed build/resume, and full
  paging/open-ended greedy recall build/resume gates. The longer high-context
  needle suite and broader model-specific graft equivalence matrix still need
  separate scheduled runs.
- Retaining the previous read-only probe cache between independent DeepSeek
  probes (`--keep-probe-cache`) now passes for the full DeepSeek-V2-Lite INT4
  gate on the 12GB card. Longer high-context runs may still hit separate memory
  limits.
- DeepSeek raw turn-50 ephemeral-boat recall now passes with folding disabled.
  DeepSeek librarian folding/consolidation is not yet validated; the first
  folded build attempt OOMed inside consolidation around turn 10.

## Current State

The Python RAM-first runtime, opt-in C++ host-store mirror, native host payload,
metadata, graph-edge checkpointing, native revision application, native route
index with active-state and kind/scope/durability/mutability filters plus multi-key arena-route
acceleration, native explicit memory-command parser, native swap-plan boundary,
native host tensor swap/evict references, TensorCUDA fused splice/evict cache
movement, TensorCUDA fused RoPE re-seat movement, TensorCUDA fused
cache-sliced raw/RoPE export primitives with paired and multi-layer paired
export boundaries, TensorCUDA functional multi-layer arena cache transaction,
and DeepSeek GRM clean build plus fresh-process resume paths are real and
tested, including a full paging/open-ended recall gate on DeepSeek-V2-Lite
INT4, including retained-probe-cache stress and raw turn-50 ephemeral-boat
recall. The full production C++/CUDA runtime is not complete until routing
policy boundaries, metadata/revision policy ownership, model-based extraction
policy hardening, DeepSeek-safe librarian consolidation, CUDA route scanning if
needed, longer needle/high-context runs, and the broader model-specific graft
equivalence matrix pass.
