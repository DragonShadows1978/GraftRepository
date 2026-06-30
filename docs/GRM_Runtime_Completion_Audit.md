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
  `load()` preserves the pending flag through a manifest round-trip. WAL
  recovery also replays semantic revision records (`MEMORY_CORRECT`,
  `MEMORY_EXTRACT_SUPERSEDE`) so superseded facts do not resurrect after a
  crash-before-manifest, and preserves fold-abort `no_fold` exemptions from
  `NODE_META` state so a rejected librarian window does not loop after
  recovery. Validated by `test_wal_recovers_text_metadata_without_manifest`,
  `test_wal_recovery_keeps_forgotten_nodes_inert`,
  `test_wal_recovery_replays_extraction_supersession`, and
  `test_wal_recovery_preserves_fold_abort_exemption`.
- Metadata/fact semantics:
  nodes carry durability, mutability, scope, write intent, confidence,
  active state, supersession fields, source grafts, and provenance.
- Explicit memory commands:
  `remember`, `forget`, `correct/update`, review fallback, ignore, and
  `flush memory now` are supported. `GRMRuntime.apply_memory_command()` now
  runs all explicit commands through the runtime finish path, so autosave
  publishes command mutations durably while `flush_immediately` and explicit
  flush commands force durability even when autosave is disabled.
- Review buffer:
  uncertain candidates can be recorded and later approved into memory.
- Conservative extractor candidate interface and runtime hook:
  `GraftRepository.apply_extraction_candidate(s)` accepts classifier-style
  candidate dictionaries, writes high-confidence non-conflicting candidates,
  sends low-confidence or inferred conflicting claims to the review buffer, and
  lets authoritative user/system assertions supersede active semantic memory
  with revision metadata. Public candidate application now finishes through
  `GRMRuntime`, reports an `extraction` runtime event, and honors autosave
  durability for direct `apply_extraction_candidate(s)` callers. Turn-triggered
  extraction remains inside the enclosing chat/add-turn runtime event to avoid
  double flushing. `GraftRepository(..., extractor=...)` now runs an optional
  extractor on newly completed chat/scripted turns, passes source turn graft ids
  into that same policy path, and records extractor errors as non-blocking WAL
  events unless configured to raise.
- Python runtime coordinator boundary:
  `core.grm_runtime.GRMRuntime` now owns the operation sequencing for
  `chat()`, `add_turn()`, `idle()`, review execution, and explicit
  memory-command execution, plus public extractor-candidate execution:
  snapshot, model/arena action, extraction/review policy, librarian folding,
  mutation marking, flush, and paging.
  `GraftRepository` remains the public API and persistence owner, but the
  hot-path orchestration is no longer spread across public methods.
- Graftability/remountability dialect metadata:
  `DialectDescriptor` now persists the model's positional cache law and graft
  semantics: `position_law`, `state_kind`, `graftability`, `remountable`, and
  `composition`. MLA records `rope_partial_mla` with seat-remountable
  latent+RoPE payloads; plain GQA records `rope_full_kv`; learned absolute
  position records `same_position_restore`; Qwen3.5-style recurrent hybrids
  record `prefix_restore_only`; and Gemma-style sliding/global KV records
  window-limited remountability. The remount boundary is RoPE: pre-RoPE
  captures can be re-RoPE'd at injection seats, while fixed absolute-position
  captures require same-position restore or a model-specific re-encoding path.
  Native `GRMSTORE7` checkpoints now persist and enforce the same profile, so
  same-shape stores with different positional/remount laws cannot load each
  other's graft payloads.
- C++ host runtime scaffold:
  `cpp/` contains `DialectDescriptor`, `HostGraftStore`, `RouterIndex`,
  `DirtyQueue`, `DurabilityWriter`, swap/evict-planning `DeviceArena`, and
  CMake build.
- Native Python boundary:
  `cpp/grm_runtime_c.h` exposes a dependency-free C ABI and
  `core/grm_native.py` drives it with `ctypes`. The host-store creation ABI now
  supports both MLA and GQA dialect descriptors.
- Opt-in native host mirror:
  `GraftRepository(..., native_lib_path=...)` mirrors RAM payload bytes into
  `NativeGraftStore`, marks nodes durable on `flush_now()`, and rebuilds the
  native mirror on manifest reload for both MLA and GQA payload families.
- Native metadata-only cold-node mirror:
  manifest reload now mirrors retired/cold durable nodes into the native store
  without reloading their payload tensors into RAM. This keeps native node ids,
  route metadata, active/inactive state, and graph references aligned with the
  Python manifest while preserving the RAM-first cold-storage boundary.
- Native page-out cleanliness:
  repository-driven native device-copy eviction now refreshes the native host
  payload before asking C++ to drop the device copy, which fixes the
  metadata-only cold-node page-in edge while keeping durable native host
  payloads clean after budgeted reload paging.
- Structured native payload tensors:
  the native C++ store now accepts named tensors with shape/dtype metadata,
  reports per-node tensor counts/bytes, and exposes tensor shape plus byte
  readback through `core/grm_native.py` so Python can reconstruct numpy arrays.
- Native host-store checkpointing:
  `HostGraftStore` can save/load a dependency-free binary checkpoint containing
  the native dialect id, node ids, text, token counts, lifecycle state, route
  vectors, lexical route keys, and every structured tensor's name, dtype, shape,
  and bytes. The C ABI and Python wrapper expose `save_checkpoint()` and
  `load_checkpoint()`, and `GRMSTORE6`
  rejects mismatched native dialect loads while keeping older checkpoints
  readable. Loading a checkpoint rebuilds the native `RouterIndex` directly
  from stored node route state.
- Repository-level native checkpoint integration:
  `GraftRepository.flush_now()` now checkpoints the native host store to
  `native/grm_store.bin` before publishing `manifest.json`, stores per-node
  `native_node_id` values in the manifest, and reloads the native checkpoint
  on repository resume when present. Metadata-only native mutations are covered
  by the same flush boundary, so Python durable state no longer leaves native
  dirty state behind. Retired durable nodes clear native host payload tensors
  before checkpointing while preserving metadata, graph state, and route keys,
  so cold payload bytes stay in NVMe node files rather than native RAM.
- Native semantic metadata persistence:
  native nodes now store the repository metadata JSON blob, mirror semantic
  metadata from `GraftRepository`, expose it through the C ABI/Python wrapper,
  and preserve it through native checkpoints.
- Native graph-edge metadata:
  source turns, source grafts, supersedes, and superseded-by edges are mirrored
  into structured native state through `grm_store_set_graph_edges()`, exposed
  through `NativeGraftStore.graph_edges()`, and preserved in `GRMSTORE4+`
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
  before applying the Python memory policy. Command execution then completes
  through `GRMRuntime`, including autosave/forced-flush durability semantics.
- Review-buffer execution:
  uncertain extraction or malformed correction candidates can now be approved,
  rejected, edited, or scope-changed through repository APIs backed by the
  `GRMRuntime` review boundary. Review items carry explicit `pending` /
  `approved` / `rejected` status, autosave-enabled review mutations publish via
  `flush_now()`, and WAL replay applies review edits and decisions over both
  WAL-only recovery and manifest-plus-WAL reload.
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
- Native MLA route-scan arena path:
  `ArenaCache.route()` now uses the native C++ route index when the native
  mirror is enabled and active MLA candidates have native route entries. The
  C++ lexical score matches Python's fractional identifier bonus, and the
  native index stores multiple route keys per node so digest/era
  child-centroid routing uses the same max-over-keys law as Python. Python
  fallback remains for GQA/raw-key scoring laws or incomplete native coverage.
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
  recall. With `--enable-folding`, DeepSeek now performs first-gen
  turn-to-digest compression and extractive digest-to-era compression on the
  same 50-turn workload: 40 raw turns retire under 10 digest nodes, six digest
  nodes retire under two era nodes, only the first 4-turn window remains
  `no_fold`, and fresh-process recall passes both for the original turn-1
  ASTRA code and for a retired turn-5 source read through folded memory.
  Identifier-bearing probes suppress recency mounts so previous recall turns
  cannot pollute point reads; no-identifier anaphora still uses recency.
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
52 passed, 2 warnings

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

PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
python3 tests/deepseek_grm_turn50_gate.py \
  --mode build \
  --repo /tmp/deepseek_grm_era_full_v1 \
  --native-lib /tmp/grm_runtime_build/libgrm_runtime.so \
  --enable-folding \
  --print-fold-diagnostics
turn 50: nodes=62 active_device=1 live_tokens=0
fold 8: kind=era sources=[12, 17, 22] accepted=True digest_idx=38 best_cov=1.0 hits=55/55
fold 12: kind=era sources=[27, 32, 37] accepted=True digest_idx=54 best_cov=1.0 hits=54/54
stats: nodes=62 kinds={'turn': 10, 'retired turn': 40, 'retired digest': 6, 'era': 2, 'digest': 4} active_device=1 ram_payload_mb=217 dirty_nodes=0 durable_nodes=62 folds_aborted=1 no_fold=4 native.nodes=62 native.dirty_nodes=0 native.durable_nodes=62 native.host_payload_tensors=124 native.route_entries=62
reload stats: nodes=62 kinds={'turn': 10, 'retired turn': 40, 'retired digest': 6, 'era': 2, 'digest': 4} active_device=1 ram_payload_mb=91 dirty_nodes=0 durable_nodes=62 cold_nodes=46 native.nodes=62 native.dirty_nodes=0 native.durable_nodes=62 native.host_payload_tensors=32 native.route_entries=62
DEEPSEEK TURN50 BUILD: PASS

PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
python3 tests/deepseek_grm_turn50_gate.py \
  --mode resume \
  --repo /tmp/deepseek_grm_era_full_v1 \
  --native-lib /tmp/grm_runtime_build/libgrm_runtime.so
turn50 probe HIT backend=native mounts=[1] resident=79 evicted=39 active_device=0 0.74s | 'ASTRA-NODE clearance code is T50-7391.'
stats: nodes=65 kinds={'turn': 10, 'retired turn': 40, 'retired digest': 6, 'era': 2, 'digest': 4, 'recall': 3} active_device=0 ram_payload_mb=95 dirty_nodes=0 durable_nodes=65 page_ins=1 no_fold=4 native.nodes=65 native.dirty_nodes=0 native.durable_nodes=65 native.host_payload_tensors=38 native.route_entries=65
DEEPSEEK TURN50 RESUME GATE: PASS

PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
python3 tests/deepseek_grm_turn50_gate.py \
  --mode resume \
  --repo /tmp/deepseek_grm_era_full_v1 \
  --native-lib /tmp/grm_runtime_build/libgrm_runtime.so \
  --enable-folding \
  --probe-turn 5
turn5 probe HIT backend=native mounts=[13] resident=229 evicted=36 active_device=0 0.75s | 'The marker is M05-0685.'
stats: nodes=64 kinds={'turn': 10, 'retired turn': 40, 'retired digest': 6, 'era': 2, 'digest': 4, 'recall': 2} active_device=0 ram_payload_mb=101 dirty_nodes=0 durable_nodes=64 page_ins=1 no_fold=4 native.nodes=64 native.dirty_nodes=0 native.durable_nodes=64 native.host_payload_tensors=38 native.route_entries=64
DEEPSEEK TURN50 RESUME GATE: PASS
```

Latest cross-architecture GRM gates on current `grm-ram-tiered-runtime` HEAD:

```text
MiniCPM3 MLA infinite-context gate
loaded: {'loaded': 'INT4 MLA', 'framework': 'tensor_cuda MiniCPM3-4B'}
42 turns deposited and consolidated in 52s
probes: 8/8 HIT
final: INFINITE: 8/8 | max resident 341 seats | repo 70.3MB |
  nodes=58 active_device=26 device_mb=40 ram_payload_mb=91
  folds_aborted=2 no_fold=7

Qwen3-4B GQA descent gate
loaded: {'loaded': 'bf16', 'framework': 'tensor_cuda bf16 (unquantized, Qwen3-4B)'}
42 turns deposited and consolidated in 513s
probes: 8/8 HIT
final: GQA-DESCENT: 8/8 | max resident 429 |
  nodes=61 active_device=32 device_mb=266 ram_payload_mb=435 page_ins=13
  folds_aborted=0 no_fold=0
```

## Not Complete

- C++ `DeviceArena` owns the host-reference byte movement for contiguous tensor
  swap and eviction, and TensorCUDA now owns fused CUDA splice/evict movement,
  fused RoPE re-seat movement, fused cache-sliced raw/RoPE export movement, and
  the functional multi-layer raw+positional arena cache transaction on the
  Project-Tensor branch. GRM now has a Python `GRMRuntime` coordinator for
  hot-path orchestration, but broader routing policy, memory-command policy,
  revision policy, and runtime ownership are still Python rather than one
  cohesive C++/CUDA runtime object.
- C++ host route-scan acceleration exists for native-backed MLA arena routes,
  including child-centroid digest/era keys. CUDA/GPU route-scan acceleration is
  not implemented, and Qwen3/GQA's raw `|q.k|` scoring law remains a Python
  route path.
- Python `GraftRepository` mirrors into `NativeGraftStore` only by opt-in;
  the C++ store owns mirrored payload lifecycle, tensor boundaries, tensor
  shapes/dtypes, payload byte reconstruction, host payload checkpointing, and
  lifecycle-aware route indexing with native kind/scope/durability/mutability
  filters for MLA and GQA dialect ids. Semantic metadata is persisted natively
  as JSON, and source/supersession graph edges are now structured native state.
  Native can apply the final revision state once Python decides the correction.
  The explicit memory-command grammar now has a native parse-plan boundary, and
  command execution completes behind `GRMRuntime`; public extractor-candidate
  execution is also runtime-coordinated. Conflict policy and extractor quality
  still live in the Python repository policy layer.
- DeepSeek-specific GRM attention hooks have passed live CUDA parity, greedy
  recall, repository lifecycle smoke, routed build/resume, and full
  paging/open-ended greedy recall build/resume gates. Current-head MiniCPM3 MLA
  and Qwen3 GQA 42-turn gates also pass, narrowing the matrix gap to longer
  high-context needles and additional model families.
- Retaining the previous read-only probe cache between independent DeepSeek
  probes (`--keep-probe-cache`) now passes for the full DeepSeek-V2-Lite INT4
  gate on the 12GB card. Longer high-context runs may still hit separate memory
  limits.
- DeepSeek turn-to-digest and extractive digest-to-era folding now pass the
  50-turn ephemeral-boat build/resume gate and a folded-source turn-5 recall
  probe. The extractive era path is deliberately an index node, not a generated
  reader digest; broader high-context era stress is still a separate scheduled
  run.

## Current State

The Python RAM-first runtime, opt-in C++ host-store mirror, native host payload,
metadata, graph-edge checkpointing, native revision application, native route
index with active-state and kind/scope/durability/mutability filters for MLA
and GQA dialect ids plus multi-key MLA arena-route acceleration, native explicit
memory-command parser, native swap-plan boundary, native host tensor swap/evict
references, TensorCUDA fused splice/evict cache movement, TensorCUDA fused RoPE
re-seat movement, TensorCUDA fused
cache-sliced raw/RoPE export primitives with paired and multi-layer paired
export boundaries, TensorCUDA functional multi-layer arena cache transaction,
and DeepSeek GRM clean build plus fresh-process resume paths are real and
tested, including a full paging/open-ended recall gate on DeepSeek-V2-Lite
INT4, including retained-probe-cache stress and raw turn-50 ephemeral-boat
recall. Folding-enabled DeepSeek turn-50 now validates two-level compression:
40 turns retire under 10 digest nodes, six digests retire under two extractive
era nodes, fresh-process ASTRA recall still passes, and a retired turn-5 fact
is recalled through folded memory. Fresh current-head cross-architecture gates
also pass on MiniCPM3 MLA and Qwen3 GQA. `GRMRuntime` now packages the Python
hot-path orchestration boundary and is covered by lifecycle tests plus a live
DeepSeek smoke gate. Dialect manifests now distinguish RoPE-remountable caches,
absolute-position prefix restore, recurrent hybrid prefix state, and
window-limited KV. The full production C++/CUDA runtime is not complete until
remaining routing/revision policy ownership moves as needed, model-specific
extraction policy hardens, longer needle/high-context runs finish, CUDA route
scanning is implemented if needed, and the broader model-specific graft
equivalence matrix passes.
