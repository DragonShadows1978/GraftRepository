# GRM RAM-Tiered Runtime Build Plan

**Status:** Python RAM-first runtime implemented through async durability,
WAL, explicit memory commands, review buffer, provenance metadata, budgeted
manifest reload, WAL replay for semantic supersession/fold-abort state, and a
compilable C++ host runtime with an opt-in Python mirror plus native explicit
memory-command parsing. TensorCUDA now owns the focused CUDA cache-surgery,
RoPE, cache-span export, and paired raw+positional export primitives, including
a multi-layer paired export boundary for compatible dialects. It also owns a
multi-layer raw+positional swap/re-seat/evict boundary and a functional cache
transaction for compatible dialects. Explicit memory-command execution,
semantic revision application, and review-buffer execution now exist in the
Python policy layer with native-backed parsing/routing/state mirrors where
available. Completed turn extraction is now an explicit runtime hook whose
candidates flow through the same conservative write/review/supersession policy;
the remaining CUDA/runtime work is packaging model-specific extraction policy,
optional CUDA route scanning, and broader orchestration into one cohesive GRM
runtime plus high-context/model-matrix GPU regression coverage.

**Context:** the GPU may be occupied by GRAPA training, so this document scopes
the runtime architecture, memory policy, and control surface before code moves.

**Implementation note, 2026-06-28:** `core/graft_repository.py` now has a
dimension-driven `DialectDescriptor`, explicit lifecycle fields
(`host_present`, `device_present`, `dirty`, `durable`, `cold_only`), an
authoritative in-RAM `host_payload` snapshot for new grafts, a dirty queue,
`flush_now()` that writes from RAM payloads instead of CUDA tensors,
thread-backed `flush_async()`/`flush_wait()`, lightweight WAL records, explicit
memory commands (`remember`, `forget`, `correct/update`, cull/split graft,
pin/unpin, mutability marking, memory introspection, `flush memory now`), review
buffer APIs, and provenance persistence. `cpp/` now contains a
compilable host-runtime scaffold with `HostGraftStore`, `RouterIndex`,
`DirtyQueue`, `DurabilityWriter`, a swap/evict-planning `DeviceArena`, and a
dependency-free C ABI exposed to Python through `core/grm_native.py`.
The descriptor also persists graftability metadata (`position_law`,
`state_kind`, `graftability`, `remountable`, `composition`) so RoPE
seat-remountable MLA/GQA caches, learned-absolute same-position restores,
recurrent hybrid prefix states, and sliding/global window-limited KV are not
collapsed into one generic cache family. The practical split is positional:
pre-RoPE extractions can be re-RoPE'd into a new seat, while fixed
learned-absolute extractions are same-position restores unless a
model-specific bridge re-encodes the graft.
`GraftRepository(..., native_lib_path=...)`, `native_enabled=True`, or an
explicit `GRM_RUNTIME_LIB` environment setting now mirrors payload lifecycle
into that native host store as reconstructable named tensors with shape/dtype
metadata, checkpoints those native host payloads to NVMe through a binary C++
store format with a persisted native dialect/profile id (`GRMSTORE7`) and
provenance JSON in `GRMSTORE9`, persists semantic metadata JSON through the
same native checkpoint path, and supports native host-store creation for both
MLA and GQA dialect descriptors.
`native_auto=False` keeps a repository Python-only even when `GRM_RUNTIME_LIB`
is set. It mirrors native route keys into the C++ `RouterIndex` and preserves
route vectors plus lexical keys through the native checkpoint path;
native route entries carry active/inactive state through
`grm_store_set_active()`, skip forgotten or superseded nodes during route
lookup, preserve active state through native checkpoints, and carry
kind/scope/durability/mutability fields for native route filtering through
`grm_store_route_filtered()`. Source turns, source grafts, supersedes, and
superseded-by edges are also mirrored into structured native state through
`grm_store_set_graph_edges()` and preserved in native checkpoints. Repository
provenance is mirrored as native JSON through
`grm_store_set_provenance_json()`, read back through
`grm_store_provenance_json()`, and consumed by `why_remember()` when the native
mirror is complete. Python metadata retains Python graft ids, while the
structured native graph edge plane is written with mapped native node ids so
native traversal remains correct even when those id spaces diverge. Folded
source descent can query recursive
source-graft closure through `grm_store_source_closure()`, and `ArenaCache`
uses that native closure for digest/era mount expansion when all returned
native ids map back to local grafts, falling back to Python `sources` otherwise.
Repository `flush_now()` now writes `native/grm_store.bin` before the Python
manifest, records the native checkpoint path plus per-node native ids in
`manifest.json`, and reloads that native checkpoint on resume when present.
Retired durable nodes clear native host payloads before checkpointing, leaving
metadata and route keys in RAM/native state while cold payload bytes live in
the repository node files.
`grm_store_apply_revision()` applies the final correction state in native code
after Python policy decides which nodes are superseded: old nodes become
inactive, replacement nodes record `supersedes`, and native route activity is
updated with the revision. `grm_store_apply_expire()` similarly applies the
final active-state transition for authoritative expire decisions after Python
selects targets and records expiry reason/timestamp metadata.
`grm_store_parse_memory_command()` now parses the deterministic explicit memory
command grammar into a JSON operation plan, and native-backed
`GraftRepository.apply_memory_command()` consumes that plan before applying
Python policy. The native grammar now includes review-buffer execution commands
(`approve review`, `reject review`, `edit review`, and review
scope/durability/mutability changes), moving the explicit review control surface
through the same command-plan boundary while leaving conflict/review/extraction
decisions in Python until those policies harden.
`grm_store_plan_review_transition()` now owns deterministic review lifecycle
validation for edit/change-scope/reject/approve transitions, including approved
and rejected terminal-state guards plus idempotent repeated approval. Python
still applies the review mutation and WAL record.
`grm_store_active_text_matches()` now performs active-only, case-insensitive text
target discovery in the C++ host store for memory commands such as pin, unpin,
ignore, show, forget, and correct. Python still applies the final mutation, WAL, and
durability semantics, with the old arena scan retained when the native mirror is
unavailable or incomplete.
`grm_store_plan_cull_spans()` now owns deterministic cull span generation and
retire-parent full-coverage validation for token-capped and tokenizer-derived
section culls; Python still discovers semantic/text boundaries, but native code
chooses the final valid token spans before payload slicing.
`GRMRuntime.apply_memory_command()` now finishes every explicit command through
the runtime durability path: autosave-enabled repositories publish remember,
forget, correct, review-fallback, ignore, and cull/split-graft decisions through
`flush_now()`. Pin/unpin and mutability-marking commands also update active-node
metadata through the same dirty/WAL/native-metadata path; read-only show/why
commands set a runtime result without forcing autosave, with native-backed why
rows reading text, metadata, and provenance from the C++ host store when
available. `flush memory now` and
`flush_immediately` plans still force a flush even when autosave is disabled.
Durability mode commands (`switch to volatile/session-safe/project-safe mode`)
now change the repository `durability_mode`, toggle WAL eligibility for that
mode, persist the checkpoint mode in `manifest.json`, and replay post-checkpoint
mode changes from `CONFIG` WAL records. In `project_safe`, explicit
project/permanent `remember` commands force the runtime flush path even when
repository autosave is disabled; session-scoped scratch memory still batches.
`GraftRepository.apply_extraction_candidate(s)` now provides the no-GPU
extractor interface: classifier-style candidates can be written directly,
queued for review, ignored, or used to supersede active semantic memory under a
conservative confidence/conflict policy. High-confidence duplicate candidates
for the same active scoped fact now reinforce the existing node instead of
creating duplicate fact nodes: source links are merged, confidence is raised,
and reinforcement metadata is WAL-recorded. WAL-only recovery now replays those
extractor supersession and metadata records, along with explicit correction
records, so a crash before the next manifest does not resurrect superseded facts
as active memory or lose duplicate-fact reinforcement. WAL recovery also
preserves `no_fold` fold-abort exemptions from metadata state records, which
keeps rejected librarian windows from looping
after recovery.
Public extractor-candidate execution now finishes through `GRMRuntime`: direct
calls to `apply_extraction_candidate(s)` autosave when the repository is in
autosave mode, report an `extraction` runtime event, and page through the same
runtime boundary as chat-triggered extraction. Turn-triggered extraction still
runs inside the enclosing chat/add-turn event so it does not double-flush.
Optional runtime extraction orchestration is now wired into completed chat and
scripted turn ingestion: `GraftRepository(..., extractor=...)` calls the
extractor on newly deposited turn/recall grafts, passes `source_grafts` and
turn text into `apply_extraction_candidate(s)`, records the last extraction
result/error, and treats extractor failures as non-blocking WAL-recorded events
unless configured to raise. Malformed candidates inside an otherwise valid
extractor batch are isolated per item, recorded as `EXTRACTION_ERROR` WAL
events with their source graft context, and do not prevent valid siblings from
being applied; `extraction_error_policy="raise"` keeps the strict fail-fast
path for tests and trusted pipelines.
`core.grm_runtime.GRMRuntime` now packages the Python hot-path orchestration
boundary for chat turns, scripted turns, deferred librarian work, review
execution, and explicit memory commands: snapshot, arena/model operation,
extraction/review policy, librarian folding, mutation marking, flush, and
paging. `GraftRepository` keeps the public API and persistence surface.
Review-buffer execution now supports approve, reject, edit, and scope-change
operations. Review item status (`pending`, `approved`, `rejected`) and
post-manifest review edits are replayed from WAL, so review decisions survive
both WAL-only recovery and manifest-plus-WAL reload. Autosave-enabled
repositories now publish review edits, rejections, and approvals through the
same runtime/`flush_now()` path as chat turns and explicit memory commands.
Manifest reload now mirrors retired/cold durable nodes into the native store as
metadata-only placeholders instead of reloading their payload tensors into RAM;
native node ids, active state, route metadata, and graph references stay aligned
with the Python manifest while cold payloads remain cold.
`ArenaCache.route()` now uses that native route index for native-backed MLA
candidates, with the C++ lexical score calibrated to Python's fractional
identifier bonus and multi-key route entries that support digest/era
child-centroid routing with the same max-over-keys law as Python. GQA native
host-store mirroring is enabled, and Qwen3-style raw `|q.k|` route scoring now
runs through the native route index with variable-length route-key lists.
`DeviceArena` now owns native host-reference swap and evict contracts for
replacing `[sink | old mounts | live tail]` with
`[sink | new mounts | live tail]` and dropping stale live spans while
preserving `[sink | mounts]`.
`core/deepseek_v2_lite_tc.py` now has the GRM hook surface (`_capture`,
`_capture_q`, `inject_kv`, `graft_seats`, `live_shift`) and a
`DeepSeekMLAArenaCache` dialect for 512+64 MLA payloads, plus a
`warm_absorbed_decode()` hook to materialize absorbed MLA decode projections
before fresh repository resume attaches graft payloads. On the Project-Tensor
branch `codex/grm-arena-cache-surgery-20260627`, tensor_cuda now exposes fused
CUDA `splice_rows` and `evict_rows` cache-surgery ops, and `ArenaCache` uses
them when present with the old slice/cat chain as fallback. TensorCUDA's fused
`rope_apply` now supports inverse rotation and DeepSeek's pair-swapped
positional layout, and TensorCUDA also exposes `export_rows` plus
`export_rope_rows` so live cache spans can be sliced and inverse-RoPE exported
in one CUDA launch per payload. `export_row_pair` groups one raw span plus one
positional span behind a single C++ binding call, and `export_row_pairs` batches
those pair exports across the layer stack behind one C++ binding call.
`ArenaCache` uses that multi-layer boundary for two-payload cache deposits when
present, with the pair, lower-level export helpers, and composed
`F.apply_rotary` fallback retained. TensorCUDA also exposes
`swap_row_pairs_with_rope` for multi-layer raw splice plus positional re-seat
and splice, `evict_row_pairs` for paired multi-layer eviction, and
`arena_row_pair_transaction` to select swap versus empty-mount eviction, validate
width/token state, and return the new mount-token state with the new cache stack.
`ArenaCache` uses that transaction where dialect shape permits. `ArenaCache` also now handles `live_turns=0`, can reset the
live cache for independent probes, clears transient CUDA allocations after
route/swap/evict, and records greedy-decode cache spans using the number of
tokens actually committed to KV.
The live DeepSeek smoke gate passed last-token graft-vs-in-context parity and
greedy access-code recall, plus repository deposit/flush/reload on CUDA with
the native mirror. `tests/deepseek_grm_arena_gate.py` passed a clean build and
fresh-process resume gate: three document grafts, three turn grafts, RAM/native
flush, reload, routed swaps, and 2/2 greedy read-only recalls using independent
probe caches. `tests/deepseek_grm_full_gate.py` passed the deeper DeepSeek
build/resume gate: five document grafts, eight live turns, a 2 MB graft-device
budget, budgeted manifest reload, native route backend, RAM page-ins, and 4/4
open-ended greedy exact-fact recalls after fresh-process resume. The same full
gate now also passes retained-probe-cache stress (`--keep-probe-cache`) on the
12GB card. DeepSeek librarian consolidation now passes the 50-turn folding
gate with 40 turns retired under 10 digest nodes plus two accepted extractive
era nodes. The extractive era path avoids the prior DeepSeek digest-to-era OOM
by preserving child digest facts verbatim as an index node, then relying on the
existing era-descent reader path. Fresh-process ASTRA recall passes, and a
folded-source turn-5 probe recalls `M05-0685` through the folded memory path.
Current-head cross-architecture gates now also pass outside DeepSeek:
MiniCPM3-4B INT4 MLA completed the 42-turn infinite-context gate at 8/8 probes
with max resident 341 seats, 40MB active device memory, and 91MB RAM payload;
Qwen3-4B BF16 GQA completed the 42-turn descent gate at 8/8 probes with max
resident 429 seats, 266MB active device memory, 435MB RAM payload, and 13 RAM
page-ins. The missing production pieces are still deeper C++/CUDA ownership for
remaining routing/revision policy where it proves useful, model-specific
extraction quality/policy, CUDA route scanning if needed, longer high-context
needle runs, and a broader model-specific graft equivalence matrix beyond
DeepSeek, MiniCPM3 MLA, and Qwen3 GQA.
Native dialect profiles now enforce the graftability contract at construction:
profiles that claim `remountable=true` must declare a RoPE/rotary/relative
position law, while fixed/absolute-position profiles are accepted only when
marked non-remountable. This prevents a fixed-position cache dialect from being
silently routed into the seat-remount path.
Native metadata now also carries structured fact identity
(`subject`/`predicate`/`value`/scope plus temporal fields) beside the JSON blob.
The C++ host store exposes active fact-target scans for equal-value and
conflicting-value matches; Python consumes that native scan for extraction
expire/conflict/reinforcement target discovery, then still applies the
authoritative-intent and temporal-validity policy checks. Unicode and time do
not cross the ABI (rule adopted 2026-07-01 after review found divergence):
the native scans fold ASCII bytes only while Python folds full Unicode;
native exact temporal identity was raw byte-equality ("2026-06-01" never
matched "2026-06-01T00:00:00+00:00"), and the native parser rejects ISO forms
Python accepts (times without seconds, numeric epoch). Python therefore
passes `temporal_mode=0`, applies effective-now and temporal-identity checks
to native scan results itself, and falls back to the Python scan for
non-ASCII queries, corpora, or identity fields on either side of the
comparison. `grm_store_fact_matches_ex()` retains the temporal modes at the
ABI for callers that guarantee canonical inputs.
The native runtime also exposes a side-effect-free extraction policy planner for
the static write/review/supersede/reinforce/expire decisions. Python supplies
the target counts and executes the actual mutation, but the review reason/action
selection now has a C++ boundary instead of being only ad hoc repository code.
Duplicate-fact reinforcement metadata merging also has a native planner:
`grm_store_plan_reinforcement()` chooses the winning write-intent rank, max
confidence, and incremented reinforcement count before Python records metadata
and WAL.
Explicit extractor target-id validation now also has a native state boundary:
`grm_store_filter_active_nodes()` deduplicates requested native ids, ignores
unknown ids, and returns only active targets in request order. Python maps local
graft ids through the native mirror and uses that filtered target set before
supersede/expire policy planning.
Explicit memory-command text targeting has the same native host-state boundary:
`grm_store_active_text_matches()` scans stored node text and active metadata in
native id order, returning only active matches for Python to mutate or display.

`tests/deepseek_grm_turn50_gate.py` now validates the original GRM ephemeral
boat on DeepSeek-V2-Lite INT4: 50 stored turn grafts, live context cleared
between turns (`live_tokens=0` at checkpoints), fresh-process reload, native
routing, RAM page-in, and a turn-1 needle recalled at turn 50. Folding is
disabled by default in that gate to isolate raw repository recall. The same
gate now completes with `--enable-folding` without the prior DeepSeek
consolidation OOM and with actual compression across both generations: 40 raw
turns retire under 10 digest nodes, six digest nodes retire under two
extractive era nodes (`folds_aborted=1`, `no_fold=4`), fresh-process turn-1
needle recall still passes, and an explicit turn-5 folded-source probe recalls
`M05-0685` through folded memory. Identifier-bearing probes now suppress
recency mounts so point reads are not polluted by previous recall turns; the
anaphora path still uses recency when there is no identifier key.

This plan extends Graft Repository Memory from a Python research harness into a
RAM-first memory runtime:

```
VRAM  -> mounted arena only
RAM   -> authoritative live graft repository and routing index
NVMe  -> async durability, cold archive, recovery checkpoints
```

The goal is not a wholesale rewrite. The goal is to split the current system
into stable planes so the hot path can become a C++/CUDA runtime while Python
keeps experiment control, policy, and model-facing orchestration.

---

## 1. Target Architecture

### 1.1 Planes

GRM should separate four concerns that are currently close together:

1. **Arena residency**
   - What is mounted in VRAM right now.
   - Owns cache surgery, graft seating, re-RoPE, un-RoPE, eviction, and
     mount rollback.

2. **Repository residency**
   - What exists as live memory in RAM.
   - Owns host payloads, metadata, routing keys, lineage, fact records, dirty
     state, and active/inactive revision state.

3. **Durability**
   - What has reached NVMe safely.
   - Owns WAL records, node blobs, index shards, manifest checkpoints, fsync
     policy, compaction, recovery, and cold archive layout.

4. **Memory policy**
   - What should be remembered, folded, promoted, pinned, forgotten, revised,
     or written durably.
   - Owns user intent, automatic extraction, mutability classification,
     conflict handling, and review buffers.

The runtime should make these boundaries explicit. A RAM-resident node is a
valid repository node even if it has not been flushed to NVMe yet. A VRAM node
is only a mounted or cached copy. NVMe is durability, not live authority.

### 1.2 Runtime Shape

```
Python API / policy
  |
  | chat(), add_turn(), remember(), forget(), flush_now(), stats()
  v
C++ host runtime
  |
  | HostGraftStore, RouterIndex, MemoryGraph, DirtyQueue, DurabilityWriter
  v
tensor_cuda / C++ CUDA
  |
  | DeviceArena, cache surgery, RoPE transforms, route scans, copy scheduling
  v
GPU cache + model adapter
```

Python decides what operation should happen. C++ owns storage, indexing,
durability, and movement. CUDA owns tensor transformations and hot-path kernels.

---

## 2. C++ Core With Python Shim

### 2.1 Keep In Python

Python should remain the control and research layer until the policy stabilizes:

- `GraftRepository.chat()`
- `add_turn()`
- `add_document()`
- `remember()`
- `forget()`
- `pin()`
- `flush_async()`
- `flush_now()`
- experiment gates
- prompt variants for consolidation
- memory extraction and classification policies
- model-specific route-law experiments

Python is also the right place for compatibility gates while the design is in
motion. It should remain easy to run the existing measured protocols.

### 2.2 Move To C++ Host Runtime

The C++ host runtime should own the durable data model and RAM-first repository:

#### `HostGraftStore`

Responsibilities:

- Allocate and retain RAM payloads.
- Track node lifecycle.
- Track dirty and durable state.
- Provide read handles for arena mounting.
- Provide immutable snapshots for async writers.
- Avoid touching CUDA state from writer threads.

Node payload states:

```
host_present       RAM payload exists
device_present     VRAM payload exists or can be reached through arena cache
dirty              metadata or payload newer than NVMe
wal_logged         recovery intent is recorded
durable            payload and metadata reached an NVMe checkpoint
cold_only          payload evicted from RAM but durable on NVMe
```

#### `RouterIndex`

Responsibilities:

- Keep routing keys resident in RAM.
- Store topical keys, lexical keys, lineage keys, and fact keys.
- Support active/inactive filtering.
- Support kind and scope filtering.
- Support mutable fact revision filtering.
- Support batch rebuild after recovery.

The current routing law can remain model-specific:

- MLA: latent centroid cosine plus lexical and child keys.
- GQA: layer-0 raw `|q.k|` scoring plus normalization, lexical, and lineage.

The index API should hide that dialect split behind a stable query interface.

#### `MemoryGraph`

Responsibilities:

- Track `source_turns`, `source_grafts`, `sources`, `supersedes`,
  `superseded_by`, `derived_from`, and contradiction edges.
- Keep raw conversational evidence separate from durable fact nodes.
- Support descent from digest/era/fact nodes to source spans.

#### `DirtyQueue`

Responsibilities:

- Track dirty node IDs.
- Track dirty byte count.
- Track dirty metadata-only updates separately from payload writes.
- Support durability priority.
- Support "flush permanent facts first."

#### `DurabilityWriter`

Responsibilities:

- Write RAM payload snapshots to NVMe in batches.
- Write WAL records when configured.
- Write node blobs before index shards.
- Write manifest checkpoint last.
- fsync at batch boundaries.
- Mark nodes durable by LSN only after the checkpoint is committed.

The writer must never read CUDA tensors. The hot path must materialize host
payloads before queueing work. Otherwise "async" durability can accidentally
synchronize the GPU and poison turn latency.

Current implementation status: `DurabilityWriter::write_checkpoint()` now
commits through `HostGraftStore::save_checkpoint()`, producing the binary
`grm_store.bin` checkpoint, fsyncing the checkpoint boundary, marking store
nodes durable only after publication, and then writing a text checkpoint
summary. `HostGraftStore::dirty_plan()` / `grm_store_dirty_plan()` now expose
ordered dirty nodes with payload-vs-metadata flags, payload byte counts, and
durability priority so permanent memories flush before project/session scratch.
The Python repository still owns the higher-level WAL/manifest publication
sequence.

### 2.3 Move To C++/CUDA Or `tensor_cuda`

The hot tensor path belongs near `tensor_cuda`:

- cache slice harvesting
- un-RoPE from live cache span
- re-RoPE at arena seats
- graft concat for mount blocks
- arena swap
- arena rollback snapshots
- route-key scans where they are tensor-heavy
- host-to-device copy scheduling
- device payload eviction
- packed graft format decode, if a custom binary format replaces `.npz`

The target is to make one Python call per turn-level operation, not one Python
loop per layer or per payload transformation.

### 2.4 Python Binding Boundary

Expose a narrow binding surface:

```python
store = grm.RuntimeStore(path, dialect, durability="session_safe")
arena = grm.DeviceArena(model_handle, store, arena_width=256)

node_id = store.add_host_node(text, payload, metadata)
ranked = store.route(query_key, lexical_keys, filters)
arena.mount(ranked[:topk])
store.mark_dirty(node_id)
store.flush_async()
store.flush_now()
```

The binding should pass handles and buffers, not serialized JSON blobs, across
the hot boundary.

---

## 3. Memory Data Model

### 3.1 Node Kinds

Use node kinds to separate evidence, knowledge, and control state:

```
turn          raw user/assistant exchange evidence
doc           imported source document
fact          extracted or explicitly written memory
preference    user preference or operating preference
instruction   durable behavioral instruction
task_state    mutable project/session state
artifact      code/file/report/model artifact reference
digest        consolidated turn memory
era           consolidated digest memory
recall        derivative retrieval answer, excluded from routing/folding
anchor        pinned always-mounted context
```

Current `turn`, `doc`, `digest`, `era`, and `recall` concepts remain valid.
The new work adds durable semantic nodes alongside raw graft evidence.

### 3.2 Fact Metadata

A durable fact is not just text. It needs identity, scope, mutability, source,
and revision semantics.

Example:

```json
{
  "kind": "fact",
  "subject": "speed of light in vacuum",
  "predicate": "equals",
  "value": "299792458 m/s",
  "scope": "global",
  "durability": "permanent",
  "mutability": "immutable",
  "write_intent": "imported",
  "confidence": 1.0,
  "created_at": "2026-06-19T00:00:00-04:00",
  "valid_from": null,
  "expires_at": null,
  "source_turns": [],
  "source_grafts": [],
  "supersedes": [],
  "active": true
}
```

Mutable project state example:

```json
{
  "kind": "task_state",
  "subject": "current GRM runtime design focus",
  "predicate": "is",
  "value": "RAM-first repository with async NVMe durability",
  "scope": "project",
  "durability": "project",
  "mutability": "mutable",
  "write_intent": "user_asserted",
  "confidence": 0.95,
  "created_at": "2026-06-19T00:00:00-04:00",
  "valid_from": "2026-06-19T00:00:00-04:00",
  "expires_at": null,
  "source_turns": [44],
  "source_grafts": [91],
  "supersedes": [73],
  "active": true
}
```

### 3.3 Classification Axes

Use these fields consistently:

#### `durability`

```
volatile       keep only in RAM unless promoted
session        keep through the current session
project        persist for this project/repo/task domain
permanent      persist until explicitly forgotten or superseded
```

#### `mutability`

```
ephemeral      expected to expire quickly
mutable        can change and needs revision semantics
stable         unlikely to change but not logically fixed
immutable      definition-level or historical fact
```

#### `scope`

```
conversation   local to current chat
session        local to active runtime session
project        local to this repository/project
user           user-level preference or instruction
domain         domain knowledge
global         broad stable knowledge
```

#### `write_intent`

```
observed        extracted from conversation
inferred        model inferred it
user_asserted   user said to remember or stated it directly
system_asserted from a trusted system source
imported        from a document/corpus
generated       produced by a consolidation pass
```

#### `confidence`

Confidence should be explicit because it affects whether automatic extraction
writes directly or enters a review buffer.

---

## 4. What Becomes A Graft And What Does Not

### 4.1 Core Rule

Conversation turns are evidence. Fact records are memory.

Every meaningful turn can be harvested as a `turn` graft, but not every turn
should become durable semantic memory. Raw turns preserve what happened.
Fact/preference/task nodes preserve what the system should remember and route
as knowledge.

### 4.2 Do Save As Grafts

Save these as graft nodes or graph-backed memory nodes:

- user-stated facts
- explicit instructions
- project decisions
- project state
- user preferences
- durable constraints
- imported documents
- codebase observations
- artifact descriptions
- task handoffs
- stable domain knowledge
- high-value generated summaries that pass fidelity gates
- raw turns that are needed as evidence for later facts

### 4.3 Do Not Promote To Durable Long-Term Fact

Do not automatically promote:

- acknowledgments
- greetings and filler
- assistant speculation
- failed retrieval answers
- low-confidence inferences
- repeated restatements
- facts lacking scope
- mutable state lacking timestamp
- conflicting claims without resolution
- derivative recall answers that add no new identifiers
- model-generated summaries that fail fidelity coverage

Such material may still exist as raw `turn` evidence and may be foldable.

### 4.4 Automatic Classifier Output

The memory classifier should emit candidates, not just a yes/no decision:

```json
{
  "candidate_type": "fact",
  "text_span": [120, 155],
  "subject": "GRM hot tier",
  "predicate": "is",
  "value": "RAM-resident before async NVMe durability",
  "durability": "project",
  "mutability": "stable",
  "scope": "project",
  "write_intent": "user_asserted",
  "confidence": 0.96,
  "action": "write_direct"
}
```

Possible actions:

```
ignore
keep_turn_only
write_direct
review_candidate
update_existing
supersede_existing
pin
expire
```

### 4.5 Explicit User Intent Overrides Classifier

Automatic extraction should be conservative. Explicit chat commands should
override classifier hesitation unless the command is unsafe or ambiguous.

Examples:

```
remember permanently: ...
remember this for the project: ...
this is temporary: ...
do not remember this
forget: ...
update memory: ...
mark this as mutable
pin this
```

---

## 5. Arena Section Harvesting

### 5.1 Problem

The current system can deposit whole turns or documents. The expanded runtime
should be able to save specific sections of the memory context as intentional
grafts:

- a single user assertion
- a specific assistant answer span
- a fact-bearing sentence
- a decision and its rationale
- the generated output influenced by specific mounted grafts
- a mounted source bundle that should be consolidated

This requires provenance over the arena, not just text storage.

### 5.2 Segment Types

Track these segment scopes:

```
sink_span             permanent sink/anchor seats
arena_mount_span      mounted graft seats
recency_span          mounted recent turn grafts
prompt_span           current user input
answer_span           generated assistant output
exchange_span         prompt + answer
fact_span             exact token range for an extracted fact
decision_span         exact token range for a project decision
derived_span          output range influenced by selected mounts
```

### 5.3 Provenance Record

Every live segment should be able to answer:

```json
{
  "segment_id": 128,
  "node_id": 91,
  "segment_type": "answer_span",
  "token_start": 302,
  "token_end": 347,
  "seat_start": 781,
  "seat_end": 826,
  "source_turn": 44,
  "mounted_grafts": [12, 19, 33],
  "route_attempt": 0,
  "clean_room": false,
  "generated_after_mounts": true
}
```

This metadata enables targeted graft creation and auditability.

### 5.4 Section Harvest Methods

#### Method 1: Text-span re-harvest

Take the exact text span and run a standalone harvest.

Pros:

- clean routing key
- clean payload
- independent of live cache pollution
- easy to validate

Cons:

- costs another forward
- not free during hot path

Use for durable facts, preferences, instructions, and stable project memory.

#### Method 2: Cache-sliced harvest

Slice the already-existing live cache span, then un-RoPE into a position-free
payload.

Pros:

- cheap if the span is already in cache
- captures what was actually generated/read in context
- useful for turn evidence and answer spans

Cons:

- routing keys may be polluted by context
- needs standalone key generation or a separate key correction path
- provenance is mandatory

Use for raw turns, recent conversation, and evidence spans.

#### Method 3: Hybrid section graft

Use cache-sliced payload plus standalone routing key.

Pros:

- preserves hot-path savings
- avoids contextualized centroid pollution
- matches the existing lesson from turn harvests

Cons:

- still requires partial forward for key generation

Use as the default for conversational turn grafts.

#### Method 4: Source-linked fact node without new K/V payload

Create a fact record that links to source spans and source grafts but does not
immediately harvest a separate fact graft.

Pros:

- cheap
- good for review buffers
- good for mutable task state

Cons:

- retrieval must mount source evidence or synthesize a fact card later

Use for low-confidence candidates, temporary state, or deferred durability.

#### Method 5: Consolidated bundle graft

Mount a selected set of source grafts, generate a digest/chronicle, run fidelity
QC, and store the result as a digest or fact bundle.

Pros:

- compresses seats
- can preserve lineage and source descent

Cons:

- model-generation fidelity gate is mandatory
- not hot-path unless explicitly requested

Use in idle time or explicit "save this as durable memory" operations.

### 5.5 Recommended Default

For normal chat:

1. Save full exchange as a RAM `turn` graft using cache-sliced payload.
2. Generate standalone route key or partial route key.
3. Run extractor on text spans.
4. Promote high-confidence or explicit facts to fact nodes.
5. Re-harvest permanent fact spans standalone during idle or strict durability.
6. Fold raw turns later, preserving fact nodes independently.

---

## 6. RAM-First Durability And Batching

### 6.1 Hot Path

The hot path should not block on full NVMe persistence.

On each chat turn:

1. Generate answer.
2. Harvest turn payload or section payload.
3. Store payload in RAM.
4. Update RAM metadata and routing index.
5. Mark node dirty.
6. Optionally append a small WAL record.
7. Queue payload for durability.
8. Return answer.

The node is immediately routable once in RAM. Durability can lag.

### 6.2 Durability Modes

#### `volatile_fast`

Properties:

- no WAL
- no blocking flush
- RAM-first only
- crash may lose recent turns

Use when raw throughput matters more than recovery.

#### `session_safe`

Properties:

- append lightweight WAL entries for node creation and metadata intent
- batch full payload writes
- manifest/index checkpoint later
- crash can recover text, metadata, and pending payload obligations

Use as the default interactive mode.

#### `project_safe`

Properties:

- explicit project facts and instructions get high-priority flush
- raw turn payloads still batch
- fsync batches during idle or memory pressure

Use when project memory must survive crashes.

#### `durable_strict`

Properties:

- synchronous flush for explicit permanent memories
- expensive but clear
- should be user-commanded or policy-commanded

Use for "remember permanently" and critical corrections.

### 6.3 Batch Triggers

Flush dirty RAM nodes when any of these triggers fire:

```
dirty_node_count >= N
dirty_bytes >= B
time_since_last_flush >= T
available_ram <= low_watermark
session_idle == true
user_calls_flush_now()
explicit_permanent_memory_written
shutdown_requested
```

Suggested starting values:

```
N = 64 nodes
B = 256 MB
T = 5 seconds for metadata, 30 seconds for payloads
low_watermark = configurable, for example 20 percent free RAM
```

These are runtime knobs, not constants.

### 6.4 Batch Write Order

Use an ordered commit:

1. Snapshot dirty node metadata and host payload references.
2. Write or append node payload blobs.
3. Write index shard updates.
4. Write graph/revision updates.
5. Write manifest checkpoint.
6. fsync manifest/checkpoint boundary.
7. Mark nodes durable by committed LSN.
8. Optionally compact older blobs.

Never mark a node durable before its payload and metadata are both reachable
from the committed checkpoint or WAL recovery path.

### 6.5 WAL Strategy

The WAL should be cheap. It does not need to store every full K/V payload in
strict mode unless desired.

WAL record examples:

```
NODE_BEGIN node_id kind created_at text_hash payload_hash
NODE_TEXT node_id compressed_text
NODE_META node_id metadata_delta
NODE_PAYLOAD_PENDING node_id expected_payload_hash
NODE_DURABLE node_id blob_id offset length payload_hash
REVISION_SUPERSEDE old_id new_id
CHECKPOINT manifest_id lsn
```

Recovery can rebuild:

- text and metadata for recent nodes
- dirty queue for payloads that were not durably written
- active/inactive revision graph
- router index from committed nodes

### 6.6 File Layout

Possible NVMe layout:

```
repo/
  manifest.json
  wal/
    000001.wal
    000002.wal
  blobs/
    000001.grmb
    000002.grmb
  index/
    route_000001.grmi
    lexical_000001.grmi
  nodes/
    optional legacy npz compatibility
```

The current `.npz` node format can remain as compatibility while the binary
blob format is introduced.

### 6.7 RAM Pressure

When RAM pressure rises:

1. Keep active routing index.
2. Keep active conversation turns.
3. Keep pinned anchors and preferences.
4. Keep dirty payloads until durable.
5. Spill durable cold payloads from RAM first.
6. Reload cold payloads from NVMe on demand.

Dirty payloads should be high priority for flushing because they block RAM
reclamation.

---

## 7. Mutable Versus Stable Memory

### 7.1 Revision Rule

Mutable facts should never be overwritten in place. Write a new revision and
deactivate the old one.

```
fact:73 active=false superseded_by=91
fact:91 active=true supersedes=[73]
```

This keeps auditability and lets old turns remain evidence without poisoning
current answers.

### 7.2 Mutability Examples

Immutable:

- speed of light in vacuum
- mathematical definitions
- historical commit hash after it exists

Stable:

- project architecture decisions
- documented model shape sheets
- measured benchmark result with date and protocol

Mutable:

- current project focus
- active branch
- current training run state
- service URL
- user preference that may change

Ephemeral:

- "today's plan"
- temporary debugging hypothesis
- current shell/session state

### 7.3 Retrieval Policy

When answering:

1. Active fact nodes outrank old turns.
2. User-asserted facts outrank inferred facts.
3. New active revisions outrank superseded revisions.
4. Immutable facts can be cached aggressively.
5. Mutable facts require recency and scope checks.
6. Superseded facts are evidence only, not answer authority.

### 7.4 Conflict Handling

If a new candidate conflicts with active memory:

- explicit user correction -> supersede old memory
- imported trusted source -> create pending conflict if user memory disagrees
- inferred candidate -> review buffer
- assistant-generated claim -> do not supersede without confirmation
- candidate-provided `supersedes` targets require user/system authoritative
  intent; non-authoritative extractor targets go to review

If a new candidate matches the same active scoped fact and passes the direct
write threshold, reinforce the existing memory node rather than creating a
duplicate fact. Low-confidence duplicates still go through review before their
source evidence is trusted.

If a reviewed candidate is explicitly approved and carries a complete
subject/predicate/value fact triple, approval is treated as user-authoritative
application of the same policy: equivalent facts reinforce the active node,
while same-scope conflicts supersede the old fact. Lower-structure review notes
still approve through the plain `remember()` path.

---

## 8. User Explicit Control In Chat

### 8.1 Commands

Support direct memory commands from the chat window:

```
remember permanently: ...
remember this for the project: ...
remember this for this session: ...
this is temporary: ...
do not remember this
forget: ...
update memory: ...
correct memory: ...
cull graft <id> max tokens <n>
cull graft <id> into sections max tokens <n>
split graft <id> into sections
select graft <id> span <start> <end> [label <name>]
mark memory mutable: <query>
mark memory stable: <query>
mark memory immutable: <query>
pin memory: <query>
unpin memory: <query>
show memory about: ...
why do you remember: ...
flush memory now
switch to volatile mode
switch to volatile-fast mode
switch to session-safe mode
switch to project-safe mode
```

### 8.2 Explicit Command Semantics

`remember permanently`

- write fact/preference/instruction node
- durability = permanent
- mutability default = stable unless user says immutable/mutable
- flush priority = strict or high

`remember this for the project`

- scope = project
- durability = project
- flush priority = high

`remember this for this session`

- scope = session
- durability = session
- may remain RAM-first

`this is temporary`

- mutability = ephemeral
- expiry required or default end-of-session

`do not remember this`

- raw turn may still exist as transient evidence until normal cleanup
- no fact/preference promotion
- no durable semantic node

`forget`

- deactivate matching fact/preference/task nodes
- append tombstone
- preserve audit record unless destructive deletion is explicitly supported

`correct memory`

- create new active revision
- mark old revision superseded
- retain source link to correction turn

### 8.3 Memory Review Buffer

Automatic extraction should write uncertain candidates to a review buffer:

```
review_candidate
  text
  proposed_kind
  proposed_scope
  proposed_durability
  proposed_mutability
  source_turn
  confidence
```

The user can approve, reject, edit, or change scope. Runtime support exists via
`approve_review()`, `reject_review()`, `edit_review()`, and
`change_review_scope()`, with WAL replay of review status and edits.

High-confidence explicit user commands bypass review.

### 8.4 Chat UX Requirements

The assistant should be able to answer:

- what it remembered
- why it remembered it
- which source turn or document supports it
- whether it is mutable
- whether it has been flushed to durable storage
- whether it superseded an older memory

This requires metadata to be first-class, not bolted onto text.

---

## 9. Implementation Phases

### Phase 0: Schema And Vocabulary

Deliverables:

- node schema
- metadata vocabulary
- durability modes
- revision semantics
- provenance record shape
- compatibility mapping from current manifest

No GPU required.

### Phase 1: Python RAM-First Refactor

Deliverables:

- split `saved` into `host_present`, `dirty`, `durable`, `device_present`
- RAM node store inside Python
- dirty queue
- `flush_now()`
- `flush_async()` placeholder
- `remember()` API
- metadata attached to node records

Validation:

- existing repository gates should still run when GPU is available
- non-GPU schema tests can run immediately

### Phase 2: Async Durability In Python

Deliverables:

- background writer thread/process
- batched node writes
- manifest checkpoint write-last rule
- optional WAL
- recovery path
- durability stats

Validation:

- simulated crash tests without GPU
- dirty queue replay tests
- manifest checkpoint consistency tests

### Phase 3: Fact And Intent Layer

Deliverables:

- explicit chat memory commands
- fact/preference/task node creation
- mutable revision graph
- review buffer
- conflict policy
- extractor interface

Validation:

- unit tests over text-only examples
- no GPU required for policy tests

### Phase 4: Arena Provenance And Section Harvesting

Deliverables:

- segment provenance records
- text-span graft creation
- cache-span graft hooks
- source-linked fact nodes
- selected-section API

Current status: segment provenance records, source-linked fact nodes, document
and cull span payload slicing, and the public selected-section API are
implemented. `GraftRepository.select_graft_span()` creates a non-retiring child
graft with selected-span provenance and sliced RAM/native payloads; native
provenance mirroring/checkpointing now preserves those explanation records for
why-memory reads. Live cache-span harvesting remains owned by the
TensorCUDA/arena export gates.

Validation:

- text-only provenance tests
- GPU gates later for payload equivalence

### Phase 5: C++ Host Runtime

Deliverables:

- `HostGraftStore`
- `RouterIndex`
- `MemoryGraph`
- `DirtyQueue`
- `DurabilityWriter`
- Python bindings

Current status: `HostGraftStore`, `RouterIndex`, `DirtyQueue`,
`DurabilityWriter`, and ctypes bindings are implemented for the RAM-authority
checkpoint/routing path. `MemoryGraph` semantics are represented through
structured source/supersession edges in the host store rather than a separate
native graph class.

Validation:

- parity with Python store behavior
- recovery tests
- batch durability tests

### Phase 6: C++/CUDA Arena Runtime

Deliverables:

- device arena handles
- C++ swap/evict/re-seat
- cohesive arena export/re-seat/swap call
- copy scheduling
- route scan acceleration

Validation:

- existing graft equivalence gates
- arena/trips gates
- paging gates
- stress tests under RAM/NVMe pressure

### Phase 7: Production Daemon Option

Deliverables:

- long-running repository daemon
- shared RAM hot graft pool
- multi-agent handles
- exclusive writer or transactional writer model
- health/status endpoint

This should wait until the Python and C++ API boundaries stop moving.

---

## 10. Design Rules

1. RAM is the authoritative live repository.
2. NVMe is durability and cold storage, not the hot path.
3. VRAM is a disposable mount cache.
4. Conversation turns are evidence.
5. Fact records are memory.
6. Mutable facts are revised, not overwritten.
7. Explicit user intent outranks automatic extraction.
8. Async durability must not touch CUDA tensors.
9. A fold that loses facts must abort.
10. Every durable memory must have source metadata.
11. Every answer-relevant mutable memory must know whether it is active.
12. Python remains the policy layer until the policy hardens.
13. C++ owns RAM storage, indexing, durability, and movement.
14. CUDA owns tensor transformations and cache surgery.

---

## 11. Open Questions

- Should permanent explicit memories force synchronous payload durability, or is
  a WAL plus high-priority batch enough?
- Should fact nodes always get their own standalone K/V payload, or should many
  begin as source-linked text records and be harvested lazily?
- How aggressive should automatic extraction be before a review buffer becomes
  noisy?
- Should user-level memories live in a separate repository from project-level
  memories?
- How should contradictions between imported documents and user-asserted
  memory rank?
- What binary payload format should replace `.npz` for high-throughput writes?
- Should RAM payload compression be used before NVMe durability, or only during
  cold compaction?
- How much of route scoring should move into CUDA versus C++ host SIMD first?

---

## 12. First Concrete Build Slice

The smallest useful implementation slice:

1. Add node metadata fields in Python.
2. Add `remember(text, durability, mutability, scope, kind)` API.
3. Keep RAM payloads authoritative.
4. Replace hot-path `save()` dependence with a dirty queue.
5. Add `flush_now()` that writes dirty nodes in commit order.
6. Add explicit memory commands in the chat wrapper.
7. Add unit tests for mutable revision and metadata filtering.

This slice proves the data model and user control without requiring GPU time.
The C++ and CUDA split can then implement the same semantics under the Python
API instead of redesigning the behavior during the port.
