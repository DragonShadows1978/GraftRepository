# GRM GQA Route-Card Database Ledger

Plan: `docs/GRM_GQA_ROUTE_CARD_DB_PLAN.md`

Narrative continuation: `docs/GRM_GEMV_ROUTER_SYNTHESIS.md`

## 2026-07-11 — Worktree and Baseline

Repository state before edits:

- source checkout: `/mnt/ForgeRealm/GraftRepository`;
- source branch: `main`;
- source HEAD: `fdc478ca28a6f74da6570748b80ba67410dc45fb`;
- source worktree clean;
- no pre-existing secondary worktrees.

Created the requested isolated worktree:

```text
/home/vader/GraftRepository-gqa-route-card-db
branch codex/gqa-route-card-db
```

No commits and no pushes.

## 2026-07-11 — Native Record, ABI, and Checkpoint

Implemented:

- first-class `RouteCardRecord` on `HostGraftNode`;
- GQA-only shape/state/finite-key validation;
- public logical fp32 layout `[slots, kv_heads, head_dim]`;
- lossless transpose into the existing GQA router's head-major layout;
- C ABI set/status/JSON/clear calls;
- Python `NativeGraftStore` bindings;
- atomic row replacement and eager immutable-snapshot publication;
- GQA checkpoint version `GRMSTOR11`;
- v1-v10 read compatibility;
- MLA writer remains on `GRMSTOR10`.

Manual native smoke:

- created a GQA node and a `4 x 2 x 4` fp32 card;
- routed it through native raw-qk;
- saved/reloaded the checkpoint;
- verified state, source hash, keys, and ranking after reload.

Build command:

```bash
cmake -S cpp -B /tmp/grm-route-card-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/grm-route-card-build -j2
```

Result: passed; static and shared libraries built.

## 2026-07-11 — Repository and Librarian Slice

Implemented behind `gqa_route_cards=False` by default:

- constructor GQA/slot/native-ABI guards;
- source/profile/payload/lineage/policy hashes;
- deterministic raw-key selector with lowest-position ties;
- fixed child-card derivation for digest/era nodes;
- provisional, pending, ready, failed, and stale lifecycle;
- inactive and recall exclusion;
- deferred explicit job queue with no background thread;
- strict JSON record validation;
- exact source evidence and answer containment;
- source-supported rare identifier gate;
- scratch-cache generation/harvest path;
- cancellation and deadline checks at registered boundaries;
- terminal three-attempt failure retaining provisional keys;
- explicit status and alias-redacted inspection APIs;
- checkpoint hydration and old-GQA provisional synthesis path;
- GQA Python/CUDA routing reads `route_card_keys_native` when enabled.

No librarian output is returned by `chat()`, mounted, appended to live K/V, or
interpreted as a memory command.

## 2026-07-11 — Focused Verification A (initial checkpoint)

New test module:

```text
tests/test_gqa_route_cards.py
```

Coverage:

- native ABI and checkpoint byte round-trip;
- card clear and route removal;
- deterministic dense selector and raw-vector provenance;
- provisional enqueue to hybrid-ready publication;
- live-cache identity/state preservation;
- alias redaction by default;
- unsupported identifier and prompt-injection rejection;
- three-attempt failure with provisional retention;
- cancellation remaining pending without attempt consumption;
- stale/inactive supersession behavior;
- repository restart preserving ready card/ranking inputs;
- MLA opt-in rejection;
- concurrent route/card replacement over immutable snapshots.

Command:

```bash
PYTEST_ADDOPTS='-p no:cacheprovider' \
  python3 -m pytest tests/test_gqa_route_cards.py -q
```

Result: `9 passed, 2 warnings in 7.64s`.

Focused existing regression command selected ten GQA, route, checkpoint,
supersession, snapshot, and CUDA-bank lifecycle tests from
`tests/test_grm_native_runtime.py`.

Result: `10 passed, 2 warnings in 26.99s`.

Running total at this checkpoint: `19 passed`.

## Development Freeze Status

Not yet frozen. Existing artifact inventory confirms:

- 19,722 Qwen3.5 capture shards are present under
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/captures`;
- target shards include layer K and Q tensors;
- the GPT-OSS full composed-session receipt and ragged route index are present
  under `artifacts/grm_e2e/full_session_20260708_P2`;
- that receipt established 7/7 fresh facts but both supersession probes failed
  route competition and used a Python path at about 756 ms.

The slot sweep and the exact availability of replayable GPT-OSS query tensors
must be receipted before selecting the constructor winner. Fresh seeds remain
closed.

## 2026-07-11 — Qwen3.5 Development-Visible Source-Only Sweep

Added:

```text
scripts/grm_gqa_route_card_dev_sweep.py
```

Protocol:

- existing target capture shards only; no fresh data;
- capture root:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/captures`;
- layer 3 raw K/Q;
- 32 deterministic nodes;
- 20 deterministic one-token queries;
- seed `20260711`;
- full ragged raw `|q.k|` versus source-only cards `{2,4,8}`;
- source identity evaluated at recall@3;
- stable node-id ties.

Command:

```bash
python3 scripts/grm_gqa_route_card_dev_sweep.py \
  --nodes 32 --queries 20 --query-tokens 1 \
  --out artifacts/gqa_route_card/dev/qwen35_slot_sweep.json
```

Artifact:

```text
artifacts/gqa_route_card/dev/qwen35_slot_sweep.json
```

Results:

| Route bank | Source recall@3 |
| --- | ---: |
| Full ragged K | 18/20 |
| 2-slot provisional | 2/20 |
| 4-slot provisional | 3/20 |
| 8-slot provisional | 4/20 |

Interpretation:

- Every registered source-only size is worse than full ragged routing on this
  replay.
- This is a red provisional-card pre-gate. It refutes any claim that the
  immediate source medoids are recall-equivalent by themselves.
- It is not the complete development gate because these capture shards do not
  contain the generated future-question alias K needed to form the registered
  hybrid cards, and the GPT-OSS probe Q tensors are not stored in the composed
  receipt.
- No slot winner is frozen. No fresh seed is opened. The selector and slot set
  are not retuned after seeing the result.

## 2026-07-11 — Native Publication TSAN and Card CPU/CUDA Parity

Updated the existing router TSAN harness so its writer publishes route cards
through `grm_store_set_route_card` while concurrent readers call
`grm_store_route_gqa`.

Command:

```bash
python3 scripts/grm_router_tsan_gate.py \
  --nodes 16 --iterations 200 --readers 2
```

The host hit its known GCC TSAN ASLR mapping failure on the first invocation.
The harness retried under `setarch x86_64 -R` and passed:

```text
router TSAN stress passed: nodes=16 iterations=200 readers=2
```

Added a card-only dense parity receipt:

```text
scripts/grm_gqa_route_card_cuda_parity.py
artifacts/gqa_route_card/dev/cpu_cuda_parity.json
```

Protocol: 64 nodes, four slots, two KV heads, eight query heads, head dim 256,
20 queries, top-5, seed `20260711`, no model load. The first sandboxed attempt
could not open CUDA (`cudaErrorNoDevice`); the approved device-access rerun
engaged the 4070 Super.

Result:

- stable CPU/CUDA top-5 parity: `20/20`;
- reused CUDA batch wall: `1.405-1.562 ms` for 20 queries;
- device route: about `1.182 ms` total, `0.06565 ms/query` after two warmups;
- first wall sample: `47.905 ms`, reported separately from reused steady state;
- native CPU total: `49.223 ms` for the same 20 queries.

This is synthetic parity evidence at 64 nodes. It does not satisfy the fresh
512-node latency gate.

## 2026-07-11 — Final Implementation Verification

The focused module grew after the initial checkpoint to cover:

- v10 GQA migration into v11 cards;
- byte-identical MLA v10 checkpoint output against the unmodified library;
- text, payload, and lineage invalidation;
- generator-version and policy-hash invalidation while already queued;
- generation, harvest, and pre-publication cancellation boundaries;
- exact schema/evidence/duplicate/identifier validation receipts;
- persistence of rejected-alias receipts with default-view text redaction;
- recall exclusion and digest derivation from child cards;
- fixed-card digest eligibility plus missing/stale/inactive exclusion in Python
  and dense CUDA fallback routing;
- restart preserving ready bytes/audit state without requeue;
- native stale cards being audit-only even when a direct caller leaves node
  metadata active, plus fresh-card reactivation.

Final focused command:

```bash
python3 -m pytest -q tests/test_gqa_route_cards.py
```

Result after the hybrid continuation: `23 passed, 2 warnings in 18.70s`.
The three added parameterized cases prove the development harness's hybrid
builder is byte-identical to the repository builder at 2, 4, and 8 slots.

Broad existing regressions:

| Suite | Result |
| --- | ---: |
| `tests/test_grm_router_baseline.py` | 21 passed |
| `tests/test_grm_runtime_lifecycle.py` | 101 passed |
| `tests/test_grm_native_runtime.py` | 121 passed |

These are 266 distinct pytest cases across the three existing modules and the
new focused module. After the final stale-state hardening, six targeted native
active/revision/expiry/concurrent-snapshot tests were rerun and passed; they are
subsets of the 121 and are not added to the distinct total.

Final compile/static receipts:

- `python3 -m py_compile` passed for the modified Python modules and scripts;
- `git diff --check` passed;
- a clean Release configure/build at `/tmp/grm-route-card-build-final` built
  both static and shared native libraries;
- the final route-card writer/readers TSAN stress passed after the harness's
  registered `setarch x86_64 -R` retry for the host GCC mapping issue.

An attempted pytest invocation of the historical GQA repository/model files
was not counted: those files are standalone `build`/`resume` programs, so
pytest initialized Qwen but collected zero cases. The actual two-phase model
gate was not opened because the development freeze has no slot winner.

## Work-Order Stop

Implementation is complete behind the opt-in flag, but adoption evidence is
not green. The development-visible source-only replay is red and the complete
hybrid/GPT replay cannot be constructed from the stored receipts alone. No
slot count, prompt/policy bundle, or fresh seed is frozen. Accordingly:

- Qwen3.5 seed `20260713` was not opened;
- GPT-OSS-20B seed `20260714` was not opened;
- blind lead verification and mutation-score adoption gates remain unopened;
- the feature remains disabled by default.

No commits and no pushes were made.

## 2026-07-11 — Operator Reopened the Untested Hybrid Question

The operator chose to continue because the red source-only result does not
adjudicate the route-card design's central claim: validated future-question K
may restore addressability that a tiny source witness set loses.

The continuation protocol is frozen in the plan before running it. It reuses
the exact prior 32-node/20-query capture selection and tests real hybrid cards
with the local Qwen3.5-9B model. The selector, slot set, validator, alias limit,
and prompt version are unchanged. A two-node non-scoring qualification precedes
the one-shot hybrid replay. Fresh seeds remain sealed.

### Qualification receipt

Command:

```bash
python3 scripts/grm_gqa_route_card_hybrid_dev.py qualify
```

Protocol SHA-256:
`29f79085e43c1d739f5ec6d8d20b5437f552ff1f353bbdbb91f29d06b134af9b`.

The local INT4 Qwen3.5-9B model loaded with the expected 32-layer hybrid
profile and attention layers `{3,7,11,15,19,23,27,31}`. Results:

| Node | Accepted aliases | Concatenated layer-3 K | Result |
| ---: | ---: | --- | --- |
| 0 | 3 | `[4,40,256]` | pass |
| 1 | 4 | `[4,56,256]` | pass |

Both generations were parseable JSON; strict validation and native-K harvest
completed without error. Qualification intentionally calculated no retrieval
score. The registered full replay is therefore open.

## 2026-07-11 — Hybrid Development-Visibility Order (Scope)

New work order accepted: the 2026-07-11 source-only sweep above was red
({2,4,8} scored 2/20, 3/20, 4/20 recall@3 vs 18/20 full ragged). The registered
card design is hybrid (half source medoids, half generated-alias medoids), and
the hybrid replay could not be built because no alias K existed. Scope for this
entry:

1. Reconnaissance receipt: inventory capture machinery for (a) obtaining GQA K
   for arbitrary text through the repository's own capture path, (b) whether
   the 32 dev nodes' source texts and 20 dev queries are recoverable from
   existing artifacts. Registered rail: if layer-0 capture for arbitrary alias
   text is not constructible without touching Project-Tensor or model-port
   internals, stop after the inventory and report the gap. If only the layer-3
   translation-capture frame is available, the hybrid sweep may run entirely
   in that frame for dev visibility only (must be labeled explicitly).
2. Freeze the alias-generation prompt (`gqa-route-card-v1`) verbatim in the
   ledger before any generation run.
3. Generate candidate aliases per dev node via local ollama Qwen3.5
   (localhost:11434, temperature 0), validate against the three-field
   contract + lexical rule, record per-node accept/reject receipts.
4. Harvest alias K in the same frame as the Step-0 decision; build hybrid
   cards per the registered greedy selector; no averaging/normalizing/
   rescaling.
5. Run the registered hybrid sweep at {2,4,8} slots vs full ragged, same 32
   nodes/20 queries/seed 20260711, recall@3, stable ties. Artifact:
   `artifacts/gqa_route_card/dev/qwen35_hybrid_sweep.json`.
6. Ledger the results table, interpretation bounded to development-visible
   evidence, and exact commands. No commits/pushes/network/pip
   installs/subagents used in this entry (ollama is a local service call via
   curl, not network egress).

Work begins below.

LEAD ANNOTATION (Fable, 2026-07-11): the scope entry above was a
Sonnet continuation order STOPPED BY THE OPERATOR mid-Step-0
(reconnaissance) when Codex quota reset; no generation, harvesting, or
sweep ran; no artifact was written. Its one useful finding, preserved:
the 32 dev nodes' source texts ARE recoverable — capture
manifest.jsonl labels point into scribe_mint_v1/manifest.jsonl by
doc_index with token_start/token_end spans, so alias generation can
reconstruct exact source substrings from existing receipts. The
hybrid-evidence leg is superseded by the operator's own Codex session.

## 2026-07-12 — Frozen Qwen3.5-9B Hybrid Replay Result

The operator's Codex continuation used the already-frozen protocol above, the
exact prior 32 target-capture paths and 20 query locations, and the local INT4
Qwen3.5-9B serving model. It did not use the stopped local-Ollama proposal.

Command:

```bash
python3 scripts/grm_gqa_route_card_hybrid_dev.py full
```

The run reused the two qualified receipts and generated the remaining 30 under
protocol SHA-256
`29f79085e43c1d739f5ec6d8d20b5437f552ff1f353bbdbb91f29d06b134af9b`.
It completed all 32 nodes in about 842 seconds of per-node
generation/harvest time:

- 42 accepted, source-grounded aliases across 12/32 nodes;
- 20/32 nodes retained provisional cards;
- of those 20, 14 hit the frozen 384-token rail with incomplete JSON, five
  returned an empty JSON array, and one returned other malformed JSON;
- validator receipts: 42 accepted, two rejected because the answer was not
  inside its evidence, one rejected because evidence was not an exact source
  substring;
- zero accepted aliases contained unsupported identifiers.

The 20 frozen query locations were scored once:

| Slots | Full ragged recall@3 | Provisional recall@3 | Hybrid recall@3 | Hybrid vs provisional | Hybrid vs full |
| ---: | ---: | ---: | ---: | --- | --- |
| 2 | 18/20 | 2/20 | 3/20 | no worse | fail |
| 4 | 18/20 | 3/20 | 2/20 | worse | fail |
| 8 | 18/20 | 4/20 | 3/20 | worse | fail |

No hybrid card reproduced the full bank's top three on any of the 20 queries.
No registered size passed. Classification:

```text
ROUTE-QUALITY-RED / LIBRARIAN-YIELD-RED
```

Receipt files:

```text
artifacts/gqa_route_card/dev/qwen35_hybrid_v1/protocol.json
  sha256 91a3c8f776e5e4accc7792513f7c4adb41de56c828464c8fac220860b49c04f6
artifacts/gqa_route_card/dev/qwen35_hybrid_v1/qualification.json
  sha256 47f59a2a2dfde42f6d6e81dcbecad05d7aac6f986903d08b0bcaff81b146b73b
artifacts/gqa_route_card/dev/qwen35_hybrid_v1/qwen35_hybrid_sweep.json
  sha256 5f138a81b1de0161731945acfd515d7075fdc74849ac00e426f227d037c03b95
```

The sampled post-run GPU residency returned to 171 MiB; this is a post-run
`nvidia-smi` snapshot, not an allocator peak.

Per the preregistered rule, this fixed `{2,4,8}` half-source/half-alias line is
stopped. The prompt, token rail, selector, slot split, and query set are not
retuned after the result. No GPT replay, fresh seed, blind lead gate, mutation
gate, default enablement, commit, or push is opened.

### Post-hoc shortlist diagnostic (not an adoption gate)

After sealing the failed top-three result, the stored ranks were read once to
check whether tiny cards could still justify a coarse-card/exact-ragged rerank.
They do not:

| Slots | Hybrid recall@8 | Hybrid recall@16 | Hybrid recall@24 |
| ---: | ---: | ---: | ---: |
| 2 | 4/20 | 11/20 | 17/20 |
| 4 | 6/20 | 13/20 | 16/20 |
| 8 | 6/20 | 12/20 | 19/20 |

At 24/32 candidates, the filter retains 75% of nodes and still loses sources.
This is too little pruning and too much recall loss to support a two-stage
route-card design. The evidence-backed successor direction is an exact ragged
GQA CUDA router (for example, offset-segmented or length-bucketed K with exact
masked maxima), not another fixed tiny-card variant.

Full retrospective and engineering interpretation:
`docs/GRM_GQA_ROUTE_CARD_DB_POSTMORTEM.md`.
