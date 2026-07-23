# GRM Revision-Aware Resolver Ledger

Execution record for `docs/GRM_REVISION_AWARE_RESOLVER_PLAN.md`.

## 2026-07-12 — Work Order Opened

Repository state:

- Source repository: `/mnt/ForgeRealm/GraftRepository`
- Local `main`: `fdc478ca28a6f74da6570748b80ba67410dc45fb`
- Successful exact-ragged branch/base:
  `8e11bbd9107436720b51db5634746a4c2360c018`
- Isolated worktree: `/home/vader/GraftRepository-revision-aware-resolver`
- Branch: `codex/grm-revision-aware-resolver`
- Opening tree: clean

Why the branch is based on `8e11bbd`, not local main: the approved concept is a
successor to exact-ragged GQA routing, and local main has not yet incorporated
that commit. Dropping the proven route path would invalidate the resolver's
CUDA-preservation gate.

Opening evidence:

- Exact-ragged leaf routing is development-green: 175/175 model-native Qwen
  queries, 512-node resident p50/p95 1.59/2.01 ms, cold build 147 ms.
- Existing composed GPT-OSS full session: 7/7 fresh facts recalled, no stale
  value emitted, but the current `orion pin` revision failed under competition.
- Sharp receipt: authoritative fact node 7 was semantic rank 3 behind prior
  probe/update turns; mount fitting seated nodes 5 and 7, and readout emitted
  node 5's unrelated `cypher bridge` value.
- `HostGraftStore` already persists `subject`, `predicate`, `value`, `scope`,
  active state, and supersession edges and exposes exact `fact_matches`.
- The missing piece is query-time family resolution and mount-policy
  integration, not another store.

Initial architecture decision:

- An exact unambiguous family pin is a post-semantic policy.
- Corrections without explicit identity may receive only grounded identity
  shared by old/replacement text or inherited from one existing family.
- Ambiguity and unsupported inputs preserve semantic behavior unchanged.
- First implementation remains opt-in.

Next action: create the deterministic competition baseline before changing
product code.

## 2026-07-12 — P0 Deterministic Competition Baseline

Instrument: `scripts/grm_revision_resolver_dev_gate.py`, twelve distinct fact
families. Every case contains an inactive old fact, a higher-semantic derived
probe turn, a higher-semantic authoritative-update turn, and the lower-semantic
active replacement fact. Query labels exactly match the family.

Command:

```text
python3 scripts/grm_revision_resolver_dev_gate.py \
  --out artifacts/grm_revision_resolver/baseline.json
```

Receipt:

- current active fact rank 1: `0/12`;
- current active fact top 3: `12/12`;
- current fact in the two-seat first mount: `0/12`;
- inactive stale fact ranked: `0/12`;
- expected baseline mode: pass.

The baseline reproduces the registered defect without a model: semantic and
generic content-word lexical routing find the correct family but derived turns
occupy the first mount ahead of the authoritative fact. Product code remained
unchanged for this receipt.

Next action: P1 grounded correction identity, then native/Python resolution.

## 2026-07-12 — P1-P3 Implementation

Implemented on the isolated branch without commits or pushes:

- grounded correction identity from explicit metadata, one inherited family,
  or the exact shared `current <label> value is <value>` correction grammar;
- Unicode-aware fail-closed Python resolution;
- a native exact subject-signature index backed by the existing fact identity,
  active state, scope, and checkpoint record;
- `grm_store_resolve_fact_query` and
  `NativeGraftStore.resolve_fact_query(...)`;
- a shared/unique fact-state publication lock around native resolver reads and
  identity/lifecycle mutations;
- opt-in `revision_aware_resolver=True`, stable semantic reordering, exact
  single-node first mount, and `inspect_last_route_resolution()`;
- an explicit `--revision-aware-resolver` composed-session driver option that
  preserves the previous top-k probe behavior on no-match and uses the exact
  single fact on a resolver hit.

The feature remains disabled by default. Resolver metadata never becomes
mounted text and does not execute commands or alter truth.

## 2026-07-12 — Deterministic And Native Development Gates

Deterministic replay:

```text
python3 scripts/grm_revision_resolver_dev_gate.py --resolver \
  --out artifacts/grm_revision_resolver/resolver_python_v2.json
```

- current fact rank 1: `12/12` (baseline `0/12`);
- current fact first single mount: `12/12` (baseline `0/12`);
- current fact top 3: `12/12`;
- stale ranked: `0/12`;
- default-disabled baseline rankings remained unchanged.

The initial 512-node native implementation was correctness-green but missed
the frozen compute rail: p50/p95 `0.429/0.539 ms` against p95 `0.25 ms`. The
cause was a union over a shared subject-token posting that visited all 512
facts. Replacing the candidate discovery step with exact sorted subject-token
signatures retained the same subset/addressability law and produced:

```text
artifacts/grm_revision_resolver/native_gate_signature_index.json
```

- exact resolution: `100/100`;
- checkpoint/restart exact resolution: `100/100`;
- p50/p95/max: `0.122/0.133/0.175 ms`;
- concurrent identity mutation errors: `0`;
- sampled RSS after warmup/measurement: `35,204/35,368 KiB`;
- native gate: pass.

Verification completed before the model stop:

- new identity, ambiguity, specificity, inactive/expired, Unicode, correction,
  exact-mount, native parity, and restart tests: `10 passed`;
- full runtime-lifecycle plus new resolver tests: `111 passed`;
- affected native selectors: `18 passed, 103 deselected`;
- CMake release static/shared build: pass.

The exact-ragged CUDA regression remained green at all registered sizes:

```text
artifacts/grm_revision_resolver/exact_ragged_regression.json
```

- parity and CUDA engagement: pass;
- 512-node bridge p50/p95: `1.597/1.830 ms`;
- 512-node cold build: `164.31 ms`;
- padding ratio: `2.1992x`;
- VRAM monotonic-growth check: pass;
- verdict: `QUALITY-GREEN`.

## 2026-07-12 — GPT-OSS Development Stop

Command:

```text
python3 scripts/grm_e2e_session.py --mode smoke \
  --session-dir artifacts/grm_revision_resolver/gpt_oss_smoke_resolver \
  --native-lib /tmp/grm_revision_resolver_build/libgrm_runtime.so \
  --revision-aware-resolver --skip-gpu-idle-check
```

The bounded session completed ten turns and a real process checkpoint/restart.
The ordinary `cypher bridge` probe passed. The superseded `orion pin` probe
produced the exact resolver receipt after restart:

- resolver backend/state: `native/exact`;
- matched family: `orion pin/value/project`;
- authoritative fact node: `3`;
- source rank: `1`;
- mount plan/fitted mounts: `[3]` / `[3]`;
- stored node text: `The current orion pin value is Kestrel-9-Tango.`;
- stale `Auric-4-Alpha` emitted: no;
- answer: `The current orion value is Kestrel.`;
- exact expected value emitted: no.

Score: `1/2` probes. Status: `probe_failures`.

This is the registered hard stop. Selection authority, lifecycle, restart,
and mount isolation all worked, but the model did not read the complete code.
The active fact graft was a bare 15-token correction payload; the richer
Harmony-formatted authoritative update turn was 92 tokens and remained a
separate node. Pinning only the metadata-authoritative bare graft removed
competition but also removed the richer readout context. The partial answer
(`Kestrel`) is evidence of lossy readout, not stale selection or a competing
fact.

The fresh 40-turn GPT-OSS gate, fresh Qwen identity corpus, TSAN campaign, and
mutation campaign were not run after this correctness failure. The branch is
preserved uncommitted as a negative development result. A successor would need
to bind family authority to an evidence-bearing, serving-dialect-compatible KV
payload (or its explicit lineage), which is a different concept and work
order.
