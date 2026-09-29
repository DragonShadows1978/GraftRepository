# GRM GQA Route-Card Database Plan

Status: implementation complete; development hybrid-alias continuation failed
its frozen Qwen gate; feature disabled by default and rollout stopped.

Branch/worktree: `codex/gqa-route-card-db` at
`/home/vader/GraftRepository-gqa-route-card-db`.

Tracking:

- Operational ledger: `docs/GRM_GQA_ROUTE_CARD_DB_LEDGER.md`
- Router narrative: `docs/GRM_GEMV_ROUTER_SYNTHESIS.md`

## Decision

Extend the existing native `HostGraftStore` and `RouterIndex`. Do not create a
second service, a second repository, or a mountable summary-graft format.

The feature is GQA-only. MLA already routes through compact latent centroids;
passing `gqa_route_cards=True` to an MLA repository is a configuration error.

A route card is a fixed-shape, fp32 index record attached to a normal graft
node. It is searched by the router and audited through explicit APIs. It is
never a node, never a source of mounted text, never a K/V payload, and never an
authority for facts. Mounting a routed node always retrieves the original
ragged graft payload.

This design follows three useful precedents without importing their serving
stacks: lightweight centroid bags from [PLAID](https://arxiv.org/abs/2205.09707),
generated query addressability with filtering from
[Document Expansion by Query Prediction](https://arxiv.org/abs/1904.08375) and
[Doc2Query--](https://arxiv.org/abs/2301.03266), and explicit cache correctness
boundaries from [Prompt Cache](https://arxiv.org/abs/2311.04934) and
[CacheBlend](https://arxiv.org/abs/2405.16444).

## House Rules

- This plan is the registered intent. Results and deviations go in the ledger.
- Failure is a result. A failed fresh gate is preserved and stops rollout.
- Existing artifacts are development-visible only. They cannot be relabeled as
  fresh evidence.
- Freeze slot count, prompt version, selector version, and validation policy
  before opening either fresh GQA gate.
- No tuning after a sealed fresh seed is opened.
- Keep model-quality, route-quality, compute, and memory claims separate.
- CUDA parity excludes exact fp32 score ties; ties retain stable node-id order.
- Sampled `nvidia-smi` residency is not an allocator peak.
- No commits or pushes in this work order.
- The feature remains disabled by default. Default adoption is a separate
  operator decision after both fresh gates pass.

## Native Schema

`HostGraftNode` gains one optional `RouteCardRecord`:

```text
present
state: provisional | pending | ready | failed | stale
dialect_profile_hash
source_text_sha256
prompt_version
generator_version
slot_count
kv_heads
head_dim
fp32 route_keys [slot_count, kv_heads, head_dim]
lexical_identifiers
aliases + validation receipts
provenance + policy hashes
attempt_count
created_at
updated_at
```

The native public card layout is slot-major. Publication transposes without
averaging into the historical GQA router layout `[kv_heads, slots, head_dim]`.
The original vectors and norms are unchanged.

Checkpoint version `GRMSTOR11` adds the optional card record for GQA stores.
MLA stores continue writing the `GRMSTOR10` byte layout. Versions 1 through 10
remain readable. A version-10 GQA load has no cards; the repository synthesizes
provisional cards from its existing layer-0 raw keys and queues upgrades.

Native C ABI:

```text
grm_store_set_route_card
grm_store_route_card_status
grm_store_route_card_json
grm_store_clear_route_card
```

Python binding:

```python
NativeGraftStore.set_route_card(...)
NativeGraftStore.route_card_status(node_id)
NativeGraftStore.route_card(node_id)
NativeGraftStore.clear_route_card(node_id)
```

Setting or clearing a card takes the native router write lock, replaces the
node's rows, prepares a new immutable snapshot, publishes it, and increments
the router epoch before returning. Existing metadata filters, active bits,
lexical bonuses, stable node IDs, graph edges, and mount behavior are reused.

## Immediate Provisional Card

Registered slot candidates are `{2, 4, 8}`. The development sweep is one-shot;
the smallest complete pass wins. Until that freeze is receipted, the code's
constructor value is a development setting, not an adoption decision.

For each KV head independently:

1. Consider finite layer-0 raw source K vectors only.
2. Select the highest-L2-norm vector; lowest source position wins a tie.
3. For every later slot, maximize:

   ```text
   (candidate_norm / maximum_source_norm)
     * min_selected(1 - abs(cosine(candidate, selected)))
   ```

4. Keep the selected raw vector. Do not average, normalize, synthesize, or
   rescale it.
5. Prefer additional distinct source vectors when alias capacity is short.
6. If the source itself is too short, duplicate its highest-norm vector.

Leaf cards select from the node's raw layer-0 K. Digest and era cards select
from their fixed child cards so a hierarchy cannot reintroduce a ragged bank.

Publication creates a dense CUDA shape
`[node, kv_head, slot, head_dim]`. CPU routing consumes the identical vectors.

## Deferred Hidden Librarian

The visible response path may create the provisional card and enqueue a job.
It performs no alias-generation model work. There is no default background GPU
thread.

Explicit control:

```python
repo.bookkeeping_tick(max_jobs=1, deadline_ms=250, cancel_event=None)
repo.route_card_status(node_id)
repo.inspect_route_card(node_id, include_aliases=False)
```

The tick uses the serving model with a local scratch cache. It does not assign
that cache to `arena.caches`, append generated text to the conversation, alter
live mounts, or create a memory node. The generator receives source text only
as a quoted JSON string inside a JSON-only contract.

The requested output is an array of at most eight records containing exactly:

```json
{"query": "...", "answer_span": "...", "evidence_quote": "..."}
```

Acceptance is mechanical:

- all three fields are non-empty strings;
- normalized questions are unique;
- `evidence_quote` is an exact, case-sensitive source substring;
- `answer_span` occurs inside that evidence;
- every number, code, and rare identifier in the query/answer already occurs
  in the source under the repository's lexical rule.

Accepted question strings are harvested through the normal layer-0 GQA K
capture using scratch state. Half the card slots remain source medoids and half
become alias medoids. If alias K has too few distinct token positions, source
medoids fill the deficit before highest-norm duplication.

Aliases are untrusted routing metadata. No code path interprets them as chat,
tool calls, memory commands, supersession instructions, or mounted content.
`inspect_route_card()` hides alias text unless `include_aliases=True` is passed.

Cancellation is checked before generation, at each hidden decode-token
boundary, before/after alias harvesting, and immediately before publication.
An interruption stays `pending` and does not consume an attempt. Ordinary
generation/validation/harvest failures increment attempts. The third failure
sets `failed`, retains the original provisional keys, and persists receipts.

## Lifecycle

- Deposit: publish a provisional fixed card synchronously, then enqueue.
- Pending: provisional keys remain searchable while hidden work is queued.
- Ready: atomically replace provisional rows with the hybrid card.
- Failure: retain provisional rows; terminal after three attempts.
- Supersession/expiry: deactivate the old node before card publication and
  retain its card as `stale` audit data; the replacement gets a provisional
  card.
- Text, payload, profile, prompt-policy, or lineage mismatch invalidates the
  prior card. The implementation records source, payload, profile, lineage,
  and policy hashes and replaces invalid rows from current source keys.
- `recall`: clear the card and routing rows.
- Inactive/retired: preserve audit data but exclude via the native active bit.
- Digest/era: derive provisional keys from fixed child cards.
- Restart: restore fp32 keys, state, aliases, receipts, timestamps, and policy
  hashes from the native checkpoint before rebuilding the CUDA view.

## Configuration

```python
GraftRepository(
    ...,
    gqa_route_cards=True,
    route_card_slots=<development winner>,
    route_card_aliases=8,
    route_card_prompt_version="gqa-route-card-v1",
)
```

Opt-in requires a native library exposing the route-card ABI. The ordinary
constructor remains disabled and behaviorally unchanged.

## Development Gate

Development-visible sources:

- existing Qwen3.5 K/Q capture shards;
- the GPT-OSS composed-session repository, scorecard, and restart receipts;
- the existing GQA CPU/CUDA bridge receipts.

Sweep `{2, 4, 8}` once. Choose the smallest size satisfying every item:

- source-node recall@3 no worse than full ragged-key routing;
- all seven GPT-OSS fresh-fact sources remain recallable;
- both supersession probes put the current source in top three and never rank
  its inactive source above it;
- hybrid cards are no worse than source-only cards;
- no accepted alias contains an unsupported identifier;
- native CPU and dense CUDA return the same stable top-k outside exact ties.

If any required replay input is absent, the gate is not silently weakened. It
stays incomplete and fresh evidence remains closed. If all three sizes fail,
stop the work order before opening a fresh seed.

### Registered hybrid continuation (2026-07-11)

The first source-only replay did not test the central generated-query claim.
The operator authorized continuing that line, so one development-visible
hybrid replay is registered before any additional result is observed:

- reuse the exact 32 target capture paths, 20 source-query locations, layer 3,
  and seed recorded by `qwen35_slot_sweep.json`;
- reconstruct each source chunk only from its stored target `token_ids` using
  the matching local Qwen3.5-9B tokenizer;
- use the same local Qwen3.5-9B serving model, greedy decode, prompt version
  `gqa-route-card-v1`, alias limit eight, and the already-implemented exact
  evidence/answer/identifier validator;
- harvest accepted alias K from Qwen3.5-9B layer 3, the same native space as
  the stored target K/Q receipts;
- build each `{2,4,8}` hybrid card with the already-frozen half-source,
  half-alias policy and selector; a node with no accepted alias retains its
  source-only provisional card;
- score the same 20 stored source queries against full ragged, provisional,
  and hybrid banks. No new query or slot-size result is used to alter the
  prompt, selector, split, or acceptance law.

A two-node protocol qualification may run first. It tests only model loading,
JSON parseability, mechanical validation, and layer-3 K shape; it does not
score retrieval or expose a slot-size result. The full replay opens only if
both qualification nodes yield at least one accepted alias and valid native K.

The line survives this continuation only if at least one registered size is no
worse than full ragged recall@3 and is no worse than its provisional card,
with zero unsupported identifiers in accepted aliases. Otherwise the
fixed-card hypothesis stops; it is not rescued by another prompt or selector
tune. GPT replay and both fresh seeds remain closed regardless until their
registered inputs and every other development gate are available.

## Fresh Dual GQA Gate

Only after the development freeze:

### Qwen3.5

- new deterministic 512-node, 100-query corpus;
- seed `20260713`;
- CPU/CUDA non-tie parity: 100%;
- CUDA engaged for every card-ready semantic route;
- resident-bank latency at 512 nodes: p50 no more than 5 ms, p95 no more than
  8 ms.

### GPT-OSS-20B

- new 40-turn session;
- ten fresh facts and two supersessions;
- eviction plus checkpoint/restart;
- seed `20260714`;
- current source top three for 12/12 probes;
- answers: 10/10 fresh and 2/2 current supersessions;
- stale values never emitted;
- restart preserves card bytes, audit data, state, and ranking.

Shared systems gates:

- enqueue overhead no more than 5 ms;
- cancellation acknowledged at one hidden token boundary, with a 250 ms wall
  scheduling rail;
- shared-state A/B proves no change to the already-visible answer or live cache;
- no monotonic VRAM growth after steady state;
- sampled residency and allocator peaks reported separately.

A quality pass above the latency rail is `QUALITY-GREEN / COMPUTE-RED`. Any
fresh recall, supersession, audit, cache-isolation, or parity failure stops
native rollout.

## Verification Matrix

Required before adoption:

- schema and C/Python binding tests;
- v10-to-v11 GQA migration and v10 MLA byte-compatibility tests;
- deterministic selector/property tests including short sources and ties;
- lifecycle, lineage, stale-hash, supersession, inactive, recall, digest, and
  era tests;
- exact-evidence, duplicate, unsupported-identifier, malformed JSON, and prompt
  injection tests;
- cancellation at generation, harvesting, and pre-publication boundaries;
- route/update snapshot concurrency plus TSAN;
- CPU/CUDA card parity;
- existing repository, runtime lifecycle, librarian, GQA arena, CUDA bridge,
  and GPT-OSS session suites.

After a passing baseline, a lead who did not implement the mutation runs the
blind adversarial verification. Registered mutations include missing source
hash invalidation, unsupported identifier acceptance, route-card mounting,
missing epoch publication, stale supersession cards, live-cache contamination,
and checkpoint audit loss. Kill threshold is at least 80%.

## Stop Rail

This work order may land an opt-in implementation and still stop before
adoption. No partial development result authorizes fresh-seed tuning or default
enablement. The dual fresh gate, blind verification, and mutation threshold are
all adoption prerequisites.
