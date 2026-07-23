# GRM Revision-Aware Resolver Plan

Status: stopped at the development model gate on 2026-07-12. The feature is
opt-in and the repository default remains unchanged. The deterministic and
native resolver gates passed, but the exact authoritative mount did not yield
an exact GPT-OSS readback, so fresh evidence is forbidden by this plan's stop
rule.

Tracking artifacts:

- Operational ledger: `docs/GRM_REVISION_AWARE_RESOLVER_LEDGER.md`
- Narrative synthesis: `docs/GRM_REVISION_AWARE_RESOLVER_SYNTHESIS.md`

Branch: `codex/grm-revision-aware-resolver`

Worktree: `/home/vader/GraftRepository-revision-aware-resolver`

Base: `8e11bbd9107436720b51db5634746a4c2360c018`, the exact-ragged GQA
development winner. Local `main` remains behind that commit at work-order
opening, so this successor deliberately preserves the proven router branch.

## Decision

Do not spend this work order making the leaf semantic scan faster. At the
registered 512-node point it already routes in roughly 1.6/2.0 ms p50/p95.
Add a small, native revision-aware resolver after semantic ranking and before
mount fitting.

The resolver recognizes an exact, unambiguous fact family and pins its current
active fact into the first mount attempt. It does not answer the question,
alter the stored payload, generate aliases, or replace semantic routing.

## Evidence That Opens The Work

The composed GPT-OSS session recovered 7/7 fresh facts and never emitted an
inactive stale value, but two probes for the current `orion pin` revision
failed under competition. At the sharpest failure, the authoritative fact node
was semantic rank 3; a prior probe turn and its wrong competitor occupied the
mount plan ahead of it. The current node was mounted alongside a competitor and
the model emitted the competitor's value.

This is not stale-readback and not a missing CUDA kernel. It is a selection
policy defect: derived conversational turns can beat the authoritative active
fact even when the query names that fact's family exactly.

## House Rules

- Extend the existing `HostGraftStore`, native C ABI, `NativeGraftStore`, and
  `GraftRepository`; do not create a service or a second memory database.
- Raw turns remain evidence. Durable fact identity and revision lineage remain
  authoritative metadata.
- Inactive, expired, retired, or superseded nodes can never be pinned.
- Resolution is fail-closed. Missing identity, partial label overlap,
  conflicting active leaves, Unicode/ABI uncertainty, or unsupported temporal
  policy returns `no_match`/`ambiguous` and leaves semantic order unchanged.
- Derived family identity must be source-grounded. It may come only from:
  explicit `subject`/`predicate` metadata; one unambiguous inherited family;
  or ordered non-glue tokens present in both the correction target text and
  replacement text. It cannot come from model generation.
- Family identity is routing metadata. It cannot create a fact, invoke tools,
  execute memory commands, change truth, or become mounted content.
- The mounted payload remains the original graft. The resolver never mounts
  metadata or synthesized text.
- Preserve semantic stable order for every unpinned node. Preserve filters,
  exclusions, precise identifiers, lineage, and dialect behavior.
- Python owns Unicode and temporal policy. Only normalized ASCII family/query
  keys cross the native ABI; uncertain cases use the Python mirror.
- Keep the feature disabled by default. Default enablement is a separate
  operator decision after fresh dual-dialect evidence.
- Stop on a stale pin, ambiguous pin, semantic-order regression, restart loss,
  or current-revision miss. Latency misses may be reported as
  `QUALITY-GREEN / COMPUTE-RED`; correctness misses stop the line.

## Resolver Record And Interfaces

Every resolution produces an audit record:

```text
state: disabled | no_match | exact | ambiguous | failed
backend: native | python
query_family_keys
matched_subject / predicate / scope
pinned_node_ids
candidate_node_ids
inactive_rejected
reason
```

Native/Python surfaces:

```text
grm_store_resolve_fact_query
NativeGraftStore.resolve_fact_query(...)

GraftRepository(
    ...,
    revision_aware_resolver=True,
)

repo.inspect_last_route_resolution()
```

`include` candidate IDs are passed explicitly so live-window exclusions,
recency mounts, policy filters, and route eligibility remain owned by the
normal route path.

## Exact Resolution Law

1. Build the existing ordered query-lex token set; generic prompt/fact glue is
   excluded by the current query-lex law.
2. Consider active `kind=fact` nodes in the current candidate set with a
   non-empty normalized subject and predicate.
3. A family is addressable only when every normalized subject token occurs in
   the query token set. Partial subject overlap is not enough.
4. Group matches by normalized `(subject, predicate, scope)`.
5. Within a group, follow revision state and retain active leaves only. A
   unique active leaf is exact. Zero leaves is no match. Multiple leaves is
   ambiguous and pins nothing.
6. Across groups, prefer the most-specific subject (largest token count). Equal
   specificity across different families is ambiguous.
7. Insert the exact node at rank 1, remove any duplicate occurrence, and keep
   all other semantic candidates in their original stable order.
8. The first mount attempt contains the exact fact alone. Wider semantic trips
   remain available if grounding fails.

The resolver does not score values and cannot use an answer value absent from
the query as an address key.

## P0 — Baseline And Frozen Gates

Before product code:

- Preserve the composed-session 7/9 receipt and its two `orion pin` misses.
- Add a deterministic competition harness reproducing the same structure:
  inactive old fact, active corrected fact, higher semantic derived turn, and
  higher semantic unrelated fact.
- Record baseline current-fact rank, mount plan, semantic order, and backend.

Development correctness gates:

- Current active fact is rank 1 and the first precise mount for 12/12
  supersession-under-competition probes.
- Inactive/stale nodes are pinned 0 times and never rank above the current
  family node.
- Fresh fact routing outside exact family matches is byte-for-byte unchanged.
- Ambiguous active families pin nothing and preserve semantic order.
- Native and Python resolver results/receipts are identical for ASCII inputs.
- Unicode family identity stays on the Python path with identical policy.
- Checkpoint/restart and WAL recovery preserve identity, active state,
  resolution, and ranking.
- Existing exact-ragged GQA CUDA routing remains engaged for eligible semantic
  routes; resolution is a post-route policy and does not rebuild the bank.

Performance gates:

- Native resolver p95 overhead <= 0.25 ms at 512 active facts.
- Total resident semantic-route-plus-resolution p95 remains <= 8 ms at the
  registered 512-node GQA point.
- Correction enqueue/identity derivation overhead <= 5 ms excluding model
  harvest and durability I/O.
- No monotonic host or device residency growth after steady state; sampled
  residency and allocator peaks are reported separately.

## P1 — Grounded Family Identity

Thread explicit fact identity through correction paths. When correction
metadata omits identity:

1. inherit one common explicit identity from the superseded targets, else
2. derive ordered family tokens present in both the correction query and the
   replacement text after the existing query-lex stop/glue law.

If the intersection is empty or ambiguous, write the correction exactly as
today with no resolvable family identity. Record the derivation source in
metadata for audit. Do not infer arbitrary predicates; the deterministic
correction grammar uses predicate `value` only when the shared label is proven.

## P2 — Native Fact-Family Resolver

Add a first-class native subject-token index over existing fact identity. It is
updated by fact-identity mutation, active-state mutation, revision, expiry, and
checkpoint load. Resolution filters the index by the caller's candidate IDs,
then applies the exact law above.

The C ABI transports normalized string blobs and stable integer IDs only.
Python mirrors the same algorithm for native-unavailable and Unicode cases.

## P3 — Route And Mount Integration

Install an optional repository-owned resolver callback on `ArenaCache`:

- semantic routing executes normally;
- exact resolution reorders the returned mount window only;
- exact resolution bypasses the expensive generic content-word rescore when
  the native result and resolver receipt fully determine the point lookup;
- `last_route_backend` continues to report the semantic backend, while the
  resolver backend/state is separately auditable;
- `step()` uses a unique exact pin as its first single-node attempt.

## P4 — Verification

Add native ABI, Python mirror, property, lifecycle, restart, WAL, filter,
Unicode, temporal, ambiguity, stable-order, prompt-injection, and concurrency
tests. Run the existing repository, runtime lifecycle, native runtime, router,
GQA CUDA, MLA CUDA, librarian, and composed-session selectors affected by the
change.

## P5 — Governed Model Gates

Development replay uses the already-visible GPT-OSS composed-session artifact
and deterministic native/Python competition corpus. No model knob or old
receipt is retuned.

Only after development passes, freeze fresh seeds and run:

- GPT-OSS-20B: a new 40-turn session with ten fresh facts, two supersessions,
  eviction, checkpoint/restart, and twelve probes.
- Qwen GQA: a new deterministic 512-node identity/semantic competition corpus
  plus the exact-ragged CUDA route bank.

Required model results:

- Current fact source rank 1 and first mount for 12/12 probes.
- Answers: 10/10 fresh facts and 2/2 current supersessions; stale values never
  emitted.
- Restart preserves resolver receipt, identity, active state, and ranking.
- No visible-response, live-cache, or mounted-payload change when resolution is
  disabled or returns no match.
- All registered latency, parity, and residency rails remain green.

If development or fresh correctness fails, stop and preserve the failure. Do
not retune the same fresh seeds.

## Adoption Boundary

Passing this work order proves an opt-in resolver. Default enablement, general
natural-language fact extraction, value-constrained readout, and multi-family
question planning are separate operator decisions.

## Stop Record

The implementation made the active replacement fact rank first and mounted it
alone after checkpoint/restart. GPT-OSS nevertheless answered `The current
orion value is Kestrel.` rather than the stored `Kestrel-9-Tango`. No stale or
competitor value was emitted, but exact current-revision readback is a hard
gate. The 40-turn GPT-OSS and fresh Qwen campaigns were therefore not run.

The negative result distinguishes repository authority from KV readability.
The resolved fact graft was a 15-token bare correction payload, while the
value-bearing update exchange was a 92-token Harmony-formatted turn. Exact
selection cannot make the compact bare graft carry the same instruction and
readout behavior as the completed exchange. Any successor must govern which
evidence-bearing payload represents the authoritative family, not merely
which metadata record wins routing.
