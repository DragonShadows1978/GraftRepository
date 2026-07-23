# GRM Revision-Aware Resolver Synthesis

## Opening Thesis

GRM's next bottleneck is no longer finding a semantically related graft fast.
MLA and GQA both have measured CUDA routes, and exact ragged GQA leaves route at
interactive latency without compressing away their signal. The remaining
composed-session failure is more semantic: the repository knows that one fact
supersedes another, yet query-time mount selection does not use that knowledge.

The failure is unusually clean. Fresh-fact recall works. Stale memory stays
inactive. The correct replacement exists and can rank in the semantic top 3.
But a value-bearing probe turn or unrelated fact can still consume the better
mount position, and free generation reads the wrong mounted value. More GEMV
optimization cannot repair that ordering.

## The Design Move

Treat routing as two distinct operations:

```text
model-native semantic route
        +
exact repository identity resolution
        ↓
auditable mount plan
```

Semantic routing answers, "what memories are related to this query?" The
revision resolver answers, "does the query exactly address one durable fact
family whose current active leaf is already known?" Only the second question
may pin a node, and only when its answer is unambiguous.

This keeps the architecture honest. The model's attention-state geometry
continues to provide broad addressability. The repository's explicit identity,
scope, active state, and revision graph enforce memory semantics. Neither is
asked to imitate the other.

## What This Is Not

- Not a route-card revival: no fixed K summary or generated query aliases.
- Not answer synthesis: the resolver returns node IDs, never values or prose.
- Not a universal natural-language parser: unsupported identity is a normal
  no-match.
- Not a stale-value suppression patch: stale nodes are already inactive; the
  defect is promotion of the current authoritative node.
- Not a new database: the native host store already owns the needed record.
- Not a kind bonus over every query: only exact family addressability may pin.

## Expected Consequence

If the work succeeds, supersession becomes an operational property rather than
only a storage property. An exact query for `orion pin` seats the one active
`orion pin/value/project` fact before derived turns, while unrelated and
ambiguous queries retain their byte-identical semantic ordering.

That is the first step toward a general GRM memory resolver: semantic recall
for breadth, explicit repository semantics for authority, and original graft
payloads for model-native reading.

## Result

The resolver worked as a resolver and failed as a complete memory fix.

On deterministic competition, the active fact moved from rank 3 to rank 1 and
the first mount in 12/12 cases. Native resolution was exact across 100/100
queries and restart, reached 0.133 ms p95 at 512 facts, and left the exact
ragged CUDA route green at 1.830 ms p95. Ambiguity, stale state, Unicode,
stable-order, lifecycle, and restart checks behaved as designed.

The real GPT-OSS smoke then isolated the unresolved layer. After checkpoint
restart, the resolver selected `orion pin/value/project`, promoted node 3 to
rank 1, and mounted node 3 alone. Its stored text contained the exact current
value `Kestrel-9-Tango`. GPT-OSS answered `The current orion value is
Kestrel.` No stale or competing value appeared, but the exact value did not
survive readout. That is a hard failure under the preregistered gate.

## Why It Failed

The design treated metadata authority and KV readability as if choosing the
former guaranteed the latter. They are separate properties.

The replacement fact was created by the normal correction path as a compact,
15-token bare sentence. The completed authoritative update was a separate,
92-token Harmony exchange containing the same value, an explicit later-answer
instruction, and an assistant acknowledgment. The resolver correctly chose
the durable fact record, but the mounted KV for that record did not induce an
exact code-shaped answer. Precise mounting removed distractors and still lost
the numeric/phonetic suffix.

This also explains why more resolver tuning is the wrong response. Subject
matching, specificity, revision state, rank, and mount isolation were already
correct on the failed probe. Changing those rules cannot add information or
serving-dialect behavior to the selected KV payload.

## Decision

Stop this branch before fresh evidence. Do not adopt or enable the resolver.
Preserve it as a useful negative result: GRM needs an authority-to-evidence
binding, not only an authority-to-node binding. A successor may resolve a fact
family to a grounded completed exchange or explicit source lineage, or encode
the authoritative fact in the serving prompt dialect. That successor must be
planned separately because it changes the mounted-evidence contract.
