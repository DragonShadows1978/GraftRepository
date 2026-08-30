# ORDER GRM-CMC1 — co-mount confusion: mechanism hunt (order-swap / attention witness / outlier clipping / identifier channel)

## Grant

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` — edits + CPU
runs AUTHORIZED; no GPU in your sandbox: build + self-test + emit run
scripts, lead runs GPU. READ-ONLY: model snapshots (HF cache),
Project-Tensor. All existing artifacts append-only. New code
`scripts/grm_cmc1_*.py` (import runtime/harness freely; production
behavior byte-identical with every new flag off). Artifacts
`artifacts/grm_cmc1/`. FORBIDDEN: git, subagents, network.

## Context (one paragraph)

Co-mount confusion is GRM's top open defect class: routing and mounting
select the RIGHT graft, yet readout returns a co-mounted sibling
(supersession ledger: fresh control 1/2; DIAG arc: turn-5 read flips
with ALL route invariants identical, and an INT8 pack→evict→rehydrate
round-trip REPAIRS it, 8/9 — supported, never convicted). Lead
mechanism hypotheses, registered: **H-INNER-CHANNEL** — readout is
token-level QK similarity with none of the outer router's channels
(no identifier bonus), so topical near-ties among co-mounted siblings
split softmax mass; **H-OUTLIER** — the near-tie is broken by
key-norm outliers (massive-activation class, cf. APAMQ): a few
freak-norm sibling keys capture the query regardless of content, and
the rehydrate "repair" is INT8 accidentally clipping exactly those
outliers; **H-POSITION** — arena adjacency/order decides the tie
(rotary proximity / nearest-pattern copying). More than one may hold.

## Fixtures (registered; reproduce-first gate)

Use the EXISTING repro cases: the supersession-ledger co-mount fresh
control and the DIAG turn-5 pattern (orion↔cypher class) on their
existing model/runtime fixtures. CMC-G0: each fixture must REPRODUCE
the wrong read under current production defaults before any arm runs
(else STOP and report — a healed fixture is a finding).

## Arms

- **T1 order-swap:** permute the arena mount order of target and
  sibling(s) (all orderings for ≤3 mounts). Record winner vs position
  vs identity per ordering.
- **T2 attention witness (the named missing instrument, built the
  robust way):** capture the query vectors at the answer-readout
  positions and ALL mounted K from the arena (GRM owns this state —
  no kernel materialization needed), recompute softmax attention
  offline in fp32, per head per layer. Report: (a) captured-mass
  split target-vs-sibling at the decisive heads; (b) mass
  concentration — what fraction of the sibling's captured mass rides
  keys with norms above its own p99; (c) per-graft key-norm
  distributions (p50/p99/max).
- **T3 selective clipping:** at mount time, clip ONLY the sibling's
  top key norms (grid: clip keys above p99 / p99.9 to that
  percentile; target graft untouched). Does the read flip as the
  full INT8 rehydrate does? Include a full-rehydrate arm same-run
  for direct comparison.
- **T4 identifier channel at readout:** additive logit bonus δ
  (grid: small values) on mounted-arena tokens exactly matching the
  query's identifier tokens (reuse the outer router's identifier
  extraction), behind a default-off flag. Does the read resolve at
  small δ? Byte-identity with flag off is a gate.

## Registered adjudication

- H-OUTLIER CONVICTED iff T2(b) shows the sibling's captured mass
  concentrated on its outlier keys (≥50% of captured mass on keys
  above its p99 norm at the decisive heads) AND T3 outlier-only
  clipping flips the read on ≥ half the repro cases.
- H-POSITION CONVICTED iff T1 winner follows position, not identity,
  on ≥ half the orderings.
- H-INNER-CHANNEL SUPPORTED iff T4 resolves the read at small δ
  without regressing the fixture's other probes.
- Verdicts are independent; report each with its receipt. If T2/T3
  disagree (mass concentrated but clipping does not flip), say so —
  that is the most informative outcome, do not harmonize it.

## Gates

- CMC-G0 reproduce-first (above).
- CMC-G1 byte-identity: all new flags off → production transcripts
  byte-identical (registered baseline hashes).
- CMC-G2 witness truth: offline fp32 recompute validated against a
  directly-materialized softmax on ONE short standard-path case
  (tolerance stated).
- CMC-G3 report `artifacts/grm_cmc1/GRM_CMC1_REPORT.md` with the
  three verdicts in the registered vocabulary + the rehydrate-mystery
  disposition (closed-as-corollary / still-open).

## Constraints

Standing laws: GPU discipline in emitted scripts (flock, ≤590 s
leases, 30 s gaps, single GPU), bf16 fidelity on any HF-side work,
content-addressed provenance, fail-loud non-finite writer, RED
honesty, no post-hoc widening. If a fixture session exceeds
known-valid instrument lengths for its model, note it and rely on
task-based reads (the GLC law) — the witness recompute itself is
exact math at any length.

## Done

Final message verbatim: CMC-G0..G3 statuses; the three H-verdicts
with key numbers; the T2 mass-split and concentration table for the
decisive heads; T3 grid outcomes vs full rehydrate; T4 δ at
resolution (if any); rehydrate-mystery disposition; files
created/modified; anything you could not do.
