# GRM S4 — Grounding-Hit Ledger (PLAN, immutable)

Successor to GRM_IMPORTANCE (predecessor closed RED at G1/G2, 262e434);
fresh registration per its ledger. Opened 2026-07-16. Lead: Fable.
Implementation seat: Sol (codex shim). Substrate: MiniCPM3 MLA arena.

## Thesis (David-approved direction, 2026-07-16)

The arena already computes, then discards, a retrieval-outcome signal
every turn: grounding. A per-node ledger of {routed, mounted,
grounded-hit} costs ZERO extra forward passes and is causally
downstream of counterfactual dependence (a grounding hit means the
answer's content tokens demonstrably came from that mount). Use it for
DEMOTION (hot→cold storage priority), never deletion, never routing.

## Registered metric definitions

- Per-node counters in metadata["importance"]["s4"]:
  {n_routed, n_mounted, n_grounded, last_grounded_turn}.
- GROUNDED-HIT attribution: within _grounded()'s existing coverage
  computation, a mounted node scores a hit iff the answer is grounded
  AND that node contributes ≥1 token to the answer's content-coverage
  set (identifier/caps tokens; substantive-word fallback for
  content-empty answers, mirroring grounding v3's ladder). Pure split
  of the existing pooled check — no new token machinery.
- Counters update ONLY on accepted attempts (rolled-back trips do not
  count; same epoch discipline as S1 telemetry).

## Gates

- **G0-S4 (attribution correctness):** CPU units on the coverage
  split; GPU smoke: grounded answer sourced from mount X → X
  increments and a co-mounted decoy does not; ungrounded/hedged answer
  → no increments; rolled-back trip → no increments. Zero-cost claim
  verified structurally (no new forwards on any path).
- **G1-S4 (predictive validity, repeat-probe):** new fixture arm —
  4 conversations, ≥4 facts each probed TWICE ≥8 turns apart, plus
  never-probed controls with graded relevance (schema of
  importance_convos). At the LATE probe, rank candidates by S4 count
  accrued through the EARLY probe; arbiter = S3 counterfactual
  (imported unchanged from the predecessor). INHERITED BARS (for
  comparability with S1/S2): median Spearman ≥ 0.5 AND top-1
  agreement ≥ 50%, same eligibility cutoff discipline (2× the G0b
  floor of 1.881, re-measured if the fixture regime differs).
  SECONDARY (registered, reported, no gate): lift = P(late hit |
  early hit) / P(late hit | no early hit).
- **G2-S4 (consumer, demotion A/B):** the payoff gate. Long session
  under a VRAM budget that forces paging; policy A = current LRU,
  policy B = S4-aware (zero-hit nodes demote first, LRU tiebreak).
  Win = late-probe recall ≥ LRU with page-ins ≤ LRU. Thresholds on
  the margins registered after G0-S4 lands, before G2-S4 runs.

## Non-goals (unchanged laws)

- S4 never enters routing scores (the attractor law stands).
- Demotion only — no deletion; cold storage is free (INT8/INT6).
- No changes to grounding SEMANTICS — the split must be a pure
  refinement: pooled verdict bit-identical before/after.

## Evidence classes

Unit test / GPU smoke / rank-agreement vs teacher-forced A/B arbiter /
recall gate under paging pressure. No claiming G2-S4 value from G1-S4
agreement.
