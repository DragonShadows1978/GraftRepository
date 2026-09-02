# LSR — Addendum 1 (2026-09-01): Phase 0 verdicts; Phase 1 re-registered

Phase 0 receipts (adjudication 49e1e38b, registered-before-any-run,
plan hash-bound):
- **H-LSR-1 NOT CONVICTED, 0/6** — premise empty: the recency
  clean-room zeroes recent-turns on point-lookup probes
  (grm_e2e_session:1146); no recent tokens exist to leak.
- Measured value provenance: ARENA ×2 (lumen, orion — wrong lineage
  node mounted, model reads it faithfully), ABSENT ×4 (value nowhere
  in window — confabulation under retrieval failure). Controls
  ARENA (correct). One probe class-switched between serves
  (instability of the failing class).
- Survey constant, on disk: **6/6 failing probes never mounted the
  node containing the expected value.**
- H-LSR-2 (route bleed): no evidence — mount receipts hold; nothing
  foreign in windows. H-LSR-3 (confabulation): supported as the
  DOWNSTREAM behavior when retrieval fails, not the root.

## Phase 1 (re-registered): why does routing mount the wrong lineage end?

Question: on the six probes, reconstruct the full routing decision —
route score table over the fixture's nodes, identifier hits, L2 M5-edge
resolution, A-DEC branch — and adjudicate per probe which stage
discarded the expected-value node:
- P1-A: route scores rank the wrong node above the expected one
  (scoring defect / key quality);
- P1-B: route ranks expected first but L2 resolution follows the M5
  edge to the wrong end (resolution defect or fixture edge direction);
- P1-C: A-DEC decisive branch prunes the expected node the route
  would have admitted at k=3 (admission interaction — note ADM1.3's
  F-PROD showed all-green on THESE fixtures' battery frame; reconcile).
Registered vocabulary: each probe gets exactly one of
ROUTE-RANK / LINEAGE-RESOLUTION / ADMISSION-PRUNE / UNRESOLVED, with
the stage's own receipts cited. Mostly CPU from existing lived
receipts + fixture manifests; fresh GPU serves only if a decision
input was not persisted (state which).

Fix design enters only after the stage verdicts — same discipline as
Phase 0.

## Phase 1 FINAL (2026-09-01 evening; adjudication 31e5c894, reconciliation 9c2f16a7)

Verdicts: **ADMISSION-PRUNE ×4** (repos complete; A-DEC planned the
right node; arena fit() at width 96 silently dropped it — long
competitor nodes unseatable by construction) and **NOT-YET-DEPOSITED
×2** (addendum term, accepted: the expected node was absent from the
repository at the lived probe turn — deposit ordering; no stage could
select it; registered-vocabulary fallback ROUTE-RANK recorded per
probe). LINEAGE-RESOLUTION exonerated structurally; key derivation
install-vs-deposit exonerated bit-for-bit on GQA (routing-index digest
match; MLA contextualized-centroid hazard remains untested and
unclaimed). Harbor control: value-duplication mask confirmed.

Principles earned: (1) rebuild-based instruments are blind to
mid-session repository states — evidence timing, not key fidelity, is
what a rebuild cannot see (the fork-from-snapshot principle at the
repository layer); (2) route receipts are not persisted at serving
time — an observability gap (candidate_count/fallbacks per turn belong
in the step-3 memory ledger).

## Phase 2 fix menu (David decisions; no fixes authored yet)

1. **Fit-stage honesty (the big one):** when fit() cannot seat the
   planned node, fail loud / trigger shuttle-trip / degrade with an
   explicit flag — never silently substitute a lower-ranked node.
   (Mount-curation design Stage B machinery; D-NGH can witness.)
2. **Not-in-memory honesty:** when the routed best is a weak match
   because the answer was never deposited, the abstention/demand-loop
   behavior applies — serve "not in memory" rather than the nearest
   wrong thing. (Stage C design; ties to GRAPA abstention curriculum.)
3. **Route-receipt persistence** in the per-turn memory ledger
   (cheap observability fix).

## Phase 2 (2026-09-01 evening/night; David's ruling: all three, SHUTTLE over fail-loud)

Orders: GRM_LSR_P2A (fit honesty + abstention), P2B (route-receipt
persistence, worktree), P2C (unseatable split/descent). Commits 60635c9
(P2A), 6a552f4→39c1be1 (P2B + merge), 2001ada (P2C). All Opus 5 seats.

- **P2A**: plan-order seating (a planned node is never displaced by
  filler), fit-drop-triggered shuttle (cap `len(rank_plan)`, registered),
  explicit UNSEATABLE degrade (`fit_unseatable`,
  `served_without_plan_head`), structural not-in-memory abstention
  (identifier tokens bind no node → constant served, deposited as
  `kind="recall"` never as fact; no threshold). G3 abstention live GREEN.
  FINDING: the 4 ADMISSION-PRUNE answer nodes (623–706 chars) exceed
  the 96-seat arena ALONE; shuttle has nothing to serialize. P2A's lived
  replay BLOCKED: DET1.3 fork snapshots capture mounted payload at a
  boundary AFTER admission; rebuilds flip A-DEC branches (principle 1
  reproduced). The `[X,Y,Z]`-from-`rank_plan=[X]` widening site:
  `scripts/grm_probe_ladder.py:115` (topk rung merged behind the plan
  rung by the drivers).
- **P2B**: `grm.route_receipt.v1` attached opt-in to the step-3 memory
  ledger (frozen mutation schema untouched); dump script reproduces all
  14 Phase-1 evidence fields on 8/8 probes from persisted receipts.
  Merge seat caught a phantom-rung receipt on abstained turns
  (`trips None` vs `[]`) and fixed it.
- **P2C**: width guard at every repository deposit path (chunk at
  sentence boundaries via cull_graft, one-time repair); fit-time split
  + identifier-child descent using the frozen ADM1
  `is_identifier_binding` predicate (the rare channel is EMPTY for the
  whole probe class); chunk shuttle; `split_oversized` sweep. Fixed a
  pre-existing packed-payload slice corruption in `cull_graft`. G2
  lived-order replay: Arm 0 (fixes OFF) reproduced 9/9 lived values and
  the Phase-1 admission receipts exactly; Arm 1 = 4 flips, 5 unchanged,
  0 regressions, 0 abstentions, `fit_unseatable` empty 9/9. G3 (e2e
  census turns) BLOCKED: Arm 0 8/10, the two late-turn divergences are
  fork-ladder-vs-in-session mechanism; Arm 1 withheld per rule.

**Verdict: the wrong-value class is CLOSED for the supersession battery
(4/4 ADMISSION-PRUNE flipped under lived-order reproduction; the 2
NOT-YET-DEPOSITED are harness deposit-ordering artifacts and correctly
unchanged).**

Principles earned: (3) the repository never holds a node the arena
cannot mount — prevention at deposit, same shape as co-mount; (4) a
reproduction arm with the fixes OFF must reproduce the lived receipts
before the fixes-ON arm counts (it caught a non-equivalent rebuild in
P2A and a non-equivalent session replay in P2C-G3); (5) the rare-token
channel is not an identifier channel — descent filters must ask the
same binding predicate admission asks.

Successors (open): greedy-coarse chunker (child can take 96/96 seats;
leaf-bias / fill-fraction, no knob registered); A-DEC insurance branch
ranks a long competitor ahead of the answer on sup_solace (harmless
now, plan still wrong); in-session fork harness so e2e turns get a
lived-equivalent replay (G3); D-NGH Stage C wiring; branch housekeeping.
