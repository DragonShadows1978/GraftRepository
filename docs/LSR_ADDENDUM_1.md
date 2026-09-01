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
