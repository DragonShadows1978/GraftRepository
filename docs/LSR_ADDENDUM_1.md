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
