# ORDER GRM-SUP-L2-ON — operator decision: L2 mount resolution DEFAULT ON

OPERATOR DECISION (David, 2026-08-30, verbatim intent: "Flip that
switch ON, it being OFF is causing issues"). Scope: L2 ONLY — the
M5-edge mount resolution / lineage-head-only lever from SUP-WO1
(GRM_SUPERSESSION_LEDGER.md; the `--resolve` arm of G2-SUP). L1
length-debias REMAINS default off (its receipt is UNDECIDABLE on MLA;
untouched by this order).

WRITABLE TARGET `/mnt/ForgeRealm/GraftRepository`; standing rules
(no git, no subagents, no network; CPU self-runs authorized, GPU
scripts for the rest — GPU queues behind in-flight CMC/ADM work on
the same flock).

## Work (mirror the probe-ladder default-on precedent)

1. Flip L2 to DEFAULT ON in production, with an escape env
   (GRM_SUP_RESOLVE=0 or the existing flag's natural inverse —
   document the exact name) restoring legacy behavior exactly.
2. Acceptance (registered):
   - Supersession battery under new default: stale 0/5, wrong-fact
     ≤ baseline, fresh controls ≥ 1/2 and byte-identical to their
     L2-on receipts (g2_debias_resolve lineage — note we are
     enabling resolve WITHOUT debias; if no resolve-only baseline
     exists, the diag_resolve_only.jsonl receipts are the anchor).
   - Escape-path check: GRM_SUP_RESOLVE=0 (or equivalent) →
     byte-identical to current default transcripts (registered
     baseline hashes).
   - Shared suites green (34/34-class), full GRM suite.
   - Re-register default baselines for any transcript that L2
     legitimately changes (supersession-involved paths ONLY —
     enumerate them; anything else changing is a RED).
3. Ledger entry appended to GRM_SUPERSESSION_LEDGER.md recording the
   operator decision, date, and new baseline hashes (append-only).
4. Coordination note: CMC1.1 arms and ADM1 fixtures currently run
   against the old default; their harnesses pin their own frames, so
   no interference expected — verify their pinned-frame assumption
   and state it.

## Done

Exact flag/env names; acceptance results with numbers; enumerated
re-registered baselines; ledger entry text verbatim; files modified;
anything you could not do.
