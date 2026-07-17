# GRM S4 → Fold Order (PLAN, immutable) — campaign section 4

Successor named in GRM_S4_LEDGER.md (G2-S4 FAIL entry). Opened
2026-07-17. Lead: Fable. Seat: Terra (Sol escalation per §9).
Substrate: MiniCPM3 MLA, deferred librarian.

## Thesis

G2-S4's failure mechanism was recency confounding: zero-hit nodes
include the just-deposited. FOLDING has no such confound — the
librarian only ever selects fold sources already past the recency
window, where zero-hit genuinely means unused. Ordering fold-source
selection by grounding hits (fewest hits fold first; hit-rich turns
stay verbatim longest) should preserve recall better than the current
age-order under equal compression, at zero signal cost. Fold-ordering
was the original importance program's registered "safe consumer."

## Registered design

- Fold-source ordering key: (n_grounded ascending, then current age
  order) over the librarian's existing eligible set — eligibility
  itself unchanged (M11 payload guard, no_fold, recall-kind exclusion,
  live-window exclusion all untouched).
- Opt-in fold_order="s4" alongside the current default ("age");
  default byte-identical.
- Applies to BOTH deferred and backpressure paths (the two-path drift
  precedent).

## Registered gate (G-FOLD, thresholds before it runs)

A/B on identical sessions (fingerprint-checked), librarian active,
enough turns that ≥4 folds fire per arm (extend the 42-turn deferred
harness pattern with the repeat-probe fixtures' probe style):
- PASS = post-fold recall(s4) ≥ recall(age) on late probes AND
  fidelity-gate ABORT count(s4) ≤ aborts(age) AND hot-path add_turn
  max latency unchanged (deferred flatness preserved, 0.27s-class).
- Report per-fold composition (which turns folded, hit counts) —
  the mechanism receipt.
- Ties on all = pass-by-equivalence, free-but-not-yet-advantageous.

## Non-goals

Routing (unchanged law), paging (closed RED, s4_protect owns the
retry), fold PROMPT changes, threshold-count changes.
