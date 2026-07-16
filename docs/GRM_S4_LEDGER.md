# GRM S4 Grounding-Hit Ledger — LEDGER (receipts, append-only)

Plan: GRM_S4_GROUNDING_LEDGER_PLAN.md (immutable, b5428f9).

## 2026-07-16 — S4-WO1 landed (Sol), G0-S4 PASS, G1-S4 PASS

- WO1 (983610f): attribution split (pooled grounding verdict
  bit-parity — CPU-tested), funnel counters {n_routed, n_mounted,
  n_grounded, last_grounded_turn} with acceptance owned by step()
  (rolled-back attempts never count), 4 repeat-probe fixtures
  (32 probes, validator green), gate harness. 118/118 shared suites
  under lead GPU re-run (seat's 5 fails = CPU-only sandbox
  constructing MLAAttentionTC — same env artifact as the SUP seat,
  same correct in-report diagnosis).
- **G0-S4 PASS** (receipt schema grm_s4_grounding_ledger_g0_v1):
  source node n_grounded=2/last_grounded_turn=7; co-mounted decoy
  0 grounded despite 1 mount / 2 routed; rollback trip uncounted.
- **G1-S4 PASS — the strongest importance signal measured in this
  program family:** 16/16 late probes eligible+ranked, median
  Spearman 0.7556 (bar 0.5), top-1 agreement 0.875 (bar 0.5),
  against the same S3 teacher-forced arbiter and inherited bars as
  the predecessor (comparison row: S1 0.4733/0.50 FAIL,
  S2 0.2191/0.25 FAIL, S4 0.7556/0.875 PASS). Evidence class:
  rank agreement vs teacher-forced logit A/B arbiter, sealed
  artifacts (grm_s4_grounding_ledger_convo_v1).
- CAVEAT (registered honestly): secondary lift undefined —
  n_no_early_hit = 0; fixture design guarantees early hits on
  targets, so the "no early hit but probed late" condition is
  empty. Predictive validity rests on rank agreement; CONSEQUENCE
  rests on G2-S4 below. Fixtures where repeat-probing defines
  importance favor a usage signal by construction — the demotion
  A/B is what tests real payoff.

## 2026-07-16 — G2-S4 THRESHOLDS REGISTERED (before the gate runs)

- Session: the 4 repeat-probe convos driven as ONE repository
  session (or an equivalent long session), vram_budget set to force
  ≥3× overcommit (paging must actually fire; report page-in counts
  to prove it).
- Policy A = current LRU paging (last-mounted). Policy B = S4-aware
  demotion: zero-hit nodes spill first, LRU tiebreak within equal
  hit classes. Same budget, same session, same probes.
- PASS = policy B late-probe recall ≥ policy A AND page-ins(B) ≤
  page-ins(A). Exact ties on both = PASS-by-equivalence with the
  note that S4 demotion is then free but not yet advantageous at
  this scale.
- Non-goals unchanged: no routing integration, demotion only.
