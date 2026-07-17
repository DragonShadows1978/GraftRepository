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

## 2026-07-16 — G2-S4 VERDICT: FAIL (registered gate; result stands)

- Receipts: artifacts/grm_s4_demotion/{grm_s4_demotion_lru.json,
  grm_s4_demotion_s4.json, grm_s4_demotion_verdict.json}; overcommit
  4.595× (≥3× prerequisite met), paging fired both arms, same session
  fingerprint both arms.
- Late-probe recall: LRU 14/16, S4 14/16 — non-inferiority met.
- Page-ins: LRU 68, S4 112 (+65%) — FAIL on the second condition.
- MECHANISM (interpretation, receipt-consistent): early in a session
  the zero-hit class includes the just-deposited nodes routing is
  about to want again; zero-hit-first spilling breaks recency
  locality and thrashes exactly where LRU stays warm. The S4 signal
  ranks importance (G1 GREEN, 0.756/0.875); as a PAGING KEY it lags
  usage until hits accumulate. Signal ≠ policy; LRU is hard to beat.
- PROGRAM VERDICT: S4 grounding-hit ledger = VALIDATED IMPORTANCE
  SIGNAL (G0+G1 green, zero forward-pass cost) with NO consumer win
  yet. spill_policy="s4" stays in-tree, flagged, default LRU.
- Successor space (fresh registration each, no retunes): hit-
  protected LRU hybrid (recency primary, hits as spill-protection
  only); longer-horizon sessions where hit statistics are dense
  before pressure; hits as DISK-tier demotion (cold→archive) where
  recency is meaningless; hits feeding fold ORDER instead of paging.

## 2026-07-17 — s4_protect REGISTERED (successor #5, before any gate)

- Policy: spill_policy="s4_protect" — LRU stays PRIMARY (recency
  ordering untouched); nodes with n_grounded > 0 are PROTECTED from
  spill while any unprotected candidate remains; protection yields
  (pure LRU among protected) only when every resident node is
  protected. The inverse of the failed zero-hit-first design: hits
  shield, never accelerate eviction; the young-node confound is
  structurally absent (young zero-hit nodes get default LRU
  treatment, not punishment).
- Gate: same A/B form and prerequisites as G2-S4 (4-convo combined
  session, ≥3× overcommit, paging fired, same fingerprint).
  REGISTERED PASS = recall(s4_protect) ≥ recall(LRU) AND
  page-ins(s4_protect) ≤ page-ins(LRU). Ties-on-both = pass-by-
  equivalence with the free-but-not-yet-advantageous note.

## 2026-07-17 — s4_protect VERDICT: FAIL — and an equivalence law

- Receipts: grm_s4_demotion_s4_protect.json vs the fingerprint-matched
  LRU arm. Recall tied 14/16; page-ins 112 vs 68 — FAIL on the
  registered second condition, identical numbers to the zero-hit-first
  policy.
- BYTE-IDENTICAL RECEIPTS vs the s4 arm (every probe record, every
  phase split). Plumbing verified engaged at source (_page dispatches
  to _s4_protect_spill_order; distinct code, CPU-tested distinct where
  nonzero hit counts differ). Verdict: BEHAVIORAL EQUIVALENCE on this
  workload — with binary-sparse hits, protect-the-grounded and
  spill-zero-hit-first select the same victims in the same order;
  they diverge only when spilling among nodes with DIFFERING nonzero
  hit counts, which this session never forces.
- LAW (registered): EVICTION IS ZERO-SUM. Under a fixed budget,
  protection is victim-relabeling: shielding hit-nodes IS evicting
  the unhit, and early-session unhit = young. The s4_protect
  registration's premise ("young zero-hit nodes get default LRU
  treatment, not punishment") was arithmetically impossible — there
  is no paging policy that both honors hits and spares the young when
  the young are the zero-hit mass. S4-as-paging-signal is now closed
  from BOTH directions (punish-zero-hit: G2-S4 FAIL; protect-nonzero:
  equivalent, FAIL). Remaining live consumer bets: fold order
  (G-FOLD, in flight) and disk-tier demotion (registered successor;
  recency-free regime).
