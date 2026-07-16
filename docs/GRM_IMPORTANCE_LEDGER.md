# GRM Importance Weighting — LEDGER (receipts, append-only)

Plan: GRM_IMPORTANCE_PLAN.md (immutable after initial commit).

## 2026-07-16 — Program opened

- Recon (lead): `w = tc.causal_softmax(s)` explicit in MLA absorbed
  decode (`core/minicpm3_tc.py:200`) → S1 tap is a read-only hook.
  Deferred librarian idle slot confirmed live in
  `core/graft_repository.py` (librarian_mode, `_librarian_jobs`,
  backpressure path) → S2 has a home.
- WO-1..WO-4 dispatched to Sonnet seats, parallel, strict file
  ownership per plan. GPU gate runs reserved to lead, serialized.

## 2026-07-16 — WO-2 (S3 harness) LANDED, lead-verified at CPU level

- tests/test_grm_importance_counterfactual.py: metrics + minus-one
  pick logic + 15 CPU unit tests (15/15 under lead re-run) + G0b gate
  behind --run-gpu (lazy imports verified; plain import touches no
  GPU). Evidence class: unit test.
- LEAD SIGN-OFF on two metric details the plan left unspecified,
  registered here BEFORE any G1 threshold registration:
  (1) S3 per-token vocab reduction = MAX |Δlogit| over vocab, then
      mean over reply tokens — matches the repo's cache-equivalence
      idiom (test_graft_mla_gate.py, scribe G0).
  (2) KL direction = KL(full ‖ minus): the full-mounted-set
      distribution is the reference; dependence = perturbation away
      from full context.
- G0b GPU run PENDING — held until WO-1 finishes editing
  core/minicpm3_tc.py (no gate runs against a half-edited tree).
- G0b fixture content borrowed from test_graft_mla_gate.py
  (coolant-manifold / osprey) since WO-4 fixtures hadn't landed;
  acceptable — G0b is a floor measurement, not a G1/G2 gate.

## 2026-07-16 — WO-4 (fixture set) LANDED, lead-verified

- tests/fixtures/importance_convos/: 6 conversations (23-24 turns),
  18 probes, every probe 6 graded candidates (0-3 scale). Validator
  6/6 under lead re-run; convo_02 spot-checked by hand: STANDING_PREF
  turn 3, zero intermediate touches, probed turn 24 (21-turn silent
  gap ≥ the plan's 10-turn floor); SUPERSEDED = shared node_id across
  original+correction, correction higher-graded. Evidence class:
  validator run + manual spot-check.
- LEAD SIGN-OFF on fixture design calls: shared node_id for
  SUPERSEDED touches; turn-4 assistant acknowledgment of the
  preference counts as the SAME graft node under (user, assistant)
  turn-pairing — not a later use (documented in fixture README).
- Identifier collision check across files reported clean by seat
  (BX-44/BX-51, C14, Kessyrn-9, LP-2231 unique per file).
