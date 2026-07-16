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

## 2026-07-16 — WO-1 (S1 telemetry) LANDED, lead-verified

- Tap: core/minicpm3_tc.py absorbed-decode path, pure read of `w`
  post-softmax (lead read the diff: no in-place ops, off-path is the
  pre-existing two statements verbatim). Decode-only = matches the
  registered metric. 9/9 CPU tests under lead re-run.
- Seat attribution VERIFIED by lead against source, not taken from
  the seat's report: cur_mounts = picks at swap (graft_arena.py:1502,
  1530); _graft_block cats in picks order re-RoPE'd from n_sink →
  _mount_seat_ranges IS the physical layout. Evidence class: source
  audit + unit test.
- LEAD SIGN-OFF on two flagged calls: (1) s1_mass keyed by graft
  index in cur_mounts (consistent with step() info["mounts"]; node-id
  translation layer deferred to consumer phase); (2) share denominator
  = ALL non-live mass incl. sink (literal plan reading; mount ordering
  unaffected by common denominator; sink share visible as 1−Σ).
- _attempt() resets telemetry per attempt incl. rollbacks — failed
  trips can't bleed mass into accepted attempts.
