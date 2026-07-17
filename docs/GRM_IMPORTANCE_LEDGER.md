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

## 2026-07-16 — G0a first run RED → tap EXONERATED, harness+engine finding

- G0a as-shipped FAILED: max|Δlogit| 0.5, s1_mass {}. Lead diagnostic
  driver (scratchpad, teacher-forced logit A/B, model loaded once):
  - off-vs-off (runs 1,2) = 0.5 → the DELTA IS NOT THE TAP.
  - warm off vs warm on = 0.0 EXACTLY (direct receipt, both
    orderings) → telemetry tap is bit-clean under matched warmth.
  - live_shift-after-feed ordering hypothesis REFUTED: with
    live_shift broadcast before feed on every run, off-vs-off
    still 0.5.
  - Pattern: FIRST RUN OF THE PROCESS differs (≤0.5 logit, bf16-noise
    scale) from all subsequent runs; all warm runs bit-identical
    regardless of telemetry/ordering.
- ENGINE FINDING (open seam, not chased here): process-warmth
  first-run effect. Candidates: kernel autotune / lazy table build.
  Measurement law (kin to Trinity's matched-reference law): any
  same-process A/B gate must warm up before capturing side A.
- s1_mass {} root cause: gate never calls step() — nothing mounts.
  Harness gap, separate from the parity question.
- Gate fix dispatched to WO-1 seat: warm-up pass before the A/B pair;
  mount via step() so attribution is exercised (assert s1_mass
  non-empty); live_shift-before-feed hygiene. Bit-identical demand
  UNCHANGED. Evidence class: teacher-forced logit A/B.

## 2026-07-16 — WO-3 (S2 salience) LANDED (ab2d83a), lead-verified

- Idle-window scoring pass in core/graft_repository.py: frozen rubric
  (module constant, primed-prefix "Assistant: Rating:" mirroring
  DIGEST_PROMPTS), strict first-position 0-3 parse, one retry, double
  failure → None never a guess; arena snapshot/restore stricter than
  consolidate(); no deposits, routing state untouched. 19/19 CPU
  under lead re-run (17 + 2 after correction below).
- LEAD CORRECTION applied by seat: kind="recall" nodes EXCLUDED from
  S2 scoring (derivative-turn hygiene law — no consumer reads a
  salience score off a wake node). Unit tests prove never-queued /
  never-scored.
- LEAD SIGN-OFFS: importance dict lives at metadata["importance"]
  (program-wide manifest location, S1 consumer-phase persistence to
  match); multi-node new_nodes all scored (matches S1 per-mount
  granularity). Seat self-caught + fixed a retry-remount bug
  (retry would have scored an empty mount).

## 2026-07-16 — G0b PASS (registered floors) + pre-existing suite RED

- G0b (warm-up-equipped, commit ab2d83a): load-bearing dependence
  mean|Δlogit| 7.970 / KL 2.599 vs decoy 1.881 / 0.0181. Ordering
  assertion PASS. REGISTERED FLOORS for G1/G2 thresholds:
  noise_floor mean|Δlogit| = 1.881, KL floor = 0.0181; dynamic range
  4.24× (|Δlogit|) / 143× (KL). Evidence class: teacher-forced logit
  A/B, sealed JSON line (schema grm_importance_s3_g0b_v1).
- OBSERVATION (not a metric change): KL separates 34× harder than
  |Δlogit|. Registered primary for S3 stays mean|Δlogit| per plan;
  any arbiter-metric change would need David + fresh registration
  BEFORE G1 runs, never after.
- Decoy floor 1.881 > the historical 0.75-0.9 cache-vs-prefill bf16
  band — expected: removing a real mount changes the attention
  denominator everywhere; different measurement class, keep separate.
- PRE-EXISTING FINDING (not this program): tests/
  test_grm_runtime_lifecycle.py 91/101 RED on clean main — FakeArena
  test double lacks _bump_cuda_gqa_epoch (call site landed e8906dc,
  July 8 bridge merge). Confirmed by lead on quiet machine + stash
  A/B; WO-3 seat's "environment contention" attribution WRONG, its
  same-both-sides A/B conclusion RIGHT. Queued in GRM_BUG_QUEUE.md.

## 2026-07-16 — G0a round 3: harness green, gate caught a REAL tap bug

- Round-3 harness (fact turn + 2 fillers pushes fact graft out of the
  live_turns=2 recency window per evict()/live_segs mechanics; mounts
  now happen). Gate then crashed IN THE TAP: minicpm3_tc.py:229
  broadcast (103,)→(77,) — S_all SHRANK mid-accumulation-window
  (trips rollback / clean-room mini-cache = cache surgery).
- Real defect class: growth-only accumulator + physical-seat keying
  across surgery = crash now, silent misattribution if merely
  clamped. Invariant registered: MASS ACCUMULATES ONLY WITHIN A
  STABLE SEATING EPOCH; surgery invalidates the accumulator.
  Note: _telemetry_mass lives on the LAYER, not the arena — fresh
  ArenaCache ≠ fresh accumulator (suspected hole in warm-up/measured
  sequencing; seat ordered to name the mechanism, not patch the
  symptom). Fix dispatched to WO-1.

## 2026-07-16 — G0a PASS. G0 PHASE COMPLETE (both gates green)

- WO-1 round-4 fix: (a) crash fixed — accumulator SHRINK (S_all <
  stored length) now discards and restarts (shrink = proof of cache
  surgery; conservative under-attribution, never misattribution);
  (b) real scoping hole closed — set_telemetry(True) previously never
  reset the layer-level accumulator (only disable did), so mass could
  leak across arena instances sharing the model. 12/12 CPU under lead
  re-run.
- Seat's static trace could NOT reconcile the 103→77 shrink with
  max_trips=0 (trips/clean-room ruled out inside step()); flagged
  honestly. Lead's candidate mechanism: end-of-step LIVE-SEGMENT
  EVICTION physically removes rows between step() and the
  teacher-forced loop. UNCONFIRMED — recorded as interpretation, not
  a gate finding. Gate green either way (discard-on-shrink is safe
  by construction).
- G0a RECEIPT: max |Δlogit| = 0.0 EXACTLY over 15 teacher-forced
  decode steps, telemetry on vs off; mounts non-empty both sides;
  s1_mass = {0: 0.600} — the mounted graft drew 60.0% of non-live
  attention mass during the reply. Evidence class: teacher-forced
  logit A/B.
- Receipt-reading note for G1 authors: info["mounts"] is 1-BASED
  (existing arena convention, graft_arena.py:2049); s1_mass keys are
  0-based raw graft indices. mounts=[1] ↔ s1_mass key 0. Checked,
  consistent, not a bug.

## 2026-07-16 — G1/G2 THRESHOLDS REGISTERED (before either gate runs)

From the G0b floors (noise_floor mean|Δlogit| 1.881). Registered by
lead; David may veto/adjust BEFORE the governed gate runs — never
after.

- G1 (per candidate signal, over all 18 fixture probes):
  PASS = median Spearman(signal ranks, S3 ranks) ≥ 0.5 AND top-1
  agreement with S3 ≥ 50% (chance ≈ 17-25% at 4-6 graded candidates).
  S3 ranks computed on the registered primary (mean|Δlogit|); mounts
  with dependence < 2× noise floor (< 3.762) are reported but their
  top-1 slots don't count against a signal (no load-bearing winner
  exists there). KL reported as diagnostic throughout.
- G2 (S2 only; prospective discriminator):
  PASS = median s2_salience(STANDING_PREF) ≥ 2 AND
  (median STANDING_PREF − median FILLER) ≥ 1 rubric point.
  S1 on STANDING_PREF is a REGISTERED EXPECTATION of failure
  (retrospective signal, zero uses by construction): report
  STANDING_PREF vs FILLER s1_mass; no pass/fail attached to S1 here.
- G3 dispatch condition per plan: G1 green for at least one signal
  OR G2 green.

## 2026-07-16 — WO-5 (G1/G2 driver) LANDED + leg-1 shakeout arc

- Driver + analysis + 43 CPU tests (lead re-ran; 31→43 through the
  arc below). Lead sign-offs: probe eligibility rule applies to BOTH
  signals' top-1 denominators (probe property, not signal property);
  defensive user+assistant turn-id mapping; S3 sweeps snapshot-
  isolated on fresh mini-caches (WO-2's gated pattern).
- Shakeout iterations, each mechanism NAMED before fixing:
  (1) fixture probe→scripted-answer structure unmodeled (probe pairs
  now consumed but NEVER deposited — the wake-turn hygiene law;
  scripted answer recorded as expected_answer_scripted; invariant
  test walks all 6 fixtures);
  (2) s1 flat 0.0 — driver never set absorbed_decode=True; the S1
  tap exists only in the absorbed-decode branch; generic SDPA path
  never fires it. Lead's stale-read hypothesis WRONG, seat's trace
  right (read ordering was already correct);
  (3) librarian fold crash, mechanism confirmed by lead's
  instrumented run (scratchpad monkeypatch): STALE WORKSPACE —
  /tmp/graftrepo_g1g2_convo_01 carried wal/ state from earlier
  crashed attempts; repo BOOTED with 11 WAL-recovery placeholder
  nodes (kind=turn, ntok=0, h=None, recovered/payload_pending);
  placeholders count as foldable → fold fired instantly → consolidate
  TypeError. native_store=False confirmed (Python fold path; seat's
  fresh-repo fold arithmetic was correct all along).
- OPERATIONAL NOTE: bare `pytest --collect-only` over tests/ is a
  GPU hazard — a legacy module loads a model at import; a leaked
  10.8GB collection process blocked leg 1 (killed after identifying
  it as this session's own leftover).

## 2026-07-16 — M11 REGISTERED: fold-after-recovery bricks librarian

## 2026-07-16 — G1/G2 PARTIAL RESULTS (15/18 probes — NOT the gate)

- Legs 1,3,4,5,6 GREEN (artifacts committed); leg 2 failed pre-run:
  convo_02 turn 19 is a SOLO scripted user turn (no assistant reply
  before the next probe) — driver's strict-pairing assumption; fix
  order = first Sol dispatch under the new protocol. Fixtures are
  gate-registered data: the DRIVER adapts, fixtures unchanged.
- PROVISIONAL numbers over 15 probes (verdict schema sealed in
  analyze output; gate verdict WAITS for all 18 registered probes):
  - S1 G1: median Spearman 0.5071 (bar 0.5), top-1 53.3% (bar 50%)
    — passing BY A HAIR; 3 missing probes can flip it. NO CALL YET.
  - S2 G1: FAIL trajectory — Spearman 0.171, top-1 25%, rankable on
    only 4/15 probes (scores collapse to 0).
  - S2 G2: FAIL trajectory — median s2(STANDING_PREF)=0.0 vs bar 2.0;
    the 4B judge rates standing preferences as throwaway. NOTE: the
    frozen rubric words the scale around "facts" — plausible wording
    artifact, but the rubric froze pre-G1; result stands. A reworded
    rubric is a SUCCESSOR arm (fresh registration), never a retune.
  - S1 on G2 (registered expectation of failure, no gate):
    STANDING_PREF 0.0568 vs FILLER 0.0277 — 2× separation at tiny
    magnitude, observation only.
- Evidence class: recall-gate-style driver receipts + analysis over
  sealed per-convo JSON artifacts (grm_importance_g1g2_convo_v1).

## 2026-07-16 — G1/G2 FINAL VERDICTS (all 18 registered probes): RED

- WO-6 (Sol, first dispatch under the shim protocol, order e8ba4f9):
  solo-user-turn handling, faithful add_turn-sequence mirroring for
  solo deposits, 53 CPU tests, lead-verified. Leg 2 GREEN; analysis
  over all 6 artifacts / 18 probes = THE registered gate.
- G1 S1 (attention mass): median Spearman 0.4733 vs bar 0.5 — FAIL
  (top-1 agreement 0.5000 met its bar exactly; the AND fails). The
  15-probe partial (0.5071) flipped with convo_02 added — the
  marginal-pass caution was warranted.
- G1 S2 (self-report): Spearman 0.219, top-1 4/7, rankable on only
  7/18 probes — FAIL.
- G2 S2: median s2(STANDING_PREF) = 0.0 vs bar 2.0 — FAIL. The 4B
  judge scores standing preferences as throwaway under the frozen
  fact-worded rubric.
- G2 S1 observation (no gate): STANDING_PREF 0.0626 vs FILLER 0.0274
  (2.3×). NUANCE: this is PROBE-TIME mass (preference mounted while
  being asked about) — it does not contradict the designed
  deposit-time failure; S1 has no deposit-time value at all.
- G3 DISPATCH CONDITION NOT MET (plan: G1 green for ≥1 signal OR G2
  green). Consumer phase does not run. Verdict, evidence-classed
  (recall-gate + teacher-forced A/B arbiter): under the registered
  thresholds, on MiniCPM3-4B with these fixtures, neither
  retrospective attention mass nor 4B self-report salience tracks
  counterfactual importance well enough to drive memory management.
  NOT DETECTED UNDER THESE LIMITS — not refuted. Un-searched space,
  named for successor registration (fresh pre-registration each, no
  retunes): (1) per-layer S1 aggregation (all-layer mean was the
  registered metric; per-layer diagnostics sit unread in the sealed
  artifacts); (2) KL-ranked S3 arbiter (KL separated 143× vs 4.24× at
  G0b; ranks recomputable from sealed artifacts — deliberately NOT
  computed post-hoc this round); (3) rubric-v2 arm (preference-aware
  wording) and/or larger judge for S2; (4) richer probe replies
  (longer generations widen S1's mass sample); (5) the silent-pass
  ARCHITECTURE itself is untouched by this red — the idle-window
  machinery (S2 pass, telemetry, fold guard) all landed and gate
  green; what failed is these two SCORING functions at this scale.

- REAL PRODUCTION BUG exposed by the leg-1 crash (not a harness
  artifact): fold-source selection never excludes payload-missing
  placeholder nodes. Any crash-recovered session (WAL replay,
  payloads gone) bricks its librarian on the first idle() —
  TypeError in consolidate. M2 guarded the FLUSH path against gap
  placeholders (2026-07-02); the FOLD path was never guarded.
  Evidence class: instrumented reproduction (graft state dumped at
  crash). Fix order dispatched (fold eligibility = resolvable
  payload only; both deferred and backpressure paths; counter-jam
  guard per 816a0a0 lesson). Queued in GRM_BUG_QUEUE.md as M11.

## 2026-07-17 — CORRECTION + exploratory registration (successor 7a/7b)

- CORRECTION: the 2026-07-16 close-out claimed per-layer S1
  diagnostics "sit unread in the sealed artifacts." FALSE — lead
  verified 2026-07-17: artifact schema carries headline s1_mass only
  (candidate keys: s1_mass, s2_salience, s3_dep_dlogit, s3_dep_kl).
  The per-layer study requires a driver extension (record
  s1_mass(per_layer=True) per candidate) + GPU leg re-runs.
- EXPLORATORY REGISTRATION (7a per-layer S1, 7b KL-ranked arbiter):
  both are HYPOTHESIS-GENERATING analyses, not gates. No pass/fail
  thresholds attached. Any metric change they motivate (e.g., KL as
  S3 primary, single-layer S1) requires a FRESH gate registration
  before any gate runs on it. 7b computes from sealed artifacts
  (s3_dep_kl per candidate); 7a re-runs legs with per-layer capture
  under the same fixtures + warm-up law.

## 2026-07-17 — VERDICT CORRECTION: S1 G1 = PASS under the registered gate

- The 7b diagnostics (CAMPAIGN-D) found a HARNESS DEVIATION in the
  published G1 verdict: compute_g1 correlated signals against
  ground_truth_grade (fixture labels), while the registered gate
  (2026-07-16 threshold registration, verbatim) demands
  "median Spearman(signal ranks, S3 ranks)". Lead verified at source
  (spearman(gt, sig_vals)) AND independently re-ran the diagnostics.
- CORRECTED VERDICT from the same sealed artifacts, registered
  semantics: S1 median Spearman vs S3 = +0.8286 (bar 0.5), top-1
  9/18 = 0.5000 (bar 0.5) → **S1 G1 PASS**. (Top-1 was computed vs
  S3 all along — only the Spearman reference deviated.)
- S2 remains FAIL: rankable on 7/18 probes; the registered gate spans
  all 18; 11 probes have no S2 ranking (score collapse). G2 (0.0 vs
  2.0) unaffected by any of this.
- LEAD ACCOUNTABILITY: the deviation shipped because gate-time
  verification checked that the harness ran and its numbers were
  internally consistent — not that the Spearman reference matched
  the registration text. New check for the discipline: verify the
  MEASURED QUANTITY against the registration, not just the result.
- RESEARCH OBSERVATION the error exposed: S1 tracks CAUSAL
  dependence (0.83 vs S3) far better than HUMAN-LABELED importance
  (0.47 vs fixture grades). The published "replicates Jain &
  Wallace" claim attached to the wrong measurement and is WITHDRAWN
  for the registered gate; attention mass vs counterfactual
  dependence is strong here, not weak. The grade-vs-S3 gap is itself
  a finding: what humans label important ≠ what the model causally
  used.
- 7b RESOLVED: S3 is metric-robust (dlogit vs KL ranks ρ 0.9429,
  top-1 18/18 identical) — no arbiter change warranted; no fresh
  registration needed.
- DOWNSTREAM: the plan's G3 dispatch condition (G1 green for ≥1
  signal) was retroactively MET. The G3 consumer question is already
  in flight via superior receipts: fold-order (section 4, S4 signal)
  and s4_protect paging; an S1-driven consumer remains an OPTION,
  not auto-dispatched. AUDIT NOTE: the S4 harness was checked for
  the same deviation and is CLEAN — its Spearman correlates against
  s3_dep_dlogit (test_grm_s4_ledger.py:383-385), the registered
  reference; S4's G1 PASS (0.7556/0.875) stands as measured. Only
  the g1g2 harness deviated.

## 2026-07-17 — S2-v2 (rubric-v2) REGISTERED (campaign 7c, before any gate)

- Arm: S2 with a reworded frozen rubric that explicitly names
  preferences / standing directives / instructions as keep-worthy
  classes alongside facts (v1's fact-only wording is the suspected
  cause of the standing-pref 0.0 collapse). Everything else
  byte-identical: primed prefix, strict parse, one retry, None on
  double failure, recall-kind exclusion. Exact v2 wording FROZEN in
  code at dispatch; opt-in rubric="v2" alongside default "v1".
- Gate (inherits the registered semantics, fresh registration):
  G1-S2v2 = median Spearman(S2v2 ranks, S3 ranks) ≥ 0.5 AND top-1 vs
  S3 ≥ 50% over ALL 18 probes (unrankable probes count against, as
  v1's did); G2-S2v2 = median s2(STANDING_PREF) ≥ 2 AND delta vs
  FILLER ≥ 1. Measured from ONE fresh set of six v2 GPU legs (same
  fixtures, warm-up law) that simultaneously capture s1_mass_per_layer
  (7a) — shared legs, independent analyses.
- Explicit rail: the G1 analysis for S2v2 uses the CORRECTED
  registered semantics (vs S3 ranks), never fixture grades.
