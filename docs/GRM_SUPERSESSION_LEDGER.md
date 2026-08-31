# GRM Supersession — LEDGER (receipts, append-only)

Plan: GRM_SUPERSESSION_PLAN.md (immutable, b5428f9).

## 2026-07-16 — SUP-WO1 landed (Sol), lead-verified; G0 baseline run

- Battery (4 fixtures, 5 probes), L1 log-length-debias
  s/√(log₂(K+1)) (extreme-value justification frozen in code),
  L2 M5-edge mount resolution; both flagged, default off (verified:
  passthrough guards). 34/34 suites with GPU (seat's 5 telemetry
  fails were CPU-only sandbox env — MLAAttentionTC allocates on
  device at init; confirmed green under lead re-run). Finding from
  seat, verified against source: repository node_id is graft-index
  IDENTITY, not stable lineage — only explicit M5
  supersedes/superseded_by edges are authoritative.
- **G0-SUP BASELINE (receipts artifacts/grm_supersession/
  g0_baseline.jsonl, route_backend=python, MiniCPM3 INT4):
  classification correct 2 / stale 2 / wrong-fact 1.**
  - **SEAM TRANSFER REFUTED IN THE ROUTING SENSE:
    competitor_over_correction_inversions = 0.** The GQA max-pool
    length bias has no MLA analogue (centroid route) — as predicted
    at dispatch. L1's designed gate is NOT DECIDABLE on this
    substrate.
  - **THE MLA FAILURE MODE IS DIFFERENT: STALE-BEATS-CORRECTION.**
    Ancestors rank 1 in every scenario (harbor_a, lumen_a, orion_a);
    corrections rank 2-3; topk=3 co-mounts ancestor+correction; the
    readout returns the STALE value on plain-correction probes
    (orion, lumen). Restatement scenario (harbor) reads correct
    despite worse correction rank — an authoritative restatement
    rescues readout. NOTE: stale values RETURNED here, which the
    GPT-OSS E2E never showed — dialect/substrate difference in the
    failure's locus (GQA: route inversion by competitor; MLA:
    stale-priority ranking + co-mount readout capture).
  - Fresh controls 1/2: praxis probe routed rank-1 correctly,
    mounted correctly, answered the co-mounted sibling's value —
    the corpus-100 co-mount confusion class, NOT supersession;
    carried as battery context, not a supersession failure.
  - First-statement-wins ranking echoes the early-turn-attractor
    class (E4 contextualization split). Mechanism unproven here —
    noted as interpretation.

## 2026-07-16 — G1/G2 THRESHOLDS REGISTERED (before either lever runs)

- **G1-SUP (--debias, L1 only), decidability caveat registered:**
  with zero baseline inversions the plan's "inversions eliminated"
  demand is vacuous on MLA. Registered demand downgraded to
  REGRESSION-ONLY: no correction rank degrades vs baseline;
  inversions stay 0; fresh controls unchanged; shared suites stay
  green. L1's decisive test lives on GQA — registered successor,
  out of scope here.
- **G2-SUP (--debias --resolve, L1+L2 as planned):**
  - stale answers 0/5 (baseline 2);
  - plain-correction and multi-hop probes answer CURRENT: harbor,
    lumen, orion all correct (baseline 1/3 of these);
  - multi-hop resolves to head C with A and B both excluded from
    the mount set when C is available;
  - fresh controls ≥ baseline (≥1/2) — L2 must not touch
    non-lineage nodes;
  - diagnostic leg (--resolve alone, no gate): isolates L2's
    contribution given L1's expected vacuity.
## 2026-07-16 — G1 PASS (vacuous, as registered), G2 PASS (L2 is the fix)

- G1 (--debias): battery IDENTICAL to baseline — every rank, every
  classification (receipts g1_debias.jsonl). Registered
  regression-only demand met; L1 empirically inert on MLA centroid
  routing. L1's decisive test remains GQA (registered successor).
- **G2 (--debias --resolve): PASS on every registered demand** —
  stale 0/5 (was 2); harbor/lumen/orion all CURRENT; multi-hop
  mounts head only (['lumen_c']; A and B excluded); non-lineage
  nodes untouched (orion keeps falcon_competitor mounted; fresh
  controls byte-identical to baseline, 1/2 with the same
  co-mount-confusion miss). Receipts g2_debias_resolve.jsonl.
- Resolve-only diagnostic IDENTICAL to L1+L2 (diag_resolve_only.
  jsonl): **L2 accounts for 100% of the improvement.** Attribution
  clean.
- Both levers remain flagged, DEFAULT OFF — enabling in production
  is an operator decision; default-path behavior unchanged (34/34
  shared suites).
- Residual, explicitly NOT closed: the praxis fresh-control miss is
  the corpus-100 co-mount confusion class (routed right, mounted
  right, read the sibling) — pre-existing, out of this program's
  scope, now with a fresh receipt.

## 2026-08-30 — GRM-SUP-L2-ON DEFAULT FLIP (operator decision)

- Operator: David. Date: 2026-08-30. Verbatim intent: **"Flip that
  switch ON, it being OFF is causing issues".** Scope is L2 only:
  M5 `metadata.supersedes` lineage-head mount resolution is now
  DEFAULT ON in `ArenaCache`. L1 length-debias remains DEFAULT OFF.
- Production resolution order is explicit constructor/CLI choice,
  then `GRM_SUP_RESOLVE`, then permanent default ON. The registered
  legacy escape is **`GRM_SUP_RESOLVE=0`**. The E2E driver also exposes
  `--sup-resolve` / `--no-sup-resolve` and freezes the resolved value
  across restart. The battery keeps `--resolve` / `--no-resolve`.
- Escape semantics are the original passthrough guard exactly:
  `revision_resolution=False` returns the proposed mount list without
  graph parsing, reordering, deduplication, or filtering.

### Baseline re-registration

- NEW DEFAULT L2 receipt anchor:
  `artifacts/grm_supersession/diag_resolve_only.jsonl`, SHA-256
  `8053f9c3437bb6ff4c0e81533159b2e5870c32ca20f895e799ca61f8585d2a78`.
  Its independent post-merge repeat has the same SHA-256.
- NEW DEFAULT transcript-projection SHA-256:
  `4ef9ad270bf3a6e5ccd798f16c0be43f9c4d6544be7d77c1d33d38542a4a1b94`.
- LEGACY ESCAPE receipt anchor:
  `artifacts/grm_supersession/g0_baseline.jsonl`, SHA-256
  `7bc14aaf1eb58a61b6df12586341f409f1768a537b5910b17dd67df9541e8ccd`.
  Escape transcript-projection SHA-256:
  `3e2dfddc0c7cfbbbf9d7f2600772924cfc5549996a2a0e3c022987aaac43b030`.
- Projection excludes only receipt-envelope keys added/evolved outside
  the transcript (`dialect`, `mode`, `record_type`, `schema`); answer,
  classification, ranks, mounts, arena info, fixture, scenario, and
  probe fields remain byte-covered. Machine-readable registration:
  `artifacts/grm_supersession/l2_default_on_registration_20260830.json`.

### Registered acceptance and scope audit

- Resolve-only anchor PASS: correct 4/5, stale 0/5 (legacy 2/5),
  wrong-fact 1/5 (legacy 1/5), fresh controls 1/2. All three revision
  probes answer current; multi-hop mounts only `lumen_c`.
- Fresh controls are byte-identical under the transcript projection,
  SHA-256
  `404a552e72a2c6242f591fb80015f7d59784cb4a301ace8c7a2761af7694f2a1`.
- Re-registered changed supersession paths ONLY:
  `harbor_restatement` (mount/arena fields; answer already current),
  `lumen_head` (stale to current plus mount/arena fields), and
  `orion_current` (stale to current plus mount/arena fields). No fresh
  or non-lineage transcript changed; any additional live drift is RED.
- CPU-safe shared suite: 39/39 PASS. Full GRM CPU collection: 483/488
  PASS; the same five telemetry constructors require a CUDA device and
  failed only with `no CUDA-capable device is detected`. The registered
  live no-env battery, `GRM_SUP_RESOLVE=0` byte check, all 44 shared
  tests, and full 488-test GRM run remain for external GPU execution via
  `python3 scripts/grm_sup_l2_default_on_gate.py --run-gpu`; the runner
  queues each leg on `/tmp/forge-gpu.lock` behind CMC/ADM work.
- Coordination pin verified after wiring: CMC1.1's arena explicitly sets
  `length_debias=False, revision_resolution=False`; ADM1's two arenas do
  the same and its E2E subprocess passes `--no-sup-resolve`. Their old-
  default experimental frames therefore do not inherit this flip.

## 2026-08-31 — GRM-ADM2.1 A-DEC DEFAULT FLIP (operator decision; live re-registration pending)

- Operator decision: David delegated the flip; the GRM-ADM2.1 amendment
  requires full A-DEC adoption and replaces ADM2's erroneous zero-answer-
  change stop with exactly one allowed and required improvement. The frozen
  rule is
  `artifacts/grm_adm1/run_20260830T093538Z_3defaa/decisiveness_rule_c304609f81475bd2.json`,
  SHA-256
  `c304609f81475bd2ae3399ad180d3bb00cc5d2b44b1a49db570387c810defb91`:
  zero identifier bindings select k=3; two or more select the complete
  identified set; one rank-1 binding, or the strict fitted margin above
  `0.1385774091529802`, selects k=1; the remaining one-hit case keeps k=3.
- A-DEC is now DEFAULT ON in `ArenaCache`. Resolution order is explicit
  constructor/CLI choice, then `GRM_ADM_DECISIVE`, then permanent default ON.
  The constructor field is `decisive_admission`; E2E exposes
  `--adm-decisive` / `--no-adm-decisive` and freezes the resolved value across
  restart. The byte-covered legacy escape is `GRM_ADM_DECISIVE=0`, which
  executes the original bounded fixed-k=3 route/admission branch and emits no
  A-DEC arena fields.

### Headline answer improvement and evidence

- The answer-change enumeration is exactly one row:
  `supersession:praxis_fresh`, wrong
  `The current Praxis dock value is Raven-9-Ivory.` under legacy k=3, to
  correct `The current Praxis dock value is Quartz-8-Jade.` under A-DEC k=1.
  The wrong read is in the CMC live G0 receipt
  `artifacts/grm_cmc1/live_arms_20260830T092447Z_4056005/g0_95d6dcb605db1bf0.json`,
  SHA-256
  `95d6dcb605db1bf0c16d2c54f70ba531040b6a7f799e50f0dcac0837c982e39c`;
  the CMC arc convicted value blending in
  `artifacts/grm_cmc2/live_sweep_20260830T170021Z_4124360/adjudication_ba6821ccf299ac17.json`,
  SHA-256
  `ba6821ccf299ac170ff27e685e91af5c76c82ba09ebc62e1150e6b40714788e3`.
  The correct A-DEC k=1 read and adoption recommendation are in
  `artifacts/grm_adm1/adm1_3_prod_run_20260830T161236Z_3eaccb/dual_frame_adjudication_c83a45c60e9fe50b.json`,
  SHA-256
  `c83a45c60e9fe50b76e63fd63e680f6dab1aaab47628cf030057a8a655600da9`.
  Every other recorded answer is byte-stable; any second answer change is
  RED.
- The frozen ADM1.3 F-PROD adoption anchors are CORPUS-100 20/20,
  supersession fresh controls 2/2, E2E-34 9/9, and P4-34 9/9. Both 34-turn
  answer-transcript projections have SHA-256
  `0318050b357bd651eb70b36e82fcf4ab203752e70049c600974322bd3b0f0889`.
  The amended full supersession target is correct 5/5, stale 0/5,
  wrong-fact 0/5.

### Enumerated mount and arena-field re-registrations

- The physical enumeration is 49 turns: 40
  `exactly_one_identifier_decisive_rank1` and 9
  `declared_synthesis_identified_set`. The exact per-row legacy/A-DEC rank
  plans, final mounts, changed result fields, and changed arena fields are in
  `artifacts/grm_adm2/default_on_20260831T150306599577Z_2/registered_acceptance.json`,
  SHA-256
  `fa8d11fea02f5878c339dbf617df2f88ccfcf1100fe11cf357377fcdb0a7f028`.
- Lookup, 22 turns: `corpus100:apiary:1`, `corpus100:apiary:4`,
  `corpus100:sonde:4`, `corpus100:sonde:1`,
  `corpus100:geothermal:7`, `corpus100:geothermal:8`,
  `corpus100:ticket:0`, `corpus100:ticket:5`, `corpus100:vessel:3`,
  `corpus100:vessel:2`, `corpus100:depot:6`, `corpus100:depot:9`,
  `corpus100:yeast:9`, `corpus100:yeast:6`, `corpus100:loco:2`,
  `corpus100:loco:3`, `corpus100:psalter:5`, `corpus100:psalter:0`,
  `corpus100:reef:8`, `corpus100:reef:7`,
  `supersession:praxis_fresh`, and `supersession:solace_fresh`.
  Lookup arena-field changes are confined to `evicted`, `mounts`, and
  `resident` on the two supersession controls; the corpus rows change their
  admission/rank-plan receipt fields while retaining their final correct
  mount. Praxis is the sole row whose answer, correctness, and wrong-read
  fields also change.
- DIAG, 9 turns: `diag:turn-5:orion-pin`,
  `diag:turn-9:cypher-bridge`, `diag:turn-13:orion-pin`,
  `diag:turn-16:lyra-dock`, `diag:turn-19:nova-key`,
  `diag:turn-22:mira-seal`, `diag:turn-24:terra-port`,
  `diag:turn-26:ember-code`, and `diag:turn-30:atlas-tone`.
- E2E-34, 9 turns: `e2e-anchor:turn-5:orion-pin`,
  `e2e-anchor:turn-9:cypher-bridge`, `e2e-anchor:turn-13:orion-pin`,
  `e2e-anchor:turn-16:lyra-dock`, `e2e-anchor:turn-19:nova-key`,
  `e2e-anchor:turn-22:mira-seal`, `e2e-anchor:turn-24:terra-port`,
  `e2e-anchor:turn-26:ember-code`, and
  `e2e-anchor:turn-30:atlas-tone`.
- P4-34, 9 turns: `p4-anchor:turn-5:orion-pin`,
  `p4-anchor:turn-9:cypher-bridge`, `p4-anchor:turn-13:orion-pin`,
  `p4-anchor:turn-16:lyra-dock`, `p4-anchor:turn-19:nova-key`,
  `p4-anchor:turn-22:mira-seal`, `p4-anchor:turn-24:terra-port`,
  `p4-anchor:turn-26:ember-code`, and `p4-anchor:turn-30:atlas-tone`.
  Across DIAG/E2E/P4, arena changes are confined to the enumerated
  `driver_topk`, `ranking_ids`, `mount_plan`, `mount_dropped_for_width`, and
  `precise_first` surfaces; result changes are confined to `arena_info` and
  the A-DEC policy/plan/production-ladder receipt fields. Answers do not
  change.

### Validation, pins, and remaining live gate

- CPU receipt/hash gate PASS; focused admission/ladder/supersession/snapshot
  suite 53/53 PASS. The broad GRM collection reached 504/509 PASS; the five
  failures are the existing importance-telemetry constructors that allocate
  real TensorCUDA tensors, and all failed with `cudaMalloc failed: no
  CUDA-capable device is detected` because this managed worker exposed no
  `/dev/nvidia*` device nodes. No admission assertion failed.
- Historical counterfactual frames are explicit: ADM1, ADM1.3, CMC1 arms,
  CMC2, and both legs of the L2-only default-on gate pin A-DEC off. DET1 needs
  no pin: its order and implementation resolve the production default at GPU
  start, record `adm_decisive` and the exact environment in the runtime-frame
  receipt, and plant the withheld winner, making admission non-variable in
  that race.
- Live new-default, `GRM_ADM_DECISIVE=0` byte-projection, amended 5/5
  supersession, both 34-turn sessions, shared suite, and full suite receipts
  remain PENDING because this worker cannot allocate CUDA. The registered
  serialized runner is:
  `PYTHONPATH=/mnt/ForgeRealm/GraftRepository python3 scripts/grm_adm2_default_on_gate.py --stage all`.
  It self-leases `/tmp/forge-gpu.lock`, caps each leg below 590 seconds,
  writes a new append-only run directory, and refuses the run on any second
  answer change. New live default/escape baseline hashes must come from that
  run; none are fabricated here.
