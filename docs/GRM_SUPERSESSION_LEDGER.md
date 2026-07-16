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