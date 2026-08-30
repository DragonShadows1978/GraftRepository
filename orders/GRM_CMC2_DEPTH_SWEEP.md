# ORDER GRM-CMC2 — quantization-depth sweep: fusing un-supersedes and co-mount into one curve

QUEUED BEHIND GRM-CMC1 (run 20260830T041553Z-4006825): reuse its
fixtures, witness, and clipping machinery. Do not begin until CMC1
artifacts exist; consume its verdicts as context (any H outcome is
compatible with this order).

## Grant

As GRM-CMC1: writable `/mnt/ForgeRealm/GraftRepository`, CPU self-runs
+ emitted GPU scripts, append-only artifacts (`artifacts/grm_cmc2/`),
new code `scripts/grm_cmc2_*.py`, production byte-identical with flags
off. FORBIDDEN: git, subagents, network.

## Hypothesis (David + lead, registered)

Three GRM ranking failures share one shape — representation statistics
voting where content should decide (route length-bias; below-floor
un-supersedes; co-mount readout). Mechanism frame: instance IDENTITY
lives in the representational tail (few distinctive components); TOPIC
lives in the bulk. Quantization clips outliers first (de-energizing
norm bullies) and destroys tail identity last (below the ~3-bit
floor). Therefore co-mount behavior vs storage depth should be
NON-MONOTONIC, and the INT8 rehydrate repair (CMC1 context) and the
3-bit un-supersedes floor (GRM_GRAFT_QUANT_LEDGER) are two ends of one
curve.

## The sweep (registered)

On the CMC1 co-mount fixtures, mount-time graft storage depth ∈
{bf16, INT8, INT6, INT4, INT3, INT2} (existing pack formats where they
exist; the below-floor depths may reuse the quant program's machinery).
Per depth, measure:
1. Wrong-read outcome (the CMC1 probe set) — repaired / confused /
   collapsed.
2. Witness mass-split (CMC1 instrument): target-vs-sibling captured
   mass AND the split's entropy — outlier-dominance (mass on few
   sibling keys) vs identity-collapse (mass flattening toward
   uniform across grafts) are DIFFERENT signatures; report both
   statistics per depth.
3. Key-norm outlier stats per graft per depth (p50/p99/max).
4. A supersession probe on the SAME fixtures (v2-outranks-v1 check,
   supersession machinery as-is) per depth — connecting this curve to
   the registered un-supersedes floor on identical material.

## Registered predictions and adjudication

- P1 (repair band): at INT8/INT6, wrong reads repair vs bf16, with
  witness showing reduced sibling outlier-mass concentration.
- P2 (identity collapse): at/below INT3, a DIFFERENT failure
  signature — supersession probe degrades AND witness mass-split
  flattens (rising entropy), i.e. confusion-because-indistinguishable
  rather than confusion-because-outshouted.
- LAW-FUSED iff P1 AND P2 both hold with their signatures. HALF-LAW:
  exactly one holds — name which, with numbers. REFUTED: neither —
  the depth curve is monotonic or flat; say so plainly.
- Do not harmonize mixed signatures; a mid-band depth showing BOTH
  signatures at once is the most informative row — report it raw.

## Gates

- CMC2-G0: CMC1 fixtures still reproduce at bf16 defaults.
- CMC2-G1: byte-identity with all flags off.
- CMC2-G2: per-depth pack/unpack round-trips content-validated
  (existing quant machinery receipts).
- CMC2-G3: report `artifacts/grm_cmc2/GRM_CMC2_REPORT.md` — the
  depth × {read outcome, mass-split, entropy, norm stats,
  supersession} table, the adjudication sentence, and one paragraph
  placing the result against GRM_GRAFT_QUANT_LEDGER's floor.

## Constraints

All standing laws (GPU discipline, provenance, non-finite writer, RED
honesty, no post-hoc widening). ## Done: gate statuses, the full sweep
table verbatim, the adjudication sentence, files, anything not done.

---

# GRM-CMC2 ADDENDUM

Context update since CMC2 was registered: CMC1.1 adjudicated the
co-mount mechanism as VALUE BLENDING — H-OUTLIER NOT CONVICTED (mass
not concentrated on outlier keys; sibling-only clipping does not flip
the read). CMC2's registered P1 ("repair band via outlier
de-energizing") therefore tests a now-disfavored mechanism — WHICH IS
THE POINT: the predictions were frozen first; run them unchanged. A
P1 failure-to-appear is independent confirmation of blending; a P1
appearance would reopen outlier contribution at specific depths.
P2 (identity collapse at/below INT3, mass-split entropy rising,
supersession probe degrading) is unaffected and remains the bridge to
the un-supersedes floor. Adjudication vocabulary unchanged
(LAW-FUSED / HALF-LAW / REFUTED). Frame: production defaults (ladder
ON, L2 ON) per ADM1.2's alignment; note it in every receipt. Use the
trued witness (engine-operand tap, CMC1.2). Reuse CMC fixtures and
machinery; self-run CPU, emit GPU scripts.
