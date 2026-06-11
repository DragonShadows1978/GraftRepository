# SCRIBE — registered thresholds (protocol §4 discipline)

Written 2026-06-11, after G0 and before any gate below runs. Amendments
by the Architect are valid only BEFORE the governed gate runs.

## G0 result (the instrument, measured)

- Pipeline isolation: minted-from-disk vs direct-harvest mount —
  **max|Δlogit| = 0.00000** on all docs (fp16 storage is exact at the
  engine's compute precision).
- Graft-vs-in-context noise floor: **0.3125 max|Δlogit|**; top-1 flips
  occur ONLY at ctx margins ≤ 0.0625 (one bf16 ULP) — the established
  exact-tie class. 3 docs, 3 domains, held-out probes.

## Registered Δ for the gates

| Gate | Pass threshold (registered) | Kill |
|---|---|---|
| **G2** logit fidelity | per-domain top-1 agreement (predicted vs exact mounts, teacher-forced probe spans) ≥ **90%**, AND mean per-position logit KL ≤ **0.5 nats** | top-1 agreement indistinguishable from chance |
| **G3** needle readback | predicted-graft needle recall ≥ **exact − 1** on a 10-needle held-out-domain set | recall ≤ 2/10 with exact ≥ 8/10 (content collapse, H2 strong form) |
| **G4** router recall | recall@1 AND @3 over a 20-graft predicted repository within **Δ = 2/20** of the same-repository exact recall | below exact by > 4/20 (H1 falsified) |
| **G5** digest tolerance | probe accuracy through predicted-sourced digests within **1/8** of exact-sourced digests (same probe set) | degradation ≥ 3/8 (compounding — predicted tier excluded from consolidation) |

Notes binding the numbers:
- "exact" arms always run in the same process/protocol as predicted arms;
  fresh process per gate.
- All teacher-forced comparisons inherit G0's flip allowance: top-1
  disagreements at ctx margins ≤ 0.125 (2 ULP) count as agreement.
- Per-domain reporting is mandatory (distribution-shift door §7); a gate
  passes only if it passes on EVERY held-out domain.
