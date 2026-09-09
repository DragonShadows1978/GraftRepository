# SCOUT-FIX-1 conditional attribution (CPU reasoning only)

E2E profile drop has NOT been observed: no GPU cells ran. The order's bisect
fork is NOT triggered. No commit was reverted, and no GPU bisect was run.
This is a source-based inspection map for the lead, not a causal diagnosis.

| Item | File and seam | Expected C2 sensitivity | Conditional lead investigation |
|---|---|---|---|
| B1 | core/grm_demand.py, final flush/answer index guard | Demand is OFF in both registered arms; direct effect is unlikely by reasoning. CPU final-flush tests still matter. | Inspect whether demand was actually OFF and an observer was attached before selecting a one-item B1 source overlay. |
| B2 | core/grm_three_pass.py, admission projection/passthrough | Additive receipt plumbing; should preserve served answers. C2 explicitly builds the production route receipt from served info. | Separate lost RT1 evidence from changed routing/answers. Overlay B2 alone in a lead-owned sidecar if the evidence points here. |
| B3 | core/graft_repository.py, optional capture mapping save/load | Directly exercised by pre-probe persistence and new-process reload. Most relevant to lost/mismatched capture metadata. | Compare before/load manifest projections and payload hashes first; legacy missing capture must stay explicitly inherited. Overlay B3 alone, never reharvest to hide a miss. |
| B5 | scripts/grm_wc1_results.py, ntok aggregation | Reporting only; C2 sums actual mounted ntok and retains raw IDs. No serving effect expected. | Compare raw rows and independent token sum with report output. Do not interpret changed reporting as changed model quality. |

If measured profile counts drop below9/9,9/10,13/14, attribute to these four
SCOUT-FIX-1 diffs first as ordered. The lead can arrange four independent
one-item reversions/overlays; preserve original sources and hash each variant.
No cumulative reversion, retuning, threshold change, automatic retry or GPU run
is authorized here. Exact commit IDs are not inferred from old reports; lead
must bind item commits before any GPU bisect. No git was used by this seat.

Prior art: SCOUT-FIX-1 report/fixtures, EB1 demand-off frame, B2/RT1 projection,
B3/RS3 ordinary manifests, B5/WC1 measured seats (project contributors,2026).
Taken: existing causal seams and isolated-item comparison. Ours: C2-specific
inspection map. No new algorithm; no prior art known to me beyond these local
systems for these repair-specific hypotheses.
