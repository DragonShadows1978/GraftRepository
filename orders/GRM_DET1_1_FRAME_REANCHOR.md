# ORDER GRM-DET1.1 — micro fix: re-anchor DET1 canonical to live registered baselines

WRITABLE TARGET `/mnt/ForgeRealm/GraftRepository`; rules as DET1.

Defect: eval_sup_1 guard fired — "instrumented served counterfactual
differs from canonical production path for sup_harbor_restatement."
DET1's frame froze BEFORE the A-DEC default flip; ADM2's gate then
legitimately re-registered harbor (mount/arena fields). Third
baseline-skew of this class (ADM1.2 probes, ADM2 orion, now this).

Work:
1. Verify the instrumented served path matches the CURRENT registered
   (post-ADM2) baseline for harbor — if it does, the guard's canonical
   was stale, not the instrumentation. If it does NOT match either
   baseline, STOP: that is a real instrumentation leak, report it.
2. GENERAL FIX (the third occurrence earns it): DET1's purity guard
   compares against the LIVE registered-baseline registry at run time
   (recording which registration hash it anchored to in every
   receipt), never a private frozen snapshot. Planted-miss semantics,
   detectors, thresholds, and the pre-data prediction registration
   remain byte-unchanged.
3. Resume the frozen run (--run-dir run_20260831T160525Z_2) through
   the race to the verdict.

Done: stale-canonical vs instrumentation-leak adjudication; the
anchor-registry diff summary; resume receipt; anything not done.
