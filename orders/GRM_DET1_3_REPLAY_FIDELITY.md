# ORDER GRM-DET1.3 — replay fidelity: why does reconstructed state refuse where lived state answers?

WRITABLE TARGET `/mnt/ForgeRealm/GraftRepository`; rules as DET1.x.

Finding (lead-verified across bisect receipts): EVERY fresh-process
arm of the DET1.2 bisection — including bare controls with zero
instrumentation — REFUSES on harbor_restatement, while the ADM2 gate's
live in-session run answers "Nacre-6-Blue" correctly. Instrumentation
exonerated. The served-counterfactual REPLAY path (reconstructed
arena/process state, defer_memory) is behaviorally divergent from the
lived session path on a knife-edge turn. This is the DOSE arc's
named-but-never-built instrument coming due: the live dual snapshot.

Work:
1. Build the dual-snapshot comparator: run the correction_then_
   restatement session LIVE to the probe turn and snapshot the full
   model-visible state (arena K/V bytes, positions/rope state, sink
   rows, masks, live tokens, admission plan); separately build the
   REPLAY state the counterfactual harness constructs for the same
   turn; diff byte-level, field by field. Name every divergence.
2. Adjudicate the divergence class: (a) replay-harness construction
   bug (fixable to bit-faithful — fix it, gate = replayed state
   byte-equals lived snapshot); (b) inherent path difference (e.g.,
   lived state includes something replay cannot reconstruct — name
   it, and the law becomes "counterfactuals must fork from live
   snapshots, not reconstruct"); state which.
3. Note the knife-edge context honestly: the turn flips to a REFUSAL,
   not a wrong fact — connect to the E2E refusal receipts and the
   GLC distilled-model findings in one paragraph (this model's
   refusal basin is adjacent to normal serving; small state error
   lands in it).
4. If (a) fixed or (b) fork-from-snapshot implemented: resume the
   race through DET-G1 (all-hooks byte-identity now against the
   faithful replay) to the verdict.

Done: the field-by-field divergence table; (a)/(b) adjudication;
fix-or-law; the knife-edge paragraph; race status; anything not done.
