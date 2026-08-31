# ORDER GRM-DET1 — the demand-detector race (missing-sibling detection: mechanistic vs verbal)

WRITABLE TARGET `/mnt/ForgeRealm/GraftRepository`; standing rules;
CPU self-runs; GPU legs self-lease. New code `scripts/grm_det1_*.py`;
artifacts `artifacts/grm_det1/`. Frame: production defaults AT RUN
TIME (ladder ON, L2 ON, admission = whatever production default holds
when the GPU legs run — record exact flags in every receipt; planted
misses make admission policy a non-variable here).

## Context (one paragraph)

The mount-curation design's Stage C needs a demand signal: the model
is missing a needed graft mid-turn — who notices? S2 proved verbalized
self-report fails for importance (7/18 rankable, median 0.0) while
mechanistic reads succeed (S1 0.8286, S4 0.756). LEAD PREDICTION,
REGISTERED PRE-DATA: the same split reproduces for need-detection —
mechanistic detectors win, the verbal ask underperforms, live-query
routing takes the race. If the verbal ask wins, the prediction is
REFUTED and that is the finding.

## Fixtures

From the certified 34-turn session + supersession battery: for each
probe turn, create a PLANTED-MISS variant (the graft the router would
rank-1 for that probe is withheld from admission; everything else
mounts normally) and keep the SERVED control (normal admission).
Minimum 12 planted-miss turns + 12 served controls; enumerate them.

## Detectors (race arms, all measured on identical turns)

- D-LQR live-query routing: during generation, score the model's
  query-fingerprint stream against the FULL repository index (the
  existing fingerprint + router machinery); trigger = an unmounted
  graft outranks every mounted graft for the current query direction.
- D-NGH no-good-home: witness machinery reads arena attention at
  readout positions; trigger = mounted-content mass below a fit-side
  threshold (mass to sinks/live-only), calibrated on served controls.
- D-ENT output-confidence: logit margin/entropy spike at answer
  positions vs served-control distribution (fit-side threshold).
- D-VERB verbal ask (the S2-class baseline): append the registered
  neutral question ("Do you have the information needed to answer?
  YES/NO") in a separate probe pass; trigger = NO. Freeze the exact
  wording before any run.

## Metrics + registered adjudication

Per detector: detection recall on planted misses, false-positive rate
on served controls, trigger latency (earliest token index of signal),
F1. Thresholds for D-NGH/D-ENT chosen fit-side on a disjoint
calibration split (freeze before eval). VIABLE = recall ≥ 0.80 at
FP ≤ 0.10. RACE WINNER = highest F1 among viable detectors.
PREDICTION VERDICT: SUPPORTED iff every mechanistic viable detector
outscores D-VERB's F1 AND D-LQR is the winner; PARTIAL iff mechanistic
beats verbal but D-LQR is not the winner; REFUTED iff D-VERB is the
winner or ties the best mechanistic. Report all four rows regardless.

## Gates

DET-G0: planted misses actually miss (the withheld graft absent from
the arena, receipts) and served controls answer correctly. DET-G1:
byte-identity of production paths with all detector instrumentation
off. DET-G2: calibration/eval split provenance frozen pre-eval.
DET-G3: report with the four-row race table + the prediction verdict
sentence.

Done: gate statuses; the race table verbatim; prediction verdict;
per-detector latency medians; files; anything not done.
