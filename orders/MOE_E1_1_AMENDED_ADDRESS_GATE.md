# ORDER MOE-E1.1 — E1 amendment: re-registered address gate (David's adjudication, 2026-08-27)

## Grant

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` — edits and CPU
runs AUTHORIZED. This order is CPU-ONLY for you (captures already on
disk); run your stages yourself to completion. READ-ONLY: everything
listed in `orders/MOE_E1_NARRATIVE_EXPERT.md` (which remains in force
except as amended below), plus `artifacts/moe_e1/*` capture/key files —
extend, never overwrite receipts. FORBIDDEN: git, subagents, network.

## Context (one paragraph)

E1's G2 fired RED under the original rule: EVAL generic FPR 0.25–0.50.
Lead per-window diagnosis (verified): the violations concentrate in
wikitext windows that ARE narrative-craft text (film themes-and-analysis
88% fire, narrative history 73%), while genuinely out-of-domain windows
sit <1–8% and code/GRM negatives are inside bounds everywhere. David's
adjudication: firing on narrative-adjacent text is IN-REMIT — "that's
what the expert is for." The generic-FPR bound was proxying a safety
property that G5 (behavioral non-interference) measures directly. This
amendment re-registers the gates accordingly. This is a pre-run
re-registration, not a post-hoc widening of a live gate: E1's G2 verdict
under the old rule stands RED in the record.

## Amended registrations (fixed before any rerun)

- G2' address gate: the narrative K4 key must qualify at ≥1 layer with
  EVAL recall ≥ 0.50, code FPR ≤ 0.05, GRM FPR ≤ 0.05 at the frozen
  fit-side τ (τ policy unchanged: feasibility on FIT uses code+GRM
  constraints only now). Generic fire-rate becomes DESCRIPTIVE:
  report it per layer and per eval window, never as a pass/fail bound.
- G5' non-interference (now THE safety gate): with expert mounted and
  gate live — (a) wikitext overall ppl delta vs base ≤ 0.5%
  (registered, pass/fail); (b) code fire rate ≤ 0.05 (registered);
  (c) descriptive: per-window wikitext ppl deltas split by fire rate —
  the high-firing narrative-ish windows (5, 13, 11, 15) reported
  individually, so we see whether the expert helps, hurts, or is
  neutral exactly where it wakes up.
- All other E1 gates (G0, G1, G3, G4) unchanged.
- Install-layer selection: among G2'-qualifying layers, FIT-side rank
  as already implemented.

## Work

1. Amend `scripts/gpt_oss20b_expert_e1.py` minimally to implement the
   G2'/G5' rules (a `--address-rule e11` flag or equivalent; default
   must preserve the original rule so E1's receipts stay reproducible).
2. Run `fit-key` yourself (CPU) under the amended rule on the existing
   captures. Report the install row verbatim.
3. Regenerate `artifacts/moe_e1/GPU_RESUME_COMMANDS.sh` for the
   remaining stages (pair captures from the chosen L\*, train,
   eval-gates incl. the amended G5' reporting, analyze), preserving the
   flock/timeout/sleep discipline exactly.
4. CPU-validate the amended paths (synthetic self-tests as before).

## Done

Final message MUST contain verbatim:
1. G2' verdict with the install row: L\*, recall, code FPR, GRM FPR,
   τ, and the descriptive generic fire rate at L\*.
2. Qualifying-layer list under G2'.
3. Confirmation the original-rule path is preserved (how invoked).
4. Exact paths of every file created or modified.
5. Anything you could not do, stated plainly.
