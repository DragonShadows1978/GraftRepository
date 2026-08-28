# ORDER MOE-E1.2-DIAG — teacher-arm mechanism hunt (bug vs real premise failure)

## Grant

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` — edits and CPU
runs AUTHORIZED. Your sandbox has no GPU (proven): build the diagnostic
modes + emit exact GPU resume commands; the lead runs them. READ-ONLY:
same paths as `orders/MOE_E1_NARRATIVE_EXPERT.md`; RT/E1 artifacts are
append-only. New code: extend `scripts/gpt_oss20b_expert_e1.py` with
diag modes or add `scripts/gpt_oss20b_e1_diag.py`. Artifacts:
`artifacts/moe_e1_diag/`. FORBIDDEN: git, subagents, network, changing
any registered gate or eval semantics (this order is diagnosis only; a
fix ships behind a new flag, default off).

## The anomaly (read the receipts first)

`artifacts/moe_e1/eval_e11/narrative_*.json`, lead-read: on ALL 16
HELDOUT windows the teacher arm (2,048-token TRAIN-guide prefix +
512-token window, NLL scored on the 511 window targets) is WORSE than
base — often absurdly: window 11 teacher mean_nll 12.14 (ppl 186,382 ≈
uniform-random over the vocab), windows 2/6/9/13 at mean_nll 8.8–9.1,
while base and expert arms on the SAME targets (sha-matched) sit at
2.4–5.9. The E1.1 chain therefore stopped with "PREMISE FINDING:
prefixes did not improve heldout ppl." A same-domain prefix making a
20B go near-random is not a plausible cognitive result; it is the
signature of a mechanics bug in the NEW eval path at prefix lengths —
position/target misalignment, RoPE/YARN table misuse, sliding-window/
sink mask breakage past 512, or prefix/window concatenation error.
Your job: find the mechanism and prove it either way. The
capture-pairs path (same prefix construction, G1 alignment GREEN) and
the base arm are reference points.

## Diagnostic plan (registered)

D1 per-position NLL profile: teacher arm on windows 11 and 13 (worst)
   + window 15 (mildest, 98 vs 58): dump NLL per target position across
   the 511 targets. Uniform-high ≈ misalignment; degrading-with-
   position ≈ mask/rope; boundary-spike ≈ concatenation.
D2 prediction sanity: at 8 sampled positions per window, dump top-5
   predicted tokens vs the actual target and vs the PREVIOUS/NEXT
   target (an off-by-one/off-by-prefix-length hit on neighbors is the
   smoking gun for index shift).
D3 prefix-content control: teacher arm with (a) a 2,048-token WIKITEXT
   prefix and (b) a 256-token guide prefix on the same windows. Bug in
   prefix mechanics → wikitext prefix explodes identically; real
   content effect → wikitext prefix ≈ base.
D4 known-good cross-check: score the same 2,560-token sequence
   (prefix+window) through the ESTABLISHED long-context path (the port
   machinery `scripts/gpt_oss20b_context_ladder.py` /
   `gpt_oss20b_realtext_ppl_gate.py` use), scoring the last 511
   targets. Disagreement with the E1 eval path localizes the bug to
   the new code.
D5 code audit: diff the eval-gates teacher forward against the
   capture-pairs teacher forward (which passed alignment G1) —
   position ids, mask construction, sink handling, logit gather
   indices. Report the exact divergence if any.

## Registered outcomes

- MECHANISM=BUG: name the defect (file:line), ship the fix behind
  `--eval-fix e12` (default off; original path preserved for
  receipts), emit resume commands for lead to re-run G4/G5' under the
  fixed path. No gate definitions change.
- MECHANISM=REAL: prefixes genuinely degrade heldout ppl through the
  known-good path too (D4 confirms) — then the E1.1 premise finding
  stands; say so plainly and stop.
- Either way: one-paragraph mechanism note with receipts.

## Gates

- DG1: D1–D2 artifacts produced for the three named windows.
- DG2: D3 controls run, numbers reported.
- DG3: D4 cross-check run through the established path, delta reported.
- DG4: mechanism verdict (BUG file:line + fix, or REAL) with receipts.
(Your CPU-only portion: build + self-test all modes on synthetic data,
audit D5 statically, emit the GPU script. Gates evaluate after the
lead-run.)

## Constraints

Standing GPU laws for the resume script: flock, ≤590 s, sleep gaps,
single GPU. RED honesty. Evidence class: mechanism diagnosis on one
model. ## Done: DG verdicts (or NOT_MEASURED + resume script path),
D5 audit result verbatim, files created/modified, anything you could
not do.
