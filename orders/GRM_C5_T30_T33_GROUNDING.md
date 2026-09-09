# GRM-C5 — Can t30/t33 close without weakening grounding? (worktree wt/grm-c5, branch grm-c5)

Origin: `docs/GRM_SCOUT_2026-09-08.md` Part C rank 5; the primer's open
item "whether grounding gets a narrower rule for the space-versus-hyphen
class that blocks one recovery in ten". David's goal: test both sides.
Two independent arms, each against its adversarial controls:
**Arm S** a contiguous, ordered value-span match (targets t30);
**Arm N** proper-name binding (targets t33, admits Polaris); and the
naive alternative **Arm W** global whitespace/hyphen folding, which the
Scout predicts FAILS the adversarial controls. The "other side" of each
is the current grounding rule unchanged (Arm 0).

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/wt/grm-c5`.

## Mission
1. Locate the grounding check the batteries use (the acceptance /
   grounding rule in the EB1/SC batteries; name file/lines) and add the
   three candidate rules as additive, flag-OFF alternatives selected by
   an explicit parameter, never by default.
2. Fixture set (CPU-scorable from served text where possible; GPU only
   where the served text must be regenerated): the positive originals
   (t30, t33 and the 8 other long-history probes that currently pass)
   plus adversarial controls per class: changed digit, omitted token,
   swapped relation, negation, scattered-word match, alias collision
   (a different proper name sharing tokens). ≥ 6 controls per class.
3. Registered acceptance: both intended recoveries (t30 under S, t33
   under N) AND zero new false acceptances on the controls; W is the
   registered negative control (prediction: it accepts at least one
   scattered-word or alias-collision control).
4. First run the four arms offline on the EXISTING served texts from
   the EB1/WC1/RT1 receipts (0 GPU); then, only if a rule changes the
   verdict of a probe whose served text is not on disk, register the
   GPU regeneration cells (≤ 0.4 GPU-h).
5. State explicitly in the ledger that a passing rule is a narrower
   grounding rule for a named class, not a prose-grounding certificate.

## Done (verbatim)
1. Rule implementations (file/lines, flag, default-OFF pin test);
   offline arm×fixture table; GPU cells if any + estimates; blocked-report;
   exact lead commands.
2. CPU gate results.
3. Prior art; deviations; RED; process safety; model id and effort.

## COMMON RULES (all Scout follow-up orders)
- Forked from `lc1-wip`. Additive only: new scripts under `scripts/`, new
  registration JSON under `artifacts/<campaign>/`, new tests; no edits to
  existing kernels, batteries, registries, `config/`, or flag defaults.
  Reuse the existing harness modules by import, never by copy-edit.
- Registration IMMUTABLE once written (fixtures, counts, thresholds,
  acceptance bars, predictions), sha-bound; later changes are amendment
  JSONs. Every receipt fingerprints the files its cell executes.
- Your sandbox has NO GPU. Build, CPU-gate (pytest, `--dry-run` enumerating
  every cell with wall estimates), write the blocked-report and exact
  `lead_commands.txt` in dependency order. The lead runs GPU cells through
  the leased runner (`/tmp/forge-gpu.lock`, ≤285 s worker / 590 s outer,
  30 s cooldown; a cell that cannot fit the rail is registered non-fit,
  never retried into a longer lease). Total GPU budget for this order
  ≤0.75 GPU-h unless stated. Never kill or signal any process you did
  not start; never clear a lock.
- Both sides of every decision are cells with equal standing; the order
  registers the prediction, the receipts decide. RED is a result.
- No git (lead commits), no subagents, no background waits, foreground
  only, < 10 min per call. Prior Art Directive at every code site.
- Reasoning effort: high.
