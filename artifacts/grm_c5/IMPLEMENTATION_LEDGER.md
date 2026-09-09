# GRM-C5 implementation ledger

## 2026-09-09T03:57Z — registration and first offline gate

Evidence classes: source inspection, historical receipt text, CPU finite validation.
Immutable plan: `orders/GRM_C5_T30_T33_GROUNDING.md` (unchanged).
Read `/mnt/Shared/HOUSE_RULES.md` and workspace `AGENTS.md` before implementation.
Worktree path and `.git` pointer / HEAD text identify `grm-c5`; no git command used.

Registration was written before any gate: `registration.json`, SHA-256
`8127ddcf3768b2a30fa789abf5f61d0ba55ef58a8e56d4230838d4fc411c8964`.
Command: `PYTHONDONTWRITEBYTECODE=1 python scripts/grm_c5_register.py` (exit 0).
It freezes 126 fixtures, 504 offline cells, 10 controls per class, acceptance
bars, predictions, CPU suite, mutations and conditional GPU rails. Inputs were
read from the main checkout because the worktree lacks campaign receipts.
84 EB1/WC1 original rows cover all 14 longhorizon probes at six geometries;
SC1 supplies the actual spaced t30 served answer and its unselected demand
trip (separate evidence labels); RT1 supplies 18 original served rows. There
are 20 constructed negative controls and two constructed positive diagnostics.

Additive files: `scripts/grm_c5_rules.py`, `scripts/grm_c5_register.py`,
`scripts/grm_c5_offline.py`, `scripts/grm_c5_pytest_receipt.py`,
`tests/test_grm_c5_grounding.py`; all reports are under `artifacts/grm_c5/`.
Existing batteries, kernels, registries, configs and defaults are untouched.
Production grounding and value scoring are reused by import.

Gate commands (both foreground, CUDA hidden):

```
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python scripts/grm_c5_offline.py --dry-run --output artifacts/grm_c5/dry_run.json
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python scripts/grm_c5_offline.py --offline --output artifacts/grm_c5/offline.json
```

Dry-run exit 0: 507 cells, including every arm/fixture and three CPU gates;
GPU estimate 0 h. Offline exit **2 (scientific RED)**: 432 scored cells,
72 blocked cells (18 RT1 rows lack exact mounted text). RT1 value-hit
comparisons ran; grounding remains null, never a fabricated reject or pass.
Per-cell verdicts: `offline.json`; full table: `offline.md`.

S rescues `SC1:t30` and its recorded demand-trip answer, `Cobalt 1 India`.
N does not rescue `EB1:t33`: actual served text is
`Not in memory: no stored record matches polaris, mark.`; mounted set empty,
`infer_calls=0`, `abstain_reason=identifier_unbound`. Source node 35 is in
the recorded excluded-live set. Both the existing candidate binder and N
already bind that source if it is supplied as eligible. This is evidence of
an eligibility/admission boundary, not a new proper-name grounding recovery.
WC1 width 64's t33 text `Marble-4-Juliet` already grounds under all arms.

S and N each introduce 0/20 new false acceptances and no observed original
grounding regression. Arm 0 already falsely accepts four N-class controls
(swapped relation, negation, alias collision, source negation); additive
arms preserve these defects. W introduces 12 new false acceptances, including
both classes' scattered-word controls and the S-class alias collision.
The negative-control prediction hits. Combined recovery acceptance stays RED.

**A passing rule is a narrower grounding rule for a named class, not a
prose-grounding certificate.** This author-run finite corpus is not blind
adversarial verification; the lead owns independent verification.

GPU decision: all selected served answers are on disk, including failed t33.
The registered missing-served-text condition is not met. No GPU cells,
0 GPU-h used or estimated. Missing RT1 mounted text is not missing output;
regenerating would replace the historical experiment rather than rescore it.

Prior art: local SC1.1 glyph projection/text arena, DET1 comparator, ADM1/RT1
binding, EB1/WC1 receipts and Scout adversarial categories (GRM, 2026) are
borrowed unchanged or imported. External leads are **unverified — lead to
check**: Thompson (1968), regular expression search; Codd (1970), relational
keys; Pearl (2000), controlled interventions; Gardner et al. (2020), contrast
sets; DeMillo, Lipton and Sayward (1978), mutation testing. Borrowed mechanisms
are exact spans, key binding, paired controls and deliberate defects; ours
is the restricted clause/value grammar, fixture composition and receipt
adapter. No prior art known to me for this exact composition; no novelty
claim. These annotations also appear at code sites and in the final report.

Deviations/findings: t30 spacing originates in SC1, while EB1/WC1 t30 is
already hyphenated and passing; included both instead of conflating them.
t33 admission eligibility prevents a grounding-only recovery. RT1 rows are
explicitly blocked for historical mount-text fidelity. No acceptance bars
were weakened. No plan or registration edits.

Process safety: foreground CPU only; no GPU/model loads, network, git
commands, subagents, background waits, live-service actions, process signals,
lock acquisition or lock clearing. CPU torch import availability was checked
without initializing CUDA. Author: GPT-6 (Codex); exact backend model ID is
not exposed to this seat. Reasoning effort: high. Receipt model:
openai/gpt-oss-20b revision 6cee5e81ee83917806bbde320786a8fb61efebee,
TensorCUDA resident_packed_mxfp4; this task did not load it.

## 2026-09-09T04:00Z — CPU receipt hook failure (RED, mechanical)

Command: `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python scripts/grm_c5_offline.py --cpu-gate --output artifacts/grm_c5/cpu_gate.json`.
The 520 test progress markers completed but pytest exited 1 during the receipt
hook, so this run is not reported as a passing gate. Exact failure:
`FileNotFoundError: [Errno 2] No such file or directory: '/mnt/ForgeRealm/wt/grm-c5/_classes.py'`.
Full transcript: `cpu_gate.log`; failure receipt: `cpu_gate_failure.json`.
The dynamic module's synthetic `__file__` is not an executed local file.
Fixed only receipt discovery: hash existing local source files and explicitly
list unresolved module metadata. Added a fallback failed-driver receipt if a
future pytest hook fails. No rule, fixture, threshold or registration change.
The failed hook could not fingerprint all its imports; this limit is explicit
in its failure receipt. The new successful run must emit the complete inventory.

## Final CPU gates and handoff

CPU repeat command: `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python scripts/grm_c5_offline.py --cpu-gate --output artifacts/grm_c5/cpu_gate_r2.json`.
Exit 0: **520 passed in 5.40 s**. `cpu_gate_r2.json` fingerprints 30 executed
local sources and explicitly lists synthetic `torch.ops` (`_ops.py`) and
`torch.classes` (`_classes.py`) metadata. Full log: `cpu_gate_r2.log`.

Mutation command: `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python scripts/grm_c5_offline.py --mutations --baseline artifacts/grm_c5/cpu_gate_r2.json --output artifacts/grm_c5/mutations.json`.
Exit 0: **5/5 non-error mutants killed**, fraction 1.0 against registered 0.8.
Mutants: default_S, accept_all, S_uses_N, W_uses_baseline, N_uses_baseline.
Only owned temporary copies under /tmp were mutated and removed. Receipt
contains each copy's hash and exact source replacement; rules remained intact.
Prior art: DeMillo, Lipton and Sayward (1978), mutation testing,
**unverified — lead to check**. Borrowed deliberate defects; ours these five
registered mutations and consequence checks. No blind verification claimed.

After the bookkeeping fix, repeated only the small offline/dry-run cells to
bind the final executable bytes: `offline_r2.json` / `offline_r2.md` (exit 2,
RED) and `dry_run_r2.json` (exit 0). Rows and gates are exactly equal to the
first offline receipt. Source rule implementation did not change. Dry-run
estimates 145.2 s across 507 cells and 0 GPU-h.

Integrity command: `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python scripts/grm_c5_offline.py --integrity --output artifacts/grm_c5/integrity.json`.
Exit 0: **623 pre-existing files unchanged**; registration/input hashes valid.

Final report: `REPORT.md`; machine blocker handoff: `blocked_report.json`;
exact dependency-ordered CPU commands: `lead_commands.txt`. No GPU commands
because the registered condition is false. S's finite named-class bar passes;
N's intended served recovery and full RT1 coverage remain RED. A passing rule
is a narrower grounding rule for a named class, not a prose-grounding certificate.

Handoff verification: shell syntax for `lead_commands.txt` passed; all final report file links resolve; every source fingerprint in the six final successful receipts matches current bytes. Fixture/cell counts are 126/504. Final SHA256SUMS covers campaign files and the five new source/test files; it excludes itself. No additional scientific or GPU runs.
