# GRM-X3 amendment 2 — r2 handoff

CPU PASS; experimental RED / GPU_BLOCKED. No r2 GPU cell was started. The X3
thesis is unproven. r1 remains RED_STRATA_OR_INCOMPLETE; its P1 failure and
unevaluable predictions are not re-scored, re-thresholded or reinterpreted.

## Frozen inputs (reasoning / preregistration)

- Order: `orders/GRM_X3_AMENDMENT_2.md`, SHA256 `ae126b0456ea1a538c6ff0cb0dd9f1c5055bc468363a2eeb9577cfeca0fef844`; read-only, unchanged.
- r2 registration: `artifacts/grm_x3/r2/registration.json`, SHA256 `99abd4dfbed92df335135cdb08f5180bba32a6cbfa8ff2e37b8d1190d236af33`.
- Recipe manifest: `artifacts/grm_x3/r2/fixtures/manifest.json`, SHA256 `b848cf7f1af0c9cf2816acaa8dd253394e2837871677574361e64b9cc38acce6`.
- Twenty NEW recipes: `fixtures/x3_r2_00.json` through `x3_r2_19.json`; 8 correct, 6 decoy, 6 refusal by intent. Disjoint from r1 in entity, value and question. Recipes are not GPU captures.
- Implementation manifest: 84 current file pins, SHA256 `a5e13f8337d305a6d6b5e380888d3a6a7c21f7d31d4ce06684f63b6296a94887`.
- Runtime fingerprint: `530dcf174adcfb71b16a7bfba04516b004561911f2afbf43576e7a123494f683`.
- r1 evidence manifest: `r1_evidence_before.json`, SHA256 `783c1cf9a3f83a360601dadfbbd39175536f4f623a414f510c338868e5629e90`; all 5,161 file pins unchanged, including raw runs, both summaries, fixtures, registrations, amendment-1 report/rows and old lead commands. `r1_evidence_verification.json` is the CPU receipt.

Registration and recipe manifest have independent compiled trust anchors in
`scripts/grm_x3_r2_anchor.py`; replacing either JSON and recomputing its own
internal hashes cannot authorize it. Source pins and runtime identity are also
checked before launch. Creation is exclusive; no re-freeze/retry operation exists.

## Scorer and decisions

`scripts/grm_x3_r2_scorer.py:24` normalizes NFKC, the registered Unicode Dash and
legacy Hyphen repertoire, collapsed whitespace, then casefold.
`:28` matches one contiguous ordered span with word/hyphen boundaries and a
conservative negation veto. `:43` produces both realization records; ambiguous
competing classes realize none. Refusal uses the six frozen r1 phrases as spans;
only the matched refusal phrase is exempted from the negation veto.

`scorer_controls.json` contains 372 registered cases: 86 positive and 286
negative cases across true and decoy values and refusal phrases. Changed digits,
omitted tokens, swapped/changed relations, prefix/suffix negation and token
boundaries must reject. Tests also cover all 34 registered Dash/Hyphen code points.
GRM-C5 arm S duplication is explicit; no C5 worktree file was read or imported.
Lead reconciliation remains pending.

0.10 nats was chosen from r1 as prior-run calibration. Each r2 sham KL >= 0.10 is a FAIL for that snapshot, never a recalibration; P1-prime fails if fewer than 18/20 pass or same-payload is not byte-identical 20/20.

P2-prime: removal KL >1 nat on >=6 realized correct cases. P3-prime: high mass
on >=4 realized decoys AND lesion accuracy >= mass accuracy on realized decoys.
Q2-prime: removal KL >1 on <=5 realized correct cases. Q3-prime: lesion accuracy
<= mass accuracy across realized cases. P1-prime and Q1-prime share the control
rule above. Guards require all20 receipts, no truncated answers, and realized
counts >=6/8 correct, >=4/6 decoy, >=4/6 refusal. Otherwise predictions P2/P3/Q2/Q3
are null and status RED_STRATA_OR_INCOMPLETE, unless the unchanged kill rule fires.

Explicit interpretation registered before tests: frozen whole-answer exact-error
labels remain classifier targets; value-span scoring changes realization. The
all20 exact-error table/accuracy and a separately labeled value-span error table
are reported beside value-span and frozen exact-equality stratum counts. P3-prime
uses amendment 2's inclusive >= comparison; Q3 retains r1's opposing <= direction.
The original kill rule remains controls comparable on >=3/20 OR no strict overall20
exact-error accuracy improvement, so an overall tie is RED even if P3-prime passes.
A valid-strata P1-prime failure is explicitly RED_P1.

## CPU evidence and lead commands

`logs/grm_x3_r2_cpu_01.txt`: initial r2 baseline 11 passed, 2.91 s.
`logs/grm_x3_r2_cpu_02.txt`: combined r2 and unchanged r1 regression suites,
62 passed, 11.59 s; two SWIG deprecation warnings and a shutdown warning.
Author baseline only, not an independent blind test or new mutation score.
`CPU_GATE_RECEIPT.json` pins the log and CPU execution receipts.

Tests in `tests/test_grm_x3_r2.py`:
- `test_registered_scorer_controls` (line 16), `test_all_registered_dashes` (25).
- `test_r2_registration_rejects_forged_and_stale` (32): forged floor, stale r1 registration, forged manifest.
- `test_r1_evidence_byte_unchanged` (43), `test_r2_disjoint_recipes_and_dry_run` (50).
- `test_r2_summary_guards_and_floor` (77), `test_r2_thresholds_and_unchanged_kill` (102).
- `test_r2_scorer_competing_answers_and_frozen_exact` (115).
- `test_r2_summary_existing_run_layout` (179): full four-cell, 20-capture CPU synthetic run through real DET1 blob/logit validation.
- `test_r2_cli_refuses_stale_fingerprint_before_gpu` (191).

Shell syntax and Python AST checks pass. `dry_run.json` enumerates all four cells
and six arms per recipe. CPU preflight exits 2 with the exact error:
`NO GPU: /dev/nvidia0 absent in dispatched sandbox`. `pending_summary.json`
contains zero observations and null predictions, explicitly pending, not E2E.

Refreshed r2 commands are `artifacts/grm_x3/r2/lead_commands.txt`; old r1 commands
remain byte-identical. Execute foreground and sequentially; stop at any nonzero
exit. Each command pins the reviewed runtime. Exact commands:

```bash
# Run foreground, sequentially; stop on any nonzero exit. No retries.
cd /mnt/ForgeRealm/wt/grm-x3
bash scripts/grm_x3_r2_lead_gpu.sh preflight 530dcf174adcfb71b16a7bfba04516b004561911f2afbf43576e7a123494f683
bash scripts/grm_x3_r2_lead_gpu.sh run cell_00 530dcf174adcfb71b16a7bfba04516b004561911f2afbf43576e7a123494f683
bash scripts/grm_x3_r2_lead_gpu.sh run cell_01 530dcf174adcfb71b16a7bfba04516b004561911f2afbf43576e7a123494f683
bash scripts/grm_x3_r2_lead_gpu.sh run cell_02 530dcf174adcfb71b16a7bfba04516b004561911f2afbf43576e7a123494f683
bash scripts/grm_x3_r2_lead_gpu.sh run cell_03 530dcf174adcfb71b16a7bfba04516b004561911f2afbf43576e7a123494f683
bash scripts/grm_x3_r2_lead_gpu.sh summary 530dcf174adcfb71b16a7bfba04516b004561911f2afbf43576e7a123494f683
```

Reservation: four cells x285 s =1140 GPU seconds (0.3167 GPU-h); both X3 run
reservations total2280 s (0.6333 GPU-h), within the amendment's 0.64 GPU-h.
Per invocation: <=240 s foreground lock wait, <=285 s worker, 30 s cooldown,
590 s outer rail. Typical planning range 180–285 s worker; r1's observed timing
is not an r2 timing guarantee. No r2 launch consumed. Four create-only launches,
no retries; failed/abandoned starts consume their reservation.

## Prior art

Unicode Consortium, UAX #15 (1998 onward): NFKC normalization; primary reference
checked: https://www.unicode.org/reports/tr15/ . Unicode 16.0 PropList (2024)
confirms the Dash and legacy Hyphen repertoire including Garay:
https://www.unicode.org/Public/16.0.0/ucd/PropList.txt . Python provides ordinary
casefold, whitespace splitting and regular expressions.

Jain and Wallace (2019), Attention is not Explanation, motivates distinguishing
attention mass from interventions; primary page checked:
https://aclanthology.org/N19-1357/ . Inherited X3 annotations: Meng et al. (2022),
Locating and Editing Factual Associations in GPT, arXiv:2202.05262, controlled
activation interventions, no weight editing. Kullback and Leibler (1951), On
Information and Sufficiency, standard KL; unverified — lead to check DOI
10.1214/aoms/1177729694. No fresh verification claim for the latter two.

House DET1/S3/S4/D-NGH/RS3/RS4/EB1/RT1/X3 (2026): capture, hydration, fixed-position
mask interventions, mass arithmetic, SHA256 receipts, leased foreground runners
and classifier direction reused. The r2 lead/worker copies isolate r1's artifacts
and semantics. House C5 arm S (2026) shares the specification, independently
implemented here. No prior art known to me for the particular relation recipes,
negation veto, quotas, or r1-calibrated 0.10 floor; no novelty claim. The amendment
supplies thresholds and experiment allocation. Annotations occur at code sites,
in the registration and ledger, and here.

## Deviations, RED and process safety

All changes are scoped X3 scripts/tests/artifacts plus an append to the existing
X3 ledger. New r2 ledger is `IMPLEMENTATION_LEDGER.md`; immutable plan is the
amendment order. r1 bytes remain unchanged. No serving/core code changed.

Registered implementation choices: one completed fact turn per new relational
recipe; token-boundary and whole-answer negation veto; refusal phrase containment;
ambiguous-class veto; exact-error labels retained; P3 inclusive comparison. These
are explicit lexical decisions, not semantic entailment. Relation reversal INSIDE
the registered value is tested; attributing the intact value to a wrong entity,
irony, arbitrary paraphrases and unrelated negation are not solved. Conservative
false negatives remain possible. No model-answer selection, threshold retuning,
C5 import or r1 reinterpretation occurred.

Inherited residuals: controlled direct attempts rather than natural routing;
canonical host value bytes rather than literal device storage; first-token KL
may precede the fact; Python alarms do not hard-bound a hung native call; external
native build required. Not claimed fixed. No new blind verification or mutation
campaign was authorized/launched; lead independent verification remains open.

No git, subagents, GPU calls/leases, background waits, service changes or process
kills/signals were used. All CPU processes completed in foreground. Target model:
openai/gpt-oss-20b revision 6cee5e81ee83917806bbde320786a8fb61efebee; target Harmony reasoning low.
Author: GPT-6 / Codex, exact deployment subvariant unavailable; reasoning high
requested by the order (runtime effort setting not independently exposed).
