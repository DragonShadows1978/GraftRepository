# GRM-C4 amendment 6 final report

CPU preparation PASS; full long-history batteries remain NON_FIT. No GPU work
ran. The six named registered NON_FIT units are explicit CPU skips, never
attempted or leased. The cap is 6,600 s. All 44 fitting units remain dispatched,
but six cannot execute because their saved-state predecessor is skipped.
This limitation is not claimed fixed and no partial score becomes a battery score.

## Amendment and projection

Immutable order: `orders/GRM_C4_AMENDMENT_6.md`, SHA256
`f003ef8c6b3f40bced73c5d753b94961bdbd4065d55d7ab8572313d3a01df8f0`.
New create-only registration: `artifacts/grm_c4/lead_a6/amendment_a8.json`,
42,570 bytes, SHA256
`4343189f992408e57797dfc0ebd13dc6df2a7e086e28d874823bdaa546f2c78e`.
Its predecessor is unchanged A5 `lead_a5/amendment_a7.json`, SHA256
`e3ab21e4974a11b2d8499233b220ed02dbd2778b9e2023ca1272f1ea416a6d65`.
The source/tests/commands and pre-registration gates were pinned before tests.
No registration or source was edited after sealing.

Evidence class: recorded E2E wall receipts plus inherited planning estimates;
this is a budget projection, not an execution or completion measurement.

| Quantity | Seconds |
| --- | ---: |
| Recorded walls, including both historic REDs | 2,157.6785489856265 |
| All 44 fitting unit estimates | 3,845 |
| Recorded + fitting | 6,002.678548985627 |
| Authorized cap | 6,600 |
| Headroom | 597.3214510143735 |
| Peak projected admission including 285 s reservation | 6,221.678548985627 |
| Excluded six NON_FIT estimates | 1,896 |

No hypothetical 285 s charge is added for a skipped unit. Actual admissions
still reserve 285 s, use a 285 s lease, wait zero for the lock, and retain the
30 s cooldown and 590 s outer bound. Cooldowns are outside GPU accounting.
Actual elapsed walls can differ from estimates; the cap still stops admission.
Inherited estimate metadata is unchanged; `plan.projection` is the A6 projection.

## Skip list and dependency residual

| Cell | Never-attempted units (estimates) | Fitting dispatches | Fitting estimate sum |
| --- | --- | ---: | ---: |
| c64_w96 | c4-lh-12a (284 s), c4-lh-13b (348 s) | 2 | 129 s |
| c96_w64 | c4-lh-12a (284 s), c4-lh-13b (348 s) | 21 | 1,858 s |
| c64_w64 | c4-lh-12a (284 s), c4-lh-13b (348 s) | 21 | 1,858 s |

A5 classified lh12a under its <200 s planning target: its 284 s estimate does
not exceed the 285 s lease. A6 preserves the lead's explicitly named skips and
that existing classification. No new timing measurement or heuristic is added.

Within each cell lh12b (63 s) and lh13a (66 s) depend on skipped lh12a state.
Unchanged A5 admission therefore produces dependency NON_FIT, without an attempt
or lease, for those six fitting segments. Their combined 387 s remains included
in the conservative 3,845 s fitting projection. There are 38 reachable new GPU
workers if every preceding reachable unit passes, all in the two unstarted cells.
The command list retains all 50 units: six explicit skips and 44 fitting worker
dispatches. It neither bypasses a gap nor promotes the old RED state into PASS.
The recorded lh12 failure is `AssertionError: Worker rail exceeded: NON_FIT`;
its actual charged wall was 306.07129970 s under the 285 s rail (A5 receipt).

Each partial cell summary remains NON_FIT, lists completed segments, and has no
battery score. The synthetic successful reachable chain lists 19 completed units
per cell (4 sup, 4 census, 11 LH), four NON_FIT segments, and no missing units.
The cross verdict is INCONCLUSIVE and prediction_met is null. These are CPU
synthetic results, not observations from new real workers.

## Files and lines

- `scripts/grm_c4_skip_a6.py:38`: fitting-only projection, skip/dependency lists.
- `scripts/grm_c4_skip_a6.py:76`: exact order, predecessor and source closure;
  `:92` binds the new budget and identity at runtime.
- `scripts/grm_c4_skip_a6.py:103`: refuses executed/charged/forged skip receipts.
- `scripts/grm_c4_skip_a6.py:117`: adapts the unchanged A5 executor in process.
- `scripts/grm_c4_skip_a6.py:124`: six CPU-only skip commands plus fitting workers.
- `scripts/grm_c4_skip_a6.py:144`: CLI preflight, skip/worker, score and summary.
- `scripts/grm_c4_register_a6.py:13`: create-only registration before execution.
- `tests/test_grm_c4_skip_a6.py:30`: projection, membership and command checks;
  `:71` and `:86` forgery/staleness; `:98` skip semantics; `:114` all-unit
  synthetic dispatch and partial summary; `:143` and `:157` forged receipts/
  partial scores; `:167` exact reserve boundary; `:176` immutable old bytes.
- `artifacts/grm_c4/lead_a6/`: pre_registration.json, before.json,
  amendment_a8.json/.sha256, IMPLEMENTATION_LEDGER.md, dry_run.json,
  cpu_gate_a6.txt, cpu_gate_legacy.txt, validation.json, this report, SHA256SUMS.
- `artifacts/grm_c4/NARRATIVE_SYNTHESIS.md`: appended A6 result to the same synthesis.
- `artifacts/grm_c4/lead_commands_a6.txt:1`: exact new lead command list.

All pre-A6 code, orders, registrations, receipts and command files remain
byte-unchanged. New worker receipts use the existing `lead_a5/runs` layout,
with A6 amendment/source/budget identities. Use the A6 entrypoint for this run;
historical entrypoints retain their historical authorization.

## CPU tests and integrity receipts

31 new cases PASS in 21.86 s (`cpu_gate_a6.txt`); 181 unchanged prior cases PASS
in 38.14 s (`cpu_gate_legacy.txt`). Both processes exited 0. Both report two
Swig deprecation warnings, plus a swigvarlink warning at interpreter exit.
These are author baseline tests, not blind verification or GPU evidence.

Tests exercise all six skips with forbidden readiness/accounting/harness calls,
zero fake lease calls, zero elapsed GPU work and no .attempt.json. Forged skip
PASS, charge, attempt, source and reason are refused. Forged manifests with
recomputed sidecars, stale bound files, and forged/stale partial scores refuse.
The all-unit fixture executes the real A5 dependency/score/summary logic with
fake sessions and leases: 38 leases, six dependency refusals, three honest
partial cells. Actual accounting accepts 6,315 s + 285 s reserve and refuses
6,315.001 s + 285 s reserve.

`validation.json` records 2,742 unchanged historical files, including all 2,567
original run files and 44 receipt/claim files. It also records unchanged prior
command files. No real A5/A6 worker root exists. CLI dry-run and `bash -n` passed.

Exact CPU commands (foreground, sequential; logs are under lead_a6):

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python scripts/grm_c4_register_a6.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m pytest -v -p no:cacheprovider tests/test_grm_c4_skip_a6.py > artifacts/grm_c4/lead_a6/cpu_gate_a6.txt 2>&1
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m pytest -v -p no:cacheprovider tests/test_grm_c4_split_a5.py tests/test_grm_c4_cap_a4.py tests/test_grm_c4_remaining_a3.py tests/test_grm_c4_resume_a2.py tests/test_grm_c4_campaign.py > artifacts/grm_c4/lead_a6/cpu_gate_legacy.txt 2>&1
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python scripts/grm_c4_skip_a6.py --dry-run > artifacts/grm_c4/lead_a6/dry_run.json
bash -n artifacts/grm_c4/lead_commands_a6.txt
```

## Exact lead commands

The complete 105-line list is `artifacts/grm_c4/lead_commands_a6.txt`, 6,699 bytes,
SHA256 `70b9935128af97766e6f4d6b86a62b40a44ac147bcfb937d9e891b53667382bb`.
It contains preflight, six CPU skips, 44 fitting-unit dispatches, three score
commands and cross summary, in registered cell order. The script retains
`set -e`; integrity faults stop the chain. NON_FIT returns zero only as an
explicit admission result. After prior foreground work exits, the exact lead
invocation replacing the unqueued A5 list is:

```bash
bash /mnt/ForgeRealm/wt/grm-c4/artifacts/grm_c4/lead_commands_a6.txt
```

This command file was generated and syntax-checked, not executed by this seat.

## Prior art

Local C4/A4/A5 and DET1 (house, 2026): reused SHA manifests and exact source
closure, runtime overlays, create-only admission receipts, saved-state dependencies,
full-worker reserve accounting, original complete-battery scoring and partial
summaries. Verified in local C4 cap/split/accounting/adapter source; the unchanged
CMC1 lease implementation supplies the guarded lease. A6 adds the lead-directed
fitting-only projection and explicit CPU skip dispatch; no novel algorithm is
claimed. For the specific inherited A5 timing extrapolation: no prior art known
to me. No external literature result is introduced or claimed verified. Matching
annotations are at code sites, in pre-registration and in the ledger.

## Deviations, RED, safety and model

No scope or policy expansion. A small additive adapter invokes A5 without editing
its frozen source. The literal goal that every fitting unit executes cannot be
met for six dependent segments under unchanged resume rules; this is explicitly
registered and tested rather than silently bypassed. All three LH batteries
remain NON_FIT and the existing rail RED is not claimed fixed.

No git commands, subagents, GPU execution, shell background jobs, background
waits, process kills/signals, lock changes or live service changes. CPU tests ran
sequentially in foreground; their leases and sessions were synthetic. Lead GPU
commands remain unexecuted. No real receipts were created or rewritten.

Model: GPT-6 (Codex; exact deployment identifier not exposed). Effort: high
(requested). No model identity beyond the exposed system context is asserted.
