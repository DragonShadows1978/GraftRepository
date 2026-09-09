# GRM-C4 amendment5 — NON_FIT; CPU preparation gates PASS

The requested all-subsegments-below200s split is impossible on the measured
turn boundary: turn90 alone took225.512218956 s. No below200 estimate has been
invented. The full untrimmed projection also exceeds5400 s. Known oversized
halves and budget overflow are registered NON_FIT; no real workers ran here.

## 1. Amendment, split, units, files and tests

Amendment: [amendment_a7.json](amendment_a7.json),36076 bytes.
SHA256: `e3ab21e4974a11b2d8499233b220ed02dbd2778b9e2023ca1272f1ea416a6d65`.
Bound order: `orders/GRM_C4_AMENDMENT_5.md`,
SHA256 `8eb729c2454a1ee1da1f43ff9a288472c74541720911623455037a4f9646f0e4`.
Bound predecessor: `lead_a4/amendment_a6.json`,
SHA256 `f28963404f0e0498d11cbc8368c83846eb0b33945df95583ddb301174425359d`.
Immutable pre-registration and before.json precede the gates. No registered
source changed after sealing; the first full suite passed.

Evidence class: lead E2E receipts and bound instrumentation. A2 lh06..11
recorded74.773,78.853,81.915,190.940,84.733,256.319 s. lh12 records306.071 s
against the285 s rail, with all turns88..95 completed, then RED. Its probe90
is the indivisible bottleneck; this is not a uniformly expensive eight-turn
segment. Probe70 and80 took115.313 and178.325 s. Non-turn overhead rose from
30.500 to38.756 s; the planning allowance is ceil39 s.

| New unit | Turns, start inclusive / stop exclusive | Estimate seconds | Planning status |
| --- | --- | ---: | --- |
| c4-lh-12a | [88,92) | 284 | NON_FIT: exceeds200 target |
| c4-lh-12b | [92,96) | 63 | FIT estimate; requires12a PASS |
| c4-lh-13a | [96,100) | 66 | FIT estimate; requires12b PASS |
| c4-lh-13b | [100,104) | 348 | NON_FIT: exceeds200 target and285 rail |

Evidence class: reasoning/planning estimates, not new measurements. lh12 uses
observed turn sums plus39 s. lh13 uses the maximum observed ordinary-turn wall
and projects probe100 as probe90 plus the largest observed adjacent probe
increase at70/80/90. Exact inputs and formula are bound in amendment.plan.
Halves are identical for all three cells. Resume starts from bound PASS lh11,
never the old RED lh12 state; each successor copies and validates bound state.
No fixture, probe, scorer, geometry or acceptance threshold changed.

Unit counts: c64_w96 recovery4; c96_w64 full23; c64_w64 full23; total50 new
units. Each full cell remains4sup+4census+15LH, all104 turns, all14LH probes.
The complete score cites23 selected worker receipts, including19 historical
PASS units for c64_w96. No partial score is presented as a battery score.

Budget: recorded2157.6785489856265 s (including both historical RED walls)
+ registered5741 s =7898.678548985627 s. Overflow2498.6785489856265 s.
Remaining-cell estimates update lh06..11 to ceil observed wall; other existing
prefix estimates remain inherited and unvalidated for these geometries.
Projection retains every requested unit, including NON_FIT units, without
reallocating hypothetical savings. The per-admission285 s reservation marks
24 units as cap NON_FIT: c96_w64/lh13b and all23 c64_w64 units. c64_w64 is
independent of prior scores but remains blocked by its registered cap status.
The planning-status rejection also blocks12a and13b in each cell. Dependencies
then prevent later halves from advancing. No whole-cell GPU completion claimed.

Files/lines (new files; earlier entrypoints and sources byte-unchanged):

- `scripts/grm_c4_split_a5.py:39`: bound timing evidence; `:58` half-range
  estimates/full projection; `:115` order/predecessor/source closure;
  `:131` runtime binding; `:156` receipt checks; `:189` independent-cell readiness;
  `:213` accounting; `:243` shard adapter; `:252` bound seed copy;
  `:266` no-GPU NON_FIT receipt; `:279` admission; `:304` worker;
  `:362` unchanged scorer arithmetic with complete-chain gating;
  `:450` partial progress; `:471` score validation; `:485` cross summary;
  `:575` NON_FIT rendering; `:591` exact command generation.
- `scripts/grm_c4_register_a5.py:13`: create-only amendment and command sealing.
- `tests/test_grm_c4_split_a5.py:135`: real schedule/estimate/command gate;
  `:161` recomputed-sidecar forgery attacks; `:177` stale source;
  `:189` independent starts; `:195` all50 synthetic units, legacy seed,104turn
  resume and full scoring; `:216` state/turn/link/geometry/source/serving attacks;
  `:243` no-GPU NON_FIT; `:253` budget/dependency rejection;
  `:261` partial summary; `:276` forged/stale summary; `:292` accounting;
  `:307` all2567 existing run files byte-unchanged.

CPU author suite: **181 passed,2 warnings in40.66 s**, exit0;35 new A5 cases
and146 unchanged prior cases. See [cpu_gate.txt](cpu_gate.txt). SWIG deprecation
warnings, plus an exit-time swigvarlink warning. No blind-verifier claim.
The synthetic layout explicitly admits cheap fake turns to exercise resume;
this does not authorize the real NON_FIT units. Real-plan refusal is separately
asserted before lease acquisition. All prior suites passed without edits.

Executed foreground CPU commands (no lead GPU chain execution):

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python scripts/grm_c4_register_a5.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m pytest -v -p no:cacheprovider tests/test_grm_c4_split_a5.py tests/test_grm_c4_cap_a4.py tests/test_grm_c4_remaining_a3.py tests/test_grm_c4_resume_a2.py tests/test_grm_c4_campaign.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python scripts/grm_c4_split_a5.py --dry-run
bash -n artifacts/grm_c4/lead_commands_a5.txt
```

All exit0. [validation.json](validation.json) records unchanged2567-file set,
all2567 hashes checked by the suite, final44 receipt/claim hashes checked again,
zero new real worker receipts, and the effective amendment/command pins.
[dry_run.json](dry_run.json) enumerates all50 units and explicit overflow.

## 2. Exact lead commands

[lead_commands_a5.txt](../lead_commands_a5.txt) is the exact sealed command list.
It contains the four wide-cell halves and score, then both full23-unit lists
and scores, then cross summary. New entrypoint: `grm_c4_split_a5.py`.

```bash
bash artifacts/grm_c4/lead_commands_a5.txt
```

This command has NOT been run by this seat. It runs admitted workers only;
registered/dependency/budget NON_FIT units create explicit zero-work receipts
and return normally, so a partial wide cell does not prevent the other cells
from being evaluated for admission. Integrity failures stop. An executed worker
that raises still stops the foreground shell; a hard timeout can leave an
orphan claim, which blocks future work. No error suppression, retry or waiver.
Per-unit lease285, outer timeout590, cooldown30 remain inherited. These timeout
commands are lead deliverables only; this seat sent no process signals.

## 3. Prior art

Verified local C4/A2/A3/A4, EB1 and DET1 (house,2026): reused SHA source closure,
immutable create-only registration/claims, saved-state copy/resume, observed
serving guards, lease reservation and original battery scoring. Added ordered
halves, explicit planning rejection, independent-cell dispatch and partial
summary assembly. No novel algorithm claimed. The specific timing extrapolation
heuristic: **no prior art known to me**. Annotations also appear at code sites
and in the ledger. No external literature claim. SHA closure is integrity
checking, not signature authentication against writers of the trusted validator.

Deviations: the requested <200 s target is not attainable by turn-range splits
on the measured turn90; registered as NON_FIT instead of fabricating estimates.
A5 is an explicit successor to the A3 runner, preserving old files/source pins;
old A3 commands retain their old ordering guard and are superseded by the new
command list. lh06..11 estimates refreshed from measurements for honest cap
accounting. No intra-turn optimization or changed model/runtime behavior.

RED: `AssertionError: Worker rail exceeded: NON_FIT` remains byte-unchanged
and charged at306.07129970006645 s, not285 s. The earlier
`AssertionError: No serving observed` remains charged too. Lead-reported
`Prior cell score missing` is addressed in the A5 successor and CPU-tested.
**Not claimed fixed:** the indivisible turn bottleneck, cap overflow, real
GPU completion, model quality, and the cross prediction. Full battery remains
NON_FIT; partial aggregation reports INCONCLUSIVE.

Process safety: no git, subagents, GPU execution, shell background jobs/waits,
process signals, process kills, service/lock changes or writes outside the
worktree. Pytest was one foreground process, collected through the tool after
it yielded. Existing2567 run files and previous sources/orders/manifests remain
unchanged. No queued GPU chain executed.

Model id: GPT-6 (Codex; exact deployment identifier not exposed).
Reasoning effort: high (requested).
