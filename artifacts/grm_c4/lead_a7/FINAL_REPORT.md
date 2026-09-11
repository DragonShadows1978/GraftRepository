# GRM-C4 amendment 7 final report

Accounting crash FIXED in CPU reproduction and live read-only admission.
34 A7 author tests PASS; legacy suite is RED: 210 PASS, two obsolete snapshot
assertions FAIL. No GPU execution or new GPU charge. A6 plan, six skips,
6,600 s cap and 6,002.678548985627 s projection are unchanged.

## 1. Root cause, fix, regression and unchanged projection

Evidence class: before-fix reproduction on actual disk, followed by CPU unit
comparison on an exact copy of the top-level accounting scan layout.
`reproduction_before.txt` contains the original exception verbatim:

```text
File "/mnt/ForgeRealm/wt/grm-c4/scripts/grm_c4_split_a5.py", line 229, in accounting
    if row['status']=='NON_FIT':
KeyError: 'status'
OFFENDING_FILE: /mnt/ForgeRealm/wt/grm-c4/artifacts/grm_c4/runs/c64_w96/runtime_frame.json
QUOTE: '  "schema": "grm.det1.runtime_frame.v1",'
```

The offending file is runtime metadata with no `status`. The same metadata
filename is present at `artifacts/grm_c4/lead_a2/runs/c64_w96/runtime_frame.json`.
The old `*/*.json` scan admitted both. Its attempt/score exclusion already
existed but ran after JSON parsing. The lead's attempt/partial-file hypothesis
was not the actual cause of this exception.

Fix: `scripts/grm_c4_split_a5.py:214` (accounting), filename exclusions at
`:229`, structured stderr diagnostics at `:226`, incomplete-worker handling
at `:240`, original historical pin checks at `:259`, completed wall checks at
`:267`, unmatched-claim blocking at `:271`, unchanged reserve/cooldown at `:278`.
Only completed PASS/RED workers contribute wall time. Attempts, scores and
non-worker filenames are excluded before JSON parsing. Valid zero-work NON_FIT
records are validated and excluded from charged work. Each excluded file emits
its path and reason; `accounting_skips.jsonl` captures three live read-only scans.
Malformed/incomplete worker-shaped files are diagnosed and excluded from the
sum, then block admission. A missing or incomplete claim counterpart also blocks;
exclusion cannot silently discard possible GPU work and open a lease.

A7 source closure: new `scripts/grm_c4_accounting_a7.py:31` verifies an explicit
old-to-new pair for A5 and A6 source bytes. `:61` returns the historical identity
pin only after verifying the current replacement bytes. A5 source comparison
at `scripts/grm_c4_split_a5.py:125` and A6 comparisons at
`scripts/grm_c4_skip_a6.py:87` use this helper. A6 identities are retained under
the separate A7 authorization, following A4's replacement pattern. No global
hash substitution or predecessor manifest rewrite. Drift/forgery gates still run.

Create-only registration, sealed before CPU gates:
`artifacts/grm_c4/lead_a7/amendment_a9.json`, 3,952 bytes, SHA256
`0d6c656ee51e564f2013f2f5dbaececa33d4e58c216fcacfc410796e8487cbc8`.
The immutable A7 order and pre_registration.json precede all gates. Original A5
and A6 source snapshots are saved as `grm_c4_split_a5.py.before` and
`grm_c4_skip_a6.py.before`. No sealed source or registration changed after gates.

Regression: `tests/test_grm_c4_accounting_a7.py:59`,
`test_exact_crashed_disk_layout_keyerror_then_completed_receipts`.
The fixture at `:25` copies the actual top-level JSON bytes and relative layout
from `runs`, `lead_a2/runs`, `lead_a3/runs`, and `lead_a5/runs`, including both
runtime frames, paired attempts, historical REDs and the five lead CPU outputs.
It extracts the original accounting function from the byte-pinned source
snapshot, asserts the exact `KeyError` and offending path, then verifies the
fixed function charges 2,157.6785489856265 s and emits exclusion reasons.
The first unstarted worker reaches readiness after accounting; the test raises
a sentinel there, before lease import/acquisition. This covers the accounting
scan, not nested session/model state or GPU execution. These campaign preflight
tests use the present receipt snapshot; they do not assert future run completion.

Other A7 cases cover exclusion before reads, malformed/partial worker JSON with
and without claims, invalid/negative/nonfinite wall values, exact A6 plan/budget/
command identity, historical artifacts, and forged/stale replacement manifests.
Existing A5/A6 tests still exercise reserve285, cooldown30, orphan claims,
historical drift, skip semantics, dependencies, partial scores and fake leases.

Evidence class: immutable planning projection plus historical completed E2E wall
receipts; this is not a new timing measurement or completion claim.

| Quantity | Seconds |
| --- | ---: |
| Recorded completed walls, including both REDs | 2,157.6785489856265 |
| All 44 fitting unit estimates | 3,845 |
| Projection | 6,002.678548985627 |
| Cap | 6,600 |
| Headroom | 597.3214510143735 |
| Peak projected reservation | 6,221.678548985627 |
| Six excluded NON_FIT estimates | 1,896 |
| New GPU work/charge in this amendment | 0 |

Never-attempted list unchanged: `c4-lh-12a` and `c4-lh-13b` in each of
`c64_w96`, `c96_w64`, `c64_w64`. The six fitting but dependency-blocked lh12b/
lh13a segments remain in the projection. No new lease, retry, state bypass,
budget, scoring rule or timing heuristic is introduced.

## 2. Exact lead commands and current-state continuation

`artifacts/grm_c4/lead_commands_a7.txt` is byte-for-byte identical to A6:
105 lines, 6,699 bytes, SHA256
`70b9935128af97766e6f4d6b86a62b40a44ac147bcfb937d9e891b53667382bb`.
Both the full file and the continuation below pass `bash -n`. Neither was run.
The full list retains all six skip commands, 44 fitting dispatches, three scores
and summary, including all original sleep, timeout and lease parameters.

Current disk already contains four NON_FIT outputs and a partial score for
c64_w96 from the failed lead invocation. Restarting the full file at line 8
would correctly fail `No retries or overwrite`. None of those outputs was
removed, rewritten or retried. The first unstarted worker is full-file line 15:

```bash
timeout --signal=KILL 590s python scripts/grm_c4_skip_a6.py worker --cell c96_w64 --battery sup --spec correction_then_restatement
```

A separate `artifacts/grm_c4/lead_resume_a7.txt` contains exactly full-file
lines 1–7 plus 15–end. It retains preflight, `set -e`, cwd, initial cooldown and
every remaining command. It omits only the already-completed CPU commands and
their sleeps. `validation.json` pins both files and verifies the selection.
For the currently inspected disk state, the exact lead invocation is:

```bash
bash /mnt/ForgeRealm/wt/grm-c4/artifacts/grm_c4/lead_resume_a7.txt
```

This continuation is derived from unchanged lead_commands_a7.txt; it is not an
authorization to retry workers. The first unstarted receipt/attempt was absent
and live read-only admission returned None (accepted) in validation.json.
If another lead invocation runs meanwhile, this recorded resume point is stale;
existing create-only guards remain in effect.

## Tests, receipts and RED

- A7: 34 passed, two Swig deprecation warnings, 18.87 s; exit 0,
  `cpu_gate_a7.txt`.
- Legacy: 210 passed, two failed, two Swig deprecation warnings, 54.35 s;
  exit 1, `cpu_gate_legacy.txt`. Both test processes also emitted the existing
  swigvarlink deprecation warning at interpreter exit.
- CPU CLI dry-run: exit 0, `dry_run.json`; live read-only accounting,
  reserve/cooldown and first-worker admission: exit 0, `validation_command.txt`
  and `validation.json`.

Legacy failures retained verbatim in the log and not weakened:

```text
FAILED tests/test_grm_c4_skip_a6.py::test_existing_artifacts_byte_unchanged_and_no_real_attempts
    assert all(c.record(p['path'])==p for p in pins)
E   assert False
FAILED tests/test_grm_c4_split_a5.py::test_real_existing_bytes_unchanged
    assert not (a5.OUT/'runs').exists()
E   AssertionError: assert not True
```

The first snapshot's sole drift is the A7-authorized A5 source replacement.
The second assumes the lead has never created A5-root outputs; five CPU files
now exist from the reported failed invocation. The original A5 before.json's
2,567 files still match. Neither old assertion is claimed fixed. The A7 gate
checks current historical bytes and explicit replacements instead. No skipped
tests, relaxed assertions, tolerance changes or repeat-to-green run.

At pre-report validation, 2,736 of 2,738 captured files were byte-unchanged;
the only differences were the two explicitly amended source files. All existing
receipts, claims, registrations and command files were unchanged. After reporting,
NARRATIVE_SYNTHESIS.md receives an append-only A7 entry; final_integrity.json
checks its original prefix and all other captured bytes. No new real run root
or claim was created. Both old rail REDs remain charged; full long-history
batteries remain NON_FIT and the cross-result remains uncompleted/INCONCLUSIVE.
GPU completion, model quality and blind verification are not claimed fixed.

Exact test commands were foreground and sequential:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python scripts/grm_c4_accounting_a7.py --register
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m pytest -v -p no:cacheprovider tests/test_grm_c4_accounting_a7.py > artifacts/grm_c4/lead_a7/cpu_gate_a7.txt 2>&1
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m pytest -v -p no:cacheprovider tests/test_grm_c4_skip_a6.py tests/test_grm_c4_split_a5.py tests/test_grm_c4_cap_a4.py tests/test_grm_c4_remaining_a3.py tests/test_grm_c4_resume_a2.py tests/test_grm_c4_campaign.py > artifacts/grm_c4/lead_a7/cpu_gate_legacy.txt 2>&1
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python scripts/grm_c4_skip_a6.py --dry-run > artifacts/grm_c4/lead_a7/dry_run.json
bash -n artifacts/grm_c4/lead_commands_a7.txt
bash -n artifacts/grm_c4/lead_resume_a7.txt
```

## Prior art

Local C4/A4/A5/A6 and DET1 (house, 2026), verified directly in
`scripts/grm_c4_cap_a4.py`, `scripts/grm_c4_split_a5.py` and
`scripts/grm_c4_skip_a6.py`: reused exact source replacement manifests,
completed-worker wall accounting, claim guards, reserves/cooldowns,
create-only receipts, dependencies and unchanged complete-battery scoring.
A7 adds filename filtering, per-file diagnostics and the exact-layout
original/fixed regression comparison; no novel algorithm is claimed.
No external literature result is introduced. Specific inherited A5 timing
heuristic: no prior art known to me; unchanged and not reimplemented here.
Annotations appear at code sites, in pre-registration and in the ledger.

## Deviations, process safety, model and effort

The source-replacement helper is necessary to keep the frozen A5/A6 source
manifests valid while fixing their live executor. The extra continuation file
is necessary to use the mandated identical command list after the lead's five
successful CPU writes. Neither changes authorization or no-retry semantics.
Partial worker evidence remains a blocking integrity fault; only irrelevant
metadata and validated zero-work records are harmless exclusions.

No git commands, subagents, GPU execution, shell background jobs, background
waits, process kills/signals, lock changes or live service changes. Test leases
were mocked; real admission scans were read-only. No process was killed.
All lead GPU commands remain unexecuted by this seat. No GPU time is charged.

Model: GPT-6 (Codex; exact deployment identifier not exposed).
Effort: high (requested). Author baseline only; lead blind review is pending.
