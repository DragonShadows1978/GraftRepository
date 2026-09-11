# GRM-C4 amendment 8 final report

A8 recovery is READY FOR LEAD RESUME on CPU evidence: 47 new CPU tests PASS.
Legacy suite retains 5 failures (95 PASS); no GPU execution or actual re-run is
claimed. All 92 original snapshot files are byte-unchanged. The first incomplete
unit is c96_w64 / census / e2e-2. Projection: 5,811.558284636121 s against 6,600 s.

## 1. Classified files (paths, shas); amendment path + sha; files/lines; tests; projection.

Evidence class: exact on-disk bytes, JSON parsing, SHA256, and CPU author gates.
Paths below are relative to `/mnt/ForgeRealm/wt/grm-c4`.

`CORRUPT_DISK_FULL`:

- Original: `artifacts/grm_c4/lead_a5/runs/c96_w64/census_e2e-2.json`
- Bytes: 16,384; SHA256: `28eb9c9dd370c07f00f2e5e10e413f987c1d80bde3d2ab11f67fe96b008b03c6`
- Preserved create-only copy: `artifacts/grm_c4/lead_a8/corrupt/28eb9c9dd370c07f00f2e5e10e413f987c1d80bde3d2ab11f67fe96b008b03c6/census_e2e-2.json`
- Copy is byte-identical and has the same SHA256. Original remains in place.
- Paired claim: `artifacts/grm_c4/lead_a5/runs/c96_w64/census_e2e-2.attempt.json`
- Claim bytes: 726; SHA256: `ecd7924d1de68f43b734cdf2f28524918fe7cde95e04182d54a35cbb6ed766f8`
- The pair represents ONE corrupt unit, not a separate abandoned unit. No complete
  top-level `gpu_seconds` field survives. Its incident charge is 0 under the lead's
  explicit corrupt-receipt accounting rule; this does not claim zero actual usage.
- Exact failure: `JSONDecodeError: Expecting property name enclosed in double quotes: line 545 column 3 (char 16384)`.

`ABANDONED_DISK_FULL`: no further unpaired claim exists in the four live receipt
roots (`runs`, `lead_a2/runs`, `lead_a3/runs`, `lead_a5/runs`). None was fabricated
or charged. Synthetic gates establish classification and the full 285 s charge
for a registered abandoned claim. Any newly appearing historical claim or damaged
receipt blocks this frozen amendment; it cannot silently gain retry authorization.

The five intact earlier c96_w64 receipts all parse PASS:

| Receipt under `artifacts/grm_c4/lead_a5/runs/c96_w64/` | SHA256 | Wall seconds |
| --- | --- | ---: |
| `sup_correction_then_restatement.json` | `159d4503b470c9ed9b8aeeeab1700204bc11fb707e5f25d5b8b880b0e0a6be92` | 47.64083048515022 |
| `sup_fresh_fact_controls.json` | `961727bd13d55f779f4f85443aa55c745b0beb40fd0bff5c3e2993b55a38c629` | 46.46300751529634 |
| `sup_multi_hop_a_b_c.json` | `ecdd9b67bdccd4e5af4f906f73cedbc47c0acb2e05f42071783725c62668efd4` | 51.76203435799107 |
| `sup_short_correction_long_competitor.json` | `4ac94124afb58deca13a00ec1fa1b8b12a6134744f2d2af69eb522dcbb3a3eeb` | 45.94713612413034 |
| `census_e2e-1.json` | `dac40e85c5554cc8da3f60ea3f249996401ffd762c790f208d67bba800921d9e` | 66.06672716792673 |

The complete e2e-1 saved-session file inventory also verifies against its receipt.
`validation.json` records this and all five parsed receipt pins.

Frozen amendment: `artifacts/grm_c4/lead_a8/amendment_a10.json` (4,826 bytes), SHA256:
`8108af9b123c31508bf1a5b99703b832653efd6176ba5e612ff5a09cb4e42431`.
A10 is the artifact sequence; this remains lead amendment 8. Pre-registration,
before.json and the amendment were created before gates. No sealed implementation
or test source changed after the first gate. A separate
`supplemental_gates_registration.json` binds the additional full-sequence test
before that test ran. The immutable order was not edited.

Implementation: new `scripts/grm_c4_recovery_a8.py` only; A5/A6/A7 sources unchanged.

| Lines | Behavior |
| --- | --- |
| 51, 82 | Decode only complete top-level recorded wall fields; classify exact corrupt/abandoned evidence and charge per lead rule. |
| 127 | `df -B1 --output=avail /mnt/ForgeRealm`; print available bytes and decimal GB; refuse below 20,000,000,000 bytes; malformed/error output fails closed. |
| 155, 163 | Select one A8 successor pathname for an incident; validate historical receipts under their original amendment and new receipts under A8. |
| 170 | Account historical PASS/RED walls, classified incident charges and new re-run walls once each. Unknown corrupt receipts/unmatched claims block. Keep reserve285/cooldown30. |
| 211, 217, 235 | Remaining units in registered order, refreshed fitting-only projection, exact continuation commands. |
| 255, 289 | Verify predecessor closure, all 92 original snapshot pins, incident/archive bytes, new source/test/command pins and amendment SHA. |
| 300, 309 | Overlay unchanged A5 executor with isolated A8 receipt/session root. Census copies and verifies the intact prior state; each worker checks free space before admission and readiness. |
| 345 | Create-only archive, command file and amendment registration. |

The authorized re-run receipt is
`artifacts/grm_c4/lead_a8/runs/c96_w64/census_e2e-2.json`, with its create-only
`.attempt.json` counterpart. An existing PASS, RED, NON_FIT, or unfinished attempt
at this successor path prevents another attempt. New units also get only one
create-only A8 pathname. Old truncated sessions cannot collide with new sessions.
The A8 census runner seeds the intact e2e-1 state into its new root, verifies the
copied SHA/byte counts, and calls the unchanged census implementation.

CPU gates:

- `tests/test_grm_c4_recovery_a8.py`: 46 PASS, 10.93 s; `cpu_gate_a8.txt`, exit 0.
  Exact bytes/inventory at line31; recorded-wall boundaries at51/56;
  abandoned285 at74; corrupt/abandoned one-shot PASS/RED/unfinished cases at86;
  two-shard census recovery at124; df boundaries/errors/early refusal at147–171;
  unknown damage at178; forged/stale closure at188/201; live projection, ordering,
  first admission and immutable originals at216.
- `tests/test_grm_c4_recovery_a8_sequence.py:15`: 1 PASS, 6.79 s;
  `cpu_gate_sequence.txt`, exit0. Full synthetic remaining sequence:38 fake
  leases total; each of three cells has19 completed and4 failed segments,
  no missing segments, all registered/dependency NON_FIT units have no attempts,
  and the cross-summary remains INCONCLUSIVE with no fabricated battery score.
- Existing A5/A6/A7 suite:95 PASS,5 FAIL,55.25 s; `cpu_gate_legacy.txt`, exit1.
  Retained failures are detailed under RED below. Tests were not weakened.
- Live CPU preflight and first-unit admission/readiness PASS; `preflight.txt`,
  `accounting_diagnostics.jsonl`, `validation.json`. Free bytes at the recorded
  preflight:41,018,793,984 (41.018793984 GB). `bash -n` continuation:exit0.
- Each pytest run emitted two existing Swig deprecation warnings and a
  swigvarlink deprecation warning at interpreter exit.

Evidence class: planning projection using recorded E2E walls and unchanged A6
estimates, not a new timing measurement or a completion guarantee.

| Quantity | Seconds |
| --- | ---: |
| Recorded walls including historical REDs and five intact new PASSes | 2,415.5582846361212 |
| Incident charges on this actual disk (one corrupt pair; no abandoned unit) | 0 |
| Remaining fitting estimates (37 fitting units; dependency-blocked estimates retained) | 3,396 |
| Refreshed total | 5,811.558284636121 |
| Cap | 6,600 |
| Headroom | 788.4417153638788 |
| Peak projected reservation | 6,030.558284636121 |
| New GPU execution/charge by this seat | 0 |

41 units remain:37 fitting and4 registered skips. Six planning NON_FIT skips
remain registered across the campaign; two already have immutable outputs.
The four completed c64_w96 NON_FIT units now use actual zero cost; their two
previously projected dependency-blocked fitting estimates (63+66 s) are no longer
remaining work. Replace the five completed c96_w64 estimates (320 s) with their
257.8797356504947 s measured walls. Thus the old 6,002.678548985627 s projection
becomes5,811.558284636121 s. No cap, estimate, skip, dependency or scoring rule
changed. All long-history batteries remain NON_FIT.

## 2. Exact lead command.

```bash
bash /mnt/ForgeRealm/wt/grm-c4/artifacts/grm_c4/lead_resume_a8.txt
```

It performs A8 preflight, retains the foreground30 s cooldown and each inherited
590 s outer timeout/285 s worker lease, and starts with:

```bash
timeout --signal=KILL 590s python scripts/grm_c4_recovery_a8.py worker --cell c96_w64 --battery census --spec e2e-2
```

The list omits every already-completed unit, preserves the registered order of
all41 remaining units, records the remaining skips, and produces A8 partial scores
and summary. It also creates a CPU-only c64_w96 partial score in the A8 root so
all summary inputs share the A8 reporting identity. The old score is untouched.
The command file SHA is in `validation.json` and `SHA256SUMS`. Neither lead command
was run by this seat; existing re-run guards prevent restarting consumed units.

Exact validation commands (foreground, sequential):

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python scripts/grm_c4_recovery_a8.py register
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m pytest -v -p no:cacheprovider tests/test_grm_c4_recovery_a8.py > artifacts/grm_c4/lead_a8/cpu_gate_a8.txt 2>&1
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python scripts/grm_c4_recovery_a8.py preflight > artifacts/grm_c4/lead_a8/preflight.txt 2> artifacts/grm_c4/lead_a8/accounting_diagnostics.jsonl
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m pytest -v -p no:cacheprovider tests/test_grm_c4_split_a5.py tests/test_grm_c4_skip_a6.py tests/test_grm_c4_accounting_a7.py > artifacts/grm_c4/lead_a8/cpu_gate_legacy.txt 2>&1
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m pytest -v -p no:cacheprovider tests/test_grm_c4_recovery_a8_sequence.py > artifacts/grm_c4/lead_a8/cpu_gate_sequence.txt 2>&1
bash -n artifacts/grm_c4/lead_resume_a8.txt
```

## Prior art

Local C4/A2/A5/A6/A7 and DET1 (house,2026), verified directly in
`scripts/grm_c4_resume_a2.py`, `grm_c4_split_a5.py`, `grm_c4_skip_a6.py`, and
`grm_c4_accounting_a7.py`: reused create-only SHA registration, historical/successor
receipt selection, incomplete-claim guards, recorded-wall-plus-reservation
accounting, cooldowns, ordered foreground commands, complete-battery scoring and
copy-and-verify state resume. Tests reuse the local A5/A6 synthetic lease/state
fixtures and A7 exact-byte/forgery approach. Python's installed standard-library
`json.JSONDecoder.raw_decode` supplies token parsing; the installed GNU coreutils
`df` supplies available-byte measurement. These are system primitives, not new
algorithms. Their original publication dates were not researched; no external
literature claim is made. Lead literature-check terms if desired: Python json
JSONDecoder raw_decode; GNU coreutils df output avail.

A8's contribution is the lead's exact incident authorization, fixed20 GB admission
threshold and isolated successor integration. No novel algorithm is claimed.
Specific inherited A5 timing heuristic: no prior art known to me; unchanged.
Prior-art annotations appear at implementation/test sites and in the ledger.

## 3. Prior art; deviations; RED; process safety; model id and effort.

Prior art is stated above. Deviations: no further abandoned claim was found,
so its gate is synthetic and no extra live285 s charge is invented. A8 uses a
separate receipt/session root for all remaining work, and reconstructs c64_w96's
CPU partial score there for consistent summary identity. Original receipts,
claims, archived damage, sources, registrations and order remain untouched.
The shared synthesis receives only an appended A8 entry, with prefix preservation
verified separately. There were no post-gate source corrections.

RED, verbatim failure lines retained in `cpu_gate_legacy.txt`:

```text
FAILED tests/test_grm_c4_split_a5.py::test_real_existing_bytes_unchanged
E       AssertionError: assert not True
FAILED tests/test_grm_c4_skip_a6.py::test_existing_artifacts_byte_unchanged_and_no_real_attempts
E       assert False
FAILED tests/test_grm_c4_accounting_a7.py::test_exact_crashed_disk_layout_keyerror_then_completed_receipts
E       AssertionError: Unfinished worker claim or incomplete worker receipt: campaign blocked: /tmp/pytest-of-vader/pytest-925/test_exact_crashed_disk_layout0/lead_a5/runs/c96_w64/census_e2e-2.json, /tmp/pytest-of-vader/pytest-925/test_exact_crashed_disk_layout0/lead_a5/runs/c96_w64/census_e2e-2.attempt.json
FAILED tests/test_grm_c4_accounting_a7.py::test_metadata_attempt_and_score_are_excluded_before_read
E       AssertionError: Unfinished worker claim or incomplete worker receipt: campaign blocked: /tmp/pytest-of-vader/pytest-925/test_metadata_attempt_and_scor0/lead_a5/runs/c96_w64/census_e2e-2.json, /tmp/pytest-of-vader/pytest-925/test_metadata_attempt_and_scor0/lead_a5/runs/c96_w64/census_e2e-2.attempt.json
FAILED tests/test_grm_c4_accounting_a7.py::test_a6_plan_budget_commands_and_historical_artifacts_unchanged
E       AssertionError: assert 18 == 5
```

A5 assumes no lead_a5 run root. The old A6 snapshot differs only by A7's already
registered `scripts/grm_c4_split_a5.py` replacement at gate time. A7's two old
live-layout accounting tests correctly fail closed on the incident; A8 alone
provides the new scoped authorization. The other A7 test assumes the old five-file
CPU-only snapshot, while18 files now exist. None is claimed fixed or suppressed.
All92 files captured in the A8 before snapshot remain unchanged.

Not claimed fixed: actual GPU re-run completion, disk exhaustion during a lease
by concurrent writers, full long-history fitting/quality, or independent blind
verification. Free-space checks are point-in-time admission checks, not disk-space
reservations. New unregistered corrupt/unfinished evidence blocks further work.
Existing historical RED walls remain charged. New tests are author baseline only;
lead blind review was not launched by this seat.

Process safety: no git, subagents, GPU execution, shell background jobs, background
waits, process kills/signals, lock/service changes or original receipt overwrites.
All test processes exited naturally; leases were mocked. No real A8 run root,
attempt or completed GPU receipt exists. Lead command file is prepared only.

Model: GPT-6 (Codex; exact deployment model identifier is not exposed).
Effort: high (requested). No alternate model or agent was dispatched.
