# GRM-X1 amendment 2 — CPU PASS; continuation ready for lead GPU execution

Continuation accepted by live CPU preflight. Author CPU baseline: **174 passed**,
2 existing SWIG warnings. Active r2 campaign is **PENDING**, with **zero r2
claims, zero r2 GPU receipts, zero r2 observation rows**. The r1 RED is retained
under `historical_red`; it does not set the r2 verdict. No GPU execution in this
seat, as required by amendment 2. Card result is not claimed fixed.

## Continuation and changes

Create-only manifest: `artifacts/grm_x1/continuation_02.json`.
SHA-256: `a49cd7a3453fde39c5a8b575fa0a0e92d5d134c2fc948b362ba8dab362204c20`.
Create-only checksum: `artifacts/grm_x1/continuation_02.sha256`.
Pre-gate registration: `artifacts/grm_x1/continuation_02_registration.json`,
SHA-256 `26dc26ddae0e505421c192a0b15ed89ad8d83b7589bc4e518054fcb3a726b293`.

Manifest binds the immutable amendment-2 order, payload-amendment handoff,
original registration, original handoff, pre-gate registration, every amended
source hash, and the exact historical claim/RED receipt hashes. It retains
the amendment-1 worker unchanged. Original handoff remains accepted when its
sources match and no continuation is installed. A present invalid continuation
never silently falls back. Dependencies still must match the original handoff.

| File / lines | Change |
| --- | --- |
| `scripts/grm_x1_campaign.py:33–42` | Fixed authority hashes and authorized budget. |
| `scripts/grm_x1_campaign.py:98–117` | Nine-cell dry-run with retry and historical/total reservation accounting. |
| `scripts/grm_x1_campaign.py:265–342` | Validate/seal create-only continuation; choose its SHA as receipt fingerprint. |
| `scripts/grm_x1_campaign.py:344–427` | Validate r1/r2 claims and receipts separately; only r2 supplies campaign metrics/verdict. |
| `scripts/grm_x1_campaign.py:455–565` | Route new controller/GPU receipts to r2 and claims to r2; expose `seal-continuation`. |
| `tests/test_grm_x1_campaign.py:217–374` | Five gates, original-handoff compatibility, and all-nine-cell CPU simulation. |
| `docs/GRM_X1_LEDGER.md:231` onward | Append pre-gate registration, authority/budget, results and process record. |
| `artifacts/grm_x1/lead_commands.txt:1–19` | Executable ordered lead commands; failure stops and summary remains available. |
| `artifacts/grm_x1/lead_commands_amendment_01.txt` | Create-only preservation of amendment-1 commands. |

Additional new artifacts: this report, the three continuation registration/seal
files, `receipts/continuation_02_gates.json`,
`receipts/continuation_02_lead_checks.json`, CPU baseline/pytest receipts, and
`continuation_02_delivery.json` (file inventory and hashes).

r2 naming: `artifacts/grm_x1/receipts/r2/gpu_{cell}_{receipt_content_sha256}.json`;
controller receipts are in the same r2 directory. Both carry `fingerprint =
a49cd7a3453fde39c5a8b575fa0a0e92d5d134c2fc948b362ba8dab362204c20`.
Claims: `artifacts/grm_x1/claims/r2/{cell}_{continuation_sha256}.json`.
Session paths remain `artifacts/grm_x1/sessions/{cell}_{fingerprint}`.
No r2 claim/receipt has been written to the real campaign by CPU simulations.

r1 receipt remains in place, unchanged:
`artifacts/grm_x1/receipts/gpu_oracle_m1_s0_0c1bf61590059cad2b207f0edef2da421f64639f69c41d4b6282cc7ba3c35bba.json`.
Its SHA is `0c1bf61590059cad2b207f0edef2da421f64639f69c41d4b6282cc7ba3c35bba`.

Only the frozen r1 RED `oracle_m1_s0` can be retried, once. Any r2 RED or
incomplete claim stops run/resume and all successors. A completed r2 cell is
also consumed. Unknown fingerprints, duplicate receipts, receipts without
claims, and changed r1 evidence are refused. The natural arm remains locked
until the complete oracle arm is positive under the original thresholds.

## Five gate results

Evidence class: author CPU unit test. All names belong to
`tests/test_grm_x1_campaign.py`; full passing baseline includes every case.

| Test name | Result |
| --- | --- |
| `test_continuation_accepted` | PASS (1 case) |
| `test_forged_continuation_refused` | PASS (9 cases) |
| `test_stale_continuation_refused` | PASS (1 case) |
| `test_second_retry_refused` | PASS (3 cases) |
| `test_r1_receipt_untouched` | PASS (1 case) |

Receipt: `artifacts/grm_x1/receipts/continuation_02_gates.json`.
Baseline: `artifacts/grm_x1/receipts/cpu_baseline_b1a5012ed9e8c8d294b710e1a02731a93e153a354b20d4a09d650a7d3eeb222c.json`.
Pytest: `artifacts/grm_x1/receipts/cpu_pytest_08ae49c9c4c2dc2f8070b2e086cbf2b2c7b5ea7d75a813bd595f0e1e1eb3e81a.json`.
Exact baseline command: `PYTHONDONTWRITEBYTECODE=1 python3 scripts/grm_x1_cpu.py baseline`.
The additional ordered-command simulation completed all nine cells using
synthetic CPU operands and checked 285 r1 + 2565 r2 = 2850 reserved seconds.
It does not establish GPU runtime, recall, latency, or actual oracle positivity.
Existing unchanged core mutation gates were not rerun; the changed controller
was checked by the new continuation/state tests and full CPU baseline.

Live command receipt: `artifacts/grm_x1/receipts/continuation_02_lead_checks.json`.
`preflight`, `--dry-run`, `summary`, and `bash -n artifacts/grm_x1/lead_commands.txt`
all returned 0. Preflight checked dependency hashes/path-stat inventory without
a GPU probe or model construction. Dry-run lists the original nine cells in
order; only `oracle_m1_s0` has `retry: true`. `next_cell()` returns that cell.
Summary reports r1 reserved 285 seconds, r2 reserved 0, cap 2850, r2 PENDING.

## Exact lead commands

Dispatch commands singly in the foreground. Stop at any RED. The last three
cells require a complete positive oracle; otherwise retain the registered stop
and run `summary`. These GPU commands were CPU-simulated, not run on the card.

```bash
#!/usr/bin/env bash
# Lead only: dispatch each command separately in the foreground (<590 s rail).
# A RED stops the sequence. Natural cells require the complete positive oracle.
# If invoked as a script, stop on error and still report summary on exit.
set -euo pipefail
cd /mnt/ForgeRealm/wt/grm-x1
trap 'bash scripts/grm_x1_lead_gpu.sh summary' EXIT
bash scripts/grm_x1_lead_gpu.sh preflight
bash scripts/grm_x1_lead_gpu.sh run oracle_m1_s0
bash scripts/grm_x1_lead_gpu.sh run oracle_m1_s1
bash scripts/grm_x1_lead_gpu.sh run oracle_m10_s0
bash scripts/grm_x1_lead_gpu.sh run oracle_m10_s1
bash scripts/grm_x1_lead_gpu.sh run oracle_m100_s0
bash scripts/grm_x1_lead_gpu.sh run oracle_m100_s1
bash scripts/grm_x1_lead_gpu.sh run natural_m1
bash scripts/grm_x1_lead_gpu.sh run natural_m10
bash scripts/grm_x1_lead_gpu.sh run natural_m100
trap - EXIT
bash scripts/grm_x1_lead_gpu.sh summary
```

## Budget, deviations, RED and process safety

Registered reservation: **2565 + 285 = 2850 seconds = 0.7916667 GPU-hours**,
rounded to 0.79. This exceeds COMMON 2700 seconds / 0.75 GPU-hours by 150 seconds,
under the lead's explicit amendment-2 decision. This is reserved allowance,
not measured use. The planned remaining nine calls reserve at most 2565 worker
seconds; no-contention estimated wall is 2835 seconds including cooldowns.
Each worker allowance remains 285 seconds, own-process work alarm 280 seconds,
lease wait 250 seconds, foreground cooldown 30 seconds, outer rail 590 seconds.

Deviations: none from the no-GPU seat scope. The campaign is enabled and handed
to the lead; GPU runs and verdict remain pending. All validation is author-run;
no independent blind verification claimed. Hashes protect local integrity under
trusted order/code storage; they are not cryptographic authorization signatures.

Historical RED verbatim: `TypeError: 'NoneType' object is not iterable`.
Historical worker wall: 34.63641106989235 seconds (r1 E2E failure receipt),
charged as a 285-second reservation. Not claimed fixed on the card.
A non-returning native call can still defeat the own-process exception timer;
the existing hard-bound limitation remains RED, unchanged by this continuation.

Process safety: no git commands, no subagents, no GPU calls, no background
jobs/waits, no kills/signals sent by this seat, no service writes, no sibling
edits. CPU test substitutions and receipt mutations ran only in temporary
copies. No core or GPU-worker source edited for amendment 2. Original orders,
registration, handoffs, claim and r1 RED receipt remain unchanged.
Author model: GPT-6; exact deployment variant/ID unavailable in session metadata.
Reasoning effort: high (requested and followed). Reader target remains
registered `openai/gpt-oss-20b`, inference effort low; no reader loaded here.

## Prior art

House X1 campaign controller (2026), verified in local source: borrowed
create-only reservations/receipts, fingerprint checks, frozen experiment
forks, and the existing leased worker protocol. NIST FIPS 180-4 (2015), SHA-256:
**unverified — lead to check**, search terms `NIST FIPS 180-4 2015 SHA-256`.
New work is the amendment-specific manifest validation, r1/r2 partition,
and single paid retry integration; no novel hash, paging, or selection
algorithm is claimed. Same annotation appears at code sites and in the ledger.
