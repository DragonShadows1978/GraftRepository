# GRM-X1 amendment 3 — CPU PASS; r3 GPU timing PENDING

Evidence class: author unit tests and CPU protocol simulations. **196 passed**,
2 existing SWIG warnings. No r3 GPU execution, model load, timing or accuracy
claim. Per-unit fit and total campaign fit are **not claimed established**.

## 1. continuation_03, units, files and gates

- Manifest: `artifacts/grm_x1/continuation_03.json`
- SHA-256: `e8d854f6a6eabbe27602992ba6a804a9b5d4a102417fe726fc8c746d0bcd5b10`
- Checksum: `artifacts/grm_x1/continuation_03.sha256`
- Pre-edit/gate registration: `artifacts/grm_x1/continuation_03_registration.json`
- Registration SHA-256: `0262299553a7123b297488383de9cfd4bf3922a1195e9f23e7068ac4f00a4729`
- Immutable order: `orders/GRM_X1_AMENDMENT_3.md`; prior manifests unchanged.

**Scheme: one query per unit.** Each lease loads its model and captures the
query's family template inside that unit, then installs, forwards and scores
all registered conditions/arms. Oracle units contain six paired turns;
natural units contain two. Global query ordinals, RNG seeds, arm rotation,
fixture values and scientific thresholds remain those of the original harness.
There is no capture dependency on another lease and no forward chunking.

| Cell, in registered order | Units | Turns per unit |
|---|---:|---:|
| oracle_m1_s0 | 12 | 6 |
| oracle_m1_s1 | 12 | 6 |
| oracle_m10_s0 | 12 | 6 |
| oracle_m10_s1 | 12 | 6 |
| oracle_m100_s0 | 12 | 6 |
| oracle_m100_s1 | 12 | 6 |
| natural_m1 | 24 | 2 |
| natural_m10 | 24 | 2 |
| natural_m100 | 24 | 2 |

144 total units: 72 oracle and 72 conditional natural; 576 total paired rows.
Counts are fixture enumeration, not GPU observations. First unit is
`oracle_m1_s0__alias_00`.

`run <cell>` consumes one next missing unit and returns. Create-only claims
use `claims/r3/<cell>__<query>__a<attempt>_<continuation_sha>.json`; unit receipts
use the same basename under `receipts/r3`. Deterministic attempt paths reject
a second write even if payload bytes differ. Receipts include rows, worker
wall, row SHA, whole-content SHA and hashes of session JSON (including page
payload digest inventories). Publication occurs in the worker before its exit,
controller cooldown or any next lease. Session payloads keep the r2 packed-host
digest format; no device-h comparison is introduced.

A cell is COMPLETE when all its units have receipts, including RED receipts.
Its aggregate still counts RED/NON_FIT units. Summary retains partial RED rows,
uses only the latest attempt per unit for metrics, sums wall across all
attempts, and keeps attempt receipts visible. Any r3 RED prevents natural
unlock. Normal resume skips a receipted RED. An explicit `--retry-unit UNIT`
may retry a non-timeout RED once; two REDs for that unit stop the campaign.
Incomplete claims stop for lead reconciliation. Timeout units are RED with
`fit_status=NON_FIT`, cannot be retried/chunked, and do not prevent trying
other missing units. A cell completed with NON_FIT prints NON_FIT; an ordinary
single RED prints UNIT_RED and the loop continues.

Budget: **5400 seconds / 1.5 GPU-hours**, including **570 seconds** for the two
historical reservations. Completed r3 leases charge measured worker wall;
incomplete claims charge 285 seconds. Each new lease requires 285 seconds of
headroom. First successful unit wall becomes the fixed planning estimate:
`570 + 144 * estimate`; also check charged cost plus remaining-unit estimate.
If either exceeds 5400, summary is NON_FIT_BUDGET and the loop exits for lead
decision. Natural units are included conservatively even before unlock.
Worker allowance 285 s; work alarm 280 s; lock wait 250 s; outer rail 590 s;
foreground cooldown 30 s. At handoff: charged 570 s, zero r3 claims/receipts,
zero r3 rows, estimate null, verdict PENDING.

### Files / lines

- `scripts/grm_x1_units.py:27`: enumeration; `:40`: binding/scope validation;
  `:72`: create-only seal; `:122`: receipt publication; `:139`: aggregate,
  historical RED and budget; `:235`: resume; `:287`: controller; `:348`: worker.
- `scripts/grm_x1_gpu.py:260`: single-query wrapper around existing capture/
  paired execution; native/core implementation unchanged.
- `scripts/grm_x1_campaign.py:98`, `:268`, `:388`, `:546`: r3 entry-point routing.
- `scripts/grm_x1_cpu.py:35`: include r3 tests in baseline.
- `tests/test_grm_x1_units.py:54`: r3 gates below.
- `tests/test_grm_x1_campaign.py:17`: live matrix uses r3 budget; historical r2
  fixtures use frozen r2 sources/commands, preserving old epoch regressions.
- `artifacts/grm_x1/lead_commands.txt:10`: ordered per-cell/per-unit loop;
  prior commands preserved create-only as `lead_commands_r2.txt`.
- `docs/GRM_X1_LEDGER.md`: append-only registration and result entries.

### Test names and results

| Test in tests/test_grm_x1_units.py | Cases | Result |
|---|---:|---|
| test_r3_enumeration_order (line 54) | 1 | PASS |
| test_r3_create_only (line 65) | 1 | PASS |
| test_r3_aggregate_sum (line 79) | 1 | PASS |
| test_r3_resume_next_missing (line 99) | 1 | PASS |
| test_r3_red_advances_and_second_red_stops (line 108) | 1 | PASS |
| test_r3_non_fit_stays_local_and_no_forward_chunking (line 119) | 1 | PASS |
| test_r3_incomplete_claim_stops (line 130) | 1 | PASS |
| test_r3_historical_bytes (line 138) | 1 | PASS |
| test_r3_dry_run (line 150) | 1 | PASS |
| test_r3_budget (line 160) | 3 | PASS |
| test_r3_estimate_is_first_completion_and_red_retry_not_double_counted (line 174) | 1 | PASS |
| test_r3_bindings (line 187) | 5 | PASS |
| test_r3_worker_cpu (line 207) | 2 | PASS |
| test_r3_run_unit_preserves_capture_and_global_order (line 232) | 1 | PASS |
| test_r3_lead_loop (line 245) | 1 | PASS |

Command: `PYTHONDONTWRITEBYTECODE=1 python3 scripts/grm_x1_cpu.py baseline`.
196 total passed, including 22 r3 cases, X1 address/campaign,
RT1 and RS3 regression gates; elapsed pytest 3.74 s. Existing payload fix
CPU gates remain green. Tokenizer/fixture checks also pass.

Receipts:

- `artifacts/grm_x1/receipts/cpu_baseline_0d8b663a1e90e0204f3a43c9b0d0eb20b7c35420ca62d40916768519ca929f07.json`
- `artifacts/grm_x1/receipts/cpu_pytest_88bea04f528ad2ce122b59ba3fd0337808d1b347b633fc1809b910526561358b.json`
- `artifacts/grm_x1/receipts/continuation_03_gates.json`
- `artifacts/grm_x1/receipts/continuation_03_lead_checks.json`
- `artifacts/grm_x1/receipts/continuation_03_shell_gates.json`

Real CPU preflight, --dry-run, summary and bash syntax all returned 0.
24 registered historical files hash-identical. Separate preregistered shell
protocol simulation ran an isolated copy with a CPU stub (no real runner):
144 calls with one RED then completion exited 0; first-unit budget NON_FIT
exited 2 after one call; natural lock exited 0 after 73 calls; registered
stop exited 1 after one call. The worker-prefix/final-JSON parser was exercised.
Author baseline only; no blind verifier dispatched. Existing unchanged-core
mutation suite was not rerun; no new mutation or independent-verification claim.

## 2. Exact lead commands

For the supplied loop (each `run` is exactly one lease):

```bash
cd /mnt/ForgeRealm/wt/grm-x1
bash scripts/grm_x1_lead_gpu.sh --dry-run
bash artifacts/grm_x1/lead_commands.txt
```

Or dispatch individual calls in the foreground:

```bash
cd /mnt/ForgeRealm/wt/grm-x1
bash scripts/grm_x1_lead_gpu.sh preflight
bash scripts/grm_x1_lead_gpu.sh run oracle_m1_s0
bash scripts/grm_x1_lead_gpu.sh summary
```

Repeat `run oracle_m1_s0` until CELL_COMPLETE or NON_FIT, then use the next
cell in the table. UNIT_COMPLETE and a single UNIT_RED mean repeat that cell.
`resume` selects the next missing eligible unit globally. NON_FIT_BUDGET exits
the supplied loop with code 2 for lead decision; any STOP status exits with
code 1. NATURAL_LOCKED ends without running conditional units. Summary runs
on loop exit. `--dry-run` does not take a lease. Never invoke `_worker` directly.

Optional explicit retry syntax for a non-timeout unit RED, subject to budget:
`bash scripts/grm_x1_lead_gpu.sh run oracle_m1_s0 --retry-unit oracle_m1_s0__alias_00`.
Normal loop never retries units automatically. No retry is currently eligible.

## 3. Prior art

House X1 harness (2026), verified in local `grm_x1_campaign.py` and
`grm_x1_gpu.py`: reused exclusive creation, SHA receipts, foreground flock,
paired query snapshots, original global ordinals, scoring and budget rails.
New integration: self-contained query-sized lease/checkpoint boundaries,
unit/attempt resume and first-completed-unit projection. No new scientific
algorithm or hash algorithm claimed. SHA-256 antecedent NIST FIPS 180-4
(2015): **unverified — lead to check**; search terms “NIST FIPS 180-4 2015
SHA-256”. Hash bindings assume trusted local storage and are not signatures.
Attribution appears at code sites and in the ledger as required.

### Deviations, RED and process safety

Evidence correction to amendment prose: the actual SHA-bound r2 receipt
`gpu_oracle_m1_s0_374c534e547b8d9116d7d43fe5cfedd5f44fd9c81af007714286440f4fe1bf7c.json`
contains **17 partial rows**, from alias_00, alias_02 and alias_04, not an empty
rows list. Its worker wall is 280.00282101891935 s. Its actual error is
`TimeoutError: GRM-X1 worker reached its 280 s work rail (285 s outer worker allowance)`;
traceback includes `core/gpt_oss20b_tc.py:1025`. r1 remains
`TypeError: 'NoneType' object is not iterable`. Both stay historical_red with
unchanged bytes and excluded from r3 metrics. The r2 partial observations do
not measure a fresh, self-contained r3 unit including model/template capture.
Per-unit wall is unknown. GPU timing/fit is **not claimed fixed** by CPU tests.

No scope deviation. Details resolved in preregistration: per-query scheme,
actual-wall accounting for completed r3 leases, conservative all-144-unit
projection, explicit single retry, and incomplete-claim stop. Timeout NON_FIT
is unit-local so other missing units can proceed, as the amendment requires.

No git commands, subagents, GPU probes/runs, background jobs/waits, process
kills, service writes, sibling/core edits or model loads. Tests were foreground
CPU subprocesses; CPU shell simulations used private temporary trees. Lead-only
worker retains cooperative own-process SIGALRM and no timeout/kill wrapper.
The cooperative timer cannot guarantee hard preemption of a non-returning
native call; receipt/cleanup overhead and real lease fit require lead evidence.

Author: **GPT-6**, exact deployment variant unavailable; reasoning **high**.
Registered reader: **openai/gpt-oss-20b**, reasoning **low**, unchanged/unloaded.
