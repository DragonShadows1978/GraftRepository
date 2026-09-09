# GRM-X1 implementation ledger

## 2026-09-08 — registration before gates

Order is immutable: `orders/GRM_X1_VIRTUAL_ADDRESSES.md`. Read
`/mnt/Shared/HOUSE_RULES.md`, local `AGENTS.md`, scout Part D/E, EB1,
RT1, RS3/RS4, SUP/census constructors and both registered configs. Read
`.git` and worktree HEAD as text only: branch is `grm-x1`; no git commands.
No sibling worktree touched, no subagents, no GPU calls, no background waits.

Created `scripts/grm_x1_register.py`; ran
`PYTHONDONTWRITEBYTECODE=1 python3 scripts/grm_x1_register.py` before any gate.
Registration SHA-256:
`defb5014f6ee66113139c1ca3fe902f9bdc51fb81d137cbc0a47e5d06f8f3e57`.
Fixture receipt: `artifacts/grm_x1/fixtures/SHA256SUMS`.
Evidence class: reasoning/pre-gate registration, not recall validation.
Frozen 24 new wordings (12 alias / 12 correction), 13 source families,
3 multiplicities, present/absent conditions, seed 20260908. Existing facts
are known fixtures; held-out means wording withheld from fitting only.
No hyperparameter fitting will occur. Lead and seat predictions are preserved
verbatim in registration. Oracle cap 1710 worker seconds, conditional cap
855 seconds, combined 2565 seconds (<2700 order maximum); these are registered
bounds, not measured timings. Six primary and three conditional cells.

Design: additive `core/grm_x1_addresses.py`, no edit to existing production
modules. OFF is direct callback passthrough; topical discovery also passes
through. Exact current-version pointers precede native `_attempt`; C verifies
actual mounted IDs at the first forward. No text/value is copied into a query.
Single-writer sidecar; explicit version conflicts fail, no implicit rollback.

## Prior art

Code-site annotations carry: Codd (1970), logical relational identity
([ACM SIGMOD bibliography](https://sigmod.org/publications/dblp/db/journals/cacm/Codd70.html));
Denning (1968), bounded working sets/page absence
([primary paper](https://denninginstitute.com/pjd/PUBS/WSModel_1968.pdf)).
Reed (1978), multiversion indirection: unverified — lead to check, search
“Naming and Synchronization in a Decentralized Computer System”.
Fisher (1935), paired blocked experiments: unverified — lead to check,
“The Design of Experiments”. Unicode normalization UAX #15: unverified —
lead to check. Dictionary token-span lookup is established; no novelty claim.
Borrow existing house EB1/SUP/RT1/RS3/RS4, native deposit/mount/reader and
fixture constructors (2026); new contribution is this sidecar and comparison.

## Pre-gate limitations and decisions

The worktree has no archived DET1 runtime frame or local `cpp/build`.
The shared native library and pinned GPT-OSS snapshot directory exist by
read-only inspection; actual loading is deferred to the lead. Construction
will reuse the existing E2E loader with explicit registered frame options.
Census decoys reuse the original SUP Falcon competitor unchanged; it names
Orion but not every census family. Source-stratified results are mandatory.
Natural arm is a limited frozen dictionary resolver and is locked behind the
registered positive oracle gate. D-NGH remains default OFF; structural demands
are recorded separately, so this is not an attention-detector comparison.
The author cannot provide blind red-team verification; lead dispatch pending.
Worker timeout raises within its own process and never kills a process.
A non-returning native call cannot be forcibly bounded under the never-kill
rule: this safety/termination conflict remains RED and must be visible in
the handoff, not concealed by a timeout wrapper that signals other processes.

## 2026-09-08 — CPU gates, corrections and final handoff

First baseline command:
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/mnt/ForgeRealm/wt/grm-x1:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 -m pytest -q -p no:cacheprovider tests/test_grm_x1_addresses.py tests/test_grm_x1_campaign.py tests/test_grm_rt1_split_child_routing.py tests/test_grm_rs3_capture_pin_seat.py`.
Receipt `logs/grm_x1_cpu_baseline_01.log`: **148 passed, 1 failed** (unit test).
Verbatim: `ValueError: expected exactly 24 unique held-out wordings`.
Root cause: identical SUP/census Orion correction wordings. Separate immutable
`artifacts/grm_x1/fixture_amendment_01.json` replaces only census query IDs
correction_06 and correction_07; active queries SHA
`abfac7da02c0c1fec32ef98c6bb27694005e5259d2581cc8b476a22c7eea3af6`.
No registration, original fixture, threshold or prediction was edited.

Registered five named CPU defects via
`PYTHONDONTWRITEBYTECODE=1 python3 scripts/grm_x1_cpu.py register-mutants`.
Then ran `python3 scripts/grm_x1_cpu.py baseline` with bytecode disabled;
initial corrected baseline passed 149 tests. Added width-fault/fallback
invariants after reviewing native bootstrap fit, then a real CPU-only flock
protocol test: final baseline **152 passed**, 2 existing SWIG deprecation
warnings. Default-OFF pin is
`tests/test_grm_x1_addresses.py::test_grm_x1_default_off_passthrough_identity`.
Final baseline receipt: `artifacts/grm_x1/receipts/cpu_baseline_6d40d49c1487aabf745aadd1bda77e4e7f83b2cdb834fdcc1380400ea7d474d0.json`.
Final mutation command: `PYTHONDONTWRITEBYTECODE=1 python3 scripts/grm_x1_cpu.py mutations`.
Receipt `artifacts/grm_x1/receipts/cpu_mutations_b940a51b38035709415ba52c331f90f51d8fa3626933be2e9b35ea537c952154.json`: **5 runnable, 5 killed, fraction 1.00** (unit test),
fixed rail .80. Mutants were temporary copies, original module SHA unchanged.
Earlier passing receipts retained; reruns followed actual boundary-test/source
changes, never altered tolerances. No GPU/model accuracy number inferred.

The exploratory `cpu_token_counts_01.json` omitted the Harmony system scaffold
in a hand reconstruction. Its shorter counts are superseded by the final
baseline's AST-read original `harmony_turn`, using the actual local tokenizer:
current target lengths **52–92**, competitors **150–159**, width **96**
(unit test). No gate used the exploratory shorter counts.

Implemented `scripts/grm_x1_campaign.py` (enumeration, strict scoring,
source/kind/condition aggregates, non-vacuous prediction forks, create-only
fingerprinted reservations/outcomes, one-cell resume),
`scripts/grm_x1_gpu.py` (native deposits/splits, shared payload snapshots,
fresh per-arm repositories, explicit addressed native reads),
`scripts/grm_x1_lead_gpu.sh` (foreground command entry),
`scripts/grm_x1_cpu.py`, and both X1 test modules. See blocked report file/line
inventory. Default-OFF and topical paths return the original callback result.

Additional prior-art annotations: DeMillo/Lipton/Sayward (1978), Hints on Test
Data Selection, for mutation testing; NIST FIPS 180-4 (2015) for SHA-256;
util-linux flock (year unverified) for advisory leases. All **unverified —
lead to check**, named phrases are search terms. Shared immutable native
payload snapshots and production splits reuse house 2026 methods. No prior
art known to me for this exact X1 duplicate-child/withheld-page test design.

Ran `python3 scripts/grm_x1_campaign.py seal`: immutable handoff SHA
`bf923a9ccb4f06a8d2ae643cb2727de5f7c116304febf429f71a2da7e9087121`.
Then CPU-only lead entry commands `preflight`, `--dry-run`, `summary`;
receipts in `logs/grm_x1_{preflight,dry_run,summary}_01.json`.
All required dependency files exist by read-only inventory; model weight
files are path/stat inventoried, not freshly content-hashed. No model load.
Dry-run enumerates 6 oracle cells (432 arm/condition turns) and 3 conditional
natural cells (144 turns), planned maxima 1710 / 2565 worker seconds.
These are reasoning/budget numbers. GPU summary has zero reservations and
zero E2E rows, `BLOCKED_NO_GPU_EXECUTED`; no prediction is adjudicated.

Created `artifacts/grm_x1/active_fixture_inventory.json`,
`artifacts/grm_x1/lead_commands.txt`, and `artifacts/grm_x1/BLOCKED_REPORT.md`.
Residuals: no independent blind verification; bounded dictionary aliases;
weak lexical stress for some census controls; single-writer non-atomic
sidecar lifecycle; GPU timing/ABI/recall unverified. A non-returning native
call cannot be unconditionally bounded without process termination; the
runner self-raises and reports overruns, never kills. This conflict is RED,
not claimed solved by a timeout flag. Operator retains absolute GPU priority.
Author GPT-6; exact deployment ID/reasoning effort not exposed. Target
openai/gpt-oss-20b, reasoning low. No git, subagents, background waits,
GPU runs, process termination, live-service writes or sibling edits.

## Final artifact verification

Re-verified immutable registration, original and amended fixtures, source/dependency
fingerprint and passing CPU-baseline source identity. GPU summary remains zero
rows / zero reservations (CPU artifact check, no GPU probe). Verified report
contains the actual handoff and registration SHAs. Final delivery manifest
is `artifacts/grm_x1/delivery_manifest.json`; it includes file paths, line counts
and SHA-256 values. No later source change was made after the passing baseline.

## 2026-09-08 — amendment 1: nonresident payload digest (pre-gate record)

Read immutable `orders/GRM_X1_AMENDMENT_1.md` and HOUSE_RULES. Original GPU
receipt `artifacts/grm_x1/receipts/gpu_oracle_m1_s0_0c1bf61590059cad2b207f0edef2da421f64639f69c41d4b6282cc7ba3c35bba.json`
is RED: `TypeError: 'NoneType' object is not iterable` at the old
`payload_digest -> for layer in node["h"]`. Confirmed verbatim locally
before editing; unit-test reproduction receipt:
`artifacts/grm_x1/receipts/payload_amendment_01_reproduction.json`.

Registered `artifacts/grm_x1/payload_amendment_01.json` create-only BEFORE
reproduction or CPU gates. This entry records that sequence; original order,
registration, fixtures, handoff and GPU receipts remain immutable. No change
to core modules, flags, predictions, gate thresholds, campaign retry rules
or GPU budgets. Scope is worker fix, CPU tests and handoff documentation.

Root cause (source inspection): `h` is optional. `_native_sync_node` resolves
packed host backing through `_ensure_host_payload`; `_load_node` reads that
backing from RAM or NPZ for device reconstruction. The worker now uses the
same host accessor on ORIGINAL repository nodes before cloning. Thus cold
NPZ backing is copied to RAM while the owning repository is available and
shared in subsequent private-arm clones. Hash sorted packed keys, shapes,
dtypes and bytes, including scales. Mark page receipts with format
`packed-host-key-shape-dtype-bytes-v1`; these digests are not comparable to
r1 device-h digests. Missing/empty backing remains an explicit error, never
an empty-payload success. No placeholder fabricated from text or node IDs.

Prior art: GraftRepository (house, 2026), `_native_sync_node`,
`_ensure_host_payload`, `_read_payload_file` and `_load_node`, verified in
local source. Borrow its packed RAM/durable-file storage access and existing
X1 SHA-256 hashing structure. New work is the X1 snapshot/digest integration;
no novel hashing, selection or paging algorithm. Existing SHA-256 antecedent
NIST FIPS 180-4 (2015) remains unverified — lead to check; search terms:
“NIST FIPS 180-4 2015 SHA-256”.

CPU baseline launched:
`PYTHONDONTWRITEBYTECODE=1 python3 scripts/grm_x1_cpu.py baseline`.
New regression: `test_payload_digest_h_none_host_backing_and_cold_file_match`;
also packed-content identity, missing backing (durable/nondurable), and empty
backing checks. Uses actual repository accessor/file reads without model or
native-store construction; resident pack method is a CPU stub. This is author
unit evidence, not independent verification or GPU clearance.

## 2026-09-08 — amendment 1: results and refreshed handoff

CPU baseline PASS: **157 passed**, 2 existing pytest SWIG warnings, in 0.87 s
(unit test). Receipt:
`artifacts/grm_x1/receipts/cpu_baseline_cbb1b285a457ec672461c6580f8c891b46b63eb12e5d3474656fd8fd6bb32bd1.json`.
Includes frozen fixture integrity, original Harmony/local tokenizer fit,
9-cell GPU dry-run enumeration and shell syntax. No GPU model instantiated.
Default-OFF pin remains passing:
`test_grm_x1_default_off_passthrough_identity`.

Ran `PYTHONDONTWRITEBYTECODE=1 python3 scripts/grm_x1_cpu.py mutations`:
**5 runnable, 5 killed, fraction 1.00 >= registered 0.80**, PASS (author unit
mutation gate). Receipt:
`artifacts/grm_x1/receipts/cpu_mutations_7112056e61db4558269ca982c03c15f6b843b44228bd749a68dc87905206e4b8.json`.
Original core-module hash unchanged; temporary mutant copies only.

CPU handoff checks: lead entry `--dry-run` and `summary` return successfully;
`preflight` rejects with `ValueError: handoff source drift; do not start a new
campaign silently`. Direct `next_cell()` rejects with
`ValueError: failed/incomplete cell: registered stop; no automatic retry`.
Both are EXPECTED safeguards, not passing GPU preflight. Original registration,
handoff and RED receipt hashes verified unchanged. Only source deltas against
r1 handoff are `scripts/grm_x1_gpu.py` and `tests/test_grm_x1_campaign.py`.
Receipt: `artifacts/grm_x1/receipts/payload_amendment_01_handoff_check.json`.

Preserved original commands as `artifacts/grm_x1/lead_commands_r1.txt` and
refreshed `artifacts/grm_x1/lead_commands.txt` with runnable CPU inspection
commands, expected preflight failure and explicitly BLOCKED GPU commands.
New create-only `artifacts/grm_x1/payload_amendment_01_handoff.json` binds amended
sources and CPU receipts; it is a review artifact, not an activated campaign.
This amendment does not change the campaign controller or retry semantics.
Lead continuation must bind the new source identity and account for the old
285-second reservation while preserving the RED evidence. That accounting
number is the registered reservation, not the observed 34.6-second worker
failure reported by the lead.

Current report: `artifacts/grm_x1/AMENDMENT_1_REPORT.md` supersedes r1's
zero-GPU-reservations status and source inventory for this repair only.
RED: original oracle cell remains RED; no GPU rerun, recall or latency claim;
native-call hard-bound residual unchanged. Not claimed fixed on the card.
Process safety: no git, subagents, background waits, GPU calls, kills, service
writes, sibling edits or existing core-code edits. Author GPT-6; exact model
deployment ID unavailable; reasoning effort high (requested and followed).
Target reader remains registered openai/gpt-oss-20b with low inference effort;
no confusion with this author session's high reasoning effort.

## 2026-09-08 — amendment 2: pre-gate registration

Read HOUSE_RULES and immutable orders/GRM_X1_AMENDMENT_2.md. Created
artifacts/grm_x1/continuation_02_registration.json before edits or gates;
it binds the order, pre-edit sources, original claim/RED receipt bytes,
and five named continuation gates. CPU author verification only; lead runs
the card under the explicit no-GPU seat rule. No core or worker changes planned.

Lead-authorized budget: 2565 + 285 = 2850 worker reservation seconds
(0.7916667 GPU-hours), exceeding COMMON 2700 seconds by 150 seconds.
This is a reservation limit, not measured GPU consumption. Preserve r1
claim and RED receipt; new r2 receipts/claims use receipts/r2 and claims/r2.
Manifest and checksum are create-only; any r2 RED or incomplete claim stops
without a second retry. Natural arm keeps its original positive-oracle gate.

Prior art: house X1 controller (2026), verified in local source, supplies
SHA-addressed receipts, exclusive creation, reservations and stop behavior.
SHA-256 antecedent NIST FIPS 180-4 (2015), unverified — lead to check; search
terms: NIST FIPS 180-4 2015 SHA-256. New integration: amendment-bound epoch
partition and exactly one paid retry; no new hash or experiment algorithm.
Hash bindings assume trusted local code/order storage; they are not signatures.

## 2026-09-08 — amendment 2: sealed continuation and CPU results

Ran `PYTHONDONTWRITEBYTECODE=1 python3 scripts/grm_x1_campaign.py seal-continuation`
after final source edits and before gates. Create-only manifest
`artifacts/grm_x1/continuation_02.json`, SHA `a49cd7a3453fde39c5a8b575fa0a0e92d5d134c2fc948b362ba8dab362204c20`.
No sealed source changed afterward. Receipt/claim partitions are
`receipts/r2/gpu_{cell}_{content_sha}.json` and
`claims/r2/{cell}_{continuation_sha}.json`; receipt fingerprint is continuation SHA.

Ran `PYTHONDONTWRITEBYTECODE=1 python3 scripts/grm_x1_cpu.py baseline`: **174 passed**,
2 existing SWIG warnings (author unit tests). Baseline receipt:
`artifacts/grm_x1/receipts/cpu_baseline_b1a5012ed9e8c8d294b710e1a02731a93e153a354b20d4a09d650a7d3eeb222c.json`.
Five named gate results: continuation accepted PASS (1); forged continuation
refused PASS (9); stale continuation refused PASS (1); second retry refused
PASS (3: RED, incomplete claim, completed claim); r1 receipt untouched PASS (1).
Exact test names/evidence in `artifacts/grm_x1/receipts/continuation_02_gates.json`.
Original-handoff compatibility and synthetic all-nine-cell ordered controller
simulation also PASS. The latter reserved 285 historical + 2565 continuation
= 2850 synthetic seconds; no GPU observations are inferred. Mutation suite
not rerun because core source and existing mutations are unchanged.

Live CPU `bash scripts/grm_x1_lead_gpu.sh preflight`, `--dry-run`, `summary`,
and `bash -n artifacts/grm_x1/lead_commands.txt` all returned 0. Receipt:
`artifacts/grm_x1/receipts/continuation_02_lead_checks.json`. Actual next cell
is oracle_m1_s0; dry-run enumerates nine cells in order and flags only its
retry. Active r2 verdict PENDING, failed=false, zero r2 claims/receipts/rows;
r1 RED remains historical and unchanged. Reservation used: r1 285, r2 0;
authorized total cap 2850 seconds (0.7916667 GPU-hours, explicit lead overage).

Refreshed executable lead_commands.txt preserves all stop conditions and
conditional natural gate; old commands saved create-only as
lead_commands_amendment_01.txt. Full file/line inventory, exact commands,
prior-art attribution and residuals: artifacts/grm_x1/AMENDMENT_2_REPORT.md.
Prior art remains house X1 (2026) verified local controller, and NIST FIPS
180-4 (2015) SHA-256, unverified — lead to check; new work is epoch partition
and lead-authorized one-retry integration, no novel algorithm claimed.

RED: r1 `TypeError: 'NoneType' object is not iterable` remains visible; not
claimed fixed on the card. GPU campaign execution is pending at the lead,
as amendment 2 explicitly forbids GPU use in this seat. No prediction or
latency outcome inferred. Existing non-returning-native-call hard-bound
limitation unchanged. No deviations from dispatched scope; no independent
blind verification. No git, subagents, GPU, background jobs/waits, kills,
service writes, sibling edits, core changes or worker changes. Author GPT-6
(exact deployment variant unavailable), reasoning high; reader GPT-OSS-20B
low unchanged and not loaded.

## 2026-09-09 — amendment 3 registration before edits/gates

Create-only `artifacts/grm_x1/continuation_03_registration.json`, SHA 0262299553a7123b297488383de9cfd4bf3922a1195e9f23e7068ac4f00a4729. Immutable order: `orders/GRM_X1_AMENDMENT_3.md`. One query per unit, in original cell/global-ordinal order: 12 per oracle cell, 24 per natural cell; 144 total. Capture, install, all paired forwards and score stay inside each lease. Work alarm 280 s, worker allowance 285 s, outer 590 s, cooldown 30 s. No per-unit wall known; r2 timeout is cell-level evidence only. Budget 5400 s including 570 historical; r3 actual completed worker wall, outstanding claims 285 s, admission reserves 285 s. First successful unit projects 144 units; NON_FIT_BUDGET requires lead decision. Unit RED advances; explicit retry once, second RED stops; timeout NON_FIT never chunks forward. Historical claims/receipts and prior manifests bound before edits. Gates/criteria frozen in registration.

Prior art: house X1 (2026), locally verified exclusive claims, SHA receipts and query pairing; new query-sized checkpoint/lease integration and budget projection, no new scientific algorithm. NIST FIPS 180-4 (2015), unverified — lead to check: NIST FIPS 180-4 SHA-256. No literature claim independently verified here. Author GPT-6, exact deployment variant unavailable; effort high. GPU reader openai/gpt-oss-20b low remains unloaded.

## 2026-09-09 — amendment 3 sealed implementation and CPU results

Sealed continuation_03 before gates: `e8d854f6a6eabbe27602992ba6a804a9b5d4a102417fe726fc8c746d0bcd5b10`; path
`artifacts/grm_x1/continuation_03.json`. Source hashes unchanged after seal.
Added unit harness and single-query GPU wrapper, routed campaign CLI, included
r3 CPU tests and isolated historical r2 fixtures, refreshed lead loop. Exact
files/lines and tests: `artifacts/grm_x1/AMENDMENT_3_REPORT.md`.

`PYTHONDONTWRITEBYTECODE=1 python3 scripts/grm_x1_cpu.py baseline`: 196 passed,
2 existing SWIG warnings, pytest 3.74 s; 22 r3 cases. Baseline receipt:
`artifacts/grm_x1/receipts/cpu_baseline_0d8b663a1e90e0204f3a43c9b0d0eb20b7c35420ca62d40916768519ca929f07.json`.
Named gates/results: `receipts/continuation_03_gates.json`. Preflight, --dry-run,
summary and bash syntax PASS; 24 historical files hash-identical; receipt
`receipts/continuation_03_lead_checks.json`. Additional lead-loop CPU protocol
gate registered before execution in `continuation_03_shell_gate_registration.json`: all-unit/one-RED path 144 calls exit 0, budget NON_FIT one call exit 2, natural lock 73 calls exit 0, registered stop one call exit 1, all PASS. Receipt `receipts/continuation_03_shell_gates.json`. No GPU/stub confused with real worker evidence.

Live state: r3 PENDING, zero claims/receipts/rows, 570 historical charged seconds
of 5400 authorized; estimate null. Next unit oracle_m1_s0__alias_00. r1 and r2
RED unchanged. Evidence correction: r2 actual receipt has 17 partial rows,
not the empty list stated by the lead prose, and wall 280.00282101891935 s;
no fresh r3 unit timing is inferable. Error verbatim: TimeoutError: GRM-X1
worker reached its 280 s work rail (285 s outer worker allowance). r1 error:
TypeError: 'NoneType' object is not iterable. No GPU fit/accuracy fix claimed.

Prior art: house X1 (2026), verified local exclusive creation, SHA receipts,
leased workers and paired query templates; new query-unit checkpoint/resume
and planning integration only. NIST FIPS 180-4 (2015) SHA-256, unverified —
lead to check those search terms. No new scientific algorithm claimed.

No scope deviation, git, subagents, GPU probes/runs, background jobs/waits,
kills, service writes or core/sibling edits. Foreground CPU only. Existing
cooperative-timer hard-preemption residual retained, not tested on GPU.
Author verification only; unchanged-core mutants not rerun; blind verification
left to lead. Author GPT-6 (exact variant unavailable), effort high; reader
openai/gpt-oss-20b low unchanged and unloaded.
