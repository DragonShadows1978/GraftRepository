# GRM-C7 lead amendment 1 — completion report

PASS for amendment binding and CPU gates. Campaign/GPU acceptance remains NOT_RUN.

## Amendment and source identity

- Amendment: `artifacts/grm_c7/amendment_lead_1.json`
- Amendment SHA256: `e7c8bb527953ae82512ce1fa13e8118288a7223d162b3b15f5f3c6c10f1984d5`
- Order: `orders/GRM_C7_AMENDMENT_1.md`, SHA256 `5e23a7f9fd06ac9442129fcbf28345a5ff88b2e149a2843171499665ca8d7215`
- Original registration SHA256: `7bc0a0d1ad4cca02a02ae7653ce4b841a8f2fcd0c1e8e051d46b5f5ed7c70c1c`
- Previous A2 SHA256: `41b70efdf1735684322a8762c2b716348180bb51ec48a6cc05c630b13c2285fb`
- Registered old `core/graft_repository.py` SHA256: `2bb38b8efd8f75f3d189ade592133f3a40b245182b98c04b485976d5b2ed4311`
- Amended current source SHA256: `fc6b9448efb45c29d5d2271fe929e3e4b6519867567cf1c69588b2a99e4773db`

Source inspection: the only pre-existing registered input delta was this core
file. The supplied SCOUT-FIX-2 adds deepcopy at line 38 and child capture
inheritance at line 1005 (copy at 1009). Old bytes recovered from C2's archived
SCOUT-FIX-2 source match C7's registered hash. Both the local archive and
`source_delta_lead_1.patch` are bound in the amendment. No product source was
edited by this seat. Commit 6b2d3c8 provenance is the lead's statement; no git
command was used to verify commit or branch identity.

Acceptance wording is UNCHANGED and not weakened. The original restart phrase
is: "same per-probe exact/wrong/abstention scores at 100/200, same persisted
metadata, different process identity". The existing all-node projection in
`scripts/grm_c2_cells.py:76` includes capture, so split children are within this
comparison. Inherited child fields now exist and must survive. Fixtures, probes,
oracle, answer/residency/fold/restart thresholds, cells and budgets are unchanged;
A2 controlled paging remains in effect. This is a source binding amendment,
separate from immutable registration/A1/A2 and the new immutable order.

## Files and behavior

- `scripts/grm_c7_common.py:27`: mandatory lead amendment verification; binds order,
  registration, A2, authorized old/new core, before archives, exact change scope,
  unchanged acceptance and lead command hash. The normal effective-input check
  at line 137 then verifies the amended inputs against live files.
- `scripts/grm_c7_run.py:25`: receipt/checkpoint binding verifies the amendment and
  carries `amendment_lead_1_sha256` and `amended_source_sha256`; dry-run adds both
  at line 58. Worker/controller/reservation/fold-start/summary/checkpoint paths
  already use this binding. GPU receipt emission was inspected, not GPU-tested.
- `tests/test_grm_c7_lead_1.py:47`: 16 forged/stale/missing amendment cases across
  verifier, dry-run, binding and summary. Critical-field forgeries are tested
  after recalculating the adjacent checksum, not only as checksum corruption.
- `tests/test_grm_c7_lead_1.py:92`: three live-source/order/command mismatch cases.
- `tests/test_grm_c7_lead_1.py:100`: exact source/amendment receipt and checkpoint
  bindings, rejection of historical/stale binding, unchanged 39 cells per arm.
- `tests/test_grm_c7_lead_1.py:130`: production CPU split/save/reload plus existing
  restart projection detects loss or alteration of child capture.

Hash checks establish integrity against these local pinned inputs, not a digital
signature or authentication against an actor able to replace the verifier itself.

## CPU gate receipts

Exact commands are registered in the amendment. All ran with
`CUDA_VISIBLE_DEVICES=''`; pytest also used `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

| Gate | Result | Receipt |
|---|---|---|
| C7 original + A1 + A2 + lead_1 suites | 44 passed, 2 warnings, 2.25s | `cpu_lead_1_c7.log` and `.json` |
| `tests/test_grm_scout_fix2_capture.py` | 10 passed, 2 warnings, 0.32s | `cpu_lead_1_scout_fix2.log` and `.json` |
| Registered scorer mutation gate, after baseline | 5/5 non-error mutants killed; 1.0 >= 0.80 | `cpu_lead_1_mutations.json` |
| `python scripts/grm_c7_run.py --dry-run` | 39 A + 39 B cells; B NON_FIT; exact registered cells | `dry_run_lead_1.json` |
| Byte/hash audit | PASS; 428 effective inputs verified | `final_integrity_lead_1.json` |

Both suites emitted SWIG type DeprecationWarnings, preserved in full logs,
including the exit-time swigvarlink warning. No suppression or tolerance change.
These are author CPU suite/mutation/CLI evidence, not blind verification or E2E
model evidence. Summary receipts for both arms remain NOT_RUN.

## lead_commands.txt

`artifacts/grm_c7/lead_commands.txt` is BYTE-UNCHANGED against its before archive,
verified by direct byte equality and SHA256 before and after the work:
`a0320cf79b5d2ba3f86f16e13673802fa316e65d0d66db2123683dc09f695ee7`.
The queued command file was not executed, rewritten or chmodded. Its existing
dry-run preflight now reads the mandatory lead amendment through the runner.

## Prior art

Local C7 A1/A2 and C2 sha-bound manifests, immutable checkpoints and all-node
capture projection (GRM contributors, 2026), verified from local files. Taken:
hash chain, explicit scope allowlists, before archives and capture comparison.
This work adds the exact SCOUT-FIX-2 authorization and its C7 receipt/checkpoint
binding. No new algorithm. No prior art known to me for this exact composition.
Code comments and the ledger carry the same annotation. The supplied core delta
credits B3/RS3 capture projection and LSR-P2C payload slicing (project contributors,
2026); that algorithm was already implemented by SCOUT-FIX-2, not this seat.
No external literature verification claim.

## Deviations; RED; process safety; model

Deviations: none from this amendment order. The original scorer-mutation driver
was called through its existing mutations() function with a new receipt name to
preserve historical CPU receipts. Exact script gate commands appear in the JSON.
No registered thresholds or historical artifacts were overwritten.

RED: campaign/GPU model, paging and restart acceptance NOT_RUN; B remains NON_FIT.
No product quality failure is claimed fixed. Author tests are not independent
blind review; lead verification remains outstanding. The pre-amendment refusal
was `ValueError: INPUT_SHA_MISMATCH: core/graft_repository.py`; the source-binding
treatment now passes the registered CPU gates. No CPU gate failure occurred.

Process safety: no git, subagents, GPU/model loads, background jobs/waits, service
changes, signals or process kills. Foreground CPU subprocesses completed normally.
Only this worktree and task-owned temporary files were written; C2 archive reads
were read-only. `artifacts/grm_c7/cells` remained absent at the final audit.

Model: GPT-6 per system identity; exact deployment/API model ID unavailable.
Effort: high as requested; effective runtime setting is not independently exposed.
