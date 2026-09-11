# GRM-C4 append-only implementation ledger

## 2026-09-08 — before registration/gates
Read /mnt/Shared/HOUSE_RULES.md and supplied AGENTS/order. Inspected WC1/EB1,
RS3 capture and seating, repository split guard and manifest sources. Read
.git pointer as plain text only: target refers to worktrees/grm-c4. No git
commands. No GPU probe, model load, subagent, background wait, process signal,
lock manipulation, or external service activity.

Source inspection evidence: _guard_deposit_width at core/graft_repository.py
1210 accepts budget; default uses arena width. Explicit fit repair uses that
same guard. RS3 live capture derives from arena.live_shift and near-live ends
at live_shift-1. WC1 varies live_shift with width, so width64 historical cell
is not a fixed-numeric-geometry control. The archive is in the main repository,
not this worktree. Existing source comparisons matched before modifications;
registration will bind the full comparison including external frozen artifacts.

Implementation: new scripts/grm_c4_adapter.py and grm_c4_campaign.py; new tests
only. No core edits, respecting additive-only scope. Production guard target:
core/graft_repository.py:1210; no grm_three_pass guard exists to modify here.
The knob exists only within the opt-in adapter context; ambient env alone
cannot alter a normal production run. Explicit fit budgets retain behavior.

Prior art at code sites and in plan: verified house LSR-P2C, RS3, WC1, EB1,
DET1 (2026). Borrow their exact splitter, geometry, scoring, session persistence
and lease. C4 contribution: independent experimental controls and token-seat
measurement. External Fisher (1935) factorial lead is unverified — lead to
check, search Fisher factorial experiments 1935. No algorithm novelty claim.

Planning estimates: each new cell 1600s (4x60 + 4x80 + 13x80), two cells 3200s.
These are unvalidated planning assumptions, not measured GPU speed. Full fixed
geometry design would add a 1600s width64 replacement, total 4800s: NON_FIT
against 3600s mission cap. Register all four with equal standing; preserve RED
missing-control status and do not relabel historical width64 as comparable.

## 2026-09-08 — initial CPU receipt and additive amendment A1
Registration written before gates: SHA-256
7fb561c70dd8c04abc18ef99a7a2a5ec4d37cbc2a22b1de1f89ab2119bd285e8.
Initial gate: `python -m pytest -q tests/test_grm_c4_campaign.py
 tests/test_grm_lsr_p2c_split_descent.py` -> 55 passed, 2 upstream SWIG warnings,
4.71s; full output cpu_gate_initial.txt. `--dry-run` passed and enumerated
4 logical cells, 21 units per new cell, 3200s new-cell and 4800s full-design
planning estimates. All 13 WC1 registration source records matched after
read-only archive/main-tree path resolution. Existing manifest covers 37 nodes.

Handoff inspection found the imported supersession loader uses module-level
NATIVE_LIB rather than the frame's native path. In this worktree that binary
is absent. A1 binds that loader's binary and model to the same verified shared
paths already used by census/LH. Also added explicit observed capture asserts,
resumed-file hash checks, an immutable worker-start marker to prevent retry
of a killed worker's partial session, and per-battery measurement summaries.
These are implementation corrections/hardening, not changed fixtures, bars,
predictions, geometry, schedule, or budget. Registration is NOT edited.
A1 binds updated executable sources with its own SHA sidecar before the next
gate. Tests add loader-path binding, synthetic rail RED/no retry, and imported
LH 13-segment saved-state copy coverage. Prior art unchanged: house WC1/EB1/
DET1 (2026); reuse their imported code, C4 adds validation and bindings.

## 2026-09-09 — A1 receipt and final safety amendment A2
A1 CPU gate: 58 passed, 2 SWIG warnings, 4.69s (cpu_gate_a1.txt).
Final process-safety inspection identified an unreceipted outer-killed worker
could leave unknown GPU usage: its marker prevented its own retry, but a
worker in another cell could still start. A2 blocks the whole campaign if any
start claim lacks a completed receipt. Also enforce measured <=285s before
PASS and require score sources equal the active amended inventory. Amendments
now form a checked SHA chain. Added a CPU test that an abandoned worker blocks
a different cell without launching GPU code. No registration, fixtures,
thresholds, decision rules, geometry, or estimates changed. Prior art: DET1
persistent claim/receipt bookkeeping (house, 2026); C4 fail-closed accounting.

## 2026-09-09 — final CPU/handoff receipts
Final gate: 59 passed, 2 upstream SWIG warnings, 4.65s; preserved verbatim in
cpu_gate_final.txt. Final dry-run succeeded with active A2 source inventory;
original registration SHA and amendment chain verified. Final check: all 13
historical WC1 fingerprints still match. No existing source file changed.

Wrote BLOCKED_REPORT.md and blocked_report.json with evidence limits, historical
scores, missing actual historical token residency, the fixed-geometry width64
control gap, and 3200s two-cell / 4800s full-design unvalidated estimates.
lead_commands.txt contains 42 interleaved worker commands and 42 separate
foreground 30s cooldown commands, then CPU scoring for both new cells. Syntax
checked with bash -n. Outer hard timeout is SIGKILL at 590s for only the worker
started by that command; commands were NOT executed. Imported lease's own
285s alarm is the normal worker stop. No marker/lock clearing or process-name
signals. All commands must be separate bounded foreground invocations.

Process safety: no git, subagents, GPU activity, lock access, live service
changes, background waits/jobs, or signals in this sandbox. Source manifest
and checksum receipt below are artifact integrity checks, not GPU acceptance.
Author GPT-6 (Codex; finer deployment id not exposed), high per order.
Prior art remains house LSR-P2C/RS3/WC1/EB1/DET1 (2026); external Fisher (1935)
lead unverified. No novel algorithm claim. GPU/full-factorial verdict RED and
INCONCLUSIVE, not claimed fixed. Blind verification remains lead-owned.

## 2026-09-09 — lead amendment 1, before gates (artifact A3)
Read HOUSE_RULES, AGENTS and immutable orders/GRM_C4_AMENDMENT_1.md.
Lead accepts r1 width64 numeric-geometry mismatch and authorizes the complete
fixed cross at 4800s GPU cap (1.333 GPU-h), superseding the 3600s cap only.
Register c64_w64 NEW at 1600s with the same 21 units, 9/10/14 batteries,
geometry, prediction/rejection, 285s worker / 590s outer / 30s cooldown.
c64_w96 and c96_w64 remain 1600s each. Total new GPU planning estimate 4800s.
WC1 width96 is CITE_WC1: RS3 registration derives sink19, capture/live115
and near-live head114. Its 13 worker receipts bind the width96 frame and
live/near-live levers; archive source fingerprints and quality receipt SHAs
are checked by the amendment writer. Historical scores: width96 9/9,9/10,13/14;
width64 8/9,10/10,14/14. Register the new c64_w64 versus historical comparison
to test survival of that benefit under fixed geometry; no prediction changed.
Implementation A1/A2 and base registration remain immutable. A3 is this lead
amendment 1, not a rewrite of the existing implementation A1. Amend binding
to apply only authorized fields and verify the order SHA. CLI, worker and
score enumerate the active registered schedule. Per-cell score reports the
diagonal deltas without assigning a factorial verdict. Refresh lead_commands
for complete-cell ordering and immediate per-cell scoring; preserve r1 copy.
Prior art: verified house RS3/WC1/EB1/DET1/C4 (2026), borrowed geometry
derivation, SHA chain, cell enumeration, deltas, resume and lease; this work
adds authorized controls and evidence pins, no novel algorithm claim. Original
Fisher (1935) external lead remains unverified — lead to check, search Fisher
factorial experiments 1935. Code-site and final-report annotations required.
RED reasoning: 4800s equals the unvalidated estimate and provides no headroom.
The unchanged full285s pre-worker reserve would require 5005s at the exact
planning unit times for the last unit; completion is not guaranteed at cap.
Do not loosen the reserve, retry or enlarge leases. Cooldowns add 1890s of
foreground wall time outside GPU usage. No r1 worker receipts/start claims
exist; no prior GPU receipts are being revalidated or relabeled.
Process safety so far: no git, subagents, GPU, background waits/jobs, process
signals, lock access, live services or external delivery. Lead commands retain
r1 outer timeout for only its own worker; no such command is executed here.
Author: GPT-6 (Codex; finer deployment id not exposed); high per order.

### Registration writer correction before A3 existed
First invocation exited 1 before writing the immutable amendment: at
scripts/grm_c4_amendment_1.py:52, `levers = row['shard']['rs3_levers']`
raised `KeyError: 'shard'`. Source diagnosis: WC1 score summaries also carry
frame_binding but are not worker receipts. Select the three explicit worker
schemas (sup_fixture, census_shard, longhorizon_shard); score source hashes
remain checked separately. No gate or GPU ran and no registered threshold
changed. Prior art: WC1 typed receipt schemas (house, 2026).

## 2026-09-09 — lead amendment 1 registered, CPU gates and handoff
A3 immutable SHA-256: 168acfe429f3b1a288ea74c0bc491ae3359c297a4e363d50fe4eedf7a2677c42.
Order SHA: 2bb3e59d54f62895a072718613dc4768aae6513b49e7203c94310f83936d03f7.
Base/A1/A2 SHA chain verified; A3 written before gates.
`python -m pytest -q tests/test_grm_c4_campaign.py tests/test_grm_lsr_p2c_split_descent.py`
-> exit0, 66 passed, 2 warnings in5.63s (cpu_gate_lead_a1.txt); includes new
diagonal execution eligibility, timeout RED/no retry, order/amendment drift
refusal, unchanged bars/geometry, historical pin, dependency schedule checks.
`python scripts/grm_c4_campaign.py preflight --dry-run` -> exit0,
dry_run_lead_a1.json: 4 logical cells,3 executable,63 units,4800s estimate.
`bash -n artifacts/grm_c4/lead_commands.txt` -> exit0. 63 explicit workers
and63 foreground cooldowns, with each of3 scores immediately after its cell.
No GPU commands executed. Prior art unchanged: local house RS3/WC1/EB1/DET1/C4
(2026), external Fisher1935 lead unverified. No novel algorithm claim.
Updated BLOCKED_REPORT.md/blocked_report.json; preserved r1 reports and
lead commands; original SHA256SUMS saved as SHA256SUMS_r1. Current checksum
manifest will be regenerated after this append and verified. Old r1 manifest
records historical bytes and is not a current-tree integrity gate.
GPU/full-factorial verdict RED/INCONCLUSIVE pending lead execution; budget
headroom and historical token-residency limitations remain. Not claimed fixed:
t33,Juniper,GPU geometry parity,historical benefit survival,chunking causality.
No deviations beyond authorized cap/cell and noted pre-registration writer
correction. No git,subagents,GPU,background jobs/waits,signals/kills,lock access,
service changes or external delivery. Model GPT-6 (Codex; deployment suffix
not exposed); reasoning high per order.

Final integrity verification: all32 current SHA256SUMS entries passed; active
A3 source binding passed,13/13 historical fingerprints match,zero C4 worker
files. Base-source comparison includes adapter differences already present
in r1; adapter current SHA2638f87714f32af58cce32065a635107058c09de8e7635ae1d9da46303fd41e4
matches the preserved r1 SHA256SUMS. This turn changed only the existing C4
campaign/test files and added the amendment writer, besides scoped artifacts.

## 2026-09-09 — lead amendment 2, diagnosis and gates registered before implementation
Order orders/GRM_C4_AMENDMENT_2.md remains immutable. Read HOUSE_RULES.
Source/receipt evidence CONFIRMS [40,48) has four fact and four supersede turns,
no probe or answer; saved instrumentation is exactly turns0..47. Stored error
is exactly `AssertionError: No serving observed` (exception-class prefix is
part of receipt format). Full probe plan:5,9,13,16,19,22,24,26,30,33,70,80,90,100.
The later-cell prediction is conditional: existing global RED guard prevents
any later worker starting. Currently only c64_w96 has receipts:13PASS,1RED.
Registered gates and isolation decision in lead_a2/pre_registration.json;
original top-level receipt hashes in lead_a2/original_receipts_before.json.
Use additive entry point and nested A4, leaving original binding/code/commands
unchanged for the live lead chain. New rerun receipts and session copies are
create-only in lead_a2/runs; commands enumerate only evidenced eligible suffixes.
Prior art: verified local C4/WC1/EB1/DET1 house systems(2026), reuse SHA-chain,
create-only claims, state-copy resume and scorer; add per-cell guard and exact
one-time recovery selection per this order. No novel algorithm claimed.
No git,subagents,GPU,background waits,signals,lock or service manipulation.
Author GPT-6(Codex; specific deployment id not exposed), effort high(order).

## 2026-09-09 — lead amendment 2 registered and CPU PASS
A4 SHA014ae519a3b27ff0598d1d5a5ae3e4dcb7c3785ae2a52b7cf966d6c6fd10a16e; nested opt-in lead_a2/amendment_a4.json.
93 tests passed,2 upstream SWIG warnings,6.13s; lead_a2/cpu_gate.txt.
Read-only validation:29 original files unchanged(13PASS,1RED);115 lastPASS
state files checked; original A3 binding and A4 preflight pass. No GPU run.
Exact eligible commands:lead_commands_a2.txt,8 c64_w96 recovery units then
score; other cells have no receipts and remain outside recovery eligibility.
Prior art:local C4/WC1/EB1/DET1(2026),same reuse/contribution as preregistration.
Full per-order ledger and report:lead_a2/IMPLEMENTATION_LEDGER.md and
lead_a2/FINAL_REPORT.md. RED/GPU recovery pending,not claimed fixed.
No git,subagents,GPU,background waits,signals,kills or original runs writes.
GPT-6(Codex;specific deployment id not exposed),high(order).
