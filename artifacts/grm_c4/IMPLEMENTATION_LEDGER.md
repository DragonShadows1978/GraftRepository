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
