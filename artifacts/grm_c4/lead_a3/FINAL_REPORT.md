# GRM-C4 amendment 3 — CPU preparation PASS; GPU results pending

Amendment: `artifacts/grm_c4/lead_a3/amendment_a5.json`
SHA256: `ffc9a66afe9c27ae9d56b6d5b35ad119666db60d6f9459687dec67f35126cdf1`

Bound to immutable `orders/GRM_C4_AMENDMENT_3.md` and recovery
`lead_a2/amendment_a4.json` (SHA256
`014ae519a3b27ff0598d1d5a5ae3e4dcb7c3785ae2a52b7cf966d6c6fd10a16e`).
Executor is additive A4 sibling `scripts/grm_c4_remaining_a3.py`.
Original and queued A4 executors/manifests/commands remain byte-unchanged.

42 units: per cell 4 supersession + 4 census + 13 long-history segments,
`c96_w64` then `c64_w64`. Registered estimate1600 s/cell,3200 worker s total.
Command file includes43 cooldowns (initial gap plus each unit):1290 s outside
GPU budget. Estimated command elapsed4490 s plus CPU scoring/summary overhead.
These are planning assumptions, not measured GPU performance.

The recorded original cost is1084.0746343242936 s, including RED lh06 cost
73.72084314282984 s. At inspection lh07–13 have no worker receipts; A4 has no
completed workers. Eight queued A4 recovery units have registered estimate640 s.
Combined projection4924.075 s exceeds cap4800 by124.075 s. Per-worker reserve285 s
can stop the chain earlier. Runtime accounting charges all original, A4 and
new walls once; retains cap4800, lease285, outer590, cooldown30, lock wait0.
The projection is a budget risk, not a measured NON_FIT result. No cap waiver.

Files/lines:

- `scripts/grm_c4_remaining_a3.py:32`: order/A4/source/SHA binding and scope checks.
- `scripts/grm_c4_remaining_a3.py:87`: prior scores and bound worker validation.
- `scripts/grm_c4_remaining_a3.py:103`: three-location wall accounting.
- `scripts/grm_c4_remaining_a3.py:134`: ordered dependencies and no overwrite.
- `scripts/grm_c4_remaining_a3.py:161`: batteries imported unchanged; A4 guard,
  complete bound state before resume; claim and receipt create-only.
- `scripts/grm_c4_remaining_a3.py:212`: original C4 scorer imported with one
  isolated root for all batteries and score.json. No scoring arithmetic edits.
- `scripts/grm_c4_remaining_a3.py:224`: registered probe membership and outcomes.
- `scripts/grm_c4_remaining_a3.py:241`: bound four-cell summary, WC1 full14-probe
  diagonal, fixed/historical64 component deltas and named Juniper/t33 outcomes.
- `scripts/grm_c4_remaining_a3.py:321`: exact foreground schedule generation.
- `scripts/grm_c4_register_a3.py:14`: create-only registration and snapshot.
- `tests/test_grm_c4_remaining_a3.py:106`: dry schedule and exact commands.
- `tests/test_grm_c4_remaining_a3.py:128`: forged/stale amendment refusals.
- `tests/test_grm_c4_remaining_a3.py:152`: prior receipt bytes and bindings.
- `tests/test_grm_c4_remaining_a3.py:161`: full synthetic battery/resume/score path.
- `tests/test_grm_c4_remaining_a3.py:201`: state/turn/serving/rail failures.
- `tests/test_grm_c4_remaining_a3.py:232`: original RED charged, cap/cooldown/orphan stops.
- `tests/test_grm_c4_remaining_a3.py:292`: synthetic four-cell outcome cases.
- `tests/test_grm_c4_remaining_a3.py:314`: missing/mixed/stale evidence refusal.

CPU evidence: `cpu_gate.txt`:85 passed,2 SWIG deprecation warnings,10.76 s,
plus exit-time swigvarlink warning. Original C4 and A4 suites included.
`dry_run.json`:42 units and3200 s. Shell syntax gate exit0.
`validation.json`:29 pre-existing receipt/claim JSON files byte-unchanged;
no real worker receipts added; original and A4 source bindings still valid.
`synthetic_four_cells/summary.txt` and `synthetic_four_cells/a5/cross_summary.json`:
CPU synthetic executable cells plus the pinned archived WC1 fourth cell.
Synthetic prediction success is NOT a campaign result. No blind review claimed.

## Exact lead commands

After the already queued `lead_commands_a2.txt` completes successfully:

```bash
cd /mnt/ForgeRealm/wt/grm-c4
bash artifacts/grm_c4/lead_commands_a3.txt
```

The file is SHA-bound in A5 and contains every explicit unit command, cooldown,
per-cell score and final summary. Do not overlap it with the queued recovery.
It starts with dry-run and30 s cooldown. Every worker is wrapped with
`timeout --signal=KILL 590s` as already registered; lead execution only.
The final commands are:

```bash
python scripts/grm_c4_remaining_a3.py score --cell c64_w64
python scripts/grm_c4_remaining_a3.py summary
```

Real outputs will be `lead_a3/runs/{c96_w64,c64_w64}/score.json` and
`lead_a3/cross_summary.json`. The c64_w96 score comes from A4, and96/96 from
WC1's archived full13/14 row, never the4/4 distant spot-probe subtotal.
Missing or stale evidence refuses summary and remains INCONCLUSIVE.

## Prior art

Verified local C4/A4, WC1, RS3, EB1 and DET1 (house,2026): reused SHA manifests,
create-only claims/receipts, dependency-ordered saved-state resume, per-cell
serving guard, lease accounting, original battery scorers and per-probe
comparisons. New contribution is executor scope and cross-summary assembly.
No novel algorithm claimed; no external literature verification claimed.
Annotations are at code sites and in IMPLEMENTATION_LEDGER.md.

Deviations: none from this amendment. A sibling executor is explicitly allowed;
it avoids changing A4's frozen source closure. No new scoring thresholds.

RED: GPU recovery and two-cell E2E unrun by this seat. The original
`AssertionError: No serving observed` remains on disk and charged. Not claimed
fixed by these CPU gates. Prediction and live factorial verdict remain pending.
Historical WC1 actual token residency remains unavailable as registered.

Process safety: no git, subagents, GPU calls, process signals, lock changes,
shell background jobs or edits to original receipt trees. A foreground CPU test
command yielded once at tool level and its same session was collected to exit0.
All exported fixture data is isolated beneath lead_a3/synthetic_four_cells.

Model id: GPT-6 (Codex; exact deployment identifier not exposed). Effort: high.
