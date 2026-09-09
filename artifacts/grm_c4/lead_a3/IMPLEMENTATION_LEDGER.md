# GRM-C4 lead amendment 3 implementation ledger

2026-09-09 — Plan is immutable `orders/GRM_C4_AMENDMENT_3.md`.
Read HOUSE_RULES and worktree AGENTS. No git or subagents invoked.

Pre-registration inspection (evidence class: local source/receipt read):
A4 binding validates unchanged; A4 recovery runner excludes unevidenced cells.
Original campaign guard refuses continuation after original RED.
Original tree has 14 finished worker receipts, total 1084.0746343242936 s;
only stored RED is c64_w96/lh06 at 73.72084314282984 s. lh07–13 have no
worker receipts (pre-lease dependency refusal), not invented zero-wall REDs.
A4 run tree has no finished workers at inspection. Both remaining cells have
no receipts or claims. Existing paths are never changed by this order.

Implementation before registration: additive `grm_c4_remaining_a3.py` sibling,
`grm_c4_register_a3.py`, and CPU author gates. Import original scorer and A4
serving/state guards. Receipt root lead_a3/runs; all three batteries and scorer
use this root. Bound A4 context is retained for mixed legacy/recovery c64_w96.
Dependency order includes prior cell score and every earlier unit. All three
roots contribute wall accounting; original authorized RED work stays charged.
No retry, overwrite, abandoned claim clearing, budget extension or rail change.

Prior art: verified local C4/A4/WC1/RS3/EB1/DET1 house systems (2026), read
from their source. Borrowed SHA binding, create-only claims, ordered state
resume, per-cell guard, scorer, accounting and per-probe comparisons. This
order adds executor scope and summary assembly. No novel algorithm claimed.
No external technique introduced; no literature verification claim.

Gate criteria will be registered in pre_registration.json and amendment_a5.json
before running tests. Author tests are a baseline; blind review belongs to lead.
Planning: 42 new units, 3200 worker s; 43 cooldowns including initial gap =1290 s.
Eight queued A4 recovery units add registered estimate640 s. Snapshot projection
1084.0746343242936+640+3200=4924.074634324294 s exceeds cap4800, before the
285 s per-worker reservation constraint. This is an unvalidated planning
projection, not permission to exceed cap or a measured non-fit gate result.

Registration receipt (before any gates):
`python scripts/grm_c4_register_a3.py` created amendment_a5.json, 62728 bytes,
SHA256 ffc9a66afe9c27ae9d56b6d5b35ad119666db60d6f9459687dec67f35126cdf1.
Order, previous A4, executor/source closure, schedule, scores and budget bound.
No implementation edits or threshold changes after registration were needed.

CPU suite receipt: `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m
pytest -q -p no:cacheprovider tests/test_grm_c4_remaining_a3.py
tests/test_grm_c4_resume_a2.py tests/test_grm_c4_campaign.py`:
85 passed in10.76 s, two SWIG deprecation warnings plus exit-time swigvarlink
warning. Full output cpu_gate.txt. Tests use fake lease and fake harness for
workers, persisted synthetic state, original score arithmetic by import.
No GPU model loaded. Foreground command tool yielded once after10 s; collected
that same foreground session to exit0. No shell background job or wait loop.

Dry-run: `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python
scripts/grm_c4_remaining_a3.py --dry-run`, exit0; dry_run.json records42 units,
3200 worker s and1290 cooldown s. `bash -n artifacts/grm_c4/lead_commands_a3.txt`
exit0. Commands are 4sup+4census+13LH, cell c96_w64 then c64_w64, each scored,
then cross summary. Initial30 s gap plus30 s after each unit; unchanged leases.

Synthetic export: invoked the SHA-bound test fixture and four_cells helper
with pytest.MonkeyPatch in a foreground CPU process, then production
cross_summary/format_summary. Saved under synthetic_four_cells/; three fake
executable-cell scores and pinned archived WC1 fourth row. This is explicitly
NOT campaign E2E evidence. Matrix uses full13/14 WC1 diagonal; historical64
is8/9,10/10,14/14 and Juniper false versus synthetic fixed Juniper true.
Suite separately covers SUPPORTED_FINITE, REJECTED_FINITE, MIXED and missing/
stale/incorrect-identity/probe-membership/no-observation refusals. Author baseline
only; no blind-review claim.

Post-gate validation.json: all29 existing receipt/claim JSON files match
pre-gate bytes; no added real worker receipts; original A3 and recovery A4
bindings validate, including their frozen source manifests. All original and
new source pins remain unchanged. Actual A4 recovery and two-cell GPU E2E
remain unrun by this seat; per-cell real scores and cross verdict pending.
Not claimed fixed: observed live campaign RED until authorized recovery passes.
No budget increase or waiving recorded failed wall time. No process signals,
lock changes, git, subagents, GPU execution or edits outside worktree.
Model: GPT-6 (Codex; exact deployment identifier not exposed). Effort: high.
