# GRM-C2 ledger (append-only)

## 2026-09-09 registration, before gates
Read /mnt/Shared/HOUSE_RULES.md and AGENTS.md; order is immutable plan.
Read metadata .git and referenced worktree HEAD: refs/heads/grm-c2; no git command.
C3 COMMON absent locally; read canonical checkout order, fingerprinted in registration.
Created config/grm_eb1_profile_registered.json, SHA256
cfee19259a071d1314dde602978d8f83880f7fd3cca04a9d41b26ed42d4fb74e.
Created registration.json, SHA256
8a3a27302c41f2980f883666657f7d871c5ad01d195953e017804717d3644eb0.
Read CPU tokenizer only: exact revision 6cee5e81ee83917806bbde320786a8fb61efebee,
Harmony sink 19 tokens, profile live shift 115; shipped width256 live shift275.
Copied frozen frame and reference census as campaign sidecars, original files read-only.
Prediction remains profile9/9,9/10,13/14; defaults5/9,9/10,13/14.
Reasoning: defaults width256 differs from EB1's width96. RT1 already default ON.
No silent change to width96 for the defaults arm.

Prior art: WC1/RT1.1/RS3/EB1/DET1 and SCOUT-FIX-1, project contributors (2026),
verified local source. Taken: fixtures, chronology, capture geometry, route/score
helpers, ordinary manifest persistence and measured mounted-ntok sum. Ours:
additive opt-in registry, pre-probe persistence checkpoints and new-process replay
orchestration, missing-evidence checks. No new route/selection algorithm; no prior
art known to me for this adapter beyond the named local systems. Standard
checkpoint/restart is established systems practice; no novelty claim.

RED before GPU: 52 cells estimate3140s exceeds2880s budget by260s. Non-fit,
not silently tuned. Estimates are planning estimates (45s sup family,55s early
session/restart group,100s long segment), not newly measured GPU timing. Cell cap
285s, outer590s, lock wait240s, cooldown30s. Lead must amend decomposition or
provide justified estimates before execution; no longer lease or automatic retry.
No GPU runs, subagents, git, background waits, signals or service changes.

## New CPU gates and historical timing check
Command: CUDA_VISIBLE_DEVICES='' timeout120 python -m pytest -q
 tests/test_grm_c2_profile.py. Receipt new_cpu_initial.log: 17 passed,2 warnings
in1.72s. Evidence class: author-run CPU unit tests, no blind or model validation.
A1 includes nine mixed-module CPU pytest paths (62 existing files total).
A2 fingerprints driver/test code and registers their gates before execution.
Dry-run enumerates52 cells, NON_FIT_BUDGET. lead_commands.txt includes the exact
commands but explicitly gates execution on a separately reviewed planning amendment.

A3 timing evidence (archived E2E receipts, CPU read only): width96 supersession
family elapsed243.512s, census turns201.836s, long-history turns933.598s;
width256 corresponding208.966s,260.218s,1174.350s. These already total3022.482s,
before C2's restart replay/snapshot overhead. WC1 profile levers were ON at
both widths; historical contention caveat applies. This is not a rigorous lower
bound or defaults measurement. It does refute confidence in reducing3140s
heuristically to fit2880s. Original estimate/registration unchanged; NON_FIT remains.

## Receipt review and A4
Static source evidence core/graft_arena.py deposit_from_cache: capture pin moves
routing-key harvest only, not its live-cache payload. Driver now grades
FRESH_CACHE_SLICE_PAYLOAD_NOT_MOVED_BY_PIN, retaining capture_span_pos0, and
checks complete geometry on defaults as well as profile. Old A2 driver preserved
as driver_A2.py, changed driver/test SHA in immutable amendment_A4_capture.json.
Registered original profile/plan remain unchanged. New final CPU log:
19 passed,2 warnings in0.72s. These are author-run unit tests, not blind verification.

## Existing battery inventory error, recorded without suppression
Two nominal CPU/mixed files invoke real tc.tensor allocations in10 tests:
test_graft_quant_format.py (5) and test_grm_importance_telemetry.py (5).
They failed with `RuntimeError: cudaMalloc failed: no CUDA-capable device is detected`
under CUDA_VISIBLE_DEVICES=''. No GPU computation succeeded. This was a test
inventory mistake, not ten newly established product regressions. Raw logs005/028
preserved; tests not edited or skipped to hide failures. Remaining CPU fake paths
continue. Do not claim that no CUDA API was attempted.

Conditional SCOUT-FIX-1 inspection map saved in SCOUT_FIX_1_TRIAGE.md. No E2E drop
observed, so no bisect/reversion performed. Demand OFF makes B1 direct influence
unlikely; B2/B5 reporting seams and B3 persistence seam are hypotheses, not proof.

## A5 full CPU inventory completion (before added gates)
Audit of originally excluded test bodies finds three additional all-CPU modules:
GPT-OSS context-ladder pure helpers, ragged-bank CPU numpy/fake-native tests,
Qwen35 translation CPU fixture tests. Also six GPT-OSS scaffold CPU-only nodes;
its remaining tests explicitly call _require_tensor_cuda. A5 registers these
commands before running them. Total66 existing files touched by CPU collection,
with33 remaining GPU programs excluded plus GPU nodes in scaffold. No existing
test/assertion modified. Earlier broad filename heuristic was over-conservative.
Native-runtime full file completed under285s: controller280.845s, no timeout.

## A6 continuation durability checks
Code review found that fresh-session continuation also reloads prior payloads.
Added the same missing/pending-payload rejection there, plus comparison against
previous cell's persisted repository hashes. Old A4 driver preserved, A6 binds
new source and registers two boundary tests before execution. No production
changes, no registration edits, no reharvest path added. New CPU final A6:
21 passed,2 warnings in0.65s; dry_run_A6.json remains NON_FIT_BUDGET.
lead_commands.txt CPU line updated to include all21 new tests. Commands remain
blocked by the immutable overbudget plan; do not claim runnable GPU certification.
Existing native CPU tests refer to canonical C++ paths; current worktree and
canonical cpp/grm_runtime.cpp and .hpp SHA256 compare equal. Python tests import
this branch. This source comparison does not broaden CPU tests to E2E evidence.

## Full existing CPU completion and final reporting
All66 CPU-selected existing file lanes completed, no timeouts. Raw aggregate:
1480 passed,58 failed,148 skipped,20 errors. New C2:21 passed. SCOUT-FIX-1:23 passed.
The exact original six-file comparison subtotal is227 passed,11 failed, matching
its reported known calibration/receipt failures. Ten additional failures are
unavailable real CUDA allocations;37 further failures and20 setup errors occur
in broader pre-existing receipt-dependent/APAMQ coverage. No blanket claim that
all failures are explained or that regression absence is certified. Full original
summary lines and node IDs pasted in CPU_GATE_RESULTS.md, with JSON/log SHA records.

Native runtime121 passed; lifecycle101 passed. Existing skips148 retained;
RS1/RS2/RS3-reference/LSR-only-skipped lanes checked zero tests and are not green
coverage. Additional A5 CPU modules and six scaffold CPU nodes all passed.
All registered inputs verify unchanged; native worktree/canonical sources equal.
Final report REPORT.md, final blocked receipt blocked_report_final.json,
final dry-run dry_run_A6.json, exact commands lead_commands.txt. No GPU gate,
bisect, blind review, production/default edit or adoption performed.
No git/subagents/background waits/process kills/services/external delivery.
Identity Codex/GPT-6, exact API ID unavailable, requested effort high.

## Lead amendment 1 registration, before CPU gates (2026-09-09)
Evidence class: registration / source inspection / planning reasoning. Immutable
lead order raises cap to3600s; r1 registration SHA 8a3a27302c41f2980f883666657f7d871c5ad01d195953e017804717d3644eb0 unchanged.
Created amendment_lead_1.json SHA 91df018adaa90e9db3fd32981ad72a5564cee2cc8360b462775248945859807c.
52 original cells /3140s unchanged; defaults96 NON_FIT +26 cells/+1570s, total4710s
(1110s over cap). Width256 shipped defaults vs width96 proposed profile is not a
width-matched comparison. Stable battery order scores sup then census then longhistory
as soon as each paired fresh/restart battery completes. Lead commands now invoke
new additive grm_c2_amended.py; original r1 runner/tests remain unchanged and preserve
the old non-fit historical contract. Resume and accounting rules and CPU gates are
registered in JSON before execution. No product edits. Old commands archived
as lead_commands_r1.txt. Registration JSON and command hashes anchored in verifier
code; normalized verifier source is itself bound without a circular hash. This is
content integrity assuming trusted code, not a cryptographic lead signature.

Prior art: C2 r1 / CMC1 / WC1 / RT1.1 / SCOUT-FIX-1, project contributors (2026),
verified local sources; reused worker, battery comparator, flock lease, immutable
reservations and persisted checkpoints. New glue: cap binding, battery scheduling,
resume and summary. SHA-256 content binding is established practice, no novelty
claim. No prior art known to me for this adapter beyond those local systems.

RED: GPU gates NOT_RUN; earlier CPU-suite RED and A3 timing caveats remain.
No GPU, git, subagents, background waits, process kills/signals or live-service
actions. Identity Codex/GPT-6; exact serving API ID unavailable; effort high.

### Amendment 1 baseline and registered mutation lane
Author CPU unit tests:45 passed,2 warnings in1.44s; receipt
amendment_lead_1_cpu_initial.log. Original21 tests unchanged,24 new cases passed.
Executable shell syntax and --dry-run passed:52 cells/3140s/3600s cap.
Actual no-measurement summary exited1 as required, receipt
amendment_lead_1_summary_not_run.txt. Registered five explicit CPU behavioral
mutations in amendment_lead_1_mutation_registration.json before running them.
Prior art: House Rules/AtlasForge mutation-testing workflow (project,2026),
reused seeded defects and >=0.80 kill threshold. New C2 defects are listed
in the registration; no prior art known to me for this specific adapter.
The source integrity guard retains original file identity intentionally so
mutation kills must come from behavioral assertions, not hash rejection.

### Amendment 1 verification complete
Evidence class: author-run CPU tests/mutations, not blind or GPU evidence.
All five registered behavioral mutants killed (5/5=1.00>=0.80), each by
assertion rather than hash rejection; logs and SHA records in
amendment_lead_1_mutation_results.json. Temporary copies only, originals
unchanged. Executable shell --dry-run: exit0,52 cells,3140s estimate,3600s cap.
Actual summary: exit1,NOT_RUN,all six rows with denominators9/10/14 and explicit
unknown seats/RT1; no invented measured scores. Synthetic CPU cases cover complete
pre/post table and named provenance fields. Commands expanded for audit in
lead_commands_expanded.txt. Final report AMENDMENT_1_REPORT.md; shared REPORT.md
continued. Historical full CPU failures remain; no broad rerun or GPU claims.
No git/subagents/GPU/background waits/process kills/signals/service actions.
Codex/GPT-6; exact serving API ID unavailable; effort high.
