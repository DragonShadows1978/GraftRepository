# GRM-RD1 implementation ledger

## 2026-09-09 — registration before gates

Evidence class: local source/receipt inspection and planning calculation.
Immutable plan: `orders/GRM_RD1_MOUNTED_READ_CONTRAST.md`; not edited.
Registration: `artifacts/grm_rd1/registration.json`, SHA-256 `941f0d9db78a4748c80165da9456601731c5b0b3c651c68ec449c70da74310a1`.
Registration written by `PYTHONDONTWRITEBYTECODE=1 python scripts/grm_rd1.py --register` before CPU gates or treatment inference.

Read `/mnt/Shared/HOUSE_RULES.md`, worktree `AGENTS.md`, order, C7 amendment-4 report, C7 scorer/worker/oracle/checkpoint implementation, EB1 serving wrapper, and C7 diagnostic doubles. No git commands used; branch name is user-supplied, not independently verified. Working directory verified with pwd.

The copied worktree lacks ignored checkpoint trees, fixture and timing JSON. Original files exist at `/mnt/ForgeRealm/wt/grm-c7`; bind that root read-only. Actual session state is `checkpoint/state.json`, alongside `checkpoint/repository/manifest.json`; no `session_state` file found. Pins include preceding checkpoint descriptor, state, manifest and predecessor worker; full tree checks deferred to registered CPU gate.

52 continuation probes = 34 answerable + 18 controls. Both sides for every arm = 520 requests. Original eight quarantined probes excluded. This includes the 30 memory failure sides and 16 oracle failure sides, plus paired comparators and controls. Fixed memory verdict denominator is the original 30, never all 52. No treatment selected or executed.

16 original checkpoint batches, all five arms and both sides in each lease. R2 whole-cell walls sum to 1572.5084202657454 s. Conservative estimate = 1.25 * original wall * 6 token-budget units. This repeats reload/prefix/fold overhead and is NOT an empirical lower bound. Full total 11793.813152 s; minimal 46 failure-side proxy 4823.666819 s; cap 2880 s. Every full cell exceeds 285 s. Register NON_FIT and stop model launch, without trimming or repacking. Medium reasoning latency is unmeasured.

Prefix finding: turn-31 `c7_folded_1_d005` needs node17, absent from preceding `A-017-023` checkpoint. It is created at turn26. A model worker must restore state and execute prefix events (including original folds/pressure), then isolate each pre-probe state. Substituting the post-cell checkpoint is forbidden. No such substitution made.

Implementation: additive `scripts/grm_rd1.py`, registration/census/requests/diff artifacts only. Literal treatment suffix is `; if unspecified, reply unknown.` replaced by `.`; reasoning medium changes live system text only, keeping final channel, sink, source payloads, stops and mounts fixed. Controls alone accept whole-string normalized unknown in any case; raw frozen scores also retained. This instruction-only effort contrast cannot establish whether a model internally used more reasoning.

Prior art: local C7/FIX4/EB1 (GRM contributors, 2026), verified at code sites: immutable tree/state/worker hashes, oracle chronology, exact scorer, wrapper/stops, conservative cell accounting, fake transport concept. Python hashlib and AST (Python contributors, 2026) provide streaming hashes and literal parity. Ours: reader-request contrasts and this NON_FIT inventory. No prior art known to me for this exact composition. No external-paper or novelty claim; no external literature needed for these local reuse claims.

Predictions and pass/stop rules are in registration. A0 fake tape is explicitly a transport fixture, not numerical checkpoint replay; the latter must remain RED if unexecuted. No blind audit claimed. No model weights loaded, GPU work, service changes, subagents, git, background waits, process kills or core edits.

Inspection errors preserved here: a broad controller/worker cat produced excessive output (then replaced by keyed JSON reads); `cat artifacts/grm_c7/timing.json` and a local fixture read failed with FileNotFoundError because ignored artifacts were absent. Resolved with the original C7 read-only root. These were discovery commands before registration, not failed acceptance gates.

## 2026-09-09 — registered CPU gates and blocked launch

Commands executed in order:
- `PYTHONDONTWRITEBYTECODE=1 python scripts/grm_rd1.py --cpu-gates --output artifacts/grm_rd1/cpu_1 > artifacts/grm_rd1/cpu_1.log 2>&1` — exit 0; PARTIAL_CPU_PASS_REPLAY_BLOCKED. Registered numerical replay gate explicitly BLOCKED, not counted as passed.
- `PYTHONDONTWRITEBYTECODE=1 python scripts/grm_rd1.py --dry-run --output artifacts/grm_rd1/dry_run.json` — exit 0, BLOCKED_NON_FIT.
- `PYTHONDONTWRITEBYTECODE=1 python scripts/grm_rd1.py --run --output artifacts/grm_rd1/blocked_report.json` — expected exit 2, BLOCKED_NON_FIT, no GPU or lease.

Checkpoint tree/state/worker parity passed for 16 dependencies. Census and 104 historical score checks passed. All 520 request diffs passed, with live source constant parity. Ten control score checks passed. Scripted A0 tape replay returned 104 original strings and rejected 416 altered requests. This is author transport verification, not a numerical checkpoint replay or blind audit. The requested numerical A0 replay remains RED; no claim of overall CPU-gate completion.

Reports written: REPORT.md (narrative synthesis), ARM_TABLE.md (40 NOT_RUN combinations), HISTORICAL_TABLE.md (actual r2 score counts), result_slots.json (520 null inference slots), lead_commands.txt. Prior art for report aggregation: C7 amendment-4 tables and receipt provenance (GRM contributors, 2026); no prior art known to me for this exact composition. Report-generation code carried the same annotation. Original plan and registration untouched; no model worker enabled past the NON_FIT rail.

## 2026-09-09 — final artifact integrity

Rechecked all 113 registered input pins, original plan/registration, and core source hashes. Verified 520 null unrun slots, 40 table combinations and all relative report links. No extra inference gate run. `final_integrity.json` is deliverable integrity only, not a numerical replay pass. Final output SHA256SUMS includes script and deliverables.
