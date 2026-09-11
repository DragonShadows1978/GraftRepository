# GRM-RD1 — NON_FIT; reader effect NOT_MEASURED

**Registered NON_FIT; no GPU/model execution. Not claimed fixed.** The conservative full design estimates **11793.813 s (3.276 GPU-h)** against **2880 s (0.8 GPU-h)**. All 16 checkpoint batches exceed 285 s under this estimate. The 46 required failure-side-only proxy is **4823.667 s (1.340 GPU-h)**, also over cap. No rows, arms or cells were trimmed.

Evidence class: local source/receipt inspection, CPU hash and request-contract checks, and a conservative planning calculation. These are not new model results. Whole-cell scaling repeats reload/prefix/fold overhead; **NON_FIT under this registered estimator is not a measured impossibility result**. The r2 receipts do not separate probe generation walls, and reasoning-medium latency is unmeasured.

## Registration, prompts and predictions

Immutable plan: `orders/GRM_RD1_MOUNTED_READ_CONTRAST.md` (unchanged).
Registration: [registration.json](registration.json).
SHA-256: `941f0d9db78a4748c80165da9456601731c5b0b3c651c68ec449c70da74310a1`.
Prepared before CPU gates. Pins cover order, rules, relevant code, source receipts, predecessor state/checkpoint/worker, cohort, requests and exact diffs. [Ledger](LEDGER.md) records discovery and gate commands.

The receipt does **not** say “say unknown if not in memory.” Its actual clause is `if unspecified, reply unknown.` Representative complete memory prompt:

```text
What is the current C7-Fresh-1 value? Reply only with the answer; if unspecified, reply unknown.
```

A2/A4 replace the exact terminal bytes `; if unspecified, reply unknown.` with `.`. The resulting prompt is:

```text
What is the current C7-Fresh-1 value? Reply only with the answer.
```

Full 520 request definitions, including every memory/oracle wrapped prompt, are in [requests.json](requests.json); every A0-relative before/after field delta is in [arm_diffs.json](arm_diffs.json), SHA-256 `41c53b06f0de94e608cdab8e03253142a5d4d101a1a49262a3732f74cdcbd56a`.

A0 = original prompt / low / 32 tokens; A1 = medium only; A2 = clause removed only; A3 = 64 tokens only; A4 = medium + clause removed. Same recorded ordered mounts, payloads, stop strings, final-channel prefix, no-deposit/deferred-memory contract. Oracle mounts remain empty. `served_from` is retained in historical data where present; a planned fixed-mount reader has a distinct provenance label. No runtime served_from value is fabricated for unrun slots.

“Reasoning medium” here is solely the local Harmony live-system instruction, not an API parameter. Existing checkpoint sink/source payloads retain their original low-reasoning text. The final channel stays fixed. Whether this produces additional internal reasoning is unmeasured; changing sink K/V or channel would require another contrast.

Registered subjective predictions (not statistical intervals): A0 0/30 exact, 0 wrong; A1 4–10 exact, 0–2 wrong; A2 6–12 exact, 1–4 wrong; A3 0–3 exact, 0–2 wrong; A4 8–16 exact, 1–4 wrong. A4 is predicted to yield the largest exact gain but could fail the wrong-value rail. Oracle: A2/A4 recover at least 8/16 failures; A1 4–10; A3 at most 3. Alias mounts lack base values, limiting grounded recovery. Four repeated underlying fresh/alias questions are not 16 independent new questions.

Registered verdict: on the **fixed original 30 memory failure probes**, an arm moves the reader only if exact rises by **at least 8 relative to replayed A0** and wrong values are **at most 2**. Require all 30 pairs and A0 text reproduction. Missing/discrepant replay makes the verdict NOT_MEASURED. Every arm currently has that verdict.

## Bound state and cells

The fork omits ignored fixture/checkpoint artifacts. Original source root `/mnt/ForgeRealm/wt/grm-c7` is used read-only; local historical copies were preserved. There are 52 continuation probes, 34 answerable and 18 controls, covering all 30 memory failure sides and 16 oracle failure sides. Both sides of every probe are retained, so 520 requested generations across five arms. The original eight quarantined probes remain excluded.

Actual session-state filename is `checkpoint/state.json`, not `session_state`. Full original checkpoint tree hashes and predecessor-worker descriptor hashes passed for all 16 dependencies. The turn-24 batch depends on preserved `r2/cells/A-017-023/checkpoint`; the other 15 depend on `r2/fix4_attempt_1/cells/*/checkpoint`. Each planned batch contains all its probes and all arms in one lease.

An essential replay dependency remains: turn31 `c7_folded_1_d005` mounts node17, which does not exist in the preceding checkpoint; it is created at turn26. A valid worker must execute the intervening original prefix, including fold/pressure actions, and snapshot identical pre-probe state for each arm. It must validate ordered actual mounts and payload fingerprints before accepting each response. No post-cell checkpoint was substituted for this state.

Estimate: the original 16 probe-bearing cells sum to 1572.508420 s. Each receives six 32-token budget units (A0/A1/A2/A4 = one each; A3 = two), then a 1.25 safety factor. All original cell overhead is scaled too, deliberately conservatively. The required-only proxy allocates each cell wall uniformly across its original two-sided probes before applying the same factor; it is a sensitivity calculation, not a reduced run plan. Neither calculation proves real decode timing.

| Cell | Preceding checkpoint | Probes | r2 wall s | Conservative estimate s |
| --- | --- | ---: | ---: | ---: |
| A-024-031 | A-017-023 | 3 | 104.882 | 786.617 |
| A-032-039 | A-024-031 | 5 | 98.490 | 738.678 |
| A-040-047 | A-032-039 | 3 | 91.686 | 687.641 |
| A-048-055 | A-040-047 | 3 | 107.945 | 809.589 |
| A-056-062 | A-048-055 | 2 | 65.130 | 488.472 |
| A-063-070 | A-056-062 | 5 | 96.331 | 722.481 |
| A-071-077 | A-063-070 | 2 | 86.017 | 645.126 |
| A-078-085 | A-071-077 | 3 | 109.406 | 820.547 |
| A-086-093 | A-078-085 | 2 | 88.572 | 664.287 |
| A-125-132 | A-117-124 | 5 | 103.443 | 775.825 |
| A-133-140 | A-125-132 | 5 | 131.930 | 989.477 |
| A-141-148 | A-133-140 | 2 | 77.104 | 578.281 |
| A-249-256 | A-241-248 | 1 | 71.243 | 534.319 |
| A-257-264 | A-249-256 | 4 | 108.597 | 814.478 |
| A-265-272 | A-257-264 | 5 | 140.551 | 1054.133 |
| A-273-280 | A-265-272 | 2 | 91.181 | 683.860 |

## CPU gates and lead commands

[CPU receipt](cpu_1/cpu_gates.json), [raw log](cpu_1.log), [dry run](dry_run.json), [launch refusal](blocked_report.json).

- PASS: 16 predecessor checkpoint trees, 1637 file hashes, state next_turn/process_id and predecessor-worker SHA bindings.
- PASS: exact census 52 probes / 30 memory failures / 16 oracle failures; all 104 historical frozen scores reproduced.
- PASS: all 520 intended-field diffs; actual EB1 wrapper and stop constants checked by AST without importing a GPU backend.
- PASS: control-only case contract, 10 adversarial examples. Only normalized whole-string unknown is case-insensitive on controls; answerable values retain case. Original frozen scores are also retained.
- PASS, limited: 104 A0 historical strings round-trip through an exact-request fake tape; 416 deliberate input mutations rejected. This tests transport and rejection, **not numerical checkpoint restoration**.
- **RED / BLOCKED_NON_FIT: requested numerical A0 checkpoint replay on a fake model was not implemented/executed.** The fake tape is not substituted for this gate. Therefore the overall CPU outcome is `PARTIAL_CPU_PASS_REPLAY_BLOCKED`.
- PASS: `--dry-run` reports BLOCKED_NON_FIT; `--run` exits 2 before any model import or lease acquisition. There is no enabled GPU worker in this NON_FIT deliverable.

Exact commands are in [lead_commands.txt](lead_commands.txt):

```bash
cd /mnt/ForgeRealm/wt/grm-rd1
PYTHONDONTWRITEBYTECODE=1 python scripts/grm_rd1.py --dry-run
# Use this fresh output path once. Choose another new path for a later audit.
PYTHONDONTWRITEBYTECODE=1 python scripts/grm_rd1.py --cpu-gates --output /tmp/grm-rd1-lead-cpu-20260909
# Expected exit code 2, BLOCKED_NON_FIT; this checks the launch refusal only.
PYTHONDONTWRITEBYTECODE=1 python scripts/grm_rd1.py --run
# Stop. A separately registered timing/replay amendment is required before GPU launch.
```

The launch command deliberately refuses; it is not a concealed full C7 rerun. Additional numerical replay implementation and a separately registered timing basis are required before a bounded GPU command can exist. The original registration must not be rewritten.

## Counts and reader verdict

[Arm × class × side table](ARM_TABLE.md) contains all 40 combinations with planned denominators, zero observed rows and explicit NOT_RUN scores. [520 result slots](result_slots.json) contain null served_text/served_from/scores, preserving absent evidence. [Historical r2 table](HISTORICAL_TABLE.md) contains actual existing exact/wrong/abstain counts, separate from unrun arms. The [A0 tape rows](cpu_1/A0_tape_rows.json) are labeled CPU fixtures.

Historical answerable exact totals are memory 4/34 and oracle 18/34; original 30 memory failures split fresh4/alias8/correction8/folded10, and oracle16 failures split fresh8/alias8. Frozen scorer partial-unknown abstentions can include wrong components, so wrong_value_error=0 is not proof of component correctness. No reader improvement, folding success, product acceptance or corrected serving is claimed.

## Prior art

Verified local prior art: C7/FIX4 checkpoint bindings, immutable registration, exact scorer and conservative cell budgeting; EB1 Harmony wrapper/stops; C7 chronological source oracle and CPU diagnostic doubles (GRM contributors, 2026). Taken: those contracts and scoring, without core changes. Python hashlib streaming and AST literal inspection (Python contributors, 2026) support bounded hashing and source parity. Ours: this reader-request contrast, cohort bindings, diagnostic tables and NON_FIT calculation. **No prior art known to me for this exact composition.** No external algorithm or paper claim; no literature novelty claimed. Annotations also appear at code sites and in the ledger.

## Deviations, RED and process safety

Deviations: source artifacts read from the original C7 worktree because ignored checkpoints/fixture are absent here; state filename corrected to its actual `state.json`; all 52 continuation probes paired for controls/comparators, with the original 30 retained for the verdict. The conservative timing model cannot establish a fit; no authorized cohort was silently reduced. Numerical replay/model execution stopped under NON_FIT, leaving the requested A0 replay gate incomplete. These are explicit limitations, not completed-work claims.

RED: numerical checkpoint replay, all treatment inference, internal reasoning effect and reader verdict unmeasured. Incomplete alias evidence, rejected folds and baseline read failures remain. **Not claimed fixed.** No blind independent audit occurred. Successor for the lead: separately register a more informative timing/replay order preserving all required rows and rails; do not reinterpret this conservative estimate as proof that batching cannot fit. Reuse original C7 prefix execution and snapshot contracts (GRM contributors, 2026); no new algorithm is proposed.

Safety: CPU foreground commands and read-only source artifacts; additive harness/report files only. No git, subagents, GPU allocation/model loading, service operations, background waits, process kills, or core edits. No branch claim beyond the user-provided `grm-rd1`. Registration/input hashes checked again after gates. No memory files modified.

Model under test: **openai/gpt-oss-20b**, revision **6cee5e81ee83917806bbde320786a8fb61efebee**, `resident_packed_mxfp4`, `tensor_cuda GptOss20B_TC`. Planned effort A0/A2/A3 low, A1/A4 medium. Agent: Codex, GPT-6 family per session instructions; exact backend model identifier/effort metadata not exposed. Requested analysis effort high; not independently attested.
