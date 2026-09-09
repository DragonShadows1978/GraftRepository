# GRM-X3 ledger

2026-09-08 — Registration and input freeze (reasoning / authoring, before gates).
Read `/mnt/Shared/HOUSE_RULES.md`, the immutable X3 order, and local AGENTS.md.
Verified worktree HEAD by reading its metadata file: refs/heads/grm-x3; no git command.
Registration: `artifacts/grm_x3/registration.json`, SHA256
`83bba1ea9f9223271526694e97ed3c92b89f11382facc6598515b2c3fed09b62`.
Twenty recipes frozen (8 intended correct, 6 intended decoy, 6 intended refusal),
with token IDs, source SHA256s, a manifest and SHA256SUMS. These are NOT GPU
snapshots or observed answer categories. Both prediction sets, scoring direction,
strata guards, controls, kill rails and four-cell resource limits are registered.

Source finding: DET1 capture/restore exports canonical host value bytes, not raw
BF16 device storage. Its existing removal compacts rows; X3 will mask only one
mount's columns, preserving other positions and tensors. Direct diagnostic
attempts retain EB1/RS3 geometry and use controlled picks; this is not a natural
RT1 routing acceptance test. No core module will be added.

Authoring errors before registration was created: formatter extraction first
raised `NameError: name 'SYSTEM_PREFIX' is not defined`; added literal formatter
constants. Input geometry checks rejected 61+40 >96 and 92+6 >96. Replaced the
sham text with the fixed standalone unrelated `Weather calm.` graft (3 tokens).
No model outputs or gate results were inspected; no threshold changed.

Prior art: Jain and Wallace (2019), Attention is not Explanation,
https://aclanthology.org/N19-1357/ (verified primary abstract): motivation for
measuring intervention beside attention. Meng et al. (2022), Locating and Editing
Factual Associations in GPT, https://arxiv.org/abs/2202.05262 (verified primary
abstract): controlled activation interventions; no ROME weight editing used.
Kullback and Leibler (1951), On Information and Sufficiency: KL statistic;
unverified — lead to check `Kullback Leibler 1951 On Information and Sufficiency
10.1214/aoms/1177729694`. House DET1/S3/S4/D-NGH/RS3/RS4/EB1/RT1 (2026): reused
capture/hydration, telemetry arithmetic and geometry. Own contribution is a
scoped mask/payload fork adapter and paired negative-control reporting. No prior
art known to me for this particular fixed classifier direction, digit-decoy rule
or fixture allocation; no novelty claim. Create-only hashing/leased dispatch
follows the existing DET1 and house runners, not a new algorithm.

2026-09-08 — CPU implementation baseline and adversarial checks (unit test).
`logs/grm_x3_cpu_baseline_01.txt`: 36 passed; initial pure fork/scoring/runner checks.
`logs/grm_x3_cpu_baseline_02.txt`: 41 passed, including real DET1 capture/restore
for every X3 arm with only tensor upload replaced by NumPy. Source: scripts
`grm_x3_diagnostic.py`, `grm_x3_gpu.py`, `grm_x3_lead.py`, `grm_x3_lead_gpu.sh`;
tests: `tests/test_grm_x3_lesions.py`. No core or serving code edited.

`mutation_registration.json` frozen after baseline; all six registered defects
caught by executed tests (6/6 valid lanes, kill fraction 1.0 against >=0.80).
`mutation_receipt.json` links each log and confirms the original diagnostic
source SHA is unchanged. Mutants existed only in child-process memory, never in
production files. This is author testing, not independent blind verification.

Source review found that direct `_attempt` bypasses `step`'s assignment of
attention `live_shift`. The GPU adapter now calls existing RS2 `_clear_boat`
before seating; `test_direct_attempt_pins_rs3_query_origin_and_width` pins the
real RS3 position calculation and width rejection. No GPU failure was observed
or hidden. `logs/grm_x3_cpu_baseline_03.txt`: 42 passed, 2 reported SWIG
DeprecationWarnings plus a shutdown swigvarlink warning; none suppressed.
The mutation receipt applies to the unchanged diagnostic module and the earlier
41-test suite. The later test adds the query-origin check; no mutation claim is
made for the GPU driver itself.

Implementation details/amendments are in
`artifacts/grm_x3/amendment_implementation_details.json`. Registration and input
fixtures remain unchanged. In particular, individual raw snapshots freeze before
their own measurements; they are not all20 concurrently captured. The read-only
external native build is needed because this worktree has no native .so. Python
alarms do not establish a hard upper bound on a hung native call. This unresolved
constraint is RED, not claimed fixed.

2026-09-08 — Handoff gates and GPU block.
Unit-test evidence: reused DET1 snapshot, RS3 capture/seat, EB1 ephemeral-frame,
and RT1 split-child suites: 185 passed, 2 SWIG DeprecationWarnings plus shutdown
warning (`logs/grm_x3_reused_cpu_gates.txt`). Combined distinct CPU baseline:
42 X3 tests + 185 reused tests = 227 passed. Not a GPU/kernel/E2E result.
All 21 fixture manifest/input SHA256 entries passed `sha256sum -c SHA256SUMS`.
Shell syntax and all X3 Python AST parses passed. `--dry-run` and runner `list`
enumerate every cell/arm; receipts in `artifacts/grm_x3/`.

`implementation_manifest.json` freezes 73 source/dependency/artifact pins, SHA
`361dc992bd19b8ba896f784e4cb04f0efbf1fe1dac029bbfa19b08af0f5f6d1f`.
Preflight reports `NO GPU: /dev/nvidia0 absent in dispatched sandbox`; no GPU
worker started, no lease taken, no measured logits or real snapshots. The empty
summary is RED with null prediction verdicts and null four-way error rates.
GPU budget consumed: 0; planned reserve 1140 seconds. Evidence classes: CPU
preflight for absence; reasoning for timing/budget estimates. Prior-art annotations
are at code sites, in the initial ledger entry, and in the final report.

Handoff artifacts: `REPORT.md`, `GPU_BLOCKED_REPORT.md`, `lead_commands.txt`,
`CPU_GATE_RECEIPT.json`, frozen registration/fixtures, implementation pins,
mutation registration/results and per-cell enumeration. No git, subagents,
service changes, serving code edits, GPU loads, background waits, or process kills.
The lead's independent blind verification remains outstanding under House §8;
the dispatched author did not launch its own verifier.
