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

2026-09-08 — Amendment 1: historical-receipt summary repair (CPU only).
Immutable amendment order read; separate pre-gate amendment registration:
`artifacts/grm_x3/amendment_1_registration.json`. No original registration,
order, fixture, implementation manifest, or run receipt was edited.

Diagnosis (source inspection / filesystem evidence): discovery finds four cells;
all workers persist outcome/category_realized. The b80b... summary is the r1
empty handoff artifact (mtime 23:13:18 EDT), older than the first start
(23:19:05 EDT). It hashes summary content, not runtime. Original lead.py:68-72
and diagnostic.py:60-65 repeatedly call DET1 load_snapshot_array, which
revalidates the whole snapshot per member (DET1:1292-1313), twice per summary.
The lead chain ends exactly 120 seconds after its final-cell line: timeout is
an inference, not a recovered exception. My pre-fix read-only diagnostic
(session 66260) also failed to finish before I interrupted only that own process
via Ctrl-C, exit 130; buffered stdout was unrecovered. No baseline timing or
throughput claim. `amendment_1_diagnosis.json` records metadata and source counts.

Changed only X3 scripts/tests: diagnostic.py:60-68 now uses DET1's existing
validated-array reader after one manifest validation; every blob is still
hashed and geometry-checked. lead.py:60-82 removes the duplicate traversal.
lead.py:85-143 selects one historical run (or requires its explicit fingerprint),
checks recorded runtime/input/cell pins and worker/per-snapshot agreement, then
records input/output provenance. Summary dispatch precedes current-runtime
implementation checks; GPU launch checks remain unchanged. Descriptive controls
and stratum counts added, without modifying registered scoring or null guards.

Unit-test evidence: `logs/grm_x3_a1_cpu_01.txt`: 50 passed, 2 SWIG deprecation
warnings plus shutdown swigvarlink warning, 9.24 s. New primary regression:
`test_summary_existing_run_directory_real_layout` at tests:272. It uses all four
cells, 20 synthetic captures, real DET1 manifests/blobs and raw logits, with
real receipt validation, all four table cells and both prediction sets; pins
20 whole-manifest validations. Companion cases reject ambiguous runs and five
corruption classes, and report missing starts RED. Author baseline only, no
blind verification or new mutation score claimed.

CPU aggregation of existing E2E receipts: command
`bash scripts/grm_x3_lead_gpu.sh summary e59ce9ad2e2073d8328db192c9a0405e3147fa0d0d660288567458debfb55008`
completed exit 0 in 10.669022 s with empty stderr. Receipt:
`artifacts/grm_x3/amendment_1_summary_execution.json`; output:
`artifacts/grm_x3/summaries/09b8527549574e99fcec7f77b36b22b46ecacb0fae8aa8ba8b186d270d9c15d9.json`.
All20 observed, no receipt errors. Low/low: 1 error / 1 (100%); low/high and
high/low empty (null); high/high: 16 errors /19 (84.2105%). Same-payload/zero
KL=0 and raw logits identical on20/20. Sham KL<0.05 on16/20, mean0.036660684,
median0.029970218, range0.009287394..0.077579186 nats; top1 changes0/20.
Joint controls16/20; comparable-to-factual controls0/20. P1 and Q1 FAIL.
P2/P3/Q2/Q3 remain UNEVALUABLE under the frozen strata guard: correct3/8,
decoy1/6, refusal0/6 realized, no truncations. Raw descriptive overall accuracy:
lesion80%, mass20%; intended-decoy100% vs0%. No semantic rescoring or promotion
of descriptive counts to registered prediction passes. Status remains
RED_STRATA_OR_INCOMPLETE. Detailed narrative: `AMENDMENT_1_REPORT.md`; per-row
CSV: `amendment_1_rows.csv`, both under artifacts/grm_x3.

Prior art: house DET1 (2026) supplies validated manifest/blob-reader separation;
house X3 (2026) supplies recorded runtime pins, content-addressed receipts and
frozen scoring/guards. Removed duplicate traversal and added reporting plumbing;
no new capture or intervention algorithm. Ordinary descriptive counts,
mean/median/extrema; no specific prior art known to me for this reporting repair
or run-selection rule, no novelty claim. Original research annotations retained;
no new external-literature verification claim.

SHA256 verification: 5,156 frozen evidence/input files unchanged against
`amendment_1_evidence_before.json`; result in
`amendment_1_evidence_verification.json`. Original stale summary preserved.
No git, subagents, GPU runs/leases, service changes, background waits or signals
to other processes. Only own CPU diagnostic interrupted; all own processes
finished. Experimental RED, hard native-hang deadline residual and independent
blind verification are not claimed fixed. Target openai/gpt-oss-20b, frozen
Harmony effort low. Author GPT-6/Codex (exact deployment subvariant unavailable),
reasoning effort high as requested.
