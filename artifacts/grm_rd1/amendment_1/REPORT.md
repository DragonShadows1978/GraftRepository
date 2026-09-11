# GRM-RD1 amendment 1 — worker implemented; numerical validity NOT RUN

**FIT_ESTIMATE: 138 requests, 3129.896567 s = 0.869415713 GPU-h under the authorized 1.0 GPU-h cap. CPU control-flow gates PASS. Numerical A0/r2 reproduction and the reader verdict remain NOT_MEASURED; not claimed fixed.**

## Registration and projection

Immutable lead order: `orders/GRM_RD1_AMENDMENT_1.md`. Original order and r1 registration/artifacts are preserved. Amendment: [amendment.json](amendment.json), SHA-256 **72f7ba59d705e2107627321557ab047de762cb149d29143380bcacfcc3634593**. Registered before gates; 211 new input pins plus the unchanged r1 bindings. [Ledger](LEDGER.md).

The fixed cohort is 30 mounted-but-unread memory sides plus 16 failed oracle sides, under A0/A2/A4: 46 x 3 = **138 scored probe _attempt calls**, in **48 checkpoint/arm batches**, at most 7 requests and 158.762869 projected seconds per batch. Each batch gets one foreground lease of at most 285 s; no per-probe leases. Actual model loading, non-probe prefix generation, folds and pressure are charged to the same 3600 s campaign cap. They are reconstruction overhead, not additional scored probe requests. The harness launches no unscored probe generation.

Projection: r1 total 11793.813152259705 / 520 = **22.680409907679 s/request**; x138 = **3129.896567259705 s**. This is the requested amortized r1 planning proxy, including r1's overhead scaling, not measured per-request timing. Repeated loads, prefix work and medium-reasoning latency remain unmeasured; actual budget exhaustion stops the campaign without trimming or retrying. Estimated overflow is zero. Dropped A1 is NOT_RUN, 46-request estimate **1043.298856 s**; dropped A3 is NOT_RUN, 46-request 64-token estimate **2086.597712 s** (two proxy units). Original full-design estimates remain in r1.

Actual selected prompt:

```text
For C7-Archive-0, give outbound then inbound tags separated by |. Reply only with the answer; if unspecified, reply unknown.
```

A2/A4 replace only terminal `; if unspecified, reply unknown.` with `.`. A4 also replaces `Reasoning: low.` with `Reasoning: medium.` in the live system prefix. A0/A2 use low. All retained arms use 32 answer tokens, original stop sequences, final channel, sink K/V and recorded mounts. [Requests](requests.json), [exact arm diffs](arm_diffs.json). The frozen C7 scorer is reused; the inherited whole-string unknown case correction applies only to controls, and no control rows are scheduled in this amendment.

## Replay worker and validity

[Worker](../../../scripts/grm_rd1_replay.py): `replay_attempt` line 148; `replay_batch` line 271; `verify_a0` line 347; `launch` line 354; `summary` line 409.

The worker validates the predecessor worker/checkpoint/tree/state boundary, copies its repository into a fresh writable session, loads the production model/repository, checks manifest projection and sink/profile geometry, and restores the recorded cold state. It executes intervening original non-probe turns via production `run_turn`, with C7 forced folds and pressure in chronological order. This includes creation of nodes absent at the preceding checkpoint. It never substitutes the post-cell checkpoint.

Earlier probe rows use C7's deferred/no-deposit ephemeral contract. Their turn_records advance without extra probe decode; historical frame/deposit/split guards must pass. This omits transient routing, paging and reader-cache history. The lead's numerical A0 gate is required to establish whether this reconstruction is valid for r2; CPU evidence alone cannot establish it. Each selected side uses a clean ephemeral cache. Ordered mounts must equal the residency row before revision resolution and after the production attempt. Payloads must exist and load. Mounted NPZ hashes are recorded and treatments must match A0 hashes. Oracle mounts are empty and its exact chronological live prompt is retained.

The original production `_attempt` runs with `deposit=False`, `defer_memory=True`. The worker writes raw/served text, prompt/token IDs, mount IDs/hashes, checkpoint SHA, arm/side/class, scores, production info and explicit serving provenance. Raw `_attempt` output is compared as UTF-8 bytes to r2 served text. A difference receipt contains both strings and UTF-8 hex, then raises `A0_R2_TEXT_DIFFERENCE`; it is never normalized away. The historical ladder may have selected among attempts or applied grounding normalization, so these are possible differences, not asserted causes. A0 must pass all **46 rows** before any A2; A4 follows all A2. Failed/partial A0 invalidates the entire contrast.

## CPU gates

[CPU receipt](cpu_gates.json), [baseline log](cpu_baseline_1.log), [mutation log](cpu_mutations_1.log), [dry-run](dry_run.json), [blocked report](blocked_report.json).

- `registration_and_138_intended_diffs`
- `checkpoint_tree_and_boundary`
- `fake_checkpoint_production_attempt_A0` (memory and oracle)
- `fake_prefix_creates_missing_mount`
- `fake_arm_fields_and_isolation` (A0/A2/A4)
- `mount_mismatch_rejected_before_forward`
- `A0_difference_receipted_and_stops`
- `A0_campaign_barrier`
- `budget_and_incomplete_summary_fail_closed`
- `dry_run_no_gpu`
- `registered_mutations_at_least_80_percent`

Baseline: **13 passed**, 8.71 s. Subsequent mutation gate: **1 passed**, all **5/5** registered mutants rejected, 5.76 s; threshold 0.80. Two SWIG deprecation warnings remain visible. Fake numerical boundaries use C7 Codec/Model/CPUArena, while actual checkpoint persistence/load and inherited production `_attempt` execute. The fake reader derives values from restored mount tokens or live source text; it never receives expected answers or a historical-output tape. The fake prefix callback exercises worker chronology and creation of a missing mount; it does not validate GPU `run_turn`, folds or pager numerics. This is author CPU evidence, not a blind audit.

The no-authority launch test exited 2 before acquiring a lease/loading a model: `ValueError: BLOCKED_NO_GPU_IN_SANDBOX`. Model directory and SHA-pinned native library exist. Shell syntax and executable bit pass. [NOT_RUN table](ARM_TABLE.md) and [summary JSON](summary_not_run.json) contain all 40 combinations; absent numeric results are null, never fabricated zeros.

## Exact lead commands

The executable, immutable [lead_commands.txt](lead_commands.txt) runs all A0 cells first, checks the 46-row barrier, then all A2, then all A4, then summary. Run once in the lead's authorized GPU environment:

```bash
/mnt/ForgeRealm/wt/grm-rd1/artifacts/grm_rd1/amendment_1/lead_commands.txt
```

Its exact content is:

```bash
#!/usr/bin/env bash
# Prior art: C7/C2 foreground leased cells (GRM contributors, 2026).
# Lead-only GPU run. Fresh immutable campaign; no retries and no process kills.
set -euo pipefail
cd /mnt/ForgeRealm/wt/grm-rd1
export PYTHONDONTWRITEBYTECODE=1
python scripts/grm_rd1_replay.py dry-run
export GRM_RD1_LEAD_GPU=1
campaign=/mnt/ForgeRealm/wt/grm-rd1/artifacts/grm_rd1/amendment_1/gpu_1
cells=(A-024-031 A-032-039 A-040-047 A-048-055 A-056-062 A-063-070 A-071-077 A-078-085 A-086-093 A-125-132 A-133-140 A-141-148 A-249-256 A-257-264 A-265-272 A-273-280)
for cell in "${cells[@]}"; do
  python scripts/grm_rd1_replay.py run-batch --batch "A0_${cell}" --campaign "$campaign"
done
python scripts/grm_rd1_replay.py verify-a0 --campaign "$campaign"
for arm in A2 A4; do
  for cell in "${cells[@]}"; do
    python scripts/grm_rd1_replay.py run-batch --batch "${arm}_${cell}" --campaign "$campaign"
  done
done
python scripts/grm_rd1_replay.py summary --campaign "$campaign" --output "$campaign/summary.json"
```

Every batch has an exclusive campaign owner and immutable reservation/receipts. Orphan reservations remain charged; failed batches cannot be retried. Shared GPU lock acquisition is nonblocking. A foreground lease alarm raises on expiration; no process is killed. Python cannot hard-preempt an uninterruptible native call, so any overrun is RED and charged in full, without clamping elapsed time. The controller will refuse further work at the budget rail.

Summary implements the registered rule on the original fixed 30 memory probes: **treatment exact minus replayed A0 exact >= 8 and wrong values <= 2**, with all 46 rows of that arm and all A0 rows required. Otherwise NOT_MEASURED or DOES_NOT_MEET_RULE, as appropriate. No reader improvement is claimed from CPU fixtures.

## Prior art

Verified local C7/FIX4/EB1/C2 and diagnostic doubles, GRM contributors (2026): checkpoint tree/state/worker SHA bindings, ephemeral frame lifecycle, production `_attempt`, chronological oracle, frozen scorer, prefix fold/pressure schedule, foreground leases, atomic ownership/reservations, and fake numerical boundaries reused. HOUSE_RULES section 8 and C7's source-copy mutation pattern supply the registered mutation procedure. Ours: fixed-residency replay orchestration, contrast accounting and summary composition. **No prior art known to me for this exact composition.** No new serving algorithm, external literature claim or novelty claim. Annotations occur at code sites and in the ledger.

## Deviations, RED, process safety, model and effort

Deviations: state filename is `checkpoint/state.json`; ignored source artifacts are read from `/mnt/ForgeRealm/wt/grm-c7`. Deferred ephemeral probe history is advanced without inference to preserve 138 scored probe calls; transient differences require numerical A0 validation. Reconstruction can generate non-probe/fold tokens, fully charged to leases. Raw attempt text is not silently converted to the historical ladder's served text. No cohort/arm trimming beyond the explicit lead amendment.

**RED:** GPT-OSS A0 reproduction, treatment outputs, internal reasoning behavior, actual GPU budget fit and reader improvement are unmeasured. No independent blind audit. **Not claimed fixed.** A1/A3 are registered NOT_RUN. Cap/timeout limitations above are explicit.

Safety: no GPU model load/allocation, git, subagents, services, background shell jobs, process kills or core edits in this seat. CPU tests set CUDA_VISIBLE_DEVICES empty; fake model uses no numerical CUDA. Tool-level yielding of foreground commands is recorded in the ledger; all completed normally. No source artifacts or memory files were modified.

Model under test: **openai/gpt-oss-20b**, revision **6cee5e81ee83917806bbde320786a8fb61efebee**, r2 recorded **resident_packed_mxfp4**, **tensor_cuda GptOss20B_TC**. A0/A2 low; A4 medium instruction. Agent: **Codex, GPT-6 family**; exact serving model ID is not exposed in session metadata. Requested reasoning effort **high**; no independent runtime attestation available.
