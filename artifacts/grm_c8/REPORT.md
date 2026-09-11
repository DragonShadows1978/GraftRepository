# GRM-C8 — CPU GREEN; GPU/decision BLOCKED; campaign NON_FIT

The opt-in profiler and leased-cell adapter are built. **88 CPU tests passed; 5/5 registered mutants rejected. No GPU work ran.** The complete campaign estimates **2,591 s (0.720 GPU-h)** against **1,800 s (0.5 GPU-h)**. It is registered NON_FIT and refuses before acquiring a lease. No observed decode/non-decode fractions, per-stage GPU means/p50/p95, demand-trip result, GPU bit-identity result, or allocation winner are claimed.

SP5 has a material context discrepancy: its section 4 says **S=2,048**, 32 synced decode steps, for standard/2P/SP **82.6/84.0/84.2 ms/token**. The order assigns these numbers to 12,288; that is actually the separate resident memory ceiling in the receipt. The order is unchanged. At the measured context, 2P saves **−1.4 ms/token (−1.695%)**, SP saves **−1.6 ms/token (−1.937%)** relative to standard. These are slowdowns, not a positive saving. Evidence class: external synced decode microbenchmark receipt, not GRM E2E performance. There is **no measured saving at 12,288 context or at the width-96 arena** in this source.

## Deliverables and exact boundaries

- [Profiler](../../scripts/grm_c8_profiler.py): source map line 52; CUDA meter line 91; OFF-by-default constructor line 145; original-value-preserving `run` line 206; reducer line 253. Flag: explicit `--profile-turn` in [cell adapter](../../scripts/grm_c8_cells.py:340). No existing product/config/default was edited.
- Exclusive stages: lexical scan, route, admission, cold fetch, seat/mount, prefill, served decode, deposit/harvest, fold, and OTHER. The source map follows Python code identity, including imported aliases and subclass implementations. `_attempt` line phases distinguish seat assembly, prompt prefill and the whole decode loop. Fold owns nested generation; deposit owns harvest/route-key forwards. Absent stages have zero measured duration; unmeasured GPU fields on CPU are null.
- Census and longhistory wrap the entire unchanged `e2e.run_turn`, including route diagnostics and transcript/receipt IO. Sup wraps unchanged `_probe_ladder_chat(defer_memory=False)`, including deposit and fold. Startup/model loading, initial fixture capture, checkpoint IO and replay cloning are outside measured turn boundaries but inside charged cells. These boundaries differ in harness overhead; report each battery separately.
- Stage wall totals conserve full instrumented-turn wall, including instrumentation gaps in OTHER. Mean and nearest-rank p50/p95 are per complete turn, including zero-stage turns. Decode share is **sum(decode wall)/sum(turn wall)**, not mean of per-turn ratios. Demand has a separate one-turn table, not a 30-turn claim.
- CUDA time is a device-synchronized event interval on the default stream, including CPU/idle gaps. It is **not kernel-active time**. Synchronization changes overlap and tracing costs time; the allocation fraction describes instrumented wall. Boundary overhead is recorded; dispatch overhead is included but not independently measured.
- Peak memory: **exact default CUDA async-pool used-memory high-water mark** per exclusive segment, then max per stage/turn. `cudaMemGetInfo` whole-device boundary maxima are separately labeled sampled lower bounds. `full_device_peak_bytes` is null: legacy allocator and other-pool transients are not measured. This is a **partial fulfillment of a whole-device peak requirement**, not claimed fixed. CPU memory metrics are null, not GPU zeros.

## Registration and cells

Immutable plan: [original order](../../orders/GRM_C8_EB1_TURN_PROFILE.md). [Registration](registration.json), SHA-256 `264ccd493f9f73fd157c47a7fa0ded3be60e0b3530e0ad3b42fae13d872d8345`, was written before gates. It binds the order, profile, source, fixtures, local model/tokenizer metadata, native library and TensorCUDA module hashes. The model weight files have revision/path/size inventory, not a new full-weight hash. A runner code anchor validates the registration and source template. [Separate demand registration](demand_registration.json) has its own digest in [SHA256.json](SHA256.json).

C2 geometry: width 96, sink 19, live/capture shift 115, ephemeral, capture pin live, seat-near-live ON, RT1 ON. Demand OFF everywhere except `demand-lh033`; original carried threshold, early-abort OFF. The demand cell reuses the exact persisted pre-lh033 state and requires `demand_fired == True` and `demand_trip_taken == True`. The prompt is not guaranteed to fire at this geometry. No fire/no actual trip is RED with no alternate prompt, threshold tuning or retry.

Three batteries: supersession nine unique probes × four matched replays = **36 turns** (not 36 independent questions), census **34 turns**, full longhistory **104 turns**, plus one separate demand turn and one identity pair. Sup repetitions restore the same durable pre-turn input before each complete turn; they do not keep answers from preceding repeats in repository state. Census/longhistory retain all original probes, scripted feed turns and continuation order.

Forecast evidence: [C2 A3 archived timing evidence](../grm_c2/amendment_A3_timing_evidence.json) gives width-96 census 201.836 s, longhistory 933.598 s, sup fixture totals 243.512 s including 63.680 s over nine probes. C8 adds a heuristic 25% instrumentation contingency and 20 s model/reload/persist cost per segment; sup repeats add 3× corresponding archived per-probe time. Identity 120 s and demand 60 s are heuristics. These are forecasts, not measured bounds, and archived timings have contention caveats. They were frozen before gates; none was lowered to force a fit.

| Cell | Profiled turns | Estimated s | Depends on |
|---|---:|---:|---|
| `gpu-identity` | 1 | 120 | none |
| `sup-correction_then_restatement` | 8 | 150 | gpu-identity |
| `sup-fresh_fact_controls` | 12 | 176 | gpu-identity |
| `sup-multi_hop_a_b_c` | 8 | 150 | gpu-identity |
| `sup-short_correction_long_competitor` | 8 | 150 | gpu-identity |
| `census-000-008` | 8 | 80 | gpu-identity |
| `census-008-016` | 8 | 80 | census-000-008 |
| `census-016-024` | 8 | 80 | census-008-016 |
| `census-024-032` | 8 | 80 | census-016-024 |
| `census-032-034` | 2 | 35 | census-024-032 |
| `longhistory-000-008` | 8 | 110 | gpu-identity |
| `longhistory-008-016` | 8 | 110 | longhistory-000-008 |
| `longhistory-016-024` | 8 | 110 | longhistory-008-016 |
| `longhistory-024-032` | 8 | 110 | longhistory-016-024 |
| `longhistory-032-040` | 8 | 110 | longhistory-024-032 |
| `longhistory-040-048` | 8 | 110 | longhistory-032-040 |
| `longhistory-048-056` | 8 | 110 | longhistory-040-048 |
| `longhistory-056-064` | 8 | 110 | longhistory-048-056 |
| `longhistory-064-072` | 8 | 110 | longhistory-056-064 |
| `longhistory-072-080` | 8 | 110 | longhistory-064-072 |
| `longhistory-080-088` | 8 | 110 | longhistory-072-080 |
| `longhistory-088-096` | 8 | 110 | longhistory-080-088 |
| `longhistory-096-104` | 8 | 110 | longhistory-088-096 |
| `demand-lh033` | 1 | 60 | gpu-identity, longhistory-032-040 |

Total 24 cells / 2,591 s. Every individual estimate ≤285 s, but **the campaign is NON_FIT at the order budget**. No cell has started and no reservation exists. This plan must not be attempted or retried into a longer lease. A different campaign or budget requires a separate lead order/amendment; this implementation does not grant one.

## CPU receipts and zero-change pins

[Raw CPU log](cpu_baseline.log): **88 passed, 2 warnings in 9.69s**; exit 0. Warnings were unsuppressed SwigPyPacked/SwigPyObject import deprecations (a swigvarlink deprecation also printed at interpreter exit). This is the relevant C8 + unchanged EB1 + unchanged C2 profile battery, **not the entire repository suite**.

Command:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_grm_c8_profiler.py tests/test_grm_eb1_ephemeral_frame.py tests/test_grm_c2_profile.py
```

Pins in [C8 tests](../../tests/test_grm_c8_profiler.py):

- `test_default_off_exact_object_no_instrumentation`: same object, no meter calls or environment changes.
- `test_fake_complete_pipeline_utf8_identity_and_state`: nonempty Unicode served bytes and mutation sequence match OFF/ON over all nine stages.
- `test_real_eb1_step_cpu_fake_byte_identity`: four-turn actual EB1 `step` path with model seams faked; projected answer/mount/frame receipts byte-identical.
- Nested fold/deposit attribution, exception identity and tracer restoration, refusal to replace another tracer, actual `_attempt` phase map, single-token-prompt handling, sum-vs-max metrics, nearest-rank quantiles and ratio-of-sums checked.
- Decision `< 0.50`, `== 0.50`, `> 0.50`; missing/failed/nonfinite/negative/nonconserving measurements; registration forgery; NON_FIT before filesystem reservation; worker parent guard; no fabricated allocation verdict checked.

[Mutation results](mutation_results.json): **5 valid, 5 killed, fraction 1.0 ≥ registered 0.80**. Mutants: default ON, ≤50% decision, missing count gate, missing fold ownership, floor p95. In-memory source copies only; original profiler hash unchanged. [Harness](../../scripts/grm_c8_cpu_mutations.py) and every mutant log are hashed in the receipt. These are author baselines; **blind verification has not run**.

Registered GPU pin ([worker](../../scripts/grm_c8_cells.py:100)): plant census prefix 0–4 once; persist pre-probe5 state; same loaded model; OFF then ON from identical repository copies. Compare nonempty raw `_forward` array hashes with shape/dtype, exact generated IDs, and served UTF-8 bytes. Gate is **NOT_RUN**, not passed by the CPU fake. `--profile-turn` is required for measuring workers, while the API remains flag-OFF by default.

[Dry run](dry_run.json), exit 0: all 24 cells enumerated, GPU false. [Preflight raw log](preflight.log), exit 1, expected stop:

```text
ValueError: NON_FIT_BUDGET: 2591s > 1800s; never execute/retry this plan
```

[Blocked report](blocked_report.json), exit 0. [Unmeasured summary](unmeasured_summary.json), exit 2, `BLOCKED_NOT_MEASURED`; all allocation decisions null. Report functions do not promote absent or partial rows to a completed battery.

## Lead commands and rails

[Exact lead_commands.txt](lead_commands.txt) is a dependency-ordered shell script with `set -euo pipefail`. Its mandatory CPU preflight **stops before the first GPU invocation** for the current NON_FIT registration. The cell lines are concrete future-review material, not permission to remove that stop. It specifies `/tmp/forge-gpu.lock`, 285 s lease, 280 s worker timeout, 240 s maximum lock wait, outer TERM at 585 s + 5 s grace, and 30 s cooldown between cells. Campaign reservations are serialized and a full 285 s is required from remaining budget before each next start. A failed/unfinished reservation blocks continuation; no retries or lock clearing. The lead has no runnable authorized GPU cell under this registration.

The registered decision is per battery: if decode <50%, session routing/admission wins the next allocation; otherwise APA decode integration is competitive and the lead reports both. No pooled average hides conflicting batteries. With no GPU rows, comparison to the nondecode share is **unavailable**. The reducer exposes the signed SP counterfactual `decode_fraction × (82.6−84.2)/82.6` as reasoning-only transfer from S=2048, not a width96 measurement. A 96-seat token band does not fix total context at 12,288.

## Prior art

Local, verified in this worktree: C2 / EB1 / RS3 / DET1 / CMC1, GRM contributors (2026): reuse profile geometry, original battery fixtures/serving, durable replay and foreground leases. New work is EB1 stage attribution and experimental receipt glue; no routing or admission algorithm was introduced.

External **unverified — lead to check**: Graham/Kessler/McKusick, *gprof* (1982), nested exclusive timing; Python `sys.settrace`, van Rossum/Python (1991 onward), execution callbacks; NVIDIA CUDA events (2007 onward), synchronized intervals; CUDA 11.2 (2020), default async-pool high-water counters; Amdahl (1967), fixed-component saving bound; DeMillo/Lipton/Sayward (1978), mutation testing. Search terms: `gprof 1982`, `Python sys.settrace`, `CUDA UsedMemHigh 11.2`, `Amdahl 1967`, `Hints on test data selection 1978`. Standard nearest-rank quantiles reused; specific originating author/year not known to me. No prior art known to me beyond these systems for this specific adapter; no novelty claim. The user-designated lead checks literature through its proxy.

## Deviations, RED, safety and model

RED: GPU prohibited in sandbox; 791 s forecast budget excess; SP5 context mismatch; whole-device transient peak not available; GPU timer behavior, GPU bit identity, demand trip, timings and allocation decision unmeasured. Instrumented wall includes trace/sync overhead. Sup repeats have nine unique inputs and a different harness-IO boundary from session batteries. No complete E2E quality or certification claim. **Not claimed fixed:** budget fit, exact whole-device memory peak, demand fire/actual trip and any live GPU gate.

No plan/registration/threshold amendments after gates; no test weakening. Added only new scripts/tests/artifacts in the writable worktree, plus a temporary registration builder in `/tmp`. No git commands, subagents, background jobs, GPU probes, GPU allocations, service operations, foreign-process signaling, process kills, or external messages were performed. Normal CPU tests/imports ran with `CUDA_VISIBLE_DEVICES=''`. Existing services were not inspected or altered. cwd and worktree pointer text were inspected; branch was not independently verified via git.

Agent identity: **GPT-6**, exact deployment identifier not exposed in this session; requested reasoning effort **high**. Served model: **openai/gpt-oss-20b**, revision **6cee5e81ee83917806bbde320786a8fb61efebee**; frozen C2 Harmony sink says **Reasoning: low** (model inference setting, distinct from agent effort).


# Amendment 1 — 2026-09-09 (supersedes r1 cap/context disposition)

**READY_FOR_LEAD; CPU author baseline PASS. GPU results NOT_MEASURED.**
The lead raised the cap to 2,700 seconds (0.75 GPU-h) and accepted the SP5
context correction. All 24 original cells remain: estimated 2,591 seconds
(0.7197 GPU-h), 109 seconds below the cap. This is a forecast from the
original registered reasoning/archived C2 timings, not a new timing result
or a guarantee of completion within the cap. Runtime overrun still stops.
The original order, registration and separate demand registration retain
their bytes and hashes. Historical r1 reports/receipts above are retained;
the r1 NON_FIT decision was correct under its then-authorized 1,800 seconds.

## 1. Amendment path + sha; test names; total estimate under the cap.

- Amendment: `artifacts/grm_c8/amendment_1.json`
- SHA-256: `91d080971bde54bf2c35ae20dc9955e533e9d112c20e915926c1aee296ca4a60`
- Original registration SHA-256: `264ccd493f9f73fd157c47a7fa0ded3be60e0b3530e0ad3b42fae13d872d8345`
- Preflight receipt: `artifacts/grm_c8/amendment_1_preflight.log`, exit0,
  `READY_FOR_LEAD`, 24 cells, estimate2591s, cap2700s.

The amendment pins the original registration, predecessor runner template,
new runner template, lead amendment order, unchanged demand registration,
new tests and exact executable commands. The runtime checks these before
GPU imports or reservations. The original registration is available through
`registration()`; execution uses an amended copy through
`effective_registration()`. The SHA is a reviewed integrity anchor, not a
cryptographic signature or protection against an attacker rewriting the
verifier itself. The registration and amendment were written before gates.

**Evidence class: CPU suite run, author baseline only.**
`amendment_1_cpu_baseline.log`: **113 passed, 2 warnings in 10.02s**, exit0.
This includes the 25 new parametrized amendment cases and all 88 original
C8/EB1/C2 cases. Original tests and profiler source were not edited.
The Swig deprecation warnings are retained verbatim in the log.

New test function names (`tests/test_grm_c8_amendment.py`):

- `test_amendment_cap_all_cells_and_original_registration_immutable`
- `test_preflight_reads_amendment_without_gpu`
- `test_forged_amendment_refused_before_reservation`
- `test_stale_amendment_semantics_even_with_rehashed_file`
- `test_stale_amendment_input_refused`
- `test_missing_amendment_refused`
- `test_resume_skips_all_started_states_without_launch`
- `test_resume_stale_receipt_is_red`
- `test_prior_red_blocks_unstarted_cell`
- `test_forecast_all_24_reservations_fit_and_overrun_refused`
- `test_unstarted_final_cell_uses_remaining_lease_cpu_fake`
- `test_complete_summary_stage_table_demand_and_decision`
- `test_summary_withholds_allocation_for_missing_red_demand_identity_or_conflict`
- `test_lead_commands_all_cells_in_order_resume_and_final_summary`

**Evidence class: author mutation baseline, not independent verification.**
`amendment_1_mutation_registration.json` froze five mutations after the
passing CPU baseline and before mutation gates. All5 were valid/rejected,
kill fraction1.0 >=0.80: ignore_cap, retry_complete, red_skip_success,
unclipped_reservation, ignore_incomplete_allocation. Receipts:
`amendment_1_mutation_results.json` and individual logs. Original source
remained unchanged. Harness: `scripts/grm_c8_amendment_cpu_mutations.py`.

**Evidence class: CPU command checks.**
`amendment_1_cpu_command_results.json`: dry-run0, preflight0,
blocked-report0, JSON summary2, text summary2, shell syntax0. Summary exit2
is expected because no GPU receipts exist. Machine data:
`amendment_1_unmeasured_summary.log`; readable per-stage table:
`amendment_1_summary_table.log`. All missing timings are NOT_MEASURED,
not zeros supplied as results. Decode/nondecode shares and allocation
remain null/WITHHELD. No cells directory or GPU reservation was created.

## 2. Exact lead commands.

Run on the authorized leased runner from this worktree:

```bash
cd /mnt/ForgeRealm/wt/grm-c8
./artifacts/grm_c8/lead_commands.txt
```

The file is executable (0755). Its exact contents follow. The EXIT trap
prints the final per-stage mean/p50/p95 table, turn wall, decode/nondecode
shares, demand-lh033 observation and registered decision outcome on both
success and early RED. `--summary --json` selects machine-readable output.

```bash
#!/usr/bin/env bash
# Lead-only GPU campaign. Prior art: C8/C2/CMC1 (GRM, 2026), sequential
# create-only cells and foreground lease rails; ours: resume and final summary.
set -euo pipefail
cd /mnt/ForgeRealm/wt/grm-c8
export PYTHONDONTWRITEBYTECODE=1
# EXIT always prints the final summary, including after a RED/preflight failure.
finish() {
    campaign_status=$?
    trap - EXIT
    set +e
    CUDA_VISIBLE_DEVICES='' python scripts/grm_c8_cells.py --summary
    summary_status=$?
    if (( campaign_status != 0 )); then
        exit "$campaign_status"
    fi
    exit "$summary_status"
}
trap finish EXIT
CUDA_VISIBLE_DEVICES='' python scripts/grm_c8_cells.py --dry-run
CUDA_VISIBLE_DEVICES='' python scripts/grm_c8_cells.py --preflight
# All 24 cells, original dependency order. COMPLETE skips; RED/unfinished
# skips without retry and stops. Never clear a reservation or a shared lock.
# Foreground: <=285s lease, child <=lease-5s; <=590s outer; 30s cooldown.
# Timeout signals apply only to the command it starts. Sandbox: do not execute.
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell gpu-identity --profile-turn --resume
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell sup-correction_then_restatement --profile-turn --resume
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell sup-fresh_fact_controls --profile-turn --resume
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell sup-multi_hop_a_b_c --profile-turn --resume
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell sup-short_correction_long_competitor --profile-turn --resume
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell census-000-008 --profile-turn --resume
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell census-008-016 --profile-turn --resume
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell census-016-024 --profile-turn --resume
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell census-024-032 --profile-turn --resume
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell census-032-034 --profile-turn --resume
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell longhistory-000-008 --profile-turn --resume
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell longhistory-008-016 --profile-turn --resume
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell longhistory-016-024 --profile-turn --resume
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell longhistory-024-032 --profile-turn --resume
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell longhistory-032-040 --profile-turn --resume
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell longhistory-040-048 --profile-turn --resume
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell longhistory-048-056 --profile-turn --resume
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell longhistory-056-064 --profile-turn --resume
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell longhistory-064-072 --profile-turn --resume
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell longhistory-072-080 --profile-turn --resume
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell longhistory-080-088 --profile-turn --resume
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell longhistory-088-096 --profile-turn --resume
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell longhistory-096-104 --profile-turn --resume
timeout --signal=TERM --kill-after=5s 585s python scripts/grm_c8_cells.py --cell demand-lh033 --profile-turn --resume
# Final summary is printed by the EXIT trap above.
```

Resume is conservative: any existing cell directory means started.
Verified COMPLETE skips without rerunning. RED, incomplete, stale and
unreadable receipts skip without rerunning, return nonzero and stop the
campaign. A restart cannot clear a RED or find a substitute prompt.
The shell was exercised with CPU fake python/timeout executables: all24
cells appear in registered order with --resume; simulated third-cell RED
stops further cells, still runs summary, preserves its exit7. This proves
shell/control wiring only, not a GPU lease or model run.

The remaining-cap reservation is clipped to min(285,floor(remaining)).
A cell must fit its registered forecast plus5 seconds before launch.
At forecast, the final60s demand cell reserves169s and times out its child
at164s. This corrects the unconditional285s reservation that would
otherwise require2816s at the last cell. Forecasts, dependencies, turn
counts and gates remain unchanged. Per-worker lease <=285s, outer <=590s,
foreground cooldown30s; actual worker time is charged and failed/unfinished
reservations fail closed. Budget accounting retains r1's worker-wall scope;
queue wait and cooldown are outside GPU compute seconds, so elapsed campaign
wall is longer than the compute forecast.

The registered decision remains per-battery ratio of summed decode wall to
summed instrumented turn wall: strictly <0.5 means session routing/admission
wins; exactly0.5 and above means APA decode integration is competitive and
the lead reports both. All batteries, demand and GPU identity must complete
before an allocation conclusion. Conflicting batteries stay separate.
CPU hand-computed tables and boundary tests validate reporting logic only.

## Side B and evidence class

SP5 report `/mnt/Shared/APA_SP5_GPTOSS20B_Model_Test_Report_2026-09-08.md`,
section4, clean decode bullet: **S=2,048 tokens,32 synced steps**.
Evidence class: **external receipt, synced decode microbenchmark**;
locally inspected source report, not remeasured by this seat.

| Arm | ms/token | Saving vs standard, ms/token |
|---|---:|---:|
| Standard | 82.6 | 0.0 |
| APA two-pass | 84.0 | -1.4 |
| APA single-pass | 84.2 | -1.6 |

APA is1.4–1.6ms/token slower here, about1.69–1.94% of standard decode
wall. No positive decode saving is demonstrated. **The order's
12,288-token decode context was the lead's error**; that number is the
resident-memory ceiling, not the context of this decode timing table.
There is no SP5 S=12,288 or C8 W96 decode-speed measurement in this claim.
The report supplies both decode and nondecode share when C8 measurements
exist; none exist here. Transferring SP5's single-pass ratio to a C8 decode
fraction d would give turn saving d*(82.6-84.2)/82.6 (negative): **reasoning
only**, not a measured C8 speedup or allocation result.

## Prior art

- **C8/C2/CMC1, GRM contributors,2026**, inspected locally: SHA bindings,
  create-only reservations, durable continuation and foreground flock lease.
  Taken: those control and receipt methods. Ours: amendment successor binding,
  skip/stop CLI, remaining-cap clipping, shell finalizer and table rendering.
- **C8 summarize, GRM contributors,2026**: existing nearest-rank empirical
  quantiles and ratio of summed components. Taken unchanged; no new
  statistical algorithm. Specific historical origin of nearest-rank
  quantiles is not known to me.
- **Amdahl,1967**: fixed-component end-to-end speedup reasoning; the actual
  strict50% allocation rule is the GRM Scout/lead's registered2026 rule.
  Taken: component-fraction reasoning, not a novel optimization. External
  bibliographic attribution unverified — lead to check: `Amdahl 1967
  Validity of the single processor approach`.
- **DeMillo/Lipton/Sayward,1978**, mutation testing: seeded-defect rejection,
  plus the local C8 in-memory module-copy harness. Ours: five amendment
  substitutions. External attribution unverified — lead to check:
  `Hints on test data selection 1978`.
- Existing profiler prior art (gprof1982, Python tracing, CUDA event/pool
  metrics) remains as recorded in r1 and code; profiler unchanged. No new
  routing, selection or profiling algorithm is claimed. No prior art known
  to me for this precise adapter beyond the local systems listed above.

## Deviations; RED; process safety; model id and effort

Authorized deviations: cap1800→2700s and correction of the lead's SP5
context error. Necessary implementation detail: clip the final reservation
within the raised cap; no cell dropped or estimate changed. No other order,
product, kernel, battery, registry, config or flag-default changes.
The original command file is archived byte-for-byte as `lead_commands_r1.txt`.
New amendment receipts supplement r1 receipts instead of overwriting them.

**RED / not claimed fixed:** GPU identity, actual stage timings, natural
W96 demand-trip success, and allocation remain NOT_MEASURED. No end-to-end
GPU result or independent blind verification is claimed. The old budget
refusal is preserved as historical evidence; the cap now permits preflight.
SP5's slower APA decode result remains negative evidence, not a speed win.
Author tests and mutants cannot replace the lead's blind verification.

Process safety: no git commands, subagents, GPU work, service changes,
background shell jobs, shared-lock clearing or process-kill operations.
CPU children ran synchronously; no process was signaled. Tool transport
briefly yielded the running CPU-suite session, which was polled to completion
without launching parallel work; this was not a detached/background job.
The timeout commands in the lead artifact were only syntax/fake-shell tested;
no real GPU/timeout campaign ran. Never kill a foreign process. No cells
or reservations were created in the actual campaign directory.

Agent model: **GPT-6; exact deployment identifier not exposed**. Effort:
**high**. Served model registration remains **openai/gpt-oss-20b**, revision
`6cee5e81ee83917806bbde320786a8fb61efebee` (immutable r1 artifact metadata).
