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
