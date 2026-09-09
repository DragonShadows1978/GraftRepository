# GRM-C3 blocked report

**CPU PASS; GPU BLOCKED / UNRUN. No demand rule is certified or recommended for adoption.**

The authorized sandbox deliverable is complete: 48 immutable new cases, four offline arms, bounded foreground GPU commands, and a passing author CPU gate. The GPU mission remains unexecuted by this seat because the order explicitly supplies no GPU in the sandbox. No measurement, miss count, empirical weak-read qualification, or preservation result is inferred from the unit tests.

## Artifacts and registration

- Immutable plan: `orders/GRM_C3_DNGH_DECOY_CALIBRATION.md`; read unchanged. Branch metadata read directly from the worktree HEAD says `grm-c3`; no git command was used. Fork ancestry was not checked under the no-git rule.
- Registration: `artifacts/grm_c3/registration.json`; SHA-256 `048e4c7e786be8bb26685abbdd21ce710c9d0f4f2f8fa5e6b3722538820c5666`.
- Case-set SHA-256: `c46535aec23824a876f9198eebf58a67e73cab7faa12c42f45980e07938c6f30`. All 48 IDs, classes and per-case hashes: `case_set.md`. Full questions and nodes: `registration.json`.
- Execution manifest: `execution_manifest.json`; SHA-256 `f120564fa2e5224f3a5a86a883d2016d239571e3e92c3a7cb6e175eb3be0a9c3`. It fingerprints 346 source, native-library, tokenizer and model files (13,808,566,585 bytes), before CPU gates. Both model shard bytes and imported local harness sources are pinned. Every GPU trace/cell binds this manifest and the registration. Parent validates all inputs before leasing and code inputs again afterwards.
- CPU receipt: `cpu_gate.json`; **29 passed in 0.28 s**, pytest return code 0. Evidence class: author CPU unit-test baseline, not blind verification.
- Cell enumeration: `dry_run.json`. Dependency-ordered commands: `lead_commands.txt` (run with `bash artifacts/grm_c3/lead_commands.txt` only on the GPU lead host before any cell has started).

## Rules and unrun outcomes

All rules consume the SAME detector-OFF answer trace. The carried and candidate lines are used at full config precision without fitting; all mass comparisons are strict `<`. Empty-band uses actual mounted token count, not node-list length or a synthetic zero for missing traces.

| Rule | Frozen criterion | TP | FP | TN | FN |
|---|---|---|---|---|---|
| carried | any mass < 0.3380523274342219 | UNRUN | UNRUN | UNRUN | UNRUN |
| candidate | any mass < 0.3181141105790933 | UNRUN | UNRUN | UNRUN | UNRUN |
| near_zero | any mass < 1e-8 | UNRUN | UNRUN | UNRUN | UNRUN |
| empty_band | actual mounted token count == 0 | UNRUN | UNRUN | UNRUN | UNRUN |

Acceptance is frozen at TP >= 22/24 and FP <= 1/24, plus no lost correct answers. Offline reports count fires on **every actually correct OFF answer**, including a correct guess on a missing-knowledge case, as `correct_answers_at_risk`. Zero at risk is a conservative offline eligibility guard. It is not a measured count of lost answers. Demand-ON preservation remains UNMEASURED, so even a passing offline candidate cannot certify the full demand-flip bar.

Predictions remain predictions: near_zero fires on empty bands but misses a nonempty decoy; carried has an avoidable false fire on a correctly answered weak target. The eventual evaluator lists all separating cases, all first-fire indices, confusion counts, qualification failures, and named prediction witnesses. It explicitly reports when no rule meets the conservative offline bar. Current recommendation: **do not flip demand on this evidence**.

## Cells and wall estimate

| Cell | Cases (one per class) | Estimate | Worker cap | Outer cap | Cooldown |
|---|---|---:|---:|---:|---:|
| trace_01 | c3_01_correct, c3_01_decoy, c3_01_unavailable, c3_01_weak | 200 s | 225 s | 590 s | 30 s |
| trace_02 | c3_02_decoy, c3_02_unavailable, c3_02_weak, c3_02_correct | 200 s | 225 s | 590 s | 30 s |
| trace_03 | c3_03_unavailable, c3_03_weak, c3_03_correct, c3_03_decoy | 200 s | 225 s | 590 s | 30 s |
| trace_04 | c3_04_weak, c3_04_correct, c3_04_decoy, c3_04_unavailable | 200 s | 225 s | 590 s | 30 s |
| trace_05 | c3_05_correct, c3_05_decoy, c3_05_unavailable, c3_05_weak | 200 s | 225 s | 590 s | 30 s |
| trace_06 | c3_06_decoy, c3_06_unavailable, c3_06_weak, c3_06_correct | 200 s | 225 s | 590 s | 30 s |
| trace_07 | c3_07_unavailable, c3_07_weak, c3_07_correct, c3_07_decoy | 200 s | 225 s | 590 s | 30 s |
| trace_08 | c3_08_weak, c3_08_correct, c3_08_decoy, c3_08_unavailable | 200 s | 225 s | 590 s | 30 s |
| trace_09 | c3_09_correct, c3_09_decoy, c3_09_unavailable, c3_09_weak | 200 s | 225 s | 590 s | 30 s |
| trace_10 | c3_10_decoy, c3_10_unavailable, c3_10_weak, c3_10_correct | 200 s | 225 s | 590 s | 30 s |
| trace_11 | c3_11_unavailable, c3_11_weak, c3_11_correct, c3_11_decoy | 200 s | 225 s | 590 s | 30 s |
| trace_12 | c3_12_weak, c3_12_correct, c3_12_decoy, c3_12_unavailable | 200 s | 225 s | 590 s | 30 s |

Reasoning-only estimate: 2,400 GPU seconds (0.667 GPU-h), plus 360 seconds of cooldown = 46 minutes, plus CPU fingerprint/preflight overhead. Maximum registered GPU worker budget: 12 x 225 = **2,700 seconds (0.75 GPU-h)**. Each cell serially loads four fresh model processes; 25 seconds load + 25 seconds planting/generation/cleanup per case is an estimate, not a C3 benchmark. No existing SC2 results were used as C3 observations. The 225-second total deadline includes child startup, planting and answer observation. Timeout records NON_FIT and forbids continuation/retry; failures are RED. A cell that cannot fit does not receive a longer lease.

The four CPU evaluation arms have equal standing and require no additional GPU cells. Each GPU cell includes every class, with deterministically rotated order. It generates one answer per case; node planting/harvest necessarily performs preparation forwards, which are not additional answer passes.

## CPU gates

Fixture validity checks 48 unique IDs/questions, balanced 12/12/12/12, hashes, planned mounts, topical decoys, absence of oracle values from decoy repositories and all prompts, empty unavailable repositories, weak target types, novelty against existing fixture JSON and GRM scripts, exhaustive cell coverage, and budget rails. No SC2 fit fixture is reused.

The hand-computed six-trace table separates all four arms: minima 0.4, 0.32, 0.30, 1e-9, empty zero, and nonempty zero. Exact threshold equality does not fire; the adjacent representable number below does. Negative, NaN, infinite, out-of-range, empty traces and inconsistent empty-band masses are rejected. A synthetic 48-row table pins TP/FP/TN/FN to literal expected integers; it is **test data only**, never model evidence. Additional tests pin the 22/24 and 1/24 boundaries, duplicate/missing rows, qualification failures, fire-on-correct accounting, immutable writes, tampered fixture detection, and the one-attempt observer adapter using a fake arena.

## Prior art

- Local **GRM-DET1, SC2 and SC1.2**, GraftRepository contributors (2026), verified in this source tree. Imported: DET1.3 `_install_lived_nodes` (chronological Harmony planting), DET1.5 workers/campaign answer comparator and cleanup, DET1 `DetectorObserver` (full-layer mass), DET1.2 model/repo loader, SC2 `_first_below`, SC1.2 `min_mounted_mass`, and production `_attempt`/EB1 reset. No harness is copied or modified.
- The fixed thresholds and empty-band comparison come from the order/Scout and config, not a new detector invention. New campaign contributions: the 48 texts, controlled mounts, registered 1e-8 tolerance, balanced cell layout, offline aggregation and bounded parent wrapper.
- Fisher (1935), *The Design of Experiments*: balanced blocking is a known design idea; this schedule is deterministic, not randomized. **Unverified — lead to check**: “Fisher 1935 randomized blocks”.
- Chow (1970), error/reject tradeoff: known selective-decision context, not the source of the D-NGH signal. **Unverified — lead to check**: “Chow 1970 optimum recognition error reject tradeoff”, “attention mass missing knowledge detection”, “attention-based uncertainty”.
- NIST SHA-256, FIPS 180-4 (2015), POSIX flock and Python subprocess/GNU timeout supply routine integrity and process control; no novelty claimed. **Unverified — lead to check** official standard/documentation editions. Hash serialization here is compact sorted-key UTF-8 JSON, not a claimed RFC 8785 implementation.

## Deviations and RED

1. Controlled mounts deliberately set the registered target/decoy/empty condition and execute the production `_attempt`; they do not measure the natural router or the full admission/grounding ladder. This is a mounted-band detector experiment. Reusing all old DET1 snapshot campaign machinery would require historical fixtures and extra answer passes, contrary to the new-text/single-answer design. Lower-level planting, observer and scoring functions are imported.
2. Weak number/name texts target the t30/t33 class. CPU checks cannot establish weak model reading or even correct answers. The runner records observed answers and full mass traces. Incorrect correct/weak controls fail achieved-class qualification; they remain in the 48-row report, are never filtered/replaced, and block eligibility. Empirical weakness is left unconfirmed instead of being defined after seeing results.
3. Demand is OFF throughout. Actual lost correct answers under demand-ON are unmeasured. No full flip claim can be made from this protocol alone; a successor preservation gate requires a separate lead order.
4. No GPU run, no empirical case separation, no GPU API integration validation and no measured wall time. These are **not claimed fixed or passed**. Blind verification is pending with the lead under HOUSE_RULES section 8; this seat did not select or launch a verifier.
5. `cpp/build/libgrm_runtime.so` is absent in this worktree. The existing main-checkout library is pinned and passed through a temporary imported-loader variable override. No build, copy, reinstall or library mutation was performed. Its compatibility with this worktree is untested on GPU and any failure is RED.
6. No registration changes or amendments were needed during this seat's CPU preparation. If lead source inputs drift, do not overwrite this manifest or registration: stop and issue a separate source amendment plus gates.

## Process safety and identity

No git commands, subagents, GPU probes, services, background jobs, external messages, process killing, lock clearing, or default/config edits. Foreground CPU commands only. Runtime lead wrapper takes 30 seconds of foreground cooldown, uses nonblocking flock on `/tmp/forge-gpu.lock`, and runs only its own child processes sequentially with the remaining shared deadline. Its timeout can terminate only a child it started. Locks are released by closing/unlocking, never unlinked. Failed/started cells cannot be rerun.

Agent identity available to this seat: GPT-6; exact backend variant not exposed. Requested reasoning effort: high. Experiment model: `openai/gpt-oss-20b`, revision `6cee5e81ee83917806bbde320786a8fb61efebee`, TensorCUDA packed MXFP4 expert path, greedy 32-token maximum with inherited Harmony sink/template and no explicit model reasoning-effort override. Actual model-info and environment are recorded per lead-run trace.

Carried-threshold caveat: THIN ENVELOPE: this threshold was fit on 2 calibration turns (served controls only, strict minimum-of-minima envelope). It is CARRIED verbatim from the GRM-DET1 race and is never refit or adjusted from serving traffic. Any receipt that reports a D-NGH decision must restate this caveat.
