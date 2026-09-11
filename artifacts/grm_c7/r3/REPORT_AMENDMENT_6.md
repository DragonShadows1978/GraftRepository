# GRM-C7 amendment 6 — CPU GREEN; r3 registered; GPU NOT_RUN

Fixture lineage, plain prompt and control scorer are implemented in the harness. **128 registered r3/core tests + 7 historical-prompt regressions pass; 5/5 source-copy mutants killed (1.00 ≥ 0.80).** The executable lead launcher passes its CPU-only preflight. No 300-turn GPU run occurred under this seat's explicit **no GPU** instruction. Alias resolution remains **RED / not claimed fixed**; all alias probes are retained.

This report supplements the C7 narrative and preserves the amendment-5 stop report and all r2 receipts. The immutable plan is `orders/GRM_C7_AMENDMENT_6_R3.md`; gates were written before edits/tests in `artifacts/grm_c7/r3/amendment_6/gates_registration.json`. Its SHA-256 is `018a2385b9395d554bd14d9c30d3d91192e8d3461c11fdf9d54573fdb8ae382a`.

## 1. Fixture diff + new sha; prompt before/after; scorer note; tests

- Fixture: `artifacts/grm_c7/r3/fixture.json`; SHA-256 **`b648ebb71072d1c357146f6cedb22f6d93dd259a69cb50d912e1e9b564624c17`**.
- Historical fixture SHA-256: `18c4bc891115ac6d96f7c443ad023f3e923ee92ab6aac21a1f1150bd14408ff0` (unchanged).
- Exact diff: `artifacts/grm_c7/r3/fixture_delta.patch`; structured receipt: `fixture_receipt.json`; manifest: `fixture_manifest.json`.
- `scripts/grm_c7_register_r3.py::fixture_bytes` changes only turn 2: `kind='supersede'`, `old_value='Mica-431'`, and `correction_command`. The registered test compares raw JSON object byte spans for all 300 turns; **299 other turn spans are identical**. Every probe record and other top-level fixture field remains identical. The retained legacy `corrections` index has 14 entries; actual supersede events now total 15, explicitly recorded in the fixture receipt.

Turn 2's command is exactly:

```text
correct memory: current C7-Vesper-0 value is Mica-431 => Correction of both original records: the current C7-Vesper-0 value is Flint-511; the current C7-Vesper-1 value is Flint-512. The previous Mica values are obsolete.
```

Production `scripts/grm_e2e_session.py::run_turn` dispatches `correction_command` for `kind='supersede'` to `repo.apply_memory_command`. Matching Vesper-0 identifies the original shared Mica node; replacement text must carry **both** Flint values because retirement is node-wide. CPU tests exercise the real repository correction API, verify the explicit supersedes link and original-node retirement, preserve both Flint values in replacement text, retain both subsequent Onyx values, and compare persisted metadata after reload. This is a CPU lifecycle test with numerical doubles, not model recall or full production-pipeline acceptance. Separate supersession paths for each entity are not claimed; node-wide retirement of shared records remains a general design constraint.

Registered prompt before (example):

```text
What is the current C7-Fresh-0 value? Reply only with the answer; if unspecified, reply UNKNOWN.
```

Historical r2 executed prompt before:

```text
What is the current C7-Fresh-0 value? Reply only with the answer; if unspecified, reply unknown.
```

r3 executed prompt after:

```text
What is the current C7-Fresh-0 value? Reply only with the answer.
```

Production template, quoted from `scripts/grm_e2e_session.py::probe_turn`:

```text
Recall probe. What is the current {fact_id} value? Reply with only the value.
```

Both production and r3 omit an abstention clause; task-specific C7 questions remain distinct. The transform is `scripts/grm_c7_run.py::effective_question`, gated on revision `r3`, and is shared by memory calls, oracle calls and restart sentinels. `effective_questions.json` records all 60 original/effective question pairs. The source-record oracle prefix is retained. Applying the transform at execution preserves the frozen probe-turn bytes.

Scorer: `scripts/grm_c7_common.py::score` casefolds **only normalized full-string UNKNOWN comparison on unanswerable controls**. `unknown`, `Unknown`, and `uNkNoWn` pass that control. Answerable exact values remain case-sensitive; prose synonyms and strings containing both an abstention and a value receive no exact-match credit. The existing abstention/error categorization is otherwise unchanged. Historical score receipts are not rewritten.

| Evidence class | Receipt under `r3/amendment_6/` | Result |
| --- | --- | --- |
| CPU RED before repair | `cpu_red.log` | 6 failed, 1 passed: missing supersede, retained clause, mixed-case control mismatch |
| CPU staging | `cpu_green_staging.log` | 83 passed, 2 failed; mechanical test-context corrections documented below |
| Focused lifecycle recheck | `cpu_lifecycle_context.log` | 1 passed after using the persisted manifest |
| Registered r3/core CPU suite | `cpu_registered.log`, `cpu_registered.json` | 128 passed |
| Unmodified FIX-4 historical-prompt CPU suite | `cpu_legacy_fix4.log`, `.json` | 7 passed |
| Source-copy mutation | `cpu_mutations.json` and five named logs | 5 killed, 0 survived, 0 errors; live sources unchanged |
| Executable lead CPU check | `lead_check.log` | exit 0; free 283,625,955,328 bytes ≥ 20,000,000,000 |

The baseline covers fixture scope, actual correction/reload, plain oracle/source receipt and sentinel calls, exact scoring, revision propagation into the leased child, registration tamper, COMPLETE-only resume/checkpoint integrity, cap boundaries, free-space boundaries and existing FIX-3/4/5/capture/fold/oracle regressions. Mutation operators remove casefolding, retain the clause, restore the wrong 7200s cap, drop the child revision, or disable free-space rejection. Author verification only; no blind audit claimed.

## 2. r3 registration path + sha; exact lead command; the memo path

Registration: **`artifacts/grm_c7/r3/registration.json`**; SHA-256 **`65c543b1904e6e3bd81d95316dbc092fd72f70ae164044cce036ce8a6ec04b1c`**. This is a fresh revision registration with explicit parent/gate/source bindings, not an edit to an older registration. The source manifest SHA-256 is `9f335bd1fc31186fca8892f5be040f9b75c7a038178b0f97f7d51ed530343245`; verification checks 467 registered inputs. `amendment_6/ready.json` binds successful CPU receipts to this exact registration.

Protocol: **39 cells, arm A only, unchanged C2 profile, FIX-3/4/5 ON in the pinned core, FIX-7 absent, cap 2.2 GPU-h = 7,920 seconds.** Old r2 checkpoints cannot enter r3. The misleadingly named `GRM_C7_FIX4` selects the historical r2 checkpoint bridge, not the core fix; r3 forbids that bridge. The launcher clears it and disables alias-follow. The C2 worker environment retains revision `r3` and its lease-parent marker. All new campaign receipts route to `artifacts/grm_c7/r3/cells/`.

The 39 original turn ranges and their 280-second worker limits are preserved. Each cell reserves against the remaining 7,920-second cap; incomplete/orphan reservations consume their full charge. Leases remain 285 seconds, foreground lock wait ≤240 seconds, outer limit 590 seconds, foreground cooldown 30 seconds. The inherited timing proxy is not a newly measured fit guarantee: FIX-5's larger generation allowance and the extra correction may increase time. Hitting a rail is RED, with no retuning/retry.

Exact **lead GPU command**, executable and resumable:

```bash
/mnt/ForgeRealm/wt/grm-c7/artifacts/grm_c7/r3/lead_commands_r3.txt
```

The script checks ≥20 decimal GB before the run and before every new cell, verifies ready receipts, and skips only COMPLETE cells with matching controller/worker bindings and a valid checkpoint tree. Started failures, orphaned directories, stale bindings or an owner marker stop the script. It uses the existing foreground lease/owned-worker machinery; it never clears shared locks. Its GPU-launching form was **not executed here**.

Exact CPU command executed:

```bash
GRM_C7_REVISION=r3 CUDA_VISIBLE_DEVICES='' python -m scripts.grm_c7_r3_cpu
```

Exact CPU-only launch check executed:

```bash
CUDA_VISIBLE_DEVICES='' /mnt/ForgeRealm/wt/grm-c7/artifacts/grm_c7/r3/lead_commands_r3.txt --check-only
```

Registered prediction, verbatim in meaning: **fresh ≥10/12; folded ≥10/15; corrections ≥6/13; aliases ≤2/12; residency bounded; restarts retained.** Aliases are expected failures, not excluded rows or credited successes. Original exhaustive acceptance still includes alias errors; prediction is separate from acceptance.

**Denominator mismatch remains explicit:** the fixture has 15 probes per class (10 answerable + 5 controls), and the order does not identify subsets matching 12/15/13/12. Optional clarification was asked during work; no answer arrived before registration. All 60 probes remain registered and all actual class/distance counts will be reported. Prediction evaluation is `UNRESOLVED_DENOMINATORS`; no invented subset, ratio conversion or acceptance weakening.

Memo: **`artifacts/grm_c7/r3/ALIAS_DESIGN_OPTIONS.md`**. It compares two-hop read, fold-merge and alias-in-base rewrite, their code surfaces, query/deposit costs, fit constraints and revision semantics. **Receipts favour investigating fold-merge first**, with two-hop read as fallback: FIX-5 accepted 13/13 existing GPU folds and preserved co-located Handle aliases and values. The exact Signal/Base evidence was folded separately, so pairing, 96-seat fit and alias recall remain unproved. No alias mechanism was implemented.

## Prior art

**Verified local GRM contributors (2026):** C7/C2 immutable registrations, source hashes, foreground leased cells, reservation accounting and checkpoint validation; production `supersede_turn`/`run_turn`/`correct_memory`; RD1 plain-prompt contrast; M5/L2 revision semantics; SC1 demand re-routing; FIX-3/4/5. Taken: these existing contracts and algorithms. New work: r3 fixture/launch integration, control-only comparison amendment, tests and evidence synthesis. **No prior art known to me for this exact composition; no new memory algorithm claimed.** Attribution appears at code sites and in the ledger.

The scorer retains the earlier exact-QA/answerability attribution to Rajpurkar et al. (2016) and Rajpurkar/Jia/Liang (2018); this control grammar is not their normalization implementation. The memo cites **[Press et al. (2022), self-ask with search](https://arxiv.org/abs/2210.03350)** for intermediate-question retrieval and **[Sarthi et al. (2024), RAPTOR](https://arxiv.org/abs/2401.18059)** for summaries as retrieval representations. These primary sources were checked via read-only web access; they support antecedents, not GRM performance claims. For deposit-time alias propagation, local supersession is verified; Gupta/Mumick materialized-view maintenance (1995) is **unverified — lead to check**, search terms recorded in the memo. No implementation code site exists for memo-only options.

## 3. Deviations; RED; process safety; model id and effort

- **Test-context corrections, not product changes:** staging called `_node_manifest` before the CPU double's flush populated `rare`, yielding `KeyError: 'rare'`. The corrected test reads the actual persisted manifest after `flush_now`. The historical FIX-4 byte test was initially run under r3's deliberately changed prompt and failed at byte 539 (`b'3' != b'4'`). Its original suite and frozen receipt remain unchanged; rerunning it in historical prompt mode passes. Separate `amendment_6/test_context_amendment.json` records both before the execution registration. No assertion, tolerance or gate threshold was weakened.
- **Scope/RED:** the no-GPU seat completed CPU implementation/registration/checks and prepared the lead-run campaign. GPU cells started: **0**; GPU time: **0 seconds**. `summary_A_NOT_RUN.json` records `NOT_RUN`, `complete=false`, residency `null`, and no restart scores. Model-level correction/fresh/folded improvement, bounded r3 residency and retained r3 restarts are **not measured**. Alias resolution is **not claimed fixed**. The prediction-denominator mismatch remains unresolved. Existing alias and fold receipts are evidence from prior runs only.
- **Integrity:** `amendment_6/final_integrity.json` verifies 423 protected files unchanged, including all inspected core source, historical r2 receipts and amendment-5 artifacts. The original fixture, orders and registrations are preserved. Harness source diff and new file receipts are under r3; no product code was edited. `SHA256SUMS_AMENDMENT_6` seals new artifacts without replacing amendment 5's checksum file.
- **Safety:** correct worktree/branch verified by reading `.git` and its referenced `HEAD`, without git commands. No subagents, GPU/model loads, background jobs/waits, process kills/signals, service actions, external messages or memory writes. All compute was foreground CPU work with owned test subprocesses. The prepared future lead command retains pre-existing owned-worker timeouts; none was exercised against a live GPU process here.
- **Engineering model:** **`gpt-6-astra`, effort `high`**, recorded in `logs/grm_c7_a6.log:6,10`. Registered model under test: **`openai/gpt-oss-20b`**, revision **`6cee5e81ee83917806bbde320786a8fb61efebee`**. No model inference executed in this seat.
