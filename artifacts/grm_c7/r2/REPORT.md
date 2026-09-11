# FIX-3 + C7 r2 — CPU GREEN; GPU NOT_RUN

**Implemented and registered. Final author CPU suite: 112 passed; six of six source-copy mutants killed (1.00 >= 0.80). No GPU execution. GPT-OSS UNKNOWN causality and post-fix digest quality remain RED-unresolved.**

Evidence: `launch_amendment/cpu_registered.log` and `.json`, `launch_amendment/cpu_mutations.json`, `launch_amendment/dry_run.json`, `launch_amendment/summary_A_NOT_RUN.json`. The earlier 107-test/5-mutant baseline remains on disk, followed by a separately registered launch correction. This continues the C7 synthesis; original r1 reports/receipts are untouched.

## 1. FIX-3 file/lines, test names, RED-before/GREEN-after evidence, site audit rulings

`core/graft_arena.py:2202` routes the existing consolidation user instruction through `_format_step_prompt`, which calls the configured **existing `harmony_turn(user_text, None)`** (`scripts/grm_e2e_session.py:108`). It retains the registered assistant primer after the final-channel opening. No duplicate Harmony formatter, feature flag, kernel change or new model adapter. `:2225` checks the configured serving stop strings before another forward; `:2235` strips them from the candidate. The four EB1 stops are `<|return|>`, `<|end|>`, `<|start|>user`, `<|start|>assistant`.

Dispatch recognizes the configured formatter's terminal `<|start|>assistant<|channel|>final<|message|>` boundary. Unconfigured models and other configured templates retain the legacy prompt and generation bytes (source reasoning and CPU controls); this is format-contract dispatch, not checkpoint-name guessing. Common QC is intentionally tightened for all generated digests. A different Harmony channel template is outside this exact serving contract and is not claimed covered.

`core/graft_arena.py:2038` adds character QC: reject three or more Unicode ellipses or six or more punctuation characters, allowing whitespace between them. Ordinary three ASCII dots remain valid. `:2248` applies QC before coverage; list relaxation cannot rescue punctuation collapse. QC-rejected candidates record coverage 0.0 without computing it; this is a rejection sentinel, not a measured lexical coverage. Extractive eras are unchanged.

**Budget:** `CONSOLIDATE_NGEN = 120` (`:1908`), still selected by `consolidate(..., ngen=None)` (`:2138`). Explicit `ngen` remains honored. `core/graft_repository.py:3261` still calls `arena.consolidate(idxs)` without an override. `docs/GRM_Methodology.md:240` onward registers the 0.70 fidelity rule, but no alternate digest token budget. C7's 32-token probe/oracle budget remains separate. No number was changed silently.

| CPU gate | Evidence |
|---|---|
| `test_fold_harmony_wrapper_and_stop` (`tests/test_grm_scout_fix3.py:17`) | All four stops × turn/digest/era; exact wrapped input, no forward after stop, no suffix leak, accepted source fact. |
| `test_fold_explicit_budget_and_legacy_prompt` (`:38`) | Explicit budget plus legacy no-template and non-Harmony-template controls. |
| `test_digest_qc_rejects_punctuation_collapse` / `test_digest_qc_accepts_ordinary_punctuation` (`:61`, `:67`) | Collapsed ellipses/repeated/mixed/spaced punctuation rejected; ordinary punctuation retained. |
| `test_degenerate_qc_precedes_coverage_even_with_list_relaxation` (`:71`) | Coverage callback must never run on degenerate candidates, even with list relaxation enabled. |
| `test_fold_coverage_bar_remains_point_seven` (`:89`) | Real fact extraction/coverage rejects 6/10 and accepts 7/10; retirement agrees with acceptance. |

RED receipt `cpu_fix3_red.log`: **17 failed, 5 passed** before core edits. Initial treatment `cpu_fix3_green.log`: **22 passed**. Expanded controls and final suite pass in the final receipt above. The remove-wrapper, remove-stops and disable-QC mutants each fail the relevant unchanged repair gates. Existing CPU recovered-fold guards, S4 fold-order/acceptance gates, C7 folded-lineage checks and SCOUT-FIX-2 tests pass in the final suite. Historical E4/GQA model-loading scripts are GPU gates and were not executed.

| Generation / restore site | Ruling |
|---|---|
| Repository `_fold_once`, `core/graft_repository.py:3261`; first-generation folds | **FIXED through shared consolidate.** Repository caller unchanged. |
| `ERA_PROMPTS` / deep digest and era re-folds, `core/graft_arena.py:1890`, `:2197` | **FIXED through the same loop**, covered for both digest and era source kinds. Optional source scaffolding still feeds the same wrapper dispatch. |
| Extractive era path, `core/graft_arena.py:2160`; `_deposit_consolidation` | **No generated continuation.** Concatenates child text and harvests/deposits a known note; unchanged. |
| Serving `_attempt`, `core/graft_arena.py:4380`; `_resume_attempt`, `:4562` | **Already compliant.** Fresh prompt is wrapped; resumed generation retains the original prefix and stop set. No fresh unwrapped prompt on resume. |
| `feed`, `:2640`; `_harvest`, `:350` and `:4751`; `_forward`, `:2631` | **Not independent free-generation paths.** Known-text ingestion/payload capture or a low-level forward used by the audited caller; unchanged. |
| Repository `_load_node`, `:3894`; `load`, `:4770`; WAL recovery, `:4074`; C7 cold restore | **No digest synthesis on restore.** Restore metadata/payloads, preserve no-fold markers; future folds use fixed consolidate. |
| Repository `_s2_generate`, `:3353` | **Further finding, unchanged.** Optional salience-rating forced continuation uses raw prompt and no configured stop check, default 24 tokens; `s2_salience_enabled=False` at `:254`. Separate rating protocol, not a fold/restore path. Not claimed Harmony-correct; queue separately before enabling it on Harmony. |

Core arena SHA-256: `cf242ddfe0ff29ac2114e98768f5cf66939f0ff46886658e50a4f492fcdb1fa4`. Unchanged repository caller SHA-256: `fc6b9448efb45c29d5d2271fe929e3e4b6519867567cf1c69588b2a99e4773db`. These are file hashes, not a new git commit. Exact delta: `core_source_delta.patch`.

## 2. Harness fixes file/lines + gate names; r2 amendment path + sha; exact lead commands

- `scripts/grm_c7_run.py:108`: lowercase **only** the final fallback instruction `if unspecified, reply UNKNOWN.` at execution. This prevents an instruction word from becoming an admission identifier under the frozen core law. Entity spelling, sources, expected answers and core admission are unchanged. The worker (`:395`), restart sentinels (`:327`) and oracle (`:128`) all use it. `effective_questions.json` records all 60 original/effective query pairs; actual prompt bytes are explicitly amended, not described as unchanged.
- `:128`: oracle establishes each layer's `live_shift = arena.live_shift` before `_attempt`, records the actual wrapped prompt and encoded IDs, and restores every saved shift/cache field even on exception. The source block and exact scorer are retained. It passes live source text, never `expected`, to the reader.
- `:119`, `:404`: probe rows carry actual `question`, original `registered_question`, and `expected` as well as both answers.
- `:26`, `:40`, `scripts/grm_c7_common.py:139`, `scripts/grm_c7_register_r2.py:31`: r2 receipt namespace, full historical amendment chain, changed core/harness hashes and current amendment in checkpoint/controller bindings. r1 execution against changed sources fails closed; r1 checkpoints cannot resume r2.
- `scripts/grm_c7_run.py:487`: explicitly retain `GRM_C7_REVISION` after the C2 environment builder removes ambient `GRM_*` variables. Without it the owned child would select r1 and reject the new sources. Found in final source audit; separately registered in `launch_amendment/registration.json` before its gate. `launch_amendment/cpu_red.log` preserves `assert None == 'r2'`; final gate passes. The initial failure repr included unrelated environment values, which were redacted; the gate now records only campaign markers.

Repair gates: `test_fixture_identifiers_mount_and_live_oracle_answers`, `test_oracle_sets_restores_layer_positions_and_records_wrapped_ids`, `test_probe_row_question_expected_and_worker_uses_effective_question` (`tests/test_grm_c7_r2.py:22,53,78`). The fake session mounts the first actual fixture source and answers Basalt-811 through both memory and live-source oracle; all 60 query/source pairs also pass the actual identifier-binding predicate. This does not certify alias/multi-source/folded model answers. Harness RED: **3 failed** in `cpu_harness_red.log`; final suite GREEN.

Launch gate: `test_leased_child_keeps_r2_after_c2_environment_filter` (`tests/test_grm_c7_r2_launch.py:17`) executes the real controller with fake lease/child boundaries and the real C2 environment builder. Additional registration gates reject rehashed chain/scope/budget/fixture/question/core/source forgeries and stale checkpoints (`tests/test_grm_c7_r2_registration.py`, `tests/test_grm_c7_r2_launch.py:51`).

The old small oracle double in `tests/test_grm_c7.py:147` gained the required layer/template interface and stronger receipt assertions; original assertions remain. Historical lead-2 diagnosis tests intentionally pin the old failures and were preserved, not relabeled as repair gates.

Registration chain:

- Base r2: `artifacts/grm_c7/r2/amendment.json` — `6b0d07b5c979ba30d8315a7ffe474a134bcb9fc52602a10518a42dff09e00048`.
- **Effective final r2 amendment:** `artifacts/grm_c7/r2/launch_amendment/amendment.json` — **`cb20ba62d3fa99ebe5b6dde12cacefbab0126098476f6b4601e3584fb6ca3712`**. This binds the launch correction and final command/CPU-runner bytes while inheriting the base core SHA and protocol. Neither amendment was rewritten.

Same frozen fixture/manifest, 300 turns, 60 probes, source/oracle records, distances/classes, restart turns, pressure protocol, C2 flags and **39 arm-A cells**. Same 39 prospective B cells remain NON_FIT. Historical cost proxy remains A 6740.77236 s / 1.87244 GPU-h, B another 6740.77236 s; no post-fix timing measurement is claimed. r2 hard cap **7200 s (2 GPU-h)**; prior r1 1.9 GPU-h per lead, total C7 **3.9 GPU-h**. Worker 280 s / lease 285 s / outer 590 s / foreground cooldown 30 s, no retries after started/RED/NON_FIT cells.

**Exact lead command (not executed here):**

```bash
cd /mnt/ForgeRealm/wt/grm-c7
bash artifacts/grm_c7/r2/lead_commands_r2.txt
```

That file contains the exact 39 sequential `run_cell A-...` invocations, exports `GRM_C7_REVISION=r2`, verifies the final SHA chain and CPU/mutation receipts, enforces no-retry/resume bindings, then runs `--summary A`. `bash -n` and the exact CPU preflight passed. Its SHA-256 is `04922d2e3700b896fb51ea4c33d17e6effe9d5ac9881dba9f3acfdbbb44b32b7`. No GPU cell was launched.

CPU reproduction command (receipts are exclusive-create; rerunning in place is intentionally refused once receipts exist):

```bash
GRM_C7_REVISION=r2 CUDA_VISIBLE_DEVICES='' python -m scripts.grm_c7_r2_cpu
```

## 3. Prior art; deviations; RED; process safety; model id and effort

### Prior art

**Local source verified:** GRM contributors (2026), EB1 `harmony_turn`/`HARMONY_STOPS` and arena early-stop contract; existing consolidation prompts/coverage/QC; C7 lead-2 numerical doubles, admission counterfactual, live-source oracle; C7/C2 SHA-bound amendment/checkpoint and lease protocols; HOUSE_RULES mutation discipline. Borrowed those contracts and test mechanisms. This work applies them to folds and repairs C7 caller/receipt/child-environment plumbing. The punctuation rule and exact combined regression stimuli are local additions: **no prior art known to me** for those exact rules/combinations. No new generation, routing, cryptographic or memory algorithm is claimed. Matching annotations are at code sites and in `LEDGER.md`; no external literature verification was needed for these locally inspected antecedents.

**Deviations:** The core edit is in the actual generation owner `graft_arena.py`, not its unchanged repository call site. The final launch boundary required a separate immutable amendment after the initial GREEN baseline; original registration/receipts were retained. The r2 fixture file is unchanged, while prompt instruction bytes are explicitly amended in an auxiliary manifest. Author CPU validation is not a blind review; no verifier was spawned under the no-subagent order.

**RED / not claimed fixed:** GPT-OSS's UNKNOWN replies are not causally explained by fake-model tests. GPT-OSS post-fix digest fluency, >=0.70 fold acceptance, attribution, aliases, paging/restarts and full product-promise acceptance remain unmeasured until lead r2. Optional S2 salience raw generation remains a separate finding. Historical GPU E4 coverage/quality gates were not run. Final arm-A summary is NOT_RUN, never PASS.

**Process safety:** no GPU work, git command, subagent, shell-background job, service mutation or process kill. Only CPU test subprocesses and fake lease/launch boundaries were used. Direct metadata read confirmed `refs/heads/grm-c7`; no commit was created. All **10,833 protected r1 files** match their pre-edit SHA256s (`r1_integrity_after.json`); r1's 39 cell receipts remain intact. Work/order registrations are unchanged; execution additions are in separate files.

**Model:** campaign `openai/gpt-oss-20b`, revision `6cee5e81ee83917806bbde320786a8fb61efebee`, never loaded. Author system identity GPT-6; exact deployment/API model ID is not exposed. Effort **high requested by order**; effective deployment setting is not independently exposed.
