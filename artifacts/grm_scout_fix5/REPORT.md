# GRM-SCOUT-FIX-5 — CPU GREEN; GPU contrast registered, NOT_RUN

**Unit/suite evidence:** 13/13 saved r2 source cases now pass the prompt-reading fake-model fold at **1.0 lexical coverage**, versus 0/13 before the fix. **62 CPU pin cases pass**, including all 25 FIX-3 cases; **45 runner/recovery/fold-order cases pass**. Both registered mutations are killed. **Real-model retention is NOT_MEASURED / not claimed fixed.**

## 1. Fix file/lines; test names with RED/GREEN evidence; budget formula

Only two core methods change, both in `core/graft_arena.py`:

- **1929–1955, `_consolidation_prompts`:** keep the existing three prompts and primers; append an explicit “Keep every one of these N facts” instruction with numbered source-bearing spans. Remove Harmony transport markers and legacy role labels from displayed spans. Preserve the source relationships, correction text, order and identifiers. Require one prose archive block, no source/list copy, and no invented times. The list is input guidance; output QC still requires the existing prose format.
- **2179–2184, `consolidate`:** default allowance is **`max(CONSOLIDATE_NGEN, 24 * len(need))`**, with the unchanged configured floor **120**, i.e. **`max(120, 24*N)`** in this campaign. Explicit `ngen` remains authoritative, including callers intentionally requesting less than 120. The zero-fact path retains its original configured allowance.

`N` is the existing unique lexical fact-token count, **not** a count of semantic propositions or list rows. Source spans preserve context rather than listing isolated lowercased tokens. No new fact extraction/coverage algorithm is introduced. Coverage **0.70**, fact extraction, QC, source retirement, payload injection and Harmony/stop handling remain structurally unchanged. [Core diff](core_delta.patch); [AST scope pin](../../tests/test_grm_scout_fix5_runner.py).

| Pin | Before | After | Evidence |
| --- | --- | --- | --- |
| `test_r2_enumerated_facts_fold_cpu`, 13 turns | 13 fail; fake sees no source list, coverage 0, all rejected | 13 pass; all facts visible, coverage 1.0, all accepted | [RED log](cpu_red.log), [GREEN log](cpu_green.log), `red_cases/`, `green_cases/` |
| `test_r2_default_budget_exhaustion`, same 13 turns | 13 fail: each attempt only 120 tokens | 13 pass: every attempt gets its N-sized allowance | same logs |
| `test_default_budget_boundary_and_explicit_override`, N=0,1,5,6,10 | 2 default cases fail (N=6,10); other 8 pass | 10 pass; explicit 19 always retained | same logs |
| `test_zero_fact_fold_byte_identical` | passes; bytes frozen before core edit | passes against original bytes | [Frozen control](zero_fact_before.json) |
| FIX-3 suite, 25 cases | 25 pass | 25 pass | same logs; [test-only expected-prompt delta](fix3_test_delta.patch) |

Zero-fact control includes Harmony/legacy templates × scaffold off/on. It compares complete model-call records, result/digest bytes, QC/coverage receipts and source/digest lifecycle fields. FIX-3's two exact expected-prompt expressions use the prompt builder now, because fact-bearing prompt text intentionally changes; wrapper equality, stop counts, explicit-budget, punctuation QC and both sides of the .70 rule are retained. Test names: `test_fold_harmony_wrapper_and_stop`, `test_fold_explicit_budget_and_legacy_prompt`, `test_digest_qc_rejects_punctuation_collapse`, `test_digest_qc_accepts_ordinary_punctuation`, `test_degenerate_qc_precedes_coverage_even_with_list_relaxation`, `test_fold_coverage_bar_remains_point_seven`.

Verbatim suite totals: **`28 failed, 34 passed, 2 warnings in 5.70s`** before; **`62 passed, 2 warnings in 5.70s`** after. The two warnings are existing SWIG type deprecations. All test files are CPU-only. The fake reader extracts identifiers solely from the live numbered source list and emits them in a single prose block. Its tokenizer is a reversible CPU double; its generation lengths do not establish GPT-OSS token geometry. Its lexical success does **not** prove preservation of relations by a model.

| Fold turn | N | New default tokens | CPU before coverage | CPU after coverage |
| --- | ---: | ---: | ---: | ---: |
| 8 | 10 | 240 | 0.0 | 1.0 |
| 17 | 10 | 240 | 0.0 | 1.0 |
| 26 | 8 | 192 | 0.0 | 1.0 |
| 55 | 8 | 192 | 0.0 | 1.0 |
| 74 | 8 | 192 | 0.0 | 1.0 |
| 85 | 9 | 216 | 0.0 | 1.0 |
| 91 | 12 | 288 | 0.0 | 1.0 |
| 95 | 9 | 216 | 0.0 | 1.0 |
| 99 | 8 | 192 | 0.0 | 1.0 |
| 103 | 8 | 192 | 0.0 | 1.0 |
| 107 | 8 | 192 | 0.0 | 1.0 |
| 111 | 8 | 192 | 0.0 | 1.0 |
| 115 | 8 | 192 | 0.0 | 1.0 |

[Runner/regression log](runner_green.log): **`45 passed, 2 warnings in 10.52s`**, covering 13 source restores using actual CPU unpacked packed K/V and the pinned local tokenizer, SHA tamper rejection, direct-worker rejection, foreground controller/no-retry accounting, unchanged-core AST scope, and existing recovered-fold/fold-order tests. This tests restoring operator inputs, not numerical GPU attention. [Mutation receipts](summary.json): remove enumeration → **13 failed**; remove scaling → **13 failed**; kill rate **2/2 = 1.0**, above the preregistered .80 bar. Mutations are applied to methods in disposable Python processes, never to the live core file.

CPU repeat command (no receipt overwrite):

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_grm_scout_fix5.py tests/test_grm_scout_fix3.py tests/test_grm_scout_fix5_runner.py tests/test_grm_fold_recovered_guard.py tests/test_grm_s4_fold_order.py
```

## 2. GPU contrast registration path + sha; exact lead command

**Registration:** `artifacts/grm_scout_fix5/gpu_registration.json` ([file](gpu_registration.json)).

**SHA-256:** `631b6ad9891149ec6efdbfbab4409afdf5a4e34c1e29cce326fd76cbf755cb46`

Exact lead command:

```bash
bash /mnt/ForgeRealm/wt/grm-c7/artifacts/grm_scout_fix5/lead_commands_fix5.txt
```

CPU-only preflight, already run successfully ([receipt](lead_check.log)):

```bash
bash /mnt/ForgeRealm/wt/grm-c7/artifacts/grm_scout_fix5/lead_commands_fix5.txt --check
```

The launcher calls `python -m scripts.grm_scout_fix5 --check` then `--run`. [Runner](../../scripts/grm_scout_fix5.py) hashes the original start/result/payload files, per-cell checkpoint source metadata and routing keys, current core, runner/loader/lease source and local model metadata/tokenizer. Old campaign registrations remain intact; this is an isolated fold-operator replay, not a campaign continuation under stale source hashes.

Each fold loads a fresh model under **one existing foreground GPU lease**, restores packed K/V through the production unpacker and checkpoint centroids through the production index unpacker, preserves source order and records original→local node IDs. It invokes real `consolidate` with the new default prompt/budget, source K/V injection and unchanged .70/QC gate. It does not re-harvest source text or rerun questions. Post-cell checkpoint keys provide unchanged source centroids; text/kind/ntok are checked against the bound fold start. Recorded null kinds are restored as absent/default `turn`, consistent with actual baseline primers rather than the old observer's incorrect ERA labels.

**Reservation:** 13 × 80-second leases = **1,040 seconds = 0.288889 GPU-h**, below 0.3. Each worker gets a 75-second cooperative deadline plus cleanup allowance. Zero lock wait. No retry/overwrite of any started namespace. A coverage rejection is a completed measured result, so remaining folds continue; a binding failure, lease failure, timeout or worker failure stops and reports incomplete. Lease alarms raise in the owned process; no process is killed. An uninterruptible native call cannot be forcibly bounded under the never-kill constraint; this is an explicit operational limitation, not a claimed hard preemption guarantee. Fit of all 13 real model runs within these rails remains unmeasured.

Planned receipts live at `artifacts/grm_scout_fix5/gpu/TTT/`: bound reservation, `prompt_N.json`, per-token `tokens.jsonl`, `receipt.json` (coverage, accepted, digest text, QC attempts, raw full decode, input/output IDs, finish category and baseline), controller and worker log. Final `gpu/summary.json` includes all 13 coverage/accepted/digest rows and reserved charging, or a RED_INCOMPLETE result. **The gpu directory was not created by this seat.**

## Prior art

**Verified local systems, GRM contributors (2026):** `_source_scaffold` / `_consolidation_prompts` supply the source-text scaffolding concept; `_fact_set` / `_coverage` / digest QC supply the unchanged fidelity contract; FIX-3/EB1 supply Harmony wrapping and stops; C7 supplies CPU numerical doubles, start/payload receipts and SHA bindings; GQA persistence supplies packed payload/index restoration; C2 supplies the registered environment and model loader; CMC1 supplies foreground flock leases. Taken: these existing interfaces, policies and verification techniques. Ours: the order-prescribed explicit fact-bearing source list, per-fact default budget, bound fold replay adapter, and test/telemetry receipts. **No prior art known to me for this exact prompt/budget/replay composition.** No external-paper or novelty claim; no external literature lookup was needed for these locally verified attributions. Same attribution appears at code sites and in the ledger.

## Deviations; RED; process safety; model id and effort

- **Census clarification:** the order's continuation glob contains 11 folds. Turns 8 and 17 are in `r2/cells`; including both produces the required 13, following amendment 4. No historical file is rewritten.
- **FIX-3 pin adaptation:** two expected prompt constructions changed as described above; its 25 behavioral tests still pass. No acceptance assertion was weakened.
- **Contrast limit:** historical baseline versus treatment is not interleaved A/B. Durable quantized payload restoration is the specified replay input; it does not certify byte identity with the historical in-memory device tensors. This run will change both prompt and default budget as ordered; it cannot isolate their separate model effects.
- **RED / not claimed fixed:** real-model fold retention, relation fidelity, prompt-versus-K/V causation and the original campaign's retrieval/oracle/E2E deficits remain unmeasured or unresolved. Model weights are identified by the original pinned local snapshot/revision; config/tokenizer/index are hashed here, but multi-GB weight files were not independently rehashed. No GPU quality, residency, restart or product certification claim is made. Author tests are not a blind independent audit.
- **Process safety:** only foreground CPU work and local reads/edits; no GPU execution, git commands, subagents, background waits, process kills, live-service changes or external messages. Worktree and branch ref were verified by filesystem reads. Source/order/registration preservation is checked in `final_integrity.json`; file receipts are listed in `SHA256SUMS`.
- **Model planned for contrast:** `openai/gpt-oss-20b`, revision `6cee5e81ee83917806bbde320786a8fb61efebee`, existing `tensor_cuda GptOss20B_TC` / `resident_packed_mxfp4` path, Harmony **Reasoning: low / Valid channel: final**. CPU model is the deterministic `EnumeratedModel` double, with no learned weights.
- **Agent:** Codex, GPT-6 family per session instructions; exact backend model ID is not exposed. Requested analysis effort: **high**; backend effort metadata is not independently exposed.

Immutable [execution registration](registration.json), separate [validation amendment](validation_amendment.json), append-only [ledger](LEDGER.md), and [machine-readable summary](summary.json) retain the receipts and boundaries.
