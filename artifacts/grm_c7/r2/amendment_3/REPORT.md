# GRM-C7 amendment 3 — attribute repaired; continuation BLOCKED_CORE

The independent optional-attribute harness repair passes CPU gates. The alias failure reproduces in **core**, triggering the order's STOP rail. No GPU continuation is armed. The real-model oracle `unknown` cause is **NOT CLAIMED FIXED**.

## Attribute repair and CPU evidence

`scripts/grm_c7_run.py:137–166` follows the real optional `live_shift` contract: snapshot using `getattr`, dynamically set the arena shift, and restore either the prior value or attribute absence. Local reference: `core/gpt_oss20b_tc.py:632–695` (constructor does not create `live_shift`; attention reads it with `getattr(..., None)` and falls back to `graft_seats`), `core/graft_arena.py:604–615` (`_capture_geometry`) and `:3367–3368` (serving setter). No model position algorithm changed.

`scripts/grm_c7_diagnose.py:65–96` now uses the real `GptOssAttentionTC` metadata-only constructor for the fake attention object; no numerical forward, weights or GPU allocation is involved. `tests/test_grm_c7_amendment3.py:25`, **test_fake_attention_surface_matches_real_constructor**, enumerates both `vars` and `dir` from a separately constructed real object and asserts exact equality, including absent `live_shift`. **test_real_attention_without_live_shift_oracle_restores_absence** exercises the real object through the oracle. The exception pin covers absent, None and integer prior shifts. The existing r2 position/cache restoration gates also pass.

Author CPU evidence, not blind review or real-model quality validation: `cpu_red.log`: **3 failed, 5 passed**, reproducing verbatim `AttributeError: 'GptOssAttentionTC' object has no attribute 'live_shift'`. `cpu_green.log`: **62 passed** across amendment-3, r2 harness, FIX-3 and existing fold gates. Baseline command:

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_grm_c7_amendment3.py tests/test_grm_c7_r2.py tests/test_grm_scout_fix3.py tests/test_grm_fold_recovered_guard.py tests/test_grm_s4_fold_order.py
```

`cpu_mutations.json`: direct-read, leaked-shift and fake-precreates-shift mutants **3/3 killed**, zero errors, kill rate 1.0 against registered 0.80; source copies only, live files unchanged. Reproducer: `python -m scripts.grm_c7_amendment3_mutations` (writes immutable receipts; do not rerun into occupied output paths).

## Oracle diagnosis — source omission refuted, model cause unresolved

Original receipt `artifacts/grm_c7/r2/cells/A-009-016/probes.jsonl:1`, probe `c7_fresh_0_d005`, has oracle answer **`unknown`**, memory answer **`Basalt-811`**, and this exact Harmony-wrapped prompt:

```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 6] The current C7-Fresh-0 value is Basalt-811.
Use only those records.
What is the current C7-Fresh-0 value? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

`diagnosis_receipt.json` quotes all eight rows and records an offline decode using the registered model's local tokenizer.json. All eight recorded token-ID arrays decode exactly to their wrapped prompts, contain every listed source record, and equal offline re-encoding. `_attempt` passes `_format_step_prompt(user_text)` to `_forward` (`core/graft_arena.py:4377`); CPU real-attempt tests observe that complete input at the model seam. The sources are supplied in live prefill input, not mounted grafts. This proves input presence; the old receipt is prepared before `_attempt` and does not expose real per-layer attention or cache contents. It cannot establish effective numerical visibility or the cause of the generated `unknown`.

The lead premise "every oracle answer is still ... unknown" is contradicted by completed cell 3: `A-017-023/probes.jsonl:2` answers **`Onyx-911`**, line 3 **`Onyx-912`**. Six of eight rows say `unknown`; two of those six are unanswerable controls. The fixed missing-attribute read did not cause those completed calls to lose their source text: it had succeeded there. No speculative prompt rewrite or core/model patch is justified. **Ruling: crash = harness; unknown-answer causality = unresolved, not demonstrated harness or core.** The scorer remains its registered case-sensitive exact grammar, including lowercase `unknown` versus expected `UNKNOWN`; no score normalization was weakened.

## Alias diagnosis — CORE STOP

The alias is deposited as scanner-visible text. `A-009-016/checkpoint/repository/manifest.json`, node 8 (49 tokens), contains exactly:

```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>C7-Signal-0 is an alias for C7-AliasBase-0.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

`A-009-016/probes.jsonl:4` serves **`Not in memory: no stored record matches c7-signal-0.`** with **0 seats**. Its receipt quotes `excluded_live_ids: [8, 9]`, `recency_mounted_ids: [8, 9]`, and `identified_candidates: []`. The analogous Signal-1 row is line 5. These are recency **nominees**, not actual mounts; the abstention occurs first.

**test_alias_recency_exclusion_reproduces_in_core_and_ladder** deposits fixture turns 1–10 as Harmony complete turns, uses two recency nominees, verifies node 8 binds the alias, and reproduces the refusal in both real `ArenaCache.step` and `_probe_ladder_chat`, with zero model calls. A read-only admission counterfactual without exclusion finds `identified_candidates == [8]`. The CPU replay uses Python routing and synthetic numerical payloads, so it establishes the shared structural failure, not native ranking or model answer quality.

Core code: `core/graft_arena.py:3372–3382` excludes recency nominees from admission; `:3412–3429` abstains before any mount. The production ladder repeats it at `scripts/grm_e2e_session.py:1198–1256`. Core's point-lookup recency rule (`eb1_charge_recency`, `:3030`) also explicitly excludes those seats. This is a core policy interaction, not missing C7 alias deposition. **STOP**, as amendment 3 requires. No recency knob, fixture text, admission threshold or core code was changed.

Registered successor for the lead: reconcile core identifier eligibility / no-stored-record abstention with excluded recency binders, preserving fit and point-lookup semantics. Minimal condition to resolve: an extant excluded binder must not be reported as no stored record. Exact policy and any paired core/ladder change require the operator's decision; not implemented here. Reader success across alias-to-base relations remains unmeasured.

## Continuation disposition and validity

CPU registration (written before tests/repairs): `registration.json`, SHA256 `1ad182e2ce24e50f87559b31ad0743ee8828483a2be8162d5f40c59bfa91a1bf`.

**Blocked disposition**, not an enabled continuation registration: `continuation_registration.json`, SHA256 **`a4422e23c93f92d7757e98c29787d7ff5a6e54fab8f02a2b549efd3b0948c577`**. It binds the final harness/test sources, diagnosis, CPU receipts, core hashes and predecessor. Requested next cell remains **A-024-031**. The core STOP prevents registering an executable resume or importing an old checkpoint under a new binding.

Cells A-001-008, A-009-016, A-017-023 retain their raw memory/residency/fold/checkpoint evidence: the repair changes only the formerly absent-attribute branch and the CPU fake, and each completed oracle call necessarily passed the old attribute read. Existing-attribute execution, prompt bytes, injection, shift value, scorer and deposit path are unchanged. This is a source-path validity ruling, not a GPU treatment-effect result. All pre-amendment oracle rows are quarantined as **NOT_MEASURED** per the lead's direction, including the two correction successes. `summary_A_quarantined.json` enforces that analysis disposition; no old receipt is edited. An oracle-relative acceptance comparison and product-promise verdict are **NOT_MEASURED**. Oracle quarantine is conservative; missing source or harness causation of the six unknown outputs was not established.

A-024-031 remains FAILED in place. No retry, overwrite, checkpoint rebinding, or directory migration occurred. Its full 280-second charge remains included: existing charged total **523.163476883434 seconds** of the unchanged 7200-second r2 cap. Any future authorized continuation needs a separate attempt namespace, compatibility registration, all historical charges, and the unresolved oracle validity decision. The legacy launch verifier correctly rejects this unarmed harness delta with `ValueError: INPUT_SHA_MISMATCH: scripts/grm_c7_run.py`.

Exact lead command (diagnostic-only; deliberately exits 1 before any GPU/lease/worker call):

```bash
bash /mnt/ForgeRealm/wt/grm-c7/artifacts/grm_c7/r2/lead_commands_r2_resume.txt
```

## Prior art

Verified local system/source: GRM contributors (2026), `GptOssAttentionTC` optional shift and `ArenaCache._capture_geometry`/`step`; borrowed optional reads and dynamic setters. Our repair preserves missing-attribute state explicitly and replaces the fake's invented initialized field with the real metadata constructor. GRM C7/EB1 CPU serving-path replays and HOUSE_RULES section 8 (2026) supply receipt, counterfactual and source-copy mutation practice. The diagnostic alias counterfactual changes no shipped policy. The successor names an inconsistency in those existing contracts, not a new admission algorithm. **No prior art known to me for this exact repair/test composition.** No external literature claim or search was needed; no new memory algorithm or optimization was implemented.

## Deviations, RED, process safety, model and effort

The core STOP supersedes the requested resume deliverable. Mission 1 is repaired on CPU; mission 2 reaches a proven core finding and an explicitly unresolved oracle cause; mission 3 is blocked. This is not a completed E2E recovery. No core fixes or executable continuation were smuggled into the harness. The typo in the repair's prior-art helper name was corrected after the passing baseline; that edit is comment-only. The final mutation suite ran after it.

RED: real-model crash repair not GPU-validated; unknown-answer cause unresolved; core alias admission interaction not fixed; oracle upper-bound and product acceptance not measured; continuation not armed; blind lead review outstanding. Old registration/launch tests are not green claims under the intentionally unregistered new source hashes. Independent registration remains immutable.

No git, subagents, GPU/model-weight load, external network, services, background jobs, sleeps/waits, signals or process kills were used. All owned CPU test/mutation processes ran sequentially in the foreground and exited. Async tool yields were polled only to collect those foreground completions. Core/order/fixture/old r2 receipts and payloads are SHA-checked in `final_integrity.json`; no existing registration was edited. Zero GPU seconds used by this seat.

Evaluated model identity from registration: **openai/gpt-oss-20b**, revision **6cee5e81ee83917806bbde320786a8fb61efebee**, TensorCUDA attention implementation. Agent: **GPT-6**, exact deployment ID not exposed; **high** effort requested and used for reasoning, runtime setting not independently exposed.
