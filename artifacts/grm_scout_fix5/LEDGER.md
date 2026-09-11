# GRM-SCOUT-FIX-5 ledger

- Registration created before gates; immutable order read and unchanged. Source snapshots saved under before/. Working directory and ref verified by filesystem reads; no git command.
- Prior art: local GRM source scaffold/fact-set/QC/coverage and C7 numerical doubles/leased SHA receipts (GRM contributors, 2026), source verified. Reused contracts; explicit source list and 24*N budget are this order's prescribed extension. No prior art known to me for this exact combination.
- Census: 11 continuation folds plus preserved turns 8 and 17 = 13. Historical prompts misclassify absent kind; replay treats recorded null as absent/turn, verified by baseline primers.
- Gates: 13 source-list replays, 13 exhaustion pins, zero-fact byte controls, FIX3 semantics, budget boundaries; then two registered mutations. GPU lead-only, 13 leases at 80s, no retry, max 1040s.

- CPU RED: cpu_red.log records 28 failed, 34 passed (13 enumeration, 13 budget, 2 boundary failures); all 13 result receipts in red_cases/. Zero-fact four-frame bytes frozen before core edits.
- Core change: source-bearing spans appended to all three generated fold prompts; default budget max(CONSOLIDATE_NGEN,24*len(need)); configured floor remains 120. Zero-fact and explicit ngen preserved; QC/coverage untouched.
- CPU GREEN: cpu_green.log, 62 passed. All 13 fake-model folds accepted at 1.0 lexical coverage; green_cases/ preserves prompts and digests. This tests prompt/budget plumbing only, not relation correctness of a real model.
- FIX3 test adaptation: two expected prompt expressions now call the prompt builder because FIX5 intentionally changes fact-bearing prompts. Wrapper, stops, exact generation lengths, legacy formatting and QC/coverage assertions retained; 25 cases pass.
- GPU harness uses exact saved packed payloads, checkpoint routing keys and original source order; maps source IDs locally. Runs one fresh foreground process and lease per fold, 80s lease / 75s worker. No subprocess timeout or kill. Alarm deadlines are cooperative; uninterruptible native calls cannot be hard-bounded without violating never-kill. GPU NOT_RUN.

- GPU registration frozen as gpu_registration.json; SHA-256 631b6ad9891149ec6efdbfbab4409afdf5a4e34c1e29cce326fd76cbf755cb46. lead_commands_fix5.txt --check passes all 13 bindings without model/GPU imports. Core files, source payloads, historical receipts, index/manifest and model metadata/tokenizer are bound.
- Separate immutable validation_amendment.json registered runner/AST checks before their execution. runner_green.log: 45 passed in 10.52s; all 13 packed source sets restored on CPU with actual tokenizer and dequantization, no model forward.
- Registered mutation results: remove_enumeration -> 13 failures; remove_scaling -> 13 failures. 2/2 killed (1.0 >= .80); live source was never mutated. Logs and summary.json retained.
- Report written with exact code lines/test names, registrations/lead command, prior art, deviations and RED limits. No GPU directory created. All original inputs and both immutable registrations rechecked for final_integrity.json. Agent GPT-6 family / exact backend ID not exposed, high effort requested.
