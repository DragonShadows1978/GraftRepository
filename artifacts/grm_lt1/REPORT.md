# GRM-LT1 — BLOCKED / RED / NON_FIT

No GPU was run. This is a frozen fixture and a negative CPU author baseline, **not** a successful long-conversation proof of concept. Registration remains unchanged.

## 1. Fixture manifest sha; turn-mix counts; gate test names and results.

- Fixture: `fixtures/lt1/dialogue.json`; readable dialogue: `fixtures/lt1/turn_plan.md`.
- Manifest: `fixtures/lt1/manifest.json`, SHA-256 `992acacf307fa5c4e7b7a3c6c5ed8adb1196c13063a5a77caec26630a218d1bb`.
- Dialogue SHA-256 `57aac20ddc8a3379fa03ded29dc5609a1f4cdc2f973ec52241853a5c2e63bb83`.
- Exactly 200 turns: **60 fact, 15 correction, 10 alias, 79 ordinary, 35 recall, 1 recap**. Seven recalls at each distance 10/25/50/100/150; each stratum contains 3 fresh, 2 corrected, 2 alias. Latest corrections and aliases determine distance; every required source is checked independently.
- Frozen scripted user/assistant replay; only recall/recap outputs are designated for model generation. No fabricated identifiers, hidden memory correction commands, repeated answer deposits, or Frontier modifications. Facts are invented for this expansion, not imported from public lore.
- UNIT/SUITE evidence: `cpu_baseline.log`, **7 passed / 3 failed, 36.06 s**; `cpu_receipt.json` binds its hash and registration. Baseline was registered first. No rerun, repair, threshold change or mutation after RED.

| Registered test | Result |
|---|---|
| `test_registration_and_frozen_fixture` | PASS: all 35 plan-source ages, production EB1 recency nominations, source text/expected-current checks, mix and distances |
| `test_recency_gate_rejects_recent_source` | PASS |
| `test_instruction_collision_and_missing_oracle_are_rejected` | PASS |
| `test_value_span_and_disjoint_error_categories` | PASS |
| `test_fake_attention_matches_actual_constructor` | PASS; real GptOssAttentionTC constructor surface, absent live_shift |
| `test_visible_reader_requires_sources_and_current_revision` | PASS |
| `test_full_fixture_fake_mounts_answers_and_restarts[A]` | **FAIL: 35/35 memory abstentions, zero mounted sources** |
| `test_full_fixture_fake_mounts_answers_and_restarts[B]` | **FAIL: 35/35 memory abstentions, zero mounted sources** |
| `test_c7_checkpoint_roundtrip_and_corruption` | **FAIL: KeyError: 'process_id'**, author test setup omitted required field |
| `test_empty_summary_and_budget_fail_closed` | PASS |

The full-session tests reached next_turn=201 across three distinct processes per arm (A: [29, 54, 79]; B: [104, 129, 154]). Before reaching their failing memory assertion they checked all 35 actual oracle prompts for source inclusion and obtained 35/35 exact spans from the visible-prose CPU fake. Actual source IDs were disjoint from recency nominees on all calls. `fake_A_receipt.json` and `fake_B_receipt.json` preserve route fields, prompts and raw answers. The fake emits trailing ellipses on some oracle outputs; those remain in receipts and pass only the registered value-span metric. This does not establish model fluency or GPT-OSS quality.

### CPU diagnosis (core boundary; not claimed fixed)

First source, turn 12: `Promenade's lamp spacing will be 9 voxels.`

First recall, turn 22: `What did we settle on for Promenade's lamp spacing?`

Actual response, both arms:

> Not in memory: no stored record matches settle, promenade, lamp, spacing.

Receipt: source node `[11]`, recency nodes `[19, 20]`, source ranked first, `rare_tokens=[]`, `admission_identified_candidates=[]`, `abstain_reason=identifier_unbound`, mounts `[]`, and zero numerical memory-reader calls. This is not a recurrence of the recent-source collision. In `core/grm_admission.py:113`, no-rare-token binding requires `['current', *ordered_query_tokens, 'value']` contiguously in the candidate. The natural source does not contain `current settle promenade lamp spacing value`. `_rare_tokens` at `core/graft_arena.py:2024` recognizes digits or all-uppercase words, not ordinary title-case names. The unchanged production predicate explains this fixture's refusal before reader execution. Core policy changes require a separate order; no synthetic-phrase rewrite or admission override was applied.

Scoring distinguishes **wrong value: 0**, **abstention: 35**, **policy-style refusal: 0** per arm. All 35 abstentions also use the admission refusal text `Not in memory: ...`; that overlapping text style is separately counted in `cpu_diagnosis.json`. Do not conflate policy refusal, memory abstention, and incorrect values.

| Distance | n per arm | Fake A memory exact | Fake B memory exact | Fake A oracle exact | Fake B oracle exact |
|---|---:|---:|---:|---:|---:|
| 10 | 7 | 0/7 | 0/7 | 7/7 | 7/7 |
| 25 | 7 | 0/7 | 0/7 | 7/7 | 7/7 |
| 50 | 7 | 0/7 | 0/7 | 7/7 | 7/7 |
| 100 | 7 | 0/7 | 0/7 | 7/7 | 7/7 |
| 150 | 7 | 0/7 | 0/7 | 7/7 | 7/7 |

**GPU both-sides comparison: NOT_MEASURED. Recap A/B: NOT_RUN, not 0/5.** `summary_A.json`, `summary_B.json`, and `comparison.json` preserve empty denominators and null quality/residency/restart fields. The fake A/B labels exercise widths 96/256; they do not simulate real capture geometry, pin or near-live treatment effects.

## 2. Registration path + sha; cell list and projection; predictions.

- Immutable `artifacts/grm_lt1/registration.json`, SHA-256 `e1913b144087ea79e6eea4a0b8eceae0bedd2a4853cd931db5f24c795488300c`; adjacent `.sha256`.
- **52 prospective cells, 26 per arm**, detailed in `cell_list.tsv`, with parent dependencies in registration. Prefix each range below with `A-` or `B-`:

001–008, 009–016, 017–024, 025–032, 033–040, 041–048, 049–056, 057–064, 065–070, 071–078, 079–086, 087–094, 095–102, 103–110, 111–118, 119–126, 127–134, 135–140, 141–148, 149–156, 157–164, 165–172, 173–180, 181–188, 189–196, 197–200.

- Boundaries: at most 8 turns/cell, splits after 70 and 140. Registered worker 280 s ≤285 s, lease 285 s, outer 590 s, foreground cooldown 30 s. Every prospective cell would reload; designated restarts add before/after fresh and corrected sentinel scoring.
- REASONING estimate: **4507.870518 s/arm, 9015.741037 s total = 2.504372510 GPU-h**, over the 9000 s cap by **15.741037 s**. Maximum cell estimate 255.030359 s. **NON_FIT on both arms; no trimming.** These are planning estimates, not measured GPU consumption.
- Derivation: historical EB1 mean-turn 8.343609 s, max-probe 7.369623 s and max-correction 30.732458 s, multiplied by 1.25; 60 s reload/IO per cell; 35 live oracles, a five-probe-equivalent 160-token recap allowance, and eight restart sentinel calls per arm. Cooldown adds 1560 s wall, outside GPU ownership. Natural prose, later persistence/folding, and B performance remain unmeasured. Copied timing evidence SHA is in registration; underlying source receipt exists and its SHA was verified as `ec626c53e5ba9ca5aac2f33a8084edafd2a584efed926eb4131fe7b0171358c5`.
- Preflight free space **325428609024 bytes**, above 20,000,000,000. No model weights or GPU loaded.
- Registered predictions: A fresh 80–100%, corrected 50–80%, aliases 0–60%; B fresh 50–80%, corrected 20–60%, weaker aggregate long-distance performance. Metadata expected retained; answer retention uncertain. Oracle pooled fresh/corrected ≥80%; recap A 3/5, B 2/5. These pre-gate forecasts are not results. Acceptance retains A ≥80% fresh/corrected **at each distance**, full coverage, restart score/metadata retention, and ≤2×width+recency residency. Small strata mean all three fresh and both corrected must pass for the 80% criterion.

## 3. Exact lead commands; prior art; deviations; RED; process safety; model id and effort.

The following executable command was run and returned **2**, after writing both arm summaries and comparison:

```bash
bash /mnt/ForgeRealm/wt/grm-lt1/artifacts/grm_lt1/lead_commands.txt
```

Its final preflight reports `NON_FIT` and `CPU_GATES_NOT_GREEN`. `bash -n artifacts/grm_lt1/lead_commands.txt` passed. The command is safe to repeat for reporting; it never starts a GPU cell. There is no bypass or automatic rerun of a started cell.

The original CPU command (already executed; do not rerun into these immutable receipt filenames) was:

```bash
cd /mnt/ForgeRealm/wt/grm-lt1
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m scripts.grm_lt1_cpu
```

### Prior art

GRM contributors (2026), locally inspected C7/C2/EB1 and C7 amendment 3: SHA-bound registration/checkpoints, leased-cell cost model, actual recency nominee method, production ladder, isolated live-source oracle, and real-attention surface fake. Borrowed contracts and source unchanged; LT1 adds a natural-language fixture, cost/report adapter, and visible-prose numerical stimulus. C5 arm S span rule follows this order. NFKC comes from Unicode UAX15 (Unicode Consortium, 2001), **unverified — lead to check**, search `Unicode UAX15 normalization history`. No prior art known to me for this exact conversation composition. No new retrieval algorithm or optimization is claimed. Annotations are also at code sites and in `LEDGER.md`.

### Deviations and remaining RED

1. **NON_FIT stops GPU implementation/execution:** the cell schedule is prospective. `--resume` is a fail-closed entrypoint, **not a completed resumable leased GPU worker**. A future separately authorized registration must supply and test that launcher; inherited C7 subprocess timeouts can kill their own child and were not reused under LT1's stricter never-kill instruction.
2. **Required fake mounting/answer gate is RED** in both arms; immutable natural wording was retained. The product admission limitation is diagnosed, not fixed. No point recall reached the numerical memory reader, so neither width demonstrates a quality advantage.
3. **Checkpoint test is RED due to my omitted `process_id`.** Its corruption assertion did not execute. Required session binding is not claimed validated by this test. Source/test/registration remain unchanged after the rail; a separate amendment is required for repair.
4. The dialogue is scripted replay, not a freely generated assistant conversation or observed human usage. Ordinary turns revisit design themes. The 200th recap has frozen five decision targets but was not generated/scored by a real model or the CPU fake. Twenty-six-cell resumption, real restart retention, profile capture, residency, and model quality remain untested.
5. The author baseline failed; planned mutation testing did not run. No blind verification was launched, consistent with no subagents and lead-owned verification.

Process safety: all started CPU child processes returned; no GPU, git commands, subagents, background shell, service changes, or process kills. Read-only branch metadata says `grm-lt1`; product files and order were not edited. No external messages sent.

Target reader: **openai/gpt-oss-20b**, revision **6cee5e81ee83917806bbde320786a8fb61efebee**, registered C2 EB1 frame explicitly says **Reasoning: low**. Author: **GPT-6**, exact deployment model ID not exposed; **high requested** by order, inference setting not independently exposed. Author effort and target model prompt effort are distinct.
