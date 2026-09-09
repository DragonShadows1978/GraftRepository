# GRM-C7 lead amendment 2 — STOP / SCOUT-FIX finding

**Core fold prompt/stop handling triggers the registered stop. No core or C7
runner fixes, r2 campaign registration, or `lead_commands_r2.txt` were made.**
The original fixtures, registration chain, command file and 39 cell receipts
remain in place. Diagnostic code and receipts are additive.

Evidence classes: read-only census of existing lead-run E2E receipts; source
inspection; author CPU replay with a fake numerical model. The latter does
not validate GPT-OSS attention, positional numerics or answer quality.

The full census corrects the brief: 47/60 oracle replies are literally
`UNKNOWN`, not 60/60. There are three exact-correct answerable replies, all
`Onyx-911`. All 13 folds are rejected, but their best coverage ranges from
0.10 to 0.375, not only 0.0–0.1. Every one of 60 memory probes takes
`identifier_unbound` with zero mounts; all 300 post-turn seat counts are zero.
`receipt_audit.json`, `oracle_exceptions.json`, and
`summary_A_unmodified.json` preserve the full census. The unchanged summary
returns `"status": "FAIL"`, `"max_token_seats": 0`,
`"residency_bounded": true`.

## 1. Four diagnoses, rulings, fixes and gates

### Admission — HARNESS query construction defect, confirmed by treatment

Verbatim from `artifacts/grm_c7/cells/A-009-016/probes.jsonl:1`:

> Not in memory: no stored record matches c7-fresh-0, unknown.

The same row has `"abstain_identifier_tokens": ["c7-fresh-0", "unknown"]`,
`"admission_ranking_before_demotion": [5, 0, 1, 4, 3, 7]`,
`"admission_identified_candidates": []`, and `"mounted_ids": []`.

`scripts/grm_c7_register.py:108` appends the all-capitals fallback instruction
to every query. `core/graft_arena.py:2024–2035` classifies a word containing a
digit **or** an all-uppercase word of length at least three as rare.
`core/grm_admission.py:85–119` case-folds these tokens and requires **all** of
them in the candidate's own text (`return rare <= have`). A source containing
`C7-Fresh-0` without `UNKNOWN` therefore fails binding. The ladder returns the
fixed abstention before mounting at `scripts/grm_e2e_session.py:1207–1255`.
This is the existing frozen admission law applied to a C7 instruction token;
neither the ASCII hyphens nor entity case is the obstacle.

CPU replay: real `GraftRepository.add_document`, real routing/admission and
`_probe_ladder_chat`, real `ArenaCache._attempt` with numerical doubles:

| Query change, diagnostic only | Mounted IDs | Answer |
|---|---|---|
| None | `[]` | Exact abstention above |
| Only `UNKNOWN` → `unknown` | `[0]` | `Basalt-811` |
| Also `C7-Fresh-0` → `c7-fresh-0` | `[0]` | `Basalt-811` |
| Bare question without reply instruction | `[0]` | `Basalt-811` |

`binding_pairs.json` records all 60 original query/source pairs and their
60 lowercase-fallback controls: original binding fails everywhere; each
control binds at least one source. This does not establish multi-source alias
or folded-answer quality, nor does it authorize changing frozen probes.

EB1 and WC1 both have the same successful historical control:

> Recall probe. What is the current orion pin value? Reply with only the value.

> Auric-4-Alpha

Each seats `[0]`, uses `identifier_tokens: ["orion", "pin"]` and an empty
rare-token set. Exact paths, SHA256s and transcript/instrumentation line 6 are
in `receipt_audit.json.comparisons`: EB1 `grm_eb1/g5_run/lh-5/session`;
WC1 `grm_wc1_astra/width_96/census/run/arm1/e2e-2/session`, both under
`/mnt/ForgeRealm/GraftRepository/artifacts/`. The CPU orion control also seats
and answers. The frozen rule is operating as coded in both cases.

**Fix: not implemented because of the core stop.** A later authorized C7
amendment must keep fallback instructions from becoming required routing
identifiers without relabeling changed probe bytes as unchanged fixtures.

Gates: `test_identifier_unknown_suffix_blocks_real_repository_admission`
(`tests/test_grm_c7_lead_2_diagnosis.py:22`) and
`test_eb1_wc1_identifier_controls_and_all_c7_binding_pairs` (`:49`).

### Oracle — HARNESS setup/observation defects; GPU answer cause unresolved

Verbatim from the same first probe row: `"answer": "UNKNOWN"`, with prompt:

```text
Exact source records, in chronological order:
[turn 6] The current C7-Fresh-0 value is Basalt-811.
Use only those records.
What is the current C7-Fresh-0 value? Reply only with the answer; if unspecified, reply UNKNOWN.
```

The source is present. C7's `oracle()` (`scripts/grm_c7_run.py:98–116`) resets
the cache and calls `_attempt` directly with no mounts. That bypasses the
identifier-abstention gate. `_attempt` formats the entire source-bearing
string (`core/graft_arena.py:4348–4351`) through the same `harmony_turn`
configured by EB1 (`scripts/grm_e2e_session.py:86–111,2193–2195`), then forwards
it as live input. `mounted_ids: []` is required for this live-source oracle;
it is not proof that the source was absent.

The unchanged oracle returns `Basalt-811` in the CPU replay, and the fake
reader's recorded first forward exactly equals `harmony_turn(live_prompt,
None)`, including the source. No expected-answer field reaches the fake
reader. `oracle.json` retains the full model-boundary transcript. The prompt
wrapper matches EB1; the inner source-record block is C7-specific. RS3 C5
instead feeds the live-source prelude separately before its question.

Two harness defects are visible. First, the oracle directly invokes `_attempt`
without establishing/restoring each layer's `live_shift`. EB1's successful
ladder sets it at `scripts/grm_e2e_session.py:1449–1451`, but C7's poisoned
queries return before that line. Core consolidation sets it to `None`
(`core/graft_arena.py:2144–2146`), and `reset_live_cache()` does not set it
(`:331–339`). The CPU first oracle forward observes `live_shift: null`.
GPT-OSS uses `graft_seats` when the layer shift is absent
(`core/gpt_oss20b_tc.py:692–702`), so the direct call does not guarantee the
registered EB1 position setup. Second, C7's `prompt_token_ids` receipt encodes
the **unwrapped** prompt (`scripts/grm_c7_run.py:112`), not the actual Harmony
input. It cannot serve as an exact input-token receipt.

These omissions belong to the C7 caller. **It is not established that either
caused the real model's UNKNOWN replies.** Numerical position behavior and
causal treatment of the actual model were not tested. The all-UNKNOWN premise
is also false: `A-040-047/probes.jsonl:3` replies `Onyx-911` correctly, whereas
`A-017-023/probes.jsonl:2` replies `Onyx-511`. Full exceptions are retained.
There is no honest basis to certify this as a fully explained, harness-only
answer failure. Not claimed fixed.

Gate: `test_oracle_exact_source_reaches_real_attempt_harmony_live_input`
(`tests/test_grm_c7_lead_2_diagnosis.py:76`). It pins live-source delivery and
the missing setup; it does not reproduce GPT-OSS's failure numerically.

### Fold generation — CORE SCOUT-FIX finding; stop triggered

Verbatim candidate from
`artifacts/grm_c7/cells/A-001-008/folds/008-230997aa79504f2ebd570d5cbc000ee5/result.json`:

> For the archive: the ………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………

That attempt has `"coverage": 0.0`, `"qc": false`. Another attempt begins
`ARCHIVE NOTE — The conversation established that the  “` followed by a long
ellipsis run and has `"coverage": 0.0`, `"qc": true`. The third includes:

```text
We need to list every name, code, number, time. Provide references.<|end|><|start|>assistant<|channel|>analysis<|message|>
```

Full strings are preserved without truncation in the original candidate
JSONL, `receipt_audit.json.fold_quotes`, and `ellipsis_qc.json`.

The C7 observer delegates unchanged (`scripts/grm_c7_run.py:156–159`); the
forced path calls `_fold_once` without `ngen` (`:377–383`). The core repository
calls `arena.consolidate(idxs)` with no budget override
(`core/graft_repository.py:3261`). This selects **120 tokens**
(`core/graft_arena.py:1908,2138–2139`). C7's 32-token inference argument does
not reach this call.

The core fold path uses legacy plain `User:/Assistant:` prompts with partial
assistant primers (`core/graft_arena.py:1875–1905,1930–1943`), never the
configured Harmony turn function. It sets all layer shifts to `None` and
injects source payloads directly (`:2144–2146,2183–2190`); source text is not
scaffolded by default (`:1906`). It performs the full `ngen` loop without the
configured stop test and afterward splits only on `User:` variants
(`:2191–2217`). Harmony channel text can therefore leak into candidates.

The CPU replay executes the real repository and core fold: three raw digest
prompts, **360 model calls (120 each)** even though the fake reader emits
`<|end|>` immediately. The stop text survives in every candidate. Core QC
counts whitespace tokens/six-word repeats (`:2040–2073`), so a long contiguous
ellipsis run can pass QC. Coverage still rejects the fold at the unchanged
0.70 bar, and `_fold_once` marks its sources `no_fold` (`:3273–3279`).

**SCOUT-FIX-C7-FOLD-PROMPT-STOP:** the model adapter's configured prompt/stop
contract is bypassed by core consolidation; degenerate character runs also
escape its QC. These are core behaviors, not C7-owned fold generation code.
The observed GPU ellipsis attractor is consistent with this incompatible
generation path; its complete numerical cause is not proven by a fake model.

Minimal successor scope for the lead to adjudicate: make core consolidation
honor its dialect's prompt/continuation and stop contracts, preserve the
0.70 coverage rail, and pin degenerate-candidate rejection. **No such fix or
alternate generation algorithm was implemented or tuned in C7.**

Gates: `test_fold_uses_core_raw_prompts_120_steps_and_ignores_harmony_stop`
(`tests/test_grm_c7_lead_2_diagnosis.py:99`) and
`test_fold_recorded_ellipsis_can_pass_core_qc_but_coverage_rejects` (`:124`).

### Probe fields — HARNESS serialization omission, confirmed

Original first row's top-level keys, verbatim sorted:

```json
["class", "distance", "memory", "oracle", "probe_id", "turn"]
```

`question` and `expected` are **absent**, not stored as JSON null. A consumer
using `.get()` would see `None`. The immutable fixture has:

```json
"question": "What is the current C7-Fresh-0 value? Reply only with the answer; if unspecified, reply UNKNOWN.",
"expected": "Basalt-811"
```

`scripts/grm_c7_run.py:363–365` constructs the row without either field.
They remain available to the scorer (`:357`) and oracle (`:362`); this is
receipt-schema omission, not lost source/answer data or a core mutation.
`probe_schema.json` confirms all 60 rows. **Fix not implemented at the stop**;
the mechanical successor is to serialize those fixture fields into the row.

Gate: `test_probe_question_expected_omitted_at_c7_serialization`
(`tests/test_grm_c7_lead_2_diagnosis.py:140`).

## 2. Registration, SHA256 and exact commands

**r2 registration path/SHA: NONE — conditional mission did not activate.**
No r2 receipts directory or GPU launch commands were created. The lead's
2 GPU-h r2 / 3.9 GPU-h cumulative authorization is retained as dispatch
context; this seat used zero GPU time and does not authorize proceeding past
the core stop. Original r1 `lead_commands.txt` remains byte-identical.

The separate, immutable **diagnostic** registration is
`artifacts/grm_c7/diagnosis_lead_2/registration.json`, SHA256:

```text
3513ca93f29ece5794ba66eb017792e460b7167b9c2171cbcb3fcf1943eb33a3
```

It was written before six author CPU pins. Initial result: **1 failed,
5 passed, 2 warnings in 4.72s** (`cpu.log`). The fold fixture used
`add_document`, selecting the era prompt branch. This was a test-fixture
error, not a production failure hidden by a retry. The old test bytes and
failure log remain on disk.

A separately registered fixture correction sets only the fake source's
kind to `turn`, matching C7's digest sources; all criteria remain unchanged.
`fixture_amendment.json` SHA256:

```text
0555a0bcfa66c2e866faa3cb496aa54d611d76b6e42aa2077689acf19d5a6f82
```

Only that failed pin was rerun: **1 passed, 2 warnings in 5.79s**
(`cpu_fixture_amended.log`, `fixture_amended/fold.json`). Thus all six
diagnostic findings have passing pins across those two runs; this is not a
claim of a single six-pass run. SWIG deprecation warnings are retained.

Exact lead review commands are in `lead_commands_diagnosis.txt`:

```bash
cd /mnt/ForgeRealm/wt/grm-c7
sha256sum -c artifacts/grm_c7/diagnosis_lead_2/SHA256SUMS
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 GRM_C7_DIAG_OUT=/tmp/grm-c7-lead2-independent-review python -m pytest -q tests/test_grm_c7_lead_2_diagnosis.py
CUDA_VISIBLE_DEVICES='' python scripts/grm_c7_run.py --summary A
```

The CPU output directory must be new; receipt writers refuse overwrite.
The summary command is read-only and was executed here, returning FAIL.

## Prior art

Verified local systems: GRM contributors (2026), EB1 ephemeral framing,
WC1's successful orion fixture, LSR-P2B E2E model doubles and runtime lifecycle
CPU payload fixtures. Borrowed: exact input counterfactuals and replacement
of numerical boundaries while exercising real repository and serving paths.
New here: C7-specific UNKNOWN-case controls, live-prompt observation, raw fold
prompt/stop traces and omission census. No new memory, routing, admission,
fold-selection or inference algorithm is claimed. **No prior art known to me**
for this exact diagnostic composition.

The proposed successor's prompt/stop contract already exists in
`ArenaCache._attempt` and EB1 `harmony_turn` (GRM contributors, 2026); this is
an integration repair recommendation, not a novel decoding technique.
No external paper claim or unverified external citation was used. Annotations
also appear at diagnostic code sites and in `artifacts/grm_c7/LEDGER.md`.

## Deviations, RED, process safety, model and effort

Deviation: one documented, separately registered CPU fixture correction;
initial failure retained. Census discrepancies with the brief are reported
above. Controller receipts sum to 2674.948811272625 charged seconds; this
differs from the lead's 1.9 GPU-h accounting and is not used to reclaim budget
or infer total device occupancy. No ledger/order/registration was rewritten.

**RED / Not claimed fixed:** the C7 run is not a valid memory-quality
measurement; the oracle is not an established upper bound; exact causal
attribution of GPT-OSS answer/ellipsis failures remains unverified; core
fold handling is unfixed; r2 is unregistered and unrun; blind review is
outstanding. Zero-seat boundedness is vacuous as an exercised-residency
result (`scripts/grm_c7_run.py:484`). Some fold source payloads are injected
directly, so zero post-turn mounts does not establish that no transient
memory input was ever presented during folding.

Diagnostic doubles use a reversible CPU codec, uniform semantic keys,
NumPy payloads, no recency mounts, legacy seating, and an unbounded fake
cache. They execute actual routing/admission/attempt/oracle/fold code but
do not reproduce C2's numerical capture/seating geometry, paging or 300 turns.
No GPU-quality inference is licensed by their successful fake answers.

Process safety: no git, subagents, GPU execution, background jobs/waits,
service changes, signals or process kills. Foreground pytest processes exited.
All substantive mutations are two new diagnostic Python files, a new
diagnostic artifact directory, and appended C7 ledger entries. Original
campaign files remain in place; final integrity verifies 564 protected core
Python and r1 JSON/JSONL/log files plus frozen inputs. Existing payload/store
binaries were not written or removed; they were not all rehashed.

Agent model identity: **GPT-6** from the system; exact deployment ID is not
exposed. Effort: **high requested**; runtime effort is not independently
inspectable. Campaign reader: GPT-OSS-20B, registered snapshot
`6cee5e81ee83917806bbde320786a8fb61efebee`; never loaded in this diagnosis.
