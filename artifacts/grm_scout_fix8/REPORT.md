# GRM-SCOUT-FIX-8 — core pins GREEN; replay registered; full suite requirement RED

**Implemented the shared identifier normalization. CPU replay changes all 24 recorded r3 refusals from zero binding/zero forwards to binding and reader execution. C2's 132 recorded plans remain byte-identical. All 44 FIX8 gates pass. The requested blanket FIX-1..6 suite PASS is not achieved: historical FIX4/FIX5 source guards reject this checkout, and FIX6 is absent from its core. No GPU run or model-quality improvement is claimed.**

Evidence classes: CPU numerical-boundary fake with real admission/route/fit/attempt; CPU real policy over frozen C2 ranking/margin; suite runs; source/hash verification; external Unicode specification; lead GPU replay **NOT_RUN**.

## 1. Fix file/lines; tests with RED/GREEN evidence; C2 byte-identity.

Only three core files changed; [core_delta.patch](core_delta.patch) compares the archived before bytes to the final files.

| File / lines | Change |
|---|---|
| [core/grm_text_norm.py:30](/mnt/ForgeRealm/wt/grm-c7/core/grm_text_norm.py:30), 36–41, 49–82 | Extend routing's existing `normalize_glyphs`: NFKC, then all 31 Unicode 17.0 `Dash` property code points to `-`. Retain existing emphasis projection and preserve case for proper-noun extraction. Explicit table includes non-Pd minus signs and newer supplementary-plane dashes. |
| [core/grm_admission.py:75](/mnt/ForgeRealm/wt/grm-c7/core/grm_admission.py:75), 75–85 | `normalized_words` uses that same function before tokenization/casefold. Own-text digest binding, recency rescue and split-child binding all inherit this fix. |
| [core/graft_arena.py:1386](/mnt/ForgeRealm/wt/grm-c7/core/graft_arena.py:1386), 1386–1415, 2025–2064 | Query content and node lexical scans use the existing `_norm_text` wrapper; identifier/lexical emissions casefold consistently. Existing LSR normalization switch remains. |
| [core/graft_arena.py:2141](/mnt/ForgeRealm/wt/grm-c7/core/graft_arena.py:2141), existing deposit path | No new deposit policy: digest own-text rare keys already call `_rare_tokens(note)`, now receiving the shared projection. Source rare-key inheritance remains routing evidence; it does not establish own-text admission binding. |

Python 3.12.3 supplies NFKC through Unicode database 15.0.0; the explicit Dash table is Unicode 17.0. The frozen scorer `scripts/grm_c7_common.py` is unchanged, SHA-256 `5f8018edd43c0e794ac8ed72424a872cc3de82ccffe6f85d65ec88b0f1531fe0`.

| Gate | Result and receipt |
|---|---|
| Actual pre-edit RED | [red.log](red.log), [red.json](red.json): exit 1, **0/24 binders, 0/24 forwards**, all 24 deterministic admission refusals. Captured before core edits. |
| Post-edit GREEN | [green.log](green.log), [green.json](green.json): exit 0, **24/24 binders, 24/24 forwards**. Same probe IDs, stored texts and checkpoint metadata. |
| Per-probe treatment | [PER_PROBE.md](PER_PROBE.md) lists all 24 IDs, before/after bindings, final CPU mounts and forward counts. Cohort: 8 answerable fresh, 6 answerable folded, 10 controls (including 3 correction controls). |
| C2 recorded-plan identity | [c2_identity.json](c2_identity.json), [c2_recorded_plans.jsonl](c2_recorded_plans.jsonl), [c2_replayed_plans.jsonl](c2_replayed_plans.jsonl): **132/132**, same in RED and GREEN. Both streams SHA-256 `eae443d91b0a18231865f397526f8e16cdee3c04aa47f8baf6db3a91114a35ce`. |
| FIX8 unit/controller gates | [fix8_gates_amendment1.log](fix8_gates_amendment1.log): **44 passed**. All 31 dash points, NFKC, Unicode casefold, preserved caps, wrong digits, separator negatives, split own-text, digest inheritance negative, C2 receipt pins, source drift, missing results, owner/orphan rejection, busy lease charge/no retry, expected-value isolation, and actual archived module origins. |
| Broad regression | [suites.log](suites.log): **400 passed, 12 failed**. Includes FIX1–5, SC1.1, RT1 split and LSR-P2C suites plus FIX8 core pins. Failures are detailed below, not suppressed. |
| FIX4 correct historical environment | [fix4_registered_env.log](fix4_registered_env.log): **1 passed, 9 failed**, all nine at `ValueError: INPUT_SHA_MISMATCH: core/graft_arena.py`. The tenth initial failure was the missing r2/FIX4 launch environment. |
| FIX6 compatibility | [fix6_before_compatibility.log](fix6_before_compatibility.log), [fix6_compatibility_retry.log](fix6_compatibility_retry.log): **10 failed before and 10 failed after**, against the sibling's CPU API suite. This core lacks FIX6's `admission_rule` / margin-first API. |

C2 scope: full pre-probe manifests, real identifier binding/RT1 demotion/policy; recorded numerical ranking and measured margin are frozen inputs. Each execution ID, worker/descriptor/manifest hash and individual compact-JSON plan bytes are in `green.json.c2`. No numerical router rerun or answer-quality parity is inferred. The 132 question strings are ASCII; this is a corpus-specific identity claim.

CPU r3 scope: private fake repositories retain checkpoint own text, kind, retirement, lineage and stored rare keys. Numerical embeddings, tokenizer geometry, payloads and model are doubles. The real serving ladder runs, but its fake answer is not scored as model evidence. Some folded fake reads mount only one relevant fragment: binding is not proof that both attributes are mounted or read correctly. No stored raw sources are revived.

Reproduce the CPU receipt program with a fresh output phase namespace if rerunning; its original `red.json` and `green.json` are immutable and writes intentionally use exclusive creation. Original commands:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python scripts/grm_scout_fix8_cpu.py --phase red
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python scripts/grm_scout_fix8_cpu.py --phase green
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_grm_scout_fix8.py tests/test_grm_scout_fix8_replay.py
```

## 2. Replay registration path + sha; exact lead command.

Registration: [replay_registration.json](replay_registration.json)

SHA-256: `f42b61df6d413551431c84042d6ceca11dfa5e3d89b600a7669f5cc76fa4283b`

Required separate amendment: [replay_amendment_1.json](replay_amendment_1.json)

SHA-256: `aed05d7cb542bf7ff2418e42de90b7ac8d0f53dc673ab1a76c69dd39ea316af5`

The amendment corrects a missing `requests.json` in one CPU test's temporary directory and adds strict before/after verification for the amended test/verifier. It also replaces a static source assertion with an actual fresh-process module-origin check. It changes no core behavior, cohort, budget, threshold, scorer or A7 registration. The initial **43 passed / 1 failed** test receipt remains in [fix8_gates.log](fix8_gates.log).

Exact lead command:

```bash
bash /mnt/ForgeRealm/wt/grm-c7/artifacts/grm_scout_fix8/lead_commands_fix8.txt
```

CPU-only preflight:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash /mnt/ForgeRealm/wt/grm-c7/artifacts/grm_scout_fix8/lead_commands_fix8.txt --check-only
```

The combined preflight passed in [lead_check.log](lead_check.log); the amended verifier and isolated A7 process passed again inside the 44-test gate. Source/order/A7 integrity is recorded in [final_integrity.json](final_integrity.json).

Sequence and registered scope:

1. Execute the **existing** amendment-7 six batches and its existing output namespace/registration. Original registration SHA remains `f4f1aa27f4db79e1fd95c995d4e72e8b70ea0f29140effaa56462bcc31a294f1`. The bridge loads the three exact archived core modules in a fresh process, verifies every original A7 input hash with that explicit path mapping, and records module origins. It changes no live file or contrast parameters. Completed A7 batches are verified and not regenerated. Original A0 raw-answer parity-before-A1 and stop rules remain.
2. Execute FIX8's six batches, four refusals each, from their registered **cell-end** r3 checkpoints. Private repository copies, current repaired core, real admission + fit + mount + production reader, original plain questions, 32-token per-attempt answer budget and original ladder flags. No forced historical mounts; no expected values passed to the model. Production splitting may add children only inside the private copy. Score with the unchanged frozen scorer, preserving exact/wrong/abstention/unsupported counts and every raw response.
3. Report PASS_ADMISSION only if all 24 rows/controllers complete, bind and mount without identifier-unbound refusal. Incomplete data is NOT_MEASURED. Reader quality is separately reported; no quality threshold is fitted after observation.

**Budget interpretation:** the new FIX8 campaign reserves **6 × 280 = 1680 seconds (0.4667 GPU-h), cap 1800 seconds (0.5 GPU-h)**. The already-registered A7 campaign keeps its separate 1680-second reservation / 1800-second cap. Combined scheduled reservations are therefore **3360 seconds (0.9333 GPU-h)**, sequenced, not concurrent. The new 0.5 GPU-h cap is not represented as a cap on both registrations combined.

Busy lease, insufficient disk, input/payload drift, existing/failed/orphaned run, timeout or overrun stops without retry or cohort trimming. Full checkpoint payload tree hashes are verified at lead open. FIX8 charges the full 280-second reservation even on success; overruns are charged in full and RED. CMC1's in-process alarm cannot hard-preempt an uninterruptible native call; no outer kill watchdog is introduced. Throughput/fit within the reservations is unmeasured. Model: `openai/gpt-oss-20b`, snapshot `6cee5e81ee83917806bbde320786a8fb61efebee`.

## 3. Prior art; deviations; RED; process safety; model id and effort.

## Prior art

**GRM contributors (2026), locally verified:** SC1.1/DET1.4 supplied the existing glyph projection; ADM1/A-DEC/RT1 supplied own-text binding and split semantics; C7/FIX4 supplied numerical doubles, checkpoint/source joins and recency serving; C2/LT1 supplied recorded-plan reconstruction from full manifests; amendment7/RD1 supplied private checkpoint reader replay; CMC1 supplied the foreground lease. Taken: these interfaces and policies. Ours: the missed normalization wiring, the order's explicit broader glyph mapping, and the per-probe/lead replay adapters. **No prior art known to me for this exact composition.** No novel retrieval algorithm is claimed.

**Unicode Consortium (2025):** NFKC follows [UAX #15](https://www.unicode.org/reports/tr15/); the dash set is the 31-code-point `Dash` property in [Unicode 17.0 PropList](https://www.unicode.org/Public/17.0.0/ucd/PropList.txt). Verified through official web sources during this seat. The mapping is spec reuse. Attributions also appear at code sites and in [LEDGER.md](LEDGER.md).

### Deviations and RED

- **Not claimed fixed: blanket FIX-1..6 suite compatibility.** FIX4's r2 source chain already predates this checkout's FIX5 arena change ([legacy_boundary_comparison.json](legacy_boundary_comparison.json)). With correct historical flags, nine tests stop at its immutable source guard. FIX5's own exact source hash and two-method-only diff guards intentionally reject FIX8's authorized additional methods. Their registrations/tests were not rewritten or weakened. The separate FIX8 registration pins the new source instead.
- **FIX6 is not integrated in grm-c7.** Its CPU tests were read from grm-lt1 and executed against both archived-before and current core: 10 failures each. Representative failure is `KeyError: 'admission_rule'`. First discovery attempt had `ModuleNotFoundError: No module named 'scripts.grm_lt1_admission'`; the explicit read-only import adapter resolved collection, with the API failures retained. Importing FIX6's admission policy would exceed this normalization-only core diff.
- **Cell-end replay, not exact pre-probe reconstruction.** This uses the recorded checkpoints available to A7. No historical metadata rollback or preceding-turn replay is asserted. CPU geometry is also narrower than the model frame. A7 retains its original parity rail; FIX8 records treatment reads without pretending they are baseline-identical.
- **Model answer quality / recovered folded pairs / control safety / alias behavior remain unmeasured under FIX8.** An identifier hit does not prove the requested attribute exists. All ten selected controls remain controls and are scored accordingly. The frozen scorer is not broadened to hide failures.
- Existing LSR switch semantics and case-bearing proper-noun extraction are retained; the requested NFKC/Dash equivalence applies beyond U+2011. The frozen DET scorer is not made into a second mutable normalizer.
- Author CPU validation only. Blind validation is the lead's responsibility under HOUSE_RULES; no subagent was launched. No cross-model or GPU acceptance is claimed.

### Process safety and identity

No GPU/model-weight load, git command, subagents, shell background execution/waits, signals, process termination, service operation or external message in this seat. Foreground CPU children were awaited to completion; existing processes were not killed. Only this task's temporary test ownership files were removed. Sibling worktrees were read for historical C2/FIX6 evidence, not edited. `.git` pointer and worktree HEAD were read as files to confirm `grm-c7`, without invoking git. The order and original registrations remain immutable; amendments are separate.

Agent model **gpt-6-astra**, reasoning effort **high**, verified from `logs/grm_fix8_r1.log:6,10`. No GPU time consumed by the seat. Registered lead runs remain **NOT_RUN**.


## 2026-09-09 order amendment 1 — F5 OOM successor (CPU only)

The lead completed F1-F4; F5 failed `RuntimeError: cudaMalloc failed: out of memory`
after two completed rows and a third attempted checkpoint copy. Failure memory
was unrecorded; transient versus own-peak remains undetermined. The original
F5 RED and full 280s charge are retained. One create-only F5-R1 successor is
registered with 120s (the original cap's slack), then F6 retains 280s; total
historical plus remaining charges =1800s. No original registration/core/receipt
was rewritten. Each new lease has a >1000 MiB preexisting-device-memory refusal
and recorded PIDs; row/failure samples retain diagnostic evidence.

[Order amendment report](resume_amendment_1/REPORT.md),
[ledger](resume_amendment_1/LEDGER.md),
[registration](resume_amendment_1/registration.json),
[lead command](lead_commands_fix8_resume.txt).
Registration SHA-256: `6a6800a0c222487d9bf7f89afd77acf82266ce0754a2e2bb146e0a81bcef244e`.
CPU author baseline 80 passed; lead --check-only passed; 136 historical hashes
unchanged. No GPU run or OOM clearance claimed. Prior art: GRM C7/CMC1/FIX8
(contributors, 2026), NVIDIA nvidia-smi documentation (accessed 2026-09-09);
reused leases/receipts/telemetry, new exact successor wiring. No prior art known
to me for this exact composition. Agent gpt-6-astra, high. Process-safety
residual: overbroad read-only discovery tool session 95902 remains unconfirmed;
no process killed. Full limits and original RED retained in amendment report.
