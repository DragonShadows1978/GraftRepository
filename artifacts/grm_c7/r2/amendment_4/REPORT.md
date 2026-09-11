# GRM-C7 amendment 4 — diagnosis complete; oracle and folds remain RED

**Receipt finding:** the reported oracle 18/52 mixes **16 answerable abstention failures** with **18 valid unanswerable abstentions penalized by case-sensitive exact matching**. All 16 answerable failures are fresh/alias queries; the eight answerable correction queries succeed. The oracle already uses the serving Harmony shape. It is not a reliable empirical upper bound: memory answers `Basalt-811` correctly on four rows where this oracle says `unknown`.

**Fold ruling: STOP AND REPORT — SCOUT-FIX-5 candidate; not claimed fixed.** FIX-3's Harmony path ran, but all 13 folds fail coverage. Of 39 candidates, 35 stop before 120 tokens and four exhaust the budget. The coverage rule binds the observed fold decisions; raising the cap alone is not supported as a remedy. The core archive prompt/path produces omitted and misattributed facts. Prompt-only causation versus multi-source K/V reading remains unisolated.

**Memory finding:** the requested 30 answerable abstention errors split **(a) no mount 0, (b) mounted reader abstained 30, (c) FIX-4 recency 0**. Across all 48 abstaining rows, the split is **0 / 47 / 1**; that recency row is a valid unanswerable control.

## Evidence and deliverables

Evidence class throughout: **CPU replay of existing lead-run model receipts**, with source inspection; causal interpretations below are labeled reasoning. No model or GPU run occurred here.

- [Oracle table](replay_final/ORACLE_TABLE.md): all 34 abstaining continuation rows, classifications, expected values, and verbatim wrapped prompts and served text. [Machine-readable rows](replay_final/oracle_abstentions.json) also preserve prompt token IDs.
- [Fold table](replay_final/FOLD_TABLE.md): all 13 events and all 39 verbatim candidates, source text, need/hits, coverage, QC, missing lexical facts, generation length and stops. [Machine-readable folds](replay_final/folds.json) retain reconstructed prompts separately from recorded prompts.
- [Memory table](replay_final/MEMORY_TABLE.md): all 48 abstentions, flags selecting the requested 30 errors, actual mounted nodes and their checkpoint source text. [Machine-readable rows](replay_final/memory_abstentions.json).
- [Oracle-only proposal prompts](replay_final/oracle_proposal_prompts.json): 52 source-derived baseline/treatment prompt pairs, provenance and treatment token IDs; model results are `NOT_RUN`.
- [Summary receipt](replay_final/summary.json), [historical oracle quarantine](replay_final/HISTORICAL_ORACLE_QUARANTINE.md), [ledger](LEDGER.md), [registration](registration.json), and [CPU replay script](../../../../scripts/grm_c7_amendment4.py).

There are **36 new continuation cells** in `r2/fix4_attempt_1/cells`, plus **three preserved cells** in `r2/cells`; all 39 controller receipts say COMPLETE. The 52-probe denominator is cells 4–39 only. The first eight probes remain excluded under the existing quarantine. Folds include turns 8 and 17 from preserved cells plus 11 continuation folds. This explains the directory/count distinctions in the order without rewriting history.

## Oracle diagnosis

| Classification | Rows | Concrete observation | Ruling |
| --- | ---: | --- | --- |
| Answerable direct fact abstention | 8 | Fresh-0/1 at distances 30,60,120,250; exact value is in a single source record | Model/serving output abstains despite source; precise cause unresolved |
| Answerable alias abstention | 8 | Signal-0/1 at the same distances; both base value and alias edge are present | Alias composition/input-presentation sensitivity is a hypothesis |
| Valid unanswerable control, exact-case mismatch | 18 | Questions ask for an inspection password that no source states; model returns `unknown`, expected is `UNKNOWN` | Correct abstention behavior; frozen exact scorer marks failure |
| Correction conflict abstention | 0 | Eight answerable current-value correction rows return the latest `Onyx-911`/`Onyx-912` | Conflicting chronological records do not explain these oracle failures |
| Proven reasoning-low truncation | 0 established | Oracle receipt lacks output IDs/count and finish reason | NOT_MEASURED; do not infer a cap hit from `unknown` |

Every recorded oracle prompt equals `harmony_turn(live_prompt, None)`, and local tokenizer encoding exactly reproduces its recorded token IDs (52/52). Source records equal the immutable fixture oracle sources, and oracle mounted IDs are empty. The prompt is already the exact configured system/user/final-channel shape; adding Harmony again would not repair a missing wrapper. These are input-construction checks, not proof of numerical attention fidelity.

Representative answerable failure, `c7_fresh_0_d030` (verbatim; complete table quotes every row):

```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 6] The current C7-Fresh-0 value is Basalt-811.
Use only those records.
What is the current C7-Fresh-0 value? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text: `unknown`. Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-032-039/probes.jsonl` (row location in the oracle table).

The alias failure's source includes both `The current C7-AliasBase-0 value is Jasper-711.` and `C7-Signal-0 is an alias for C7-AliasBase-0.`; the served text is still `unknown`. No old/new competing assignments exist in either failing answerable family. The same strings and outputs recur at all four distances; these are repeated probes of four underlying answerable questions, not 16 independent novel question types.

The control mismatch is a harness/scoring contract finding: `scripts/grm_c7_run.py:115-123` lowercases the fallback instruction; `scripts/grm_c7_common.py:161-189` preserves case for exact equality but casefolds abstention detection. Thus `unknown` and `Unknown.` are recognized as abstentions yet differ from expected `UNKNOWN`. No scorer edit or retrospective acceptance change is made. There are 34 answerable rows: oracle answers 18 and abstains on 16. The headline 18/52 is the frozen exact score, not a pure measure of fact-reading capacity.

The four memory-over-oracle rows are `c7_fresh_0_d030`, `_d060`, `_d120`, `_d250`. **Reasoning:** full information offers a conceptual ceiling, but a particular prompted model execution need not dominate another representation. This measurement already violates per-row dominance and should not certify a product gap against an ideal oracle.

## Registered oracle-only proposal

Registration SHA-256: `fa59abaf6ea34b224d03d34cdb5dc5ef127d1773a8b1f7bb0cb8359f808fc37a`. Evidence amendments are separate immutable files; they do not change this registration or the campaign acceptance rules.

The additive CPU script prepares a **source-grounded latest-value presentation** inside the same `harmony_turn` wrapper. It parses the fixed C7 source grammar into entity values, alias edges and inbound/outbound fields; chronological overwrite supplies the latest assignment for each entity/field. Alias expansion uses only an explicitly recorded base value. It retains the original chronological records below the rendered lines. The transform accepts only `(turn, source_text)` records: no expected answer, answerability flag or question is supplied to the transform.

Examples produced from source records:

```text
C7-Fresh-0: latest value = Basalt-811 [turn 6].
C7-AliasBase-0: latest value = Jasper-711 [turn 5].
C7-Signal-0: alias of C7-AliasBase-0 [turn 9].
C7-Signal-0: latest value through C7-AliasBase-0 = Jasper-711 [value turn 5; alias turn 9].
```

The effective question stays byte-identical, as do the 32-token answer budget, Harmony wrapper/stops and scorer. No inspection password or absent base value is synthesized. Every rendered assertion has exact source-substring provenance. CPU hand cases check chronological ordering, overwrite, direct alias propagation, missing alias target, and empty inputs. Original records remain present, including superseded values, so this is explicit interpretation added to the ceiling input, not a product memory change.

Status: **REGISTERED / CPU-REPLAYABLE / MODEL EFFECT NOT_RUN**. Replay command from the worktree (choose a fresh output directory):

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/grm_c7_amendment4.py --output /tmp/grm-c7-amendment4-replay
```

A later lead-run comparison would feed each paired user prompt through the existing oracle state isolation and `_attempt(..., [], 32, ..., defer_memory=True)`, retaining the current wrapper/stops. It must preserve baseline receipts and negative controls, log output token IDs and finish reasons, and report all pairs. No such model run is authorized or executed by this CPU diagnosis. Improvement is not claimed. This proposal does not repair the case-sensitive control score, and must not be presented as product retrieval improvement.

## Folds: budget, coverage and what is lost

| Turn | Best hits / need | Required hits at 0.70 | Best coverage | Tokens, candidates 0/1/2 | Binding observation |
| --- | --- | ---: | ---: | --- | --- |
| 8 | 3/10 | 7 | 0.3000 | 60/59/67 | All stop early; relations crossed, times invented |
| 17 | 2/10 | 7 | 0.2000 | 20/15/80 | Retains Fresh-2; drops AliasBase and other Fresh records |
| 26 | 2/8 | 6 | 0.2500 | 21/22/102 | Drops alias edges and latest Vesper-0; one alias points to wrong base |
| 55 | 2/8 | 6 | 0.2500 | 63/63/51 | Drops Vesper updates and one inbound relation; wrong ownership |
| 74 | 2/8 | 6 | 0.2500 | 16/16/59 | Retains Outpost-02; drops Archive-2 and Outpost-00/01 |
| 85 | 3/9 | 7 | 0.3333 | 37/31/120 | Outpost-06 assigned Pearl-1205 instead of Slate-1206; p2 hits cap |
| 91 | 3/12 | 9 | 0.2500 | 42/51/120 | Outpost-10 assigned Slate-1209 instead of Slate-1210; p2 hits cap |
| 95 | 2/9 | 7 | 0.2222 | 20/120/120 | Omits most outposts/alias, misattributes values; p1/p2 hit cap |
| 99 | 2/8 | 6 | 0.2500 | 44/16/82 | Retains one pair, drops three; invented time in p0 |
| 103 | 2/8 | 6 | 0.2500 | 14/16/87 | Retains Outpost-22; drops Outpost-19/20/21 |
| 107 | 2/8 | 6 | 0.2500 | 40/16/80 | Retains Outpost-26; drops Outpost-23/24/25 |
| 111 | 2/8 | 6 | 0.2500 | 41/16/87 | Focuses Outpost-29; p1 attaches Slate-1228 instead of Slate-1229 |
| 115 | 2/8 | 6 | 0.2500 | 14/16/81 | Outpost-32 assigned Slate-1234 instead of Slate-1232 |

All candidate text, all missing lexical fact sets and all sources are quoted in the linked fold table. `digest_text` is null for every event; these are rejected candidates, not deposited digests. For example, turn 91 candidate 0 is:

```text
For the archive: the  name “C7-Outpost-10” refers to the value “Slate-1209”; the name “C7-Handle-10” is an alias for “C7-Outpost-10.”
```

This gets three lexical hits out of 12 while assigning the wrong value. **Reasoning:** token-set coverage is a necessary retention check here, not a relation-correctness proof. Lowering it would admit candidates that already misbind facts.

FIX-3 evidence: the worker's executed-source hash matches each fold start binding; the preserved pre-FIX4 core archive is hash-verified and its consolidation/QC/coverage/prompt methods and constants are AST-identical to current core. The configured serving wrapper is Harmony. All 35 early-stopping decode traces end with `<|return|>` and exhibit the configured stop-loop short circuit. Candidate text is reproduced exactly from the raw final decode and source-bound primer. The four cap hits occur at turn85/p2, turn91/p2, turn95/p1 and p2. Fold input token IDs were not recorded: reconstructed wrapped prompts are labeled as reconstructions, not direct prompt receipts.

Twelve candidates fail shape QC (third candidate for every fold except turn55). These receive recorded coverage zero; the table separately shows lexical coverage before QC. This includes table separator punctuation and the three-bullet turn8 candidate. None of the 39 candidates reaches 0.70 even before QC. Every event has at least one short, stopped prose candidate with poor coverage. Thus 120 tokens constrain four attempts, but do not explain the 13-event failure pattern or demonstrate that a larger budget would fix it.

Instrumentation finding: `start.json` claims the wrong prompt family at **turns 8,85,99,107,115**. The observer captures missing kind as null (`scripts/grm_c7_run.py:188-199`) and treats null as deep; core defaults an absent kind to turn (`core/graft_arena.py:2155`). All actual candidate primers match DIGEST prompts. The report preserves the inconsistent start prompts and reconstructs the matching prompts from pinned code. Turn17 is consistent; this corrects provisional progress wording about the inherited cells. No observer edit was made.

### SCOUT-FIX-5 boundary report

Core locations: `core/graft_repository.py:3261` calls consolidation; `core/graft_arena.py:1875-1887` owns the three DIGEST prompts; `:1903-1906` defaults text scaffold OFF and generation to 120; `:2190-2200` concatenates and injects source K/V; `:2202-2220` wraps the archive instruction and adds its primer; `:2245-2250` applies shape QC; `:2291` rejects insufficient coverage. The prompt asks for the conversation “above,” while source facts are injected K/V and absent from the live archive instruction. It also repeatedly asks for times, although these fixtures carry no times; candidates fabricate `12:00`, `10:00 AM` and `12:29`.

**Reasoning / successor recommendation:** the existing core prompt/path is inadequate on this model/fixture and needs a separately authorized SCOUT-FIX-5 investigation. A bounded contrast can isolate the same mounted sources under a task-specific, source-enumerating archive instruction, and compare source-in-live versus multi-source K/V with exact input/output token receipts. Reuse the existing optional source-scaffold concept (`_source_scaffold` / `_consolidation_prompts`, GRM contributors, 2026); no new algorithm is claimed. Keep the 0.70 gate and 120 budget fixed initially to isolate the cause. Do not infer a prompt-only root cause or silently turn on the scaffold in this campaign. **STOP at core; no patch, new model run, threshold change or budget increase. Not claimed fixed.**

## Memory: admission versus reader versus recency

| Probe class | Answerable abstention errors | (a) Refused/no mount | (b) Mounted reader | (c) FIX-4 recency |
| --- | ---: | ---: | ---: | ---: |
| Fresh | 4 | 0 | 4 | 0 |
| Alias | 8 | 0 | 8 | 0 |
| Correction | 8 | 0 | 8 | 0 |
| Folded | 10 | 0 | 10 | 0 |
| Total | 30 | 0 | 30 | 0 |

The full abstaining superset includes 18 unanswerable controls: 47 mounted-reader rows and one recency row. `c7_correction_2_d005` serves `Unknown.` from node15 (`served_from=recency_mount`), appropriately abstaining about an unstated inspection password. Category c takes precedence over b because recency also mounts. A recency nomination alone is not classified as recency service.

Alias dominates category b **8/8**: actual mounts are `[8]` or `[9]`, whose text states only the Signal→AliasBase edge. The Jasper value resides in node4 and is not mounted. This is an admitted but incomplete evidence set, not zero-hit refusal. Correction dominates category b **8/8**: mounts `[0,12]` or `[0,14]` include the original Mica records and a current Onyx fact; the reader still serves `Unknown.`. The same correction queries succeed in the live oracle. These observations narrow the failure surface without proving whether representation, conflict handling or numeric reading is responsible.

Folded errors include partial answers such as `Reed-611 | unknown` and `Outbound: Cedar-1012 | Inbound: unknown`. The frozen scorer counts these as abstentions, so **wrong_value_error=0 does not imply there are no wrong relation/value components**. The latter answer puts an inbound value in the outbound slot. No folded digest was accepted or mounted; `folded_path_exercised=false` remains RED. The 93-seat observed maximum is residency evidence only.

## Prior art

Verified local systems: C7 chronological-source oracle, exact scorer and SHA-bound receipts; EB1 `harmony_turn`/stop contract; FIX-3 archive prompts, stop-aware generation and QC; SC1 glyph projection; GRM optional source scaffold (GRM contributors, 2026). Taken unchanged: wrapping, tokenization, scoring, fact-set/coverage/QC functions, provenance and receipt discipline. Python AST tooling supplies pure-function extraction and structural comparison (Python contributors, 2026); no numerical runtime is imported.

Ours: receipt census, decode-trace reconstruction, discrepancy reporting, and the C7-grammar source-only latest-field/alias rendering for an oracle diagnostic. Chronological overwrite is an ordinary latest-write rule. **No prior art known to me for this exact audit/renderer composition.** No new retrieval, memory, compression or proof algorithm is claimed. No external paper claim or literature novelty claim is made; literature was not searched. Code-site annotations and the ledger carry these same attributions.

## Deviations, RED, validation and process safety

The immutable order and registration remain intact; three separate evidence amendments bind historical core/parity, explain the observer mismatch and pin controller/manifest evidence. Early CPU attempts failed on audit assumptions (whole-file core equality across FIX4, then trusting start-prompt family); failure directories and ledger entries remain. Exact replay assertions were preserved, and the final replay passes. This is author verification, not a blind independent audit.

RED: oracle treatment effect unmeasured; oracle output token/finish telemetry absent; 16 answerable oracle abstentions unresolved; exact-case control mismatch unmodified; all 13 folds rejected; fold prompt versus K/V causation unisolated; memory 30 answerable abstention errors unresolved; historical quarantine remains; no product acceptance or restart-quality certification issued. **Not claimed fixed.**

Process safety: CPU-only foreground commands, local receipt/tokenizer reads, additive analysis artifacts and script. No git, subagents, model loading, GPU work, network, background waits, process kills, service changes or core edits. Input hashes are rechecked after replay; historical campaign sources and receipts are preserved.

Model under test: **openai/gpt-oss-20b**, revision **6cee5e81ee83917806bbde320786a8fb61efebee**, `resident_packed_mxfp4`, `tensor_cuda GptOss20B_TC`; prompt effort **Reasoning: low / Valid channel: final**. Agent: **Codex, GPT-6 family as identified by session instructions; exact backend model ID is not exposed**. Requested analysis effort: **high**; backend effort metadata is not independently exposed. CPU verification details are in `replay_final/summary.json` and the final integrity receipt.
