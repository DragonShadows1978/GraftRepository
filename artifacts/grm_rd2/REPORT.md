# GRM-RD2 — two mechanisms; correction fixture omission and incomplete alias retrieval

**RED; not claimed fixed. Class (b) = 8, class (c) = 8, class (a) = 0, class (d) = 0. There is no unique dominant class. No retired node was mounted.** All eight corrections serve the original **Mica** value while both Mica and current Onyx are mounted. All eight aliases serve **“I’m sorry, but I can’t provide that.”** The frozen scorer labels this `wrong_value_error=1`; these eight are refusals, not stale-value answers.

**Ruling:** the correction lineage is incomplete because the C7 harness records its first correction as an ordinary fact. L2 has no edge from the current Onyx nodes to the original Mica node. A-DEC ranks the still-active Mica node first, and RS3 places it immediately before the live question. Alias admission binds the Signal edge but omits the Jasper base. Repairing general alias dependency retrieval crosses the core boundary; **STOP: no implementation or further experiments after the registered CPU diagnosis.** No retired-node filtering defect is demonstrated. The exact causal contribution of position versus text/payload form to the reader's stale choice remains unmeasured.

Evidence classes: **existing end-to-end receipts** for answers/mount IDs and C2 scores; **CPU metadata replay** for admission, L2, fit, seating and correction targets; **source inspection** for harness/prompt semantics; **reasoning** for causal interpretation and recommendations. No new inference, native ranking, GPU, or numerical K/V replay. This report extends the RD1 evidence narrative; the older RD1 reports and immutable orders remain untouched.

## The 16 rows

The exact A2 question is quoted, including its remaining answer instruction. Mounted IDs are **zero-based**; the legacy attempt `mounts` display is one-based and must not be substituted for `mounted_ids`. The quoted mounted content is the exact user-content span. [rows.json](rows.json) includes each full stored Harmony text, all related node texts, retirement metadata, per-turn route receipt, source receipt path/line, checkpoint boundary, and CPU seat coordinates. [ROW_TABLE.md](ROW_TABLE.md) adds per-turn routing traces. Full turn nodes include the low-reasoning system frame and assistant `Recorded.`; bare fact nodes 12/14 do not.

| Turn / probe | Question (verbatim A2) | Expected | Served (verbatim) | Admission plan → actual mounts; exact mounted content | Class |
|---|---|---|---|---|---|
| 39 / c7_alias_0_d030 | What is the current value for C7-Signal-0? Reply only with the answer. | Jasper-711 | I’m sorry, but I can’t provide that. | [8] → [8]<br>8 **ALIAS EDGE Signal-0**: C7-Signal-0 is an alias for C7-AliasBase-0. | (c) |
| 40 / c7_alias_1_d030 | What is the current value for C7-Signal-1? Reply only with the answer. | Jasper-712 | I’m sorry, but I can’t provide that. | [9] → [9]<br>9 **ALIAS EDGE Signal-1**: C7-Signal-1 is an alias for C7-AliasBase-1. | (c) |
| 47 / c7_correction_0_d030 | After all corrections, what is the current C7-Vesper-0 value? Reply only with the answer. | Onyx-911 | Mica-431 | [0, 11, 12] → [0, 12]<br>0 **SUPERSEDED semantically (Mica), still active**: The current C7-Vesper-0 value is Mica-431. The current C7-Vesper-1 value is Mica-432.<br>12 **CURRENT Onyx-911 fact**: The current C7-Vesper-0 value is Onyx-911. | (b) |
| 48 / c7_correction_1_d030 | After all corrections, what is the current C7-Vesper-1 value? Reply only with the answer. | Onyx-912 | Mica-432 | [0, 13, 14] → [0, 14]<br>0 **SUPERSEDED semantically (Mica), still active**: The current C7-Vesper-0 value is Mica-431. The current C7-Vesper-1 value is Mica-432.<br>14 **CURRENT Onyx-912 fact**: The current C7-Vesper-1 value is Onyx-912. | (b) |
| 69 / c7_alias_0_d060 | What is the current value for C7-Signal-0? Reply only with the answer. | Jasper-711 | I’m sorry, but I can’t provide that. | [8] → [8]<br>8 **ALIAS EDGE Signal-0**: C7-Signal-0 is an alias for C7-AliasBase-0. | (c) |
| 70 / c7_alias_1_d060 | What is the current value for C7-Signal-1? Reply only with the answer. | Jasper-712 | I’m sorry, but I can’t provide that. | [9] → [9]<br>9 **ALIAS EDGE Signal-1**: C7-Signal-1 is an alias for C7-AliasBase-1. | (c) |
| 77 / c7_correction_0_d060 | After all corrections, what is the current C7-Vesper-0 value? Reply only with the answer. | Onyx-911 | Mica-431 | [0, 11, 12] → [0, 12]<br>0 **SUPERSEDED semantically (Mica), still active**: The current C7-Vesper-0 value is Mica-431. The current C7-Vesper-1 value is Mica-432.<br>12 **CURRENT Onyx-911 fact**: The current C7-Vesper-0 value is Onyx-911. | (b) |
| 78 / c7_correction_1_d060 | After all corrections, what is the current C7-Vesper-1 value? Reply only with the answer. | Onyx-912 | Mica-432 | [0, 13, 14] → [0, 14]<br>0 **SUPERSEDED semantically (Mica), still active**: The current C7-Vesper-0 value is Mica-431. The current C7-Vesper-1 value is Mica-432.<br>14 **CURRENT Onyx-912 fact**: The current C7-Vesper-1 value is Onyx-912. | (b) |
| 129 / c7_alias_0_d120 | What is the current value for C7-Signal-0? Reply only with the answer. | Jasper-711 | I’m sorry, but I can’t provide that. | [8] → [8]<br>8 **ALIAS EDGE Signal-0**: C7-Signal-0 is an alias for C7-AliasBase-0. | (c) |
| 130 / c7_alias_1_d120 | What is the current value for C7-Signal-1? Reply only with the answer. | Jasper-712 | I’m sorry, but I can’t provide that. | [9] → [9]<br>9 **ALIAS EDGE Signal-1**: C7-Signal-1 is an alias for C7-AliasBase-1. | (c) |
| 137 / c7_correction_0_d120 | After all corrections, what is the current C7-Vesper-0 value? Reply only with the answer. | Onyx-911 | Mica-431 | [0, 11, 12] → [0, 12]<br>0 **SUPERSEDED semantically (Mica), still active**: The current C7-Vesper-0 value is Mica-431. The current C7-Vesper-1 value is Mica-432.<br>12 **CURRENT Onyx-911 fact**: The current C7-Vesper-0 value is Onyx-911. | (b) |
| 138 / c7_correction_1_d120 | After all corrections, what is the current C7-Vesper-1 value? Reply only with the answer. | Onyx-912 | Mica-432 | [0, 13, 14] → [0, 14]<br>0 **SUPERSEDED semantically (Mica), still active**: The current C7-Vesper-0 value is Mica-431. The current C7-Vesper-1 value is Mica-432.<br>14 **CURRENT Onyx-912 fact**: The current C7-Vesper-1 value is Onyx-912. | (b) |
| 259 / c7_alias_0_d250 | What is the current value for C7-Signal-0? Reply only with the answer. | Jasper-711 | I’m sorry, but I can’t provide that. | [8] → [8]<br>8 **ALIAS EDGE Signal-0**: C7-Signal-0 is an alias for C7-AliasBase-0. | (c) |
| 260 / c7_alias_1_d250 | What is the current value for C7-Signal-1? Reply only with the answer. | Jasper-712 | I’m sorry, but I can’t provide that. | [9] → [9]<br>9 **ALIAS EDGE Signal-1**: C7-Signal-1 is an alias for C7-AliasBase-1. | (c) |
| 267 / c7_correction_0_d250 | After all corrections, what is the current C7-Vesper-0 value? Reply only with the answer. | Onyx-911 | Mica-431 | [0, 11, 12] → [0, 12]<br>0 **SUPERSEDED semantically (Mica), still active**: The current C7-Vesper-0 value is Mica-431. The current C7-Vesper-1 value is Mica-432.<br>12 **CURRENT Onyx-911 fact**: The current C7-Vesper-0 value is Onyx-911. | (b) |
| 268 / c7_correction_1_d250 | After all corrections, what is the current C7-Vesper-1 value? Reply only with the answer. | Onyx-912 | Mica-432 | [0, 13, 14] → [0, 14]<br>0 **SUPERSEDED semantically (Mica), still active**: The current C7-Vesper-0 value is Mica-431. The current C7-Vesper-1 value is Mica-432.<br>14 **CURRENT Onyx-912 fact**: The current C7-Vesper-1 value is Onyx-912. | (b) |


## Correction mechanism: both mounted, missing lineage, stale head nearest live

1. **Harness omission at turn 2.** `scripts/grm_c7_register.py:25-40` calls `fact(...)` for both the Mica original and the prose “Correction of both original records ... Flint ... obsolete.” No `kind='supersede'` or `correction_command` is installed for turn 2. `scripts/grm_e2e_session.py:2655-2670` invokes `apply_memory_command` only for `supersede`. Original instrumentation in `/mnt/ForgeRealm/wt/grm-c7/artifacts/grm_c7/r2/cells/A-001-008/session/instrumentation.jsonl` records turn 2 as `fact`, `supersession_calls=0`. This is a missing authoritative mutation, not failed retirement of an issued target.
2. **Narrow later targets cannot repair that omission.** Turns 17/18 (`scripts/grm_c7_register.py:74-79`) issue exactly:
   ```text
   correct memory: current C7-Vesper-0 value is Flint-511 => The current C7-Vesper-0 value is Onyx-911.
   correct memory: current C7-Vesper-1 value is Flint-512 => The current C7-Vesper-1 value is Onyx-912.
   ```
   `correct_memory` (`core/graft_repository.py:1953-2001`) matches the query against active text. Node 0 contains Mica, so neither Flint query targets it. Turn 17 retires the **whole shared Flint node 1**, producing node 12 with `supersedes=[1]`. Turn 18 finds that shared node inactive; node 14 has `supersedes=[]`. This also exposes the compound-node targeting hazard: updating one entity retires a node that contains another entity. The exact Python method replay starting from the turn-16 checkpoint reproduces `[1]` then `[]`; [correction_CPU_replay.json](correction_CPU_replay.json). Native lookup and persistence were stubbed; checkpoint metadata independently agrees.
3. **State at every failing correction:** node 0 = original Mica pair, `retired=false`, `no_fold=true`, `metadata.active=true`, `supersedes=[]`; node 1 = intermediate Flint pair, `retired=true`, `no_fold=true`, inactive, absent from mounts. Node 12/14 = current Onyx bare fact, 15 tokens, active, `no_fold=false`. Full Onyx turn nodes 11/13 are 46 tokens and active. They are not split children. Node 11 has `no_fold=true`; node 13 is false in the distance-30 checkpoints and true at later distances. The per-turn projections in rows.json preserve these distinctions.
4. **L2 cannot infer an unrecorded edge.** `_revision_mount_heads` (`core/graft_arena.py:1282-1339`) traverses explicit `supersedes`; it does not derive semantic correction history from prose. Neither 12 nor 14 reaches node 0, so `[0,11,12]` and `[0,13,14]` survive resolution. This is consistent with the stored metadata. No evidence that retired node 1 leaked through filtering: `_route_cand_base` excludes retired nodes at `core/graft_arena.py:1447-1449`. `no_fold` is a folding exemption, not a retirement flag; fidelity-abort code sets it while retaining sources (`core/graft_repository.py:3273-3279`). It does not bar routing.
5. **Admission/fit:** all correction receipts identify three candidates, choose `declared_synthesis_identified_set`, and retain node 0 as plan head. Node 0 costs 61 seats; the full current turn costs 46, so `61+46=107>96`. The 15-token current fact fits: actual `[0,12]` or `[0,14]`, total 76. The full current turn is pending on a shuttle (`[[11]]`/`[[13]]`), not missing from storage. RT1 has `admission_split_child_demoted=false`, empty demoted/family IDs; it cannot repair these ordinary nodes. Margin 0.0 has `route_margin_evaluated=false`: do not infer a measured score tie. CPU replay reproduced these decisions from the recorded native ranking; it did not recompute why the latent router originally ranked 0 first.
6. **Actual mount list versus effective RS3 order:** RD1 resets the live cache before `_attempt` (`scripts/grm_rd1_replay.py:164-185`), so bootstrap seating applies. With width 96, sink 19 and live shift 115, `_rs3_seat_plan` rotates the head to the end (`core/graft_arena.py:692-714`; injection at `4392-4414`). For both correction families the effective order is **current Onyx [39,54), then original Mica [54,115), then live question beginning at 115**. Stored `cur_mounts=[0,12/14]` remains in plan order; it is not the physical injection order. Thus RS3 favors the stale ranked head. This position calculation is CPU-derived from pinned code/geometry, not a captured attention-weight measurement. The current fact's 15 bare tokens and the stale 61-token complete turn also differ in format; their separate effects are untested.

The frozen original ladder also recorded extra trips. A2 is the authorized **fixed recorded mount `_attempt` contrast**, not a new full-ladder run with the changed question. It does not reroute or execute every historical shuttle. The full historical trip arrays are retained in ROW_TABLE.md/rows.json; do not mistake a plan member pending on another trip for a member of the A2 read.

## Alias mechanism: edge-only evidence

All eight rows mount only node 8 or 9 (49 tokens), the exact Signal→AliasBase sentence. **CURRENT/base node 4 is absent** and contains both Jasper values. It is active, `retired=false`, `no_fold=true`. There is no superseded alias value in these records.

`is_identifier_binding` accepts rare identifier membership in the candidate's own text (`core/grm_admission.py:98-109`); the question contains `c7-signal-0` or `c7-signal-1`, not the AliasBase identifier. Only edge 8/9 binds. A-DEC then takes `exactly_one_identifier_decisive_rank1` (`core/grm_admission.py:338-339`), with singleton rank plan `[8]`/`[9]`. It does not follow the edge's referenced entity. RT1 fields are false/empty; no split-family demotion occurred. L2 sees no supersession edge to add a base (and is a revision filter, not alias traversal).

At distances 30/60, node 4 appears at rank 3 in `admission_ranking_before_demotion`. It loses **admission priority**, not existence: the later `[edge,0,4]` fallback treats 0 and 4 as filler and drops both for width. At distances 120/250, node 4 is absent from the six-entry returned ranking; admission still selects only the edge. Every exact ranking and dropped-filler trace is in the per-turn appendix. The CPU identifier scan also confirms that the active base never becomes an identified candidate.

There is a second constraint: **49+61=110>96**. Simply adding base 4 to the rank plan cannot seat both full payloads together under this profile. Dependency retrieval would also need a registered fitting, segmentation or evidence-carry strategy. Neither alias edge node has structural source links; the “edge” is natural-language text. **Core capability finding:** this A-DEC path treats an alias relation as decisive evidence but does not retrieve its value-bearing dependency. Core locations: `core/grm_admission.py:98-109,309-342,403-410,546-570`. No generic core fix was attempted.

The A2 oracle provides both chronological source texts and gets all eight aliases correct. That supports missing evidence as the actionable retrieval problem; it does not prove which internal mechanism produced the refusal with edge-only K/V. The frozen refusal regex (`scripts/grm_c7_common.py:178-188`) recognizes “can't recall/find/know,” but not “can't provide that,” explaining the wrong-value label. No scorer change was made or used to improve RD1's verdict.

## What C2 actually establishes

Current read of `/mnt/ForgeRealm/wt/grm-c2/artifacts/grm_c2/epochs/scout-fix-2/scores/sup.json`: profile **9/9 fresh and 9/9 restart**; [C2_comparison.json](C2_comparison.json) retains all probe IDs, answers and mounts. The overall score artifact is **RED**, with defaults 4/9 fresh and 3/9 restart. The profile result must not become a whole-campaign pass claim.

| Dimension | C2 supersession battery | C7/RD1 A2 failures |
|---|---|---|
| Lineage | Fixture `supersedes` edges explicitly installed by `_install_lived_nodes`, `scripts/grm_det1_3_gpu.py:729-741`; multi-hop A→B→C linked | Mica→Flint edge missing; Onyx-911→Flint only; Onyx-912 has no edge |
| Meaning of 9/9 | Three correction/revision queries plus two fresh-fact and four unrelated-value controls; not nine unique correction chains | Two corrected entities and two aliases, each repeated at distances 30/60/120/250 |
| Residency | Corrected-value profile rows mount a single current-value-bearing node: Orion [1], Lumen [2], Harbor [1] | Correction reads combine active Mica with Onyx; aliases mount edge only |
| Distance/pressure | Small fresh fixture bank, then replay of saved probe state; no C7 300-turn residency schedule | Scheduled distances ≥30; folds/pressure; two scored stress restarts after 100/200; every leased cell also reloads |
| Fold role | Width splitting exercises RT1; explicit lineage is available | Mounted nodes here are unfolded turn/fact nodes, not digests/split children; failed folds can set `no_fold` |
| Prompt | “What is the current Orion pin value?” / “What is the current Lumen seal value?” | “After all corrections, what is the current C7-Vesper-0 value? Reply only with the answer.”; alias asks “current value for C7-Signal-0” |
| Scoring | Expected-value hit plus no rejected-value hit; sentences/typographic hyphens can pass | Frozen normalized whole-answer equality; refusal categorization as above |

C2's installer adds revision links while initially leaving nodes active; it relies on L2's explicit lineage resolution. It does **not** validate C7's natural-language-to-command construction or promise that a prose correction triggers retirement. Both batteries already use “current X value”; that wording alone cannot explain their difference. C7's missing edge and edge-only alias admission exist at the first distance-30 failures, before either scored restart, so two restarts are not necessary causes. Full relevant metadata persists across the examined boundaries. Distance, fold effects on ranking, payload form and prompt wording were not independently varied here; no isolated causal estimate is claimed.

## Production ladder / EB1 prompt

**No C7-style abstention clause is present in the production probe template or the inspected EB1/C2 battery questions.** `scripts/grm_e2e_session.py:188-190` says exactly:

```text
Recall probe. What is the current {fact_id} value? Reply with only the value.
```

The system frame at `scripts/grm_e2e_session.py:87-92` says:

```text
You are ChatGPT. Reasoning: low. Valid channel: final.
```

C2/EB1 supersession fixture example (`tests/fixtures/supersession_battery/short_correction_long_competitor.json:32`):

```text
What is the current Orion pin value?
```

The C7 clause is instead appended by its fixture and is preserved in the historical question:

```text
Reply only with the answer; if unspecified, reply unknown.
```

Registration originally used `UNKNOWN`; r2/A0 served prompts use lowercase `unknown`. A2 removes exactly `; if unspecified, reply unknown.` and leaves the terminal period. Production has programmatic identifier-grounding/abstention branches (`scripts/grm_e2e_session.py:1212-1252`); their existence is distinct from adding an abstention instruction to the model prompt. The inspected frame/battery templates do not add such a clause.

## Recommendation and cost

**Recommend a separately registered RD3 correction-harness contrast first; queue alias retrieval separately. Neither is implemented here.**

- **Correction fix candidate, harness scope:** create an amended fixture with an explicit authoritative mutation for every correction, including the initial Mica→Flint step. Keep unrelated entities in separate authoritative records so correcting one does not retire another's only active record. Verify complete supersession paths and all stale replicas retired before serving. Do not merely reverse mounts or increase width: that leaves the missing lineage unresolved. This reuses the local e2e correction command and C2 explicit-lineage contract. Actual regenerated payloads/IDs must be receipted; it is a lifecycle treatment, not a bit-identical payload claim.
- **RD3 contrast:** preserve all eight failing correction probes and their distances. Pair A2-style baseline with the repaired-fixture condition, same model/frame/profile/budget. CPU preflight must show complete lineage, no active stale value for either entity, unrelated-value preservation, and correct behavior across reload. Then 8 baseline + 8 treatment = **16 scored memory attempts**. Do not call it fixed until the original failures clear without new wrong values. If valid lineage leaves stale values eligible/mounted, stop and report a core lifecycle defect.
- **Alias successor, core scope:** register dependency-aware evidence selection and a width-safe way to retain edge+base evidence. Existing oracle success already supplies the complete-evidence comparator. A minimal reader-only diagnostic could place the missing base source in the live prompt beside the fixed edge mount, but that would be an evidence-sufficiency contrast, not production retrieval acceptance. No alias closure algorithm is implemented or certified here.

Cost: RD2 used **0 GPU seconds**, one successful CPU replay (tool wall about 0.28 s), and source/receipt analysis. RD3 correction contrast has a **planning proxy of 16 × 22.680409907679 ≈ 362.887 s = 0.1008 GPU-h**, using RD1 amendment's amortized estimate. Fresh corrected capture, prefix/fold reconstruction and reload costs remain unmeasured; this proxy is **not a fit guarantee or launch authorization**. A separate eight-baseline/eight-complete-evidence alias diagnostic would add 16 calls and the same crude proxy. Core alias implementation engineering cost is unestimated until its representation/width contract is specified. All successor GPU work requires its own order and registered resource rail.

## Prior art

Verified local prior art: **GRM contributors (2026)** — C7/RD1 immutable receipts, chronological oracle and checkpoint-bound replay; ADM1/A-DEC identifier admission and plan-priority fit; M5/L2 explicit supersession; RS3 near-live seating; RT1 split-family demotion; C2/DET1 explicit fixture revision links. Taken: their existing contracts and unchanged CPU method bodies. Ours: the 16-row evidence join, metadata-only replay and this diagnosis; no new routing or serving algorithm. **No prior art known to me for this exact diagnostic composition.** Python AST and hashlib are existing standard-library tools, used for method extraction and byte integrity, not claimed as new techniques.

Proposed correction work reuses those local explicit-revision contracts. For general alias dependency resolution, **unverified — lead to check**: search terms “relational reference traversal,” “multi-hop retrieval,” “entity-scoped supersession,” and “dependency-aware memory admission.” No specific external paper attribution is asserted from these searches; no external literature review was performed under this receipt/CPU-only mission. No prior art known to me for the exact proposed GRM integration. These annotations also appear in the diagnostic code and ledger; no product code site was changed.

## Deviations, RED, process safety, model and effort

- **Registration:** immutable plan `orders/GRM_RD2_STALE_VALUE_DIAGNOSIS.md`; [registration.json](registration.json) and [registration.sha256](registration.sha256) written before CPU gates. Separate [LEDGER.md](LEDGER.md), [CPU_receipt.json](CPU_receipt.json), [SOURCE_EVIDENCE.md](SOURCE_EVIDENCE.md), [INPUT_SHA256.json](INPUT_SHA256.json). Historical RD1 reports were not rewritten to describe later GPU evidence.
- **Deviations/limitations:** no scope changes. Ignored original C7/C2 artifacts read from their original worktrees. No per-probe metadata dump exists in A2; the report uses each preceding checkpoint, checks relevant fields against A2 session end, and checks prefix events for relevant mutation. RS3 coordinates are reconstructed from unchanged code and pinned geometry. One initial diagnostic run failed on the nested score schema before producing outputs; corrected diagnostic access and reran successfully. That failure remains in the ledger. The original native ranking is an input, not recomputed CPU evidence.
- **RED:** all 16 A2 failures remain; aliases have incomplete evidence; correction metadata is malformed relative to intended history; reader treatment effect and proposed fixes untested. No attention/logit explanation, universal recall claim, numerical CPU replay, full C7 rerun or blind independent audit. **Not claimed fixed.** Core-capability boundary respected after diagnosis.
- **Safety:** no git, no subagents, no GPU allocation/model load, no background jobs/waits, no process kills/signals, no live services or external delivery. Foreground CPU only. All additions are confined to `artifacts/grm_rd2/`. [SOURCE_INTEGRITY.json](SOURCE_INTEGRITY.json) verifies **204 RD1-pinned local core/scripts files unchanged**, including all load-bearing code. Branch name `grm-rd1` is supplied by the order, not independently checked with git. No memory file writes.
- **Model under test (existing A2 worker receipts):** `openai/gpt-oss-20b`, revision `6cee5e81ee83917806bbde320786a8fb61efebee`, `resident_packed_mxfp4`, `tensor_cuda GptOss20B_TC`; A2 low-reasoning instruction, final channel, 32 answer tokens. **Diagnostic agent: `gpt-6-astra`, effort `high`**, recorded in `logs/grm_rd2_r1.log:6,10`.
