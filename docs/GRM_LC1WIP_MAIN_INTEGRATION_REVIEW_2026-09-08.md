# GRM-C6 — lc1-wip → main integration review — 2026-09-08

**RED: integration is not yet defensible.** The Scout prediction is supported for the first blockers: receipt portability and the baseline frame are stale. The capabilities exist, but the shipped defaults do not select the best measured profile, ordinary persistence drops capture provenance, and no current acceptance result binds that profile across restart. No merge or default change was performed.

This is a **source review + CPU suite run**, with separately labelled historical E2E evidence. The immutable plan is [the C6 order](../orders/GRM_C6_LC1WIP_MAIN_REVIEW.md). The [registration](grm_c6_review/registration.json), [SHA binding](grm_c6_review/registration.sha256), [append-only ledger](grm_c6_review/IMPLEMENTATION_LEDGER.md), [CPU dry-run](grm_c6_review/cpu_dry_run.json), and [lead commands](grm_c6_review/lead_commands.txt) accompany this review. Ancillary audit drivers and receipts live under `docs/grm_c6_review/` to respect the prohibition on edits to `core/` and `scripts/`.

## Snapshot and evidence boundaries

Read-only branch inspection recorded:

- `main`: `d2999f8ce3921f834e6214ab2f0396d892e7b758`.
- `lc1-wip` and this seat's `HEAD` (`grm-c6`): `d0ff96a1800c86e090039e182db9e7f2f7ddd7ef`.
- Live difference: **142 commits ahead, 0 behind; 247 files, 136,018 insertions, 183 deletions**. The order's 141/243 is an older snapshot; the immutable order was not corrected.
- The worktree's `core/`, `scripts/`, `tests/`, `config/`, and `cpp/` matched `lc1-wip` before testing. See [identity](grm_c6_review/branch_identity.txt), [complete branch diff](grm_c6_review/branch_diff.patch), [file list](grm_c6_review/branch_names.tsv), and [worktree comparison](grm_c6_review/worktree_vs_lc1.patch).
- “Canonical checkout” below means `/mnt/ForgeRealm/GraftRepository`, **not the `main` branch**. An artifact found there is not thereby shipped by `main` or tracked by either branch. The tracked-file inventory distinguishes these cases.

Execution continued into September 9 local time; the requested September 8 artifact name is retained. No `main` tests were run. A failure here establishes a candidate/environment problem, not automatically a regression relative to `main`. No GPU acceptance, full repository-wide pytest, blind review or mutation campaign is claimed.

## Ranked blockers and ownership

| Rank | Blocker and evidence | Necessary action | Owner / acceptance |
|---|---|---|---|
| **1** | **Receipt closure is not portable.** Both registries point into ignored campaigns absent here. RS1, RS2, RS3 tables and P2C.1 census can exit pytest successfully while running no assertions. WC1's retained aggregate has 31 old `artifacts/grm_wc1/` references whose actual bytes are under `artifacts/grm_wc1_opus/`. | Ship/reference a content-addressed evidence bundle and add an immutable relocation manifest; make required input absence a release-preflight failure. Keep historical receipts unchanged. | **Lead direct:** explanatory docs, indexes and relocation manifest. **Seat order:** portable input resolution and mandatory release preflight. Zero GPU. |
| **2** | **Active registry is the ADM2.2 persistent-frame registry, not EB1 + RS3 + RT1.** Its only resolved flags are A-DEC, probe ladder and supersession. It cannot certify capture geometry or restart. | New registration with explicit model/dialect/tokenizer/native-library hashes, EB1 frame, width, capture/seating, RT1, payload provenance, comparator and exact controls. Retain ADM2 as historical. | **Seat order**, followed by **GPU acceptance** before adoption. Changing registry meaning is not a docs-only edit. |
| **3** | **Required persistence/receipt contracts are incomplete.** Deposit puts capture fields at graft top level; `_node_manifest` and reload omit them. The route receipt drops RT1 admission-demotion/family/ranking fields. | C1-class CPU repair order: ordinary save/load and route projection fixtures, then complete durable fields. Require capture provenance before fresh/legacy/restart grading. | **Seat order**, `core/` changes outside C6. **Not claimed fixed.** Static findings corroborate Scout B2/B3. |
| **4** | **The claimed best profile is not the default.** EB1/L2/A-DEC/LSR/RT1 are ON, but pin is OFF and near-live seating OFF. Width 96 is an E2E-driver default, not the generic arena's 256. Historical WC1 scores do not certify this combination at current HEAD across restart. | Resolve explicit deployment profile; compare shipped-default and proposed-profile cells with identical fixtures, then replay the same correct IDs after process restart. Grade inherited unpinned payloads separately. | **Seat order + GPU acceptance**, then lead decides adoption. No default flip in C6. |
| **5** | **No demonstrated complete acceptance plan fits ≤0.4 GPU-h.** Candidate-only historical 13-cell worker wall is 1,629.37 s (0.45260 h), before full control/restart/legacy comparisons. Existing lease defaults are 580 s and lock waits 7,200 s. | Register an importer-based runner with a ≤285 s worker, ≤590 s outer rail, 30 s cooldown and ≤1,440 s total budget; reject any non-fit plan before execution. | **Seat order, then lead GPU runner.** Current full plan is **BLOCKED / NON-FIT**; do not increase the lease or quietly omit arms. |
| **6** | **Optional demand serving has a final-flush exception path.** Abort is set on the cache-commit forward, then swallowed; `finish()` retains the extra row because `aborted_at` is set and raises `DemandError`. The config claims unconditional serving invariance. | Add the final-flush-only CPU reproducer and repair in a C1 order; preserve completed answer and observer accounting. Keep threshold/adoption separate. | **Seat order**. Demand remains OFF; its thin earlier-frame calibration is not EB1 evidence. **Not claimed fixed.** |
| **7** | **Whole-branch scope exceeds GRM.** Qwen3.8 LC and GPT-OSS fork-mask changes, plus GLC/MoE harnesses, are in the 247-file diff. GRM's 9/9, 9/10, 13/14 cannot certify their device paths. | Lead chooses the integration scope; attach independent backend/default compatibility receipts and label optional LC/MoE REDs. | **Lead scope decision + separate backend seat/acceptance orders** where current receipts are missing. Outside C6's GRM GPU cap. |
| **8** | **Documentation and test execution framing drift.** Qwen report still says default chunk=6,208, baseline tokens are placeholders and GPU gates unrun; current source and retained G0 receipt contradict those statements. Some CPU tests compile another checkout, and one CPU-named family allocates CUDA tensors. | Update narrative/indexes using current receipts; register CPU/device/input dependencies explicitly. Preserve original order text and add amendments rather than editing historical registrations. | **Lead direct:** docs and links. **Seat order:** harness isolation/CPU-GPU classification. |

The blockers are not an instruction to add another retrieval capability. Ranks 3 and 6 are concrete correctness/provenance repairs, and rank 5 is a resource/acceptance-design gap. The historical default-on decisions remain historical facts, not a present merge certificate.

## Default and flag differences

[The exhaustive syntactic inventory](grm_c6_review/defaults_with_receipt_leads.tsv) records **3,019 changed/new/removed records** across every changed Python file: function defaults, CLI arguments/actions/choices, environment reads and uppercase assignments. It includes internal and experiment-fixture constants deliberately, with main/lc1 values and line numbers. “ABSENT” means the symbol/flag/file did not exist on main, not a default of False. [Config delta](grm_c6_review/config_delta.json) carries every field of the two added JSON registries. [Raw diff](grm_c6_review/branch_diff.patch) also covers shell exports and non-Python defaults.

The inventory is syntactic: aliases, dynamically computed policy outcomes and strings that merely look like constants are not independent adoption decisions. The following table resolves the serving changes and names the evidence that can justify them. Harness-only defaults in the appendix are not promoted into production settings; where no per-default adoption receipt was established, the appendix says so explicitly. Historical result references are read in the canonical checkout, with [availability and hashes](grm_c6_review/evidence_availability.json).

| Surface | `main` → `lc1-wip` effective value | Justification / limitation |
|---|---|---|
| `ArenaCache(ephemeral=...)`, `GRM_PERSISTENT_BOAT` | Constructor `False` → `None`, resolved **ephemeral ON**. New escape true selects persistent; absent/unknown selects ephemeral. | `orders/GRM_EB1_EPHEMERAL_BOAT_FRAME.md`; `artifacts/grm_eb1/grm_eb1_summary.json`. EB1 is the specification decision, not a uniformly positive recall result: 5/9 supersession, 9/10 census, 13/14 long session; three supersession regressions against P2C arm 1. |
| `revision_resolution`, `GRM_SUP_RESOLVE`, E2E `--[no-]sup-resolve` | Constructor `False` → `None` → **ON**. Explicit caller wins; absent env ON, false/unknown OFF. CLI new, default None. | `artifacts/grm_supersession/l2_default_on_20260830T165648866160Z_4123111/gate_result.json`, hash `f2d82d10…`, verified; `orders/GRM_SUP_L2_DEFAULT_ON.md`. Frozen persistent-frame evidence. |
| `decisive_admission`, `GRM_ADM_DECISIVE`, `--[no-]adm-decisive` | Absent/fixed k admission → **ON** via None. False/unknown env OFF. | ADM2.2 `gate_result.json`, hash `e2ff8073…`, verified. Frozen A-DEC rule `c304609f…`, margin **0.1385774091529802** in `core/grm_admission.py:23`. Not fitted from current serving or revalidated for EB1 by that receipt. |
| `GRM_LSR_FIXES`; explicit `lsr_fixes=None` | New **ON**, explicit false tokens OFF, unknown/absent ON. Enables fit honesty, split/descent and associated serving repair. | P2A/B/C immutable orders; `artifacts/lsr_p2c/lsr_p2c_g2_summary_d5efcb7969279860.json`. P2C.1's fit arithmetic is reconstructed counterfactual evidence, not observed live fit transitions. |
| `GRM_RT1_RULE` | New isolated RT1 override; otherwise follows LSR resolver. Only explicit false tokens disable. | Restored `artifacts/grm_rt1/grm_rt1_g2_results.json`: with LSR ON and RS3 pair ON, isolated rule OFF **8/9**, ON **9/9**, no recorded regressions. Use RT1-only switch for this control; LSR OFF changes splitting too. |
| `GRM_CAPTURE_PIN`, `deposit(capture_pin=None)`, capture overrides | New flag/API, effective **off**; accepted pin values `mount`, `live`. Ambient unknown OFF; invalid explicit pin raises. | `artifacts/grm_rs3/grm_rs3_results.json` and `grm_rs3_part4.json` justify testing the live pin; no default-ON adoption receipt established. New payload geometry cannot retroactively repair already captured tensors. |
| `GRM_SEAT_NEAR_LIVE`, seating override | New **OFF**, explicit true token ON, absent/unknown OFF. | Same RS3 receipts, RT1 A/B and WC1. Pair ON is the best measured profile; current constructor/flag default does not adopt it. |
| `GRM_DEMAND_NGH`, `demand_ngh=None` | New **OFF**; explicit caller wins, unknown OFF. Supported observer limited to standard GPT-OSS full-attention layers. | `orders/GRM_SC1_DEMAND_LOOP_NGH.md`, `artifacts/grm_sc1_2/results.json`: earlier-frame eligible misses/recoveries. No EB1 demand adoption receipt. |
| `GRM_DEMAND_EARLY_ABORT` | New **ON conditional on demand ON**, false tokens disable; absent/unknown ON. | `artifacts/grm_sc2/sc2_results_13b1d3a5cdea7ae9.json`; seven rerun early-abort pairs, not ten fresh pairs. Final-flush boundary remains a static RED. |
| Demand threshold and trip cap | No demand config → carried **0.3380523274342219**, strictly below, one trip/turn. Candidate **0.3181141105790933** stored but **not adopted**. | Config records two-turn served-control envelope and hashes into DET1 race. `adopted_by` is absent. SC2 candidate is a separate report; neither threshold has an EB1 adoption certificate. |
| `GRM_REGISTERED_BASELINE_REGISTRY` | New override; default `config/grm_live_registered_baselines.json`. | `scripts/grm_det1_baseline_registry.py:21`. New active registry names ADM2.2, historical SUP-L2 retained. This is a frozen comparator input, not a runtime feature toggle. |
| Route receipt persistence | Main has no canonical per-turn route record → `GRMRuntime.chat` always builds one; repository ledger if configured, otherwise `info['route_receipt_record']` with sink reason. New `configure_route_receipt_ledger(..., enabled=True)`. | `orders/GRM_LSR_P2B_ROUTE_RECEIPT_PERSISTENCE.md`, CPU P2B fixtures. This is an unflagged contract change. Projection omits RT1 provenance; unavailable state hash/write failure is explicitly representable and must not be mistaken for complete durable evidence. |
| Width guard defaults | New split-at-deposit/runtime guard; boundary `section`, budget derived from arena width minus recency reserve. New `split_oversized(boundary='section', budget=None)` repair API. | P2C order/summary and 39 CPU split/descent tests. No automatic migration of all inherited payloads or pin provenance is established. |
| GPT-OSS fork mask | New `allowed_mask=None` and optional `_det1_fork_allowed_mask`; absent leaves ordinary path. Consumed once; nonstandard attention rejects the mask. | DET1.3/DET1.4 snapshot/fork orders; CPU snapshot and mask/backend tests. A diagnostic opt-in extension, not a new standard-attention default or model-quality receipt. |
| Qwen context, prefill and KV block | New `max_context=4096`, `prefill_chunk=64`, `kv_block_rows=256`; previously fixed default 4K path. | LC1 report and F4/F5 orders. CPU 38-test battery; retained `logs/lc1_g0_merged_final.log` reports 3/3 default token arrays matching, historical scope only. |
| Qwen `kv_int8`, `kv_host`, `force_tiled_attention`, `cache_read_only` | New **False** defaults (last on load API); default uses legacy attention + concat KV; LC/INT8/host/tiled selects preallocation. CLI `--kv-int8`, `--kv-host` new store_true, mutually exclusive. | F4 preserves default compatibility; F5 addresses INT8 readback. Optional long-context/device paths require their own device gates. GRM acceptance cannot cover them. |
| Qwen lm-head row chunk | API `31040` constant default → `None` → **31040 for concat/default; 6208 for preallocated LC**. CLI env override retained, otherwise None. | F4 order and historical merged-final G0 receipt (loaded chunk=31040, legacy/concat). Report's blanket “new default 6208” is stale. |
| Qwen budget and `--force-alloc` | Budget helper `context_tokens=128` → `DEFAULT_MAX_CONTEXT=4096`. New projected whole-device check rejects >12,000 MiB; new CLI `--force-alloc` defaults **False**, explicit diagnostic bypass. | LC1/F3/F4 evidence; fixed 1,185 MiB overhead is receipt-calibrated, not a formal peak guarantee. `main` contains the F3 order but not the implementation. Opt-in forced allocation is not release acceptance. |
| Frozen GPU arena gates | E4, GQA and NC17 add explicit `ephemeral=False`. | Their source comments say they measure one persistent cache. These pins preserve historical meaning; they do not test EB1 production. |

**Unchanged values that must not be reported as flips:** generic arena width **256**, E2E width **96**, recency mounts **2**, driver top-k **3**, probe ladder ON, route `auto`, optional ragged-CUDA routing, and the driver's forced storage-bits/CUDA-route settings are not new branch-default changes. The existing default 31,040 Qwen row chunk and sigmoid output gate remain the ordinary default. None of these facts makes the full best-measured profile an adopted default.

## Stale references and release-input closure

The [reference inventory](grm_c6_review/references_inventory.tsv) records 1,556 path-like occurrences from changed code, docs, orders and config, distinguishing local existence from tracking in each branch and availability in the canonical checkout. It is a conservative candidate list: line suffixes, illustrative paths, output paths, directory references and template strings are not all broken input dependencies. The confirmed operational cases are below.

| Reference / consumer | Observed state and consequence | Repair boundary |
|---|---|---|
| Active registry's ADM2.2 gate, scorecard, stage markers and battery | Absent in this worktree; present in canonical checkout with **all five active receipt hashes matching**. Both JSON registries are absent from `main`. New code merged with tracked files alone will still lack these ignored inputs. | Export a minimal hashed bundle/locator contract. Do not “fix” a missing historical input by generating a fresh model output under current defaults. |
| Demand threshold and SC2 calibration provenance | DET1 threshold/race and SC2 inputs are ignored campaign artifacts. Config's calibration is earlier-frame and thin. CPU demand assertions fail when receipts are absent. | Portable provenance + new-frame registration before adopting any threshold. |
| `grm_wc1_opus/grm_wc1_results.json` nested `artifacts/grm_wc1/...` paths | **31/31 candidate relocations hash-match**, including registration and score/fixture receipts. See [resolution manifest](grm_c6_review/wc1_path_resolution.json). The old literal paths generally do not exist. | Lead can add a relocation index/docs references while preserving old receipt bytes. Operational resolver changes require an order. |
| `tests/test_grm_native_runtime.py:16`, `build_native` | Hard-coded canonical `ROOT` compiles `cpp/grm_runtime.cpp` outside the candidate. Five C++ files currently hash-match the candidate; see [identity check](grm_c6_review/external_cpp_identity.json). A future source divergence could yield a false candidate pass. Lifecycle tests import this helper. | Seat order for explicit candidate-root input. Current results qualified by verified byte identity. |
| `tests/test_grm_lsr_p2b_route_receipt.py:348` | Searches worktree **then canonical checkout** for lived DET1/P1 receipts. Its 14 passes here include that external-data fallback; they do not demonstrate a self-contained worktree. | Register the chosen input root and full receipt closure. Do not remove the assertion or count missing data as success. |
| Original RT1 registration and deleted worktrees | RT1's original `afaa65bb…` registration is declared destroyed. The replacement `58c70ea0…` explicitly says `RESTORATION_STANDIN`. Current G2 binds the replacement correctly; the restoration record's historical original hash necessarily cannot match it. Orders also name deleted LSR-P2B/RT1/WC1 worktrees. | Retain the distinction. Use `GRM_RT1_1_RESTORE_RECEIPTS.md` and restoration manifest; add order-location amendments/indexes, never rewrite immutable orders or pretend original bytes survive. |
| Scout links to EB1, RS3, RT1, WC1, SC1/SC2, LSR receipts | Available in canonical ignored storage, absent here and not shipped in branch trees. Relative documentation links alone are not an artifact distribution plan. | Lead docs/receipt bundle. Files added on lc1-wip, such as `core/grm_frame.py`, are not stale simply because absent on pre-merge main: the full diff carries them. |
| `docs/QWEN38_LC1_REPORT.md` | Still calls `BASELINE_TOKEN_IDS` placeholders and all GPU gates unrun; source has real token arrays. Retained merged-final G0 reports PASS for code/chat/factual. It also overgeneralizes 6,208-row default and has drifted numeric source-line references. | Lead update with scope/date/hash of retained receipt. Do not extrapolate default G0 to LC/INT8 recall or every backend feature. |
| `scripts/grm_wc1_g1_pytest.py` | A **transcriber of historical logs**, not a runner of current pytest. Hard-coded comparison points into canonical artifacts. Its “main checkout” text does not mean the main branch. | Cite as history only. Current counts are below. |
| Old GPU lease wrappers | WC1/EB1 580 s defaults, SC1.2 520 s lease and 500 s worker, 7,200 s wait; shared lease permits up to 590 s. No mandatory 30 s cooldown in that helper. | Explicit bounded successor required. Do not run old commands unchanged under C6 acceptance. |

Historical fingerprints of `core/graft_arena.py` or `scripts/grm_e2e_session.py` differing from current source are expected version drift, not evidence that a frozen receipt was corrupted. The raw availability inventory also includes non-file `path` fields; its missing-entry count is not a count of confirmed blockers.

## Battery frame and registry map

| Battery family | Expected frame / comparator | What branches ship and integration meaning |
|---|---|---|
| SUP-L2 default-on and ADM2 default-on | Persistent live window; frozen byte/transcript and ordered mount comparators. ADM2 active flags A-DEC/ladders/L2 ON. | Main has older source/receipts, no live registry. lc1 ships ADM2 registry but defaults to EB1. Reproduce historical gates only under their original explicit frame. |
| DET1 baseline registry, snapshots, source-auth, achieved/census/envelope | Earlier lived DET1 campaign plus ADM2 registry. `certified_34_turn`: same-model exact answer bytes + ordered authoritative mounts. `supersession_battery_on_gpt_oss`: **MiniCPM3 producer**, GPT-OSS consumer alias; expected/rejected values + ordered fixture-local physical IDs, not cross-model exact answer bytes. | lc1 includes loaders/tests and frozen registry, mostly lacks tracked campaigns. Replacing the registry with EB1 rows without a new schema/registration would invalidate the earlier comparator. Synthetic source-auth passes do not restore absent lived receipts. |
| LSR P2A/P2B/P2C/P2C.1 | Lived DET1/P1 snapshots; P2C arms compare fixes with fixed runtime frame. P2C census explicitly distinguishes EB1 from persistent and disclaims persistent Arm-0 reproduction under a changed frame. | Code present on lc1; P2B fallback can read canonical inputs. P2C.1 module is skipped without frozen census. P2C.1 reconstructed indices are not live transition observations. |
| E2E/EB1 | Ephemeral turn opening, repository recency mounts charged, no old live chat tokens. EB1 historical 5/9, 9/10, 13/14. | Generic source defaults now EB1. Original no-regression supersession result is RED; “frame obeyed” cannot replace recall acceptance. |
| RS1/RS2 | RS1 combines **ephemeral A0** receipts with a **persistent live-ID** source. RS2 adds geometry arms/amendments. | Earlier-frame inputs are deliberate controls, not production defaults. Module-level missing-registration skips hide all their tests here. |
| RS3/RS4 | Registered capture and seat arms; RS4 matched readout ceiling and row partitioning. | Flags OFF by default. RS3 core geometry tests pass; evidence-table tests need ignored registrations/results. RS4 mass comparison is not general logit equivalence. |
| RT1 | EB1, capture live and near-live ON, LSR ON in both arms; only RT1 rule differs. | ON default exists, but RS3 pair remains OFF. Restored 8/9→9/9 sup A/B is historical; no current complete restart acceptance. |
| WC1 | Derived width-specific runtime frames, fixed EB1/RS3 levers, four sup fixtures + four census shards + five long-history shards. | Width 96 historical **9/9, 9/10, 13/14**; long-history aggregate `correct=4,total=4` covers only distant probes. Use `all_probes`, not 4/4, for full-session bar. |
| SC1/SC1.1/SC1.2/SC2 | Carried DET1 threshold, earlier-frame miss/recovery pairs and SC2 provenance split; glyph tests also synthetic. | Demand OFF; early abort conditional ON; candidate threshold not adopted. Requires EB1 recalibration and boundary repair before a flip. |
| Importance/S4/three-pass/probe-ladder/runtime/native | Mostly synthetic/unit contracts, C++ fake/native host stores; some mixed CUDA allocation. | CPU success says nothing about 9/9/9/10/13/14 model recall. Native source compiled from canonical is qualified above. |
| E4/GQA/NC17 and standalone DeepSeek arena/descent/corpus gates | Persistent-frame GPU checks and dialect-specific model gates. Some files execute at import rather than define pytest tests. | Pins are explicit on lc1. Excluded from CPU execution; rc=5/no collected tests cannot certify their GPU behavior. |
| Qwen LC and GPT-OSS backend/fork-mask | Separate model/backend compatibility, default token arrays and optional LC/INT8/host gates. | CPU tests here plus historical default G0; require independent device evidence for changed device paths. Not certified by a GRM-only acceptance profile. |

## Escapes, skips and current CPU runs

The original assertions were executed with CUDA hidden, single-thread numerical-library settings, ambient `GRM_*` variables removed to exercise shipped defaults, Python bytecode/cache output disabled and pytest plugin autoload disabled. No marks, assertions, thresholds or source files were edited. The test runner calls pytest on original modules; it does not copy harness implementations. Each cell has a registration/source/driver/log hash and JUnit XML. Native/lifecycle suites are split into six-function cells (parametrization can produce more than six tests), with a 55 s per-cell rail.

Confirmed escapes:

- RS1 registration absent: **34 skipped**; RS2 registration absent: **32 skipped**; RS3 tables registration absent: **24 skipped**; P2C.1 frozen census absent: **46 skipped**. Each whole module returns rc=0 while passing **zero** tests.
- RS4 baseline/gates: three registration and two results gates skipped (**5**), and two RS3-registration reads fail. RS4 row-split: **5** registration-bound skips.
- DET1.11 envelope lineage: **2** explicit missing-shard/missing-analysis skips; absent envelope lineage also causes assertions/IndexError rather than clean skips.
- P2B route receipt gate has an explicit missing-input skip, but here executes via canonical fallback.
- Static scan found **no pytest xfail declarations** in repository test Python files. No xfail was added. Skips must still count as incomplete required release coverage.
- `test_grm_importance_telemetry.py` is mixed CPU/GPU: five tests construct MLA norm tensors and report exactly `RuntimeError: cudaMalloc failed: no CUDA-capable device is detected`. Seven other tests pass. This is an unavailable-device result, not a GRM logic regression and not 12 CPU passes.
- Router baseline stopped at the registered 55 s rail, rc=124, no complete JUnit counts. Partial progress dots are not promoted to passes. It was not retried with a longer timeout.

**Current run totals: 1328 passed, 52 failed, 20 errors, 148 skipped; 1 timed-out battery without complete counts; 0 xfailed / 0 xpassed observed.** 88 registered cells across 54 files, all attempted. Completed XML contains 1548 unique test cases; no duplicates across shards. These totals include the five unavailable-device failures, not GPU work. [Machine summary](grm_c6_review/cpu_summary.json).

| Original test file (`tests/`) | Passed | Failed | Errors | Skipped | Status |
|---|---:|---:|---:|---:|---|
| [test_deepseek_grm_hooks_static.py](grm_c6_review/test_deepseek_grm_hooks_static.log) | 4 | 0 | 0 | 0 | PASS |
| [test_gpt_oss20b_e14_backend_fix.py](grm_c6_review/test_gpt_oss20b_e14_backend_fix.log) | 3 | 0 | 0 | 0 | PASS |
| [test_grm_adm1_3_dual_frame.py](grm_c6_review/test_grm_adm1_3_dual_frame.log) | 9 | 0 | 0 | 0 | PASS |
| [test_grm_adm1_probe_adjudication.py](grm_c6_review/test_grm_adm1_probe_adjudication.log) | 1 | 1 | 0 | 0 | RED |
| [test_grm_adm1_snapshot.py](grm_c6_review/test_grm_adm1_snapshot.log) | 1 | 0 | 0 | 0 | PASS |
| [test_grm_admission.py](grm_c6_review/test_grm_admission.log) | 11 | 0 | 0 | 0 | PASS |
| [test_grm_arena_step_rollback.py](grm_c6_review/test_grm_arena_step_rollback.log) | 2 | 0 | 0 | 0 | PASS |
| [test_grm_det1_11_achieved.py](grm_c6_review/test_grm_det1_11_achieved.log) | 41 | 0 | 0 | 0 | PASS |
| [test_grm_det1_11_census.py](grm_c6_review/test_grm_det1_11_census.log) | 8 | 10 | 0 | 0 | RED |
| [test_grm_det1_11_envelope_lineage.py](grm_c6_review/test_grm_det1_11_envelope_lineage.log) | 19 | 4 | 0 | 2 | RED |
| [test_grm_det1_3_snapshot.py](grm_c6_review/test_grm_det1_3_snapshot.log) | 39 | 0 | 0 | 0 | PASS |
| [test_grm_det1_5_campaign.py](grm_c6_review/test_grm_det1_5_campaign.log) | 63 | 0 | 0 | 0 | PASS |
| [test_grm_det1_7_registry.py](grm_c6_review/test_grm_det1_7_registry.log) | 23 | 0 | 0 | 0 | PASS |
| [test_grm_det1_7_source_auth.py](grm_c6_review/test_grm_det1_7_source_auth.log) | 6 | 0 | 0 | 0 | PASS |
| [test_grm_det1_8_source_auth.py](grm_c6_review/test_grm_det1_8_source_auth.log) | 6 | 0 | 0 | 0 | PASS |
| [test_grm_det1_9_findings.py](grm_c6_review/test_grm_det1_9_findings.log) | 5 | 0 | 0 | 0 | PASS |
| [test_grm_det1_9_source_auth.py](grm_c6_review/test_grm_det1_9_source_auth.log) | 7 | 0 | 0 | 0 | PASS |
| [test_grm_det1_baseline_registry.py](grm_c6_review/test_grm_det1_baseline_registry.log) | 1 | 7 | 0 | 0 | RED |
| [test_grm_eb1_ephemeral_frame.py](grm_c6_review/test_grm_eb1_ephemeral_frame.log) | 44 | 0 | 0 | 0 | PASS |
| [test_grm_ensure_h_error.py](grm_c6_review/test_grm_ensure_h_error.log) | 3 | 0 | 0 | 0 | PASS |
| [test_grm_fold_recovered_guard.py](grm_c6_review/test_grm_fold_recovered_guard.log) | 11 | 0 | 0 | 0 | PASS |
| [test_grm_importance_counterfactual.py](grm_c6_review/test_grm_importance_counterfactual.log) | 15 | 0 | 0 | 0 | PASS |
| [test_grm_importance_g1g2.py](grm_c6_review/test_grm_importance_g1g2.log) | 56 | 0 | 0 | 0 | PASS |
| [test_grm_importance_salience.py](grm_c6_review/test_grm_importance_salience.log) | 29 | 0 | 0 | 0 | PASS |
| [test_grm_importance_telemetry.py](grm_c6_review/test_grm_importance_telemetry.log) | 7 | 5 | 0 | 0 | RED |
| [test_grm_lsr_p2a_fit_honesty.py](grm_c6_review/test_grm_lsr_p2a_fit_honesty.log) | 20 | 0 | 0 | 0 | PASS |
| [test_grm_lsr_p2a_probe_path.py](grm_c6_review/test_grm_lsr_p2a_probe_path.log) | 9 | 0 | 0 | 0 | PASS |
| [test_grm_lsr_p2b_e2e_fixture.py](grm_c6_review/test_grm_lsr_p2b_e2e_fixture.log) | 5 | 0 | 0 | 0 | PASS |
| [test_grm_lsr_p2b_route_receipt.py](grm_c6_review/test_grm_lsr_p2b_route_receipt.log) | 14 | 0 | 0 | 0 | PASS |
| [test_grm_lsr_p2c_split_descent.py](grm_c6_review/test_grm_lsr_p2c_split_descent.log) | 39 | 0 | 0 | 0 | PASS |
| [test_grm_native_runtime.py](grm_c6_review/cpu_summary.json) | 121 | 0 | 0 | 0 | PASS (19 cells) |
| [test_grm_probe_ladder.py](grm_c6_review/test_grm_probe_ladder.log) | 15 | 0 | 0 | 0 | PASS |
| [test_grm_route_seams_gate.py](grm_c6_review/test_grm_route_seams_gate.log) | 2 | 0 | 0 | 0 | PASS |
| [test_grm_router_baseline.py](grm_c6_review/test_grm_router_baseline.log) | — | — | — | — | TIMEOUT 55s; counts incomplete |
| [test_grm_rs1_read_strength.py](grm_c6_review/test_grm_rs1_read_strength.log) | 0 | 0 | 0 | 34 | INCOMPLETE: skipped |
| [test_grm_rs2_mount_read.py](grm_c6_review/test_grm_rs2_mount_read.log) | 0 | 0 | 0 | 32 | INCOMPLETE: skipped |
| [test_grm_rs3_capture_pin_seat.py](grm_c6_review/test_grm_rs3_capture_pin_seat.log) | 64 | 0 | 0 | 0 | PASS |
| [test_grm_rs3_tables.py](grm_c6_review/test_grm_rs3_tables.log) | 0 | 0 | 0 | 24 | INCOMPLETE: skipped |
| [test_grm_rs4_baseline_and_gates.py](grm_c6_review/test_grm_rs4_baseline_and_gates.log) | 18 | 2 | 0 | 5 | RED |
| [test_grm_rs4_row_split.py](grm_c6_review/test_grm_rs4_row_split.log) | 27 | 0 | 0 | 5 | INCOMPLETE: skipped |
| [test_grm_rt1_split_child_routing.py](grm_c6_review/test_grm_rt1_split_child_routing.log) | 38 | 0 | 0 | 0 | PASS |
| [test_grm_runtime_lifecycle.py](grm_c6_review/cpu_summary.json) | 101 | 0 | 0 | 0 | PASS (17 cells) |
| [test_grm_s4_demotion.py](grm_c6_review/test_grm_s4_demotion.log) | 16 | 0 | 0 | 0 | PASS |
| [test_grm_s4_fold_order.py](grm_c6_review/test_grm_s4_fold_order.log) | 15 | 0 | 0 | 0 | PASS |
| [test_grm_s4_ledger.py](grm_c6_review/test_grm_s4_ledger.log) | 23 | 0 | 0 | 0 | PASS |
| [test_grm_sc1_1_grounding_glyphs.py](grm_c6_review/test_grm_sc1_1_grounding_glyphs.log) | 164 | 0 | 0 | 0 | PASS |
| [test_grm_sc1_2_e2e_pairs.py](grm_c6_review/test_grm_sc1_2_e2e_pairs.log) | 15 | 12 | 20 | 0 | RED |
| [test_grm_sc1_demand_loop.py](grm_c6_review/test_grm_sc1_demand_loop.log) | 50 | 1 | 0 | 0 | RED |
| [test_grm_sc2_calibration_early_abort.py](grm_c6_review/test_grm_sc2_calibration_early_abort.log) | 53 | 10 | 0 | 0 | RED |
| [test_grm_supersession_battery.py](grm_c6_review/test_grm_supersession_battery.log) | 21 | 0 | 0 | 0 | PASS |
| [test_grm_three_pass.py](grm_c6_review/test_grm_three_pass.log) | 7 | 0 | 0 | 0 | PASS |
| [test_grm_wc1_sweep.py](grm_c6_review/test_grm_wc1_sweep.log) | 39 | 0 | 0 | 0 | PASS |
| [test_lsr_p2c_1_census.py](grm_c6_review/test_lsr_p2c_1_census.log) | 0 | 0 | 0 | 46 | INCOMPLETE: skipped |
| [test_qwen38_cpu.py](grm_c6_review/test_qwen38_cpu.log) | 38 | 0 | 0 | 0 | PASS |

Exact representative REDs (full failures are in per-cell logs/XML):

```text
FileNotFoundError: [Errno 2] No such file or directory:
'/mnt/ForgeRealm/wt/grm-c6/artifacts/grm_det1/run_20260831T160525Z_2/registration_62cb6c09cbec211d.json'
AssertionError: expected an append-only series of envelopes
IndexError: list index out of range
RuntimeError: cudaMalloc failed: no CUDA-capable device is detected
```

These failures were not suppressed or fixed by transplanting artifacts. The running counts belong to this worktree at the pinned source state, not to a copied historical WC1 G1 table.

## GPU acceptance: bounded plan and explicit non-fit

**0 GPU used by C6.** [The detailed plan](grm_c6_review/gpu_acceptance_plan.json) fingerprints all timing receipts and enumerates the existing candidate cells. Historical timing is measured wrapper/worker wall, **not a GPU-busy-time measurement or a proof that no faster harness can exist**. For lease budgeting it is a conservative basis. The 0.4 h cap remains 1,440 s; it was not expanded to the Scout's broader rank-2 estimate.

| Existing width-96 candidate cell, in dependency order | Historical worker wall s | Planning estimate s | 285 s worker fit |
|---|---:|---:|---|
| Sup correction_then_restatement | 89.84 | 90 | observed fit |
| Sup fresh_fact_controls | 56.03 | 60 | observed fit |
| Sup multi_hop_a_b_c | 50.02 | 55 | observed fit |
| Sup short_correction_long_competitor | 47.61 | 50 | observed fit |
| Census e2e-1 | 69.62 | 70 | observed fit |
| Census e2e-2 | 60.00 | 65 | observed fit |
| Census e2e-3 | 100.90 | 105 | observed fit |
| Census e2e-4 | 73.28 | 75 | observed fit |
| Long history lh-1, through turn 34 | 232.89 | 235 | observed fit, limited slack |
| Long history lh-2, through turn 52 | 246.93 | 250 | observed fit, limited slack |
| Long history lh-3, through turn 71 | 210.81 | 215 | observed fit |
| Long history lh-4, through turn 91 | 237.63 | 240 | observed fit, limited slack |
| Long history lh-5, through turn 104 | 153.80 | 155 | observed fit |
| **Candidate-only total** | **1,629.37** | **1,665** | **campaign NON-FIT under 1,440 s** |

The actual acceptance matrix must include **both** shipped-default control and proposed width96/EB1/live-pin/near-live/RT1 profile on freshly captured inputs, plus post-process-restart comparisons for every originally correct probe. An inherited unpinned repository is a separate input condition, graded without silently recapturing it. Store and verify capture metadata, model/dialect/native library and payload hashes. Demand OFF in all these cells. Strict profile bars: **9/9 supersession, 9/10 census, 13/14 full long history**, no loss of original-correct controls, stable results and provenance across restart. Preserve per-ID failures; aggregate totals cannot conceal swapped regressions.

Current shard restarts are evidence of continuation through checkpoints, **not a repeated before/after answer comparison for every probe**. The long-history 34-turn prefix resembles the census, but reusing it to remove a battery or changing process-load schedules would be a new harness/registration decision. C6 does not assume that optimization or invent timings for missing restart/control/inherited cells. Those additional cells are **BLOCKED: estimate/fit not established**, with equal standing and no right to consume an unregistered extended budget.

Consequently no complete GPU acceptance execution is authorized by this review's command file. It contains exact CPU preflight/replay commands and a dependency-ordered blocking instruction for the lead. A follow-up seat must make a reviewable ≤1,440 s plan or report NON-FIT; it must not stretch leases, clear the shared lock, use a superseded frame to obtain easy passes, or label candidate-only reproduction “acceptance.” The model profile cannot be adopted merely because the historical individual cells each fit 285 s.

## Prior art

This order implements no retrieval, selection, proof or optimization algorithm. Audit code uses the **Python Software Foundation standard library (`ast`, `hashlib`, `subprocess`, `xml.etree`; Python 3.12 first released 2023; installed 3.12.3, 2026 audit)** and **pytest team, pytest 9.0.2 JUnit reporting (installed 2026 audit snapshot)**; the house immutable-registration/ledger/receipt method is reused. Our contribution is the branch-specific inventory, actual test receipts, relocation verification and integration judgement. Comments at both audit code sites and the ledger state this provenance. [Executed driver snapshots](grm_c6_review/executed_driver_snapshots.json) retain the exact pre-annotation bytes bound by receipts; later edits add attribution comments only.

For the follow-up contracts and acceptance protocol, the known conceptual antecedents are **Mohan et al., ARIES (1992)** for recovery/provenance, **Denning, working sets (1968)** for bounded residency, and the house **EB1/RS3/RT1/WC1 orders (2026)** for the exact frame, fixtures and split controls. External bibliography is **unverified — lead to check**; those author/title/year strings are the search terms, carried from Scout Part E/P2. No claim that a GRM control protocol or file manifest is a novel algorithm. No new optimization is proposed to force acceptance into the budget.

## Deviations, RED, process safety, model and effort

- **Deviations:** actual branch size differs from the immutable brief; recorded without changing it. Audit drivers/receipts are added under `docs/` because this order explicitly forbids `scripts/` edits. Native tests use canonical C++ sources; current equality is verified, isolation is not claimed. One selected CPU-labelled family attempted unavailable CUDA allocation; it remained RED, no GPU work ran. Router hit the registered CPU rail; no extension/retry. No complete ≤0.4 h GPU matrix is demonstrated; reported NON-FIT rather than supplied as a fictional executable acceptance command.
- **RED:** merge readiness, required artifact coverage, capture/route provenance, conditional demand final-flush contract, and complete budgeted GPU acceptance. **Not claimed fixed.** Passing unit tests do not clear those gates. The lead's blind verification remains outstanding; C6 did not dispatch its own verifier.
- **Process safety:** no subagents, GPU/model runs, background shell jobs, service changes, lock operations or git mutations. Read-only git inspection was explicitly permitted by C6 despite the common no-git boilerplate. Only C6's own pytest child was terminated on its registered timeout. No unrelated process was killed/signalled, no shared lock cleared. Original orders, registrations, defaults, kernels, tests and scripts were preserved.
- **Model / effort:** GPT-6 (Codex), high reasoning effort. The exact deployment/model suffix is not exposed to this seat, so no more specific API model id is asserted.
