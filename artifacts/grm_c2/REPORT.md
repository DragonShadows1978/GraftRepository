# GRM-C2 — BLOCKED at the budget rail; default adoption unmeasured

**Registry and CPU preparation delivered. GPU campaign is NON_FIT as planned:
52 cells estimate3,140 seconds (0.8722 GPU-h), exceeding2,880 seconds (0.8 GPU-h).
No GPU cells ran. Profile stays explicitly opt-in; no default adoption or EB1
SCOUT-FIX-1 E2E validation is claimed.** Archived timing evidence does not justify
reducing those estimates. This is a blocked handoff, not a runnable GPU release.

## 1. Registry, pins, cells, commands

Create-only registry: `config/grm_eb1_profile_registered.json`.
SHA256: `cfee19259a071d1314dde602978d8f83880f7fd3cca04a9d41b26ed42d4fb74e`.
Immutable registration: `artifacts/grm_c2/registration.json`.
SHA256: `8a3a27302c41f2980f883666657f7d871c5ad01d195953e017804717d3644eb0`.
Immutable plan: `orders/GRM_C2_EB1_PROFILE_REGISTRATION.md`; no order edit.
The existing `config/grm_live_registered_baselines.json` is byte-unchanged.

Target model: `openai/gpt-oss-20b`, revision
`6cee5e81ee83917806bbde320786a8fb61efebee`, pinned local snapshot. Both arms
use the imported EB1 Harmony frame, model prompt effort **low**,19 sink tokens,
32 generated-token budget, single turn pipeline, topk3, max_trips1, live_turns2,
max_live4096,8-bit graft storage, demand OFF, and the common frozen evaluation
backend/routing settings. Tokenizer was loaded on CPU to verify sink token IDs.
This is the shipped profile levers within the registered evaluation harness,
not a claim that every generic runtime constructor parameter has been tested.

| Selected lever | Explicit profile | Shipped profile lever defaults |
|---|---:|---:|
| arena width |96|256|
| capture pin |live|off, observed unpinned shift receipted|
| seat near live |true|false|
| RT1 split-child rule |true|true (already default ON)|
| live shift |115|275|

The shipped ArenaCache default is256, whereas EB1/WC1's evaluation width was96.
The registered default prediction5/9,9/10,13/14 is retained as instructed;
reproduction at256 is **not assumed**. All unrelated evaluation settings stay
pinned to the recorded frame. Loading a profile is additive and requires explicit
selection; no production caller was changed to select it automatically.

Flag-pin tests, in `tests/test_grm_c2_profile.py`:
`test_profile_off_unless_explicitly_selected`,
`test_profile_loading_sets_exact_registered_flags_and_nothing_else`,
`test_registry_matches_shipped_defaults_and_exact_frame`.
Additional capture/payload gates live in `tests/test_grm_c2_capture.py` and
`tests/test_grm_c2_payload.py`.

| Registered cells, both sides combined | Count | Estimate per cell | Sum |
|---|---:|---:|---:|
| fresh supersession, four families per side |8|45s|360s|
| fresh census, stops8/17/25/34 |8|55s|440s|
| fresh long-history, stops8/17/25/34 |8|55s|440s|
| long-history continuation, stops52/71/91/104 |8|100s|800s|
| restart supersession, groups of at most4 probes |6|55s|330s|
| restart census, groups of at most4 probes |6|55s|330s|
| restart long-history, groups of at most4 probes |8|55s|440s|
| **Total** |**52**||**3,140s**|

Every exact cell ID, probe grouping, turn range and dependency is in
`registration.json` and `dry_run_A6.json`. Cells alternate sides at each
corresponding stage. Estimates are planning estimates, not new GPU measurements.
Per-worker rail285s, actual subprocess timeout280s, outer cap590s, lock wait240s,
foreground cooldown30s. No cell is estimated above285s; the **total campaign**
is non-fit. No retries or extended lease are implemented.

Exact lead commands: `artifacts/grm_c2/lead_commands.txt`, in dependency order.
Latest dry run: `artifacts/grm_c2/dry_run_A6.json`.
Latest blocked report: `artifacts/grm_c2/blocked_report_final.json`.
CPU reproduction: `CUDA_VISIBLE_DEVICES='' timeout 120 python -m pytest -q
 tests/test_grm_c2_profile.py tests/test_grm_c2_capture.py
 tests/test_grm_c2_payload.py`.
Dry run: `python scripts/grm_c2_cells.py --dry-run`.

The lead commands intentionally refuse GPU execution under this registration.
A separate, evidence-supported decomposition/budget planning amendment **and its
consumer** are still required before those commands can execute. Do not edit the
frozen registry/order, silently lower estimates, extend a lease, or label this
handoff GPU-ready. The order allows a non-fit blocked report; that rail is used.

Persistence design: fresh capture through imported lived installers/run_turn;
ordinary repository flush and immutable **pre-probe** manifest/payload snapshots;
restart worker is a new process, reloads each snapshot without transcript refeed,
then runs the same original probe path. Replaying all old probes against the
final104-turn repository would alter historical facts and is not substituted.
Continuation inputs also validate prior persisted content hashes and reject
pending/missing payloads. Each measured probe records summed mounted ntok,
raw mount IDs, arena count, all four RT1 provenance fields, production B2 route
receipt, capture mapping, manifest metadata equality and new-process evidence.
Missing/unattested capture remains RED; defaults are explicitly unpinned.
RS3 live-cache slices are labeled `FRESH_CACHE_SLICE_PAYLOAD_NOT_MOVED_BY_PIN`,
with their actual span retained, rather than misreported as pinned harvests.

## 2. CPU gates — actual counts, not a clean-suite claim

**New author-run CPU gate:** `21 passed, 2 warnings in 0.65s`.
Receipt: `new_cpu_A6.log`. It includes ordinary manifest/payload persistence
and a new CPU subprocess reload, with recapture forbidden. This does not validate
GPT-OSS GPU serving, restart scores, whole-driver behavior, or blind correctness.

**Full existing CPU-selected battery (66 files):**
**1,480 passed,58 failed,148 skipped,20 errors; zero timeouts.**
All per-file original pytest summary lines, exact commands, failure node IDs and
error examples are pasted in `CPU_GATE_RESULTS.md`; machine receipt is
`cpu_summary.json`, with raw logs and hashes under `cpu/`.
The full inventory is original `cpu_inventory.json` plus amendmentsA1/A5.
33 GPU programs and real-device scaffold nodes are outside CPU collection.
The 148 skips came from existing test logic; no skip was introduced. Some
receipt-dependent lanes ran zero checks and are incomplete, not green evidence.

The six-file SCOUT-FIX-1 comparison subset reproduces **227 passed,11 failed**:
LSR-P2B route receipt, three-pass, SC1 demand, SC2 calibration, RS3 capture/seat,
WC1 sweep. All **23 SCOUT-FIX-1 fixtures pass**. Native runtime:
`121 passed, 2 warnings in 280.19s`; lifecycle:
`101 passed, 2 warnings in 133.05s`.

Failure accounting:

- 11 known missing calibration/campaign receipt failures in SC1/SC2.
- 10 failed real-device allocation tests inadvertently included in quant-format
 and importance-telemetry's mixed files: `RuntimeError: cudaMalloc failed: no
 CUDA-capable device is detected`. CUDA was hidden; no GPU computation ran.
 This was an inventory error and is retained, not called a CPU regression.
- 37 additional failures plus20 setup errors in broader existing coverage.
 These include absent ADM/DET/RS3/SC1 artifacts and the known APAMQ receipt test.
 Exact failures remain in the raw logs; no product repair is claimed.

Representative errors, verbatim:
`CalibrationError: missing persisted receipt: /mnt/ForgeRealm/wt/grm-c2/artifacts/grm_det1/run_20260831T160525Z_2/det1_4/campaign/calibration/thresholds_2149a44b6136fd1b.json`.
`BaselineRegistryError: active registration receipt hash/size record does not validate`.
`AssertionError: expected an append-only series of envelopes`.

All originally inventoried production sources, existing tests, fixtures, registries
and the order still match their registration SHA256 values. No separate original
source-version comparison was run here; therefore broader regression absence is
not certified. Native tests reference canonical C++ paths; worktree/canonical
`cpp/grm_runtime.cpp` and `.hpp` hashes match. No existing battery was edited.

## Prior art

**Local verified source:** DET1/EB1/WC1, RS3, RT1/RT1.1 and SCOUT-FIX-1,
project contributors (2026). Taken: battery fixtures and chronology, semantic
comparators, explicit capture/seating levers, split-child routing, ordinary
manifest persistence, B2 receipt projection and B5 mounted-token sum. C2 adds
an opt-in registry and paired checkpoint/restart orchestration; it does not add
a routing/selection algorithm. Checkpoint/restart is established systems practice;
no novelty claim. **No prior art known to me** for this particular adapter beyond
these local systems. No external literature verification or new paper claim was
needed. Annotations appear at code sites and in `LEDGER.md`.

## 3. Deviations, RED, process safety, identity

Budget non-fit is the primary RED. Archived WC1 timings (A3, fingerprinted source
receipts) total3,022.482s for the six fresh batteries alone, before C2's extra
checkpoint/replay overhead. They use levers ON at both widths and may contain
contention, so this is planning evidence, not a certified lower bound or measured
defaults result. It provides no basis to reduce estimates to fit.

AmendmentsA1/A5 correct CPU inventory; A2 pins initial driver gates; A3 adds
archived timing evidence; A4/A6 strengthen capture/payload validation. Earlier
driver versions and logs remain preserved. Order and registry are unchanged.
C3 COMMON was read from canonical checkout because it is absent locally.
Frozen frame/census copied only into campaign sidecars; canonical inputs read-only.

**Not claimed fixed:** E2E profile reproduction, restart model quality, adoption,
58 existing failures,20 errors,148 skipped tests, independent blind verification.
No profile drop is observed because no GPU cells ran. Conditional B1/B2/B3/B5
CPU reasoning is in `SCOUT_FIX_1_TRIAGE.md`; no commit reversal or GPU bisect.

No git commands, subagents, shell background waits, lock clearing, service changes,
external delivery or signals to others' processes. All tests ran foreground in
bounded calls. Ten failed CUDA API allocations are disclosed above; zero actual
GPU cells/model loads/computation. No timeout fired, and no process was killed.
No production files, shipped defaults or live baseline registry were changed.

Seat identity supplied by system: **Codex / GPT-6**; exact serving API model ID
is not exposed. Requested reasoning effort: **high**; no runtime effort change
is claimed. Target model prompt effort is separately pinned **low** by Harmony.
