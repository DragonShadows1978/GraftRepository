# GRM3P-DIAG — Unbounded-arena mechanism report

Date: 2026-07-23  
Order: `orders/GRM3P_DIAG_UNBOUNDED.md`  
Scope: diagnosis only; no production fix.

## Pre-registration

The order's registered hypotheses remain frozen:

- **H-COMOUNT:** a wider or different co-mounted set poisons the read.
- **H-FIT:** residency changes which ranked nodes fit in the 96-token arena.
- **H-ECHO:** the two-turn live window supplies the contaminating value.
- **H-ROUTE:** unbounded and 4 MB runs rank different nodes.

ARM A exposes one additional discriminator before any new GPU run:

- **H-REHYDRATE:** the decisive budget effect is payload lifecycle, not
  logical route width. With `GRM_GRAFT_STORAGE_BITS=8`, the 4 MB pager
  snapshots a device graft into the packed host format, evicts it, and
  materializes it again before mounting. The unbounded leg can mount the
  original device-resident payload without that pack/unpack page-in. This
  hypothesis is intentionally stated at the resident-versus-rehydrated
  payload-path level; the current instruments do not yet distinguish
  quantization error from another loader/materialization effect.

The following additional arms and decision rules are registered now, before
their results:

- **C2-TOPK2:** F-FULL `single`, defaults except `--topk 2`. This is a
  one-variable intermediate-width control and runs only if C1 does not
  convict H-COMOUNT.
- **C2-LIVE1:** F-FULL `single`, defaults except `--live-turns 1`. This is a
  one-variable H-ECHO control and runs only if no earlier arm convicts.
- **C3-EARLY-REHYDRATE:** F-FULL `single`, defaults except
  `--restart-after 5`, so the registered flush/reload occurs after turn 4
  and before the first miss probe. The session remains unbounded. A 9/9
  result convicts H-REHYDRATE only if turn 5 retains ARM B's ranking,
  source rank, fitted/mounted IDs, and logical live-window IDs while the
  on-disk node receipt confirms packed `storage_bits=8`. A score change
  without those invariant receipts is not a conviction.
- **CONFIRM:** repeat the first decisive causal arm exactly once and require
  the same scorecard, miss/pass identities, and transcript SHA-256. Stop
  after that confirmation. If no causal arm is decisive, proceed to ARM D
  and report ambiguity if its registered witness is also non-identifying.

## Results

Results are appended only after each frozen arm completes.

### Infrastructure interruption (not an arm result)

The first locked ARM B launch failed before model construction:

```text
RuntimeError: cudaMalloc failed: no CUDA-capable device is detected
```

A second locked check returned:

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

The NVIDIA kernel modules and PCI-bound GPU were present, but all
`/dev/nvidia*` device nodes were absent. No model load or turn completed, so
this launch is excluded from ARM B. Receipt:
`artifacts/grm_three_pass/diag_unbounded_arm_b_single/LAUNCH_FAILURE.txt`.

### ARM B — F-FULL single reproduction

Frame: `--mode full --turn-pipeline single`, all other registered CLI
defaults (`topk=3`, `live_turns=2`, `arena_width=96`, `restart_after=17`,
`ngen=32`, unbounded VRAM), env
`GRM_GQA_CUDA_ROUTE=1 GRM_GRAFT_STORAGE_BITS=8`.

Result line:

```text
ARM B F-FULL single: 7/9; FAIL turn 5 orion pin -> "The current Orion PIN value is Vortex-3-Sierra."; FAIL turn 13 orion pin -> "Vortex-3-Sierra."; all other probes PASS.
```

The two registered misses and their contaminating CYPHER BRIDGE value
reproduced exactly. Turn 5 retained source rank 1, ranking/mount plan
`[0,2,1]`, fitted/mounted IDs `[0]`, and logical live IDs `[3,4]`. Turn 13
retained source rank 3, ranking head/mount plan `[5,6,7]`, fitted/mounted IDs
`[5,7]`, and logical live IDs `[12,14]`. Passing control turn 9 retained
source rank 1, mount plan `[4,5,6]`, fitted/mounted IDs `[4]`, and logical
live IDs `[8,9]`.

Transcript SHA-256:
`06ef1fd2e18680238d241dc7d972d0b269924b43d77e20f2e9f7d997b5731dc5`.
Session:
`artifacts/grm_three_pass/diag_unbounded_arm_b_single_r2`.

### ARM C1 — top-k 1

Frame: ARM B with the sole change `--topk 1`.

Probe scorecard line:

```text
C1 --topk 1: 7/9; FAIL turns 5 and 13 with the same CYPHER BRIDGE answers; PASS turns 9, 16, 19, 22, 24, 26, and 30.
```

Turn 5 requested and mounted only node `[0]` yet still answered
`"The current Orion PIN value is Vortex-3-Sierra."`. Turn 13 requested and
mounted only rank-1 node `[5]` yet still answered `"Vortex-3-Sierra."`;
the intended node remained rank 3. The transcript SHA-256 was byte-identical
to ARM B:
`06ef1fd2e18680238d241dc7d972d0b269924b43d77e20f2e9f7d997b5731dc5`.

Decision: the registered 9/9 H-COMOUNT conviction condition did not fire.
H-COMOUNT is disfavored at top-k granularity; proceed to C2-TOPK2.
Session:
`artifacts/grm_three_pass/diag_unbounded_c1_topk1`.

### ARM C2-TOPK2 — top-k 2

Frame: ARM B with the sole change `--topk 2`.

Probe scorecard line:

```text
C2-TOPK2 --topk 2: 7/9; FAIL turns 5 and 13 with the same CYPHER BRIDGE answers; PASS turns 9, 16, 19, 22, 24, 26, and 30.
```

Turn 5 used plan `[0,2]` and fitted/mounted only `[0]`. Turn 13 used plan
`[5,6]` and fitted/mounted only `[5]`; the intended node remained rank 3.
The transcript SHA-256 again matched ARM B and C1 byte-for-byte:
`06ef1fd2e18680238d241dc7d972d0b269924b43d77e20f2e9f7d997b5731dc5`.

Decision: intermediate route width did not change recall or generated bytes.
No conviction condition fired; proceed to C2-LIVE1.
Session:
`artifacts/grm_three_pass/diag_unbounded_c2_topk2`.

### ARM C2-LIVE1 — one-turn live window

Frame: ARM B with the sole change `--live-turns 1`.

Probe scorecard line:

```text
C2-LIVE1 --live-turns 1: 1/9; PASS turn 5 only; FAIL turns 9, 13, 16, 19, 22, 24, 26, and 30.
```

Turn 5 flipped to pass while retaining source rank 1, ranking
`[0,2,1]`, plan `[0,2,1]`, and fitted/mounted `[0]`; its logical live IDs
changed from `[3,4]` to `[4]`. This was not a clean repair: prior passing
control turn 9 failed, its ranking changed from `[4,5,6,2,1,7]` to
`[4,8,6,2,1,7]`, and turn 13's source rank changed from 3 to 2. Seven
additional previously passing probes also failed. Transcript SHA-256:
`f493eef67a1122757c3221281f1cdaa6db552668aaa03cb4cb6716a81e31fdb0`.

Decision: H-ECHO is not convicted. Reducing the live window changes route
eligibility/ranking and broadly degrades the readout, so the isolated cause
of the original two misses is not identified by this control. Proceed to
C3-EARLY-REHYDRATE.
Session:
`artifacts/grm_three_pass/diag_unbounded_c2_live1`.

### ARM C3-EARLY-REHYDRATE — restart-after 5 (r3 seat)

Provenance note (not an arm result): an earlier launch of this arm exists on
disk at `artifacts/grm_three_pass/diag_unbounded_c3_early_rehydrate`. That
session is TRUNCATED — 24 transcript rows / 6 probes only (it stopped at
turn 23; transcript sha
`d6be965f40c2e89fc8349f2a58e7002d02e6f5f47fea70c8fb96d95d442a74f7`), so it is
NOT a valid C3 receipt and is excluded. C3 was re-run to completion into a
fresh session dir; that complete run is the receipt below.

Frame: ARM B with the sole change `--restart-after 5` (flush/reload after
turn 4, before the first miss probe; session remains unbounded), env
`GRM_GQA_CUDA_ROUTE=1 GRM_GRAFT_STORAGE_BITS=8`, all other registered
defaults.

Probe scorecard line:

```text
C3 --restart-after 5: 8/9; PASS turn 5 orion pin -> "Auric-4-Alpha" (FLIP from ARM B FAIL); PASS turn 13 orion pin -> "Kestrel-9-Tango" (FLIP from ARM B FAIL); FAIL turn 24 terra port -> "I'm sorry, but I can't comply with that." (NEW refusal, was PASS in ARM B); PASS turns 9, 16, 19, 22, 26, and 30.
```

Both original misses flipped to PASS, but only one flip meets the frozen
invariant condition, and a new failure appeared:

- Turn 5 (source_turn 0, pre-restart-boundary): FLIP FAIL->PASS with EVERY
  route invariant held identical to ARM B — source_rank 1, ranking `[0,2,1]`,
  mount_plan `[0,2,1]`, fitted/mounted `[0]`, logical live IDs `[3,4]`. The
  sole differing variable versus ARM B is the payload lifecycle flag
  (`resumed` False->True): the turn-5 orion node was packed to
  `storage_bits=8`, evicted at the restart, and rehydrated before mounting.
  The on-disk node receipt confirms the pack: `repository/nodes/0000.npz`
  carries `storage_bits=8`, `group_size=32`, INT8 `k_codes`/`v_codes`. This
  is exactly the invariant-preserving repair the frozen C3 rule names as the
  H-REHYDRATE signature.
- Turn 13 (supersession): FLIP FAIL->PASS but the route CHANGED — source_rank
  3->2, ranking `[5,6,7,...]`->`[6,7,4,...]`, fitted/mounted `[5,7]`->`[6]`.
  The restart renumbers downstream native node IDs, so turn 13 cannot be
  repaired "with invariant ranking" by construction; its flip is confounded
  by that route change and does NOT satisfy the frozen invariant condition.
- Turn 24 (terra port): NEW FAIL — a model-level refusal
  (`"I'm sorry, but I can't comply with that."`), not a recall contamination,
  absent in ARM B. This holds the overall score at 8/9.

Transcript SHA-256:
`5a3be671f8442da6377e39ba2e753cb3ec4c1825afa03de4dbcb1ab77ff23652`.
Session:
`artifacts/grm_three_pass/diag_unbounded_c3_early_rehydrate_r3`.

Decision (frozen rule applied verbatim, not adjusted post-hoc): the
pre-registered conviction condition is "9/9 (or the two misses flipping) WITH
invariant rankings/source-rank/mount ids, plus the storage_bits=8 node
receipt." The result is 8/9, not 9/9. Turn 5's flip satisfies the invariant
sub-condition AND the storage_bits=8 receipt; turn 13's flip does not (route
changed); and turn 24 introduces a new refusal. Full conviction to the frozen
9/9 bar therefore does NOT fire. What DOES fire is a decisive, single-probe,
invariant-preserving H-REHYDRATE receipt on turn 5 (same route, same mounts,
same live window; only the rehydration lifecycle differs). Proceed to CONFIRM,
then — because C3 leaves ambiguity at the arm level — to ARM D.

### CONFIRM — repeat C3 exactly once

Frame: byte-identical to C3 (`--restart-after 5`, same env, fresh session
dir).

Probe scorecard line:

```text
CONFIRM (repeat C3): 8/9; identical scorecard to C3 — PASS turn 5 "Auric-4-Alpha", PASS turn 13 "Kestrel-9-Tango", FAIL turn 24 refusal, PASS turns 9, 16, 19, 22, 26, 30.
```

The scorecard, every route invariant, the miss/pass identities, and the
transcript SHA-256 all matched C3 byte-for-byte:
`5a3be671f8442da6377e39ba2e753cb3ec4c1825afa03de4dbcb1ab77ff23652`. The C3
result — including the turn-5 invariant-preserving flip, the turn-13 route
change, and the turn-24 refusal — is deterministic and confirmed.
Session:
`artifacts/grm_three_pass/diag_unbounded_c3_confirm_r3`.

### ARM D pre-registration

Registered here BEFORE its result. ARM D fires because C3+CONFIRM leave
arm-level ambiguity (8/9, not 9/9; turn 13 route-confounded; turn 24 a new
refusal). Registered witness: on the ARM B frame (where turns 5 and 13 still
MISS), a per-mounted-node vs live-window attention-mass readout on the two
miss turns, additive and default-off, with the default smoke transcript SHA-256
required to stay
`68da84b8824eb79a14f65a692a602215bc53d84232cd041002410f51657bba4f`. Sharpest
question (per r3 order): during the wrong answer, does attention mass sit on
the LIVE-WINDOW cypher-turn tokens rather than the mounted orion node? A
decisive live-window-dominant reading on both miss turns would convict the
live-window echo path; a mounted-node-dominant reading would exonerate it.

### ARM D — witness not available on this stack (honest non-result)

The registered S1-style attention-mass tap
(`ArenaCache.set_telemetry`/`s1_mass`, `MLAAttentionTC._telemetry_mass`/
`reset_telemetry`) exists only on the MLA attention path
(`core/minicpm3_tc.py`). The GRM3P diagnostic runs the GPT-OSS-20B stack
(`core/gpt_oss20b_tc.py`), whose attention layers carry no telemetry
accumulator: `set_telemetry(True)` would set a `.telemetry` flag on layers
that never populate `_telemetry_mass`, so `s1_mass()` returns zeros. GPT-OSS
attention is computed inside FUSED kernels (`sink_attention_tc`,
`apa_blend_softmax_sink`/`sink_apa_blend_attention_tc`,
`sliding_sink_attention_tc`); the per-position softmax weights are never
materialized in Python and are not returned by any kernel. Producing the
per-mounted-node vs live-window mass readout therefore requires new
attention-mass instrumentation inside the model forward / fused kernels — a
change to the default forward path with real byte-identity risk, which the
additive/default-off/`68da84b8...`-preserving rail and the no-fix diagnostic
scope forbid. No forward-path code was written; `git diff --stat` for `core/`
and `scripts/` is empty, so the default smoke SHA-256 is preserved trivially,
but the registered witness could not be executed. ARM D yields no reading on
this stack.

### Verdict and residuals

Verdict — H-REHYDRATE is POSITIVELY SUPPORTED but NOT convicted to the frozen
9/9 bar; the mechanism is not fully identified under the available instruments.

Decisive receipt in hand: C3+CONFIRM (deterministic, transcript sha
`5a3be671...`) repair the turn-5 miss FAIL->PASS with the orion node's route
held fully invariant to ARM B (source_rank 1, ranking/plan `[0,2,1]`,
fitted/mounted `[0]`, live IDs `[3,4]`), the sole changed variable being the
pack(`storage_bits=8`, receipt `nodes/0000.npz`)->evict->rehydrate lifecycle.
That isolates the payload rehydration path — the same path the 4 MB budget
forces — as sufficient to flip the read on the pre-restart-boundary probe,
without changing route width or eligibility (unlike the confounded C2-LIVE1).
This is a positive H-REHYDRATE receipt for turn 5.

Why not a full conviction (frozen rule, un-adjusted): the pre-registered bar is
9/9 or both misses flipping WITH invariant ranking. C3 is 8/9. Turn 13's flip
is route-confounded (the restart renumbers downstream nodes, so an
invariant-ranking repair of turn 13 is impossible by construction), and a new
turn-24 refusal appeared. The registered ARM D tiebreaker (live-window vs
mounted-node attention mass) is unavailable on the GPT-OSS fused-kernel stack.

Un-searched space (named, per honesty rail): (1) the fused-kernel
attention-mass witness — whether, during the ARM B wrong answer, mass sits on
the live-window cypher tokens vs the mounted orion node — requires
kernel-level instrumentation not present on the GPT-OSS path and out of scope
for a no-fix diagnostic seat; (2) an unconfounded turn-13 test would need a
restart scheme that rehydrates WITHOUT renumbering downstream native node IDs
(not exposed by the current CLI); (3) the turn-24 restart-induced refusal is
uncharacterized — it is a generation-level refusal, not shown to be a
recall/mount effect.

Residuals: prior-seat C3 dir
`artifacts/grm_three_pass/diag_unbounded_c3_early_rehydrate` is truncated
(6/24, sha `d6be965f...`) and excluded — left on disk untouched for the lead.
No git actions taken; no `core/`/`scripts/` edits; GPU runs all held
`flock -w 7200 /tmp/forge-gpu.lock`.

Named successors (one-liners, no fixes): S-D1 — port the MLA S1 attention-mass
tap to the GPT-OSS fused-kernel path (default-off) to run the ARM D witness;
S-D2 — add a rehydrate-without-renumber restart mode to give turn 13 an
invariant-ranking test; S-D3 — characterize the restart-induced turn-24
refusal (recall vs pure generation).

---

## GRM3P-DIAG-DOSE — budget dose-response + payload diff

Date: 2026-07-23  
Order: `orders/GRM3P_DIAG_DOSE.md`  
Scope: diagnosis only; no production fix; no git; no subagents.

### Pre-registration (frozen before any dose run)

Copied from the order; not adjusted after rows:

Under H-REHYDRATE, per-probe recall is predicted by the source node's
lifecycle, NOT the budget size: a probe passes **iff** its source node was
packed→evicted→rehydrated at least once before the probe turn ("washed").
A miss on a washed node, or a pass on a never-washed node at any budget,
counts **AGAINST** H-REHYDRATE and must be reported as such.

Baseline anchors already on disk:

- unbounded = 7/9 with misses t5/t13 = never-washed sources  
  (`artifacts/grm_three_pass/diag_unbounded_arm_b_single_r2`)
- 4MB = 9/9 all-washed  
  (`artifacts/grm_three_pass/p4_fcold_single`, same F-FULL single frame,
  `vram_budget_mb=4`)

### Wash determination method (receipt constraint)

`--turn-pipeline single` leaves `step_io` events empty: `TurnStepIOTracker`
is entered but `current_step` is never set on the single path, so
`graft_page_in` events are not recorded (contrast `p4_fcold_three_pass`,
which has 186 page-in events). Wash is therefore reconstructed offline from
pager physics + instrumentation mounts/rankings (not invented live counters):

- Node device bytes = `ntok * vals_per_tok_layer * 2 * n_layers` with
  GPT-OSS descriptor values `vals_per_tok_layer=1024`, `n_layers=24`
  → **4.03125 MB** for the 86-token turn-0 orion node.
- After each deposit / turn, simulate repository `_page()` LRU spill under
  the session's `vram_budget_mb` (None = no spill).
- Restart boundary (`resumed` flips True): all pre-existing nodes marked
  packed+evicted (host durable; device cleared).
- Mount/`_ensure_h` candidates taken from `route_ranking.ranking_ids`,
  `mount_fitted`, and 1-indexed `info.mounts` (fillers).
- **washed** (lifecycle) = source node is in the ever-spilled set at probe
  start (packed+evicted at least once; mount therefore uses rehydrate path
  if cold, or already holds dequant tensors if warm after a prior
  rehydrate). This is the operational reading of "pack→evict→rehydrate
  lifecycle" when single-path page-in receipts are absent.
- Evidence strings record whether the source was cold at probe start and
  any prior rehydrate turn under the same simulation.

This is **pager-state reconstruction**, not a live `graft_page_in` receipt.
Where it disagrees with a future live capture, the live capture wins.

### ARM E — dose-response sweep

Frame (identical except budget): `--mode full --turn-pipeline single`, env
`GRM_GQA_CUDA_ROUTE=1 GRM_GRAFT_STORAGE_BITS=8`, defaults
(`topk=3`, `live_turns=2`, `arena_width=96`, `restart_after=17`, `ngen=32`),
sole variable `--vram-budget-mb ∈ {8, 16, 32}`. GPU held under
`flock -w 7200 /tmp/forge-gpu.lock`.

#### Probe scorecard lines (verbatim)

```text
E-8MB: 9/9; PASS turns 5, 9, 13, 16, 19, 22, 24, 26, 30.
```

Transcript SHA-256:
`f43582ad5de8f4c643fbfa7eb2cadf6d10a63c3757f10f38d12913c53db7d8e9`.  
Session: `artifacts/grm_three_pass/diag_dose_e_8mb_single`.

```text
E-16MB: 7/9; FAIL turn 5 orion pin -> 'The current Orion PIN value is Vortex-3-Sierra.'; FAIL turn 13 orion pin -> 'Vortex-3-Sierra.'; PASS turns 9, 16, 19, 22, 24, 26, 30.
```

Transcript SHA-256:
`4160c8d073abfb8b850baaecd998c5df79a16e415a945057aca24ee33caebb71`.  
Session: `artifacts/grm_three_pass/diag_dose_e_16mb_single`.

```text
E-32MB: 7/9; FAIL turn 5 orion pin -> 'The current Orion PIN value is Vortex-3-Sierra.'; FAIL turn 13 orion pin -> 'Vortex-3-Sierra.'; PASS turns 9, 16, 19, 22, 24, 26, 30.
```

Transcript SHA-256:
`06ef1fd2e18680238d241dc7d972d0b269924b43d77e20f2e9f7d997b5731dc5`
(**byte-identical** to ARM B unbounded transcript).  
Session: `artifacts/grm_three_pass/diag_dose_e_32mb_single`.

Dose summary (score only): **4MB 9/9 · 8MB 9/9 · 16MB 7/9 · 32MB 7/9 · unbounded 7/9**.
The failure mode is not a smooth function of budget size; it is a step from
full pass (≤8MB) to the same two CYPHER-BRIDGE contamination misses as
unbounded (≥16MB), with 32MB collapsing onto unbounded bytes.

#### Per-probe tables (washed from pager reconstruction)

**E-8MB** (`diag_dose_e_8mb_single`)

| turn | fact | pass | src | washed | rank | evidence |
|------|------|------|-----|--------|------|----------|
| 5 | orion pin | True | 0 | True | 1 | spilled_cold_at_probe_start+rehydrate_before@t3 |
| 9 | cypher bridge | True | 4 | True | 1 | spilled_cold_at_probe_start |
| 13 | orion pin | True | 7 | True | 2 | spilled_cold_at_probe_start+rehydrate_before@t9 |
| 16 | lyra dock | True | 13 | True | 2 | spilled_cold_at_probe_start |
| 19 | nova key | True | 8 | True | 1 | spilled_cold+rehydrate_before@t13+post_restart |
| 22 | mira seal | True | 20 | True | 2 | spilled_cold+post_restart_or_prior_spill |
| 24 | terra port | True | 16 | True | 1 | spilled_cold+rehydrate_before@t19+post_restart |
| 26 | ember code | True | 23 | True | 1 | spilled_cold+rehydrate_before@t24+post_restart |
| 30 | atlas tone | True | 30 | True | 1 | spilled_cold+post_restart_or_prior_spill |

**E-16MB** (`diag_dose_e_16mb_single`)

| turn | fact | pass | src | washed | rank | evidence |
|------|------|------|-----|--------|------|----------|
| 5 | orion pin | False | 0 | False | 1 | never_spilled_device_original |
| 9 | cypher bridge | True | 4 | True | 1 | spilled_cold_at_probe_start |
| 13 | orion pin | False | 7 | **True** | 3 | spilled_cold_at_probe_start (**AGAINST: miss on washed**) |
| 16 | lyra dock | True | 13 | False | 2 | never_spilled_device_original (**AGAINST: pass on never-washed**) |
| 19 | nova key | True | 8 | True | 1 | rehydrate_before@t13+post_restart |
| 22 | mira seal | True | 21 | False | 2 | never_spilled_device_original (**AGAINST**) |
| 24 | terra port | True | 16 | True | 1 | spilled_cold+rehydrate_before@t19+post_restart |
| 26 | ember code | True | 24 | True | 1 | rehydrate_before@t24+post_restart |
| 30 | atlas tone | True | 31 | False | 1 | never_spilled_device_original (**AGAINST**) |

**E-32MB** (`diag_dose_e_32mb_single`)

| turn | fact | pass | src | washed | rank | evidence |
|------|------|------|-----|--------|------|----------|
| 5 | orion pin | False | 0 | False | 1 | never_spilled_device_original |
| 9 | cypher bridge | True | 4 | False | 1 | never_spilled_device_original (**AGAINST**) |
| 13 | orion pin | False | 7 | False | 3 | never_spilled_device_original |
| 16 | lyra dock | True | 13 | False | 2 | never_spilled_device_original (**AGAINST**) |
| 19 | nova key | True | 8 | True | 1 | spilled_cold+rehydrate_before@t13+post_restart |
| 22 | mira seal | True | 20 | False | 2 | never_spilled_device_original (**AGAINST**) |
| 24 | terra port | True | 16 | True | 1 | rehydrate_before@t19+post_restart |
| 26 | ember code | True | 23 | True | 1 | rehydrate_before@t24+post_restart |
| 30 | atlas tone | True | 30 | False | 1 | never_spilled_device_original (**AGAINST**) |

**ANCHOR-unbounded** (`diag_unbounded_arm_b_single_r2`, 7/9)

| turn | fact | pass | src | washed | rank | evidence |
|------|------|------|-----|--------|------|----------|
| 5 | orion pin | False | 0 | False | 1 | never_spilled_device_original |
| 9 | cypher bridge | True | 4 | False | 1 | never_spilled_device_original (**AGAINST**) |
| 13 | orion pin | False | 7 | False | 3 | never_spilled_device_original |
| 16 | lyra dock | True | 13 | False | 2 | never_spilled_device_original (**AGAINST**) |
| 19 | nova key | True | 8 | True | 1 | spilled_cold+post_restart |
| 22 | mira seal | True | 20 | False | 2 | never_spilled_device_original (**AGAINST**) |
| 24 | terra port | True | 16 | True | 1 | rehydrate_before@t19+post_restart |
| 26 | ember code | True | 23 | False | 1 | never_spilled_device_original (**AGAINST**) |
| 30 | atlas tone | True | 30 | False | 1 | never_spilled_device_original (**AGAINST**) |

**ANCHOR-4MB** (`p4_fcold_single`, 9/9)

| turn | fact | pass | src | washed | rank | evidence |
|------|------|------|-----|--------|------|----------|
| 5 | orion pin | True | 0 | True | 1 | spilled_cold+rehydrate_before@t3 |
| 9 | cypher bridge | True | 4 | True | 1 | spilled_cold_at_probe_start |
| 13 | orion pin | True | 7 | True | 2 | spilled_cold+rehydrate_before@t9 |
| 16 | lyra dock | True | 13 | True | 2 | spilled_cold+rehydrate_before@t13 |
| 19 | nova key | True | 8 | True | 1 | spilled_cold+rehydrate_before@t13+post_restart |
| 22 | mira seal | True | 20 | True | 2 | spilled_cold+rehydrate_before@t19+post_restart |
| 24 | terra port | True | 16 | True | 1 | spilled_cold+rehydrate_before@t19+post_restart |
| 26 | ember code | True | 23 | True | 1 | spilled_cold+rehydrate_before@t24+post_restart |
| 30 | atlas tone | True | 30 | True | 1 | spilled_cold+post_restart_or_prior_spill |

#### Cross-budget correlation (45 probe rows = 27 dose + 18 anchors)

| washed \ pass | FAIL | PASS | row total |
|---------------|------|------|-----------|
| never_washed  | 5    | 12   | 17        |
| washed        | 1    | 27   | 28        |
| col total     | 6    | 39   | **45**    |

Frozen prediction agreement (pass iff washed): **32/45**.  
**AGAINST** rows: **13/45**.

- **miss on washed (1):** E-16MB turn 13 src=7 (spilled cold at probe start; still answers Vortex-3-Sierra).
- **pass on never-washed (12):** E-16MB t16/t22/t30; E-32MB t9/t16/t22/t30; unbounded t9/t16/t22/t26/t30.

#### ARM E verdict (frozen prediction, unadjusted)

**H-REHYDRATE as a biconditional (pass iff washed) is FALSIFIED** under the
registered counting rule: 13 AGAINST rows, including one miss-on-washed and
twelve pass-on-never-washed.

What remains as **partial, non-biconditional support** (descriptive, not a
redefinition of the frozen claim):

1. Tight budgets that force wash of the t5/t13 sources (4MB, 8MB) are 9/9;
   budgets that leave t5's source never-washed (16MB, 32MB, unbounded) miss
   t5 with the same CYPHER-BRIDGE contamination.
2. 32MB is not an intermediate phenotype: its transcript is byte-identical to
   unbounded ARM B, so above ~one-to-few node residency the budget knob is
   idle for this script.
3. Wash is **not sufficient** (16MB t13 washed+FAIL) and **not necessary**
   (many never-washed PASSes, especially non-orion probes and post-deposit
   nodes that never spill).

Per the order: the falsification arm is as reportable as confirmation. The
frozen biconditional does not hold.

### ARM F — payload diff (turn-5 source node 0; ARM B vs C3)

Sessions:

- unbounded ARM B: `artifacts/grm_three_pass/diag_unbounded_arm_b_single_r2`
- C3 early-rehydrate: `artifacts/grm_three_pass/diag_unbounded_c3_early_rehydrate_r3`

#### On-disk packed payload (end-of-session `repository/nodes/0000.npz`)

| field | ARM B | C3 | equal? |
|-------|-------|-----|--------|
| file SHA-256 | `4860e405f69e8099a028fd380b6ad1da3f6f2faed149e45cb4cbf40785f9f22c` | same | **YES (byte-identical file)** |
| format_version / storage_bits / group_size | 1 / 8 / 32 | same | YES |
| k_codes / v_codes shape, dtype | (24,8,86,2,32) int8 | same | YES, n_diff=0 |
| k_scales / v_scales | (24,8,86,2,1) float32 | same | YES, maxdiff=0 |
| k_shape / v_shape | [24,8,86,64] | same | YES |
| dequantized K max\|Δ\| / mean\|Δ\| | 0 / 0 | — | identical |
| dequantized V max\|Δ\| / mean\|Δ\| | 0 / 0 | — | identical |
| index `rkey_0000` | float32 (8,86,64) | same | maxdiff=0 |
| dialect_descriptor | GQA, rope_full_yarn, seat_remountable, 8×64 | same | YES |
| manifest ntok / native_node_id / retired | 86 / 0 / true | same | YES |
| host_present / device_present (end state) | False / False | True / True | end-of-session only; not turn-5 |

Quant-grid scale (from packed scales; **not** a measured original-vs-dequant
diff, because the original device deposit is not on disk):

- mean k_scale ≈ 0.1986 → typical code step scale/127 ≈ **1.56e-3**
- mean v_scale ≈ 0.0465 → typical step ≈ **3.66e-4**
- dequant K range ≈ [-236, 174], mean|K|≈2.90; V range ≈ [-61.5, 38.5], mean|V|≈1.98

#### Classification

**INSUFFICIENT ARTIFACTS to classify NUMERIC-ONLY / STRUCTURAL / BOTH** for
the registered comparison (device-resident deposit payload **at turn 5** vs
rehydrated payload **at turn 5**).

Reason (receipt-level):

1. Both sessions only persist the **packed** `storage_bits=8` host form at
   flush/end. ARM B's never-washed turn-5 mount used the **original device
   `h` tensors** produced at deposit; those tensors are never snapshotted to
   disk. After the session, B and C3 node 0 packs are **byte-identical**, so
   end-state packs cannot separate "what B mounted at t5" from "what C3
   mounted at t5 after rehydrate."
2. No on-disk capture of un-RoPE seating positions, arena seat indices, or
   per-layer device dtypes at probe time exists in either session's
   instrumentation (only route/mount IDs and token-level `evicted`/`resident`
   live-window counters).
3. Manifest `host_present`/`device_present` differ only as **end-of-session**
   residency bookkeeping, not as a turn-5 payload diff.

Therefore ARM F cannot measure dequantized value max/mean per layer between
device-resident deposit and rehydrated forms from artifacts alone.

#### Live capture required (do not build; listed only)

To complete the registered ARM F comparison, a future instrumented run must
capture, for node 0:

1. **T0+ deposit snapshot:** per-layer device K/V tensors (or fp16 host
   clone) immediately after turn-0 plant, **before** any
   `pack_node`/`_ensure_host_payload`/`_page` (true device-resident original).
2. **T5 mount snapshot (ARM B unbounded):** the exact `h` tensor object (or
   byte clone) used at turn-5 `_ensure_h`/`swap` for node 0, plus arena seat
   positions, RoPE/un-RoPE flags, mount_plan/fitted, and whether
   `node_loader` ran.
3. **T5 mount snapshot (C3 after restart):** same fields after
   pack→evict→`unpack_node` rehydrate, still with route invariants held.
4. Optional: packed intermediate `host_payload` bytes at the C3 restart
   flush for node 0 (should match end-state pack if no further mutation).
5. Then compute per-layer max/mean |B_device − C3_rehyd| on K and V and
   classify NUMERIC-ONLY vs STRUCTURAL vs BOTH against those two live
   tensors — not against end-of-session packs.

### Dose + F residuals

- Single-path `step_io` page-in receipts are empty by construction; wash
  tables are offline LRU reconstructions. Cross-check against
  `p4_fcold_three_pass` page-ins is consistent for the 4MB all-washed claim
  but is not a substitute for live single-path page-in IDs.
- 16MB t13 miss-on-washed is a hard AGAINST under the frozen rule; mechanism
  for that row is not identified here (could be ranking/mount
  confounds — source_rank 3, fitted `[5,7]` — rather than payload path).
- Pass-on-never-washed rows show many facts are recallable from original
  device deposits; the t5/t13 orion+CYPHER pattern is special, not universal.
- ARM F cannot close the numeric-vs-structural question without live capture.
- No `core/` or `scripts/` edits; no git; GPU under flock; diagnostic only.

Named successors (one-liners, no fixes): S-E1 — enable single-path step_io
`current_step` attribution so wash is a live receipt not a reconstruction;
S-E2 — live ARM F dual snapshot (deposit h vs post-rehydrate h) on node 0 at
t5; S-E3 — isolate 16MB t13 washed-miss (rank-3 / multi-mount) from pure
payload-path effects.

---

## GRM3P-DIAG-CONTAM — contaminator-wash conviction (Opus/Grok seat)

Date: 2026-07-23  
Order: `orders/GRM3P_DIAG_CONTAM.md`  
Scope: diagnosis only; no production fix; no git; no subagents.

### Pre-registration (frozen before any contam run; copied from order)

THEORY (lead): the miss mechanism is contaminator-side — the turn-3
cypher deposit (node 3), recency-mounted and perpetually LRU-hot at
budgets >=16MB, has never been pack/unpack-canonicalized; its device
payload hijacks the read (attractor class).

PREDICTIONS (frozen; not edited after results):
- **P1:** per-probe, pass <=> the Vortex-carrying node in the mounted set
  was pack->evict->rehydrate washed before that turn (correlation keyed
  on the CONTAMINATOR, not the source).
- **P2:** at 8MB, node 3 IS evicted+rehydrated between turn 3 and turn 5;
  at 16MB it is NOT (telemetry, not reconstruction).
- **P3:** node 3's mount-time device payload differs between the 8MB and
  16MB legs (key statistics / bytes); the packed host copies do not.

Any receipt violating P1–P3 is falsifying and reported as such.

Identity note (observational, not a prediction edit): on the F-FULL
script, the Vortex-carrying plant is turn 4 / **node 4** (`cypher bridge`);
node 3 is the turn-3 filler. P2/P3 are still evaluated against **node 3**
as registered; analysis also dumps the cypher node for honesty.

### Work item 1–2 — instrumentation landed (additive, default-off)

| piece | path | enable |
|-------|------|--------|
| Paging telemetry module | `core/paging_telemetry.py` | `GRM_PAGING_TELEMETRY_PATH`, `GRM_MOUNT_SNAPSHOT_DIR` |
| Evict / page_in / pack hooks | `core/graft_repository.py` (`_page`, `_load_node`, `_ensure_host_payload`, `_free_retired`) | same env; no-op when unset |
| Turn context + probe snapshot | `scripts/grm_e2e_session.py` | snapshots after `_attempt` for live∪fitted nodes |
| Runner | `scripts/grm3p_contam_run.sh` | flock GPU; smoke then 8/16/unbounded |
| Analyzer | `scripts/grm3p_contam_analyze.py` | P1 table, P2 timelines, P3 diffs |

Default path: env paths **unset** → telemetry disabled; no session files
written; no pre-probe `_ensure_h`; control flow of pack/evict/page-in
unchanged.

### RED residual — seat cannot execute GPU runs (sandbox)

This seat was launched with **`--sandbox strict`**. Landlock/seccomp
blocks open of:

- `/home/vader/.local/lib/python3.12/site-packages/*` (numpy, torch, …)
- `/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/*`
- `/mnt/ForgeRealm/Project-Tensor/tensor_cuda/*`

Receipt of the block (verbatim):

```text
ModuleNotFoundError: No module named 'numpy'
ModuleNotFoundError: No module named 'tensor_cuda'
```

Also blocked: `bwrap` uid map, `nsenter`, `systemd-run --user`,
`ssh localhost`, `pip install`. CUDA userspace lib loads (`libcuda.so.1`)
but the Python ML stack and model weights are not readable.

Therefore the registered post-wiring default-smoke SHA recheck and the
three telemetry legs **could not be executed in this seat**. No scorecard
rows, no P1/P2/P3 live receipts were produced by this seat. That is a RED
execution residual, not a theory verdict.

Pre-existing on disk (created before this seat's instrumentation edits,
**not** a post-wiring byte-identity proof):

- `artifacts/grm_three_pass/contam_baseline_smoke_single`
- transcript SHA-256:
  `68da84b8824eb79a14f65a692a602215bc53d84232cd041002410f51657bba4f`
  (matches registered default-smoke SHA; **pre-wire**, do not count as
  work-item-1 gate)

### Required Done lines — STATUS

```text
default-smoke byte-identity: NOT RUN POST-WIRE (sandbox RED); pre-wire baseline sha 68da84b8824eb79a14f65a692a602215bc53d84232cd041002410f51657bba4f present on disk
8MB scorecard: NOT RUN (sandbox RED)
16MB scorecard: NOT RUN (sandbox RED)
unbounded scorecard: NOT RUN (sandbox RED)
P1 table: NOT RUN (sandbox RED)
P2 timelines: NOT RUN (sandbox RED)
P3 diff numbers: NOT RUN (sandbox RED)
```

### Verdict on frozen predictions

**NO VERDICT** — P1/P2/P3 cannot be convicted or falsified without the
three live telemetry legs. Prior DOSE work remains: source-side
H-REHYDRATE biconditional is falsified; contaminator-side theory is still
open. Instrumentation is ready; evidence is not.

### Session dirs (expected after unsandboxed runner)

| leg | session |
|-----|---------|
| default smoke post-wire | `artifacts/grm_three_pass/contam_default_smoke_postwire_single` |
| 8MB + telemetry | `artifacts/grm_three_pass/contam_08mb_single` |
| 16MB + telemetry | `artifacts/grm_three_pass/contam_16mb_single` |
| unbounded + telemetry | `artifacts/grm_three_pass/contam_unbounded_single` |
| analysis JSON | `artifacts/grm_three_pass/contam_analysis.json` |

### Exact re-run (unsandboxed seat / lead)

```bash
cd /home/vader/GraftRepository-three-pass
# Requires: readable ~/.local site-packages, HF snapshot, tensor_cuda,
# GPU under flock. Do NOT use --sandbox strict.
bash scripts/grm3p_contam_run.sh
# Then append real scorecards / P1 / P2 / P3 / verdict to this report.
```

Env contract for a single leg:

```bash
export GRM_GQA_CUDA_ROUTE=1 GRM_GRAFT_STORAGE_BITS=8
export GRM_PAGING_TELEMETRY_PATH='{session}/paging_events.jsonl'
export GRM_MOUNT_SNAPSHOT_DIR='{session}/mount_snapshots'
flock -w 7200 /tmp/forge-gpu.lock \
  python3 scripts/grm_e2e_session.py --mode full --turn-pipeline single \
    --vram-budget-mb 8 \
    --session-dir artifacts/grm_three_pass/contam_08mb_single \
    --skip-gpu-idle-check
```

### Residuals

1. **Sandbox RED** blocks post-wire smoke SHA + all three legs (this seat).
2. Full-script node identity: Vortex plant is node 4, not node 3; P2/P3 as
   registered still target node 3; analyzer also reports cypher node.
3. No `core/`/`scripts/` production fixes; telemetry is default-off only.
4. No git. No subagents. No live service calls.

Named successor: **S-CONTAM-RUN** — re-dispatch this order (or only the
runner) **without** `--sandbox strict` so GPU + site-packages + HF weights
are readable; then append the Done lines and prediction verdicts.

## Lead analysis — MECHANISM IDENTIFIED (Fable, 2026-07-23, from CONTAM telemetry)

Per-node lifecycle receipts (contam_{08mb,16mb,unbounded}_single) close it.
"Wash" was a PRESENCE PROXY. The functional mechanism:

**A value-bearing distractor node that is PHYSICALLY resident in the
mounted/live context wins value-shaped questions whenever the true
source is stale (t5) or out-ranked (t13).**

- t5 (all legs): Vortex node 4 sits in the logical recency set [3,4]
  everywhere. 8MB: evicted@t4, NO page-in before probe → physically
  absent (recency seats silently skip non-resident payloads) → PASS.
  16MB/unbounded: resident → present → FAIL. Same rankings, same
  mounts, same flags — presence is the only variable.
- t13: routing ranks Vortex-carrying node rank 1 (true source rank 3);
  QUESTION mounts force page-in → present regardless of budget → FAIL
  at 16MB/unbounded. (Question mounts page in; recency mounts don't —
  the asymmetry that made the dose ladder.)
- Budget ≤8MB "fixes" recall by ACCIDENT: eviction pressure removes
  the distractor payload before it can echo.
- H-REHYDRATE (payload-bytes theory) CLOSED: presence, not bytes.
  P3's missing snapshot arrays are moot.

**Root cause (driver, not arena):** scripts/grm_e2e_session.py's Fork-A
probe path bypasses Arena.step — no precise-first collapse, no
recency-topical-only exclusion (registered RECENCY LAW 2026-06-11:
"for point lookups the previous turn IS the echo source"), no grounding
ladder. The registered production laws that kill BOTH classes exist in
core/graft_arena.py and never run on this path. Corpus-100/E4 echo +
grounded-but-wrong classes reappeared exactly where their defenses
were bypassed.

**Fix (licensed, flagged):** driver-level enforcement of the existing
laws on probe turns — point-lookup detection → exclude recency/live
value carriers; precise-first when rank-1 covers probe identifiers;
identifier-aware grounding rejection. Default-off flag (byte-identical
default); flag-on gates: F-FULL 9/9, F-COLD 9/9 unchanged, smoke 2/2.
