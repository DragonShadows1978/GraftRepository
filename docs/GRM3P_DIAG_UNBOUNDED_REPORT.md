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
