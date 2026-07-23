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
