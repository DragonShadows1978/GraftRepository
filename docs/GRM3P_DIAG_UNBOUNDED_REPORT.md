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
