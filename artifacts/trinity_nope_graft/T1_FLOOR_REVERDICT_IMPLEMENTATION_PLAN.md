# T1 Floor Re-Verdict — Implementation Plan (IMMUTABLE)

**Order:** T1 follow-up — fix generation floor, then re-verdict.  
**Date:** 2026-07-09  
**Working dir:** `/mnt/ForgeRealm/GraftRepository`  
**Writes only:** `scripts/trinity_*`, `artifacts/trinity_nope_graft/`  
**Rails:** no git commit; GPU ≤10 min/run; Project-Tensor + models read-only.

## Context (pre-registered)

- Prior width sweep: `artifacts/trinity_nope_graft/width_sweep_20260709_045806/`
- Relative finding solid: no width transition 0/117/405/789 (all `bos_loop`)
- Absolute blocked: baseline free-gen = `bos_loop`
- Prior report blamed INT4 free-gen
- **Unruled-out alternative:** invented chat template (GPT-OSS seam-1 ghost precedent)
- P1 real template hash: `c295e73aea820982584a1f874fa71c61b1f3e6856adc6ef1d7efe339b936f2ad`
- BOS id 0, EOS id 3
- fp32 T4 parity: 8/8 sensible greedy tokens on natural prompt (`The capital of France is` → Paris…)

## Pre-registered gates / forks

### Stage 1 — Template
1. Extract chat template from tokenizer (`chat_template.jinja` / loaded `tok.chat_template`).
2. `apply_chat_template` on probe messages; ledger **exact rendered bytes** + sha256.
3. Re-run **ONE arm**: width 96, mounted, INT4 weights + fp32 compute, real template.
4. Also pure-baseline chat + pure natural (positive control) under same compute.

**Fork:**
- If gen clean English on pure and/or w96 mount → template was the floor → Stage 3 in INT4+fp32.
- If still `bos_loop` (or equivalent non-English) → Stage 2.

### Stage 2 — Compute mode (bf16 layer-stream)
1. Same single arm class: establish free-gen floor under bf16 layer-stream weights (fp32 compute if needed for rope).
2. **Projection before launch:** ~33 s/forward; 16 tokens must fit 600 s rail.
3. Pure natural + pure chat first; mount arm only if pure chat cleans and wall allows.

**Fork:**
- Clean free-gen under bf16 → INT4 free-gen residual confirmed as floor; Stage 3 in this mode.
- Still broken on pure (no arena) → STOP; report both receipts verbatim (arena adapter not implicated if pure fails).
- Pure clean + arena mount broken → new seam (arena adapter); STOP and report.

### Stage 3 — T1 verdict run
In whichever mode produced a clean floor:
- widths **96 vs 768**, **mounted + control**, same probe fact (`Vortex-3-Sierra`).

**Registered verdict labels:**
| Label | Criteria |
|---|---|
| `T1_CONFIRMED` | Clean floor at BOTH widths + value recovered from mounted graft at 768 |
| `T1_CLEAN_VALUE_MISS` | Clean floor both widths, value missed both → route/readout, not hole |
| `T1_RELATIVE_HOLD` | No width-worsened collapse vs w96, but absolute clean unavailable |
| `T1_REFUTED` | Large width worsens collapse class vs w96 (hole-law-like) |
| `T1_FLOOR_UNRESOLVED` | Stage 1+2 both fail to produce clean free-gen floor |

## Deliverables
- `scripts/trinity_t1_floor_reverb.py`
- `artifacts/trinity_nope_graft/floor_reverb_<stamp>/` stage receipts
- Plan (this file), Ledger, Narrative Synthesis
- Final T1 verdict table + honest read

## Out of scope
- Product code edits in Project-Tensor
- APA engagement
- Git commits / pushes
- Changing T1 prediction text in TRINITY_NANO_PORT_PLAN.md
