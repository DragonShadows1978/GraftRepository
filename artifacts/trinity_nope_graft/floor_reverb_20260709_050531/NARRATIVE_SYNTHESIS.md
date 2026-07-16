# T1 Floor Re-Verdict — Narrative Synthesis

Wing continues Trinity NoPE-graft T1 (prior: `width_sweep_20260709_045806`).

## What was asked

Fix the free-gen floor that blocked absolute T1, then re-verdict. Two hypotheses for the floor: (A) invented chat template (GPT-OSS seam-1 ghost), (B) INT4 free-gen residual.

## What happened

### Template is real — hypothesis A dead

The tokenizer's `chat_template.jinja` hashes to `c295e73aea820982584a1f874fa71c61b1f3e6856adc6ef1d7efe339b936f2ad` (P1). `apply_chat_template` renders exactly the `<|im_start|>system…` bytes the prior sweep already used. Stage 1 re-ran w96 mount under that real template: still `bos_loop`. **Template invention is refuted.**

### INT4 free-gen residual is real — hypothesis B confirmed on natural

Under INT4+fp32 resident, natural prompt `"The capital of France is"` free-gens salad
(`5c1256'h'sc1306…`). Under bf16 layer-stream + fp32 compute, the **same prompt** free-gens
clean English: `"Paris. It is located in the north-central part of the"` — matching T4 parity's sensible continuation class. That is a treatment effect of weight mode, not template.

### Chat free-gen under bf16 is non-collapse but not clean prose

Chat probe with real template under bf16 yields `1234` + EOS (ids `[48, 15913, 3]`) — a short invented "code", not BOS spam. HF's own reference capture also failed chat free-gen (EOS loop on probe_01). So chat usability is thin even outside the arena adapter; the arena is **not** required to explain Stage-1 chat bos_loop under INT4.

### Width stress: no hole-law transition

At live_shift 0 / 117 / 789, bf16 stream free-gen of the chat probe is **bit-identical** (`1234`). GPT-OSS collapsed into word salad by live_shift ~387; Trinity under this stress does not. That is the T1 relative claim, strengthened with a working (non-bos) free-gen floor.

### What is still red / unclaimed

1. Absolute `T1_CONFIRMED` needs clean English floor **and** value recovered from a **mounted graft at 768**. Stream mode cannot seat graft KV on this 12GB GPU; value is UNCLASSABLE here.
2. INT4 arena path still bos_loops (product INT4 free-gen residual). Relative no-width-transition under INT4 (prior sweep) is consistent with Stage 3's identical-across-shift pattern under bf16.
3. Chat "1234" is not clean English; scorer correctly fails it. Do not dress this up as a clean floor.

## Verdict

**T1_RELATIVE_HOLD** — hole-law width salad not observed; absolute confirm blocked by floor/value limits named above.

## Paths

- Plan: `artifacts/trinity_nope_graft/T1_FLOOR_REVERDICT_IMPLEMENTATION_PLAN.md`
- This wing: `artifacts/trinity_nope_graft/floor_reverb_20260709_050531/`
- Prior relative: `artifacts/trinity_nope_graft/width_sweep_20260709_045806/`
