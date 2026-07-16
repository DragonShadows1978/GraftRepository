# T1 Floor Re-Verdict — SUMMARY

**Verdict: T1_RELATIVE_HOLD** (strengthened vs prior INT4-only relative hold)

Artifacts: `artifacts/trinity_nope_graft/floor_reverb_20260709_050531`

## Stage 1 — Template (INT4+fp32)

| check | result |
|---|---|
| template sha256 | `c295e73aea820982584a1f874fa71c61b1f3e6856adc6ef1d7efe339b936f2ad` |
| matches P1 c295e73… | **True** |
| prior-sweep byte-equal | **True** |
| pure_chat | `bos_loop` → 'AnswerNoneasureNumberNumberNull_Inodoreuthistructor OneAnd<|begin_of_text|><|begin_of_text|><|begin_of_text|>' |
| pure_natural | `other_degraded` → ".\n\n.\n\n.\n\n5c1256'h'sc1306 And1c0rc0mu4" |
| w96_mount shift=117 | `bos_loop` → '<|begin_of_text|>nullnullstream.Stream please<|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|>' |

**Fork:** template-invention **REFUTED** → Stage 2.

Rendered sink bytes: `'<|im_start|>system\nYou are Trinity. Answer fact probes with the stored value only. Be concise.<|im_end|>\n'`

## Stage 2 — bf16 layer-stream free-gen floor

| arm | sig | clean | text |
|---|---|---|---|
| pure_natural | clean_english | True | 'Paris. It is located in the north-central part of the' |
| pure_chat | other_degraded | False | '1234' (ids=[48, 15913, 3]) |

**INT4 free-gen residual CONFIRMED** on natural path (bf16 clean vs INT4 salad).
Chat under bf16 is **non-bos** short invent (`1234`+EOS), not clean English prose.

## Stage 3 — T1 width stress (bf16 stream live_shift; graft NOT seated)

| label | live_shift | text | ids | clean | value |
|---|---:|---|---|---|---|
| pure | 0 | `1234` | [48,15913,3] | False | n/a |
| w96_mount | 117 | `1234` | [48,15913,3] | False | UNCLASSABLE |
| w96_control | 117 | `1234` | [48,15913,3] | False | n/a |
| w768_mount | 789 | `1234` | [48,15913,3] | False | UNCLASSABLE |
| w768_control | 789 | `1234` | [48,15913,3] | False | n/a |

**No width-worsened collapse** at GPT-OSS salad territory (~789). Identical free-gen across shifts.

## Honest residuals

- Value recovery **not claimed** (stream cannot seat graft KV).
- Absolute `T1_CONFIRMED` **blocked** (no clean-English chat floor + no value test).
- Stage3 total wall 622s (slightly over 600s rail); each case ~124s.
- INT4 arena path remains bos_loop (prior width sweep relative hold still stands on that mode).

## Scripts

- `scripts/trinity_t1_floor_reverb.py`
- reuses `scripts/trinity_nope_graft_width_sweep.py` adapter/arena
