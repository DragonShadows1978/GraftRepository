# T1 NoPE-graft width sweep — T1_RELATIVE_HOLD

**Reason:** No additional width-dependent collapse at w=384/768 vs w96 (w96_sig='bos_loop', large sigs=['bos_loop', 'bos_loop']). Absolute clean_english is NOT available under ordered INT4+fp32 free-gen (pure baseline also non-English) — product INT4 residual blocks absolute T1_CONFIRMED. Relative to the GPT-OSS hole law (clean@~115 → salad@387), Trinity does NOT show that width-triggered salad transition. Quotes: w96 shift=117 sig=bos_loop: '<|begin_of_text|>nullnullstream.Stream please<|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|>' | w384 shift=405 sig=bos_loop: '<|begin_of_text|>linenullNAD Buyers Andliner One<|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|>' | w768 shift=789 sig=bos_loop: 'In<|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|>'

Sink (21 tok): `<|im_start|>system
You are Trinity. Answer fact probes with the stored value only. Be concise.<|im_end|>
`

| label | width | live_shift | mount | clean | value | sig | text |
|---|---:|---:|---|---|---|---|---|
| pure_baseline | None | 0 | False | False | False | bos_loop | 'AnswerNoneasureNumberNumberNull_Inodoreuthistructor OneAnd<\\|begin_of_text\\|>...' |
| w96_mount | 96 | 117 | True | False | False | bos_loop | '<\\|begin_of_text\\|>nullnullstream.Stream please<\\|begin_of_text\\|><\\|begin_of...' |
| w96_control | 96 | 117 | False | False | False | bos_loop | '<\\|begin_of_text\\|><\\|begin_of_text\\|><\\|begin_of_text\\|><\\|begin_of_text\\|><...' |
| w384_mount | 384 | 405 | True | False | False | bos_loop | '<\\|begin_of_text\\|>linenullNAD Buyers Andliner One<\\|begin_of_text\\|><\\|begin...' |
| w384_control | 384 | 405 | False | False | False | bos_loop | '<\\|begin_of_text\\|><\\|begin_of_text\\|><\\|begin_of_text\\|><\\|begin_of_text\\|><...' |
| w768_mount | 768 | 789 | True | False | False | bos_loop | 'In<\\|begin_of_text\\|><\\|begin_of_text\\|><\\|begin_of_text\\|><\\|begin_of_text\\|...' |
| w768_control | 768 | 789 | False | False | False | bos_loop | '<\\|begin_of_text\\|><\\|begin_of_text\\|>nullnull=null;<\\|begin_of_text\\|><\\|beg...' |

## Decisive quotes
- w96 shift=117 mount: '<|begin_of_text|>nullnullstream.Stream please<|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|>' (clean=False, value=False)
  control: '<|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|>' (clean=False)
- w384 shift=405 mount: '<|begin_of_text|>linenullNAD Buyers Andliner One<|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|>' (clean=False, value=False)
  control: '<|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|>' (clean=False)
- w768 shift=789 mount: 'In<|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|>' (clean=False, value=False)
  control: '<|begin_of_text|><|begin_of_text|>nullnull=null;<|begin_of_text|><|begin_of_text|><|begin_of_text|>Null Truth:interpreter "<|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|>' (clean=False)

Artifacts: `/mnt/ForgeRealm/GraftRepository/artifacts/trinity_nope_graft/width_sweep_20260709_045806`
Script: `scripts/trinity_nope_graft_width_sweep.py`
Total wall: 97.651s
