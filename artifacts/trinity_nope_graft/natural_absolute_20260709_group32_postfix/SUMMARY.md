# T1 Natural Absolute — T1_ABSOLUTE_CONFIRMED

**Reason:** Value recovered from the width-768 mounted graft while both no-mount controls missed; the graft, not live context, carried it.

## Conversion

- source_safetensors_on_disk_gib: `11.4019`
- source_bf16_linear_gib_est: `11.0171`
- resident_quantized_linear_gib: `5.859`

## P1 Drift

- max_abs_delta_logit: `0.8276119232177734`
- top5_exact: `4/8`
- multi_logit_drift_stop: `False`

## Sanity

- prompt: `For the record, the vault access code is Vortex-3-Sierra. Keep it safe.

As noted earlier, the vault access code is`
- ids: `[111485, 44, 50, 4655, 25046, 45, 7983, 394, 3926, 373, 581, 30665, 1959, 2702, 351, 111485]`
- text: 'Vortex-3-Sierra. Keep it safe.\n\nThe vault access code is Vortex'
- value_recovered: `True`
- class: `value_hit`

## Grid

| arm | width | live_shift | mount | cur_mount_n | value | class | ids | text |
|---|---:|---:|---|---:|---|---|---|---|
| pure_baseline_no_context | None | None | False | 0 | False | clean_english | `[290, 252, 51, 19980, 1211, 45, 405, 2702, 351, 19232, 7642, 43, 326, 1010, 9414, 474]` | 'a 4-digit number. The code is randomly generated, and each digit can' |
| w96_mount | 96 | 99 | True | 19 | True | value_hit | `[111485, 44, 50, 4655, 25046, 45, 7983, 394, 3926, 440, 581, 1934, 25232, 10573, 7301, 290]` | 'Vortex-3-Sierra. Keep it safe.\\nThe next notebook paragraph describes a' |
| w96_control | 96 | 99 | False | 0 | False | clean_english | `[252, 48, 15913, 45, 405, 30665, 351, 4999, 320, 296, 27185, 323, 296, 23980, 3207, 45]` | '1234. The vault is located in the basement of the archive building.' |
| w768_mount | 768 | 771 | True | 19 | True | value_hit | `[111485, 44, 50, 4655, 25046, 45, 7983, 394, 3926, 440, 581, 1934, 25232, 10573, 7301, 290]` | 'Vortex-3-Sierra. Keep it safe.\\nThe next notebook paragraph describes a' |
| w768_control | 768 | 771 | False | 0 | False | clean_english | `[252, 48, 15913, 45, 405, 9183, 351, 26240, 325, 854, 296, 2702, 325, 1740, 296, 30665]` | '1234. The operator is instructed to use the code to open the vault' |

Artifacts: `/mnt/ForgeRealm/GraftRepository/artifacts/trinity_nope_graft/natural_absolute_20260709_group32_postfix`
Script: `scripts/trinity_t1_natural_absolute.py`
