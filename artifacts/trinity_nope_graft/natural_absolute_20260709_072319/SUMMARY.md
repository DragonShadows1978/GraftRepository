# T1 Natural Absolute — INT8_DRIFT_STOP

**Reason:** INT8-resident P1 teacher-forced logits exceeded the registered multi-logit drift rail (10.055524826049805 > 1.0); stop before sanity/grid. This points at the quant/dequant path itself, not INT4 bit width.

## Conversion

- source_safetensors_on_disk_gib: `11.4019`
- source_bf16_linear_gib_est: `11.0171`
- resident_quantized_linear_gib: `5.6011`

## P1 Drift

- max_abs_delta_logit: `10.055524826049805`
- top5_exact: `0/8`
- multi_logit_drift_stop: `True`

Artifacts: `/mnt/ForgeRealm/GraftRepository/artifacts/trinity_nope_graft/natural_absolute_20260709_072319`
Script: `scripts/trinity_t1_natural_absolute.py`
