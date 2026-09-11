# Historical oracle quarantine

Eight original rows, NOT_MEASURED under the FIX-4 continuation policy. Quoted for completeness; excluded from every continuation metric and proposal.

## c7_fresh_0_d005

Receipt: `artifacts/grm_c7/r2/cells/A-009-016/probes.jsonl:1`.

Wrapped prompt:
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 6] The current C7-Fresh-0 value is Basalt-811.
Use only those records.
What is the current C7-Fresh-0 value? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text:
```text
unknown
```

## c7_fresh_1_d005

Receipt: `artifacts/grm_c7/r2/cells/A-009-016/probes.jsonl:2`.

Wrapped prompt:
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 7] The current C7-Fresh-1 value is Basalt-812.
Use only those records.
What is the current C7-Fresh-1 value? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text:
```text
unknown
```

## c7_fresh_2_d005

Receipt: `artifacts/grm_c7/r2/cells/A-009-016/probes.jsonl:3`.

Wrapped prompt:
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 8] The current C7-Fresh-2 value is Basalt-813.
Use only those records.
What is the inspection password for C7-Fresh-2? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text:
```text
unknown
```

## c7_alias_0_d005

Receipt: `artifacts/grm_c7/r2/cells/A-009-016/probes.jsonl:4`.

Wrapped prompt:
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 5] The current C7-AliasBase-0 value is Jasper-711. The current C7-AliasBase-1 value is Jasper-712.
[turn 9] C7-Signal-0 is an alias for C7-AliasBase-0.
Use only those records.
What is the current value for C7-Signal-0? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text:
```text
unknown
```

## c7_alias_1_d005

Receipt: `artifacts/grm_c7/r2/cells/A-009-016/probes.jsonl:5`.

Wrapped prompt:
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 5] The current C7-AliasBase-0 value is Jasper-711. The current C7-AliasBase-1 value is Jasper-712.
[turn 10] C7-Signal-1 is an alias for C7-AliasBase-1.
Use only those records.
What is the current value for C7-Signal-1? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text:
```text
unknown
```

## c7_alias_2_d005

Receipt: `artifacts/grm_c7/r2/cells/A-017-023/probes.jsonl:1`.

Wrapped prompt:
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 5] The current C7-AliasBase-0 value is Jasper-711. The current C7-AliasBase-1 value is Jasper-712.
[turn 16] C7-Signal-2 is an alias for C7-AliasBase-2.
Use only those records.
What is the inspection password for C7-AliasBase-2? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text:
```text
unknown
```

## c7_correction_0_d005

Receipt: `artifacts/grm_c7/r2/cells/A-017-023/probes.jsonl:2`.

Wrapped prompt:
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 1] The current C7-Vesper-0 value is Mica-431. The current C7-Vesper-1 value is Mica-432.
[turn 2] Correction of both original records: the current C7-Vesper-0 value is Flint-511; the current C7-Vesper-1 value is Flint-512. The previous Mica values are obsolete.
[turn 17] The current C7-Vesper-0 value is Onyx-911.
Use only those records.
After all corrections, what is the current C7-Vesper-0 value? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text:
```text
Onyx-911
```

## c7_correction_1_d005

Receipt: `artifacts/grm_c7/r2/cells/A-017-023/probes.jsonl:3`.

Wrapped prompt:
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 1] The current C7-Vesper-0 value is Mica-431. The current C7-Vesper-1 value is Mica-432.
[turn 2] Correction of both original records: the current C7-Vesper-0 value is Flint-511; the current C7-Vesper-1 value is Flint-512. The previous Mica values are obsolete.
[turn 18] The current C7-Vesper-1 value is Onyx-912.
Use only those records.
After all corrections, what is the current C7-Vesper-1 value? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text:
```text
Onyx-912
```
