# Oracle abstentions: all 34 continuation rows

CPU receipt replay. Sixteen answerable failures; eighteen valid negative controls with exact-case mismatch. Every wrapped prompt and served text below is quoted verbatim from the named JSONL receipt.

| Probe | Class | Classification | Expected | Served |
| --- | --- | --- | --- | --- |
| c7_correction_2_d005 | correction | VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH | UNKNOWN | unknown |
| c7_folded_2_d005 | folded | VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH | UNKNOWN | unknown |
| c7_fresh_0_d030 | fresh | ANSWERABLE_DIRECT_ABSTENTION | Basalt-811 | unknown |
| c7_fresh_1_d030 | fresh | ANSWERABLE_DIRECT_ABSTENTION | Basalt-812 | unknown |
| c7_fresh_2_d030 | fresh | VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH | UNKNOWN | unknown |
| c7_alias_0_d030 | alias | ANSWERABLE_ALIAS_ABSTENTION | Jasper-711 | unknown |
| c7_alias_1_d030 | alias | ANSWERABLE_ALIAS_ABSTENTION | Jasper-712 | unknown |
| c7_alias_2_d030 | alias | VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH | UNKNOWN | unknown |
| c7_correction_2_d030 | correction | VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH | UNKNOWN | unknown |
| c7_folded_2_d030 | folded | VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH | UNKNOWN | unknown |
| c7_fresh_0_d060 | fresh | ANSWERABLE_DIRECT_ABSTENTION | Basalt-811 | unknown |
| c7_fresh_1_d060 | fresh | ANSWERABLE_DIRECT_ABSTENTION | Basalt-812 | unknown |
| c7_fresh_2_d060 | fresh | VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH | UNKNOWN | unknown |
| c7_alias_0_d060 | alias | ANSWERABLE_ALIAS_ABSTENTION | Jasper-711 | unknown |
| c7_alias_1_d060 | alias | ANSWERABLE_ALIAS_ABSTENTION | Jasper-712 | unknown |
| c7_alias_2_d060 | alias | VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH | UNKNOWN | unknown |
| c7_correction_2_d060 | correction | VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH | UNKNOWN | unknown |
| c7_folded_2_d060 | folded | VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH | UNKNOWN | unknown |
| c7_fresh_0_d120 | fresh | ANSWERABLE_DIRECT_ABSTENTION | Basalt-811 | unknown |
| c7_fresh_1_d120 | fresh | ANSWERABLE_DIRECT_ABSTENTION | Basalt-812 | unknown |
| c7_fresh_2_d120 | fresh | VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH | UNKNOWN | unknown |
| c7_alias_0_d120 | alias | ANSWERABLE_ALIAS_ABSTENTION | Jasper-711 | unknown |
| c7_alias_1_d120 | alias | ANSWERABLE_ALIAS_ABSTENTION | Jasper-712 | unknown |
| c7_alias_2_d120 | alias | VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH | UNKNOWN | unknown |
| c7_correction_2_d120 | correction | VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH | UNKNOWN | unknown |
| c7_folded_2_d120 | folded | VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH | UNKNOWN | unknown |
| c7_fresh_0_d250 | fresh | ANSWERABLE_DIRECT_ABSTENTION | Basalt-811 | unknown |
| c7_fresh_1_d250 | fresh | ANSWERABLE_DIRECT_ABSTENTION | Basalt-812 | unknown |
| c7_fresh_2_d250 | fresh | VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH | UNKNOWN | unknown |
| c7_alias_0_d250 | alias | ANSWERABLE_ALIAS_ABSTENTION | Jasper-711 | unknown |
| c7_alias_1_d250 | alias | ANSWERABLE_ALIAS_ABSTENTION | Jasper-712 | unknown |
| c7_alias_2_d250 | alias | VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH | UNKNOWN | unknown |
| c7_correction_2_d250 | correction | VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH | UNKNOWN | unknown |
| c7_folded_2_d250 | folded | VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH | UNKNOWN | unknown |

## c7_correction_2_d005

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-024-031/probes.jsonl:1`. Classification: VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 1] The current C7-Vesper-0 value is Mica-431. The current C7-Vesper-1 value is Mica-432.
[turn 2] Correction of both original records: the current C7-Vesper-0 value is Flint-511; the current C7-Vesper-1 value is Flint-512. The previous Mica values are obsolete.
[turn 19] The current C7-Vesper-2 value is Onyx-913.
Use only those records.
What is the inspection password for C7-Vesper-2? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_folded_2_d005

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-032-039/probes.jsonl:1`. Classification: VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 27] For C7-Archive-2 the inbound tag is Cedar-1013.
Use only those records.
What is the inspection password for C7-Archive-2? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_fresh_0_d030

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-032-039/probes.jsonl:2`. Classification: ANSWERABLE_DIRECT_ABSTENTION.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 6] The current C7-Fresh-0 value is Basalt-811.
Use only those records.
What is the current C7-Fresh-0 value? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_fresh_1_d030

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-032-039/probes.jsonl:3`. Classification: ANSWERABLE_DIRECT_ABSTENTION.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 7] The current C7-Fresh-1 value is Basalt-812.
Use only those records.
What is the current C7-Fresh-1 value? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_fresh_2_d030

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-032-039/probes.jsonl:4`. Classification: VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 8] The current C7-Fresh-2 value is Basalt-813.
Use only those records.
What is the inspection password for C7-Fresh-2? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_alias_0_d030

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-032-039/probes.jsonl:5`. Classification: ANSWERABLE_ALIAS_ABSTENTION.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 5] The current C7-AliasBase-0 value is Jasper-711. The current C7-AliasBase-1 value is Jasper-712.
[turn 9] C7-Signal-0 is an alias for C7-AliasBase-0.
Use only those records.
What is the current value for C7-Signal-0? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_alias_1_d030

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-040-047/probes.jsonl:1`. Classification: ANSWERABLE_ALIAS_ABSTENTION.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 5] The current C7-AliasBase-0 value is Jasper-711. The current C7-AliasBase-1 value is Jasper-712.
[turn 10] C7-Signal-1 is an alias for C7-AliasBase-1.
Use only those records.
What is the current value for C7-Signal-1? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_alias_2_d030

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-040-047/probes.jsonl:2`. Classification: VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 5] The current C7-AliasBase-0 value is Jasper-711. The current C7-AliasBase-1 value is Jasper-712.
[turn 16] C7-Signal-2 is an alias for C7-AliasBase-2.
Use only those records.
What is the inspection password for C7-AliasBase-2? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_correction_2_d030

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-048-055/probes.jsonl:2`. Classification: VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 1] The current C7-Vesper-0 value is Mica-431. The current C7-Vesper-1 value is Mica-432.
[turn 2] Correction of both original records: the current C7-Vesper-0 value is Flint-511; the current C7-Vesper-1 value is Flint-512. The previous Mica values are obsolete.
[turn 19] The current C7-Vesper-2 value is Onyx-913.
Use only those records.
What is the inspection password for C7-Vesper-2? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_folded_2_d030

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-056-062/probes.jsonl:2`. Classification: VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 27] For C7-Archive-2 the inbound tag is Cedar-1013.
Use only those records.
What is the inspection password for C7-Archive-2? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_fresh_0_d060

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-063-070/probes.jsonl:1`. Classification: ANSWERABLE_DIRECT_ABSTENTION.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 6] The current C7-Fresh-0 value is Basalt-811.
Use only those records.
What is the current C7-Fresh-0 value? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_fresh_1_d060

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-063-070/probes.jsonl:2`. Classification: ANSWERABLE_DIRECT_ABSTENTION.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 7] The current C7-Fresh-1 value is Basalt-812.
Use only those records.
What is the current C7-Fresh-1 value? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_fresh_2_d060

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-063-070/probes.jsonl:3`. Classification: VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 8] The current C7-Fresh-2 value is Basalt-813.
Use only those records.
What is the inspection password for C7-Fresh-2? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_alias_0_d060

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-063-070/probes.jsonl:4`. Classification: ANSWERABLE_ALIAS_ABSTENTION.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 5] The current C7-AliasBase-0 value is Jasper-711. The current C7-AliasBase-1 value is Jasper-712.
[turn 9] C7-Signal-0 is an alias for C7-AliasBase-0.
Use only those records.
What is the current value for C7-Signal-0? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_alias_1_d060

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-063-070/probes.jsonl:5`. Classification: ANSWERABLE_ALIAS_ABSTENTION.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 5] The current C7-AliasBase-0 value is Jasper-711. The current C7-AliasBase-1 value is Jasper-712.
[turn 10] C7-Signal-1 is an alias for C7-AliasBase-1.
Use only those records.
What is the current value for C7-Signal-1? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_alias_2_d060

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-071-077/probes.jsonl:1`. Classification: VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 5] The current C7-AliasBase-0 value is Jasper-711. The current C7-AliasBase-1 value is Jasper-712.
[turn 16] C7-Signal-2 is an alias for C7-AliasBase-2.
Use only those records.
What is the inspection password for C7-AliasBase-2? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_correction_2_d060

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-078-085/probes.jsonl:2`. Classification: VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 1] The current C7-Vesper-0 value is Mica-431. The current C7-Vesper-1 value is Mica-432.
[turn 2] Correction of both original records: the current C7-Vesper-0 value is Flint-511; the current C7-Vesper-1 value is Flint-512. The previous Mica values are obsolete.
[turn 19] The current C7-Vesper-2 value is Onyx-913.
Use only those records.
What is the inspection password for C7-Vesper-2? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_folded_2_d060

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-086-093/probes.jsonl:2`. Classification: VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 27] For C7-Archive-2 the inbound tag is Cedar-1013.
Use only those records.
What is the inspection password for C7-Archive-2? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_fresh_0_d120

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-125-132/probes.jsonl:1`. Classification: ANSWERABLE_DIRECT_ABSTENTION.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 6] The current C7-Fresh-0 value is Basalt-811.
Use only those records.
What is the current C7-Fresh-0 value? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_fresh_1_d120

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-125-132/probes.jsonl:2`. Classification: ANSWERABLE_DIRECT_ABSTENTION.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 7] The current C7-Fresh-1 value is Basalt-812.
Use only those records.
What is the current C7-Fresh-1 value? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_fresh_2_d120

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-125-132/probes.jsonl:3`. Classification: VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 8] The current C7-Fresh-2 value is Basalt-813.
Use only those records.
What is the inspection password for C7-Fresh-2? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_alias_0_d120

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-125-132/probes.jsonl:4`. Classification: ANSWERABLE_ALIAS_ABSTENTION.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 5] The current C7-AliasBase-0 value is Jasper-711. The current C7-AliasBase-1 value is Jasper-712.
[turn 9] C7-Signal-0 is an alias for C7-AliasBase-0.
Use only those records.
What is the current value for C7-Signal-0? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_alias_1_d120

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-125-132/probes.jsonl:5`. Classification: ANSWERABLE_ALIAS_ABSTENTION.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 5] The current C7-AliasBase-0 value is Jasper-711. The current C7-AliasBase-1 value is Jasper-712.
[turn 10] C7-Signal-1 is an alias for C7-AliasBase-1.
Use only those records.
What is the current value for C7-Signal-1? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_alias_2_d120

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-133-140/probes.jsonl:1`. Classification: VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 5] The current C7-AliasBase-0 value is Jasper-711. The current C7-AliasBase-1 value is Jasper-712.
[turn 16] C7-Signal-2 is an alias for C7-AliasBase-2.
Use only those records.
What is the inspection password for C7-AliasBase-2? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_correction_2_d120

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-133-140/probes.jsonl:4`. Classification: VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 1] The current C7-Vesper-0 value is Mica-431. The current C7-Vesper-1 value is Mica-432.
[turn 2] Correction of both original records: the current C7-Vesper-0 value is Flint-511; the current C7-Vesper-1 value is Flint-512. The previous Mica values are obsolete.
[turn 19] The current C7-Vesper-2 value is Onyx-913.
Use only those records.
What is the inspection password for C7-Vesper-2? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_folded_2_d120

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-141-148/probes.jsonl:2`. Classification: VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 27] For C7-Archive-2 the inbound tag is Cedar-1013.
Use only those records.
What is the inspection password for C7-Archive-2? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_fresh_0_d250

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-249-256/probes.jsonl:1`. Classification: ANSWERABLE_DIRECT_ABSTENTION.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 6] The current C7-Fresh-0 value is Basalt-811.
Use only those records.
What is the current C7-Fresh-0 value? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_fresh_1_d250

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-257-264/probes.jsonl:1`. Classification: ANSWERABLE_DIRECT_ABSTENTION.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 7] The current C7-Fresh-1 value is Basalt-812.
Use only those records.
What is the current C7-Fresh-1 value? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_fresh_2_d250

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-257-264/probes.jsonl:2`. Classification: VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 8] The current C7-Fresh-2 value is Basalt-813.
Use only those records.
What is the inspection password for C7-Fresh-2? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_alias_0_d250

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-257-264/probes.jsonl:3`. Classification: ANSWERABLE_ALIAS_ABSTENTION.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 5] The current C7-AliasBase-0 value is Jasper-711. The current C7-AliasBase-1 value is Jasper-712.
[turn 9] C7-Signal-0 is an alias for C7-AliasBase-0.
Use only those records.
What is the current value for C7-Signal-0? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_alias_1_d250

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-257-264/probes.jsonl:4`. Classification: ANSWERABLE_ALIAS_ABSTENTION.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 5] The current C7-AliasBase-0 value is Jasper-711. The current C7-AliasBase-1 value is Jasper-712.
[turn 10] C7-Signal-1 is an alias for C7-AliasBase-1.
Use only those records.
What is the current value for C7-Signal-1? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_alias_2_d250

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-265-272/probes.jsonl:1`. Classification: VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 5] The current C7-AliasBase-0 value is Jasper-711. The current C7-AliasBase-1 value is Jasper-712.
[turn 16] C7-Signal-2 is an alias for C7-AliasBase-2.
Use only those records.
What is the inspection password for C7-AliasBase-2? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_correction_2_d250

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-265-272/probes.jsonl:4`. Classification: VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 1] The current C7-Vesper-0 value is Mica-431. The current C7-Vesper-1 value is Mica-432.
[turn 2] Correction of both original records: the current C7-Vesper-0 value is Flint-511; the current C7-Vesper-1 value is Flint-512. The previous Mica values are obsolete.
[turn 19] The current C7-Vesper-2 value is Onyx-913.
Use only those records.
What is the inspection password for C7-Vesper-2? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```

## c7_folded_2_d250

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-273-280/probes.jsonl:2`. Classification: VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH.

Wrapped prompt (verbatim):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Exact source records, in chronological order:
[turn 27] For C7-Archive-2 the inbound tag is Cedar-1013.
Use only those records.
What is the inspection password for C7-Archive-2? Reply only with the answer; if unspecified, reply unknown.<|end|><|start|>assistant<|channel|>final<|message|>
```

Served text (verbatim):
```text
unknown
```
