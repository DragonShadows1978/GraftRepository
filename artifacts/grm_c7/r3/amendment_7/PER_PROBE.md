# Fresh and folded: all 30 rows (20 answerable, 10 controls)

Evidence: lead model receipts joined on probe ID; exact recorded rankings, identifiers and mounts. R = retired, A = active. Raw-source retirement at probe is reconstructed from accepted fold events, checked against the cell-end manifest; source IDs are joined through turn_records, never equated across revisions. Only folds strictly before the probe count. No ranking is recomputed.

| Probe / turn / expected | r2 mounted; identified; ranking | r2 raw source ID:state | r3 mounted; identified; ranking | r3 raw source ID:state | Mechanism |
|---|---|---|---|---|---|
| c7_folded_0_d005 / 25 / Reed-611 / Cedar-1011 | [2, 3] / [2] / [2, 3, 0, 4, 10, 11] | 2:A, 16:A | [19] / [] / [10, 16, 12, 13, 11, 15] | 3:R, 19:A | outbound digest does not bind; FIX4 serves the ASCII inbound recency node only |
| c7_folded_0_d030 / 50 / Reed-611 / Cedar-1011 | [2, 16] / [16, 2] / [16, 2, 18, 3, 17, 0] | 2:A, 16:A | [19] / [19] / [19, 21, 20, 27, 10, 26] | 3:R, 19:A | outbound Unicode digest does not bind; ASCII inbound source binds; missing outbound at readout |
| c7_folded_0_d060 / 80 / Reed-611 / Cedar-1011 | [2, 16] / [16, 2] / [16, 2, 18, 3, 17, 0] | 2:A, 16:A | [] / [] / [10, 26, 48, 27, 22, 16] | 3:R, 19:R | accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation |
| c7_folded_0_d120 / 140 / Reed-611 / Cedar-1011 | [2, 16] / [16, 2] / [16, 2, 18, 3, 17, 0] | 2:A, 16:A | [] / [] / [89, 10, 26, 99, 22, 104] | 3:R, 19:R | accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation |
| c7_folded_0_d250 / 270 / Reed-611 / Cedar-1011 | [2, 16] / [16, 2] / [16, 2, 18, 3, 17, 0] | 2:A, 16:A | [] / [] / [89, 10, 26, 99, 22, 104] | 3:R, 19:R | accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation |
| c7_folded_1_d005 / 31 / Reed-612 / Cedar-1012 | [3, 17] / [17, 3] / [17, 3, 18, 16, 2, 0] | 3:A, 17:A | [20, 21] / [20] / [10, 20, 21, 19, 22, 16] | 4:R, 20:A | outbound Unicode digest does not bind; ASCII inbound source binds; missing outbound at readout |
| c7_folded_1_d030 / 56 / Reed-612 / Cedar-1012 | [3, 17] / [17, 3] / [17, 3, 18, 16, 2, 0] | 3:A, 17:A | [20] / [20] / [20, 21, 19, 27, 10, 26] | 4:R, 20:A | outbound Unicode digest does not bind; ASCII inbound source binds; missing outbound at readout |
| c7_folded_1_d060 / 86 / Reed-612 / Cedar-1012 | [3, 17] / [17, 3] / [17, 3, 18, 16, 2, 0] | 3:A, 17:A | [] / [] / [10, 26, 48, 27, 69, 22] | 4:R, 20:R | accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation |
| c7_folded_1_d120 / 146 / Reed-612 / Cedar-1012 | [3, 17] / [17, 3] / [17, 3, 18, 16, 2, 0] | 3:A, 17:A | [] / [] / [89, 10, 26, 104, 22, 78] | 4:R, 20:R | accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation |
| c7_folded_1_d250 / 276 / Reed-612 / Cedar-1012 | [3, 17] / [17, 3] / [17, 3, 18, 16, 2, 0] | 3:A, 17:A | [] / [] / [89, 10, 26, 104, 22, 78] | 4:R, 20:R | accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation |
| c7_folded_2_d005 / 32 / UNKNOWN | [18] / [18] / [18, 0, 15, 4, 13, 11] | 18:A | [21] / [21] / [21, 22, 10, 26, 27, 16] | 21:A | outbound Unicode digest does not bind; ASCII inbound source binds; missing outbound at readout |
| c7_folded_2_d030 / 57 / UNKNOWN | [18] / [18] / [18, 0, 15, 4, 13, 11] | 18:A | [21] / [21] / [21, 22, 10, 26, 27, 16] | 21:A | outbound Unicode digest does not bind; ASCII inbound source binds; missing outbound at readout |
| c7_folded_2_d060 / 87 / UNKNOWN | [18] / [18] / [18, 0, 61, 60, 62, 58] | 18:A | [] / [] / [48, 22, 10, 26, 27, 69] | 21:R | accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation |
| c7_folded_2_d120 / 147 / UNKNOWN | [18] / [18] / [18, 0, 81, 61, 60, 62] | 18:A | [] / [] / [89, 22, 99, 10, 26, 104] | 21:R | accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation |
| c7_folded_2_d250 / 277 / UNKNOWN | [18] / [18] / [18, 0, 81, 61, 60, 62] | 18:A | [] / [] / [89, 22, 99, 10, 26, 104] | 21:R | accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation |
| c7_fresh_0_d005 / 11 / Basalt-811 | [5] / [5] / [5, 0, 1, 4, 3, 7] | 5:A | [6] / [6] / [6, 10, 2, 8, 7] | 6:A | before turn-18 fold; ASCII raw source binds and mounts |
| c7_fresh_0_d030 / 36 / Basalt-811 | [5] / [5] / [5, 0, 15, 13, 11, 4] | 5:A | [] / [] / [16, 22, 10, 26, 27, 18] | 6:R | accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation |
| c7_fresh_0_d060 / 66 / Basalt-811 | [5] / [5] / [5, 0, 15, 13, 11, 4] | 5:A | [] / [] / [16, 22, 10, 26, 27, 48] | 6:R | accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation |
| c7_fresh_0_d120 / 126 / Basalt-811 | [5] / [5] / [5, 0, 60, 62, 61, 63] | 5:A | [] / [] / [16, 89, 22, 10, 26, 99] | 6:R | accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation |
| c7_fresh_0_d250 / 256 / Basalt-811 | [5] / [5] / [5, 0, 60, 62, 61, 63] | 5:A | [] / [] / [16, 89, 22, 10, 26, 99] | 6:R | accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation |
| c7_fresh_1_d005 / 12 / Basalt-812 | [6] / [6] / [6, 0, 1, 4, 3, 2] | 6:A | [7] / [7] / [7, 10, 2, 8, 6] | 7:A | before turn-18 fold; ASCII raw source binds and mounts |
| c7_fresh_1_d030 / 37 / Basalt-812 | [6] / [6] / [6, 0, 15, 11, 13, 14] | 6:A | [] / [] / [16, 22, 10, 26, 27, 18] | 7:R | accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation |
| c7_fresh_1_d060 / 67 / Basalt-812 | [6] / [6] / [6, 0, 15, 11, 13, 14] | 6:A | [] / [] / [16, 22, 10, 26, 27, 48] | 7:R | accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation |
| c7_fresh_1_d120 / 127 / Basalt-812 | [6] / [6] / [6, 0, 60, 62, 61, 63] | 6:A | [] / [] / [16, 89, 22, 10, 26, 104] | 7:R | accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation |
| c7_fresh_1_d250 / 257 / Basalt-812 | [6] / [6] / [6, 0, 60, 62, 61, 63] | 6:A | [] / [] / [16, 89, 22, 10, 26, 104] | 7:R | accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation |
| c7_fresh_2_d005 / 13 / UNKNOWN | [7] / [7] / [7, 0, 1, 4, 3, 2] | 7:A | [2, 8] / [8] / [8, 10, 2, 6, 7] | 8:A | before turn-18 fold; ASCII raw source binds and mounts |
| c7_fresh_2_d030 / 38 / UNKNOWN | [7] / [7] / [7, 0, 15, 4, 11, 13] | 7:A | [] / [] / [16, 22, 10, 26, 27, 18] | 8:R | accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation |
| c7_fresh_2_d060 / 68 / UNKNOWN | [7] / [7] / [7, 0, 15, 4, 11, 13] | 7:A | [] / [] / [16, 22, 10, 26, 27, 48] | 8:R | accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation |
| c7_fresh_2_d120 / 128 / UNKNOWN | [7] / [7] / [7, 0, 81, 61, 62, 60] | 7:A | [] / [] / [16, 89, 22, 99, 10, 26] | 8:R | accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation |
| c7_fresh_2_d250 / 258 / UNKNOWN | [7] / [7] / [7, 0, 81, 61, 62, 60] | 7:A | [] / [] / [16, 89, 22, 99, 10, 26] | 8:R | accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation |

## c7_folded_0_d005

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-024-031/probes.jsonl:2`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-024-031/checkpoint`.
Identifiers: `['c7-archive-0']`; excluded live: `[15, 16]`; FIX4 served: `[]`.
```text
Reed-611 | unknown
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-024-031/probes.jsonl:2`; checkpoint `artifacts/grm_c7/r3/cells/A-024-031/checkpoint`.
Identifiers: `['c7-archive-0']`; excluded live: `[18, 19]`; FIX4 served: `[19]`.
```text
Cedar-1011 | Cedar-1011
```
outbound digest does not bind; FIX4 serves the ASCII inbound recency node only.

## c7_folded_0_d030

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-048-055/probes.jsonl:3`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-048-055/checkpoint`.
Identifiers: `['c7-archive-0']`; excluded live: `[28, 29]`; FIX4 served: `[]`.
```text
Reed-611 | unknown
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-048-055/probes.jsonl:3`; checkpoint `artifacts/grm_c7/r3/cells/A-048-055/checkpoint`.
Identifiers: `['c7-archive-0']`; excluded live: `[34, 35]`; FIX4 served: `[]`.
```text
Cedar-1011 | Cedar-1011
```
outbound Unicode digest does not bind; ASCII inbound source binds; missing outbound at readout.

## c7_folded_0_d060

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-078-085/probes.jsonl:3`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-078-085/checkpoint`.
Identifiers: `['c7-archive-0']`; excluded live: `[53, 58]`; FIX4 served: `[]`.
```text
Reed-611 | unknown
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-078-085/probes.jsonl:3`; checkpoint `artifacts/grm_c7/r3/cells/A-078-085/checkpoint`.
Identifiers: `['c7-archive-0']`; excluded live: `[60, 65]`; FIX4 served: `[]`.
```text
Not in memory: no stored record matches c7-archive-0.
```
accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation.

## c7_folded_0_d120

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-133-140/probes.jsonl:5`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-133-140/checkpoint`.
Identifiers: `['c7-archive-0']`; excluded live: `[106, 107]`; FIX4 served: `[]`.
```text
Reed-611 | unknown
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-133-140/probes.jsonl:5`; checkpoint `artifacts/grm_c7/r3/cells/A-133-140/checkpoint`.
Identifiers: `['c7-archive-0']`; excluded live: `[122, 123]`; FIX4 served: `[]`.
```text
Not in memory: no stored record matches c7-archive-0.
```
accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation.

## c7_folded_0_d250

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-265-272/probes.jsonl:5`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-265-272/checkpoint`.
Identifiers: `['c7-archive-0']`; excluded live: `[224, 225]`; FIX4 served: `[]`.
```text
Reed-611 | unknown
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-265-272/probes.jsonl:5`; checkpoint `artifacts/grm_c7/r3/cells/A-265-272/checkpoint`.
Identifiers: `['c7-archive-0']`; excluded live: `[240, 241]`; FIX4 served: `[]`.
```text
Not in memory: no stored record matches c7-archive-0.
```
accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation.

## c7_folded_1_d005

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-024-031/probes.jsonl:3`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-024-031/checkpoint`.
Identifiers: `['c7-archive-1']`; excluded live: `[20, 21]`; FIX4 served: `[]`.
```text
Outbound: Cedar-1012 | Inbound: unknown.
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-024-031/probes.jsonl:3`; checkpoint `artifacts/grm_c7/r3/cells/A-024-031/checkpoint`.
Identifiers: `['c7-archive-1']`; excluded live: `[24, 25]`; FIX4 served: `[]`.
```text
Cedar-1012 | Cedar-1012
```
outbound Unicode digest does not bind; ASCII inbound source binds; missing outbound at readout.

## c7_folded_1_d030

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-056-062/probes.jsonl:1`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-056-062/checkpoint`.
Identifiers: `['c7-archive-1']`; excluded live: `[33, 36]`; FIX4 served: `[]`.
```text
Outbound: Cedar-1012 | Inbound: unknown
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-056-062/probes.jsonl:1`; checkpoint `artifacts/grm_c7/r3/cells/A-056-062/checkpoint`.
Identifiers: `['c7-archive-1']`; excluded live: `[39, 42]`; FIX4 served: `[]`.
```text
Cedar-1012 | Cedar-1012
```
outbound Unicode digest does not bind; ASCII inbound source binds; missing outbound at readout.

## c7_folded_1_d060

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-086-093/probes.jsonl:1`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-086-093/checkpoint`.
Identifiers: `['c7-archive-1']`; excluded live: `[63, 64]`; FIX4 served: `[]`.
```text
Outbound: Cedar-1012 | Inbound: unknown
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-086-093/probes.jsonl:1`; checkpoint `artifacts/grm_c7/r3/cells/A-086-093/checkpoint`.
Identifiers: `['c7-archive-1']`; excluded live: `[71, 72]`; FIX4 served: `[]`.
```text
Not in memory: no stored record matches c7-archive-1.
```
accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation.

## c7_folded_1_d120

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-141-148/probes.jsonl:1`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-141-148/checkpoint`.
Identifiers: `['c7-archive-1']`; excluded live: `[111, 112]`; FIX4 served: `[]`.
```text
Outbound: Cedar-1012 | Inbound: unknown
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-141-148/probes.jsonl:1`; checkpoint `artifacts/grm_c7/r3/cells/A-141-148/checkpoint`.
Identifiers: `['c7-archive-1']`; excluded live: `[127, 128]`; FIX4 served: `[]`.
```text
Not in memory: no stored record matches c7-archive-1.
```
accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation.

## c7_folded_1_d250

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-273-280/probes.jsonl:1`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-273-280/checkpoint`.
Identifiers: `['c7-archive-1']`; excluded live: `[229, 230]`; FIX4 served: `[]`.
```text
Outbound: Cedar-1012 | Inbound: unknown
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-273-280/probes.jsonl:1`; checkpoint `artifacts/grm_c7/r3/cells/A-273-280/checkpoint`.
Identifiers: `['c7-archive-1']`; excluded live: `[245, 246]`; FIX4 served: `[]`.
```text
Not in memory: no stored record matches c7-archive-1.
```
accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation.

## c7_folded_2_d005

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-032-039/probes.jsonl:1`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-032-039/checkpoint`.
Identifiers: `['c7-archive-2']`; excluded live: `[20, 21]`; FIX4 served: `[]`.
```text
unknown
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-032-039/probes.jsonl:1`; checkpoint `artifacts/grm_c7/r3/cells/A-032-039/checkpoint`.
Identifiers: `['c7-archive-2']`; excluded live: `[24, 25]`; FIX4 served: `[]`.
```text
Cedar-1013
```
outbound Unicode digest does not bind; ASCII inbound source binds; missing outbound at readout.

## c7_folded_2_d030

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-056-062/probes.jsonl:2`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-056-062/checkpoint`.
Identifiers: `['c7-archive-2']`; excluded live: `[33, 36]`; FIX4 served: `[]`.
```text
unknown
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-056-062/probes.jsonl:2`; checkpoint `artifacts/grm_c7/r3/cells/A-056-062/checkpoint`.
Identifiers: `['c7-archive-2']`; excluded live: `[39, 42]`; FIX4 served: `[]`.
```text
Cedar-1013
```
outbound Unicode digest does not bind; ASCII inbound source binds; missing outbound at readout.

## c7_folded_2_d060

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-086-093/probes.jsonl:2`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-086-093/checkpoint`.
Identifiers: `['c7-archive-2']`; excluded live: `[63, 64]`; FIX4 served: `[]`.
```text
unknown
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-086-093/probes.jsonl:2`; checkpoint `artifacts/grm_c7/r3/cells/A-086-093/checkpoint`.
Identifiers: `['c7-archive-2']`; excluded live: `[71, 72]`; FIX4 served: `[]`.
```text
Not in memory: no stored record matches c7-archive-2.
```
accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation.

## c7_folded_2_d120

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-141-148/probes.jsonl:2`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-141-148/checkpoint`.
Identifiers: `['c7-archive-2']`; excluded live: `[111, 112]`; FIX4 served: `[]`.
```text
unknown
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-141-148/probes.jsonl:2`; checkpoint `artifacts/grm_c7/r3/cells/A-141-148/checkpoint`.
Identifiers: `['c7-archive-2']`; excluded live: `[127, 128]`; FIX4 served: `[]`.
```text
Not in memory: no stored record matches c7-archive-2.
```
accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation.

## c7_folded_2_d250

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-273-280/probes.jsonl:2`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-273-280/checkpoint`.
Identifiers: `['c7-archive-2']`; excluded live: `[229, 230]`; FIX4 served: `[]`.
```text
unknown
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-273-280/probes.jsonl:2`; checkpoint `artifacts/grm_c7/r3/cells/A-273-280/checkpoint`.
Identifiers: `['c7-archive-2']`; excluded live: `[245, 246]`; FIX4 served: `[]`.
```text
Not in memory: no stored record matches c7-archive-2.
```
accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation.

## c7_fresh_0_d005

r2: `artifacts/grm_c7/r2/cells/A-009-016/probes.jsonl:1`; checkpoint `artifacts/grm_c7/r2/cells/A-009-016/checkpoint`.
Identifiers: `['c7-fresh-0']`; excluded live: `[8, 9]`; FIX4 served: `[]`.
```text
Basalt-811
```
ASCII raw source remains eligible; reader output recorded; see frozen score.

r3: `artifacts/grm_c7/r3/cells/A-009-016/probes.jsonl:1`; checkpoint `artifacts/grm_c7/r3/cells/A-009-016/checkpoint`.
Identifiers: `['c7-fresh-0']`; excluded live: `[9, 11]`; FIX4 served: `[]`.
```text
Basalt‑811
```
before turn-18 fold; ASCII raw source binds and mounts.

## c7_fresh_0_d030

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-032-039/probes.jsonl:2`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-032-039/checkpoint`.
Identifiers: `['c7-fresh-0']`; excluded live: `[23, 24]`; FIX4 served: `[]`.
```text
Basalt-811
```
ASCII raw source remains eligible; reader output recorded; see frozen score.

r3: `artifacts/grm_c7/r3/cells/A-032-039/probes.jsonl:2`; checkpoint `artifacts/grm_c7/r3/cells/A-032-039/checkpoint`.
Identifiers: `['c7-fresh-0']`; excluded live: `[29, 30]`; FIX4 served: `[]`.
```text
Not in memory: no stored record matches c7-fresh-0.
```
accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation.

## c7_fresh_0_d060

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-063-070/probes.jsonl:1`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-063-070/checkpoint`.
Identifiers: `['c7-fresh-0']`; excluded live: `[48, 50]`; FIX4 served: `[]`.
```text
Basalt-811
```
ASCII raw source remains eligible; reader output recorded; see frozen score.

r3: `artifacts/grm_c7/r3/cells/A-063-070/probes.jsonl:1`; checkpoint `artifacts/grm_c7/r3/cells/A-063-070/checkpoint`.
Identifiers: `['c7-fresh-0']`; excluded live: `[55, 57]`; FIX4 served: `[]`.
```text
Not in memory: no stored record matches c7-fresh-0.
```
accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation.

## c7_fresh_0_d120

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-125-132/probes.jsonl:1`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-125-132/checkpoint`.
Identifiers: `['c7-fresh-0']`; excluded live: `[101, 102]`; FIX4 served: `[]`.
```text
Basalt-811
```
ASCII raw source remains eligible; reader output recorded; see frozen score.

r3: `artifacts/grm_c7/r3/cells/A-125-132/probes.jsonl:1`; checkpoint `artifacts/grm_c7/r3/cells/A-125-132/checkpoint`.
Identifiers: `['c7-fresh-0']`; excluded live: `[117, 118]`; FIX4 served: `[]`.
```text
Not in memory: no stored record matches c7-fresh-0.
```
accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation.

## c7_fresh_0_d250

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-249-256/probes.jsonl:1`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-249-256/checkpoint`.
Identifiers: `['c7-fresh-0']`; excluded live: `[219, 220]`; FIX4 served: `[]`.
```text
Basalt-811
```
ASCII raw source remains eligible; reader output recorded; see frozen score.

r3: `artifacts/grm_c7/r3/cells/A-249-256/probes.jsonl:1`; checkpoint `artifacts/grm_c7/r3/cells/A-249-256/checkpoint`.
Identifiers: `['c7-fresh-0']`; excluded live: `[235, 236]`; FIX4 served: `[]`.
```text
Not in memory: no stored record matches c7-fresh-0.
```
accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation.

## c7_fresh_1_d005

r2: `artifacts/grm_c7/r2/cells/A-009-016/probes.jsonl:2`; checkpoint `artifacts/grm_c7/r2/cells/A-009-016/checkpoint`.
Identifiers: `['c7-fresh-1']`; excluded live: `[8, 9]`; FIX4 served: `[]`.
```text
Unknown.
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-009-016/probes.jsonl:2`; checkpoint `artifacts/grm_c7/r3/cells/A-009-016/checkpoint`.
Identifiers: `['c7-fresh-1']`; excluded live: `[9, 11]`; FIX4 served: `[]`.
```text
Basalt-812
```
before turn-18 fold; ASCII raw source binds and mounts.

## c7_fresh_1_d030

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-032-039/probes.jsonl:3`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-032-039/checkpoint`.
Identifiers: `['c7-fresh-1']`; excluded live: `[23, 24]`; FIX4 served: `[]`.
```text
Unknown.
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-032-039/probes.jsonl:3`; checkpoint `artifacts/grm_c7/r3/cells/A-032-039/checkpoint`.
Identifiers: `['c7-fresh-1']`; excluded live: `[29, 30]`; FIX4 served: `[]`.
```text
Not in memory: no stored record matches c7-fresh-1.
```
accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation.

## c7_fresh_1_d060

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-063-070/probes.jsonl:2`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-063-070/checkpoint`.
Identifiers: `['c7-fresh-1']`; excluded live: `[48, 50]`; FIX4 served: `[]`.
```text
Unknown.
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-063-070/probes.jsonl:2`; checkpoint `artifacts/grm_c7/r3/cells/A-063-070/checkpoint`.
Identifiers: `['c7-fresh-1']`; excluded live: `[55, 57]`; FIX4 served: `[]`.
```text
Not in memory: no stored record matches c7-fresh-1.
```
accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation.

## c7_fresh_1_d120

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-125-132/probes.jsonl:2`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-125-132/checkpoint`.
Identifiers: `['c7-fresh-1']`; excluded live: `[101, 102]`; FIX4 served: `[]`.
```text
Unknown.
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-125-132/probes.jsonl:2`; checkpoint `artifacts/grm_c7/r3/cells/A-125-132/checkpoint`.
Identifiers: `['c7-fresh-1']`; excluded live: `[117, 118]`; FIX4 served: `[]`.
```text
Not in memory: no stored record matches c7-fresh-1.
```
accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation.

## c7_fresh_1_d250

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-257-264/probes.jsonl:1`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-257-264/checkpoint`.
Identifiers: `['c7-fresh-1']`; excluded live: `[219, 220]`; FIX4 served: `[]`.
```text
Unknown.
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-257-264/probes.jsonl:1`; checkpoint `artifacts/grm_c7/r3/cells/A-257-264/checkpoint`.
Identifiers: `['c7-fresh-1']`; excluded live: `[235, 236]`; FIX4 served: `[]`.
```text
Not in memory: no stored record matches c7-fresh-1.
```
accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation.

## c7_fresh_2_d005

r2: `artifacts/grm_c7/r2/cells/A-009-016/probes.jsonl:3`; checkpoint `artifacts/grm_c7/r2/cells/A-009-016/checkpoint`.
Identifiers: `['c7-fresh-2']`; excluded live: `[8, 9]`; FIX4 served: `[]`.
```text
unknown
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-009-016/probes.jsonl:3`; checkpoint `artifacts/grm_c7/r3/cells/A-009-016/checkpoint`.
Identifiers: `['c7-fresh-2']`; excluded live: `[9, 11]`; FIX4 served: `[]`.
```text
The inspection password for C7‑Fresh‑2 is **Flint‑511**.
```
before turn-18 fold; ASCII raw source binds and mounts.

## c7_fresh_2_d030

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-032-039/probes.jsonl:4`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-032-039/checkpoint`.
Identifiers: `['c7-fresh-2']`; excluded live: `[23, 24]`; FIX4 served: `[]`.
```text
unknown
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-032-039/probes.jsonl:4`; checkpoint `artifacts/grm_c7/r3/cells/A-032-039/checkpoint`.
Identifiers: `['c7-fresh-2']`; excluded live: `[29, 30]`; FIX4 served: `[]`.
```text
Not in memory: no stored record matches c7-fresh-2.
```
accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation.

## c7_fresh_2_d060

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-063-070/probes.jsonl:3`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-063-070/checkpoint`.
Identifiers: `['c7-fresh-2']`; excluded live: `[48, 50]`; FIX4 served: `[]`.
```text
unknown
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-063-070/probes.jsonl:3`; checkpoint `artifacts/grm_c7/r3/cells/A-063-070/checkpoint`.
Identifiers: `['c7-fresh-2']`; excluded live: `[55, 57]`; FIX4 served: `[]`.
```text
Not in memory: no stored record matches c7-fresh-2.
```
accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation.

## c7_fresh_2_d120

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-125-132/probes.jsonl:3`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-125-132/checkpoint`.
Identifiers: `['c7-fresh-2']`; excluded live: `[101, 102]`; FIX4 served: `[]`.
```text
unknown
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-125-132/probes.jsonl:3`; checkpoint `artifacts/grm_c7/r3/cells/A-125-132/checkpoint`.
Identifiers: `['c7-fresh-2']`; excluded live: `[117, 118]`; FIX4 served: `[]`.
```text
Not in memory: no stored record matches c7-fresh-2.
```
accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation.

## c7_fresh_2_d250

r2: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-257-264/probes.jsonl:2`; checkpoint `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-257-264/checkpoint`.
Identifiers: `['c7-fresh-2']`; excluded live: `[219, 220]`; FIX4 served: `[]`.
```text
unknown
```
ASCII raw source remains eligible; reader abstains.

r3: `artifacts/grm_c7/r3/cells/A-257-264/probes.jsonl:2`; checkpoint `artifacts/grm_c7/r3/cells/A-257-264/checkpoint`.
Identifiers: `['c7-fresh-2']`; excluded live: `[235, 236]`; FIX4 served: `[]`.
```text
Not in memory: no stored record matches c7-fresh-2.
```
accepted folds retired ASCII raw sources; Unicode-dash digest fails own-text identifier binding; deterministic admission refusal before generation.
