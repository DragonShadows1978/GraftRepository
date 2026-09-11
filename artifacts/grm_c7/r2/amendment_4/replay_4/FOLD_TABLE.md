# Fold table: all 13 rejected events

CPU replay of source-bound receipts. No accepted digest exists; candidate texts below are verbatim. Token counts are reconstructed from the exact generation loop and decode-call trace, not retokenized estimates.

| Turn | Origin | Best hits / need | Required hits | Coverage | Tokens p0/p1/p2 | Cap-hit prompts |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | inherited_pre_FIX4 | 3/10 | 7 | 0.3 | 60/59/67 | none |
| 17 | inherited_pre_FIX4 | 2/10 | 7 | 0.2 | 20/15/80 | none |
| 26 | FIX4_continuation | 2/8 | 6 | 0.25 | 21/22/102 | none |
| 55 | FIX4_continuation | 2/8 | 6 | 0.25 | 63/63/51 | none |
| 74 | FIX4_continuation | 2/8 | 6 | 0.25 | 16/16/59 | none |
| 85 | FIX4_continuation | 3/9 | 7 | 0.3333333333333333 | 37/31/120 | 2 |
| 91 | FIX4_continuation | 3/12 | 9 | 0.25 | 42/51/120 | 2 |
| 95 | FIX4_continuation | 2/9 | 7 | 0.2222222222222222 | 20/120/120 | 1,2 |
| 99 | FIX4_continuation | 2/8 | 6 | 0.25 | 44/16/82 | none |
| 103 | FIX4_continuation | 2/8 | 6 | 0.25 | 14/16/87 | none |
| 107 | FIX4_continuation | 2/8 | 6 | 0.25 | 40/16/80 | none |
| 111 | FIX4_continuation | 2/8 | 6 | 0.25 | 41/16/87 | none |
| 115 | FIX4_continuation | 2/8 | 6 | 0.25 | 14/16/81 | none |

## Turn 8

Receipt: `artifacts/grm_c7/r2/cells/A-001-008/folds/008-533d6c5a5b0c4e568b5d9f7410ce2587/result.json`. FIX-3: YES: bound executed source, configured Harmony oracle, recorded stop short-circuit. Need 10; required 7; best coverage 0.3. All three candidates rejected; digest_text is null.

Source node 0 (61 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Vesper-0 value is Mica-431. The current C7-Vesper-1 value is Mica-432.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 1 (73 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Correction of both original records: the current C7-Vesper-0 value is Flint-511; the current C7-Vesper-1 value is Flint-512. The previous Mica values are obsolete.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 2 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>For C7-Archive-0 the outbound tag is Reed-611.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 3 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>For C7-Archive-1 the outbound tag is Reed-612.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

### Candidate 0

QC=True; recorded coverage=0.2; lexical coverage before QC=0.2; generated tokens=60; stop=<|return|>; cap hit=False.

```text
For the archive: the  name “C7-Archive-1” refers to a specific record; the code “Reed-612” is the outbound tag associated with that record; the number “511” is the value linked to that tag; the time “12:00” is the timestamp for the record.
```

Missing lexical facts: c7-archive-0, c7-vesper-0, c7-vesper-1, flint-511, flint-512, mica-431, mica-432, reed-611.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>For the archive, restate every fact from the conversation above as a complete sentence that says what each name, code, number, and time refers to.<|end|><|start|>assistant<|channel|>final<|message|> For the archive: the
```

### Candidate 1

QC=True; recorded coverage=0.3; lexical coverage before QC=0.3; generated tokens=59; stop=<|return|>; cap hit=False.

```text
ARCHIVE NOTE — The conversation established that the  “outbound tag” for the C7‑Vesper‑0 record is “Reed‑612,” and the “outbound tag” for the C7‑Vesper‑1 record is “Reed‑512.” The prior “Mica” values are no longer relevant.
```

Missing lexical facts: c7-archive-0, c7-archive-1, flint-511, flint-512, mica-431, mica-432, reed-611.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Write a brief archive note covering everything above in complete sentences, preserving every name, code, number, and time verbatim and stating what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> ARCHIVE NOTE — The conversation established that the
```

### Candidate 2

QC=False; recorded coverage=0.0; lexical coverage before QC=0.2; generated tokens=67; stop=<|return|>; cap hit=False.

```text
The facts to archive are:   

- **C7‑Vesper‑0** – the outbound tag that is currently in use.  
- **Reed‑612** – the value that the tag has been updated to.  
- **Mica** – the previous tag that has been replaced.  

These are the only items that have been mentioned.
```

Missing lexical facts: c7-archive-0, c7-archive-1, c7-vesper-1, flint-511, flint-512, mica-431, mica-432, reed-611.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>List every fact from the conversation above: every name, code, number, and time, and what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> The facts to archive are:
```

## Turn 17

Receipt: `artifacts/grm_c7/r2/cells/A-017-023/folds/017-ac4a09b2490441678f6a5e9289942d5d/result.json`. FIX-3: YES: bound executed source, configured Harmony oracle, recorded stop short-circuit. Need 10; required 7; best coverage 0.2. All three candidates rejected; digest_text is null.

Source node 4 (61 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-AliasBase-0 value is Jasper-711. The current C7-AliasBase-1 value is Jasper-712.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 5 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Fresh-0 value is Basalt-811.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 6 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Fresh-1 value is Basalt-812.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 7 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Fresh-2 value is Basalt-813.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

### Candidate 0

QC=True; recorded coverage=0.2; lexical coverage before QC=0.2; generated tokens=20; stop=<|return|>; cap hit=False.

```text
For the archive: the  name “C7‑Fresh‑2” refers to the value “Basalt‑813.”
```

Missing lexical facts: basalt-811, basalt-812, c7-aliasbase-0, c7-aliasbase-1, c7-fresh-0, c7-fresh-1, jasper-711, jasper-712.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>For the archive, restate every fact from the conversation above as a complete sentence that says what each name, code, number, and time refers to.<|end|><|start|>assistant<|channel|>final<|message|> For the archive: the
```

### Candidate 1

QC=True; recorded coverage=0.2; lexical coverage before QC=0.2; generated tokens=15; stop=<|return|>; cap hit=False.

```text
ARCHIVE NOTE — The conversation established that the  current C7‑Fresh‑2 value is Basalt‑813.
```

Missing lexical facts: basalt-811, basalt-812, c7-aliasbase-0, c7-aliasbase-1, c7-fresh-0, c7-fresh-1, jasper-711, jasper-712.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Write a brief archive note covering everything above in complete sentences, preserving every name, code, number, and time verbatim and stating what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> ARCHIVE NOTE — The conversation established that the
```

### Candidate 2

QC=False; recorded coverage=0.0; lexical coverage before QC=0.4; generated tokens=80; stop=<|return|>; cap hit=False.

```text
The facts to archive are:   

| Item | Reference | Meaning |  
|------|-----------|---------|  
| **C7‑Fresh‑2** |  | The current value is **Basalt‑812**. |  
| **C7‑Fresh‑1** |  | The current value is **Basalt‑811**. |  

No other names, codes, numbers or times were mentioned.
```

Missing lexical facts: basalt-813, c7-aliasbase-0, c7-aliasbase-1, c7-fresh-0, jasper-711, jasper-712.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>List every fact from the conversation above: every name, code, number, and time, and what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> The facts to archive are:
```

## Turn 26

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-024-031/folds/026-681f4264c9fa4c2c9a7096d30106a8a0/result.json`. FIX-3: YES: bound executed source, configured Harmony oracle, recorded stop short-circuit. Need 8; required 6; best coverage 0.25. All three candidates rejected; digest_text is null.

Source node 8 (49 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>C7-Signal-0 is an alias for C7-AliasBase-0.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 9 (49 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>C7-Signal-1 is an alias for C7-AliasBase-1.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 10 (49 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>C7-Signal-2 is an alias for C7-AliasBase-2.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 11 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Vesper-0 value is Onyx-911.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

### Candidate 0

QC=True; recorded coverage=0.25; lexical coverage before QC=0.25; generated tokens=21; stop=<|return|>; cap hit=False.

```text
For the archive: the  alias “C7-Signal-0” refers to “C7-AliasBase-2.”
```

Missing lexical facts: c7-aliasbase-0, c7-aliasbase-1, c7-signal-1, c7-signal-2, c7-vesper-0, onyx-911.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>For the archive, restate every fact from the conversation above as a complete sentence that says what each name, code, number, and time refers to.<|end|><|start|>assistant<|channel|>final<|message|> For the archive: the
```

### Candidate 1

QC=True; recorded coverage=0.25; lexical coverage before QC=0.25; generated tokens=22; stop=<|return|>; cap hit=False.

```text
ARCHIVE NOTE — The conversation established that the  “C7‑Signal‑2” is an alias for “C7‑AliasBase‑2.”
```

Missing lexical facts: c7-aliasbase-0, c7-aliasbase-1, c7-signal-0, c7-signal-1, c7-vesper-0, onyx-911.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Write a brief archive note covering everything above in complete sentences, preserving every name, code, number, and time verbatim and stating what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> ARCHIVE NOTE — The conversation established that the
```

### Candidate 2

QC=False; recorded coverage=0.0; lexical coverage before QC=0.375; generated tokens=102; stop=<|return|>; cap hit=False.

```text
The facts to archive are:   

| **Item** | **Value** | **What it points to** |  
|-----------|-----------|-----------------------|  
| **C7‑Signal‑0** | “Onyx‑911” | The alias that is used for the signal. |  
| **C7‑AliasBase‑0** | “Onyx‑911” | The base alias that the signal points to. |  

These are the only names, codes, numbers, and times that have been mentioned.
```

Missing lexical facts: c7-aliasbase-1, c7-aliasbase-2, c7-signal-1, c7-signal-2, c7-vesper-0.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>List every fact from the conversation above: every name, code, number, and time, and what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> The facts to archive are:
```

## Turn 55

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-048-055/folds/055-1b44742dab454a6685c75b4e4a614895/result.json`. FIX-3: YES: bound executed source, configured Harmony oracle, recorded stop short-circuit. Need 8; required 6; best coverage 0.25. All three candidates rejected; digest_text is null.

Source node 13 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Vesper-1 value is Onyx-912.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 15 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Vesper-2 value is Onyx-913.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 16 (47 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>For C7-Archive-0 the inbound tag is Cedar-1011.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 17 (47 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>For C7-Archive-1 the inbound tag is Cedar-1012.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

### Candidate 0

QC=True; recorded coverage=0.25; lexical coverage before QC=0.25; generated tokens=63; stop=<|return|>; cap hit=False.

```text
For the archive: the  name “C7-Archive-1” refers to a specific item or record, the code “Cedar-1012” is an identifier linked to that item, the number “1012” is part of that identifier, and the time “C7” is a label or tag associated with it.
```

Missing lexical facts: c7-archive-0, c7-vesper-1, c7-vesper-2, cedar-1011, onyx-912, onyx-913.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>For the archive, restate every fact from the conversation above as a complete sentence that says what each name, code, number, and time refers to.<|end|><|start|>assistant<|channel|>final<|message|> For the archive: the
```

### Candidate 1

QC=True; recorded coverage=0.25; lexical coverage before QC=0.25; generated tokens=63; stop=<|return|>; cap hit=False.

```text
ARCHIVE NOTE — The conversation established that the  inbound tag for the C7‑Vesper‑2 system is “Cedar‑1012.”  The user’s statement was recorded verbatim.  The system’s tag is a code that identifies the inbound tag for the C7‑Vesper‑2 system.  The user’s statement was recorded.
```

Missing lexical facts: c7-archive-0, c7-archive-1, c7-vesper-1, cedar-1011, onyx-912, onyx-913.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Write a brief archive note covering everything above in complete sentences, preserving every name, code, number, and time verbatim and stating what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> ARCHIVE NOTE — The conversation established that the
```

### Candidate 2

QC=True; recorded coverage=0.25; lexical coverage before QC=0.25; generated tokens=51; stop=<|return|>; cap hit=False.

```text
The facts to archive are:   

- **C7‑Archive‑1** – the name of the archive.  
- **Cedar‑1012** – the inbound tag that is linked to that archive.  

No other names, codes, numbers or times were mentioned.
```

Missing lexical facts: c7-archive-0, c7-vesper-1, c7-vesper-2, cedar-1011, onyx-912, onyx-913.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>List every fact from the conversation above: every name, code, number, and time, and what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> The facts to archive are:
```

## Turn 74

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-071-077/folds/074-542293a8731d432da4e42a6af5120f3c/result.json`. FIX-3: YES: bound executed source, configured Harmony oracle, recorded stop short-circuit. Need 8; required 6; best coverage 0.25. All three candidates rejected; digest_text is null.

Source node 18 (47 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>For C7-Archive-2 the inbound tag is Cedar-1013.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 33 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-00 value is Pearl-1200.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 38 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-01 value is Pearl-1201.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 43 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-02 value is Pearl-1202.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

### Candidate 0

QC=True; recorded coverage=0.25; lexical coverage before QC=0.25; generated tokens=16; stop=<|return|>; cap hit=False.

```text
For the archive: the  C7‑Outpost‑02 refers to the Pearl‑1202.
```

Missing lexical facts: c7-archive-2, c7-outpost-00, c7-outpost-01, cedar-1013, pearl-1200, pearl-1201.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>For the archive, restate every fact from the conversation above as a complete sentence that says what each name, code, number, and time refers to.<|end|><|start|>assistant<|channel|>final<|message|> For the archive: the
```

### Candidate 1

QC=True; recorded coverage=0.25; lexical coverage before QC=0.25; generated tokens=16; stop=<|return|>; cap hit=False.

```text
ARCHIVE NOTE — The conversation established that the  current C7‑Outpost‑02 value is Pearl‑1202.
```

Missing lexical facts: c7-archive-2, c7-outpost-00, c7-outpost-01, cedar-1013, pearl-1200, pearl-1201.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Write a brief archive note covering everything above in complete sentences, preserving every name, code, number, and time verbatim and stating what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> ARCHIVE NOTE — The conversation established that the
```

### Candidate 2

QC=False; recorded coverage=0.0; lexical coverage before QC=0.25; generated tokens=59; stop=<|return|>; cap hit=False.

```text
The facts to archive are:   

| Item | Value | Reference |  
|------|-------|------------|  
| **C7‑Outpost‑02** | “Pearl‑1202” | The current value of the outpost. |  

No other names, codes, numbers or times were mentioned.
```

Missing lexical facts: c7-archive-2, c7-outpost-00, c7-outpost-01, cedar-1013, pearl-1200, pearl-1201.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>List every fact from the conversation above: every name, code, number, and time, and what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> The facts to archive are:
```

## Turn 85

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-078-085/folds/085-d15b5a88d2d94e56a2915cde4a8b0dfa/result.json`. FIX-3: YES: bound executed source, configured Harmony oracle, recorded stop short-circuit. Need 9; required 7; best coverage 0.3333333333333333. All three candidates rejected; digest_text is null.

Source node 48 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-03 value is Pearl-1203.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 53 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-04 value is Pearl-1204.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 58 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-05 value is Pearl-1205.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 60 (63 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-06 value is Slate-1206. C7-Handle-06 is an alias for C7-Outpost-06.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

### Candidate 0

QC=True; recorded coverage=0.3333333333333333; lexical coverage before QC=0.3333333333333333; generated tokens=37; stop=<|return|>; cap hit=False.

```text
For the archive: the  name “C7-Outpost-06” refers to the value “Pearl-1205”; the name “C7-Handle-06” refers to the same value.
```

Missing lexical facts: c7-outpost-03, c7-outpost-04, c7-outpost-05, pearl-1203, pearl-1204, slate-1206.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>For the archive, restate every fact from the conversation above as a complete sentence that says what each name, code, number, and time refers to.<|end|><|start|>assistant<|channel|>final<|message|> For the archive: the
```

### Candidate 1

QC=True; recorded coverage=0.3333333333333333; lexical coverage before QC=0.3333333333333333; generated tokens=31; stop=<|return|>; cap hit=False.

```text
ARCHIVE NOTE — The conversation established that the  current C7‑Outpost‑06 value is Pearl‑1205.  The alias C7‑Handle‑06 refers to the same value.
```

Missing lexical facts: c7-outpost-03, c7-outpost-04, c7-outpost-05, pearl-1203, pearl-1204, slate-1206.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Write a brief archive note covering everything above in complete sentences, preserving every name, code, number, and time verbatim and stating what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> ARCHIVE NOTE — The conversation established that the
```

### Candidate 2

QC=False; recorded coverage=0.0; lexical coverage before QC=0.3333333333333333; generated tokens=120; stop=None; cap hit=True.

```text
The facts to archive are:   

|  Item |  Meaning |  Notes |  
|---|---|---|  
| **C7‑Outpost‑06** |  The current “outpost” code that the user is referencing. |  The user says it is “Pearl‑1205.” |  
| **Pearl‑1205** |  The value that the user says is associated with the outpost. |  No other context is given. |  
| **C7‑Handle‑06** |  The alias that the user says refers to the same outpost. |
```

Missing lexical facts: c7-outpost-03, c7-outpost-04, c7-outpost-05, pearl-1203, pearl-1204, slate-1206.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>List every fact from the conversation above: every name, code, number, and time, and what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> The facts to archive are:
```

## Turn 91

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-086-093/folds/091-b0a742d7fa0f498ea4ab9d653be59f4c/result.json`. FIX-3: YES: bound executed source, configured Harmony oracle, recorded stop short-circuit. Need 12; required 9; best coverage 0.25. All three candidates rejected; digest_text is null.

Source node 61 (63 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-07 value is Slate-1207. C7-Handle-07 is an alias for C7-Outpost-07.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 62 (63 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-08 value is Slate-1208. C7-Handle-08 is an alias for C7-Outpost-08.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 63 (63 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-09 value is Slate-1209. C7-Handle-09 is an alias for C7-Outpost-09.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 64 (63 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-10 value is Slate-1210. C7-Handle-10 is an alias for C7-Outpost-10.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

### Candidate 0

QC=True; recorded coverage=0.25; lexical coverage before QC=0.25; generated tokens=42; stop=<|return|>; cap hit=False.

```text
For the archive: the  name “C7-Outpost-10” refers to the value “Slate-1209”; the name “C7-Handle-10” is an alias for “C7-Outpost-10.”
```

Missing lexical facts: c7-handle-07, c7-handle-08, c7-handle-09, c7-outpost-07, c7-outpost-08, c7-outpost-09, slate-1207, slate-1208, slate-1210.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>For the archive, restate every fact from the conversation above as a complete sentence that says what each name, code, number, and time refers to.<|end|><|start|>assistant<|channel|>final<|message|> For the archive: the
```

### Candidate 1

QC=True; recorded coverage=0.25; lexical coverage before QC=0.25; generated tokens=51; stop=<|return|>; cap hit=False.

```text
ARCHIVE NOTE — The conversation established that the  current value for the code **C7‑Outpost‑10** is **Slate‑1209**.  The handle **C7‑Handle‑10** is an alias that points to the same value, **Slate‑1209**.
```

Missing lexical facts: c7-handle-07, c7-handle-08, c7-handle-09, c7-outpost-07, c7-outpost-08, c7-outpost-09, slate-1207, slate-1208, slate-1210.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Write a brief archive note covering everything above in complete sentences, preserving every name, code, number, and time verbatim and stating what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> ARCHIVE NOTE — The conversation established that the
```

### Candidate 2

QC=False; recorded coverage=0.0; lexical coverage before QC=0.3333333333333333; generated tokens=120; stop=None; cap hit=True.

```text
The facts to archive are:   

| **Name / Code** | **Value** | **What it points to** | **Notes** |
|------------------|-----------|-----------------------|-----------|
| **C7‑Outpost‑10** | **Slate‑1207** | The current value of the outpost. |  |
| **C7‑Handle‑10** | **Alias for C7‑Outpost‑10** |  |
| **C7‑Outpost‑09** | **Slate‑1207** | The outpost’s value. |  |
| **C7‑Handle
```

Missing lexical facts: c7-handle-07, c7-handle-08, c7-handle-09, c7-outpost-07, c7-outpost-08, slate-1208, slate-1209, slate-1210.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>List every fact from the conversation above: every name, code, number, and time, and what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> The facts to archive are:
```

## Turn 95

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-094-100/folds/095-3df9be1aee50442c8a0fe437490940c1/result.json`. FIX-3: YES: bound executed source, configured Harmony oracle, recorded stop short-circuit. Need 9; required 7; best coverage 0.2222222222222222. All three candidates rejected; digest_text is null.

Source node 65 (63 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-11 value is Slate-1211. C7-Handle-11 is an alias for C7-Outpost-11.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 66 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-12 value is Slate-1212.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 67 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-13 value is Slate-1213.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 68 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-14 value is Slate-1214.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

### Candidate 0

QC=True; recorded coverage=0.2222222222222222; lexical coverage before QC=0.2222222222222222; generated tokens=20; stop=<|return|>; cap hit=False.

```text
For the archive: the  name “C7-Outpost-14” refers to the value “Slate-1212.”
```

Missing lexical facts: c7-handle-11, c7-outpost-11, c7-outpost-12, c7-outpost-13, slate-1211, slate-1213, slate-1214.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>For the archive, restate every fact from the conversation above as a complete sentence that says what each name, code, number, and time refers to.<|end|><|start|>assistant<|channel|>final<|message|> For the archive: the
```

### Candidate 1

QC=True; recorded coverage=0.2222222222222222; lexical coverage before QC=0.2222222222222222; generated tokens=120; stop=None; cap hit=True.

```text
ARCHIVE NOTE — The conversation established that the  current C7‑Outpost‑12 value is Slate‑1212. The user’s statement that “C7‑Outpost‑12 is Slate‑1212” was recorded verbatim. The system’s reply, “The current C7‑Outpost‑12 value is Slate‑1212,” was also recorded. The user’s statement that “C7‑Outpost‑12 is Slate‑1212” was recorded. The system’s reply, “The current C7‑Outpost‑12 value is Slate‑1212,” was recorded. The user’s statement that “C7
```

Missing lexical facts: c7-handle-11, c7-outpost-11, c7-outpost-13, c7-outpost-14, slate-1211, slate-1213, slate-1214.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Write a brief archive note covering everything above in complete sentences, preserving every name, code, number, and time verbatim and stating what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> ARCHIVE NOTE — The conversation established that the
```

### Candidate 2

QC=False; recorded coverage=0.0; lexical coverage before QC=0.3333333333333333; generated tokens=120; stop=None; cap hit=True.

```text
The facts to archive are:   

| Item | Reference | Meaning |  
|------|-----------|---------|  
| **C7‑Outpost‑14** | 1 | The current “value” of the outpost, which is **Slate‑1214**. |  
| **Slate‑1214** | 2 | The value that the outpost is currently set to. |  
| **C7‑Handle‑11** | 3 | An alias that points to the same value as **C7‑Outpost‑14**. |  

No other names, codes, numbers or times were
```

Missing lexical facts: c7-outpost-11, c7-outpost-12, c7-outpost-13, slate-1211, slate-1212, slate-1213.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>List every fact from the conversation above: every name, code, number, and time, and what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> The facts to archive are:
```

## Turn 99

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-094-100/folds/099-71a7a9d8b7324a65934f918d78828724/result.json`. FIX-3: YES: bound executed source, configured Harmony oracle, recorded stop short-circuit. Need 8; required 6; best coverage 0.25. All three candidates rejected; digest_text is null.

Source node 69 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-15 value is Slate-1215.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 70 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-16 value is Slate-1216.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 71 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-17 value is Slate-1217.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 72 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-18 value is Slate-1218.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

### Candidate 0

QC=True; recorded coverage=0.25; lexical coverage before QC=0.25; generated tokens=44; stop=<|return|>; cap hit=False.

```text
For the archive: the  name “C7‑Outpost‑17” refers to a location, the code “Slate‑1218” refers to a value, and the time “10:00 AM” refers to a time.
```

Missing lexical facts: c7-outpost-15, c7-outpost-16, c7-outpost-18, slate-1215, slate-1216, slate-1217.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>For the archive, restate every fact from the conversation above as a complete sentence that says what each name, code, number, and time refers to.<|end|><|start|>assistant<|channel|>final<|message|> For the archive: the
```

### Candidate 1

QC=True; recorded coverage=0.25; lexical coverage before QC=0.25; generated tokens=16; stop=<|return|>; cap hit=False.

```text
ARCHIVE NOTE — The conversation established that the  current C7‑Outpost‑18 value is Slate‑1218.
```

Missing lexical facts: c7-outpost-15, c7-outpost-16, c7-outpost-17, slate-1215, slate-1216, slate-1217.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Write a brief archive note covering everything above in complete sentences, preserving every name, code, number, and time verbatim and stating what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> ARCHIVE NOTE — The conversation established that the
```

### Candidate 2

QC=False; recorded coverage=0.0; lexical coverage before QC=0.25; generated tokens=82; stop=<|return|>; cap hit=False.

```text
The facts to archive are:   

| Item | Reference | Meaning |  
|------|-----------|---------|  
| **C7‑Outpost‑18** | 1 | The current “value” or designation of the outpost. |  
| **Slate‑1217** | 2 | The value that the outpost is currently set to. |  

No other names, codes, numbers or times were mentioned.
```

Missing lexical facts: c7-outpost-15, c7-outpost-16, c7-outpost-17, slate-1215, slate-1216, slate-1218.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>List every fact from the conversation above: every name, code, number, and time, and what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> The facts to archive are:
```

## Turn 103

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-101-108/folds/103-2d04027c210443289c63a9655b90470e/result.json`. FIX-3: YES: bound executed source, configured Harmony oracle, recorded stop short-circuit. Need 8; required 6; best coverage 0.25. All three candidates rejected; digest_text is null.

Source node 73 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-19 value is Slate-1219.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 74 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-20 value is Slate-1220.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 75 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-21 value is Slate-1221.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 76 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-22 value is Slate-1222.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

### Candidate 0

QC=True; recorded coverage=0.25; lexical coverage before QC=0.25; generated tokens=14; stop=<|return|>; cap hit=False.

```text
For the archive: the  C7‑Outpost‑22 is Slate‑1222.
```

Missing lexical facts: c7-outpost-19, c7-outpost-20, c7-outpost-21, slate-1219, slate-1220, slate-1221.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>For the archive, restate every fact from the conversation above as a complete sentence that says what each name, code, number, and time refers to.<|end|><|start|>assistant<|channel|>final<|message|> For the archive: the
```

### Candidate 1

QC=True; recorded coverage=0.25; lexical coverage before QC=0.25; generated tokens=16; stop=<|return|>; cap hit=False.

```text
ARCHIVE NOTE — The conversation established that the  current C7‑Outpost‑22 value is Slate‑1222.
```

Missing lexical facts: c7-outpost-19, c7-outpost-20, c7-outpost-21, slate-1219, slate-1220, slate-1221.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Write a brief archive note covering everything above in complete sentences, preserving every name, code, number, and time verbatim and stating what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> ARCHIVE NOTE — The conversation established that the
```

### Candidate 2

QC=False; recorded coverage=0.0; lexical coverage before QC=0.25; generated tokens=87; stop=<|return|>; cap hit=False.

```text
The facts to archive are:   

| Item | Reference | Meaning |  
|------|-----------|---------|  
| **C7‑Outpost‑22** | 1 | The current “value” of the outpost, a code that is being tracked. |  
| **Slate‑1221** | 2 | The value that the outpost is currently set to. |  

No other names, codes, numbers or times were mentioned.
```

Missing lexical facts: c7-outpost-19, c7-outpost-20, c7-outpost-21, slate-1219, slate-1220, slate-1222.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>List every fact from the conversation above: every name, code, number, and time, and what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> The facts to archive are:
```

## Turn 107

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-101-108/folds/107-037ea1021147420cae7505b99d7bc897/result.json`. FIX-3: YES: bound executed source, configured Harmony oracle, recorded stop short-circuit. Need 8; required 6; best coverage 0.25. All three candidates rejected; digest_text is null.

Source node 77 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-23 value is Slate-1223.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 78 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-24 value is Slate-1224.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 79 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-25 value is Slate-1225.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 80 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-26 value is Slate-1226.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

### Candidate 0

QC=True; recorded coverage=0.25; lexical coverage before QC=0.25; generated tokens=40; stop=<|return|>; cap hit=False.

```text
For the archive: the  name “C7-Outpost-26” refers to a location, the code “Slate-1226” refers to a value, and the time “1226” refers to a time.
```

Missing lexical facts: c7-outpost-23, c7-outpost-24, c7-outpost-25, slate-1223, slate-1224, slate-1225.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>For the archive, restate every fact from the conversation above as a complete sentence that says what each name, code, number, and time refers to.<|end|><|start|>assistant<|channel|>final<|message|> For the archive: the
```

### Candidate 1

QC=True; recorded coverage=0.25; lexical coverage before QC=0.25; generated tokens=16; stop=<|return|>; cap hit=False.

```text
ARCHIVE NOTE — The conversation established that the  current C7‑Outpost‑26 value is Slate‑1226.
```

Missing lexical facts: c7-outpost-23, c7-outpost-24, c7-outpost-25, slate-1223, slate-1224, slate-1225.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Write a brief archive note covering everything above in complete sentences, preserving every name, code, number, and time verbatim and stating what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> ARCHIVE NOTE — The conversation established that the
```

### Candidate 2

QC=False; recorded coverage=0.0; lexical coverage before QC=0.25; generated tokens=80; stop=<|return|>; cap hit=False.

```text
The facts to archive are:   

| Item | Reference | Meaning |  
|------|-----------|---------|  
| **C7‑Outpost‑26** | 1 | The current “value” or designation of the outpost. |  
| **Slate‑1225** | 2 | The value that the outpost was assigned. |  

No other names, codes, numbers or times were mentioned.
```

Missing lexical facts: c7-outpost-23, c7-outpost-24, c7-outpost-25, slate-1223, slate-1224, slate-1226.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>List every fact from the conversation above: every name, code, number, and time, and what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> The facts to archive are:
```

## Turn 111

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-109-116/folds/111-3ea748aa13dd40cfb399e9c3b3b7ce82/result.json`. FIX-3: YES: bound executed source, configured Harmony oracle, recorded stop short-circuit. Need 8; required 6; best coverage 0.25. All three candidates rejected; digest_text is null.

Source node 81 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-27 value is Slate-1227.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 82 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-28 value is Slate-1228.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 83 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-29 value is Slate-1229.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 84 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-30 value is Slate-1230.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

### Candidate 0

QC=True; recorded coverage=0.25; lexical coverage before QC=0.25; generated tokens=41; stop=<|return|>; cap hit=False.

```text
For the archive: the  name “C7-Outpost-29” refers to a location, the code “Slate-1229” refers to a value, and the time “12:29” refers to a time.
```

Missing lexical facts: c7-outpost-27, c7-outpost-28, c7-outpost-30, slate-1227, slate-1228, slate-1230.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>For the archive, restate every fact from the conversation above as a complete sentence that says what each name, code, number, and time refers to.<|end|><|start|>assistant<|channel|>final<|message|> For the archive: the
```

### Candidate 1

QC=True; recorded coverage=0.25; lexical coverage before QC=0.25; generated tokens=16; stop=<|return|>; cap hit=False.

```text
ARCHIVE NOTE — The conversation established that the  current C7‑Outpost‑29 value is Slate‑1228.
```

Missing lexical facts: c7-outpost-27, c7-outpost-28, c7-outpost-30, slate-1227, slate-1229, slate-1230.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Write a brief archive note covering everything above in complete sentences, preserving every name, code, number, and time verbatim and stating what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> ARCHIVE NOTE — The conversation established that the
```

### Candidate 2

QC=False; recorded coverage=0.0; lexical coverage before QC=0.25; generated tokens=87; stop=<|return|>; cap hit=False.

```text
The facts to archive are:   

| Item | Reference | Meaning |  
|------|-----------|---------|  
| **C7‑Outpost‑29** | 1 | The current “value” of the outpost, a code that is being tracked. |  
| **Slate‑1228** | 2 | The value that the outpost is currently set to. |  

No other names, codes, numbers or times were mentioned.
```

Missing lexical facts: c7-outpost-27, c7-outpost-28, c7-outpost-30, slate-1227, slate-1229, slate-1230.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>List every fact from the conversation above: every name, code, number, and time, and what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> The facts to archive are:
```

## Turn 115

Receipt: `artifacts/grm_c7/r2/fix4_attempt_1/cells/A-109-116/folds/115-636a4f13086f4af6bdbd6dbf0b5d1614/result.json`. FIX-3: YES: bound executed source, configured Harmony oracle, recorded stop short-circuit. Need 8; required 6; best coverage 0.25. All three candidates rejected; digest_text is null.

Source node 85 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-31 value is Slate-1231.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 86 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-32 value is Slate-1232.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 87 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-33 value is Slate-1233.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

Source node 88 (46 tokens):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>The current C7-Outpost-34 value is Slate-1234.<|end|><|start|>assistant<|channel|>final<|message|>Recorded.<|end|>
```

### Candidate 0

QC=True; recorded coverage=0.25; lexical coverage before QC=0.25; generated tokens=14; stop=<|return|>; cap hit=False.

```text
For the archive: the  C7‑Outpost‑32 is Slate‑1234.
```

Missing lexical facts: c7-outpost-31, c7-outpost-33, c7-outpost-34, slate-1231, slate-1232, slate-1233.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>For the archive, restate every fact from the conversation above as a complete sentence that says what each name, code, number, and time refers to.<|end|><|start|>assistant<|channel|>final<|message|> For the archive: the
```

### Candidate 1

QC=True; recorded coverage=0.25; lexical coverage before QC=0.25; generated tokens=16; stop=<|return|>; cap hit=False.

```text
ARCHIVE NOTE — The conversation established that the  current C7‑Outpost‑32 value is Slate‑1234.
```

Missing lexical facts: c7-outpost-31, c7-outpost-33, c7-outpost-34, slate-1231, slate-1232, slate-1233.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>Write a brief archive note covering everything above in complete sentences, preserving every name, code, number, and time verbatim and stating what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> ARCHIVE NOTE — The conversation established that the
```

### Candidate 2

QC=False; recorded coverage=0.0; lexical coverage before QC=0.25; generated tokens=81; stop=<|return|>; cap hit=False.

```text
The facts to archive are:   

| Item | Reference | Meaning |  
|------|-----------|---------|  
| **C7‑Outpost‑32** | 1 | The current “value” or identifier for the outpost. |  
| **Slate‑1234** | 1 | The value that was assigned to the outpost. |  

No other names, codes, numbers or times were mentioned.
```

Missing lexical facts: c7-outpost-31, c7-outpost-33, c7-outpost-34, slate-1231, slate-1232, slate-1233.

Wrapped prompt (source-bound reconstruction, not a recorded input):
```text
<|start|>system<|message|>You are ChatGPT. Reasoning: low. Valid channel: final.<|end|><|start|>user<|message|>List every fact from the conversation above: every name, code, number, and time, and what each one refers to.<|end|><|start|>assistant<|channel|>final<|message|> The facts to archive are:
```
