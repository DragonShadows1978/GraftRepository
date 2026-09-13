# GRM-XM1 X2 — Qwen final decode and receipt diagnosis

Implemented and CPU-validated final-channel handling; GPU parity **not claimed fixed**.
No GPU rerun occurred. The original thinking-window falsifier remains RED.
The fresh-fact comparisons have a second defect: **Praxis C5 contains no
Quartz-8-Jade fact; Solace C3l mounts Tundra's split child, with no Raven-9-Ivory
fact.** These are historical source-text contrasts, not same-fact parity tests.

## 1. Files:lines; final-channel mechanism and OFF byte-identity

- `scripts/grm_xm1_qwen_final.py:15`: strict GRM_QWEN35_FINAL_CHANNEL resolver,
  default ON for XM1 Qwen cells; invalid flag is an error.
- `scripts/grm_xm1_qwen_final.py:54`: Qwen adapter uses native
  `apply_chat_template(..., enable_thinking=False)`, emitting the template's
  closed empty thinking block. Capture/feed texts stay byte-identical.
  Generation remains greedy and bounded at 32 tokens. Delimiters/thoughts are
  filtered by token IDs, raw text/tokens/observations retained. No final or
  no EOS completion is NO_FINAL / TRUNCATED_FINAL, never parity PASS.
- `scripts/grm_xm1_qwen_final.py:24`: final content token index i selects
  observation i+1 (the token's **input query**). Excludes question prefill,
  thinking, controls/EOS and leading boundary whitespace. The last token gets
  a flush forward when the cap truncates generation. This is an explicit
  change from XM1's predictor-row window. RS4 key-band partition stays intact;
  its raw answer-band keys mean generated history, possibly including thought.
- `scripts/grm_xm1_parity.py:336`: dispatches only Qwen through the new helper;
  `:345` preserves the legacy body. OFF calls that body without template
  kwargs, extra forwards, token filtering or added result fields.
- `scripts/grm_xm1_parity.py:44`: separate XM2 source-pin overlay;
  `:70` preserves explicit OFF through legacy environment stripping;
  `:436` binds final-mode Qwen receipts to XM2; `:524` rejects incomplete final
  completion as PASS. Original XM1 registrations/amendment and GPU bytes stay
  unchanged; the exact old worker is retained under `sources/`.
- `scripts/grm_xm1_cpu.py:67`: CPU codec accepts the template keyword; its
  previous rendered output is unchanged. `tests/test_grm_xm1_amendment_1.py:120`
  now proves the old worker hash against its preserved source and current
  worker hash against the separate XM2 pin. No numeric tolerance weakened.
- `tests/test_grm_xm2_final.py:29`: all ten real recorded traces select zero
  final positions. `:40` replays actual recorded reasoning tokens plus an
  explicitly synthetic final continuation through the CPU native-attention
  double; checks served value, input-query indices and independent raw-row
  mean. Full 32-token recorded prefix is separately checked with continuation
  by the selector. This does not invent an observed GPU final answer.
- `tests/test_grm_xm2_final.py:93`: ten C3l/C5 CPU cells compare full serialized
  results and exact forward-call sequences to the frozen pre-edit worker,
  byte-equal OFF. This is CPU protocol identity, not a new GPU byte-identity
  measurement. `:112` checks all four native capture/seat geometries.

## 2. Diagnosis table and hypotheses/falsifiers

Evidence class: copied GPU receipt audit plus CPU replay of the pinned local
Qwen tokenizer. Full numbers, text, IDs, token pieces and source receipt hashes:
`artifacts/grm_xm2/diagnosis.json`. No new model inference.

Full-attention layer index is zero-based. Cells below are **C5 fed minus C3l
mounted mass**, averaged over the original 32 thinking-window positions.
They cannot be relabeled final-answer measurements.

| Layer | Harbor | Praxis | Solace | Tundra | Meridian |
|---:|---:|---:|---:|---:|---:|
| 3 | +0.0107 | +0.2860 | +0.0404 | -0.0132 | -0.0336 |
| 7 | -0.0527 | +0.1644 | +0.1651 | -0.0151 | +0.0143 |
| 11 | -0.0139 | +0.2443 | +0.1811 | +0.0037 | +0.0235 |
| 15 | -0.0485 | +0.2177 | +0.2096 | -0.0274 | -0.0380 |
| 19 | -0.1296 | +0.1292 | +0.1861 | -0.1402 | -0.2003 |
| 23 | -0.0304 | +0.1582 | +0.1810 | -0.0295 | -0.1182 |
| 27 | +0.1187 | +0.2644 | +0.2709 | +0.1142 | -0.0426 |
| 31 | -0.0292 | +0.1434 | +0.1070 | -0.0340 | -0.0884 |
| Mean | -0.0219 | +0.2009 | +0.1677 | -0.0177 | -0.0604 |

Praxis is already deficient at the first full-attention layer (3), with gaps
>0.10 at all eight. Solace first exceeds 0.10 at layer 7 and stays above it.
Both show a large late-layer gap at 27, but neither deficit is only a deep
layer event. Harbor and Tundra also exceed 0.10 locally at 27 while satisfying
the registered aggregate criterion. This is a profile, not causal localization.

| Probe | C5 / C3l mean | Mounted source | Qwen ntok | Seat positions [start,end) | live_shift / inferred capture shift | C5 ntok |
|---|---|---|---:|---|---|---:|
| Harbor | 0.4913 / 0.5131 | harbor_b correction (installed 1 -> recaptured 4) | 48 | [231,279) | 279 / 279 | 183 |
| Praxis | 0.5490 / 0.3480 | praxis_fact (installed 0 -> recaptured 5) | 38 | [242,280) | 280 / 280 | 184 |
| Solace | 0.5049 / 0.3373 | Tundra split child (installed 4 -> recaptured 5) | 80 | [200,280) | 280 / 280 | 184 |
| Tundra | 0.5195 / 0.5372 | Tundra split child (installed 4 -> recaptured 5) | 80 | [200,280) | 280 / 280 | 184 |
| Meridian | 0.4703 / 0.5308 | Meridian split child (installed 5 -> recaptured 6) | 79 | [213,292) | 292 / 292 | 196 |

Node IDs above are **historical RS4 provenance**, not Qwen repository IDs:
XM1 synthesizes a Qwen payload by recapturing `capture_texts`, with no native
Qwen node installation. Physical mount band is [0,ntok), n_sink=0; RoPE
coordinates place its last token at live_shift-1. Native width is
max(capture/feed tokenizer lengths)+96. All five native captures explicitly
use that live shift. Every historical RS4 recapture also reports pin=live,
shift=115 (19 sink + 96 arena). Historical recaptured token lengths are
Harbor 64, Praxis 52, Solace/Tundra 75, Meridian 65; they are GPT-OSS counts,
not Qwen counts. Installed original capture pin is not fully proven by these
recapture receipts and those installed payloads are not reused by native XM1.
A differential fresh-node capture-pin explanation is therefore unsupported
for the actual Qwen run. RS3 lever effects themselves remain unmeasured here.

Tokenizer facts below use the local revision replayed against all ten receipt
source-ID sequences, exact equality. A leading space is included in identifier
and value tokenization; displayed pieces omit its visible BPE marker.

| Probe | Identifier pieces | Expected value pieces | Expected mentions C3l / C5 | Identifier mentions C3l / C5 |
|---|---|---|---:|---:|
| Harbor | Harbor / token | N / acre / - / 6 / - / Blue | 2 / 2 | 2 / 4 |
| Praxis | Praxis / dock | Quartz / - / 8 / -J / ade | 2 / 0 | 2 / 2 |
| Solace | Sol / ace / key | Raven / - / 9 / -I / v / ory | 0 / 2 | 1 / 4 |
| Tundra | T / undra / ledger | S / able / - / 0 / -C / opper | 3 / 3 | 3 / 3 |
| Meridian | Mer / idian / do / cket | Delta / - / 4 / - / Dr / ift | 3 / 3 | 3 / 3 |

Every C3l source has one user/assistant pair and six structural marker tokens;
every C5 source has two pairs and twelve markers. Each assistant demonstration
contains an empty think block. Qwen question windows originally have 18 tokens
(Harbor/Praxis), 19 (Solace/Tundra), 20 (Meridian). Capture scaffold differs:
Harbor uses Correction/replacing/Updated; Praxis uses The current/Recorded;
reserve children use longer ownership/preservation instructions and Archived.
Solace and Tundra have identical source IDs in each corresponding arm but ask
different questions. Their difference cannot be attributed to mount length or
identifier fragmentation alone. C5 feed-band mass also aggregates unrelated
text and is not fact-specific attribution. Missing DeltaNet recurrent state
remains another architectural difference between C3l and C5; no state-transfer
treatment is authorized or run here.

Frozen hypotheses and falsifiers (full wording in registration):

| ID | Hypothesis | Falsifier / pre-run status |
|---|---|---|
| H1 | Thinking-window contamination explains fresh deficits | Either fresh final, nonempty EOS-complete C5-minus-C3l mean remains >0.10. Untested; incomplete final is RED. |
| H2 | Fresh/comparator captures used different pins | All native captures use live; historical recaptures all live/115. Falsified as this attribution. Separate RS3 lever prediction is >=0.05 absolute mounted-mass movement on at least one fresh probe; falsified if both probe movements <0.05 at both seat settings. |
| H3 | Near-live seating improves fresh mass | Falsified if seat-on gain <0.05 on both fresh probes at each capture pin. Existing receipts already seat-on; treatment untested. |
| H4 | Unequal fact/scaffold sources confound parity | Falsifier is source replay showing both missing expected facts and matched sequences; audit instead confirms mismatches. Causal contribution unmeasured; same-text successor is outside registered matrix. |
| H5 | Identifier fragmentation alone separates failures | Falsified by one-name-token Praxis failing and split-name-token Tundra/Meridian passing aggregate parity. Other lexical effects untested. |

## 3. Registration path + SHA; lead commands

`artifacts/grm_xm2/registration.json`

SHA-256: `7d537ab25ba485094b0235c9b1407d75b542dcbba21ebdd587a6534d16d73c3d`

Frozen before any gate; no registration amendment was needed. Original XM1
registration SHA remains 758fd3ac1231ed44937714a86009b557f554f026e9d85773cb730074e04439e8.
The separate XM2 registration overlays only new implementation/source pins.

`artifacts/grm_xm2/lead_commands.txt`: sixteen repo-root commands, append
`--dry-run` for validation. Worker `scripts/grm_xm1_x2_run.py:23` checks all
pins and the ten original GPT-OSS reference receipts; dry-run takes no lease,
loads no model, queries no device. `:40` reserves before loading, leases
itself fail-fast, writes create-only attempt/result receipts, rejects changed
resume bindings, checks tokenizer/template hashes and model revision.
Do not use an outer flock. Explicit `--run` is required; `--dry-run` wins.

Five probes x C3l/C5 = ten baselines. Praxis/Solace each add capture off +
seat off, capture live + seat off, and capture off + seat on; baseline C3l
supplies live + on. Total sixteen unique cells, 110 s per reservation =
1,760 s (0.4889 GPU-h), including load/capture/decode. No retries or budget
extensions. Gap guard requires >=30 s after the previous reservation end;
the lead schedules commands accordingly, worker does not wait. Controlled
capture OFF means cleared native shift=0, not arbitrary production residue.

Runtime fit is unproven: copied warm cells took 64–69 s, but the first cold
Harbor C3l took 129.70 s, above the new 110 s per-cell cap. A cold attempt may
therefore return RED before completion. Short direct-final decoding may help
but is not measured. The budget is a hard cap, not a claim all 16 will finish.
The original >0.10 falsifier is retained; expected-correct-to-refusal changes
must be reported too. Because Praxis C5 lacks the fact, a refusal there is not
valid evidence of failure to read that fact. No GPU results for XM2 exist.

## Prior art

Qwen Team (2026), **Qwen3.5-9B chat template**, revision
c202236235762e1c871ad0ccb60c8ee5ba337b9a. Source actually read:
`artifacts/grm_xm2/sources/chat_template.jinja`, especially its final
`add_generation_prompt` / `enable_thinking` branch. Cached source verified;
upstream attribution **unverified — lead to check**: "Qwen3.5-9B
chat_template.jinja enable_thinking thinking mode".
Upstream lead: https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/chat_template.jinja
Taken: native empty think block and delimiter convention. Ours: final-token
input-query indexing, raw trace retention, and CPU replay/identity receipts.

GRM contributors (2026), RS3 capture pin/constant-delta seat, RS4 band
partition/mean, XM1/LT1 immutable cells and CMC1 GPU lease — directly reused.
Su et al. (2021), RoFormer rotary position composition is inherited through
XM1 seating; external attribution unverified — lead to check "RoFormer
2104.09864". No new attention algorithm or optimization is claimed. Prior-art
annotations also appear at code sites and in the ledger.

## 4. Deviations, RED, safety, seat; pytest LAST

- RED: original fresh-probe aggregate falsifier, thinking-only windows, source
  mismatches, and final GPU answer quality/parity remain unresolved.
  **Not claimed fixed**: Qwen GPU parity, same-fact equivalence, model quality,
  recurrent-state transfer, cold-start fit, or a successful full GPU panel.
- Scope choice: final decoding lives in the authorized XM1 adapter/harness
  scripts; `core/qwen35_tc.py` attention/model math is unchanged.
- CPU trace continuation is synthetic and explicitly labeled. Author tests
  are baseline only; no blind red team was dispatched (NO subagents).
- First CPU gate: 5 failed, 94 passed, 2 warnings in 8.24s, all failures from
  an omitted original runtime-frame pin. Restored exact-hash canonical bytes
  into this worktree; changed no registration or numerical tolerance.
  Affected recheck: 9 passed in 17.27s, including all 16 new and 60 old command
  dry-runs. Final full-suite receipt is appended below.
- Process safety: no git commands, no subagents, no GPU execution/device query,
  no background jobs/watchers/wait loops, no process killed/signaled, no live
  service or running grm-xm1 queue touched. Canonical and grm-xm1 only read.
  Foreground CPU child calls remained <10 min; tool polling only collected our
  own foreground pytest child. Test-generated temporary GPU-mode receipts
  are CPU stubs, never device results. Scratch cleanup is confined to our
  `artifacts/grm_xm2/tmp`; final parent only records output and cleans scratch.
- Seat as dispatched: **gpt-6-astra, high**; no separate runtime model-selector
  attestation was available. Target model: **Qwen/Qwen3.5-9B**, pinned revision
  above. No claim about another Qwen revision or model size.

Final command, verbatim (the final validation subprocess; only receipt writes
and owned scratch cleanup follow in its foreground parent):

```sh
python3 -m pytest -q --basetemp artifacts/grm_xm2/tmp tests/test_grm_xm1*.py tests/test_grm_xm2*.py
```

Final pytest receipt: **99 passed, 2 warnings in 22.76s**; exit code 0.
Raw output: `artifacts/grm_xm2/pytest_LAST.log`; structured receipt:
`artifacts/grm_xm2/pytest_LAST.json`. Owned basetemp removed: True. No XM2 GPU run directory: True.
