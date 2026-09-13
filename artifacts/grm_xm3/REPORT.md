# GRM-XM3 Trinity diagnosis — measurement confounded; serving remains RED

**Verdict (historical receipts + CPU source tests):** 0/10 correct served answers. The historical comparison does not test identical text: Praxis C5 lacks its answer fact; Solace C3l mounts the Tundra record. Decode-position averages cover EOS/BOS loops or garbled output. Band accounting and local-key rotation passed their CPU checks. The cause of the INT4/fp32 generation collapse is **not claimed fixed**; no GPU ran. The weight/compute path is documented at `scripts/trinity_nope_graft_width_sweep.py:609` and `:624`; a model-numerics diagnosis needs a valid live control and is unresolved here.

Seat: `gpt-6-astra`, effort `high`. Immutable plan: `orders/GRM_XM1_X3_TRINITY.md`. This report and `IMPLEMENTATION_LEDGER.md` are the X3 narrative and append-only receipts. Evidence labels below limit the claims.

## 1. Served-text table (10 cells) + template/geometry findings (file:line).

The served column is the complete stored `served` string, JSON-quoted without truncation. Full special-token-preserving decodes and exact generated IDs are in `historical_audit_amendment_1.json`; the original receipts remain untouched. Each receipt is `artifacts/grm_xm1/amendment_1_run/cells/gpu/trinity/{arm}__{probe}.json`.

| Probe | Arm | Complete served text | Generated steps | Raw-token diagnosis |
|---|---|---|---:|---|
| sup_harbor_restatement | C3l | `""` | 1 | EOS immediately: [3] |
| sup_praxis_fresh | C3l | `""` | 32 | 32 BOS tokens: [0] x 32 |
| sup_reserve_meridian_docket | C3l | `""` | 32 | 32 BOS tokens: [0] x 32 |
| sup_reserve_tundra_ledger | C3l | `"linehereThisnullstream.Stream,linehereAndscribe,linehereAndnullnullnullstream.Stream,route=\"0rangle,"` | 32 | garbled text plus BOS tokens |
| sup_solace_fresh | C3l | `"lessmith oneinternet nothereInnoc onerroribia &nullnullnullandverity Elder Brothers AndAndAndnullnullnullnullnullnull"` | 32 | garbled text plus BOS tokens |
| sup_harbor_restatement | C5 | `"lineAndnullstream.Stream.linerBreaker stretchnullstreamAsStreamection andlineAndnullnullstreamAsStreamection andliner With"` | 32 | garbled text plus BOS tokens |
| sup_praxis_fresh | C5 | `"nullstreamAsStreamection andstream.nullerves \""` | 32 | garbled text plus BOS tokens |
| sup_reserve_meridian_docket | C5 | `""` | 32 | 32 BOS tokens: [0] x 32 |
| sup_reserve_tundra_ledger | C5 | `"liner Inliner Inliner Inliner In"` | 32 | garbled text plus BOS tokens |
| sup_solace_fresh | C5 | `"nullnullstream waterway Elder Inliner Inliner Buses Elder Inlines.nullafari innullnullstream.Stream,nullstream.Stream,nullstream.Stream,And"` | 32 | garbled text plus BOS tokens |

**Template (local source replay, upstream unverified).** `scripts/grm_xm1_parity.py:90` converts Harmony role boundaries with the loaded tokenizer; question rendering is at `:334`. The shipped template is snapshotted at `artifacts/grm_xm3/evidence/chat_template.jinja:23` (user), `:25` (assistant), and `:63` (generation prompt). Its assistant prefix is `<|im_start|>assistant\n`; the blank lines around stored assistant content are produced by this shipped template and reproduce in all ten receipts. The audit re-encoded every source and question: **10/10 exact token-ID matches**, all rendered source strings exact, all skip-special served decodes exact. The wrong-native-template-selection hypothesis is refuted at this local replay boundary. No new system message or hand-written template was introduced. Fidelity to the model training template remains unverified.

Model source lead: [Arcee AI Trinity-Nano-Preview chat template](https://huggingface.co/arcee-ai/Trinity-Nano-Preview/blob/main/chat_template.jinja), **unverified — lead to check**. Both attempted web opens were rejected; no upstream content was obtained. The local model card identifies the chat-tuned Preview model (`evidence/model_README.md:31`, `:92`). Local filename `trinity-nano` is not proof of current upstream revision.

**Layer inventory and seating (source + CPU native-function AST).** Config `evidence/config.json:12`, `:18`, `:87`, `:97`: 56 layers, local window 2048, head dimension 128. Full NoPE layers (zero-based): **3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55**. The other 42 layers are local RoPE. The receipt inventory agrees.

`scripts/trinity_nope_graft_width_sweep.py:282` captures K **before RoPE** and V. `scripts/grm_xm1_parity.py:184` copies the payload and rotates local K by `delta = width - n`; `:200` explicitly skips NoPE K. The native injection at `scripts/trinity_nope_graft_width_sweep.py:310` applies row rotations `R(j)`. Their composition is `R(j) R(width-n) K_raw = R(width-n+j) K_raw` (reasoning, tested with an independent trigonometric oracle at both seats, tolerance 1e-5). Local K is **seated at the new position**, not left at capture position and not double-counting the capture rotation. NoPE K and all V retain exact bytes. Actual upstream hidden states/INT4 numerics are outside this CPU proof.

**Historical geometry (receipt evidence).** Source lengths are C3l/C5; question length is 16 throughout. `width=max(native capture count,native feed count)+96` at `scripts/grm_xm1_parity.py:338`, so 96 is additional padding, not the historical seat capacity. Historical C5 source positions start at width and its question at width+fed_n; the mount ends at width-1 and its question starts at width. Physical mount rows stay [0,n); logical RoPE position and physical partition index are different quantities. Uniform translation alone is not claimed a defect; unequal source contexts and unequal decode trajectories prevent the stronger inference.

| Probe | Source tokens C3l / C5 | Width | C3l logical start | C5 question start |
|---|---:|---:|---:|---:|
| sup_harbor_restatement | 43 / 162 | 258 | 215 | 420 |
| sup_praxis_fresh | 33 / 163 | 259 | 226 | 422 |
| sup_solace_fresh | 68 / 163 | 259 | 191 | 422 |
| sup_reserve_tundra_ledger | 68 / 163 | 259 | 191 | 422 |
| sup_reserve_meridian_docket | 64 / 169 | 265 | 201 | 434 |

**Token/content confound (receipt audit, case/dash normalized as the existing scorer).** Expected value is present in 8/10 source sets. Praxis C5 contains Solace and Tundra text, no Quartz-8-Jade; Solace C3l contains Tundra text, no Raven-9-Ivory. All five arm pairs differ in source IDs/counts. The original registration itself records this historical contrast in `artifacts/grm_xm1/registration.json` under `deviations_registered`; it cannot establish David's identical-file/tokenization premise.

**Answer window (source + decoded IDs).** `scripts/grm_xm1_parity.py:356` samples at most 32 tokens; `:232` observes the last query row. Step 0 is the final question/prefix token predicting the first output (answer band width zero). Later steps query the previous output to predict the next. `scripts/grm_rs4_row_split.py:227` calls their reduction “answer positions” without a semantic-answer gate. Harbor C3l averages only one EOS prediction versus 32 steps in C5; three other cells are 32 BOS tokens. None of the ten windows establishes a correct answer. These are decode-position means, not valid evidence that the model attended while answering correctly.

**Band decomposition (CPU arithmetic audit of all original rows).** **16,184/16,184** layer rows cover S with contiguous bands; zero probability/width failures. Maximum partition error 1.1920928955078125e-07; maximum subbands-minus-live error 1.341104507446289e-07. S ranges from 49 to 216, below the 2048 local eviction window. Thus this panel never reaches the unhandled evicted-cache geometry. Observer ignores the causal-mask flag but selects the final query; every physical key is causally visible there for these non-evicted rows. Native causal mask is bottom-right aligned (`evidence/tensor_functional.py:20`, `:31`; attention entry `evidence/trinity_nano_tc.py:122`). This audit refutes a band-sum error on these receipts; it does not certify the observer beyond this S range.

## 2. Hypotheses + falsifiers; fixes landed (flag, OFF byte-identity).

Immutable hypotheses/falsifiers are in `registration.json` (template, local relocation, fit96, capture pin, matched tokens/positions, decode window). Outcomes: local template selection, missing local re-rotation, and historical 96-token truncation are refuted within the CPU/source limits above. Historical answer-window validity is rejected. Capture-pin effects on real multilayer tensors and the cause of corrupted generation remain open. CPU fit is token/cache geometry only; no new GPU memory/quality claim.

The authorized repair is isolated in `scripts/grm_xm1_trinity_x3.py:59`: `--xm3-matched` defaults OFF and delegates to the original `native_cell` before mutation (`:61`). ON uses exactly the same captured record and token IDs in BOTH C5 and mounted treatments, aligns C5 prefix/question positions to the near-live mount (`:78`), receipts capture shift/digest/seat (`:100`), and labels semantic validity separately from decode steps (`:137`). It fails closed if source n exceeds 96 or total cache reaches the 2048 eviction seam. The original Trinity adapter and XM1 worker are unchanged, so their immutable baseline pins still validate.

CPU gates: `tests/test_grm_xm3_trinity.py:29` checks canonical serialized OFF result bytes for both arms on both probes; `:37` checks native injected local K against independent trig at off/on seats while proving payload/NoPE/V byte identity. `:65` checks all four lever combinations and matched source/question IDs. The coupled two-layer test (`:196`) propagates local attention output into NoPE attention and matches fed/mounted mass within 1e-6. Weight projections and engine are doubles; no language quality or full 56-layer weight parity is claimed. Negative tests cover fit, unknown pin, drift, missing/incorrect C5, budget, inter-lease gap and a mocked GPU entry point with a worker-owned lease/reservation. This is an author baseline, not blind verification.

## 3. Registration path + sha; lead commands.

Immutable registration: `artifacts/grm_xm3/registration.json`
`sha256 a29173ea186cbf13368c4b720dbd01c6d52600393f9f97ca584e784c5f3b5ea9`

Current implementation binding: `artifacts/grm_xm3/implementation_amendment_3.json`
`sha256 9298939093e3b0e801d315d53bc14d79f9839d0bf40839b9033e6366ce750dfd`

It chains to `implementation_amendment_2.json`, `implementation_amendment_1.json` and `implementation_pins.json`; original registrations and pins were not rewritten. Source pin keys are repo-relative. External native/template/config snapshots have repo-relative pins and read-only origin metadata. Model weights, tokenizer binary and the entire TensorCUDA transitive dependency tree are not exhaustively content-pinned; lead should preserve the inspected installation. The worker verifies the listed external live files against their pinned snapshots before loading a model.

Commands: `artifacts/grm_xm3/lead_commands.txt`, 14 lines, each exercised with appended `--dry-run` by the CPU suite. Historical GPT-OSS receipt barrier is preserved before GPU. Worker leases itself with `gpu_lease(120,0)`; **no outer flock**, no GPU launched here. Four legacy controls plus ten matched cells reserve at most **1680 s (0.4667 GPU-h)** under a 1800 s total cap. Whole reservations are charged even on failure, with no refunds. Worker refuses an invocation until previous reservation end plus 30 s; there is no background wait or polling. Commands are ordered legacy controls, matched C5, then mounted factorial. Historical controls may receipt RED; an incorrect matched C5 hard-blocks that probe's mounted arms. Timeouts, busy lease, drift, overflow, invalid bands and exhausted budget stop the run.

| Matched treatment | Capture pin | Near-live seat |
|---|---|---|
| C5 | live feed reference | source ends immediately before question |
| C0 | off | off |
| C1l | live | off |
| C2 | off | on |
| C3l | live | on |

Both probes use n_sink=0 and width=96. Native source lengths are Praxis=33, Harbor=43. C5 text starts at 63/53 respectively; question starts at 96. Capture OFF=0, LIVE=63/53; seat OFF=0, ON=63/53. NoPE K remains unrotated. This explicitly pins capture to the **actual live reference source start**, unlike historical XM1 capture at the question shift 259/258. That difference is registered, not a silent replay claim. Four CPU lever tests do not establish a model treatment effect.

Registered semantic parity falsifier: correct matched C5 required, then fed-minus-mounted full-layer mean >0.10 is a failure on that probe; wrong served values remain RED independently. The worker applies this registered falsifier through `parity_comparison` and writes `RED_PARITY` even if the value is correct; it also rejects missing/nonfinite metrics. Full and local masses, raw/served IDs, capture digest and seating are retained for every completed treatment. This compares free decode trajectories, not teacher-forced states. Since no GPU measurement ran, no winning arm, parity clearance or production flip is claimed.

## 4. Prior art, deviations, RED, process safety, model id + effort; pytest LAST.

### Prior art

- GRM contributors (2026): RS3 capture pin/near-live seating, RS4 row partition/reduction, XM1 native capture/seat and AST CPU doubles, LT1 immutable receipts, CMC1 self-leasing worker. Reused mechanisms; this work isolates the matched control and makes decode semantics explicit. Local source evidence at the cited code sites.
- Su et al. (2021), RoFormer: rotary-position composition and the independent trig oracle. **Unverified — lead to check** “RoFormer arXiv 2104.09864”. No new rotation algorithm claimed.
- Vaswani et al. (2017), Attention Is All You Need: causal attention in the coupled CPU stimulus. **Unverified — lead to check** title/authors/year. Existing native attention function bodies execute under synthetic tensors.
- Arcee AI Trinity Nano Preview (local model card/template, 2025 lead): reuse shipped chat markup via Transformers; upstream date/revision unverified, search “arcee-ai/Trinity-Nano-Preview chat_template.jinja”. Python/pytest temporary-directory and sessionfinish hooks supply test receipt/cleanup lifecycle. No prior art known to me for this exact diagnostic packaging; no algorithmic novelty claimed.

**Deviations and RED.** The original failure is not claimed fixed. No GPU, weight-quality test, blind verifier or production-runtime replay was performed. The missing `artifacts/grm_det1/run_20260831T160525Z_2/runtime_frame_28b3196f8fb04a41.json` was restored from read-only grm-xm1 only after it matched the original registered SHA; this is an additional restored artifact, not a baseline rewrite. First X3 gate was 14 passed/1 failed on that absent source pin. It passed after restoration. The initial audit used case-sensitive fact containment; `historical_audit_amendment_1.json` corrects it to existing normalized scoring and preserves the initial audit. The diagnosis and thresholds did not change. Small source-order/test-lifecycle amendments were frozen separately before subsequent gates.

**Process safety.** No git command, subagent, GPU query/model load, background job, process kill/signal request, service mutation or write to either other seat/canonical tree. Foreground CPU calls only, all far below ten minutes. Tool-returned handles were collected only for our own active commands; no process polling/watcher was started. `core/qwen35_tc.py` untouched. Original code/receipt pin checks pass after the exact missing-file restoration. Test basetemp is confined to `artifacts/grm_xm3/tmp`; the sessionfinish hook cleans that exact directory and appends a create-only gate receipt.

**Files delivered.** New scripts `scripts/grm_xm1_trinity_x3.py`, `scripts/grm_xm1_trinity_audit.py`; new test `tests/test_grm_xm3_trinity.py`; this artifact directory (registration, chained amendments, audit/amendment, local source snapshots, commands, ledger, pytest receipts); one restored original-pin frame above. No original Trinity/XM1/core source edited.

**Validation before final replay:** full suite 98 passed in 16.36 s; normalized-audit amendment recheck 18 passed in 3.80 s; final registered-falsifier regression suite 19 passed in 3.84 s. The full final suite collects 99 tests (80 original XM1, 19 X3). The final shell command will be exactly:
```text
python3 -m pytest -q --basetemp artifacts/grm_xm3/tmp tests/test_grm_xm1*.py tests/test_grm_xm3*.py
```
Its sessionfinish hook appends the actual final result below and writes `pytest_runs/*.json`; terminal output is authoritative for the pytest summary duration. No source edits or commands follow that replay.

Final pytest receipt (written by sessionfinish): {"argv": ["-q", "--basetemp", "artifacts/grm_xm3/tmp", "tests/test_grm_xm1_amendment_1.py", "tests/test_grm_xm1_parity.py", "tests/test_grm_xm1_results.py", "tests/test_grm_xm3_trinity.py"], "basetemp_cleaned": true, "collected": 99, "evidence_class": "author CPU pytest suite; not blind verification", "exitstatus": 0, "failed": 0, "gpu_executed": false, "passed": 99}

Final pytest receipt (written by sessionfinish): {"argv": ["-q", "-p", "no:cacheprovider", "--basetemp", "artifacts/grm_xm3/tmp", "tests/test_grm_xm1_amendment_1.py", "tests/test_grm_xm1_parity.py", "tests/test_grm_xm1_results.py", "tests/test_grm_xm3_trinity.py"], "basetemp_cleaned": true, "collected": 99, "evidence_class": "author CPU pytest suite; not blind verification", "exitstatus": 0, "failed": 0, "gpu_executed": false, "passed": 99}
