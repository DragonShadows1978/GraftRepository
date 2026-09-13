# GRM-XM1 X0 — BLOCKED / RED, 2026-09-12

**The GPU barrier is BLOCKED. X0's strict whole-production-path CPU gate is RED; not claimed fixed.** There is no new GPU parity result and David's same-token cross-model question remains unresolved. The Python harness, registration, 60 CPU dispatch receipts and dry-runnable lead commands are on disk. Evidence classes below distinguish recorded-data checks, CPU source-function doubles, and unrun GPU work.

## Files and measured gates

- `scripts/grm_xm1_parity.py:42`: immutable source validation; `:120` GPU loaders; `:154` capture; `:177` constant-delta seating; `:213` observer; `:321` native cells; `:373` exact comparator; `:382` unchanged RS4 replay; `:420` barrier; `:446` create-only cell receipts; `:504` CLI/lease/budget.
- `scripts/grm_xm1_cpu.py:117`: compile unmodified, pinned native attention-function ASTs against CPU tensor/projection doubles, avoiding CUDA initialization; `:139` adapter-specific numerical seams; `:190` loader. Native function bodies execute, but constructors, weights, surrounding model and full GPT repository replay are not exercised.
- `scripts/grm_xm1_registration.py:48`: registration builder. `scripts/grm_xm1_results.py:18`: joins, completeness and registered falsifier, with CPU evidence excluded from model claims.
- `tests/test_grm_xm1_parity.py:24`: exact historical reassembly; `:49` all-cell receipts, resume and partition law; `:73` payload/seat invariants; `:97` last-bit drift rejection; `:108` CPU cannot unlock GPU; `:144` all lead commands dry-run.
- Unit/suite evidence: [pytest_pre_final.log](pytest_pre_final.log), **131 passed in 12.63 s**. Includes both XM1 test files and both existing RS4 test files. The identical requested test selection is rerun as the final command after this report; its terminal output is the final-run receipt.
- Recorded-data evidence: [cpu_reference_barrier.json](cpu_reference_barrier.json), **10/10 historical C3l/C5 registered rows reassembled at exact float equality**, before the CPU matrix. This reads raw recorded per-layer masses and uses RS4's own assembler; it is not model re-execution.
- CPU execution evidence: [cpu_gate.json](cpu_gate.json), **60 dispatch receipts: 40 PASS, 20 NO_GRAFT_PATH**. All observed physical band widths sum to the actual `S`; imported RS4 probability partitions close at RS4's existing tolerance. Per-position/per-layer rows, source/question/generated token IDs, served answers and value-span scores are in `cells/cpu/`.
- Author mutation evidence: [mutation_results.json](mutation_results.json), **4/5 non-error runtime mutants killed (0.80)**. `capture_pin_ignored` SURVIVED: the tests do not prove that pinning the capture shift changes the intended context. This is a named coverage failure queued for the lead; no post-result tolerance or code change. [mutation_registration.json](mutation_registration.json) predates the run. No blind red-team claim.

## Adapter capture and seat mechanisms

| Model, lead order | What the graft is / source | Capture and seat in this harness | Status |
|---|---|---|---|
| GPT-OSS-20B | Projected pre-RoPE K plus V, with learned sinks and full/sliding GQA; `core/gpt_oss20b_tc.py:672` | GPU calls RS4 `run_arm`, production deposit(pin=`live`), `_rs3_seat_plan` and `_rs3_rotate_injection` (`core/graft_arena.py:561`, `:644`, `:749`). Physical mount remains after sink; its final positional seat is `live_shift-1`. CPU native source-function double includes learned sink and sliding RS4 observer methods. | Native CPU 10/10; production GPU reference unrun; production-replay CPU gate RED. |
| Qwen3.5-9B | Post-qk-norm pre-RoPE K and V on eight attention layers; DeltaNet state is separate. `core/qwen35_tc.py:321` | `_capture` at `:338`; `inject_kv`, `graft_seats`, `live_shift` at `:346`. Only rotary key dimensions move by the constant delta; final mounted position is `live_shift-1`. | Native source-function CPU 10/10. DeltaNet recurrent state is not transferred. |
| MiniCPM3-4B | Post-norm MLA latent `c_n` and shared pre-RoPE `k_pe`; `core/minicpm3_tc.py:154`, capture `:182` | Capture own latent pair; move only `k_pe`; inject `(c, kpe)` with `Sg=c.shape[1]`. `c` stays unchanged. Expanded standard attention (absorbed decode OFF). Final seat `live_shift-1`. | Native source-function CPU 10/10. GPU unrun. |
| Trinity-Nano | Post-qk-norm K/V; full layers NoPE, local layers RoPE. Existing T1 driver `scripts/trinity_nope_graft_width_sweep.py:266` over read-only Project-Tensor model. | Reuse existing `_arena_attention_call` and `TrinityArenaModelAdapter`. Full-layer keys remain unrotated; local keys receive constant-delta relocation. Both are physical prefix rows; local final position `live_shift-1`; NoPE has no rotary position. | Native source-function CPU 10/10. GPU unrun. |
| OLMoE-1B-7B | `scripts/olmoe_e2_experiment.py` captures MoE router inputs; no attention-graft interface found in the named E-series testbed. `scripts/moe_e2_*` does not exist here. | No fabricated capture or seat. | Dropped with 10 NO_GRAFT_PATH receipts. |
| Gemma-4-12B | `core/gemma4_tc.py:506` has normalized live K/V, ring caches and a KV storage hook, but no `_capture`, `inject_kv`, `graft_seats`, or `live_shift` reader in this tree. | No fabricated capture or seat; requires a separate production adapter order. | Additional blocked prerequisite: 10 NO_GRAFT_PATH receipts. |

Source/config inspection and exact line inventories: [adapter_inventory.json](adapter_inventory.json). A model config being present is not a weight-completeness or fit gate.

## Registration and lead execution

Immutable registration: `artifacts/grm_xm1/registration.json`.

SHA256: **758fd3ac1231ed44937714a86009b557f554f026e9d85773cb730074e04439e8**.

The prediction and falsifier are copied verbatim from the immutable Shared lead plan. Source pins use repo-relative paths only. Separate immutable files: `implementation_pins.json`, `results_pins.json`, `mutation_registration.json`. The Shared plan was not edited.

Five probes: `sup_harbor_restatement`, `sup_praxis_fresh`, `sup_solace_fresh`, `sup_reserve_tundra_ledger`, `sup_reserve_meridian_docket`: three registered refusers and two controls from RS4. Only the `registered` variant is included; RS4's secondary `solace_fact` variant is excluded.

| Model | Cells | Estimate per cell | Planned GPU time |
|---|---:|---:|---:|
| GPT-OSS | 10 | 150 s | 1,500 s / 0.417 h |
| Qwen3.5 | 10 | 120 s | 1,200 s / 0.333 h |
| MiniCPM3 | 10 | 120 s | 1,200 s / 0.333 h |
| Trinity | 10 | 150 s | 1,500 s / 0.417 h |
| OLMoE / Gemma | 10 each | 0 s (unavailable path) | No measured cell scheduled |

These are registered planning estimates, not timing measurements. The runner reserves 285 s per GPU attempt against a 1,800 s/model cap, charges a lost attempt its full reservation, and refuses a new reservation that would overrun. Full-session RS4 replay can exceed a cell estimate; do not relax the registered rail if that happens. Each GPU process uses the existing fail-fast `/tmp/forge-gpu.lock` lease (wait=0), 285 s worker cap, and a 590 s outer timeout. No monitor or background queue.

[lead_commands.txt](lead_commands.txt) has **60 exact foreground commands** in lead order. Run from the worktree root. Every executable line was run with `--dry-run` appended during the test gate and exited zero without loading weights or taking a lease. STOP on any nonzero exit. The first command is:

```sh
timeout 590s python3 scripts/grm_xm1_parity.py --model gpt-oss --arm C3l --probe sup_harbor_restatement
```

The GPU must produce all ten GPT receipts with every recorded mass at float equality and served text/correctness/answer count equal. `reference_diff` persists differences; the barrier prevents every other model from counting. CPU receipts cannot unlock it. Resume verifies registration/implementation bindings and never overwrites a receipt. An actual recognized GPU OOM is receipted as NON_FIT and returns zero so the lead can continue; failed non-fit attempts still consume budget. Fit candidates flagged in every dry-run: Trinity (INT4 weights but float32 compute and transient MoE/attention memory); Gemma, if a graft path later lands (12B plus mixed-width cache). No model is declared NON_FIT from an estimate alone.

Each GPU cell records before/after device-wide total/used/free memory via nvidia-smi. This is a snapshot, **not peak or per-process allocator memory**. CPU receipts explicitly carry null GPU memory measurements.

## Parity and protocol RED items

1. **Full RS4 production replay on a loader-only CPU double is not delivered.** The CPU source-function tests exercise native capture/cache/attention functions plus the shared native harness, while the real GPT reference branch delegates to RS4's production repository runner. That outer-path difference is explicitly RED. The GPU path is unverified, not claimed fixed.
2. **RS4's arm texts are not identical.** For example its standard Solace C3l arm recaptures a Tundra split child, while C5 feeds Solace and Tundra turns. The frozen registration carries both text sets. Historical parity does not establish the stronger same-token causal claim in David's question. A separately registered matched-text panel is needed before that claim.
3. Non-GPT direct native cells use lossless host payload, no physical system sink, native chat templates, and a band derived from the larger arm token count plus 96. GPT's reference uses the original production sink, width 96, 8-bit stored grafts, route/prelude, and sink attention. This is an adapter instrument, not a certified cross-model production-ladder comparison. No source-token/geometry equivalence is claimed across model tokenizers.
4. Qwen grafts omit recurrent DeltaNet state. Trinity remains a driver patch; Gemma has no graft path. Sliding-cache eviction beyond the observed row geometry raises an error instead of silently rebasing bands. No large-context/ring-wrap gate ran.
5. [rs4_source_audit.json](rs4_source_audit.json) shows GPT adapter and frame file hashes equal to RS4, but `graft_arena.py` and `grm_demand.py` differ. RS4 arena SHA256 was `24f227be9e5f7584d86dbb3580e735c406796a5c49497b07d89c001adace9ffc`; current is `3c97a3ea3afc7c2f6f8dce75bcc43e24e8e03ce535f14eec669fa6277ad02a62`. No core commit SHA is recorded in the selected receipts; no git was used to invent one. Actual round-2 env readers are pinned OFF, old LSR remains ON, demand NGH OFF and admission `all_tokens_bind`. There is no assumed magic `GRM_LEGACY_DEFAULTS` reader. Unflagged FIX-9 changes may still prevent equality; GPU must STOP with the exact diff if so.
6. The ignored-capture-pin mutant survived. Refusal detection is a registered lexical diagnostic; the raw served strings remain authoritative for the lead's refusal falsifier. Source/config availability is not proof that complete model weights or external runtime dependencies load.

## Prior art

- **GRM contributors, 2026:** RS3 capture pin and constant-delta seating, RS4 band decomposition/observer/value-span comparator, C7/LT1 CPU seams and immutable receipts. Taken unchanged where possible; new work is adapter dispatch, source-function doubles and receipt/barrier joins. No new attention algorithm.
- **Su et al., 2021, RoFormer:** constant-delta rotary composition; **DeepSeek-AI, 2024, DeepSeek-V2:** MLA latent versus expanded attention representation. **Unverified — lead to check:** `RoFormer arXiv 2104.09864`, `DeepSeek-V2 arXiv 2405.04434`.
- **DeMillo, Lipton and Sayward, 1978:** mutation-testing method. **Unverified — lead to check:** `Hints on Test Data Selection Help for the Practicing Programmer`.
- Python AST/compile and NumPy are implementation tools. No prior art known to me for this exact cross-adapter CPU loader composition. Literature verification is left to the lead's proxy as ordered.

## Process safety and final command

Seat: **gpt-6-astra, reasoning high**, as dispatched. No git, subagents, GPU execution, foreign-process signals, kills, service changes or background jobs. Only authorized worktree artifacts/scripts/tests were written; canonical repo and Project-Tensor source remained read-only. No repository was copied by these tests. Mutation temporary directories were under `artifacts/grm_xm1/tmp` and cleaned by context managers.

Final command after all report/ledger writes:

```sh
python3 -m pytest -q --basetemp artifacts/grm_xm1/tmp/pytest tests/test_grm_xm1*.py tests/test_grm_rs4*.py
```
