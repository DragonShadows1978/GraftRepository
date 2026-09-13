# GRM-XM1 X0 — cross-model graft-read parity harness (lead order, 2026-09-12)

Seat: Codex Astra (`gpt-6-astra`, reasoning high). YOUR WRITABLE TARGET is
the git worktree `/mnt/ForgeRealm/wt/grm-xm1` (branch `grm-xm1`, forked
from `lc1-wip`) — edits, builds and CPU runs there are AUTHORIZED. Read-only:
`/mnt/ForgeRealm/GraftRepository` (canonical), `/mnt/ForgeRealm/Project-Tensor`
(engine), `/mnt/Shared/LEAD_PLAN_XM1_GRAFT_PARITY_2026-09-12.md` (the plan —
read it first). No git (the lead commits). No subagents. Foreground only,
every Bash call < 10 min, no background waits, no monitor idling. NEVER kill
or signal a process you did not start. Your sandbox has NO GPU: build, gate
on CPU doubles, deliver a blocked report + exact lead commands; the lead runs
GPU under the lease (`/tmp/forge-gpu.lock`, 285 s worker / 590 s outer).
Tests that copy repositories use `--basetemp` under
`artifacts/grm_xm1/tmp` and clean up. Registrations pin REPO-RELATIVE paths
only. Prior Art Directive applies (code site + ledger + report).

## The question (David, verbatim)
"It SHOULD attend to it equally. Is the GPT-OSS different than any OTHER
model in the weight it gives grafts vs. K/V it computed itself from the
same tokens?"

## What exists
- RS4 (`scripts/grm_rs4_*.py`, `artifacts/grm_rs4/`, receipts committed):
  arms C0/C2/C3l/C5 on GPT-OSS-20B via the production ladder; the band
  decomposition (sink / mount / prior / fed / question / answer) in
  `grm_rs4_registration.py` and `core/grm_demand.py::_full_mass`; result:
  fed-text − mount-band mass = +0.035 on refusers, negative on controls
  (parity). Reproduce this row BIT-EQUAL before anything else.
- Adapters with graft injection: `core/gpt_oss20b_tc.py`,
  `core/qwen35_tc.py` (post-qk-norm pre-RoPE keys, `inject_kv`,
  `graft_seats`, `live_shift`), `core/minicpm3_tc.py` (MLA latent grafts,
  `Sg` seats), `core/gemma4_tc.py`, `core/qwen38_tc.py`; Trinity-Nano
  (NoPE) receipts under `artifacts/trinity_nope_graft/`; OLMoE via the
  MoE E-series scripts (`scripts/moe_e2_*` — confirm whether a graft
  path exists; if not, say so and drop it with a receipt).
- Models: HF hub caches for Qwen3.5-9B, MiniCPM3-4B, OLMoE-1B-7B,
  Trinity-Nano, gemma-4-12B-it (+qat), gpt-oss-20b; `/mnt/ForgeRealm/models/`.

## Mission
1. `scripts/grm_xm1_parity.py`: the RS4 C5-vs-C3l pair, parameterized by
   adapter — model loader, capture path (each adapter's own graft capture),
   seat-near-live injection (`GRM_SEAT_NEAR_LIVE` semantics per adapter),
   band accounting reused from RS4 verbatim, per-layer mass over fed-text
   rows vs mount-band rows, served answer both arms, value-span scoring.
   Probe panel = RS4's (sup_harbor_restatement, fresh-fact controls,
   multi-hop) with per-model tokenization of the same texts. One leased
   cell per (model, arm, probe); resumable; create-only receipts;
   device-memory receipt per cell; NON_FIT-and-continue for models that
   do not fit with a graft on 12 GB (say which in the dry-run).
2. Parity barrier: the GPT-OSS cells must reproduce RS4's recorded C3l and
   C5 masses at float equality (or state the exact reason they cannot:
   RS4 ran on core sha X; this tree has FIX-9/F1/F2/F5 — pin all round-2
   flags OFF and `GRM_LEGACY_DEFAULTS`-equivalent env; if masses still
   differ, STOP with the diff).
3. Per-adapter CPU double that traverses the same code path (loader
   stubbed only) — gate: every (model, arm) cell runs on the double and
   writes a receipt; the band decomposition sums to S on every row.
4. Registration (immutable, amendments separate): models in lead order
   (GPT-OSS → Qwen3.5-9B → MiniCPM3 → Trinity-Nano → OLMoE → Gemma-4),
   cells, per-cell estimate, ≤ 0.5 GPU-h per model, the lead's prediction
   and falsifier VERBATIM from the plan, repo-relative pins.
   `artifacts/grm_xm1/lead_commands.txt` (every command runs with
   `--dry-run` appended; add that gate). Blocked report.
5. `python3 -m pytest -q --basetemp … tests/test_grm_xm1*.py
   tests/test_grm_rs4*.py` verbatim last.

## Done (verbatim)
1. Files:lines; per-adapter capture/seat mechanism table (what each
   adapter's graft IS: pre-RoPE keys / MLA latent / NoPE; where the seat
   lands).
2. Parity-barrier status on the CPU double + what the GPU must show.
3. Registration path + sha; cells + estimates; NON_FIT candidates.
4. Prior art; deviations; RED; process safety (nothing killed, no GPU);
   model id + effort. pytest LAST.
