# GRM-C4 — CPU handoff ready; GPU and full-factorial verdict RED

**59 CPU tests passed. No GPU run occurred. The chunking hypothesis remains
INCONCLUSIVE.** Two new cells are ready for lead execution. The requested full
fixed-geometry cross is not complete: WC1's width64 diagonal has different
numeric geometry, and replacing it exceeds the registered planning estimate.
Not claimed fixed: t33, Juniper, GPU geometry parity, or the causal explanation.

## Deliverables and pins

- Knob: `GRM_C4_DEPOSIT_CHUNK_TOKENS`, OFF when absent/0/off/empty.
  `scripts/grm_c4_adapter.py:15` defines it; lines 30–63 scope it to the existing
  `GraftRepository._guard_deposit_width` at `core/graft_repository.py:1210`.
  Importing the adapter or setting the env alone does not modify production.
  The campaign opts in with 64/96. No core/three-pass/kernel/battery/registry/
  config files were edited. This harness-scoped implementation follows the
  additive-only boundary; it is not an installed production API knob.
- Explicit fit-time budgets are preserved, including when 96-token deposit
  children exceed width64. The later repair remains an observed mechanism.
- Geometry: construct with a 96-token RoPE band, then set admission capacity
  before repository native configuration; capture/live begins at `n_sink+96`,
  near-live plan head ends at `n_sink+95`. Actual numeric observations accompany
  receipts. `n_sink` is tokenizer-derived; live_turns=2, max_live=4096, EB1
  ephemeral frame. GPU equivalence of this intervention is **not validated**.
- `registration.json` was written before gates, SHA-256:
  `7fb561c70dd8c04abc18ef99a7a2a5ec4d37cbc2a22b1de1f89ab2119bd285e8`.
  A1/A2 are separate SHA-bound implementation amendments; the base is unchanged.
  Latest amendment SHA-256: `3e4fb3416bda6b29350139bc76efcc4738cd83a20485ecae2484b7b0201c881e`.
- Machine report: `blocked_report.json`; CPU output: `cpu_gate_final.txt`;
  every cell/segment and estimate: `dry_run_final.json`; exact dependency-ordered
  commands: `lead_commands.txt`; history: `IMPLEMENTATION_LEDGER.md`.

## Cells and evidence

| Chunk | Width | Evidence/status | Sup | Census | Full long-history | New GPU estimate |
|---|---|---|---|---|---|---|
| 64 | 64 | WC1 historical only; geometry mismatch | 8/9 | 10/10 | 14/14 | Replacement 1600s, full design NON_FIT |
| 96 | 96 | Cite WC1; original fingerprints match | 9/9 | 9/10 | 13/14 | 0s |
| 64 | 96 | NEW, GPU pending | — | — | — | 1600s |
| 96 | 64 | NEW, GPU pending | — | — | — | 1600s |

Historical score source: `/mnt/ForgeRealm/GraftRepository/artifacts/grm_wc1_opus/grm_wc1_results.json`,
SHA-256 `087338e42a7d8be5340876b58f172023583732d9b6356836365b82486ff8d09d`.
All 13 WC1 registration fingerprints matched after resolving missing local
artifact/native paths to the main repository. No historical diagonal was rerun.
The inherited WC1 width64 pin was `n_sink+64`, so its scores cannot be assigned
to the requested fixed-geometry cell. Historical token residency is **missing**;
the original graft-count column is not reused as token seats.

New receipts retain original route information, capture and seat observations,
parent/child split events by deposit versus explicit fit repair, and end-of-
segment split census. Actual mounted seats sum each current mount's `ntok` and
check `cur_mount_n`, with sink/live/physical cache tokens separately recorded.
The score command aggregates per-attempt and per-step seats by battery and
uses all 14 EB1 rows, not only its four distant spots. Observations are taken
at attempt/step returns; they are not a continuous allocator peak measurement.

Registered Scout prediction: c64_w96 recovers t33, reaches 14/14, and keeps
Juniper. Required battery scores are 9/9, 10/10, 14/14. Reject the chunking claim
if improvement needs narrower seating. Missing/non-comparable cells remain
INCONCLUSIVE; the scorer cannot certify the full factorial from these two new
cells plus mismatched historical geometry. Both hypotheses retain equal standing.

## CPU gate and limits

`python -m pytest -q tests/test_grm_c4_campaign.py tests/test_grm_lsr_p2c_split_descent.py`
returned **59 passed, 2 warnings in 4.65s**. Warnings are upstream SWIG type
`__module__` deprecations; the full transcript preserves them.

Unit evidence covers 64/96 splitting independent of width, full parent text,
explicit-fit budget preservation, numeric RS3 capture/seat geometry in a CPU
stub, token sums versus graft counts, frozen 9/10/14 membership, SHA drift,
missing results, imported loader path binding, timeout RED/no retry, abandoned
worker blocking, and 13-segment saved-state copy behavior.

OFF parity: all **37 nodes** of the existing WC1 width96 census manifest feed
the real guard with stubbed payload/persistence. OFF retains original callable
identity and produces byte-identical serialized stub results/node state versus
the original callable; the source manifest bytes remain unchanged. This is a
CPU guard test, not a restored-model K/V or on-disk flush parity result.
Author-run gates are baseline evidence. Blind verification remains lead-owned.

## GPU budget and resume handoff

Each new cell has four supersession fixtures (60s estimated each), four census
shards (80s each), and thirteen 8-turn EB1 segments (80s each), 1600s total.
Two cells estimate **3200s / 0.889 GPU-h** under the mission's explicit 1 GPU-h
cap. These are unvalidated planning assumptions, not measured timing claims.
The full numeric-geometry design estimates **4800s / 1.333 GPU-h: NON_FIT**.

The existing EB1 `run_shard` is imported; only its in-memory stop map becomes
8,16,…,104. Its DET1 boundary flushes repository, scorecard, and restart state;
the next lease copies and resumes that state. Original script, values and all
14 probes remain intact. Each prior receipt binds all saved session files,
which are rehashed before resume. Census uses its unchanged four shard stops.

Lead commands interleave corresponding units across the two cells, with one
foreground lease per unit. Worker cap 285s, outer timeout 590s, cooldown 30s,
lock `/tmp/forge-gpu.lock`, lock wait zero. A full 285s is reserved against
remaining campaign budget before each unit. Timeouts stop the campaign;
immutable start markers prevent reuse/deletion of partial session data. An
unreceipted marker blocks subsequent cells because GPU usage is unknown.
Never retry into a longer lease or clear a marker/lock. Execute commands in
separate bounded foreground invocations, not the whole handoff as one tool call.

## Deviations and RED

The additive scope puts the knob in a new adapter rather than modifying core.
Numeric geometry is pinned separately, revealing that historical width64 cannot
satisfy the fixed-geometry requirement. No missing control is substituted, no
extra diagonal GPU run is scheduled, and no budget increase is assumed. A1
corrects the imported supersession loader path; A2 closes abandoned-worker
accounting. Neither changes registered decision bars or fixtures.

No GPU/model serving, git commands, subagents, background jobs/waits, service
changes, lock access, or process signals occurred. CPU fake-lease tests never
open the GPU lock. GPU work is blocked here by the sandbox's lack of GPU;
the incomplete fixed-geometry factorial is a separate RED limitation.

## Prior art

Verified local prior art: **LSR-P2C, RS3, WC1, EB1, DET1 (house systems, 2026)**.
Borrowed: existing section/sentence splitter and retained-parent lineage,
near-live/capture formulas, harness import/rebinding, original scorers,
resumable sessions, leased workers and claim/receipt bookkeeping. C4 adds
independent experimental controls and token-seat observations. No new chunking,
routing, or persistence algorithm is claimed. The same annotations are at code
sites and in the ledger. External factorial lead: **Fisher, The Design of
Experiments (1935), unverified — lead to check**; search `Fisher factorial
experiments 1935`. No external literature verification was performed here.

Author model: **GPT-6 (Codex; no more specific deployment id exposed)**;
reasoning effort: **high, per order**. Reader: `openai/gpt-oss-20b`, revision
`6cee5e81ee83917806bbde320786a8fb61efebee`, inherited from the frozen frame;
no model-weight payload hash or reader-model execution is claimed in this sandbox.
