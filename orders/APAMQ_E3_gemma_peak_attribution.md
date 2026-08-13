# APAMQ-E3 — Gemma-4 Local Peak Attribution + Perfect-APA Upper Bound

YOUR WRITABLE TARGET is /mnt/ForgeRealm/GraftRepository — specifically
`scripts/`, `artifacts/apamq_e3/` (create it), and the NEW report file
`docs/APAMQ_E3_REPORT.md`. Edits there, runs of your own scripts, and
GPU model runs are AUTHORIZED. Run first, report after; do not ask
permission for anything non-destructive. A registered order IS the
permission.

HARD READ-ONLY: `core/` (instrument via wrappers/monkeypatching from
your scripts, or if a tiny in-place probe is unavoidable, REVERT it
before finishing and say so), all existing `docs/`, `orders/`,
/mnt/ForgeRealm/models/ (weights), and ALL of
/mnt/ForgeRealm/Project-Tensor including the built tensor_cuda .so
(shared runtime — never rebuild it). No git — the lead commits. No
subagents. RED honesty; a wall or a failed cell is a result. No
monitor-idling.

GPU: serialize GPU work under `flock -w 7200 /tmp/forge-gpu.lock`.
Another seat runs synthetic kernel sweeps on the same card; hold the
lock around runs, release between. Operator has absolute right of way.
Timebox: total order leash is 6h. Budget the matrix so 4K–16K completes
before any 32K attempt (A0 receipt: APA 32K prefill alone took 1619s —
at most ONE 32K cell per mode, only with ≥90 min left on the clock).

## Context (pre-nailed premise — do not re-litigate)

Plan: /mnt/ForgeRealm/Project-Tensor/docs/APA_MQA_ROOTCAUSE_PLAN.md —
read it first. Read also: `docs/GEMMA4_APA_AUDIT_A1.md` (the open
~110MB item), `docs/GEMMA4_APA_EXTENSION_PLAN.md`,
`docs/GEMMA4_PORT_LEDGER.md` (the 16 registered traps — respect them).

Question: on Gemma-4 12B local (`core/gemma4_tc.py`, exact QAT q4_0
import), WHERE does the peak VRAM actually live, standard vs APA, and
how much could a PERFECT APA (zero ring overhead, fused path wherever
legal) ever reclaim? Prior A0 measurement (2026-07-04, this card):
APA peak +0.4–3.3% vs standard, no OOM either mode through 32K.

Known methodology laws (violating these voided results before):
- ONE MODE PER PROCESS — an OOM episode fragmentation-pins ~1GB; never
  compare modes measured in the same process.
- TRUE peaks: 1s nvml poller in a side thread AND engine pool
  high-water; report both.
- The chunked-scoring scar: ring-storage quantization silently no-ops
  under chunked scoring — ring measurements must drive L=1 decode and
  ASSERT the stored dtype in-process.

Evidence class: INSTRUMENTED PORT MEASUREMENT — memory and speed
attribution only. NO quality/perplexity claims. ms/tok recorded as
secondary data only.

## Tasks

**T-audit (first, read-only, no GPU needed):** trace which attention
path the port ACTUALLY takes today for global layers, prefill and
decode, standard vs APA mode. TC_APA_MAXD in the engine is now 512
(kernels.cu:1053), so the June-era "fused kernel forbidden at D=512"
premise may be stale. Report: the exact conditions (file:line in
gemma4_tc.py — apa_min_context, the fused-vs-blend dispatch lever,
decode L==1 branch) and which branch fires at S ∈ {4K, 8K, 16K, 32K}
for prefill chunks and decode steps in each mode. If the fused path is
LEGAL but unwired on decode, say so explicitly — that is hypothesis
H-D and it matters.

**T-attrib:** instrumented runs, S ∈ {4096, 8192, 16384} both modes
(+ optional single 32K per mode per the timebox). For each: prefill to
S, then 64 decode steps. One mode per process; synthetic/repeated
prompt tokens are fine (memory does not care about token identity).
Record per run: phase peaks (prefill vs decode, pool + nvml), and
PER-LAYER-CLASS attribution — wrap the sliding-layer and global-layer
attention calls separately and accumulate per-class pool high-water
deltas, so the report can say "at S=16K prefill, sliding attention
transients X MiB, global attention transients Y MiB, weights Z,
mask caches W, other U."

**T-components (APA mode):** measured sizes vs S for: kqb ring
resident bytes (assert dtype), quantize-step transients (cold-start
chunked and incremental), refine/tail rescore transients, and any APA
state beyond standard mode's.

**T-110MB:** at S=4096, chase AUDIT_A1's ~110MB unexplained allocator
high-water with a live pool trace (allocation-site logging if the pool
exposes it; otherwise bisect by phase). If you cannot attribute it,
report what you excluded — a narrowed suspect list is a valid result.

**T-bound (arithmetic, no GPU):** from the measured numbers, compute
the inputs for the perfect-APA reclaimable bound at S=16K (and 32K if
measured): eliminable global-layer score/softmax transients under the
fused path, minus irreducible APA-specific state. Show the arithmetic
line by line. Do NOT declare a verdict against the plan's T2
threshold — numbers only; adjudication is the lead's.

## Deliverables

`scripts/apamq_e3_attrib.py` (+ helpers), `artifacts/apamq_e3/*.json`
(raw per-run records), `docs/APAMQ_E3_REPORT.md` — sections: Path
audit / Attribution tables / Component sizes / 110MB trace / Bound
arithmetic / Anomalies. Factual, receipts inline, no verdicts.

## Done

Final message MUST contain verbatim: exact commands run, the
attribution table for S=16K both modes pasted inline, the kqb ring
and quantize-transient numbers, the T-audit branch findings with
file:line, the bound arithmetic, and any cell skipped/failed with
why. If core/ was touched for probes, confirm the revert (diff
clean). Honest partial > polished incomplete.
