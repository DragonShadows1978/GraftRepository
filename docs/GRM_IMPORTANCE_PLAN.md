# GRM Importance Weighting — Silent Second Pass (PLAN, immutable)

Working name: GRM-IMPORTANCE (placeholder; naming is David's).
Opened: 2026-07-16. Lead: Fable (planner). Implementation: Sonnet seats.
Substrate: MiniCPM3 MLA arena (most-gated stack). GQA port = successor, not in scope.

## Thesis (David, 2026-07-16)

After the model replies, a **silent second pass** runs in the idle window
(the deferred librarian's `idle()` slot) and manages memory before the
user's next input. The management primitive is a per-node **importance
weight**. Open question — which signal actually measures importance:

- **S1 RETRO** — pass-1 attention telemetry: per-mount attention mass
  logged during reply decode (read-only tap at the absorbed-decode
  softmax, `core/minicpm3_tc.py` MLAAttentionTC). Retrospective usage.
- **S2 PROSPECT** — self-report salience: primed-prefix scoring pass in
  `idle()` (digest-machinery lessons apply: acknowledgment trap, content
  QC). The only prospective channel ("this will matter later").
- **S3 CAUSAL** — counterfactual unmount: teacher-forced replay of the
  reply with one mount absent; importance = measured logit dependence.
  Offline calibrator / ground truth. Never a production path.

David's directive: build ALL arms, gate them against each other.
S3 is the arbiter; S1/S2 are the candidates.

## Registered metric definitions

- S1 mass(node) = sum over reply decode steps of softmax probability
  mass on the node's arena seats, **mean over heads and ALL layers**
  (per-layer numbers are diagnostics only — no post-hoc layer picking).
  Reported as share of total non-live mass.
- S2 salience(node) ∈ {0..3} from a fixed primed-prefix rubric prompt,
  frozen in the harness before G1 runs; one retry ladder max, mirroring
  librarian QC.
- S3 dependence(node) = mean |Δlogit| over reply tokens, teacher-forced,
  mounted-set-minus-node vs full mounted set. Also report KL.
- Node manifest gains an `importance` dict; each arm writes ONLY its own
  key: `s1_mass`, `s2_salience`, `s3_dep` (test-only). Consumer reads are
  Phase 2.

## Gates (structure registered now; numeric thresholds registered after
## G0 floors, before each governed gate — SCRIBE pattern)

- **G0a telemetry parity** — telemetry-on vs telemetry-off decode,
  teacher-forced: max |Δlogit| = 0 exactly (the hook is a pure read;
  bit-identical is the demand, not the bf16 floor).
- **G0b counterfactual dynamic range** — scripted turn with one
  load-bearing mount + one decoy mount: unmount(load-bearing) ≫
  unmount(decoy), decoy within the plain cache-vs-prefill bf16 floor
  (0.75–0.9 band per prior receipts; re-measure, don't assume).
  Establishes S3 noise floor + dynamic range. G1/G2 thresholds get
  registered from these floors.
- **G1 signal agreement (retrospective)** — labeled conversations
  (fixture, see WO-4), k mounts of graded relevance per probe: Spearman
  rank corr + top-1/top-3 agreement of S1 and S2 vs S3.
- **G2 prospective discriminator** — standing-preference facts deposited,
  ZERO uses for N≥10 turns, then probed. Pre-registered expectation:
  S1 scores them at floor (**designed failure** — retrospective signal
  cannot see prospective value); S2 must rank them above filler.
  If S2 also fails G2, the prospective channel is unsupported and the
  silent pass degrades to telemetry bookkeeping — that is a result.
- **G3 consumer closed-loop (Phase 2, dispatched only if G1 or G2
  green)** — importance-driven fold ordering + paging priority vs
  current threshold/LRU baseline under compression pressure; E4-style
  42-turn recall + paging-20/20 harnesses reused. Win = recall ≥
  baseline at equal-or-better residency; fidelity gate behavior intact.

## Registered non-goals (successors, not scope)

- Routing-score integration — attractor risk, three measured instances
  (style attractors, live-window echo, cache-latent pollution). The
  weight stays OUT of the router in this program.
- Supersession-under-competition (registered E2E seam) — consumes the
  weight later; separate program.
- GQA/GPT-OSS telemetry port; production fused-path hook cost.

## Work orders (parallel, strict file ownership, no subagent spawning)

- **WO-1 (S1)** owns `core/minicpm3_tc.py`, `core/graft_arena.py`:
  telemetry tap + per-mount mass accounting + G0a harness
  `tests/test_grm_importance_telemetry.py`.
- **WO-2 (S3)** owns `tests/test_grm_importance_counterfactual.py`:
  harness-only, reuses teacher-forced A/B machinery; G0b.
- **WO-3 (S2)** owns `core/graft_repository.py` (idle-slot scoring pass)
  + `tests/test_grm_importance_salience.py`: frozen rubric prompt,
  primed prefix, writes `s2_salience`.
- **WO-4 (fixture)** owns `tests/fixtures/importance_convos/`: labeled
  conversation set — probed-later facts, standing preferences (G2
  class), filler chit-chat, superseded facts; JSON format with
  ground-truth relevance grades per (probe, node).

GPU discipline: agents deliver code + CPU-verifiable tests; ALL
GPU gate runs are executed serially by the lead (12GB card, no
co-resident model loads).

## Evidence classes

Every claim in the ledger names one: unit test / teacher-forced logit
A/B / recall gate / latency receipt. No claiming consumer value (G3)
from signal agreement (G1) — separate gates.
