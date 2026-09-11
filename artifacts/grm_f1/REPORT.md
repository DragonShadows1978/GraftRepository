# GRM-F1 — source retention after folds: REPORT

Seat: Opus 5 (opus-max), reasoning effort MAX. CPU only.
Branch `grm-f1`. Order `orders/GRM_F1_FOLD_RETAIN_SOURCES.md`.
Ledger (the receipts): `docs/GRM_F1_FOLD_RETAIN_LEDGER.md`.

## The finding, in one paragraph

LT1.1 r2 answered every one of its 57 correct rows off a fold digest or an
era index — never off the turn that carried the fact. That is not a ranking
accident: FIX-5 consolidation RETIRES a fold's sources the moment the digest
clears `MIN_FOLD_KEEP`, and `_route_cand_base` excludes retired nodes, so
after a fold the digest's prose is the only routable record. `GRM_FOLD_
RETAIN_SOURCES` makes the digest ADDITIVE instead of substitutive. Measured
through the real LT1.1 worker on 4 cells: routable nodes 12 → 24, probes
mounting a source node 4/7 → 6/7, and 2/7 probes mounted a RETAINED source
that under OFF did not exist to be reached — with residency still bounded
(max 94 seats against width 96) and answer quality unchanged on the CPU
double's regex reader (5/7 both arms). Whether the newly reachable sources
change ANSWERS is a GPU question, and that is exactly what r3 registers.

## Scale of what r2 was throwing away

From r2 arm A's end-of-campaign manifest: 199 nodes, 36 active, 163
inactive — of which **148 were retired by a fold alone** (124 turns + 24
digests) and only 15 by a correction. Under the flag the routable candidate
base is predicted to grow 36 → 184 (5.1×) at the same 199 total nodes. The
15 correction-retired nodes stay retired in both arms, which is the point:
retention is a compression policy, never a correctness one.

## Why this cannot quietly break supersession

Three independent guards, each measured rather than argued:

1. **A correction reaches MORE nodes under the flag, never fewer.**
   `correct_memory` only considers nodes whose `metadata.active` is True.
   OFF, the stale source was already fold-retired and invisible to the
   correction (it supersedes the digest alone). ON, the source is active, so
   the correction retires it too. Measured: `supersedes [digest]` →
   `supersedes [source, digest]`.
2. **GRM-A1 is invariant.** `_alias_consolidate` pins `retain=False`. A1
   always supersedes its alias edge — a bare active edge that wins admission
   and answers nothing IS the RD2 defect A1 exists to fix — and its own
   explicit retire/un-retire decisions run after the fold returns either
   way.
3. **The width guard, `MIN_FOLD_KEEP`, the fidelity abort, the era rules,
   route ranking and the admission ladder are untouched.** A fold that fails
   coverage still ABORTS and still leaves its sources alone, in both arms.

## Honest RED / things that did not go well

* **The registered C2 132-plan replay gate cannot run on this machine.** Its
  source campaign `/mnt/ForgeRealm/wt/grm-c2/...` was pruned with the grm-c2
  fork, so it collects 0 rows and asserts. Verified PRE-EXISTING (fails
  identically on baseline core). This is the round-1 absolute-pin lesson,
  already a permanent receipt. I built a substitute: a baseline-vs-F1
  fingerprint over the full node table, routing base, correction lineage and
  fold history — IDENTICAL in every section
  (`artifacts/grm_f1/off_identity/`).
* **I broke the OFF arm once, and the fingerprint caught it.** Completing
  the digest's metadata unconditionally in `_fold_once` changed which nodes
  an OFF-arm correction superseded (`[0,10]` → `[0,8,10]`). Fixed, and
  pinned by two regression tests. This is the defect that would have
  silently corrupted the control arm, and it was caught only because the
  proof compared against baseline core rather than against my expectations.
* **The flag would have been stripped from the r3 campaign.**
  `environment()` deletes every ambient `GRM_*`; without the re-apply in
  `arm_environment`, r3 would have measured OFF while reporting ON.
* **Exact-correct did not move on the CPU double** (5/7 both arms). Stated
  plainly: this run is evidence about routing topology and receipt plumbing,
  not model quality.
* **Budget conflict, unresolved by design.** The order caps r3 at ≤1.3
  GPU-h; r2's reused cell schedule reserves 2.06 GPU-h of LEASE (estimate
  1.25 GPU-h, under the cap). Setting the budget below the reservation sum
  would reproduce the exact failure LT1.1 amendment 2 was written to correct.
  Registered as two separate numbers with a **lead decision requested**.

## Known gap left open, owned by F2

A fold digest that PARAPHRASES a fact escapes `correct_memory`'s text match,
so the stale copy inside it is never superseded. Measured in BOTH arms: OFF
retires nothing at all; ON retires the retained source but the digest still
escapes. F1 neither causes nor fixes it. Registered as `known_gaps` F1-N1
and as the reason corrections are predicted at ≥9/10, not 10/10.

## What the lead is asked to do

1. Decide the budget question (§6 of the ledger).
2. Check the three UNVERIFIED external citations (RAPTOR, LFS/Gray-Reuter,
   Haber-Stornetta) through the proxy — search terms are in the ledger.
3. Run r3 from `artifacts/grm_f1/lead_commands.txt` after removing
   `--dry-run` deliberately.
