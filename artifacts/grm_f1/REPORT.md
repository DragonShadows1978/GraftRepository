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

---

# §r3 — the three-arm campaign on the merged core

Receipts: `artifacts/grm_f1/lt1_1_r3/run_{A0,Aprime,Aplusprime}` (26/26 cells
each), summaries `lead_{A0,Aprime,Aplusprime}_summary.json`. Registered
scorer = `scripts/grm_lt1.score` (primary); c2 = amendment 9's second column.
Every number below is read from those receipts by this seat, not copied from
the dispatch note.

## Scores

**Primary scorer (`scripts/grm_lt1.score`) — recomputed by this seat from
`run_*/cells/*/probes.jsonl`:**

| class | r2 arm A | A0 (all OFF) | A′ (F1+F2+F5) | A+′ (A1+F1+F2+F5) |
|---|---|---|---|---|
| fresh | 10/15 | **10/15** | 11/15 | **13/15** |
| corrections | 9/10 | **9/10** | 9/10 | **10/10** |
| aliases | 10/10 | **10/10** | **7/10** | 10/10 |
| recap | 4/5 | **4/5** | **5/5** | 4/5 |
| total | 33/40 | **33/40** | 32/40 | **37/40** |

**Column 2 (amendment 9's second scorer) — NOT VERIFIABLE FROM THESE
RECEIPTS.** The dispatch reports A0 fresh 13, A′ 12, A+′ 14/15 and an A+′
total of 38/40. The c2 verdict is computed by a separate scorer and is
**not persisted in any r3 artifact**: `probes.jsonl` score objects carry
`exact_correct` / `category` / the abstention fields and no `col2_*` key,
and `lead_*_summary.json` carries only `correct` / `n` / `expected_n` /
`exact_rate` per class. Those c2 figures are therefore recorded here as
**reported by the lead, unverified by this seat** — they are not used in any
conclusion below. Every claim in this section rests on the primary column.

**A0 reproduces r2 arm A exactly on every class.** The registration's
`attribution_rule` is therefore satisfied: core drift between r2 and today is
not in play, and movement in A′/A+′ is attributable to the flags.

## Rows that differ (11, not 9)

The dispatch named 9; the receipts show **11** — `recall_1_25` and
`recall_1_50` are two further A′/A+′ gains over A0 that the summary classes
absorb.

| row | A0 | A′ | A+′ | mount plan A′ (seats vs width 96) | cause |
|---|---|---|---|---|---|
| recall_1_25 | ✗ | ✓ | ✓ | `[19,11,6]` seated `[11,19]` | F1 gain: retained raw source 11 seats |
| recall_1_50 | ✗ | ✓ | ✓ | `[19,11,6]` seated `[11,19]` | F1 gain: same |
| recall_5_25 | ✗ | ✓ | ✓ | `[16]` | F1 gain: correction node 16 seats alone |
| recap_3 | ✗ | ✓ | ✓ | `[55,24,14]` seated `[55]` | era index serves; A0's plan head was wrong |
| **recall_3_10** | ✓ | **✗** | ✓ | `[12,5,10]` = 146 → drop **10** | **width eviction** |
| **recall_5_50** | ✓ | **✗** | ✓ | `[24,13,32]` = 180 → drop **32** | **width eviction + cross-entity leak** |
| **recall_7_10** | ✓ | **✗** | ✓ | `[12,5,18]` = 154 → drop **18** | **width eviction of the alias edge** |
| **recall_7_25** | ✓ | **✗** | ✓ | `[12,5,18]` = 154 → drop **18** | same |
| **recall_7_50** | ✓ | **✗** | ✓ | `[12,5,18]` = 154 → drop **18** | same |
| recall_3_100 | ✗ | ✗ | ✓ | `[55,24,14]` = 188 → drop `24,14` | A+′ plans `[14]` alone (53) |
| recap_2 | ✓ | ✓ | **✗** | — | A+′ mounts raw turn 13; vague assistant half |

### The single mechanism behind every A′ loss

`route_info.mount_dropped_for_width` names it on all five rows. F1's
retention adds the retained raw source to the rank plan, so the plan grows
from ONE member to THREE and the seat sum runs **146–188 against a width of
96**. The fit seats the first two and drops the third — and on these rows the
third is the only node carrying the identifier the question asked for.

`recall_7_10`, measured:

| arm | plan | ntok | seated | dropped |
|---|---|---|---|---|
| A0 | `[12]` | 35 | `[12]` | — |
| A′ | `[12, 5, 18]` | 35+56+63 = **154** | `[5, 12]` (91) | **`[18]`** |
| A+′ | `[20]` | 48 | `[20]` | — |

Admission IDENTIFIED node 18 in A′ (`admission_identified_candidates: [18]`,
branch `margin_insurance_k3_identifier_tiebreak`) and the fit then evicted
it. The abstention "We're still working on that" is therefore **honest**: the
mounted nodes (5, 12) say the *Lantern* launches 18 October 2196 and nothing
says the Beacon IS the Lantern, so the ladder had no grounded path from
"Beacon" to the date. It is not a grounding bug; it is a width eviction of
the one node that mattered.

**recall_5_50** is the same eviction with a worse surface. A′ seats nodes 13
(Commtower crew) and 24 (Commtower crew + Breakwater coords) — neither
mentions Medibay — and answers "keep it at 12". The "12" is the **z-coordinate
of Breakwater's `(-31, 48, 12)`** sitting in node 24: a cross-entity numeric
leak out of a co-mounted node, not a model prior and not a retrieved Medibay
value. Node 16 ("Medibay's bed count will be 16 beds") was identified and
never planned.

**recap_2** is A+′'s only regression. A0/A′ mount the ERA INDEX (nodes 61/55)
whose prose names "Iona Vale"; A+′ plans `[13]`, the raw turn, whose user half
says "Commtower's maintenance crew will be Iona Vale" but whose assistant half
is "That gives this area a clearer identity" — the model paraphrased to "the
crew that handled the tower's upkeep". A raw turn is a worse reader than a
digest when the fact lives only in the user half.

## The "Beacon" node per arm (item 3) — F2 is vindicated

| arm | nodes mentioning "Beacon" | verdict |
|---|---|---|
| A0 | 3 (turn 18 retired, **digest 24 retired**, **era 61 ACTIVE**) | **capture present** |
| A′ | 1 (turn 18, ACTIVE, the raw alias edge) | **no capture** |
| A+′ | 2 (turn 19 retired, digest 20 ACTIVE = A1 merge) | **no capture** |

A0 node 24, over sources `[13, 14, 17, 18]`:

> "the maintenance crew of **the Beacon** is located at Iona Vale, and the map
> position of **the Beacon** is (-31, 48, 12)."

Commtower's crew and Breakwater's coordinates, both re-filed onto "the
Beacon", and it propagated into the ACTIVE era node 61. **This is exactly the
defect F2 exists to prevent, present in the control arm and absent from both
F2 arms.**

The proof is a same-window comparison, not an inference: in A′ the digest at
the same index has sources `[13, 14]` — the alias turns 17 and 18 were
EXCLUDED from the window — and reads "the maintenance crew of **Commtower**
… the map position of **Breakwater**". Correct attribution, same fold, same
cell. (`fold_guard_history` is not persisted into the cell receipts, so the
window-source comparison is the receipt; noted as a successor.)

## Interaction statement (item 2)

**Yes — and the receipts state the mechanism precisely.**

1. F2 removes fact-less alias/rename turns from every fold window. Correct,
   and it kills the capture defect.
2. Because they are never folded, those turns are never superseded either. In
   A′ the Beacon edge (node 18) ends the campaign `active=True`,
   `digest_of=None`, and **the source of nothing** — a lone raw turn holding
   a relation and no value.
3. F1 keeps every folded source routable, which inflates rank plans from one
   member to three.
4. The lone alias edge is admitted, ranked **third**, and evicted for width.

So the alias relation is present, reachable in principle, and structurally
unable to reach a seat alongside the digest that holds the value. **A1's merge
removes exactly that failure**: node 20 carries `alias_merge=True`,
`supersedes=[19]`, and states relation AND value in ONE 48-seat node — "the
Kestrel's cargo allowance is 37 crates, and the Lantern's launch date is 18
October 2196. The Lantern is also known as the Beacon." One mount answers the
Beacon question, which is why A+′ scores 10/10 on aliases.

**What A′ would need to serve aliases with alias-merge OFF.** Any ONE of:
(a) a co-mount rule that seats the alias edge WITH its base rather than
ranking them as competitors — the width receipt says 63+35 = 98 > 96, so this
needs a narrower edge node or a widened arena, and FIX-7 was already STOPPED
as impossible at this width; (b) plan-head protection, so an
identifier-bearing member is never the one evicted; or (c) a two-hop read that
resolves the alias before routing, which the A1 design deliberately rejected
in favour of the write-time join.

**Registered finding: the four flags are a SET at this arena width.** F1+F2
without A1 is not a partial improvement — it is a *regression on aliases*
(10/10 → 7/10) created by the interaction of two individually-correct
treatments. F2 makes alias turns unfoldable; F1 makes the plans too wide to
seat them. A1 is what closes the loop. **Do not merge F1+F2 without A1 on the
strength of A′.**

And on the column that can be audited from these receipts, A′ is a **net
regression**: 32/40 against the control's 33/40. The fresh and recap gains
(+4 rows) do not pay for the alias and Breakwater losses (−5 rows). That is
the whole case for treating the flags as a set.

## Residency (all arms bounded)

| arm | nodes | active | retained-source | rows | max seats (width 96) | max retained seats | rows w/ retained mount |
|---|---|---|---|---|---|---|---|
| A0 | 199 | 36 | 0 | 323 | 94 | 0 | 0 |
| A′ | 228 | 213 | 156 | 322 | **95** | 95 | 71 |
| A+′ | 200 | 175 | 113 | 309 | 94 | 80 | 33 |

No `RESIDENCY_BOUND_EXCEEDED` on any row in any arm. The registered
prediction "residency bounded, max seats reported" HOLDS. But the A′ numbers
are the warning the prediction could not anticipate: a routable base of 213
nodes against 96 seats does not overflow the bound — it overflows the *plan*,
and the eviction lands on the node the question named.

## Verdict against the registered predictions

| arm | predicted | measured | verdict |
|---|---|---|---|
| A0 | within ±1 of r2 arm A, all classes | exact match, all classes | **MET** |
| A′ | fresh ≥14/15 | 11/15 (c2 12) | **REFUTED as stated** |
| A′ | corrections ≥9/10 | 9/10 | MET |
| A′ | aliases 10/10 ("must not move at all") | **7/10** | **FAILED** |
| A′ | recap ≥4/5 | 5/5 | MET |
| A+′ | fresh ≥14/15 | 13/15, **c2 14/15** | MET on c2, missed on primary |
| A+′ | corrections ≥9/10 | 10/10 | MET |
| A+′ | aliases 10/10 | 10/10 | MET |
| A+′ | recap ≥4/5 | 4/5 | MET |

The A′ alias row is the registered falsifier firing verbatim: *"any alias row
below 10/10 means F2's window exclusion is interacting … and the pair must be
separated before either is merged."* It fired on A′, not on A+′, and the
separation is now diagnosed rather than merely detected.

Honest reading of A+′. On the column I can verify it is **37/40 against the
control's 33/40** — a real and substantial gain. The "38/40 c2, best on
record" headline is the lead's, on a scorer whose output no r3 receipt
contains, so I neither confirm nor dispute it. And FOUR flags moved together
in that arm: the receipts license "the SET A1+F1+F2+F5 beats the control on
this fixture", never a single-flag claim.

**Successor: persist the column-2 verdict into `probes.jsonl` and the
summary.** Amendment 9 bound c2 as a registered scoring column; a campaign
that reports it should leave a receipt for it, or the column cannot be
audited from the artifacts the campaign produced.

## RED, root-caused: the follow-up-3 gate was never running the CPU double

`test_real_resume_route_through_pending_and_one_run_cell` failed once with
`rc == 2`, passed ten times, and I twice wrote it up as an unexplained flake.
Both write-ups were wrong.

Iteration 5 of a reproduction hunt failed **all three arms** in 32.78 s
(instead of 102 s) and captured the child's `worker.log`:

```
scripts/grm_lt1_1.py:880   lt1_1_worker -> worker.execute(..., gpu_loader)
scripts/grm_lt1_1.py:891   gpu_loader -> e2e.load_model_and_repo(...)
scripts/grm_e2e_session.py:2197  GptOss20B_TC.from_pretrained(...)
```

The child was loading the **real GPT-OSS-20B model**. The test patched
`worker.spawn_argv` before calling `resume()`, and `resume()` then enters
`lt1_1_seams`, which assigns `worker.spawn_argv = spawn_argv` (the real
`--worker` argv). The seam silently overwrote the patch, so every run —
including the "passing" ones — spawned a real model load; whether it failed
fast enough to surface as `WORKER_EXIT_1` varied with page-cache state. The
tell was in the timings the whole time: 102 s for what should be a CPU-double
spawn.

Fixed by patching `runner.spawn_argv`, the function the seams *install*, so
the patch survives them. The gate now runs in **15.9 s** and genuinely
exercises the CPU double.

Method note worth keeping: I ruled out five hypotheses by check and never
asked what the child actually did. The `worker.log` was always on disk.
"Ruled out by check" is not "diagnosed".
