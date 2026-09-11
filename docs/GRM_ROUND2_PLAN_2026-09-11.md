# Lead's to-do, round 2 — GRM fold fidelity at distance (plan, 2026-09-11 11:40 EDT)

David: "you pick a project and do round 2." Lead's pick: the top residual
of round 1. Seats = Opus 5 (opus-max) only; no Codex. Lead plans, verifies,
commits, runs GPU. Plan immutable after commit; ledger on the board;
synthesis on Shared. Base: GraftRepository `lc1-wip` f6c607a.

## The residual (round-1 receipts, LT1.1 r2)
Source nodes are never mounted (0/57 hits in both arms); every answer is
read from a fold digest or era. Arm A (flag OFF, shipping core): fresh
10/15 (13/15 glyph-tolerant), corrections 9/10, aliases 10/10, recap 4/5.
Misses: fresh facts at d≥100 (Breakwater coordinates confabulated; the
mounted node is an unrelated fact each time), a chronicle fold that bound
Commtower's crew and Breakwater's coordinates onto "the Beacon"
(Lantern's alias), and one abstention.

## Items
F1. Source retention (core, flag `GRM_FOLD_RETAIN_SOURCES`, default OFF):
    FIX-5 folds retire their sources on coverage ≥ 0.7, so the only
    routable record of a fact becomes its digest. Variant: sources stay
    active and routable after a fold (digest added, not substituted);
    residency/width accounting measured. Registered prediction on LT1.1
    arm A (flag-tolerant column c2): fresh 13 → ≥ 14/15; residency
    bounded (max seats reported, expected higher); corrections/aliases
    unchanged (≥ 9/10, 10/10).
F2. Chronicle fold alias capture (core, flag `GRM_FOLD_ALIAS_GUARD`,
    default OFF): an alias/rename turn inside a fold window must not
    rebind other sources' facts onto the alias. Fixture = arm A digest 24
    (sources turns 13,14,17,18) RED→GREEN on the CPU double; production
    prompt change (FIX-5 enumeration names the entity per fact) or
    exclusion of alias turns from chronicle windows — seat picks by
    evidence, both stated. Prediction: no "Beacon"-bound digest in the
    LT1.1 run; Breakwater/Commtower facts stay entity-correct in every
    digest that carries them.
F3. Routing at distance (diagnosis first, core only if a defect is
    proven): for the Breakwater misses (A d=100/150, recap_3) and the
    Commtower fabrications (A+ d=100/150), the per-row route receipts:
    what ranked, why the intact record did not (identifier set of the
    digest; margin; recency; era shadowing). Deliverable: cause table +
    registered fix proposal; a fix lands only with a CPU RED→GREEN and
    stays flag-gated.
F4. Scorer: the glyph-tolerant column (D1 amendment 9) becomes the
    registered secondary for round 2; abstention regex extended with the
    observed phrasing ("we're still working on that") — registered before
    any run, applied to r2 rows too (report both).
L1. LT1.1 r3, arm A only (alias flag OFF), same frozen conversation,
    same worker, F1+F2(+F3 if landed) ON vs the r2 A baseline;
    ≤ 1.3 GPU-h. Registration pins REPO-RELATIVE paths only.
P2. Product smoke r4: the 20-turn `grm_chat.py` transcript with the
    round-2 flags ON (≤ 0.2 GPU-h); gate unchanged (fresh ≥ 3/4).
S2. Synthesis on Shared + board + decisions (flip F1/F2 defaults = David).

## Rules carried from round 1
Every registration pins repo-relative paths; tests that copy repositories
use `--basetemp` on NVMe and clean up; a seat's CPU proof traverses the
exact code path the GPU run takes; receipts are the small JSON, never
`repository/native/*.bin` or checkpoints in git; never kill foreign
processes; every guard states when it stops applying.
