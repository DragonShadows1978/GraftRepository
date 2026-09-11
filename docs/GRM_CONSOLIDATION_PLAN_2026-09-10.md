# Lead's to-do list — GRM consolidation arc (plan, 2026-09-10 22:50 EDT)

Goal (David, 2026-09-10): "build and execute your personal to do list of
projects, run to completion. we can run Opus as execution agents, I want
to save my codex for later in the week." Seats = Opus 5 (opus-max) only;
no Codex dispatch. Lead plans, verifies, commits, runs GPU. This plan is
immutable after commit (house rule); execution details go to the ledger
(AI_Research_Board.md entries dated 2026-09-10/11) and the synthesis at
/mnt/Shared/GRM_Consolidation_Result_2026-09-1x.md.

## Phase 1 — GRM (GraftRepository, branch lc1-wip)

M1. Merge the Scout follow-through branches into one line:
    grm-fix1 → grm-c2 → grm-c7 → grm-lt1 → grm-rd1 onto lc1-wip
    (worktree wt/grm-merge, branch grm-merge; fast-forward lc1-wip
    when the full CPU suite is green). Acceptance: every FIX-1..6/8
    fixture suite green on the merged tree; C2's 132 recorded plans
    byte-identical under the default rule. Lead does the git; a seat
    resolves conflicts only if there are any.

R1. Margin-first regression replay (the registered 31 execution cells,
    9 distinct C2 questions, wt/grm-lt1 artifacts/grm_scout_fix6/
    c2_replay_cells.json): build the replay worker (seat), lead runs on
    GPU under the lease (budget ≤ 1.0 GPU-h). Both sides per cell:
    today's rule vs margin_first from the same recorded checkpoint.
    Registered prediction: ≤ 2 of 31 executions change their answer;
    0 correct→wrong on supersession. Verdict rule: margin_first is
    adoptable as the profile default iff correct→wrong = 0 on sup and
    ≤ 1 elsewhere.

A1. Alias resolution, option (b) fold-merge (memo
    wt/grm-c7/artifacts/grm_c7/r3/ALIAS_DESIGN_OPTIONS.md): librarian
    pairs an alias edge with its current base as one fold job; FIX-5
    digest carries both names, relation, value; flag-gated
    (GRM_ALIAS_FOLD_MERGE, default OFF); two-hop read NOT built.
    CPU fixtures RED→GREEN on the 8 RD2 alias rows; GPU contrast
    from the C7 r3 checkpoints (≤ 0.5 GPU-h). Prediction: aliases
    ≥ 5/8 exact from a single digest mount, 0 controls broken.

D1. LT1 residual diagnosis: the 5 wrong corrections and 5 wrong
    aliases of LT1 arm A, per row, from the LT1 checkpoints (which
    node mounted; stale vs current; alias edge vs base) — same shape
    as RD2. CPU + ≤ 0.3 GPU-h. Output: a per-row cause table and the
    recap battery redesigned as five answerable questions (registered
    before LT1.1).

LT1.1. Re-run LT1 arm A (profile, margin_first) on the merged core
    with A1 ON and the D1 recap, same frozen 200-turn conversation
    (~1.7 GPU-h). Registered prediction: exact ≥ 28/35 (fresh ≥ 14/15,
    corrections ≥ 7/10, aliases ≥ 7/10), recap ≥ 3/5, 0 abstentions,
    residency bounded, restarts retained.

P1. Product surface: `scripts/grm_chat.py` — a runnable interactive
    session (GPT-OSS-20B, EB1 frame, C2 profile, chat log never in
    context, all recall via GRM, restart-resumable), the thing David
    sits down at for the live 200-turn proof. Ships with a smoke gate
    on the fake model and a 10-turn GPU smoke. Defaults stay as
    shipped; the profile is one switch (GRM_PROFILE=eb1_c2).

S1. Synthesis + docs: results file on Shared, RESULTS_INDEX + primer
    rows, board entries, decision list for David (profile default,
    margin_first default, alias flag, abstention prompt).

## Phase 2 — only after Phase 1 closes
Q1. Qwen3.8-27B lc1-wip merge state + LC1 speed item (memory
    project_qwen38_27b): verify, merge or record blocker.
K1. APA-SP1.1 decode split-K (memory project_apamq): registered
    successor; build + kernel sweep receipts.

## Not on this list (David's decisions, untouched)
Delve L2Q/L2C/e2/SD1; Frontier INT6 merge and fix/map-context-live;
Project-Frontier GitHub repo keep/delete; demand threshold flip; GRAPA
training (power-safety pause).
