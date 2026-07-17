# CAMPAIGN-A: M10 lifecycle FakeArena fix + _ensure_h named error

YOUR WRITABLE TARGETS in /mnt/ForgeRealm/GraftRepository (main tree):
- tests/test_grm_runtime_lifecycle.py
- core/graft_arena.py
- tests/test_grm_ensure_h_error.py (new)
Everything else READ-ONLY (sibling seats work other files in this tree
— never touch tests/test_grm_s4_demotion.py, core/graft_repository.py,
or tests/test_grm_importance_g1g2.py). No git. No subagents. No GPU /
no model loads. RED honesty; no monitor-idling.

## Part 1 — M10 (docs/GRM_BUG_QUEUE.md, read the M10 entry first)
tests/test_grm_runtime_lifecycle.py is 91/101 RED on main: its
FakeArena double lacks _bump_cuda_gqa_epoch (call site
core/graft_repository.py:3779, landed with the July-8 bridge merge).
Add the method (no-op) to the double AND audit the double against the
current ArenaCache surface used by graft_repository.py — every method
graft_repository calls on self.arena must exist on FakeArena (grep the
call sites; list what you added in the report). Goal: the suite runs
GREEN again so the bug queue's rule-2 gate protects future fixes.
Do not weaken any assertion to get there — if a test fails for a REAL
behavioral reason after the double is fixed, report it as a finding.

## Part 2 — _ensure_h named error (docs/GRM_BUG_QUEUE.md M11 entry,
secondary item)
core/graft_arena.py _ensure_h currently lets an unbackable payload
fall through silently (h stays None; callers crash later with a bare
TypeError — that's how M11 presented). Design: a named exception
(e.g. GraftPayloadMissingError carrying the node ids that could not
be resolved) raised by _ensure_h when, after its heal attempts, any
requested graft still has h=None. AUDIT EVERY CALLER first
(consolidate, swap/_graft_block, _graft_pair_blocks, any others) and
state per call site whether raising is safe — mounting or folding an
unbacked node is never valid, so the expectation is yes everywhere,
but verify against the code, not the expectation. The M11 fold guard
already keeps folds away from placeholders; this makes the remaining
paths fail LOUD and named instead of deep and bare.
tests/test_grm_ensure_h_error.py: unit tests — unbackable node →
named error listing ids; healable node (host_payload / disk) →
heals, no raise; mixed set → error names only the unbackable ones.

## Verify
py_compile; python3 -m pytest tests/test_grm_runtime_lifecycle.py tests/test_grm_ensure_h_error.py tests/test_grm_fold_recovered_guard.py tests/test_grm_importance_telemetry.py -q
(telemetry tests construct MLAAttentionTC which allocates on device —
if your sandbox lacks CUDA they fail with cudaMalloc; that exact
signature is a known environment artifact, report it as such and do
not chase it.)

## Done
Verbatim: per-file diff summary; the methods added to FakeArena; the
per-call-site raise-safety audit; pytest summary line; spec
ambiguities; "no git, no GPU, no files outside grant".
