# GRM-SCOUT-FIX-1 registration (immutable)

Plan: orders/GRM_SCOUT_FIX_1.md, unchanged. Registered before executing gates.
Scope: B1, B2, B3, B5 production fixes and new C1 CPU tests only. No GPU,
git commands, subagents, background waits, kills, services, or existing battery edits.

Predictions and gates:
- B1: real DemandObserver wrapper + real ArenaCache attempt with CPU logits/mass
  seams. Flush-only fire should RED with DemandError before and GREEN after;
  preserve raw flush record, exclude unused prediction from decision, no abort state.
  Compare abort OFF/ON; first-token and stop-token abort/resume must preserve exact
  text, forward count, live bookkeeping and teardown. Stop with and without fire.
- B2: all four RT1 fields survive canonical receipt projection (demoted true/false),
  additive admission prefix and keys; no existing projection values changed.
- B3: ordinary save/reload preserves all capture metadata (pin live/off and live
  cache source); manifests without metadata retain absent capture fields and payload.
- B5: retain historical graft-count key; explicitly label graft count and token-seat
  count. Token seats use recorded route_receipt.fit.cur_mount_n (the serving sum of
  mounted ntok), over the SAME turns as the old count. Missing observations yield
  null with coverage/reason, never a fabricated zero or incomplete mean. Empty
  fits with explicit zero count as zero. Duplicate resumed turns still count once.
  Existing WC1 receipts are absent here but readable at canonical
  /mnt/ForgeRealm/GraftRepository/artifacts/grm_wc1_opus; read only, regenerate
  before/after in this worktree. Sup receipts lack token lengths: report unavailable
  unless a recorded exact sum exists. Recall cells must be unchanged.
- Run new tests against saved original sources before edits and after fixes.
  Run relevant existing CPU batteries untouched. Report missing external artifact
  dependencies and unrelated failures explicitly; do not relax assertions.
- After passing baseline, mutation checks on disposable copies: undo B1 guard,
  B2 prefix, B3 serialize, B3 reload, B5 token-count extraction. Require all five
  killed (>=0.80 non-error house threshold). No original sources mutated.

Rails: no thresholds/default/frame/grounding changes. If CPU fixture reveals broader
semantics, report and leave outside scope. No blind verification by author;
lead supplies independent verification and later EB1 GPU E2E gate.

Prior art: repository GRM-SC2, LSR-P2B, RS3 and WC1 (project contributors, 2026),
verified in local source. Reuse existing contracts and sums, only boundary and
projection repairs; no new algorithm claimed. External prior art: no prior art
known to me for these specific repairs. Tests reuse existing CPU fixture seams.

Identity: Codex, GPT-6 (system identity; exact API model identifier not exposed),
requested reasoning effort high. No claim of a runtime effort-setting change.
