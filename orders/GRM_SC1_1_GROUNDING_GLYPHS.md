# GRM-SC1.1 — Grounding v3 obeys "value comparison is semantics, not glyphs"

Lead-authored 2026-09-02 morning from the SC1 finding. SC1's demand trip
fetched the withheld node, mounted it, and the model said the right
value 2/2 — and grounding v3 rejected both, because the model emits
U+2011 (non-breaking hyphen) and `_rare_tokens`' character class
`[A-Za-z0-9][\w:.,\-]*` does not contain it: "Quartz‑8‑Jade" shatters
into `{"8"}` while the mounted node's ASCII "Quartz-8-Jade" tokenizes
whole. The DET1 comparator (`normalize_value_text`) already collapses
U+2010/U+2011 and Markdown emphasis; grounding does not. This is the
principle minted in the race ("value comparison is semantics, not
glyphs"; three incidents) applied to the one comparator it never
reached. Same principle, no new threshold.

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` (branch
`lc1-wip`, HEAD e99840d) — edits, builds, test runs, and BOUNDED GPU
runs AUTHORIZED. A registered order IS the permission: run first,
report after.

## Context (read fully, in order)

1. `orders/GRM_SC1_DEMAND_LOOP_NGH.md` and the SC1 receipts
   (`artifacts/grm_sc1/sc1_g3_summary_*.json`:
   `recovery_blocked_by_grounding_only = 2`).
2. `core/graft_arena.py::_rare_tokens` (~1608), `_caps_tokens`,
   `_grounding_attribution` (~2238; the three branches: hedge, identifier
   gate `qrare & mounted`, content ⊆ have / substantive-words fallback),
   `_grounded`.
3. `scripts/grm_det1_common.py::normalize_value_text` (~221) and the
   DET1.10 separator registration (~186–215): the registered glyph
   classes (Markdown emphasis, U+2010, U+2011; separator-vs-space is
   a comparator-only rule — NOT for grounding, see below).
4. `scripts/grm_e2e_session.py::lsr_fixes_enabled` / `GRM_LSR_FIXES`
   (P2C's fixes switch; Arm 0 = OFF reproduces lived) and
   `scripts/lsr_p2c_replay_gpu.py`, `scripts/grm_sc1_recovery_gpu.py`,
   `scripts/grm_sc1_lived_battery_gpu.py`.

## Mission

1. **One normalizer, shared.** Add `normalize_glyphs(text)` in
   `core/grm_admission.py` (or a new tiny `core/grm_text_norm.py`) that
   collapses U+2010/U+2011 → "-" and strips Markdown emphasis markers
   (`**`, `__`, `*`, `_` around a token) EXACTLY as
   `normalize_value_text` does for those two classes. Do NOT fold
   separator-to-space or space-to-separator (DET1.10's rule is a
   comparator equivalence for scoring, not a tokenizer rule; folding it
   in grounding would let "Cobalt 1 India" ground against "Cobalt-1-
   India" AND against any text containing "cobalt", "1", "india"
   separately — state this in the docstring). A test pins byte-equality
   of `normalize_glyphs` vs `normalize_value_text` on a corpus of ≥ 50
   strings covering both classes, and pins that the separator/space
   classes are NOT folded.
2. **Apply it in grounding, gated.** `_rare_tokens` and `_caps_tokens`
   (and the substantive-words fallback) normalize their input first
   when the LSR fixes switch is ON; OFF = byte-identical legacy
   behavior (test pins both). Every consumer of `_rare_tokens`
   (routing `rare` caches, descent identifier filters, digest QC,
   P2A/P2C fit paths) goes through the same normalized path when ON —
   enumerate every call site in the report and say which are affected
   and which are on the OFF path.
3. **Grounding receipts.** `info["grounding_normalized"] = True|False`
   and `info["grounding_glyph_rescued"] = True` when the normalized
   verdict is True and the legacy verdict would have been False
   (compute both; cheap). `fit_`/`demand_` prefixes don't cover it, so
   add `"grounding_"` to `ROUTE_RECEIPT_INFO_PREFIXES` (one line + one
   test — the only edit to `core/grm_three_pass.py`).

## Gates (synchronous, FOREGROUND, in-call waits only; RED is a result)

Register before any gate run (`artifacts/grm_sc1_1/registration.json`):
the glyph classes folded (exactly two), the classes deliberately NOT
folded, and the predictions below.

G1. `python3 -m pytest -q tests/ --continue-on-collection-errors` —
    pre-existing failure/error set byte-identical to
    `logs/grm_sc1_pytest_final.log`; new tests as above.
G2. **SC1 recovery re-run** (`scripts/grm_sc1_recovery_gpu.py`, same 3
    reachable pairs, demand ON, fixes ON). Registered prediction:
    recovery 2/2 (`demand_served = "demand_trip"` on both planted
    misses), served value = expected on both; detection recall and
    false fires unchanged from SC1 (1.0, 1).
G3. **Lived battery** (`scripts/grm_sc1_lived_battery_gpu.py`, demand
    OFF and demand ON, fixes ON): the 9 sup probes serve values
    IDENTICAL to SC1 G4 / P2C Arm 1 (4 corrected + 5 unchanged);
    report per probe the legacy vs normalized grounded verdict and
    `grounding_glyph_rescued`. Prediction: served values 9/9 identical;
    `grounding_glyph_rescued = True` on the 4 P2C flips (their answers
    carry U+2011 and were served as ungrounded-kept-first — verify from
    the P2C receipts, and say so if the prediction is wrong).
G4. **Arm 0 still reproduces lived**: `scripts/lsr_p2c_replay_gpu.py`
    Arm 0 (fixes OFF) = 9/9 reproduction as in P2C (the normalization
    is on the OFF path's far side; this proves it).

GPU conventions: `/tmp/forge-gpu.lock` lease via the repo's
`gpu_lease`, single GPU, every run ≤ 10 min wall, ≥30 s gaps, operator
has absolute right of way. Fork-from-snapshot only.

## File boundary

Modify ONLY: `core/graft_arena.py`, `core/grm_admission.py` (or the new
`core/grm_text_norm.py`), `core/grm_three_pass.py` (the one-line prefix
+ its test ONLY), `scripts/grm_e2e_session.py` (flag plumbing/receipts
only), `scripts/grm_sc1_*.py` (extend receipts only), `tests/`,
`artifacts/grm_sc1_1/`, `logs/`. Read-only: every `scripts/grm_det1_*.py`
and `scripts/lsr_*.py` (import only), `docs/`, `orders/`, all receipts.

## Principles (binding)

Thresholds registered before gates, never adjusted after; no new
numeric constants at all in this order. Determinism. NO git (lead
commits). NO subagents. **NO background processes for waiting**: no
`run_in_background`, no watcher/poll loops, no "block until X exits"
helpers; foreground commands only, wait in-call; keep every single
Bash call under 10 minutes so the harness never auto-backgrounds it
(split long gates into shards). RED honesty. Verify your own claims
against artifacts before writing them.

## Done (verbatim in the final message)

1. pytest summary line + pre-existing set unchanged.
2. Registration block as written before gates.
3. G2 table (pair × fired / fetched / served-from / value / recovered);
   G3 table (9 probes × served value / identical / legacy grounded /
   normalized grounded / rescued); G4 reproduction count.
4. Files created/modified; every `_rare_tokens`/`_caps_tokens` call
   site with ON/OFF path status.
5. Ambiguities, deviations, residual risks, anything RED — and the
   updated flip numbers (recovery at 3 pairs; the 7 E2E pairs remain
   the SC1.2 successor).
