# ORDER IME-E1b — Repair the Gemma dense control instrument

YOUR WRITABLE TARGET is /mnt/ForgeRealm/GraftRepository-wt-e1 (a worktree
checkout of GraftRepository) — edits and GPU runs inside it are
AUTHORIZED. Also writable: `/tmp`. Everything else is READ-ONLY reference
(Project-Tensor engine, model weight dirs, the canonical checkout, the
sibling worktree -wt-e2).

Read `docs/HYBRID_IME_E1E2_PLAN.md` first (immutable — plan wins on
conflict; report conflicts, don't resolve silently).

## Context

The E1 primary run (qwen35 hybrid) is COMPLETE and parity-clean; its
artifacts are in `logs/ime_e1_qwen35.npz` / `logs/ime_e1_summary.json`.
The dense-model CONTROL arm failed. `tests/gemma4_attn_dist.py` is
bit-rotted against the current Gemma port:

    File "tests/gemma4_attn_dist.py", line 65, in prob_call
        k = new_kv[0]
    TypeError: 'KVRing' object is not subscriptable

The instrument was written when the attention mixer returned a plain
tuple; `core/gemma4_tc.py` now returns a `KVRing` object. Model load,
weights, and VRAM were all fine (48/48 layers, peak 7316 MiB) — the
failure is purely this interface change, at the first decode step.

## Task

Repair the instrument minimally and run the control.

1. Read `core/gemma4_tc.py` to find how `KVRing` exposes its K and V
   tensors (accessor methods / attributes / an `ordered()`-style call) and
   what the row ordering and valid-row count are. Do NOT guess the API —
   read it. Do NOT modify `core/gemma4_tc.py`; the fix belongs in the
   test instrument.
2. Update `tests/gemma4_attn_dist.py` so it reads K (and V if it uses it)
   from the ring correctly, honoring valid-row bounds. Keep every
   statistic and the output format it already computes — this is an
   interface repair, NOT a redesign, and its numbers must stay comparable
   to what the instrument was always meant to produce.
3. Make it emit, in addition to whatever it already prints, the SAME
   summary columns as the qwen35 run so the two are directly comparable:
   per attention layer — mean entropy, effective key count, and mean
   fraction of attention mass in the top 5% / 10% / 15% of keys, averaged
   over heads and the captured decode steps. Gemma has 40 sliding + 8
   global layers: report sliding and global layers as clearly separate
   groups (they are different regimes; do not pool them).
4. Run it and write artifacts alongside the qwen35 ones:
   `logs/ime_e1_gemma.npz`, `logs/ime_e1_gemma_summary.json`, printed
   table to `logs/ime_e1_gemma_run.log`.

## Rails

- GPU: wrap every GPU-touching invocation in
  `flock -w 3600 /tmp/forge-gpu.lock <cmd>`; run ≤10 min; ONE model
  resident at a time (Gemma INT4 ~6.8 GB peaked 7316 MiB on a 12 GB card;
  the sibling E2 seat shares this lock — queuing is expected, not a hang).
- Launch DETACHED (`nohup` + flock, as the earlier E1 runs did) so the run
  survives your turn; end your turn after launching rather than polling,
  then report when it lands.
- NO git operations — the lead commits. NO subagents.
- RED honesty: if the ring API makes a faithful repair impossible, or the
  run OOMs, or numbers come out non-comparable, SAY SO plainly with the
  error text. A control that cannot be run is a valid, reportable outcome;
  a fabricated or silently-approximated one is not.
- Evidence class: instrument measurement only. No model-quality claims.

## Done

Final message must contain, verbatim:
- The exact `KVRing` accessor you used and the line(s) you changed.
- The printed Gemma summary table (sliding and global groups separate).
- The parity/sanity check the instrument performs, as printed.
- Exact paths of every file created or modified.
- Wall-clock and peak VRAM of the run.
- Any deviation from this order or the plan, stated as a deviation.
