# step() generate mechanical diagnosis

## Verdict
- **step() vs manual loop body: REFUTED** (shared KV snapshot → identical tokens; both HIT `Vortex-3-Sierra`).
- **Real class: (a) position/live_shift misalignment** at session `arena_width=384` (`live_shift≈387`).
- Candidates (c) stop-slicing and (d) sampling: not primary. (b) resume: secondary amplifier via poisoned plants.

## Divergence receipt (shared-state A/B)
- Step: **none** (A≡B token-for-token).
- Setup: `pos=97`, `S=116`, `live_shift=115`, mounts=`[0]`, fact `Vortex-3-Sierra` deposited+evicted+remounted.
- Prompt (step harmony): sha256 `9025cc17865f2b94e8a4df8822ec0333347363728197752deb443441bb0d47f0`.
- A/B text: `I will respond with "Vortex-3-Sierra" as the stored value`.

## live_shift sweep (fact-plant, same prompt bytes)
| case | live_shift | answer head |
|------|------------|-------------|
| pure | None | 'I’m sorry, but I can’t comply with that.<|return|><|return|>' |
| sess_w384 | 387 | 'User: 3-V> < 3-4> <5 6-7-' |
| sess_w96 | 99 | 'Sure! If you have any questions or need assistance, feel fre' |
| sess_w16 | 19 | 'Sure! If you have any questions or need assistance, feel fre' |
| sess_w0 | 4 | 'I’m sorry, but I can’t comply with that.' |
| sys_w96 | 115 | 'I’m sorry, but I can’t comply with that.' |
| sys_w384 | 403 | '**. The user.' |
| sess_w384_shift_nsink | 3 | 'I’m sorry, but I can’t comply with that.' |
| sess_w384_shift0 | 0 | 'I’m sorry, but I can’t comply with that.' |
| sess_w1 | 4 | 'I’m sorry, but I can’t comply with that.' |

## Minimal fix (DO NOT implement here)
1. `scripts/grm_e2e_session.py` `--arena-width` default **384→96** (Leg-1 proven) and pass Harmony `sink_text=SYS`.
2. Prefer `feed()` complete turns for fact plants (Leg-1), or keep free-gen only under safe live_shift.
3. Product investigation: GPT-OSS YARN + large arena RoPE hole (`graft_arena.py:77`, `gpt_oss20b_tc.py:686-691`).

## Artifacts
- shared A/B: `artifacts/grm_e2e/step_vs_manual_ab_shared_20260708_155615`
- width sweep: `artifacts/grm_e2e/live_shift_width_ab_20260708_155826`
- rebuild A/B (false positive knife-edge): `artifacts/grm_e2e/step_vs_manual_ab_20260708_155312`
