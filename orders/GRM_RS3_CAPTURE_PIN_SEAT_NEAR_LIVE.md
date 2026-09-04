# GRM-RS3 — Close the mounted-read gap: pin capture geometry, seat the plan head next to the live band

David, 2026-09-04 morning: "You can keep going. Figure it out."

The gap (RS1 e3f525d): the same text read LIVE draws ~0.74 attention
mass and answers; MOUNTED it draws ~0.18 and refuses. RS2 (887e96a)
found no single cause but two levers that move it, and the seam under
one of them:

- **Capture geometry is unpinned.** `ArenaCache.deposit` harvests with
  queries at `[0, L)` (`live_shift=None` → fallback); any served turn
  leaves `live_shift = n_sink + arena_width = 115`; the identical
  `deposit(text)` afterwards yields a DIFFERENT graft (layer 0 same,
  layers 1–23 changed; storage-quant round trip matches neither).
  Capturing at 115 (RS2 B1p) lifted harbor 0.18 → 0.30 and flipped it.
- **Band position.** Seating the mount adjacent to the live band (RS2
  B3a) moved 26% of the gap — the largest movement — but the probe
  script's makeshift injection broke generation. Production seats every
  mount at `[n_sink, n_sink + ntok)` (the `_attempt` bootstrap branch
  concatenates sink + mounts; `GptOssAttentionTC.__call__` RoPEs the
  block at `cos.slice(0, 0, graft_seats)`).

This order builds both levers as PRODUCTION seams behind flags (default
OFF), measures them alone and together against the live-band ceiling,
and — if the pair reproduces the ceiling — runs both lived batteries
with the pair ON. The flip stays David's.

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` (branch
`lc1-wip`, HEAD 887e96a) — edits, builds, test runs, and BOUNDED GPU
runs AUTHORIZED. A registered order IS the permission: run first,
report after.

## Context (read fully, in order)

1. `orders/GRM_RS1_READ_STRENGTH_PROBE.md`, `orders/GRM_RS2_MOUNT_READ_DEFICIT.md`,
   `artifacts/grm_rs1/grm_rs1_results.json`, `artifacts/grm_rs2/grm_rs2_results.json`
   (esp. RS2 §4 "seams the probe script implements that production
   lacks" — items 3, 4, 5 are this order's spec), and the RS2 probe
   scripts (`scripts/grm_rs2_*.py`: `LayerTypeMassObserver`,
   `_deposit_with_prefix`, `_positioned_injection`, the digests
   e8192df3… / ce417319… that pin the capture-position finding).
2. `core/graft_arena.py`: `deposit` (~505), `deposit_from_cache` (~518),
   `_harvest`, `_attempt` (bootstrap branch; injection/seating order),
   `feed`, `live_shift` handling, `eb1_begin_turn`; `core/gpt_oss20b_tc.py::GptOssAttentionTC.__call__`
   (~705, `cos.slice(0, position_offset + shift, L)`, `graft_seats`).
   `docs/GRM_Methodology.md` §4 (pre-RoPE relocatable keys — the
   invariant that must survive: a graft re-seats exactly at any
   position).
3. `core/grm_frame.py` (flag resolver pattern: explicit bool outranks
   env; unknown token fails closed), `core/grm_demand.py`,
   `core/grm_three_pass.py` prefixes.
4. The lived batteries under the spec frame: `scripts/lsr_p2c_replay_gpu.py`
   (sup, deposit-order) and `scripts/lsr_p2c_e2e_gpu.py` (census
   34-turn sharded) as EB1 ran them; EB1 receipts
   (`artifacts/grm_eb1/`, sup 5/9, census 9/10).

## Part 1 — Capture-geometry pin (`GRM_CAPTURE_PIN`, default OFF)

Every harvest path (`deposit`, `deposit_from_cache`, the split-child
deposits from P2C, the abstention deposit) captures with the query
position pinned to a REGISTERED geometry instead of whatever
`live_shift` the previous turn left behind. Two candidate pins, both
implemented, selected by the flag value:
- `mount`: queries at the geometry the graft will be READ in when
  mounted at the band start (`n_sink`, i.e. as the block `[sink |
  text]`); this is what "a graft is the text to the model" literally
  requires when seating is at the band start.
- `live`: queries at `live_shift` (RS2 B1p's geometry) — the geometry
  of text fed live immediately before a question.
OFF = legacy (unpinned; byte-identical grafts to today, test pins the
digest e8192df3… on the RS2 fixture text). A capture receipt field
`capture_shift` and `capture_pin` on every graft.

## Part 2 — Seat the plan head next to the live band (`GRM_SEAT_NEAR_LIVE`, default OFF)

Through PRODUCTION's injection path (not the RS2 probe hack): when ON,
the mount block is seated so the plan head's LAST token is adjacent to
`live_shift` (filler and other mounts below it, sink untouched), i.e.
the band is filled from the top down with the plan head nearest the
question. Pre-RoPE relocatable keys make this a positioning change
only; prove it: a test that mounts the same graft at both ends and
checks the K/V payload is unchanged while the RoPE-applied keys differ
only by the rotation. OFF = byte-identical to today (test pins served
text on a fixture turn). Receipt fields: `seat_offset_plan_head`,
`seat_order`.

Sliding-window note: with the head adjacent to the live band it is
trivially inside the 128-row window on sliding layers; report the
sliding-layer mounted mass so the RS2 ~15% number has a comparison.

## Part 3 — Measurement (GPU, bounded; reuse RS2's runner and observer)

Probes: RS1/RS2's five (+ the solace-fact variant). Arms, spec frame,
fixes ON, demand OFF:
- C0: both OFF (must reproduce RS2 B0 bit-equal).
- C1m: capture pin `mount`, seat OFF. C1l: capture pin `live`, seat OFF.
- C2: capture OFF, seat ON.
- C3m: pin `mount` + seat ON. C3l: pin `live` + seat ON.
- C5: the live ceiling (RS2 B5) as the reference row.
For capture arms, the installed fixture grafts are RE-CAPTURED under
the pin (same texts, via the same installer path with the flag ON) —
say exactly which deposit path produced each graft and give its digest.

Registered predictions (lead): C3l or C3m reaches mounted-band mass
≥ 0.50 on the refusers and flips ≥ 2/3 with correct values; C1 alone
≈ RS2 B1p (harbor only); C2 alone moves mass ≥ 0.10 without breaking
generation (the RS2 B3a movement, now with production seating). If
C3 stays ≤ 0.35 on the refusers, the residual is sink/RoPE distance
itself and this order reports that as its result.

## Part 4 — Lived batteries with the winning pair (only if Part 3's
best arm meets the mass ≥ 0.50 OR flips ≥ 2/3 line; otherwise SKIP and
say so)

Sup battery (9) and census (10), spec frame, fixes ON, demand OFF, the
winning arm's flags ON. Registered prediction: sup ≥ 7/9 (the three
refusers recover, nothing correct becomes wrong); census 9/10 with 0
regressions (t33's admission miss is not this order's). Then the
long-horizon spot check: the 4 distance-36–60 probes from EB1 G5 stay
correct.

## Gates (synchronous, FOREGROUND, in-call waits only; RED is a result)

Register before any gate run (`artifacts/grm_rs3/registration.json`):
both pins' exact geometry (numbers derived from the arena, not typed),
the seating rule, arms, predictions. **The registration file is
IMMUTABLE after it is written**: any amendment goes in a SEPARATE file
that cites the registration's sha256 (RS1's discipline, not RS2's).

G1. `python3 -m pytest -q` sharded under 10 min per call — pre-existing
    set unchanged vs RS2's logs; new tests: flags OFF byte-identity
    (graft digest e8192df3… reproduced; served text on a fixture turn
    unchanged); pin geometry receipt; relocatable-keys invariant under
    seat ON; seat-order receipt; flag resolvers fail closed.
G2. C0 reproduces RS2 B0 5/5 bit-equal.
G3. Part 3 table: probe × arm × (served, correct, mounted mass full,
    mounted mass sliding, live mass, sink mass, top layer, graft digest,
    seat offset). Verdict per lever and for the pair in the registered
    vocabulary: CLOSES (≥ 0.50 mass or ≥ 2/3 flips), MOVES (measurable,
    below both), NOTHING; predictions hit/miss.
G4. Part 4 tables (or the SKIP statement with the number that
    triggered it).

GPU conventions: `/tmp/forge-gpu.lock` lease via the repo's
`gpu_lease`, single GPU, every run ≤ 10 min wall, ≥30 s gaps inside
the wrapper, operator has absolute right of way. Frozen run tree never
written.

## File boundary

Modify ONLY: `core/graft_arena.py`, `core/gpt_oss20b_tc.py` (the RoPE
slice / seating seam ONLY — no attention-math changes; APA/attention
kernels untouched), `core/grm_frame.py` (flags), `core/graft_repository.py`
(capture receipt plumbing only), `core/grm_three_pass.py` (prefix line +
test only if `capture_`/`seat_` need persisting), `scripts/grm_rs3_*.py`
(new; import RS2's runner/observer), `scripts/lsr_p2c_replay_gpu.py` /
`scripts/lsr_p2c_e2e_gpu.py` (flag plumbing + receipt fields only),
`tests/`, `artifacts/grm_rs3/`, `logs/`. Read-only: every
`scripts/grm_det1_*.py`, `scripts/grm_sc*.py`, `scripts/grm_eb1_*.py`,
`scripts/grm_rs1_*.py`, `scripts/grm_rs2_*.py` (import only), `docs/`,
`orders/`, all receipts.

## Principles (binding)

Both flags default OFF; OFF paths byte-identical (test-pinned); the
flip is David's with the Part 4 receipts. Pre-RoPE relocatable keys
remain an invariant (test). Registration immutable; amendments
separate. Determinism. NO git (lead commits). NO subagents. **NO
background processes for waiting**: no `run_in_background`, no
watcher/poll loops, no "block until X exits" helpers; foreground only
with explicit `timeout`, wait in-call, every Bash call under 10
minutes. **Never kill, signal, or interrupt any process you did not
start** (multiple projects share this machine; no kill/pkill/killall/
fuser -k/gpu-reset/systemctl; pgrep/ps hits are off-limits; wait on the
flock or report blocked). Acknowledge process safety in the report.
RED honesty. Verify your own claims against artifacts before writing.

## Done (verbatim in the final message)

1. pytest summary line(s) + pre-existing set status.
2. Registration block as written before gates (pins' geometry numbers,
   seating rule, predictions) + its sha256; any amendment file.
3. G2 reproduction; G3 table + verdicts + predictions hit/miss; G4
   tables or SKIP statement.
4. Files created/modified; the exact production sites changed; new
   receipt fields; the OFF-path byte-identity receipts.
5. Ambiguities, deviations, residual risks, anything RED; process-
   safety acknowledgement; and the numbers David flips on.
