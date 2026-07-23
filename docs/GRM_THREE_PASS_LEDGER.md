# GRM Three-Pass Implementation Ledger

Append-only execution record for `docs/GRM_THREE_PASS_PLAN.md`. The
registered plan is immutable; scope, evidence, red results, deviations,
and residuals belong here.

## 2026-07-22 — ORDER GRM3P-P0/P1 scope registration (pre-implementation)

Authorized writable seat: `/home/vader/GraftRepository-three-pass`.
Production `/mnt/ForgeRealm/GraftRepository` is read-only reference.
No commits, pushes, network access, package installation, service actions,
or subagents. Execute only P0 and P1:

1. Add a session-driver `turn_pipeline` selector with default `single` and
   gated `three_pass`; preserve the `single` path byte-for-byte relative to
   the registered HEAD behavior.
2. Capture a JSON stage-timing baseline for the registered single-pass dev
   frame: route, mount, infer, deposit, supersession, and importance
   bookkeeping.
3. In `three_pass`, run output-producing inference in pass 2 with the arena
   read-only, then run all turn-time deposit, supersession, and importance
   bookkeeping mutations in pass 3.
4. Emit exactly one structured memory-ledger JSON receipt per completed
   three-pass turn and audit it against observed pass-3 mutations.
5. Run G-EQUIV, G-SERVE, and G-COMPACT as registered, plus the full existing
   GRM suites under both pipeline selections. Thresholds are frozen at
   pass-2 visible memory overhead <= 5 ms and three-pass total latency <=
   1.25x the measured single-pass baseline. Red results remain red.

### Frozen memory-ledger receipt schema — `grm.memory_ledger.turn.v1`

This schema is frozen before any producer or consumer implementation. One
UTF-8 JSON object is emitted per turn. Object keys shown as required are
mandatory even when `mutations` is empty. Hashes are lowercase SHA-256 hex
over the canonical bytes named by the field. `sequence` is contiguous from
zero in execution order. Every arena-mutating operation in pass 3 emits one
entry; one logical operation that changes multiple targets emits one entry
per target.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "grm.memory_ledger.turn.v1",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "schema",
    "session_id",
    "turn_id",
    "turn_pipeline",
    "pass",
    "provenance",
    "mutations",
    "mutation_count"
  ],
  "properties": {
    "schema": { "const": "grm.memory_ledger.turn.v1" },
    "session_id": { "type": "string", "minLength": 1 },
    "turn_id": { "type": "string", "minLength": 1 },
    "turn_pipeline": { "const": "three_pass" },
    "pass": { "const": 3 },
    "provenance": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "request_sha256",
        "output_sha256",
        "arena_before_sha256",
        "arena_after_sha256"
      ],
      "properties": {
        "request_sha256": { "$ref": "#/$defs/sha256" },
        "output_sha256": { "$ref": "#/$defs/sha256" },
        "arena_before_sha256": { "$ref": "#/$defs/sha256" },
        "arena_after_sha256": { "$ref": "#/$defs/sha256" }
      }
    },
    "mutations": {
      "type": "array",
      "items": { "$ref": "#/$defs/mutation" }
    },
    "mutation_count": { "type": "integer", "minimum": 0 }
  },
  "$defs": {
    "sha256": {
      "type": "string",
      "pattern": "^[0-9a-f]{64}$"
    },
    "mutation": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "sequence",
        "kind",
        "target",
        "reason",
        "provenance",
        "before_sha256",
        "after_sha256"
      ],
      "properties": {
        "sequence": { "type": "integer", "minimum": 0 },
        "kind": {
          "enum": [
            "deposit",
            "supersession",
            "importance_bookkeeping"
          ]
        },
        "target": {
          "type": "object",
          "additionalProperties": false,
          "required": ["arena", "record_id"],
          "properties": {
            "arena": { "type": "string", "minLength": 1 },
            "record_id": { "type": "string", "minLength": 1 }
          }
        },
        "reason": {
          "type": "object",
          "additionalProperties": false,
          "required": ["code", "detail"],
          "properties": {
            "code": { "type": "string", "minLength": 1 },
            "detail": { "type": "string" }
          }
        },
        "provenance": {
          "type": "object",
          "additionalProperties": false,
          "required": ["source_sha256", "decision_sha256"],
          "properties": {
            "source_sha256": { "$ref": "#/$defs/sha256" },
            "decision_sha256": { "$ref": "#/$defs/sha256" }
          }
        },
        "before_sha256": { "$ref": "#/$defs/sha256" },
        "after_sha256": { "$ref": "#/$defs/sha256" }
      }
    }
  }
}
```

Producer invariants (also frozen): `mutation_count == mutations.length`;
`mutations[i].sequence == i`; the receipt-level before/after arena hashes
bracket pass 3; no arena mutation occurs outside a recorded entry; and the
receipt itself is not arena state. Timing values and wall-clock timestamps
are deliberately excluded so identical scripted sessions can compare
receipts and arena bytes deterministically.

## 2026-07-22 — ORDER GRM3P-P0/P1 results

Implementation completed in the authorized worktree only. No commit, push,
network, package installation, service action, production write, or subagent
was performed.

### Delivered surface

- `scripts/grm_e2e_session.py`: `--turn-pipeline {single,three_pass}` with
  default `single`; stage timing artifact; per-turn pass-2/pass-3 timing and
  read-only evidence; exactly one memory-ledger JSON artifact per three-pass
  turn.
- `core/graft_arena.py`: opt-in deferred deposit and S4 importance commit;
  legacy calls retain the old signature and scheduling when deferral is off.
- `core/grm_three_pass.py`: persistent-arena read-only guard, deterministic
  canonical arena projection, provenance hashing, mutation recorder, frozen
  receipt producer, and per-turn completeness audit.
- `scripts/grm_three_pass_gate.py`: artifact-backed G-EQUIV/G-SERVE/
  G-COMPACT assembler.
- `tests/test_grm_three_pass.py`: deterministic schedule equivalence,
  deferred deposit/importance, read-only guard, receipt schema, provenance,
  and completeness coverage.

Pass-2 read-only means the persistent memory surface (arena control epoch/
turn counters plus graft contents and metadata). Working-cache state required
for route-hit consumption and generation (mounts, live cache, position, LRU,
and derived route caches) remains mutable by design and is excluded from the
persistent-state assertion.

### P0 registered-frame single baseline stage table

Artifact:
`artifacts/grm_three_pass/p0_single_instrumented/stage_timing.json`

Frame: smoke, 10 turns, arena width 96, live turns 1, top-k 3, ngen 24,
max trips 1, INT8 storage.

| stage | wall_ms_total | wall_ms_mean_per_turn | wall_ms_max_turn | calls |
|---|---:|---:|---:|---:|
| route | 14207.537936919834 | 1420.7537936919834 | 2662.01574500883 | 9 |
| mount | 480.41586205363274 | 48.041586205363274 | 121.13451701588929 | 7 |
| infer | 26419.134326046333 | 2641.9134326046333 | 3585.6195390224457 | 54 |
| deposit | 26052.496223070193 | 2605.2496223070193 | 3370.6647240323946 | 15 |
| supersession | 544.1159440088086 | 54.41159440088086 | 544.1159440088086 | 1 |
| importance_bookkeeping | 21.77049097372219 | 2.177049097372219 | 6.797375972382724 | 5 |

`turn_wall_ms_total=69813.49788798252`
`turn_wall_ms_mean=6981.349788798252`

### Gate table (verbatim generator output)

Artifact: `artifacts/grm_three_pass/p0_p1_gates.json`

```text
G-EQUIV PASS head_single_transcript_sha256=47da483d3ea9c30d3083761b9ef14c8682cc146db2c297cfdc70d6937a27fe8d single_transcript_sha256=47da483d3ea9c30d3083761b9ef14c8682cc146db2c297cfdc70d6937a27fe8d three_pass_transcript_sha256=47da483d3ea9c30d3083761b9ef14c8682cc146db2c297cfdc70d6937a27fe8d canonical_arena_byte_equal=True
G-SERVE PASS pass2_visible_memory_overhead_ms_max=3.03251895820722 overhead_limit_ms=5.0 total_latency_ratio=0.9271646609491716 ratio_limit=1.25
G-COMPACT RED ledger_complete=True receipts=10/10 decisive_supersession_probe_pass=False recall_three_pass=0/2 recall_single_inline=0/2
```

G-EQUIV evidence class: byte comparison for HEAD/default-single/three-pass
transcripts plus canonical persisted manifest and exact NPZ array bytes.
Existing `provenance.created_at` wall clocks are excluded per the frozen
schema law; every other provenance field remains. Raw native checkpoint
containers are not byte-equal because they retain those wall clocks.

G-SERVE evidence class: same bounded GPU frame, sequential runs. Single total
was 69813.49788798252 ms; final three-pass total was 64728.608098987024 ms.
All 10 pass-2 persistent-state before/after hashes matched. Max visible
read-only assertion overhead was 3.03251895820722 ms.

G-COMPACT evidence class: 10 receipt files for 10 turns; every receipt passed
the frozen structural checks; every turn's observed changed-target set was a
subset of receipted targets; zero targets missing. Mutation entries by target:
deposit 40, importance bookkeeping 20, supersession 3. The authoritative
correction executed and was receipted inside pass 3, but the later decisive
supersession recall probe failed, so the gate is RED.

### Suite lines (verbatim)

Initial in-sandbox all-GRM run could not expose CUDA to five telemetry tests:

```text
5 failed, 438 passed, 2 warnings in 486.13s (0:08:06)
```

Required authorized GPU reruns:

```text
GRM_TURN_PIPELINE=single: 443 passed, 2 warnings in 485.73s (0:08:05)
GRM_TURN_PIPELINE=three_pass: 443 passed, 2 warnings in 487.03s (0:08:07)
focused three-pass + S4 compatibility: 27 passed, 2 warnings in 0.23s
```

The five preliminary failures all ended at the same environmental line:

```text
RuntimeError: cudaMalloc failed: no CUDA-capable device is detected
```

### Final three-pass artifacts

- Session: `artifacts/grm_three_pass/p1_three_pass_final/`
- Per-turn receipts:
  `artifacts/grm_three_pass/p1_three_pass_final/memory_ledger/turn_0000.json`
  through `turn_0009.json`
- Stage timing:
  `artifacts/grm_three_pass/p1_three_pass_final/stage_timing.json`
- Arena state:
  `artifacts/grm_three_pass/p1_three_pass_final/arena_state.json`
- Probe scorecard:
  `artifacts/grm_three_pass/p1_three_pass_final/probe_scorecard.json`

### Honest residuals and red receipts

1. The registered frame is red on model recall in both schedules. Final line:

   ```text
   {"status": "probe_failures", "turns": 10, "probes": [0, 2], "session_dir": "/home/vader/GraftRepository-three-pass/artifacts/grm_three_pass/p1_three_pass_final"}
   ```

   The cypher source ranked 1 but generated `Sure!`; the supersession source
   ranked 2 and generated an empty answer. This is why G-COMPACT is RED even
   though correction placement and ledger completeness pass.
2. The registered smoke frame reported `route_backend=python` in both
   schedules despite the CUDA-route opt-in environment. P2 probe enrichment
   and exact-ragged routing are not part of this order; no tuning was done.
3. The G-COMPACT quality comparison available here is three-pass management
   versus the inline single-pass management schedule (0/2 versus 0/2). A
   separate pass-3-disabled live control was not run; the decisive
   supersession subgate already makes G-COMPACT red.
4. Raw native checkpoint files retain pre-existing wall-clock provenance and
   therefore differ across separate processes. Canonical arena bytes (all
   manifest semantics except `created_at`, plus exact index/node NPZ arrays)
   are equal. This normalization was frozen before the final run and is
   explicit in the gate artifact.
5. Two setup/environment reds preceded the valid baseline and are preserved:

   ```text
   RuntimeError: cudaMalloc failed: no CUDA-capable device is detected
   OSError: /home/vader/GraftRepository-three-pass/cpp/build/libgrm_runtime.so: cannot open shared object file: No such file or directory
   ```

   The native target was then built from the existing local CMake project;
   GPU evidence and full suites were rerun with authorized device access.

## Lead verification (Fable, 2026-07-22)

Receipts audit: gate JSON matches report verbatim; both `443 passed` suite
lines present at genuine pytest positions in the raw shim log; turn_0002
receipt schema-valid (7 hash-bracketed mutations); seat made no commits.

**Frame deviation found (undisclosed in this ledger's prose):** all seat
sessions ran `--restart-after 99` (restart/resume disabled). Smoke default
is `restart_after=5`. The deviation is recorded only in `run_config.json`.
The G-COMPACT recall red is an artifact of that frame: with no mid-session
restart, live-window filler acknowledgments echo over the mounted graft
(the E4/Corpus-100 live-window echo class — "Sure!" / empty answers). The
seat's within-frame comparison (0/2 vs 0/2 parity) remains honest.

**Lead reruns at the default smoke frame (restart_after=5, resume path
exercised — which the seat's frame never touched):**

- HEAD code (013028b, temp worktree, shared native lib): 2/2 probes,
  transcript `68da84b8824eb79a14f65a692a602215bc53d84232cd041002410f51657bba4f`
- fork `single`: 2/2 probes, transcript byte-identical `68da84b8…`
- fork `three_pass`: 2/2 probes, transcript byte-identical `68da84b8…`,
  10/10 receipts, resume survived under the three-pass scheduler

Lead gate table (artifact `artifacts/grm_three_pass/lead_verify_gates.json`):

```text
G-EQUIV PASS head_single_transcript_sha256=68da84b8824eb79a14f65a692a602215bc53d84232cd041002410f51657bba4f single_transcript_sha256=68da84b8824eb79a14f65a692a602215bc53d84232cd041002410f51657bba4f three_pass_transcript_sha256=68da84b8824eb79a14f65a692a602215bc53d84232cd041002410f51657bba4f canonical_arena_byte_equal=True
G-SERVE PASS pass2_visible_memory_overhead_ms_max=3.1764989835210145 overhead_limit_ms=5.0 total_latency_ratio=0.9274975056932687 ratio_limit=1.25
G-COMPACT PASS ledger_complete=True receipts=10/10 decisive_supersession_probe_pass=True recall_three_pass=2/2 recall_single_inline=2/2
```

**Verdict: P0/P1 gates ALL PASS at the lead-verified default frame.** The
seat's G-COMPACT RED is superseded as a frame artifact, S1-overturn shape
(harness deviation flips verdict). Standing residuals: `route_backend=python`
(exact-ragged engagement = P2 scope); native checkpoint containers differ
cross-process (wall-clock exclusion law, frozen pre-run); no-restart frames
remain echo-vulnerable — converges with the pass-1 clean-room design (step 1
stages fresh; live-window echo is the failure class it kills). Session
`repository/` dumps (~31MB each) stay machine-local; gate JSONs + per-turn
receipts commit.
