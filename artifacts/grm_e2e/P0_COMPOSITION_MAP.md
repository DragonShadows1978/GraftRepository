# P0 Composition Map — GRM Composed E2E Receipt

Repo: `/mnt/ForgeRealm/GraftRepository` @ 846cebc, branch grm-cuda-bridge-overhead.
Read-only. Refs are `file:line` at this commit.

## 1. Turn execution — ArenaCache.step()

`core/graft_arena.py:1611-1750`. Per turn: sets `live_shift` on every layer
(1620-21); ephemeral-mode reset (1623-30, N/A for E2E, needs a persistent live
window); `live_idx` = seated grafts ∪ recency mounts (1632); `route_limit =
(max_trips+1)*topk` (1633); `ranking = self.route(...)` (1634, §4); snapshots
pre-attempt state as a 6-tuple for rollback (1635-36); PRECISE-MOUNT check for
identifier-shaped probes (1637-52); builds an `attempts` ladder — primary
(era-expanded) → descent (era+digest-expanded) → clean-room retry (1653-1707,
`fit()` truncates to `self.width` arena seats); per attempt calls
`self.swap(picks)` + `self._attempt(...)` (1709-1738), accepting the first
`_grounded()` (1506) hit or falling back to the first attempt (1739-1750).
`cur_mounts`/`cur_mount_n` (arena-seat bookkeeping) update inside `swap()`
(1388-1423) and `_attempt`'s bootstrap branch (1764-67). Deposit fires inside
`_attempt` (1806-1830): after generation, if `deposit=True`,
`deposit_from_cache` or `deposit` runs, then `live_segs.append((gidx,
seg_cache_ntok))` and `self.evict()` (1831, §3). Live tokens occupy seats from
`live_shift` on, growing until `evict()` trims them; mounts occupy the
fixed-width arena prefix `[n_sink, n_sink+width)`, replaced wholesale by
`swap()` — the two never overlap positionally (module docstring :1-27).

**Production callers**: exactly one, `core/grm_runtime.py:87-99`
(`GRMRuntime.chat()`), wrapping `step()` with `_snapshot_state`,
`_extract_from_new_turns` (librarian), `_finish_turn_event` (flush/page). All
other call sites are test/gate scripts (`tests/test_graft_gqa_arena.py:53`,
`test_graft_e4_arena.py:53`, `deepseek_grm_arena_gate.py:101`,
`test_graft_e4_trips.py:43`, `test_graft_corpus100.py:131`,
`test_graft_e4_consolidated.py:62`, `test_graft_gqa_features.py:46,83`). **No
script calls `step()` on GPT-OSS-20B at all** — every `gpt_oss20b_*_gate.py`
script uses subprocess `stream_forward_smoke.py` capture + a separate greedy
child (`scripts/gpt_oss20b_multifact_graft_gate.py:270-358`,
`gpt_oss20b_bulk_graft_gate.py:181-235`). `GRMRuntime.chat` is model-agnostic
and would work on GPT-OSS via `GraftRepository(arena_cls=GQAArenaCache, ...)`
— this composition has never been exercised.

## 2. Live witnessed deposit — NOT a gap, already exists and defaults on

`deposit()` (301-312): standalone dedicated forward, harvests fresh, stores
device tensors. `deposit_from_cache()` (314-341): **reads directly from the
LIVE `self.caches` tensors already in VRAM** via `_export_cache_payloads`/
`_export_cache_payload` (234-255, 197-221), which for GQA un-RoPEs the K span
by rotation composition at its absolute positions (`export_row_pair(s)` C++
path, `_export_cache_tensor` Python fallback at 182-195) — no re-forward.
Routing centroid still comes from a partial standalone forward through
`route_layer` (`_node_key`, GQA override 1895-99) unless `key_from_cache=True`
— a deliberate split (326, "MEASURED SPLIT"): payload from cache, key from
clean forward, since a contextualized centroid pollutes routing (early turns
become attractors, 5/6 regression noted in the comment).

`ArenaCache.__init__` defaults `cache_deposits=True` (44), so `step()`'s
deposit path (1806-1810: `deposit_from_cache if self.cache_deposits else
deposit`) **already runs the live-witnessed path in production** whenever
`step()` runs with `deposit=True` (GRMRuntime.chat's default). This
contradicts the plan's framing that "all recall gates capture grafts
OFFLINE" — true for every GPT-OSS gate script (§1), false for the
DeepSeek/GQA arena test suite, which already exercises `deposit_from_cache`
end-to-end (`test_graft_gqa_arena.py`, `deepseek_grm_arena_gate.py:91-93` via
`add_turn`→`feed()`). **There is no live K/V extraction gap at the arena
layer.** The real gap is narrower: GPT-OSS-20B has never been driven through
`ArenaCache`/`GQAArenaCache` at all (§1), so `deposit_from_cache`'s
cache-slicing math (generic over `self.PAYLOAD`) is UNTESTED against GPT-OSS's
actual cache tuple shape (k/v pairs per layer, sliding-window layers
included) — mechanism is dialect-generic and should apply unchanged, but
unverified live (§3).

`deposit_from_cache` requires only `text` and `seg_ntok` (cached token count
for the span, computed by `_attempt` itself at 1804, not caller-supplied) —
exactly what a live per-turn deposit needs. A session driver need only pass
`deposit=True` into `step()`/`feed()`; no new extraction primitive required.

## 3. Eviction / live window

No ring buffer or window-trim exists in the GPT-OSS driver itself:
`sliding_window` (`GptOssAttentionTC.__init__:632`, config default 128,
`core/gpt_oss20b_tc.py:59`) only builds an attention **mask**
(`_gpt_oss_attention_mask:612-619`, or the fused `sliding_sink_attention_tc`
path) — the K/V cache concat at :703-705 is unconditional regardless of
layer type. GPT-OSS layers never drop cache rows on their own; the model
always accumulates full history in `kv_cache`.

The ONLY live-window policy is `ArenaCache.evict()` (1425-1446): drops
`live_segs` beyond `self.live_turns`, splicing cache tensors down via
`_evict_cache_tensor`/`_evict_cache_payloads`. Count-of-turns policy
(`live_turns` kwarg, default 2), not token-budget-aware beyond that. Called
unconditionally at the end of every `_attempt` (1831) and inside `feed()`
(1477). **This is exactly the "turn N leaves live context" hook** the plan
needs — already wired, parametrized by `live_turns`.

Separately, `GraftRepository._page()` (`core/graft_repository.py:2914-2940`)
and `_free_retired()` (2947-2953) are a DIFFERENT axis: VRAM-budget paging of
mounted/repository graft tensors (device↔host), keyed on `last_used`
(mount-clock LRU, `_ensure_h`, `core/graft_arena.py:1224-1237`), gated by
`vram_budget_mb`. They never touch `live_segs` — they free device tensors for
deposited/retired nodes not currently mounted. Retirement (`retired=True`) is
a repository-level op (supersession/forget/expire —
`graft_repository.py:3121-3181` `retire_node`/`expire_nodes`), unrelated to
live-window membership. **No call site couples "turn leaves live window"
(evict) to "graft becomes eligible for cold storage" (page/retire)** — a
turn's graft is already an independent repository node the instant `deposit`
returns (§2); live-cache eviction is a VRAM/positional op on the *duplicate*
live copy, harmless in either order.

## 4. Route path / CUDA engagement

`step()` → `self.route(...)` (1634) → `route()` (343-400) → epoch-cached
eligible base list `_route_cand_base()` (368) → `_native_route_order(...)`
(373-374, GQA override 2127-2166), the CUDA/native dispatch point: sets
`self.last_route_backend = "cuda"` (762 MLA / 2124 GQA) or `"native"`
(849/866/874 MLA; 2162 GQA) or falls through to `"python"` (377, 84 default).
GQA gate is `_cuda_route_enabled()` (1925-1927): env var `GRM_GQA_CUDA_ROUTE`
truthy. Bank build/reuse is epoch-gated (`_cuda_route_bank_inputs`, GQA at
2000+; epoch = `_cuda_gqa_epoch`, bumped at every graft mutation via
`_bump_cuda_gqa_epoch`, called from `deposit` 311, `deposit_from_cache` 340,
`step`'s rollback 1717/1749, every `graft_repository.py` mutation site). The
route path DOES engage post-P1/P2 epoch machinery from inside `step()`
unmodified — no new wiring needed to have CUDA route active in a live
session; only `GRM_GQA_CUDA_ROUTE=1` in the driver's environment.

**Per-turn route wall instrumentation**: no existing hook times `route()`
itself. The reference pattern (`deepseek_grm_arena_gate.py:100-110`) wraps
the WHOLE `step()` call in `time.perf_counter()`, then reads
`last_route_backend` post-hoc. E3 (route wall ≤5ms) needs either (a) a
one-line product change returning/stashing a timestamp since `route()` is
called INSIDE `step()`, not by the caller, or (b) monkeypatching
`ArenaCache.route` from the driver to wrap-and-record without touching
product code — (b) satisfies "no code changes beyond genuine gaps."

## 5. Checkpoint / restart

`GraftRepository.flush_now()` (`core/graft_repository.py:3686-3759`): writes
WAL checkpoint record, per-dirty-node `.npz` payload (`pack_node`/packed
format, §6), `index.npz` (`pack_index`, centroids), `manifest.json` (dialect,
route_layer, wal_lsn, node metadata, native checkpoint pointer), and a native
store checkpoint if attached (`_native_save_checkpoint`). **Survives**:
grafts (text, ntok, tags, sources, retired flag, rare-token cache, centroid,
K/V payload if durable, native_node_id), WAL LSN, dialect descriptor,
route_layer. **Does NOT survive**: `arena.caches` (live KV tensors),
`arena.pos`, `arena.live_segs`, `arena.cur_mounts`/`cur_mount_n` — none
written by `flush_now` or read by `load()` (3765-3825+, only ever assigns
`self.arena.grafts = []` then rebuilds from manifest+npz, never touches
`arena.caches`). A fresh `GraftRepository` after restart calls `load()` in
`__init__` (252-253) — live window is implicitly empty (`self.caches = None`
from `ArenaCache.__init__:78`, never re-set by `load()`).

**Restart leg**: (a) construct a new `GraftRepository` at the same `path`
(triggers `load()`); (b) repository/grafts/route index/native ids/epoch
restore as a byproduct of `load()` + `_native_configure_arena` (native
re-attach at 247-248); (c) the live conversation window is genuinely GONE —
no "remount previous live turns" step exists. A resumed N≥30-turn session
must either (i) accept the live window resets empty (next turn re-routes
cold — the intended selective-amnesia semantics per the docstring :20-22) or
(ii) the P1 driver explicitly re-`feed()` the last `live_turns` turns from
its own transcript log to rebuild `live_segs`/`caches` before continuing —
a P1 DRIVER-LEVEL responsibility, not a product gap (live cache is
ephemeral scratch by design; grafts are the durable state). CUDA route bank
and native ids need no extra restart work beyond `load()` succeeding.

## 6. Packed store (GRM_GRAFT_STORAGE_BITS=8)

`GQAArenaCache.__init__` (1857-1881): `storage_bits` ctor kwarg or
`GRM_GRAFT_STORAGE_BITS` env var, validated against `SUPPORTED_BITS` (16
rejected as a no-op value). `pack_node()` (GQA override, 2182-2200): stacks
k/v fp16 then calls `pack_kv_arrays` when `storage_bits` set — invoked ONLY at
`flush_now()` time (`graft_repository.py:3717-3720`, `_ensure_host_payload` →
`_atomic_savez_compressed`), once per dirty node per flush, not per mount.
`unpack_node()` (2202-2215): auto-detects packed payload (`is_packed_payload`)
and dequantizes before device upload — invoked at `load()` (3810) and at
`_load_node` (`graft_repository.py:2956+`, the descent/cold-reload path used
by `_ensure_h` when a mounted-but-paged-out node is re-requested, :1224-1237).
**The 3.76×-slower mount cost lands at these two call sites** — checkpoint
restart's `load()` loop (once per durable node at process start) and any
mid-session descent reload of a previously retired/paged node (`_load_node`,
CPU dequant on the reload's critical path, directly inside a `step()` call
via `_ensure_h`). Nodes that stay device-resident all session (never paged,
never restarted) pay the packed cost exactly once — NOT a per-turn recurring
cost unless `vram_budget_mb` paging is active and churns nodes in/out.

## 7. Probe scorecard reuse

**No logit-margin scorer exists in this repo** (grepped `logit`/`margin`
broadly — hits are teacher-forced GT/parity harnesses for other work,
Gemma4/DeepSeek dumps, unrelated to recall grading). The actual recall grader
is lexical: `_grounded()` (`core/graft_arena.py:1506+`) checks answer tokens
against mounted-source content + question, filtering `HEDGES` (1483-1485) and
`SCAFFOLD` dialogue words (1490), plus `_caps_tokens` proper-noun extraction
(1492-1504). Gate scripts additionally do exact-substring accept-list checks
against planted values (`deepseek_grm_arena_gate.py:97-110`: `accepts` list,
`any(a in ans_l for a in accepts)`) and report `last_route_backend` per
probe. **P1's probe scorecard should reuse this pattern directly** —
planted-fact accept-lists + `_grounded()`'s own accept/reject ladder — rather
than build a margin-based scorer; nothing resembling "logit margin
machinery" exists to adapt.

## Gap list (ranked by risk)

1. **[BUILD, highest risk]** GPT-OSS-20B has never been driven through
   `ArenaCache`/`GQAArenaCache.step()`/`GRMRuntime.chat()` in one process —
   every GPT-OSS+GRM gate is subprocess-capture + separate greedy child.
   `GraftRepository(arena_cls=GQAArenaCache, **gpt_oss_grm_dialect_kwargs(cfg))`
   against the live `GptOss20B_TC` instance is UNTESTED: does
   `_attention_module()`'s `layer.self_attn` lookup, `_harvest`'s
   `harvest_kv` capture-flag mechanism, and `deposit_from_cache`'s
   `_export_cache_payload` PAYLOAD slicing round-trip correctly against
   GPT-OSS's real cache layout (full vs sliding layer types, YARN-scaled
   RoPE)? Nothing today exercises this combination live — P1's central risk.
2. **[BUILD]** Per-turn route-wall instrumentation (E3) — no hook isolates
   `route()`'s own wall time from `step()`'s total; needs a monkeypatch
   wrapper or a minimal product-side timestamp return.
3. **[BUILD, driver-level]** Restart-leg live-window continuation — `load()`
   restores the repository but never `live_segs`/`caches`; if "session
   continuation" must mean more than "repository intact, next turn routes
   cold," the driver must re-`feed()` prior turns from its own transcript
   log. Explicit P1 design decision (this map takes no position).
4. **[WIRE]** `GRMRuntime.chat()` already does the full turn sequence
   (route→swap→generate→deposit→librarian→flush→page) — P1 constructs it for
   GPT-OSS and calls it in a loop; no new orchestration logic.
5. **[WIRE]** `evict()` and `_bump_cuda_gqa_epoch` already fire automatically
   inside `step()`/`feed()` — zero new eviction/invalidation code.
6. **[WIRE]** `GRM_GRAFT_STORAGE_BITS=8` is a pure env-var flip already
   load-bearing at `flush_now`/`load`/`_load_node`.
7. **[WIRE]** `GRM_GQA_CUDA_ROUTE=1` is a pure env-var flip already
   load-bearing inside `route()` — read `last_route_backend` per turn.
8. **[WIRE]** Probe scorecard — reuse `_grounded()` + planted accept-lists
   exactly as `deepseek_grm_arena_gate.py` does; no new scorer.

## P1 architecture sketch (session driver loop, real function names)

```
repo = GraftRepository(
    model=gpt_oss_model, encode=tok.encode, decode=tok.decode,
    path=SESSION_DIR, arena_cls=GQAArenaCache,
    **gpt_oss_grm_dialect_kwargs(cfg),
    native_lib_path=NATIVE_LIB, storage_bits=8,
    vram_budget_mb=BUDGET, live_turns=LIVE_TURNS)
runtime = GRMRuntime(repo)                    # core/grm_runtime.py
os.environ["GRM_GQA_CUDA_ROUTE"] = "1"

turn_log = []                                 # for the restart leg
for turn_idx, (user_text, is_probe) in enumerate(SCRIPTED_TURNS):
    ans, info = runtime.chat(user_text, ngen=NGEN, max_trips=MAX_TRIPS)
    route_wall = <captured via wrapped ArenaCache.route, gap #2>
    turn_log.append((user_text, ans))
    record_instrumentation(turn_idx, {           # E3 gate: route_wall<=5ms
        "route_wall_ms": route_wall * 1000,
        "route_backend": repo.arena.last_route_backend,
        "mounts": info["mounts"], "resident": info["resident"],
        "evicted": info["evicted"], "live_tokens": info["live_tokens"],
        "repo_nodes": len(repo.arena.grafts), "vram_mb": measure_vram()})
    if is_probe:
        score_probe(ans, PROBE_ACCEPTS[turn_idx])   # _grounded()-style, §7

    if turn_idx == CHECKPOINT_TURN:
        repo.flush_now()                            # §5 durability leg
        del repo, runtime
        repo = GraftRepository(model=gpt_oss_model, encode=..., decode=...,
                               path=SESSION_DIR, arena_cls=GQAArenaCache,
                               **gpt_oss_grm_dialect_kwargs(cfg),
                               native_lib_path=NATIVE_LIB, storage_bits=8)
        # load() in __init__ restores repo/route index/native ids/epoch.
        # Live window is NOT restored (§5) — driver re-seeds it (gap #3):
        for prior_user, prior_ans in turn_log[-LIVE_TURNS:]:
            repo.arena.feed(f"User: {prior_user}\nAssistant: {prior_ans}\n",
                            deposit=False)     # grafts already durable
        runtime = GRMRuntime(repo)

final_stats = repo.get_stats()                # E4: stable VRAM check
repo.flush_now()
```
