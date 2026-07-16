"""M11-class bug regression: librarian fold-source selection must exclude
WAL-recovery placeholder nodes (docs/GRM_BUG_QUEUE.md — new work order,
2026-07-16, filed by the lead while running the G1/G2 legs).

Mechanism (verified by reading the code, not just the report): a
GraftRepository booted from a workspace with wal/ state but incomplete
node payloads rebuilds grafts via _wal_placeholder_graft() with
kind="turn" (default), ntok=0, retired=False, h=None, host_payload=None,
payload_pending=True, recovered=True — text/metadata-authoritative but
with NO K/V payload anywhere (RAM or disk). These count as foldable
"turn" nodes under the pre-fix _foldable()/_active() filter (which only
checked retired/live/kind/no_fold, never payload resolvability), so
_fold_once() could select one as a fold source. arena.consolidate()
(graft_arena.py, out of this work order's ownership) mounts sources by
indexing grafts[i]["h"][li] directly — its own preceding _ensure_h() call
routes through node_loader() -> _load_node(), and _load_node() already
downgrades a missing-file node via _mark_payload_missing() (sets h=None,
payload_pending=True) INSTEAD of raising, so grafts[i]["h"][li] on a
freshly-downgraded h=None is the first place this ever surfaces as a bare
TypeError. Fix: exclude non-resolvable-payload nodes from fold selection
entirely, in BOTH the Python-fallback path (_foldable's `ok` filter,
feeding both the deferred idle() path and the inline-backpressure path —
both funnel through the same _librarian_jobs()/_foldable() choke point)
and the native-store-backed path (_native_foldable's output filter, which
had its own independent eligibility check with no payload-resolvability
test at all — the two-paths-drift precedent the lead flagged).

CPU only, no GPU, no model load. FakeArena mirrors
tests/test_grm_runtime_lifecycle.py's convention, WITH _bump_cuda_gqa_
epoch as a no-op (M10: that method landed on GraftRepository's mutation
choke points 2026-07-08 and the lifecycle suite's FakeArena double never
grew it — not re-introducing that gap here) and a real consolidate() that
reproduces the exact grafts[i]["h"][li] indexing the production bug
report describes, so the pre-fix regression genuinely raises TypeError
here too (not just "the count changed").
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graft_repository import GraftRepository


# ============================================================================
# CPU fixtures
# ============================================================================

class FakeSelfAttn:
    def __init__(self):
        self.live_shift = 32
        self.inject_kv = None
        self.graft_seats = 0


class FakeLayer:
    def __init__(self):
        self.self_attn = FakeSelfAttn()


class FakeConfig:
    num_layers = 2
    hidden_dim = 128
    kv_lora_rank = 32
    qk_rope_head_dim = 8


class FakeModel:
    config = FakeConfig()

    def __init__(self):
        self.layers = [FakeLayer() for _ in range(self.config.num_layers)]


class FakeArena:
    """Mirrors tests/test_grm_runtime_lifecycle.py's FakeArena, plus a
    real consolidate() that reproduces the production indexing bug
    (grafts[i]["h"][li]) so this suite proves the guard at the SAME site
    the bug report names, not a stand-in assertion."""
    PAYLOAD = (("c", 1), ("kpe", 2))
    VALS_PER_TOK_LAYER = 48
    ENABLE_ERA_FOLDING = True
    TEXT_SCAFFOLD_CONSOLIDATION = False

    def __init__(self, model, encode, decode, route_layer=1, live_turns=2,
                **_):
        self.m = model
        self.encode = encode
        self.decode = decode
        self.route_layer = route_layer
        self.live_turns = live_turns
        self.live_segs = []
        self.grafts = []
        self.node_loader = None
        self.page_ins = 0

    def _bump_cuda_gqa_epoch(self):
        # M10: GraftRepository's mutation choke points call this
        # unconditionally; the lifecycle suite's double lacked it and
        # went red on every mutating test. Not repeating that gap here.
        self._cuda_gqa_epoch = getattr(self, "_cuda_gqa_epoch", 0) + 1

    @staticmethod
    def _rare_tokens(text):
        return {w.lower() for w in text.split() if any(c.isdigit() for c in w)}

    def deposit(self, text):
        idx = len(self.grafts)
        self.grafts.append({
            "h": None,
            "cent": np.full((32,), float(idx), dtype=np.float32),
            "ntok": len(self.encode(text)),
            "text": text,
        })
        return idx

    def feed(self, turn_text):
        idx = self.deposit(turn_text)
        self.grafts[idx]["kind"] = "turn"
        self.grafts[idx]["h"] = self._real_payload()
        self.live_segs.append((idx, self.grafts[idx]["ntok"]))
        self.evict()

    def evict(self):
        # Mirrors ArenaCache.evict()'s live_segs bookkeeping (no tensor
        # cache to trim here): turns beyond the recency window drop out
        # of live_segs, which is what makes them fold-eligible at all
        # (_active() excludes anything still in live_segs).
        if len(self.live_segs) > self.live_turns:
            self.live_segs = (self.live_segs[-self.live_turns:]
                              if self.live_turns > 0 else [])

    def _real_payload(self):
        return [{"c": np.zeros((1, 4), dtype=np.float32),
                "kpe": np.zeros((1, 1, 4), dtype=np.float32)}
               for _ in self.m.layers]

    def _ensure_h(self, idxs):
        # Mirrors ArenaCache._ensure_h: re-load via node_loader if h is
        # missing, but does NOT invent a payload — a loader that returns
        # None (production's _load_node on a missing file) leaves h=None,
        # exactly reproducing the unguarded crash site.
        for i in idxs:
            g = self.grafts[i]
            if g.get("h") is None and self.node_loader is not None:
                g["h"] = self.node_loader(i)

    def _set_inject(self, att, blk):
        att.inject_kv = (blk["c"], blk["kpe"])
        att.graft_seats = 1

    def _clear_transients(self):
        pass

    def consolidate(self, idxs, ngen=None):
        """Minimal but FAITHFUL reproduction of ArenaCache.consolidate()'s
        crash site: mount every source by indexing grafts[i]["h"][li].
        A placeholder that reached here (h=None, no loader-side recovery
        possible) raises the exact TypeError the production bug report
        describes — this is the assertion that the guard actually
        prevents the crash, not just that node counts moved."""
        self._ensure_h(idxs)
        for li in range(len(self.m.layers)):
            for i in idxs:
                _ = self.grafts[i]["h"][li]        # production crash site
        text = "ARCHIVE NOTE. " + " ".join(
            self.grafts[i]["text"] for i in idxs)
        didx = self.deposit(text)
        self.grafts[didx]["kind"] = "digest"
        self.grafts[didx]["sources"] = list(idxs)
        self.grafts[didx]["h"] = self._real_payload()
        for i in idxs:
            self.grafts[i]["retired"] = True
        return didx, text

    def pack_node(self, h):
        return {"payload_id": np.asarray([0], dtype=np.int64)}

    def unpack_node(self, z):
        return {"id": 0}

    def pack_index(self):
        cents = [g["cent"] for g in self.grafts]
        return {"cents": np.stack(cents) if cents else np.zeros((0, 32),
                                                                np.float32)}

    def unpack_index(self, z, i):
        return z["cents"][i].astype(np.float32)


def enc(text):
    return text.split()


def dec(ids):
    return " ".join(str(i) for i in ids)


def make_repo(tmp_path, **kwargs):
    return GraftRepository(FakeModel(), enc, dec, str(tmp_path),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=1, **kwargs)


def add_real_turns(repo, n, prefix="turn"):
    for k in range(n):
        repo.add_turn(f"user says {prefix} {k}", f"assistant answers {k}")


def add_foldable_real_turns(repo, n_foldable, prefix="turn"):
    """Add enough real turns that n_foldable of them clear FakeArena's
    recency window (live_turns) and become fold-eligible — mirrors
    ArenaCache.feed()'s evict() dropping old live_segs entries. A turn
    only becomes a fold candidate once it's no longer "live"."""
    add_real_turns(repo, n_foldable + repo.arena.live_turns, prefix=prefix)


def add_recovered_placeholder(repo, text="", kind="turn"):
    """Synthesize a WAL-recovery placeholder exactly as
    _wal_placeholder_graft() would produce it (mirrors that function's
    field set, not a simplified stand-in)."""
    idx = len(repo.arena.grafts)
    repo.arena.grafts.append({
        "kind": kind,
        "text": text,
        "ntok": 0,
        "sources": [],
        "retired": False,
        "no_fold": False,
        "tags": [],
        "rare": set(),
        "cent": np.zeros(32, dtype=np.float32),
        "metadata": {"kind": kind, "active": True},
        "provenance": [],
        "host_payload": None,
        "host_present": False,
        "device_present": False,
        "dirty": False,
        "durable": False,
        "cold_only": False,
        "payload_pending": True,
        "recovered": True,
        "h": None,
    })
    repo._ensure_lifecycle(idx, repo.arena.grafts[idx])
    return idx


# ============================================================================
# 1. _has_resolvable_payload: the new predicate, unit-tested in isolation
# ============================================================================

def test_resolvable_payload_true_for_resident_h(tmp_path):
    repo = make_repo(tmp_path)
    idx = add_recovered_placeholder(tmp_path and repo)
    repo.arena.grafts[idx]["h"] = [{"c": 1}]
    assert repo._has_resolvable_payload(idx, repo.arena.grafts[idx]) is True


def test_resolvable_payload_true_for_host_payload(tmp_path):
    repo = make_repo(tmp_path)
    idx = add_recovered_placeholder(repo)
    repo.arena.grafts[idx]["host_payload"] = {"payload_id": np.asarray([0])}
    assert repo._has_resolvable_payload(idx, repo.arena.grafts[idx]) is True


def test_resolvable_payload_true_when_disk_file_present(tmp_path):
    repo = make_repo(tmp_path)
    idx = add_recovered_placeholder(repo)
    os.makedirs(os.path.join(str(tmp_path), "nodes"), exist_ok=True)
    open(repo._payload_file_path(idx), "wb").close()
    assert repo._has_resolvable_payload(idx, repo.arena.grafts[idx]) is True


def test_resolvable_payload_false_for_bare_placeholder(tmp_path):
    repo = make_repo(tmp_path)
    idx = add_recovered_placeholder(repo)
    assert repo._has_resolvable_payload(idx, repo.arena.grafts[idx]) is False


# ============================================================================
# 2. Python-fallback selection path (_foldable / _active), the always-on
#    path in this environment (no native lib) — covers BOTH callers that
#    reach it: the deferred idle() path and the inline-backpressure path,
#    since both are the same _librarian_jobs()/_foldable() choke point.
# ============================================================================

def test_foldable_excludes_placeholder_turn(tmp_path):
    repo = make_repo(tmp_path)
    add_foldable_real_turns(repo, 2)
    placeholder_idx = add_recovered_placeholder(repo)

    turns = repo._foldable(("turn",))

    assert placeholder_idx not in turns
    assert len(turns) == 2


def test_librarian_jobs_never_selects_placeholder_as_fold_source(tmp_path):
    # deferred mode: turns accumulate without auto-folding, so the plan
    # can be inspected before anything consumes it (inline mode would
    # fold as turns land, leaving nothing above threshold to observe).
    repo = make_repo(tmp_path, librarian_mode="deferred")
    placeholder_idx = add_recovered_placeholder(repo)
    add_foldable_real_turns(repo, repo.TURNS_HIGH)  # trips the deferred threshold

    jobs = repo._librarian_jobs()

    assert jobs, "expected a digest job once real turns cross TURNS_HIGH"
    kind, idxs = jobs[0]
    assert kind == "digest"
    assert placeholder_idx not in idxs


def test_idle_folds_real_turns_and_never_crashes_on_placeholder(tmp_path):
    """End-to-end: this is the exact production scenario — a
    crash-recovered session (placeholder present) that then accumulates
    enough real traffic to trip a fold. Pre-fix, this raised TypeError
    inside FakeArena.consolidate() at grafts[i]["h"][li]; post-fix it
    must fold cleanly and never touch the placeholder."""
    repo = make_repo(tmp_path, librarian_mode="deferred")
    placeholder_idx = add_recovered_placeholder(repo)
    add_foldable_real_turns(repo, repo.TURNS_HIGH)

    done = repo.idle(max_jobs=1)

    assert done == 1
    assert repo.arena.grafts[placeholder_idx]["retired"] is False
    assert repo.arena.grafts[placeholder_idx]["h"] is None
    digest_nodes = [g for g in repo.arena.grafts if g.get("kind") == "digest"]
    assert len(digest_nodes) == 1
    assert placeholder_idx not in digest_nodes[0]["sources"]


def test_inline_backpressure_path_also_excludes_placeholder(tmp_path):
    """The inline last-resort path (_librarian()'s else-branch,
    deferred_backpressure=True) funnels through the SAME _foldable() as
    the deferred idle() path — this proves that shared choke point, not
    a second independent code path, so there is nothing left to drift.

    deferred mode's backpressure valve fires INSIDE add_turn() itself
    once foldable turns cross TURNS_HIGH*2 (no idle() call needed) — so
    driving enough turns through add_turn() directly exercises the real
    call site (_librarian()'s else-branch) rather than invoking
    _librarian_jobs(deferred_backpressure=True) by hand after the fact,
    which the valve may have already pre-empted."""
    repo = make_repo(tmp_path, librarian_mode="deferred")
    placeholder_idx = add_recovered_placeholder(repo)

    add_foldable_real_turns(repo, repo.TURNS_HIGH * 2)  # trips 2x backpressure

    digest_nodes = [g for g in repo.arena.grafts if g.get("kind") == "digest"]
    assert digest_nodes, "expected the backpressure valve to have folded"
    for d in digest_nodes:
        assert placeholder_idx not in d["sources"]
    assert repo.arena.grafts[placeholder_idx]["retired"] is False
    assert repo.arena.grafts[placeholder_idx]["h"] is None


# ============================================================================
# 3. Counters are not jammed by excluded placeholders (fold-exempt-
#    counting precedent, commit 816a0a0)
# ============================================================================

def test_placeholder_does_not_inflate_foldable_count_toward_threshold(tmp_path):
    """A placeholder must not count toward TURNS_HIGH at all — not as a
    foldable node (it would wrongly help trip the threshold early) and
    not as a permanent jam (excluded forever, the window can never
    advance past it). Below-threshold real-turn count must NOT fold even
    with placeholders padding the raw node list."""
    repo = make_repo(tmp_path)
    for _ in range(5):
        add_recovered_placeholder(repo)
    add_foldable_real_turns(repo, repo.TURNS_HIGH - 1)  # one short of tripping

    jobs = repo._librarian_jobs()

    assert jobs == [], (
        "placeholders must not count toward TURNS_HIGH — the real-turn "
        "count alone is one short of the fold threshold")


def test_repeated_idle_calls_do_not_jam_on_placeholder(tmp_path):
    """Fold-exempt-counting precedent (816a0a0): counting excluded nodes
    against the threshold can make idle() re-fire the same aborting/
    no-op plan forever. Here: repeated idle() calls with a permanently
    unresolvable placeholder present must keep making real progress
    (each real-turn batch above TURNS_HIGH folds) rather than getting
    stuck re-selecting or re-counting the placeholder."""
    repo = make_repo(tmp_path, librarian_mode="deferred")
    placeholder_idx = add_recovered_placeholder(repo)
    add_foldable_real_turns(repo, repo.TURNS_HIGH)

    first = repo.idle(max_jobs=1)
    assert first == 1

    add_foldable_real_turns(repo, repo.TURNS_HIGH)
    second = repo.idle(max_jobs=1)
    assert second == 1

    # the placeholder was never touched across either round
    assert repo.arena.grafts[placeholder_idx]["h"] is None
    assert repo.arena.grafts[placeholder_idx]["retired"] is False
    digest_nodes = [g for g in repo.arena.grafts if g.get("kind") == "digest"]
    assert len(digest_nodes) == 2
    for d in digest_nodes:
        assert placeholder_idx not in d["sources"]


# ============================================================================
# 4. Native-backed selection path (_native_foldable) — narrow, targeted
#    fake of just the surface that method touches (foldable_nodes() plus
#    hasattr-gated no-ops for the sync calls), proving the SAME guard
#    line closes the independent eligibility filter on that path too.
#    Not a full native-store simulation (that risks its own test-double
#    drift, M10's lesson) — just enough to exercise the new `continue`.
# ============================================================================

class FakeNativeStore:
    """Deliberately minimal: `add_node`, `set_route`, and `foldable_nodes`
    are implemented (set_route is the one unconditional, non-hasattr-
    gated call site in _native_set_route's fallback branch). Every other
    native_store.* call site in graft_repository.py is hasattr-gated
    (set_tensor, set_metadata, set_active, configure_arena, clear_route,
    add_structured_node, set_route_key_list, set_route_keys), so their
    absence here is a silent no-op, not a missing feature."""

    def __init__(self):
        self._next_id = 0

    def add_node(self, text, blob, ntok=0):
        node_id = self._next_id
        self._next_id += 1
        return node_id

    def set_route(self, node_id, route_key, lexical_keys):
        pass

    def foldable_nodes(self, kind, excluded_native):
        # Native store has no notion of payload resolvability — it would
        # happily hand back every "turn"-kind node it knows about,
        # INCLUDING the placeholder's native id. That's the point: prove
        # graft_repository.py's post-processing filter is what excludes
        # it, not the native store.
        return list(range(self._next_id))


def test_native_foldable_excludes_placeholder_even_when_store_returns_it(
        tmp_path):
    repo = make_repo(tmp_path)
    repo.native_store = FakeNativeStore()
    repo.arena.native_store = repo.native_store

    add_foldable_real_turns(repo, 2)
    placeholder_idx = add_recovered_placeholder(repo)

    turns = repo._foldable(("turn",))

    assert placeholder_idx not in turns
    assert len(turns) == 2
