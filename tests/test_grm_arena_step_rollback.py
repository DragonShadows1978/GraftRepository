from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import numpy as np

import core.graft_arena as graft_arena
from core.graft_arena import ArenaCache


class _FakeTC:
    @staticmethod
    def no_grad():
        return nullcontext()


class _FakeLogits:
    @staticmethod
    def numpy():
        return np.array([[[0.0, 1.0]]], dtype=np.float32)


class _ConsumingBackend:
    """Mirror GPT-OSS/Gemma4's destructive outer cache-list contract."""

    def __init__(self):
        self.layers = [SimpleNamespace(self_attn=SimpleNamespace())]
        self.observed = []
        self.calls = 0

    def __call__(self, _ids, *, kv_caches, position_offset,
                 last_token_only):
        del position_offset, last_token_only
        self.observed.append(None if kv_caches is None else kv_caches[0])
        if kv_caches is not None:
            kv_caches[0] = None
        self.calls += 1
        return _FakeLogits(), [(f"successor-k-{self.calls}",
                               f"successor-v-{self.calls}")]


def _arena_with_consuming_backend(monkeypatch, ranking, grounded_attempt):
    monkeypatch.setattr(graft_arena, "tc", _FakeTC)
    backend = _ConsumingBackend()
    arena = ArenaCache.__new__(ArenaCache)
    baseline = (object(), object())
    arena.caches = [baseline]
    arena.pos = 17
    arena.live_segs = []
    arena.cur_mounts = []
    arena.cur_mount_n = 0
    arena.grafts = [
        {"kind": "digest", "sources": [1], "ntok": 1, "text": "parent"},
        {"kind": "turn", "ntok": 1, "text": "descent child"},
        {"kind": "turn", "ntok": 1, "text": "alternate rank"},
    ]
    arena.ephemeral = False
    arena.recency_mounts = 0
    arena.live_shift = 9
    arena.m = backend
    arena.topk = 1
    arena.width = 8
    arena.stop_sequences = ()
    arena.revision_resolution = False
    arena.native_store = None
    arena.last_route_receipt = None
    arena._s4_turn = 0
    arena.route = lambda _text, exclude, limit: list(ranking)
    arena._rare_tokens = lambda _text: set()
    arena._commit_s4_attempt = lambda *_args, **_kwargs: None

    attempt_starts = []
    attempt_picks = []

    def attempt(self, _user_text, picks, _ngen, _deposit, _stops):
        attempt_starts.append(
            None if self.caches is None else self.caches[0])
        attempt_picks.append(list(picks))
        ArenaCache._forward(self, [1])
        return f"attempt-{len(attempt_starts)}", {}

    arena._attempt = MethodType(attempt, arena)
    arena._grounding_attribution = lambda *_args: (
        len(attempt_starts) == grounded_attempt, set())
    return arena, backend, baseline, attempt_starts, attempt_picks


def test_step_nonclean_descent_retry_restores_consumed_cache_list(monkeypatch):
    """Production step() must retry from the intact pre-attempt cache list."""
    arena, backend, baseline, starts, picks = _arena_with_consuming_backend(
        monkeypatch, ranking=[0], grounded_attempt=2)

    answer, info = arena.step(
        "retry through the digest", ngen=1, deposit=False, max_trips=1)

    assert answer == "attempt-2"
    assert info["trip"] == 1
    assert picks == [[0], [1]]
    assert starts == [baseline, baseline]
    assert backend.observed == [baseline, baseline]


def test_step_keeps_snapshot_pristine_for_later_nonclean_retry(monkeypatch):
    """Every non-clean retry must get a fresh outer list from the snapshot."""
    arena, backend, baseline, starts, picks = _arena_with_consuming_backend(
        monkeypatch, ranking=[0, 2], grounded_attempt=4)

    answer, info = arena.step(
        "walk the full retry ladder", ngen=1, deposit=False, max_trips=3)

    assert answer == "attempt-4"
    assert info["trip"] == 3
    assert picks == [[0], [1], [1], [2]]
    assert starts == [baseline, baseline, None, baseline]
    assert backend.observed == [baseline, baseline, None, baseline]
