from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

import core.graft_arena as graft_arena
from core.graft_arena import GQAArenaCache
from scripts.grm_adm1_gpu import restore_arena, snapshot_arena


class _FakeTC:
    @staticmethod
    def no_grad():
        return nullcontext()

    @staticmethod
    def evict_rows(tensor, dim, head_tokens, drop_tokens):
        size = tensor.shape[dim]
        if (head_tokens < 0 or drop_tokens < 0 or head_tokens > size
                or drop_tokens > size - head_tokens):
            raise RuntimeError("evict_rows: drop exceeds live tail")
        head = [slice(None)] * tensor.ndim
        tail = [slice(None)] * tensor.ndim
        head[dim] = slice(0, head_tokens)
        tail[dim] = slice(head_tokens + drop_tokens, size)
        return np.concatenate((tensor[tuple(head)], tensor[tuple(tail)]), axis=dim)


def _consuming_gpt_oss_forward(caches, current_tokens):
    """Mirror GptOss20B_TC's destructive outer-list cache contract."""
    prior = caches[0]
    sequence = (
        current_tokens
        if prior is None
        else prior[0].shape[2] + current_tokens
    )
    caches[0] = None
    shape = (1, 8, sequence, 64)
    return [(np.zeros(shape, np.float16), np.zeros(shape, np.float16))]


def test_adm1_k1_snapshot_survives_consuming_cache_list(monkeypatch):
    """A repeated k=1 arm must not cold-start with stale mount occupancy."""
    arena = GQAArenaCache.__new__(GQAArenaCache)
    baseline_shape = (1, 8, 252, 64)
    arena.caches = [(
        np.zeros(baseline_shape, np.float16),
        np.zeros(baseline_shape, np.float16),
    )]
    arena.pos = 0
    arena.live_segs = [(3, 61), (4, 86)]
    arena.live_turns = 2
    arena.cur_mounts = [0]
    arena.cur_mount_n = 86
    arena.n_sink = 19
    arena.grafts = [{"ntok": 86}]
    arena.m = SimpleNamespace(layers=[])
    monkeypatch.setattr(graft_arena, "tc", _FakeTC)

    observed_shapes = []

    def run_counterfactual_arm():
        before = snapshot_arena(arena)
        try:
            # k=1 keeps the already-mounted node, so swap() is a seat no-op.
            arena.swap([0])
            arena.caches = _consuming_gpt_oss_forward(arena.caches, 47)
            observed_shapes.append(arena.caches[0][0].shape)
            arena.live_segs.append((None, 47))
            assert arena.evict() == 61
        finally:
            restore_arena(arena, before)

    run_counterfactual_arm()
    run_counterfactual_arm()

    # The regression was a cold 47-token cache paired with the still-live
    # token boundary head=19+86=105 and drop=61.  Both arms must instead see
    # the preserved 252-token baseline before adding their 47 probe tokens.
    assert observed_shapes == [(1, 8, 299, 64), (1, 8, 299, 64)]
    assert arena.caches[0][0].shape == baseline_shape
