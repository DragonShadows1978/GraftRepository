"""GRM arena cache-surgery gate for TensorCUDA splice/evict helpers.

Run explicitly because it allocates CUDA tensors:

  PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
  python3 tests/grm_arena_splice_gate.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensor_cuda as tc
from core.graft_arena import ArenaCache


def _rope_tables(t, d):
    inv = 1.0 / (1e4 ** (np.arange(0, d, 2, np.float32) / d))
    ang = np.arange(t, dtype=np.float32)[:, None] * inv[None, :]
    emb = np.concatenate([ang, ang], -1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def main():
    arena = ArenaCache.__new__(ArenaCache)
    arena.n_sink = 2
    arena.cur_mount_n = 3
    arena.width = 4
    arena.ROPE_KEYS = ("kpe",)
    arena.ROPE_PAIR_SWAP = False
    cs_np, sn_np = _rope_tables(16, 64)
    arena.m = type("DummyModel", (), {
        "rope_cos": tc.tensor(cs_np),
        "rope_sin": tc.tensor(sn_np),
    })()

    old_np = np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3)
    ins_np = (np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
              + 1000.0)
    old = tc.tensor(old_np)
    ins = tc.tensor(ins_np)
    out = arena._splice_cache_tensor(old, ins, 1, arena.n_sink,
                                     arena.n_sink + arena.cur_mount_n)
    exp = np.concatenate([old_np[:, :2], ins_np, old_np[:, 5:]], axis=1)
    assert np.array_equal(out.numpy(), exp)

    ev = arena._evict_cache_tensor(old, 1, arena.n_sink + arena.cur_mount_n, 2)
    ev_exp = np.concatenate([old_np[:, :5], old_np[:, 7:]], axis=1)
    assert np.array_equal(ev.numpy(), ev_exp)

    kpe_np = np.arange(1 * 1 * 8 * 2, dtype=np.float32).reshape(1, 1, 8, 2)
    kpe_ins_np = np.full((1, 1, 2, 2), 22.0, dtype=np.float32)
    kpe = tc.tensor(kpe_np)
    kpe_ins = tc.tensor(kpe_ins_np)
    kpe_out = arena._splice_cache_tensor(kpe, kpe_ins, 2, 2, 5)
    kpe_exp = np.concatenate([kpe_np[:, :, :2], kpe_ins_np,
                              kpe_np[:, :, 5:]], axis=2)
    assert np.array_equal(kpe_out.numpy(), kpe_exp)

    c_np = np.arange(1 * 8 * 5, dtype=np.float32).reshape(1, 8, 5)
    kpe_np = np.random.default_rng(2).standard_normal(
        (1, 1, 8, 64)).astype(np.float32)
    with tc.no_grad():
        c = tc.tensor(c_np)
        kpe_plain = tc.tensor(kpe_np)
        kpe_rot = tc.rope_apply(kpe_plain, arena.m.rope_cos,
                                arena.m.rope_sin, 0)
        seg = arena._export_cache_payload((c, kpe_rot), 4, 4)
    assert set(seg) == {"c", "kpe"}
    assert np.array_equal(seg["c"].numpy(), c_np[:, 4:8])
    np.testing.assert_allclose(seg["kpe"].numpy(), kpe_np[:, :, 4:8],
                               rtol=1e-6, atol=1e-6)

    c2_np = c_np + 500.0
    kpe2_np = kpe_np * 0.25
    with tc.no_grad():
        c2 = tc.tensor(c2_np)
        kpe2_plain = tc.tensor(kpe2_np)
        kpe2_rot = tc.rope_apply(kpe2_plain, arena.m.rope_cos,
                                 arena.m.rope_sin, 0)
        arena.caches = [(c, kpe_rot), (c2, kpe2_rot)]
        dev = arena._export_cache_payloads(3, 5)
    assert len(dev) == 2
    assert np.array_equal(dev[0]["c"].numpy(), c_np[:, 5:8])
    assert np.array_equal(dev[1]["c"].numpy(), c2_np[:, 5:8])
    np.testing.assert_allclose(dev[0]["kpe"].numpy(), kpe_np[:, :, 5:8],
                               rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dev[1]["kpe"].numpy(), kpe2_np[:, :, 5:8],
                               rtol=1e-6, atol=1e-6)

    ins0_np = np.full((1, 2, 5), 31.0, dtype=np.float32)
    ins1_np = np.full((1, 2, 5), 37.0, dtype=np.float32)
    pins0_np = np.random.default_rng(5).standard_normal(
        (1, 1, 2, 64)).astype(np.float32)
    pins1_np = pins0_np * -0.5
    with tc.no_grad():
        ins0, ins1 = tc.tensor(ins0_np), tc.tensor(ins1_np)
        pins0, pins1 = tc.tensor(pins0_np), tc.tensor(pins1_np)
        arena.caches = [(c, kpe_rot), (c2, kpe2_rot)]
        arena.grafts = [{"h": [{"c": ins0, "kpe": pins0},
                               {"c": ins1, "kpe": pins1}]}]
        swapped, n_new = arena._swap_cache_payloads([0], 5)
        pin0_rot = tc.rope_apply(pins0, arena.m.rope_cos,
                                 arena.m.rope_sin, 2)
        pin1_rot = tc.rope_apply(pins1, arena.m.rope_cos,
                                 arena.m.rope_sin, 2)
    assert n_new == 2
    assert np.array_equal(swapped[0][0].numpy(),
                          np.concatenate([c_np[:, :2], ins0_np,
                                          c_np[:, 5:]], axis=1))
    assert np.array_equal(swapped[1][0].numpy(),
                          np.concatenate([c2_np[:, :2], ins1_np,
                                          c2_np[:, 5:]], axis=1))
    np.testing.assert_allclose(
        swapped[0][1].numpy(),
        np.concatenate([kpe_rot.numpy()[:, :, :2], pin0_rot.numpy(),
                        kpe_rot.numpy()[:, :, 5:]], axis=2),
        rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        swapped[1][1].numpy(),
        np.concatenate([kpe2_rot.numpy()[:, :, :2], pin1_rot.numpy(),
                        kpe2_rot.numpy()[:, :, 5:]], axis=2),
        rtol=1e-6, atol=1e-6)

    with tc.no_grad():
        arena.caches = [(c, kpe_rot), (c2, kpe2_rot)]
        evicted = arena._evict_cache_payloads(5, 2)
    assert np.array_equal(evicted[0][0].numpy(),
                          np.concatenate([c_np[:, :5],
                                          c_np[:, 7:]], axis=1))
    assert np.array_equal(evicted[1][0].numpy(),
                          np.concatenate([c2_np[:, :5],
                                          c2_np[:, 7:]], axis=1))
    assert np.array_equal(evicted[0][1].numpy(),
                          np.concatenate([kpe_rot.numpy()[:, :, :5],
                                          kpe_rot.numpy()[:, :, 7:]], axis=2))
    assert np.array_equal(evicted[1][1].numpy(),
                          np.concatenate([kpe2_rot.numpy()[:, :, :5],
                                          kpe2_rot.numpy()[:, :, 7:]], axis=2))
    arena.width = 1
    try:
        arena._swap_cache_payloads([0], 5)
        raise AssertionError("arena transaction width guard missing")
    except RuntimeError:
        pass
    print("GRM ARENA SPLICE GATE: PASS")


if __name__ == "__main__":
    main()
