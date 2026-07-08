"""P3 unit tests: packed on-disk graft-quantization format.

Covers:
  - core.graft_quant round-trip (pack/unpack bit-identical to the P1
    reference transform, all packable bit depths) incl. fail-closed
    behavior on unknown format_version/storage_bits.
  - format-1 hook (core.graft_arena.GQAArenaCache.pack_node/unpack_node):
    opt-in storage_bits, default path unaffected, packed path dequantizes
    correctly, save/load cycle through a real npz file with storage_bits
    configured.
  - default-path-unchanged: a plain fp16 payload is unaffected by any of
    this — pack_node with storage_bits unset returns exactly the same
    {"k","v"} dict shape as before P3, and a plain shard is never
    misdetected as packed.

These are CPU-only, no GPU / no live model required for the core.graft_quant
half; the GQAArenaCache half uses tensor_cuda (device tensors are part of
pack_node/unpack_node's contract) but no model weights or CUDA kernels
beyond tensor construction/dtype-cast, matching the house convention (see
tests/test_grm_router_baseline.py for the same "import core/scripts
directly, assert in pytest" pattern).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.graft_quant import (  # noqa: E402
    FORMAT_VERSION,
    GraftQuantFormatError,
    is_packed_payload,
    load_packed_npz,
    pack_kv_arrays,
    quantize_dequantize_symmetric_group32,
    quantize_symmetric_group32,
    save_packed_npz,
    unpack_kv_arrays,
)


def _rand_kv(seed=0, shape=(1, 8, 64, 64)):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) * 3).astype(np.float16)


# --------------------------------------------------------------------------
# core.graft_quant: round-trip vs the P1 reference, all packable bit depths
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [8, 6, 4, 3, 2])
def test_pack_unpack_bit_identical_to_reference_transform(bits):
    k = _rand_kv(seed=1)
    v = _rand_kv(seed=2)

    ref_k = quantize_dequantize_symmetric_group32(k, bits, 32)
    ref_v = quantize_dequantize_symmetric_group32(v, bits, 32)

    packed = pack_kv_arrays({"k": k, "v": v}, bits, 32)
    unpacked = unpack_kv_arrays(packed, ["k", "v"])

    np.testing.assert_array_equal(ref_k, unpacked["k"])
    np.testing.assert_array_equal(ref_v, unpacked["v"])


def test_bits_16_is_identity_and_not_packable():
    k = _rand_kv()
    identity = quantize_dequantize_symmetric_group32(k, 16, 32)
    np.testing.assert_array_equal(k, identity)

    with pytest.raises(GraftQuantFormatError):
        pack_kv_arrays({"k": k}, 16)


def test_pack_unpack_npz_roundtrip_on_disk(tmp_path):
    k = _rand_kv(seed=3)
    v = _rand_kv(seed=4)
    path = tmp_path / "shard.npz"
    save_packed_npz(path, {"k": k, "v": v}, bits=8, group_size=32, compressed=False)

    with np.load(path) as z:
        assert is_packed_payload(z)

    loaded = load_packed_npz(path, ["k", "v"])
    ref_k = quantize_dequantize_symmetric_group32(k, 8, 32)
    ref_v = quantize_dequantize_symmetric_group32(v, 8, 32)
    np.testing.assert_array_equal(ref_k, loaded["k"])
    np.testing.assert_array_equal(ref_v, loaded["v"])


def test_plain_fp16_payload_is_not_detected_as_packed(tmp_path):
    k = _rand_kv(seed=5)
    path = tmp_path / "plain.npz"
    np.savez(path, k=k)
    with np.load(path) as z:
        assert not is_packed_payload(z)


# --------------------------------------------------------------------------
# Fail-closed: unknown format_version / storage_bits never silently misread
# --------------------------------------------------------------------------

def test_unpack_rejects_unknown_format_version():
    k = _rand_kv()
    payload = pack_kv_arrays({"k": k}, 8, 32)
    payload["format_version"] = np.asarray(FORMAT_VERSION + 999, dtype=np.int64)
    with pytest.raises(GraftQuantFormatError):
        unpack_kv_arrays(payload, ["k"])


def test_unpack_rejects_missing_format_version():
    with pytest.raises(GraftQuantFormatError):
        unpack_kv_arrays({"k_codes": np.zeros((1,), dtype=np.int8)}, ["k"])


def test_unpack_rejects_unsupported_storage_bits():
    k = _rand_kv()
    payload = pack_kv_arrays({"k": k}, 8, 32)
    payload["storage_bits"] = np.asarray(5, dtype=np.int64)  # not in SUPPORTED_BITS
    with pytest.raises(GraftQuantFormatError):
        unpack_kv_arrays(payload, ["k"])


def test_unpack_rejects_storage_bits_16_in_packed_payload():
    k = _rand_kv()
    payload = pack_kv_arrays({"k": k}, 8, 32)
    payload["storage_bits"] = np.asarray(16, dtype=np.int64)
    with pytest.raises(GraftQuantFormatError):
        unpack_kv_arrays(payload, ["k"])


def test_quantize_symmetric_group32_rejects_bits_16():
    k = _rand_kv()
    with pytest.raises(GraftQuantFormatError):
        quantize_symmetric_group32(k, 16, 32)


def test_pack_kv_arrays_rejects_non_fp16_input():
    k = _rand_kv().astype(np.float32)
    with pytest.raises(ValueError):
        pack_kv_arrays({"k": k}, 8, 32)


def test_group_size_must_divide_last_axis():
    k = _rand_kv(shape=(1, 8, 64, 48))  # 48 % 32 != 0
    with pytest.raises(ValueError):
        pack_kv_arrays({"k": k}, 8, 32)


# --------------------------------------------------------------------------
# format-1 hook: core.graft_arena.GQAArenaCache.pack_node/unpack_node
# --------------------------------------------------------------------------

class _FakeModel:
    """Minimal stand-in: GQAArenaCache.unpack_node only reads len(self.m.layers)."""
    def __init__(self, num_layers):
        self.layers = [object() for _ in range(num_layers)]


def _make_gqa_arena(storage_bits=None):
    """Construct a GQAArenaCache without running its (heavy, model-requiring)
    __init__ — pack_node/unpack_node only touch self.storage_bits and
    self.m.layers, both set here directly. Mirrors the house pattern of
    testing dialect methods in isolation (cf. test_grm_router_baseline.py
    importing core.grm_cuda_router functions directly)."""
    from core.graft_arena import GQAArenaCache
    arena = GQAArenaCache.__new__(GQAArenaCache)
    arena.storage_bits = storage_bits
    arena.m = _FakeModel(num_layers=3)
    return arena


def _fake_node_h(num_layers=3, num_kv_heads=8, seq=16, head_dim=64, seed=0):
    import tensor_cuda as tc
    rng = np.random.default_rng(seed)
    h = []
    for _ in range(num_layers):
        k_np = (rng.standard_normal((1, num_kv_heads, seq, head_dim)) * 2).astype(np.float16)
        v_np = (rng.standard_normal((1, num_kv_heads, seq, head_dim)) * 2).astype(np.float16)
        h.append({"k": tc.tensor(k_np), "v": tc.tensor(v_np)})
    return h


def test_pack_node_default_path_unchanged():
    arena = _make_gqa_arena(storage_bits=None)
    h = _fake_node_h()
    payload = arena.pack_node(h)
    assert set(payload.keys()) == {"k", "v"}
    assert payload["k"].dtype == np.float16
    assert not is_packed_payload(payload)


def test_pack_node_opt_in_produces_packed_payload():
    arena = _make_gqa_arena(storage_bits=8)
    h = _fake_node_h()
    payload = arena.pack_node(h)
    assert is_packed_payload(payload)
    assert int(np.asarray(payload["storage_bits"])) == 8


def test_unpack_node_roundtrips_default_and_packed():
    h = _fake_node_h(seed=7)

    arena_plain = _make_gqa_arena(storage_bits=None)
    plain_payload = arena_plain.pack_node(h)
    restored_plain = arena_plain.unpack_node(plain_payload)
    for orig, rest in zip(h, restored_plain):
        np.testing.assert_array_equal(
            orig["k"].float().numpy(), rest["k"].float().numpy())
        np.testing.assert_array_equal(
            orig["v"].float().numpy(), rest["v"].float().numpy())

    arena_packed = _make_gqa_arena(storage_bits=8)
    packed_payload = arena_packed.pack_node(h)
    restored_packed = arena_packed.unpack_node(packed_payload)
    # dequantized (lossy at 8 bits) but must match the reference transform's
    # dequantized output bit-for-bit, same law as the round-trip gate above.
    for orig, rest in zip(h, restored_packed):
        k_np = np.ascontiguousarray(orig["k"].float().numpy()[0]).astype(np.float16)
        ref_k = quantize_dequantize_symmetric_group32(k_np[None], 8, 32)
        np.testing.assert_array_equal(ref_k[0], rest["k"].float().numpy()[0])


def test_unpack_node_fails_closed_on_unknown_format_version():
    arena = _make_gqa_arena(storage_bits=8)
    h = _fake_node_h(seed=9)
    payload = arena.pack_node(h)
    payload["format_version"] = np.asarray(FORMAT_VERSION + 42, dtype=np.int64)
    with pytest.raises(GraftQuantFormatError):
        arena.unpack_node(payload)


def test_pack_node_save_load_npz_cycle_with_storage_bits(tmp_path):
    """format-1 node save/load cycle: pack_node -> np.savez_compressed (the
    same call GraftRepository.flush_now makes) -> np.load -> unpack_node,
    with storage_bits configured end-to-end through an actual npz file."""
    arena = _make_gqa_arena(storage_bits=6)
    h = _fake_node_h(seed=11)
    payload = {
        key: np.ascontiguousarray(value)
        for key, value in arena.pack_node(h).items()
    }
    assert payload["format_version"].shape == (1,)

    node_path = tmp_path / "0000.npz"
    np.savez_compressed(node_path, **payload)

    with np.load(node_path) as z:
        loaded = {key: z[key] for key in z.files}
    restored = arena.unpack_node(loaded)

    for orig, rest in zip(h, restored):
        k_np = np.ascontiguousarray(orig["k"].float().numpy()[0]).astype(np.float16)
        ref_k = quantize_dequantize_symmetric_group32(k_np[None], 6, 32)
        np.testing.assert_array_equal(ref_k[0], rest["k"].float().numpy()[0])


def test_storage_bits_env_var_default(monkeypatch):
    """GQAArenaCache.__init__ honors GRM_GRAFT_STORAGE_BITS when the
    constructor kwarg is not given (matches the GRM_GQA_CUDA_ROUTE env-var
    convention already in this class). Exercised against a stub model that
    is just enough for the pre-super().__init__ code (cfg lookup + an empty
    layers list) to run to completion without needing real weights."""
    from core.graft_arena import GQAArenaCache

    class _Cfg:
        num_kv_heads = 8
        head_dim = 64

    class _StubModel:
        config = _Cfg()
        layers = []

    monkeypatch.setenv("GRM_GRAFT_STORAGE_BITS", "6")
    stub_self = GQAArenaCache.__new__(GQAArenaCache)
    # super().__init__ would need a real model/encode/decode; call the raw
    # function up to where it sets self.storage_bits by catching the
    # (expected) failure once it reaches super().__init__ with a stub model.
    with pytest.raises(AttributeError):
        GQAArenaCache.__init__(stub_self, _StubModel(), lambda t: [], lambda i: "", "/tmp/unused")
    assert stub_self.storage_bits == 6


def test_storage_bits_rejects_invalid_value_via_constructor_kwarg():
    """storage_bits=16 (or any value outside the packable set) must raise
    ValueError before any model/super().__init__ work happens."""
    from core.graft_arena import GQAArenaCache

    class _Cfg:
        num_kv_heads = 8
        head_dim = 64

    class _StubModel:
        config = _Cfg()
        layers = []

    stub_self = GQAArenaCache.__new__(GQAArenaCache)
    with pytest.raises(ValueError):
        GQAArenaCache.__init__(stub_self, _StubModel(), storage_bits=16)

    stub_self2 = GQAArenaCache.__new__(GQAArenaCache)
    with pytest.raises(ValueError):
        GQAArenaCache.__init__(stub_self2, _StubModel(), storage_bits=5)
