import numpy as np
import pytest

from core.graft_arena import GQAArenaCache
from core import grm_cuda_router
from scripts.grm_gqa_exact_ragged_cuda_gate import _ragged_scores


def _arena(rows, *, node_ids=None):
    arena = GQAArenaCache.__new__(GQAArenaCache)
    arena.native_store = None
    arena.last_route_backend = "python"
    if node_ids is None:
        node_ids = range(101, 101 + len(rows))
    arena.grafts = [
        {
            "native_node_id": int(node_id),
            "cent": row,
            "rare": set(),
            "text": f"node-{idx}",
            "kind": "doc",
            "retired": False,
        }
        for idx, (node_id, row) in enumerate(zip(node_ids, rows))
    ]
    return arena


def test_mixed_length_bank_preserves_source_bits_and_uses_zero_only_tails():
    row0 = np.asarray(
        [[[1.0, -0.0, 3.25], [-7.5, 2.0, -4.0]]], dtype=np.float32)
    row1 = np.asarray(
        [[[9.0, -3.0, 0.5], [4.0, 5.0, 6.0], [-2.0, 8.0, 1.5],
          [0.25, -0.5, 0.75]]], dtype=np.float32)

    bank, receipt = GQAArenaCache._cuda_build_dense_route_bank([row0, row1])

    assert bank.shape == (2, 1, 4, 3)
    assert bank.flags.c_contiguous
    assert bank.dtype == np.float32
    np.testing.assert_array_equal(
        bank[0, :, :row0.shape[1], :].view(np.uint32), row0.view(np.uint32))
    np.testing.assert_array_equal(
        bank[1, :, :row1.shape[1], :].view(np.uint32), row1.view(np.uint32))
    np.testing.assert_array_equal(
        bank[0, :, row0.shape[1]:, :], np.zeros((1, 2, 3), np.float32))

    assert receipt == {
        "schema": "grm_gqa_cuda_route_bank_layout_v1",
        "layout": "zero_padded",
        "node_count": 2,
        "kv_heads": 1,
        "head_dim": 3,
        "row_token_counts": (2, 4),
        "max_tokens": 4,
        "raw_value_count": 18,
        "padded_value_count": 24,
        "raw_bytes": 72,
        "padded_bytes": 96,
        "padding_ratio": 4 / 3,
    }


def test_same_shape_bank_retains_dense_layout_receipt():
    rows = [
        np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        np.arange(12, 24, dtype=np.float32).reshape(2, 2, 3),
    ]
    bank, receipt = GQAArenaCache._cuda_build_dense_route_bank(rows)

    np.testing.assert_array_equal(bank, np.stack(rows))
    assert receipt["layout"] == "dense"
    assert receipt["row_token_counts"] == (2, 2)
    assert receipt["padding_ratio"] == 1.0


def test_randomized_raw_and_zero_padded_scores_and_rankings_are_exact():
    rng = np.random.default_rng(20260712)
    arena = GQAArenaCache.__new__(GQAArenaCache)

    for _ in range(40):
        lengths = rng.integers(1, 13, size=9).tolist()
        rows = [
            rng.standard_normal((2, tokens, 8), dtype=np.float32)
            for tokens in lengths
        ]
        query = rng.standard_normal((4, 5, 8), dtype=np.float32)
        bank, _receipt = arena._cuda_build_dense_route_bank(rows)

        raw_scores = [arena._key_score(query, row) for row in rows]
        padded_scores = [arena._key_score(query, bank[idx])
                         for idx in range(len(rows))]
        assert padded_scores == raw_scores

        raw_order = sorted(range(len(rows)), key=lambda idx: -raw_scores[idx])
        padded_order = sorted(
            range(len(rows)), key=lambda idx: -padded_scores[idx])
        assert padded_order == raw_order


def test_gate_ragged_reference_uses_production_fp32_accumulation_order():
    # Keep this reference routed through _key_score. Reassociating the GQA
    # heads into a grouped einsum is algebraically valid but can reverse a
    # sub-ulp near tie and create a false CPU/CUDA parity failure.
    rng = np.random.default_rng(8173)
    rows = [
        rng.standard_normal((2, tokens, 8), dtype=np.float32)
        for tokens in (3, 7, 11)
    ]
    query = rng.standard_normal((4, 2, 8), dtype=np.float32)
    arena = GQAArenaCache.__new__(GQAArenaCache)

    expected = np.asarray([arena._key_score(query, row) for row in rows])
    np.testing.assert_array_equal(_ragged_scores(query, rows), expected)


@pytest.mark.parametrize(
    "rows, match",
    [
        ([], "at least one"),
        ([np.empty((1, 0, 2), np.float32)], "non-empty rank-3"),
        ([np.zeros((1, 2), np.float32)], "non-empty rank-3"),
        ([np.zeros((1, 1, 2), np.float32),
          np.zeros((2, 1, 2), np.float32)], "share KV-head"),
        ([np.asarray([[[np.nan, 0.0]]], np.float32)], "must be finite"),
    ],
)
def test_dense_bank_builder_rejects_inexact_or_malformed_inputs(rows, match):
    with pytest.raises(ValueError, match=match):
        GQAArenaCache._cuda_build_dense_route_bank(rows)


def test_signature_accepts_only_mixed_length_compatible_leaf_rows():
    row0 = np.ones((2, 1, 4), np.float32)
    row1 = np.ones((2, 5, 4), np.float32)
    arena = _arena([row0, row1])

    signature = arena._cuda_route_bank_signature()
    assert signature is not None
    assert signature[0].tolist() == [101, 102]
    assert [row.shape for row in signature[2]] == [(2, 1, 4), (2, 5, 4)]

    arena.grafts[1]["cent"] = np.ones((1, 5, 4), np.float32)
    assert arena._cuda_route_bank_signature() is None

    arena.grafts[1]["cent"] = row1
    arena.grafts[1]["child_cents"] = [row0]
    assert arena._cuda_route_bank_signature() is None

    del arena.grafts[1]["child_cents"]
    arena.grafts[1]["native_node_id"] = None
    assert arena._cuda_route_bank_signature() is None

    arena.grafts[1]["native_node_id"] = 102
    arena.grafts[1]["cent"] = np.full((2, 5, 4), np.inf, np.float32)
    assert arena._cuda_route_bank_signature() is None


def test_mixed_length_bank_is_epoch_cached_and_rebuilt_after_shape_change():
    rows = [
        np.ones((1, 1, 2), np.float32),
        np.ones((1, 3, 2), np.float32),
    ]
    arena = _arena(rows)
    arena._cuda_gqa_epoch = 0

    first = arena._cuda_route_bank_inputs()
    assert first is arena._cuda_route_bank_inputs()
    assert first[0].shape == (2, 1, 3, 2)
    assert arena._cuda_gqa_padding_receipt["row_token_counts"] == (1, 3)

    arena.grafts[0]["cent"] = np.ones((1, 5, 2), np.float32)
    arena._bump_cuda_gqa_epoch()
    second = arena._cuda_route_bank_inputs()

    assert second is not first
    assert second[0].shape == (2, 1, 5, 2)
    assert arena._cuda_gqa_padding_receipt["row_token_counts"] == (5, 3)


def test_mixed_length_route_attaches_padded_bank_and_uses_cuda(monkeypatch):
    class ExactCudaStore:
        def __init__(self):
            self._cuda_gqa_bank = None
            self._cuda_gqa_bank_signature = None
            self.node_ids = None
            self.configure_calls = 0

        def configure_cuda_gqa_route_bank(self, route_bank, node_ids):
            self._cuda_gqa_bank = np.asarray(route_bank, np.float32).copy()
            self.node_ids = np.asarray(node_ids, np.uint64).copy()
            self.configure_calls += 1

        def route_gqa_cuda(self, query, *, topk=3, **_kwargs):
            query = np.asarray(query, np.float32)
            repeated = np.repeat(
                self._cuda_gqa_bank,
                query.shape[0] // self._cuda_gqa_bank.shape[1],
                axis=1,
            )
            scores = np.abs(np.einsum(
                "hqd,nhkd->nhqk", query, repeated
            ) / np.sqrt(query.shape[2])).max(axis=(2, 3)).mean(axis=1)
            order = sorted(
                range(len(scores)), key=lambda idx: (-scores[idx], idx))
            return [int(self.node_ids[idx]) for idx in order[:int(topk)]]

        def route_gqa(self, *_args, **_kwargs):
            raise AssertionError("eligible mixed-length route used CPU")

    monkeypatch.setenv("GRM_GQA_CUDA_ROUTE", "1")
    monkeypatch.setenv("GRM_ROUTE_QUERY_LEX", "0")
    rows = [
        np.asarray([[[0.05, 0.0]]], np.float32),
        np.asarray([[[0.1, 0.0], [2.0, 0.0], [0.3, 0.0]]], np.float32),
        np.asarray([[[0.4, 0.0], [0.2, 0.0]]], np.float32),
    ]
    query = np.asarray([[[1.0, 0.0]]], np.float32)
    arena = _arena(rows)
    arena.native_store = ExactCudaStore()
    arena._probe_key = lambda _text: query

    assert arena.route("plain probe", exclude=set(), limit=3) == [1, 2, 0]
    assert arena.last_route_backend == "cuda"
    assert arena.native_store.configure_calls == 1
    assert arena.native_store._cuda_gqa_bank.shape == (3, 1, 3, 2)
    assert arena._cuda_gqa_padding_receipt["layout"] == "zero_padded"

    assert arena.route("plain probe", exclude=set(), limit=2) == [1, 2]
    assert arena.native_store.configure_calls == 1


def test_sidecar_accepts_epoch_signature_without_rehashing(monkeypatch):
    class FakeDeviceArena:
        setup_wall_ms = 1.25

        def close(self):
            pass

    class FakeProbe:
        def __init__(self):
            self.keys = None

        def create_arena(self, keys):
            self.keys = np.asarray(keys).copy()
            return FakeDeviceArena()

    def forbidden_hash(*_args, **_kwargs):
        raise AssertionError("trusted arena signature must skip content hash")

    monkeypatch.setattr(
        grm_cuda_router, "gqa_route_bank_signature", forbidden_hash)
    sidecar = grm_cuda_router.CudaGQARouteSidecar.__new__(
        grm_cuda_router.CudaGQARouteSidecar)
    sidecar._build_temp = None
    sidecar._probe = FakeProbe()
    signature = ((101, (1, 2, 3), "<f4", 99),)
    keys = np.arange(12, dtype=np.float32).reshape(2, 1, 2, 3)
    node_ids = np.asarray([101, 102], np.uint64)

    bank = sidecar.create_bank(keys, node_ids, signature=signature)

    assert bank.signature is signature
    np.testing.assert_array_equal(sidecar._probe.keys, keys)
    bank.close()


def test_arena_passes_epoch_signature_only_to_capable_native_store():
    class SignatureStore:
        supports_cuda_gqa_epoch_signature = True

        def __init__(self):
            self._cuda_gqa_bank = None
            self._cuda_gqa_bank_signature = None
            self.got_signature = None

        def configure_cuda_gqa_route_bank(
                self, _route_bank, _node_ids, *, signature=None):
            self.got_signature = signature
            self._cuda_gqa_bank = object()

    arena = GQAArenaCache.__new__(GQAArenaCache)
    store = SignatureStore()
    signature = ((101, (1, 1, 2), "<f4", 77),)
    bank_inputs = (
        np.ones((1, 1, 1, 2), np.float32),
        np.asarray([101], np.uint64),
        signature,
    )

    assert arena._ensure_cuda_route_bank(store, bank_inputs)
    assert store.got_signature is signature
    assert store._cuda_gqa_bank_signature is signature


class _ExactRaggedCudaStore:
    """CPU stand-in for the sidecar's stable raw-GQA rank contract."""

    def __init__(self):
        self._cuda_gqa_bank = None
        self._cuda_gqa_bank_signature = None
        self.node_ids = None
        self.configure_calls = 0

    def configure_cuda_gqa_route_bank(self, route_bank, node_ids, **_kwargs):
        self._cuda_gqa_bank = np.asarray(route_bank, np.float32).copy()
        self.node_ids = np.asarray(node_ids, np.uint64).copy()
        self.configure_calls += 1

    def route_gqa_cuda(self, query, *, topk=3, **_kwargs):
        ref = GQAArenaCache.__new__(GQAArenaCache)
        query = np.asarray(query, np.float32)
        scores = [ref._key_score(query, self._cuda_gqa_bank[idx])
                  for idx in range(self._cuda_gqa_bank.shape[0])]
        # Explicit node-row tie order mirrors the CUDA development gate.
        order = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))
        return [int(self.node_ids[idx]) for idx in order[:int(topk)]]

    def route_gqa(self, *_args, **_kwargs):
        raise AssertionError("exact CUDA-eligible route used the native CPU path")


def test_route_backend_python_forces_the_unchanged_python_gqa_path(
        monkeypatch):
    """The explicit control arm must bypass both native and CUDA routing."""
    monkeypatch.setenv("GRM_ROUTE_QUERY_LEX", "0")
    rows = [
        np.asarray([[[0.2, 0.0]]], np.float32),
        np.asarray([[[2.0, 0.0], [0.3, 0.0]]], np.float32),
        np.asarray([[[0.5, 0.0]]], np.float32),
    ]
    query = np.asarray([[[1.0, 0.0]]], np.float32)
    arena = _arena(rows)
    arena.native_store = _ExactRaggedCudaStore()
    arena.route_backend = "python"
    arena._probe_key = lambda _text: query

    expected = arena._python_gqa_exact_order(query, [0, 1, 2], 3)
    got = arena.route("plain probe", exclude=set(), limit=3)

    assert got == expected == [1, 2, 0]
    assert arena.last_route_backend == "python"
    assert arena.native_store.configure_calls == 0


def test_explicit_cuda_ragged_selector_preserves_stable_ties_and_receipt(
        monkeypatch):
    monkeypatch.delenv("GRM_GQA_CUDA_ROUTE", raising=False)
    monkeypatch.setenv("GRM_ROUTE_QUERY_LEX", "0")
    row = np.asarray([[[1.0, 0.0], [0.0, 0.5]]], np.float32)
    rows = [row.copy(), row.copy(), row.copy()]
    query = np.asarray([[[1.0, 0.0]]], np.float32)
    arena = _arena(rows)
    arena._cuda_gqa_epoch = 0
    arena.native_store = _ExactRaggedCudaStore()
    arena.route_backend = "cuda_ragged"
    arena.route_profile = True
    arena.route_parity_check = True
    arena._probe_key = lambda _text: query

    got = arena.route("plain probe", exclude=set(), limit=3)

    assert got == [0, 1, 2]
    assert arena.last_route_backend == "cuda"
    assert arena.last_route_receipt["requested_backend"] == "cuda_ragged"
    assert arena.last_route_receipt["parity"] == {
        "checked": True,
        "ok": True,
        "reference": "python_raw_gqa_key_score_stable_ties",
    }
    assert arena.last_route_receipt["ranking_prefix"] == [0, 1, 2]
    assert arena.last_route_receipt["rebuilds"]["cuda_bank"]["status"] == "rebuilt"
    assert arena.last_route_receipt["rebuilds"]["cuda_reverse_map"]["status"] == "rebuilt"


def test_ragged_cuda_reverse_map_reuses_until_epoch_changes(monkeypatch):
    monkeypatch.setenv("GRM_ROUTE_QUERY_LEX", "0")
    rows = [
        np.asarray([[[0.1, 0.0]]], np.float32),
        np.asarray([[[1.0, 0.0], [2.0, 0.0]]], np.float32),
    ]
    query = np.asarray([[[1.0, 0.0]]], np.float32)
    arena = _arena(rows)
    arena._cuda_gqa_epoch = 0
    arena.native_store = _ExactRaggedCudaStore()
    arena.route_backend = "cuda_ragged"
    arena.route_profile = True
    arena._probe_key = lambda _text: query

    assert arena.route("plain probe", exclude=set(), limit=2) == [1, 0]
    first_bank = arena._cuda_gqa_bank_cache
    first_map = arena._cuda_gqa_native_to_idx_cache
    assert arena.native_store.configure_calls == 1

    assert arena.route("plain probe", exclude=set(), limit=2) == [1, 0]
    assert arena._cuda_gqa_bank_cache is first_bank
    assert arena._cuda_gqa_native_to_idx_cache is first_map
    assert arena.last_route_receipt["rebuilds"]["cuda_bank"]["status"] == "reused"
    assert (arena.last_route_receipt["rebuilds"]
            ["cuda_reverse_map"]["status"] == "reused")

    arena.grafts[0]["cent"] = np.asarray(
        [[[3.0, 0.0], [0.2, 0.0], [0.1, 0.0]]], np.float32)
    arena._bump_cuda_gqa_epoch()

    assert arena.route("plain probe", exclude=set(), limit=2) == [0, 1]
    assert arena._cuda_gqa_bank_cache is not first_bank
    assert arena._cuda_gqa_native_to_idx_cache is not first_map
    assert arena.native_store.configure_calls == 2
    assert arena.last_route_receipt["rebuilds"]["cuda_bank"]["status"] == "rebuilt"
    assert (arena.last_route_receipt["rebuilds"]
            ["cuda_reverse_map"]["status"] == "rebuilt")


def test_ragged_bank_native_id_marshaling_round_trips_to_graft_indices():
    rows = [
        np.ones((1, 1, 2), np.float32),
        np.ones((1, 3, 2), np.float32),
        np.ones((1, 2, 2), np.float32),
    ]
    arena = _arena(rows, node_ids=[9001, 42, 7007])
    arena._cuda_gqa_epoch = 0

    bank_inputs = arena._cuda_route_bank_inputs()
    native_to_idx, graft_idx_set = arena._cuda_gqa_native_to_idx(bank_inputs)

    assert bank_inputs[1].tolist() == [9001, 42, 7007]
    assert native_to_idx == {9001: 0, 42: 1, 7007: 2}
    assert graft_idx_set == frozenset((0, 1, 2))
    # This is the exact return path used after CUDA top-k: no row-order or
    # native-id transformation can silently remap a candidate.
    assert [native_to_idx[node_id] for node_id in (7007, 9001, 42)] == [2, 0, 1]


def test_ineligible_ragged_bank_is_cached_until_the_epoch_moves(monkeypatch):
    row = np.ones((1, 2, 2), np.float32)
    arena = _arena([row])
    arena._cuda_gqa_epoch = 0
    arena.grafts[0]["child_cents"] = [row]
    calls = []
    original = arena._cuda_route_bank_signature

    def counted_signature():
        calls.append(1)
        return original()

    monkeypatch.setattr(arena, "_cuda_route_bank_signature", counted_signature)

    assert arena._cuda_route_bank_inputs() is None
    assert arena._cuda_route_bank_inputs() is None
    assert len(calls) == 1

    arena._bump_cuda_gqa_epoch()
    assert arena._cuda_route_bank_inputs() is None
    assert len(calls) == 2
