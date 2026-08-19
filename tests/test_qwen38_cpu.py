import inspect
import json
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

import core.qwen38_tc as q38
import scripts.qwen38_generate as q38_generate
import scripts.qwen38_longctx_gate as longctx
from core.qwen38_tc import (DEFAULT_MODEL_DIR, INT3PackCache, Qwen38Config,
                            Qwen38DeviceKVCache, Qwen38HostKVCache,
                            Qwen38AttentionTC, Qwen38_TC,
                            PackedRowChunkedQuantLinearTC,
                            _bf16_u16_to_float, _device_zeros,
                            _float_to_bf16_u16,
                            compute_qwen38_memory_budget, dequantize_kv_int8_np,
                            gate_int3_pack, gate_kv_int8_pack,
                            gate_online_softmax_tiled_np,
                            online_softmax_tiled_np, quantize_kv_int8_np,
                            select_qwen38_attn_path,
                            validate_qwen38_memory_budget)


def test_qwen38_vector_pack_is_engine_bit_exact():
    assert gate_int3_pack()["max_byte_diff"] == 0


def test_qwen38_real_config_ratios_and_gate():
    cfg = Qwen38Config.from_model_dir(DEFAULT_MODEL_DIR)
    with open(f"{DEFAULT_MODEL_DIR}/config.json") as fh:
        checkpoint_text_config = json.load(fh)["text_config"]
    assert (cfg.hidden_dim, cfg.num_layers, cfg.intermediate_dim) == (5120, 64, 17408)
    assert cfg.n_v_heads // cfg.n_k_heads == 3
    assert cfg.num_heads // cfg.num_kv_heads == 6
    assert checkpoint_text_config["output_gate_type"] == "swish"
    assert cfg.output_gate_type == "sigmoid"
    assert cfg.attention_layer_indices() == list(range(3, 64, 4))


def test_qwen38_computed_budget_is_below_order_ceiling():
    budget = compute_qwen38_memory_budget(DEFAULT_MODEL_DIR)
    assert budget["projected_whole_device_peak"] < 12000 * 2**20
    assert budget["host_bf16_embedding_ram"] == 248320 * 5120 * 2
    validate_qwen38_memory_budget(budget)


def test_qwen38_lm_head_default_caps_dequant_below_64_mib():
    ranges = PackedRowChunkedQuantLinearTC.row_ranges(248320)
    assert len(ranges) == 40
    assert ranges[0] == (0, 6208)
    assert ranges[-1] == (242112, 248320)
    assert 6208 * 5120 * 2 <= 64 * 2**20


def test_qwen38_kv_int8_cpu_pack_unpack_is_bit_exact():
    assert gate_kv_int8_pack()["unpack_bit_exact"] is True
    x = np.array([[[[0.0, -1.0, 0.5, 2048.0]]]], np.float32)
    packed, scale = quantize_kv_int8_np(x)
    assert packed.dtype == np.uint8
    np.testing.assert_array_equal(
        dequantize_kv_int8_np(packed, scale),
        (packed.astype(np.int16).astype(np.float32) - 128.0) * scale)


def test_qwen38_online_softmax_tiled_matches_standard_fp32():
    result = gate_online_softmax_tiled_np()
    assert result["status"] == "PASS"
    assert result["defect_caught"] is None
    assert [case["tiles"] for case in result["cases"]] == [1, 2, 7]
    assert result["cases"][1]["remainder"] != 0
    assert result["cases"][2]["remainder"] != 0
    assert max(case["rel_l2"] for case in result["cases"]) <= 1e-6


def test_qwen38_attention_path_dispatch_is_configuration_level():
    assert select_qwen38_attn_path(4096) == "legacy"
    assert select_qwen38_attn_path(4097) == "tiled"
    assert select_qwen38_attn_path(4096, kv_int8=True) == "tiled"
    assert select_qwen38_attn_path(4096, kv_host=True) == "tiled"
    assert select_qwen38_attn_path(4096, force_tiled=True) == "tiled"


def test_qwen38_budget_selects_long_context_rungs_before_load():
    bf16_16k = compute_qwen38_memory_budget(DEFAULT_MODEL_DIR, 16384)
    int8_16k = compute_qwen38_memory_budget(
        DEFAULT_MODEL_DIR, 16384, kv_int8=True)
    host_262k = compute_qwen38_memory_budget(
        DEFAULT_MODEL_DIR, 262144, kv_host=True)
    assert bf16_16k["projected_whole_device_peak"] > 12000 * 2**20
    assert int8_16k["projected_whole_device_peak"] <= 12000 * 2**20
    assert host_262k["projected_whole_device_peak"] <= 12000 * 2**20


def test_qwen38_budget_validator_pass_and_red_branches():
    assert validate_qwen38_memory_budget(
        {"projected_whole_device_peak": 2**20}, ceiling_mib=1) == 2**20
    with pytest.raises(MemoryError, match="exceeds fixed ceiling"):
        validate_qwen38_memory_budget(
            {"projected_whole_device_peak": 2**20 + 1}, ceiling_mib=1)


def test_qwen38_lc1_function_signatures_are_importable_and_bindable():
    cases = [
        (quantize_kv_int8_np, (np.zeros((1, 1), np.float32),), {}),
        (dequantize_kv_int8_np,
         (np.zeros((1, 1), np.uint8), np.ones((1, 1), np.float32)), {}),
        (gate_kv_int8_pack, (), {"seed": 1}),
        (online_softmax_tiled_np,
         (np.zeros((1, 1, 1, 2), np.float32),
          np.zeros((1, 1, 1, 2), np.float32),
          np.zeros((1, 1, 1, 2), np.float32), 1), {}),
        (gate_online_softmax_tiled_np, (), {"seed": 1}),
        (select_qwen38_attn_path, (4096,), {}),
        (_device_zeros, (1, 2), {"dtype": "uint8"}),
        (_float_to_bf16_u16, (np.zeros(1, np.float32),), {}),
        (_bf16_u16_to_float, (np.zeros(1, np.uint16),), {}),
        (Qwen38DeviceKVCache.__init__, (object(), 1, object(), 8), {"int8": True}),
        (Qwen38DeviceKVCache._pack, (object(),), {}),
        (Qwen38DeviceKVCache._unpack, (object(), object()), {}),
        (Qwen38DeviceKVCache.append, (object(), object(), object()), {}),
        (Qwen38DeviceKVCache.block, (object(), 0, 1), {}),
        (Qwen38DeviceKVCache.all, (object(),), {}),
        (Qwen38HostKVCache.__init__, (object(), 1, object(), 8), {}),
        (Qwen38HostKVCache.append, (object(), object(), object()), {}),
        (Qwen38HostKVCache.block, (object(), 0, 1), {}),
        (Qwen38AttentionTC._streaming_standard_attention,
         (object(), object(), object(), 0, 8), {}),
        (Qwen38AttentionTC.__call__,
         (object(), object(), object(), object()), {"position_offset": 0}),
        (Qwen38_TC.__init__, (object(),), {"max_context": 8}),
        (Qwen38_TC.new_caches, (object(),), {"batch_size": 1}),
        (Qwen38_TC.take_preallocated_caches, (object(),), {"batch_size": 1}),
        (Qwen38_TC.__call__, (object(), np.zeros((1, 1), np.int64)), {}),
        (Qwen38_TC._forward, (object(), np.zeros((1, 1), np.int64)), {}),
        (Qwen38_TC._computed_vram_map, (object(),), {"context_tokens": 8}),
        (Qwen38_TC.load_weights, (object(),), {"model_dir": "model"}),
        (Qwen38_TC.from_pretrained, (), {"model_dir": "model", "max_context": 8}),
        (compute_qwen38_memory_budget, (), {"context_tokens": 8}),
        (validate_qwen38_memory_budget,
         ({"projected_whole_device_peak": 1},), {"ceiling_mib": 1}),
        (q38_generate.greedy, (object(), object(), np.zeros((1, 1)), 1), {}),
        (q38_generate.print_budget, ("model",), {"max_context": 8}),
        (q38_generate.validate_budget_for_load,
         ({"projected_whole_device_peak": 1},), {}),
        (longctx.vault_code, (1,), {}),
        (longctx._encode, (object(), "text"), {}),
        (longctx.seeded_filler_ids, (object(), 8, 1), {}),
        (longctx.build_needle_ids, (object(), 64, 0.5, 1), {}),
        (longctx.build_coherence_ids, (object(), 64, 1), {}),
        (longctx.load_model, (object(), object()), {}),
        (longctx.validate_budget_for_run,
         (object(), {"projected_whole_device_peak": 1}), {}),
        (longctx.is_tensor_cuda_oom, (RuntimeError("cudaMalloc failed"),), {}),
        (longctx.int8_teacher_forced_agreement,
         (object(), object(), np.zeros((1, 1)), 1, object()), {}),
        (longctx.teacher_forced_tiled_margins,
         (object(), object(), object()), {}),
        (longctx.parse_args, (), {}),
        (longctx.main, (), {}),
    ]
    for fn, args, kwargs in cases:
        inspect.signature(fn).bind(*args, **kwargs)


def test_qwen38_zero_allocator_avoids_fp32_staging(monkeypatch):
    calls = []

    class FakeZeros:
        def astype(self, dtype):
            calls.append(("astype", dtype))
            return ("cast", dtype)

    fake = FakeZeros()

    class FakeTC:
        @staticmethod
        def zeros(*shape, dtype):
            calls.append(("zeros", shape, dtype))
            return fake

    monkeypatch.setattr(q38, "tc", FakeTC)
    assert _device_zeros(2, 3, dtype="bfloat16") == ("cast", "bfloat16")
    assert calls == [("zeros", (2, 3), "uint8"), ("astype", "bfloat16")]
    calls.clear()
    assert _device_zeros(2, 3, dtype="uint8") is fake
    assert calls == [("zeros", (2, 3), "uint8")]


def test_qwen38_host_kv_bookkeeping_roundtrip_and_capacity(monkeypatch):
    cfg = SimpleNamespace(num_kv_heads=2, head_dim=4)
    cache = Qwen38HostKVCache(1, cfg, 3)

    class HostInput:
        def __init__(self, value):
            self.value = np.asarray(value, np.float32)
            self.shape = self.value.shape

        def float(self):
            return self

        def numpy(self):
            return self.value

    monkeypatch.setattr(q38, "tc", SimpleNamespace(
        tensor=lambda value: np.asarray(value, np.float32)))
    monkeypatch.setattr(q38, "_cast", lambda value: value)
    k = HostInput(np.arange(16, dtype=np.float32).reshape(1, 2, 2, 4) / 8)
    v = HostInput(-k.value)
    cache.append(k, v)
    assert cache.count == 2
    assert cache.storage_bytes == 1 * 2 * 3 * 4 * 2 * 2
    got_k, got_v = cache.block(0, 2)
    np.testing.assert_array_equal(got_k, _bf16_u16_to_float(_float_to_bf16_u16(k.value)))
    np.testing.assert_array_equal(got_v, _bf16_u16_to_float(_float_to_bf16_u16(v.value)))
    with pytest.raises(MemoryError, match="host KV capacity exceeded"):
        cache.append(k, v)


def test_qwen38_device_kv_storage_bookkeeping_counts_k_and_v():
    bf16 = Qwen38DeviceKVCache.__new__(Qwen38DeviceKVCache)
    bf16.capacity = 3
    bf16.int8 = False
    bf16.kb = SimpleNamespace(shape=(1, 2, 3, 4))
    assert bf16.storage_bytes == 1 * 2 * 3 * 4 * 2 * 2
    int8 = Qwen38DeviceKVCache.__new__(Qwen38DeviceKVCache)
    int8.capacity = 3
    int8.int8 = True
    int8.kb = SimpleNamespace(shape=(1, 2, 3, 4))
    assert int8.storage_bytes == 1 * 2 * 3 * (2 * 4 + 4)


def test_qwen38_last_token_is_sliced_before_lm_head():
    from core.qwen38_tc import Qwen38_TC

    class FakeTensor:
        def __init__(self, shape):
            self.shape = tuple(shape)

        def astype(self, _dtype):
            return self

        def half(self):
            return self

        def slice(self, dim, start, length):
            shape = list(self.shape)
            shape[dim] = length
            return FakeTensor(shape)

    seen = []
    model = Qwen38_TC.__new__(Qwen38_TC)
    model.config = SimpleNamespace()
    model.layers = []
    model._rope_len = 4096
    model.max_context = 4096
    model.prefill_chunk = 64
    model.kv_host = False
    model.kv_int8 = False
    model.embed_tokens = lambda ids: FakeTensor((*ids.shape, 5120))
    model.norm = lambda h: h
    model.lm_head = lambda h: seen.append(h.shape) or FakeTensor(
        (*h.shape[:-1], 248320))
    logits, caches = model(np.zeros((1, 37), dtype=np.int64),
                           last_token_only=True)
    assert seen == [(1, 1, 5120)]
    assert logits.shape == (1, 1, 248320)
    assert caches == []


def test_qwen38_chunked_prefill_control_flow_and_cache_consumption(monkeypatch):
    model = Qwen38_TC.__new__(Qwen38_TC)
    model.max_context = 16
    model.prefill_chunk = 2
    model._preallocated_caches = ["reserved"]
    calls = []

    def forward(ids, caches, position_offset, last_token_only=False,
                max_layers=None):
        calls.append((ids.copy(), caches, position_offset, last_token_only,
                      max_layers))
        return f"logits-{position_offset}", caches

    model._forward = forward
    assert model.take_preallocated_caches() == ["reserved"]
    assert model._preallocated_caches is None
    model.new_caches = lambda batch_size=1: [f"fresh-{batch_size}"]
    empty_calls = []
    monkeypatch.setattr(q38, "tc", SimpleNamespace(
        empty_cache=lambda: empty_calls.append(True),
        cat=lambda values, dim: (tuple(values), dim)))
    ids = np.arange(5, dtype=np.int64).reshape(1, 5)
    logits, caches = model(ids, last_token_only=False, position_offset=3)
    assert [call[2] for call in calls] == [3, 5, 7]
    assert [call[0].shape[1] for call in calls] == [2, 2, 1]
    assert logits == (("logits-3", "logits-5", "logits-7"), 1)
    assert caches == ["fresh-1"]
    assert len(empty_calls) == 3
    with pytest.raises(ValueError, match="exceeds configured context"):
        model(np.zeros((1, 14), np.int64), position_offset=3)


def test_qwen38_attention_baseline_path_resolves_functional_api(monkeypatch):
    class TraceTensor:
        def __init__(self, shape):
            self.shape = tuple(shape)

        def reshape(self, shape):
            return TraceTensor(shape)

        def slice(self, dim, _start, length):
            shape = list(self.shape)
            shape[dim] = length
            return TraceTensor(shape)

        def transpose(self, a, b):
            shape = list(self.shape)
            shape[a], shape[b] = shape[b], shape[a]
            return TraceTensor(shape)

        def sigmoid(self):
            return self

        def __mul__(self, _other):
            return self

    class FakeF:
        rotary_calls = 0
        sdpa_calls = 0

        @classmethod
        def apply_rotary(cls, x, cos, sin):
            cls.rotary_calls += 1
            return x

        @classmethod
        def scaled_dot_product_attention(cls, q, k, v, **kwargs):
            cls.sdpa_calls += 1
            return q

    class FakeTC:
        @staticmethod
        def cat(parts, dim=0):
            shape = list(parts[0].shape)
            shape[dim] = sum(part.shape[dim] for part in parts)
            return TraceTensor(shape)

    cfg = SimpleNamespace(num_heads=2, num_kv_heads=1, head_dim=4,
                          partial_rotary_dim=2, rms_norm_eps=1e-6,
                          output_gate_type="sigmoid")
    attn = Qwen38AttentionTC.__new__(Qwen38AttentionTC)
    attn.cfg = cfg
    attn.attn_path = "legacy"
    attn.q_proj = lambda x: TraceTensor((1, 3, 16))
    attn.k_proj = lambda x: TraceTensor((1, 3, 4))
    attn.v_proj = lambda x: TraceTensor((1, 3, 4))
    attn.o_proj = lambda x: x
    attn.q_norm_w = attn.k_norm_w = object()
    cache = Qwen38DeviceKVCache.__new__(Qwen38DeviceKVCache)
    cache.int8 = False
    cache.capacity = 4096
    cache.count = 0
    cache.append = lambda k, v: setattr(cache, "count", k.shape[2])
    cache.all = lambda: (TraceTensor((1, 1, cache.count, 4)),
                         TraceTensor((1, 1, cache.count, 4)))
    monkeypatch.setattr(q38, "F", FakeF)
    monkeypatch.setattr(q38, "tc", FakeTC)
    monkeypatch.setattr(q38, "_cast", lambda value: value)
    monkeypatch.setattr(q38, "_per_head_rmsnorm",
                        lambda x, w, eps, B, L, H, D: TraceTensor((B, L, H * D)))
    monkeypatch.setattr(q38, "_repeat_kv", lambda value, rep: value)
    got, returned_cache = attn(
        TraceTensor((1, 3, 8)), TraceTensor((4096, 2)),
        TraceTensor((4096, 2)), kv_cache=cache)
    assert got.shape == (1, 3, 8)
    assert returned_cache is cache
    assert FakeF.rotary_calls == 2
    assert FakeF.sdpa_calls == 1


def test_qwen38_streaming_attention_empty_cache_is_fail_loud():
    attn = Qwen38AttentionTC.__new__(Qwen38AttentionTC)
    attn.cfg = SimpleNamespace(num_kv_heads=1)
    q = SimpleNamespace(shape=(1, 2, 3, 4), reshape=lambda shape: q)
    with pytest.raises(RuntimeError, match="attention cache is empty"):
        attn._streaming_standard_attention(
            q, SimpleNamespace(count=0), position_offset=0, block_rows=2)


def test_qwen38_lm_head_invokes_two_stage_once_per_packed_chunk(monkeypatch):
    import core.qwen38_tc as q38

    calls = []

    class FakeTC:
        @staticmethod
        def intn_linear(x, packed, scales, zeros, bits, in_features, group_size):
            calls.append((x, packed, scales, zeros, bits, in_features, group_size))
            return f"logits-{packed}"

        @staticmethod
        def cat(parts, dim):
            return tuple(parts), dim

    monkeypatch.setattr(q38, "tc", FakeTC)
    head = PackedRowChunkedQuantLinearTC.__new__(PackedRowChunkedQuantLinearTC)
    head.chunks = [(f"p{i}", f"s{i}", f"z{i}") for i in range(40)]
    head.bits = 3
    head.in_features = 5120
    head.group_size = 128
    got = head("hidden")
    assert len(calls) == 40
    assert all(call[0] == "hidden" and call[4:] == (3, 5120, 128)
               for call in calls)
    assert got == (tuple(f"logits-p{i}" for i in range(40)), -1)


class _TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [sum(map(ord, word)) % 997 for word in text.split()]

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids)


def test_qwen38_needle_and_coherence_builders_cover_validation_branches():
    tokenizer = _TinyTokenizer()
    ids, meta = longctx.build_needle_ids(tokenizer, 128, 0.5, 7)
    ids_again, meta_again = longctx.build_needle_ids(tokenizer, 128, 0.5, 7)
    assert ids.shape == (1, 128)
    np.testing.assert_array_equal(ids, ids_again)
    assert meta == meta_again
    assert meta["expected"] == longctx.vault_code(7)
    assert 0 < meta["needle_token_start"] < 128
    assert longctx.build_coherence_ids(tokenizer, 96, 7).shape == (1, 96)
    with pytest.raises(ValueError, match="at least 64"):
        longctx.build_needle_ids(tokenizer, 63, 0.5, 7)
    with pytest.raises(ValueError, match="strictly between"):
        longctx.build_needle_ids(tokenizer, 128, 1.0, 7)
    with pytest.raises(ValueError, match="coherence length too small"):
        longctx.build_coherence_ids(tokenizer, 1, 7)


def test_qwen38_longctx_parse_args_branches(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gate", "--baseline-check"])
    args = longctx.parse_args()
    assert args.max_context == q38.DEFAULT_MAX_CONTEXT
    monkeypatch.setattr(sys, "argv", [
        "gate", "--length", "128", "--max-new-tokens", "3"])
    assert longctx.parse_args().max_context == 131
    monkeypatch.setattr(sys, "argv", ["gate", "--force-alloc"])
    assert longctx.parse_args().force_alloc is True
    monkeypatch.setattr(sys, "argv", ["gate"])
    assert longctx.parse_args().force_alloc is False
    monkeypatch.setattr(sys, "argv", ["gate", "--kv-int8", "--kv-host"])
    with pytest.raises(SystemExit):
        longctx.parse_args()


def test_qwen38_force_alloc_budget_override_is_explicit_and_default_stays_loud(
        capsys):
    budget = {"projected_whole_device_peak": 12001 * 2**20}
    args = SimpleNamespace(force_alloc=False, max_context=131072,
                           kv_int8=False, kv_host=False)
    with pytest.raises(MemoryError, match="exceeds fixed ceiling"):
        longctx.validate_budget_for_run(args, budget)
    assert "BUDGET_OVERRIDE" not in capsys.readouterr().out

    args.force_alloc = True
    assert longctx.validate_budget_for_run(args, budget) is False
    line = capsys.readouterr().out.strip()
    assert line.startswith("BUDGET_OVERRIDE ")
    receipt = json.loads(line.removeprefix("BUDGET_OVERRIDE "))
    assert receipt == {
        "status": "WARNING", "error": receipt["error"],
        "max_context": 131072, "kv_mode": "device_bf16",
        "projected_peak_mib": 12001.0, "force_alloc": True,
    }
    assert receipt["error"].startswith("MemoryError: projected whole-device")


def test_qwen38_tensor_cuda_oom_classifier_matches_engine_messages():
    assert longctx.is_tensor_cuda_oom(
        RuntimeError("cudaMalloc failed: out of memory"))
    assert longctx.is_tensor_cuda_oom(
        RuntimeError("cudaMallocAsync failed: out of memory"))
    assert longctx.is_tensor_cuda_oom(
        RuntimeError("CUDA error at sync: out of memory"))
    assert not longctx.is_tensor_cuda_oom(RuntimeError("unrelated engine error"))
    assert not longctx.is_tensor_cuda_oom(MemoryError("budget wall"))


def test_qwen38_longctx_load_model_forwards_every_lc1_argument(monkeypatch, capsys):
    calls = []

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append((args, kwargs))
            return "model", {"loaded": True}

    class FakeSampler:
        def set_phase(self, phase):
            calls.append(("phase", phase))

        def sample_now(self, kind):
            calls.append(("sample", kind))

    args = SimpleNamespace(model_dir="m", cache_dir="c", lm_head_chunk_rows=11,
                           max_context=22, kv_int8=True, kv_host=False,
                           prefill_chunk=3, kv_block_rows=4, tf_margin=False)
    monkeypatch.setattr(longctx, "Qwen38_TC", FakeModelClass)
    assert longctx.load_model(args, FakeSampler()) == "model"
    assert calls[1] == (("m", "c"), {
        "lm_head_chunk_rows": 11, "max_context": 22, "kv_int8": True,
        "kv_host": False, "prefill_chunk": 3, "kv_block_rows": 4,
        "force_tiled_attention": False, "cache_read_only": True})
    assert "LOAD_RESULT" in capsys.readouterr().out


def test_qwen38_int8_teacher_forced_control_flow(monkeypatch):
    class FakeLogits:
        def __init__(self, winner):
            self.winner = winner

        def float(self):
            return self

        def numpy(self):
            out = np.zeros(8, np.float32)
            out[self.winner] = 1
            return out

    class FakeModel:
        kv_host = True
        kv_int8 = False

        def __init__(self):
            self.calls = 0

        def new_caches(self, batch_size):
            return [SimpleNamespace(key_stats={"vectors": batch_size})]

        def __call__(self, ids, **kwargs):
            winner = (2, 3)[self.calls]
            self.calls += 1
            return FakeLogits(winner), kwargs["caches"]

    phases = []
    sampler = SimpleNamespace(set_phase=lambda phase: phases.append(phase))
    monkeypatch.setattr(longctx, "greedy", lambda *args, **kwargs: {
        "token_ids": [2, 3]})
    monkeypatch.setattr(longctx, "tc", SimpleNamespace(
        empty_cache=lambda: None, synchronize=lambda: None))
    model = FakeModel()
    result = longctx.int8_teacher_forced_agreement(
        model, _TinyTokenizer(), np.zeros((1, 4), np.int64), 2, sampler)
    assert result["status"] == "PASS"
    assert result["agreement"] == 1.0
    assert model.kv_host is False and model.kv_int8 is True
    assert phases == ["prefill:kv_int8_teacher_forced",
                      "decode:kv_int8_teacher_forced"]


def test_qwen38_tiled_margin_reports_every_step_and_disagreement(monkeypatch,
                                                                 capsys):
    baselines = {"code": [1, 2], "factual": [1, 2], "chat": [1, 2]}
    winners = iter([1, 0, 1, 2, 1, 2])

    class FakeLogits:
        def __init__(self, winner):
            self.winner = winner

        def float(self):
            return self

        def numpy(self):
            out = np.zeros(4, np.float32)
            out[3] = 3.0
            out[self.winner] = 5.0
            return out

    class FakeModel:
        attn_path = "tiled"
        max_context = 32

        def take_preallocated_caches(self, batch_size):
            return [batch_size]

        def __call__(self, ids, **kwargs):
            return FakeLogits(next(winners)), kwargs["caches"]

    phases = []
    sampler = SimpleNamespace(set_phase=lambda phase: phases.append(phase))
    monkeypatch.setattr(longctx, "BASELINE_TOKEN_IDS", baselines)
    monkeypatch.setattr(longctx, "chat_ids",
                        lambda tokenizer, prompt: np.zeros((1, 2), np.int64))
    monkeypatch.setattr(longctx, "tc", SimpleNamespace(
        empty_cache=lambda: None, synchronize=lambda: None))
    result = longctx.teacher_forced_tiled_margins(
        FakeModel(), _TinyTokenizer(), sampler)
    assert result["status"] == "REPORT_ONLY"
    assert result["cases"]["code"]["matches"] == 1
    disagreement = result["cases"]["code"]["disagreements"]
    assert disagreement == [{
        "status": "REPORT_ONLY", "case": "code", "step": 1,
        "baseline_token": 2, "tiled_top1": 0,
        "top1_agreement": False, "logit_margin_top1_minus_top2": 2.0,
    }]
    assert len([line for line in capsys.readouterr().out.splitlines()
                if line.startswith("TF_MARGIN_STEP ")]) == 6
    assert phases == [
        "prefill:tf_margin_code", "decode:tf_margin_code",
        "prefill:tf_margin_factual", "decode:tf_margin_factual",
        "prefill:tf_margin_chat", "decode:tf_margin_chat"]


def test_qwen38_baseline_main_executes_all_registered_cases(monkeypatch, capsys):
    args = SimpleNamespace(
        model_dir="model", cache_dir="cache", length=128, depth=0.5, seed=7,
        max_new_tokens=64, max_context=4096, prefill_chunk=64,
        kv_block_rows=256, lm_head_chunk_rows=6208, kv_int8=False,
        kv_host=False, coherence_smoke=False, baseline_check=True,
        kv_agreement=False, tf_margin=False, dry_run=False, force_alloc=False)

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _TinyTokenizer()

    class FakeSampler:
        def start(self):
            pass

        def stop(self):
            pass

        def receipt(self):
            return {"peak_mib": {}}

    seen = []

    def fake_greedy(model, tokenizer, ids, count, sampler, phase):
        name = phase.removeprefix("baseline_")
        seen.append(name)
        return {"token_ids": longctx.BASELINE_TOKEN_IDS[name]}

    monkeypatch.setitem(sys.modules, "transformers",
                        SimpleNamespace(AutoTokenizer=AutoTokenizer))
    monkeypatch.setattr(longctx, "parse_args", lambda: args)
    monkeypatch.setattr(longctx, "print_budget", lambda *values: {
        "projected_whole_device_peak": 1})
    monkeypatch.setattr(longctx, "VramSampler", FakeSampler)
    monkeypatch.setattr(longctx, "load_model", lambda args, sampler: object())
    monkeypatch.setattr(longctx, "chat_ids",
                        lambda tokenizer, prompt: np.zeros((1, 2), np.int64))
    monkeypatch.setattr(longctx, "greedy", fake_greedy)
    longctx.main()
    assert seen == list(longctx.DEMO_PROMPTS)
    output = capsys.readouterr().out
    assert 'BASELINE_RESULT {"cases":' in output
    assert '"status": "PASS"' in output


@pytest.mark.parametrize("oom_phase", ["load", "prefill", "decode"])
def test_qwen38_force_alloc_cuda_oom_exits_cleanly_by_phase(
        monkeypatch, capsys, oom_phase):
    args = SimpleNamespace(
        model_dir="model", cache_dir="cache", length=128, depth=0.5, seed=7,
        max_new_tokens=1, max_context=131072, prefill_chunk=64,
        kv_block_rows=256, lm_head_chunk_rows=6208, kv_int8=False,
        kv_host=False, coherence_smoke=False, baseline_check=False,
        kv_agreement=False, tf_margin=False, dry_run=False, force_alloc=True)

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _TinyTokenizer()

    class FakeSampler:
        instances = []

        def __init__(self):
            self.phase = "load"
            self.stopped = False
            type(self).instances.append(self)

        def start(self):
            pass

        def stop(self):
            self.stopped = True

        def receipt(self):
            assert self.stopped
            return {"peak_mib": {self.phase.split(':', 1)[0]: 11999}}

    oom = RuntimeError("cudaMallocAsync failed: out of memory")

    def fake_load_model(_args, sampler):
        sampler.phase = "load"
        if oom_phase == "load":
            raise oom
        return object()

    def fake_greedy(model, tokenizer, ids, count, sampler, phase):
        sampler.phase = f"{oom_phase}:{phase}"
        raise oom

    monkeypatch.setitem(sys.modules, "transformers",
                        SimpleNamespace(AutoTokenizer=AutoTokenizer))
    monkeypatch.setattr(longctx, "parse_args", lambda: args)
    monkeypatch.setattr(longctx, "print_budget", lambda *values: {
        "projected_whole_device_peak": 12001 * 2**20})
    monkeypatch.setattr(longctx, "VramSampler", FakeSampler)
    monkeypatch.setattr(longctx, "load_model", fake_load_model)
    monkeypatch.setattr(longctx, "greedy", fake_greedy)

    with pytest.raises(SystemExit) as caught:
        longctx.main()
    assert caught.value.code == 1
    assert FakeSampler.instances[-1].stopped is True
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("BUDGET_OVERRIDE ") for line in lines)
    result_line = next(line for line in lines
                       if line.startswith("RECALL_RESULT "))
    result = json.loads(result_line.removeprefix("RECALL_RESULT "))
    assert result["status"] == "RED"
    assert result["error"] == "CUDA_OOM: cudaMallocAsync failed: out of memory"
    assert result["phase"] == oom_phase
    assert result["vram"]["peak_mib"] == {oom_phase: 11999}


def test_qwen38_nonforced_cuda_oom_is_not_silently_downgraded(
        monkeypatch, capsys):
    args = SimpleNamespace(
        model_dir="model", cache_dir="cache", length=128, depth=0.5, seed=7,
        max_new_tokens=1, max_context=128, prefill_chunk=64,
        kv_block_rows=256, lm_head_chunk_rows=6208, kv_int8=False,
        kv_host=False, coherence_smoke=False, baseline_check=False,
        kv_agreement=False, tf_margin=False, dry_run=False, force_alloc=False)

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _TinyTokenizer()

    class FakeSampler:
        phase = "load"

        def start(self): pass
        def stop(self): pass
        def receipt(self): return {"peak_mib": {}}

    monkeypatch.setitem(sys.modules, "transformers",
                        SimpleNamespace(AutoTokenizer=AutoTokenizer))
    monkeypatch.setattr(longctx, "parse_args", lambda: args)
    monkeypatch.setattr(longctx, "print_budget", lambda *values: {
        "projected_whole_device_peak": 1})
    monkeypatch.setattr(longctx, "VramSampler", FakeSampler)
    monkeypatch.setattr(
        longctx, "load_model",
        lambda *values: (_ for _ in ()).throw(
            RuntimeError("cudaMalloc failed: out of memory")))
    with pytest.raises(RuntimeError, match="cudaMalloc failed"):
        longctx.main()
    assert "CUDA_OOM:" not in capsys.readouterr().out


def test_qwen38_generate_greedy_context_and_int8_receipt(monkeypatch):
    class FakeLogits:
        def __init__(self, winner):
            self.winner = winner

        def float(self):
            return self

        def numpy(self):
            out = np.zeros(8, np.float32)
            out[self.winner] = 1
            return out

    class FakeModel:
        max_context = 8
        kv_int8 = True
        config = SimpleNamespace(eos_token_id=7)

        def __init__(self):
            self.calls = 0

        def take_preallocated_caches(self, batch_size):
            return [SimpleNamespace(key_stats={"vectors": batch_size})]

        def __call__(self, ids, **kwargs):
            winner = (2, 7)[self.calls]
            self.calls += 1
            return FakeLogits(winner), kwargs["caches"]

    monkeypatch.setattr(q38.tc, "synchronize", lambda: None)
    result = q38_generate.greedy(
        FakeModel(), _TinyTokenizer(), np.zeros((1, 3), np.int64), 3)
    assert result["token_ids"] == [2, 7]
    assert result["kv_key_stats_first_prefill_chunk"] == [
        {"attention_layer": 0, "vectors": 1}]
    with pytest.raises(ValueError, match=r"prompt\+decode needs"):
        q38_generate.greedy(
            FakeModel(), _TinyTokenizer(), np.zeros((1, 7), np.int64), 2)


def test_qwen38_print_budget_forwards_chunk_and_kv_options(monkeypatch, capsys):
    seen = []

    def fake_budget(*args, **kwargs):
        seen.append((args, kwargs))
        return {"host_bf16_kv_seq9": 10, "projected_whole_device_peak": 20}

    monkeypatch.setattr(q38_generate, "compute_qwen38_memory_budget", fake_budget)
    got = q38_generate.print_budget(
        "model", max_context=9, kv_int8=True, kv_host=False,
        lm_head_chunk_rows=10, prefill_chunk=11, kv_block_rows=12)
    assert got["projected_whole_device_peak"] == 20
    assert seen == [(("model",), {
        "context_tokens": 9, "kv_int8": True, "kv_host": False,
        "lm_head_chunk_rows": 10, "prefill_chunk": 11,
        "kv_block_rows": 12})]
    assert "PER-COMPONENT MEMORY MAP" in capsys.readouterr().out


def test_qwen38_cache_quantizer_matches_engine_reference():
    import torch
    from tensor_cuda.quantization import quantize_affine_per_group

    rng = np.random.default_rng(38)
    weight = rng.standard_normal((17, 256)).astype(np.float32)
    name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        cache = INT3PackCache(cache_dir=tmp, row_chunk=5)
        entry = cache._quantize_tensor(torch.from_numpy(weight), name)
        packed = np.load(f"{tmp}/{entry['packed']}")
        scales = np.load(f"{tmp}/{entry['scales']}")
        zeros = np.load(f"{tmp}/{entry['zeros']}")
    expected = quantize_affine_per_group(weight, 3, 128)
    np.testing.assert_array_equal(packed, expected.packed)
    np.testing.assert_array_equal(scales, expected.scales)
    np.testing.assert_array_equal(zeros, expected.zeros)
