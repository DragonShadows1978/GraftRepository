import ctypes
import os
import subprocess
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from core.graft_arena import ArenaCache, GQAArenaCache
from core.graft_repository import GraftRepository
from core.grm_native import NativeGraftStore
from core.grm_runtime import GRMRuntime


ROOT = "/mnt/ForgeRealm/GraftRepository"


def test_arena_step_prompt_template_formats_prompt_and_deposit_turn():
    arena = ArenaCache.__new__(ArenaCache)
    arena.prompt_template = (
        lambda user, assistant: (
            f"<|start|>user<|message|>{user}"
            f"<|end|><|start|>assistant<|channel|>final<|message|>"
            if assistant is None
            else f"<|start|>user<|message|>{user}"
                 f"<|end|><|start|>assistant<|channel|>final<|message|>"
                 f"{assistant}<|end|>"
        )
    )
    arena.stop_sequences = ("<|end|>",)

    assert arena._format_step_prompt("Probe A") == (
        "<|start|>user<|message|>Probe A"
        "<|end|><|start|>assistant<|channel|>final<|message|>")
    assert arena._format_step_turn("Probe A", "BLUE") == (
        "<|start|>user<|message|>Probe A"
        "<|end|><|start|>assistant<|channel|>final<|message|>BLUE<|end|>")
    assert arena.stop_sequences == ("<|end|>",)


def build_native(tmp_path, extra_cxxflags=(), extra_ldflags=()):
    suffix = "default" if not extra_cxxflags else "_".join(
        flag.strip("-").replace("=", "_") for flag in extra_cxxflags)
    lib = tmp_path / f"libgrm_runtime_{suffix}.so"
    subprocess.run([
        "g++", "-std=c++17", "-shared", "-fPIC",
        *extra_cxxflags,
        "-I", f"{ROOT}/cpp",
        f"{ROOT}/cpp/grm_runtime.cpp",
        "-o", str(lib),
        *extra_ldflags,
    ], check=True)
    return str(lib)


def test_cpp_durability_writer_commits_binary_checkpoint(tmp_path):
    source = tmp_path / "durability_writer_test.cpp"
    source.write_text(r'''
#include "grm_runtime.hpp"

#include <filesystem>
#include <string>

int main(int argc, char** argv) {
  if (argc != 2) {
    return 2;
  }
  grm::DialectDescriptor dialect;
  dialect.model_type = "DeepSeekV2Lite_TC";
  dialect.num_layers = 27;
  dialect.hidden_dim = 2048;
  dialect.payload_kind = grm::PayloadKind::MLA;
  dialect.vals_per_tok_layer = 576;
  dialect.route_layer = 3;
  dialect.latent_rank = 512;
  dialect.rope_dim = 64;
  dialect.position_law = "rope_partial_mla";
  dialect.state_kind = "mla_latent_plus_rope";

  grm::HostGraftStore store(dialect);
  grm::HostGraftNode node;
  node.text = "native writer checkpoint";
  node.ntok = 4;
  node.metadata.kind = "doc";
  grm::HostTensor tensor;
  tensor.name = "payload";
  tensor.dtype = "uint8";
  tensor.shape = {4};
  tensor.bytes = {1, 2, 3, 4};
  node.payload.tensors.push_back(tensor);
  const auto node_id = store.add_node(node);

  if (store.stats().dirty_nodes != 1) {
    return 3;
  }
  grm::DurabilityWriter writer(argv[1]);
  writer.write_checkpoint(store);

  const auto root = std::filesystem::path(argv[1]);
  if (!std::filesystem::exists(root / "grm_store.bin")) {
    return 4;
  }
  if (!std::filesystem::exists(root / "checkpoint.txt")) {
    return 5;
  }
  const auto committed = store.stats();
  if (committed.dirty_nodes != 0 || committed.durable_nodes != 1) {
    return 6;
  }

  grm::HostGraftStore restored(dialect);
  restored.load_checkpoint(argv[1]);
  const auto* got = restored.get(node_id);
  if (got == nullptr || got->text != "native writer checkpoint") {
    return 7;
  }
  if (got->payload.tensor_count() != 1 || got->payload.bytes() != 4) {
    return 8;
  }
  if (std::filesystem::file_size(root / "grm_store.bin") == 0) {
    return 9;
  }
  return 0;
}
''')
    exe = tmp_path / "durability_writer_test"
    subprocess.run([
        "g++", "-std=c++17",
        "-I", f"{ROOT}/cpp",
        f"{ROOT}/cpp/grm_runtime.cpp",
        str(source),
        "-o", str(exe),
    ], check=True)
    subprocess.run([str(exe), str(tmp_path / "writer_ckpt")], check=True)


def test_cpp_dialect_profile_rejects_fixed_position_remountable(tmp_path):
    source = tmp_path / "dialect_profile_test.cpp"
    source.write_text(r'''
#include "grm_runtime.hpp"

#include <stdexcept>
#include <string>

static grm::DialectDescriptor absolute_profile(bool remountable) {
  grm::DialectDescriptor dialect;
  dialect.model_type = "AbsoluteKV_TC";
  dialect.num_layers = 12;
  dialect.hidden_dim = 768;
  dialect.payload_kind = grm::PayloadKind::GQA;
  dialect.vals_per_tok_layer = 384;
  dialect.route_layer = 0;
  dialect.num_kv_heads = 4;
  dialect.head_dim = 64;
  dialect.position_law = "absolute_learned";
  dialect.state_kind = "kv";
  dialect.graftability = "fixed_position";
  dialect.remountable = remountable;
  dialect.composition = "single_mount";
  return dialect;
}

int main() {
  try {
    grm::HostGraftStore bad(absolute_profile(true));
    return 3;
  } catch (const std::invalid_argument&) {
  }

  grm::HostGraftStore ok(absolute_profile(false));
  if (ok.dialect().profile_id() !=
      "absolute_learned|kv|fixed_position|0|single_mount") {
    return 4;
  }
  return 0;
}
''')
    exe = tmp_path / "dialect_profile_test"
    subprocess.run([
        "g++", "-std=c++17",
        "-I", f"{ROOT}/cpp",
        f"{ROOT}/cpp/grm_runtime.cpp",
        str(source),
        "-o", str(exe),
    ], check=True)
    subprocess.run([str(exe)], check=True)


def test_native_store_lifecycle_via_ctypes(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        assert store.dialect_id() == "DeepSeekV2Lite_TC:27x2048:r512"
        assert store.dialect_profile() == (
            "rope_partial_mla|mla_latent_plus_rope|seat_remountable|1|"
            "multi_mount")

        node_id = store.add_node("native graft", b"abcdef", ntok=2)
        assert node_id == 0

        stats = store.stats()
        assert stats.nodes == 1
        assert stats.dirty_nodes == 1
        assert stats.durable_nodes == 0
        assert stats.host_payload_bytes == 6
        assert stats.host_payload_tensors == 1
        assert stats.route_entries == 0
        assert store.tensor_info(node_id, "payload").dtype == "uint8"
        assert store.get_tensor(node_id, "payload").tolist() == [
            ord(c) for c in "abcdef"]

        store.set_route(node_id, [0.0, 1.0, 0.0], ["native", "graft"])
        stats = store.stats()
        assert stats.route_entries == 1
        assert store.route([0.0, 0.9, 0.0], topk=1) == [0]

        store.mark_durable(node_id)
        stats = store.stats()
        assert stats.dirty_nodes == 0
        assert stats.durable_nodes == 1

        store.evict_device_copy(node_id)


def test_native_store_clear_route_removes_persistent_route_entry(tmp_path):
    lib = build_native(tmp_path)
    ckpt = tmp_path / "clear_route_ckpt"
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        node_id = store.add_node("route clear target", b"", ntok=3)
        store.set_route(node_id, [1.0, 0.0], ["clear-me"])
        assert store.stats().route_entries == 1
        assert store.route([1.0, 0.0], ["clear-me"], topk=1) == [node_id]
        store.clear_route(node_id)
        assert store.stats().route_entries == 0
        assert store.route([1.0, 0.0], ["clear-me"], topk=1) == []
        store.save_checkpoint(ckpt)

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as restored:
        restored.load_checkpoint(ckpt)
        assert restored.stats().route_entries == 0
        assert restored.route([1.0, 0.0], ["clear-me"], topk=1) == []


def test_native_route_upsert_replaces_existing_entry(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        node_id = store.add_node("replace route", b"", ntok=1)
        old_runner = store.add_node("old lexical runner", b"", ntok=1)
        store.set_route(node_id, [1.0, 0.0], ["old"])
        store.set_route(old_runner, [0.1, 0.9949874], ["old"])
        assert store.stats().route_entries == 2
        assert store.route([1.0, 0.0], ["old"], topk=1) == [node_id]

        store.set_route(node_id, [0.0, 1.0], ["new"])
        assert store.stats().route_entries == 2
        assert store.route([1.0, 0.0], ["old"], topk=1) == [old_runner]
        assert store.route([0.0, 1.0], ["new"], topk=1) == [node_id]


def test_native_route_drops_non_finite_scores(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        bad = store.add_node("nan route", b"", ntok=1)
        good = store.add_node("finite route", b"", ntok=1)
        store.set_route(bad, [float("nan"), 0.0], [])
        store.set_route(good, [1.0, 0.0], [])

        assert store.route([1.0, 0.0], topk=2) == [good]


def test_native_memory_mutation_plan_via_ctypes(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        assert store.plan_memory_mutation(
            command="forget", has_query=False, target_count=0) == {
                "action": "no_op",
                "reason": "empty query",
                "target_count": 0,
                "apply_expire": False,
                "apply_revision": False,
                "write_replacement": False,
                "update_metadata": False,
            }
        forget = store.plan_memory_mutation(
            command="forget", has_query=True, target_count=2)
        assert forget["action"] == "expire_targets"
        assert forget["apply_expire"] is True
        assert forget["target_count"] == 2

        correct = store.plan_memory_mutation(
            command="correct", has_query=True, target_count=1,
            has_replacement=True)
        assert correct["action"] == "supersede_targets"
        assert correct["write_replacement"] is True
        assert correct["apply_revision"] is True

        standalone = store.plan_memory_mutation(
            command="correct", has_query=True, target_count=0,
            has_replacement=True)
        assert standalone["action"] == "write_replacement"
        assert standalone["write_replacement"] is True
        assert standalone["apply_revision"] is False

        metadata = store.plan_memory_mutation(
            command="update_metadata", has_query=True, target_count=3)
        assert metadata["action"] == "metadata_update"
        assert metadata["update_metadata"] is True


def test_native_librarian_plan_via_ctypes(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        plan = store.plan_librarian(
            foldable_turn_count=8, foldable_digest_count=6,
            turns_high=8, turns_fold=4,
            digests_high=6, digests_fold=3,
            era_enabled=True)
        assert plan["pending_jobs"] == 2
        assert plan["digest_source_count"] == 4
        assert plan["era_source_count"] == 3
        assert plan["deferred_backpressure"] is False

        disabled = store.plan_librarian(
            foldable_turn_count=0, foldable_digest_count=6,
            turns_high=8, turns_fold=4,
            digests_high=6, digests_fold=3,
            era_enabled=False)
        assert disabled["pending_jobs"] == 0
        assert disabled["era_source_count"] == 0

        below = store.plan_librarian(
            foldable_turn_count=15, foldable_digest_count=6,
            turns_high=8, turns_fold=4,
            digests_high=6, digests_fold=3,
            era_enabled=True, deferred_backpressure=True)
        assert below["pending_jobs"] == 0
        assert below["reason"] == "below deferred backpressure threshold"

        pressure = store.plan_librarian(
            foldable_turn_count=16, foldable_digest_count=6,
            turns_high=8, turns_fold=4,
            digests_high=6, digests_fold=3,
            era_enabled=True, deferred_backpressure=True)
        assert pressure["pending_jobs"] == 1
        assert pressure["digest_source_count"] == 4
        assert pressure["era_source_count"] == 0
        assert pressure["reason"] == "deferred turn backpressure"


def test_native_foldable_nodes_track_kind_active_and_no_fold(tmp_path):
    lib = build_native(tmp_path)
    ckpt = tmp_path / "foldable_ckpt"
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        turn0 = store.add_node("foldable turn 0", b"a", ntok=1)
        turn1 = store.add_node("foldable turn 1", b"b", ntok=1)
        turn2 = store.add_node("inactive turn 2", b"c", ntok=1)
        digest = store.add_node("foldable digest 0", b"d", ntok=1)
        store.set_metadata(turn0, {"kind": "turn", "active": True})
        store.set_metadata(turn1, {"kind": "turn", "active": True})
        store.set_metadata(turn2, {"kind": "turn", "active": True})
        store.set_metadata(digest, {"kind": "digest", "active": True})
        store.set_no_fold(turn1, True)
        store.set_active(turn2, False)

        assert store.foldable_nodes("turn") == (turn0,)
        assert store.foldable_nodes("digest") == (digest,)
        assert store.foldable_nodes("turn", excluded_node_ids=(turn0,)) == ()
        store.save_checkpoint(ckpt)

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as restored:
        restored.load_checkpoint(ckpt)
        assert restored.foldable_nodes("turn") == (turn0,)
        assert restored.foldable_nodes("digest") == (digest,)


def test_native_profile_requires_reseatable_position_law(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="AbsoluteKV_TC", num_layers=12,
            hidden_dim=768, vals_per_tok_layer=384, route_layer=0,
            payload_kind="gqa", num_kv_heads=4, head_dim=64,
            position_law="absolute_learned", state_kind="kv",
            graftability="fixed_position", remountable=False,
            composition="single_mount") as store:
        assert store.dialect_profile() == (
            "absolute_learned|kv|fixed_position|0|single_mount")

    with pytest.raises(ValueError, match="fixed-position"):
        NativeGraftStore(
            lib, model_type="AbsoluteKV_TC", num_layers=12,
            hidden_dim=768, vals_per_tok_layer=384, route_layer=0,
            payload_kind="gqa", num_kv_heads=4, head_dim=64,
            position_law="absolute_learned", state_kind="kv",
            graftability="fixed_position", remountable=True)

    with pytest.raises(ValueError, match="RoPE or relative"):
        NativeGraftStore(
            lib, model_type="UnknownPos_TC", num_layers=12,
            hidden_dim=768, vals_per_tok_layer=384, route_layer=0,
            payload_kind="gqa", num_kv_heads=4, head_dim=64,
            position_law="learned_bias", state_kind="kv",
            graftability="seat_remountable", remountable=True)


def test_native_store_gqa_dialect_via_ctypes(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="Qwen3_TC", num_layers=36,
            hidden_dim=2560, vals_per_tok_layer=2048, route_layer=0,
            payload_kind="gqa", num_kv_heads=8, head_dim=128) as store:
        assert store.dialect_id() == "Qwen3_TC:36x2560:g8x128"
        assert store.dialect_profile() == (
            "rope_full_kv|kv|seat_remountable|1|multi_mount")

        node_id = store.add_structured_node(
            "native GQA graft",
            {"k": np.zeros((2, 8, 3, 128), np.float16),
             "v": np.ones((2, 8, 3, 128), np.float16)},
            ntok=3)
        assert node_id == 0
        assert store.payload_stats(node_id).tensor_count == 2
        assert store.tensor_info(node_id, "k").shape == (2, 8, 3, 128)

        store.set_route(node_id, [0.0, 1.0], ["qwen", "gqa"])
        assert store.route([0.0, 0.9], ["gqa"], topk=1) == [node_id]


def test_native_gqa_raw_route_matches_python_law(tmp_path):
    def raw_score(query, key):
        h, _, dh = query.shape
        kk = np.repeat(key, h // key.shape[0], axis=0)
        sc = np.einsum("hqd,hkd->hqk", query, kk) / np.sqrt(dh)
        return float(np.abs(sc).max(axis=(1, 2)).mean())

    lib = build_native(tmp_path)
    q = np.asarray([
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 1.0], [1.0, -1.0]],
    ], dtype=np.float32)
    weak = np.asarray([[[0.1, 0.0]]], dtype=np.float32)
    good = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    best = np.asarray([[[1.0, 1.0]]], dtype=np.float32)

    with NativeGraftStore(
            lib, model_type="Qwen3_TC", num_layers=36,
            hidden_dim=2560, vals_per_tok_layer=2048, route_layer=0,
            payload_kind="gqa", num_kv_heads=1, head_dim=2) as store:
        n_good = store.add_node("good raw GQA route", b"", ntok=1)
        n_best = store.add_node("best variable raw GQA route", b"", ntok=1)
        n_lex = store.add_node("lexical raw GQA route", b"", ntok=1)
        store.set_route_key_list(n_good, [good], [])
        store.set_route_key_list(n_best, [weak, best], [])
        store.set_route_key_list(n_lex, [weak], ["a17"])

        scores = {
            n_good: raw_score(q, good),
            n_best: max(raw_score(q, weak), raw_score(q, best)),
        }
        expected = sorted(scores, key=lambda node_id: -scores[node_id])
        assert store.route_gqa(q, topk=2) == expected
        assert store.route_gqa(q, lexical_keys=["a17"], topk=3)[0] == n_lex


def test_native_gqa_cuda_route_requires_explicit_bank(tmp_path):
    lib = build_native(tmp_path)
    q = np.zeros((1, 1, 2), dtype=np.float32)

    with NativeGraftStore(
            lib, model_type="Qwen3_TC", num_layers=36,
            hidden_dim=2560, vals_per_tok_layer=2048, route_layer=0,
            payload_kind="gqa", num_kv_heads=1, head_dim=2) as store:
        with pytest.raises(RuntimeError, match="route bank is not configured"):
            store.route_gqa_cuda(q, topk=1)


class _DummyCudaRouteBank:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def route_topk(self, *_args, **_kwargs):
        raise AssertionError("stale CUDA route bank was used")


def test_native_gqa_route_mutation_clears_cuda_bank(tmp_path):
    lib = build_native(tmp_path)
    q = np.zeros((1, 1, 2), dtype=np.float32)

    with NativeGraftStore(
            lib, model_type="Qwen3_TC", num_layers=36,
            hidden_dim=2560, vals_per_tok_layer=2048, route_layer=0,
            payload_kind="gqa", num_kv_heads=1, head_dim=2) as store:
        node_id = store.add_node("cuda route stale", b"", ntok=1)
        dummy = _DummyCudaRouteBank()
        store._cuda_gqa_bank = dummy

        store.set_route_key_list(
            node_id, [np.zeros((1, 1, 2), dtype=np.float32)])

        assert dummy.closed
        assert store._cuda_gqa_bank is None
        with pytest.raises(RuntimeError, match="route bank is not configured"):
            store.route_gqa_cuda(q, topk=1)


def test_native_gqa_eligibility_mutation_clears_cuda_bank(tmp_path):
    lib = build_native(tmp_path)
    q = np.zeros((1, 1, 2), dtype=np.float32)

    with NativeGraftStore(
            lib, model_type="Qwen3_TC", num_layers=36,
            hidden_dim=2560, vals_per_tok_layer=2048, route_layer=0,
            payload_kind="gqa", num_kv_heads=1, head_dim=2) as store:
        node_id = store.add_node("cuda eligibility stale", b"", ntok=1)
        dummy = _DummyCudaRouteBank()
        store._cuda_gqa_bank = dummy

        store.set_active(node_id, False)

        assert dummy.closed
        assert store._cuda_gqa_bank is None
        with pytest.raises(RuntimeError, match="route bank is not configured"):
            store.route_gqa_cuda(q, topk=1)


def test_native_mla_cuda_route_requires_explicit_bank(tmp_path):
    """MLA P2 mirror of test_native_gqa_cuda_route_requires_explicit_bank."""
    lib = build_native(tmp_path)
    q = np.zeros(4, dtype=np.float32)

    with NativeGraftStore(
            lib, model_type="RouterMlaLifecycle", num_layers=1,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=0,
            payload_kind="mla", latent_rank=512, rope_dim=64) as store:
        with pytest.raises(RuntimeError, match="route bank is not configured"):
            store.route_mla_cuda(q, topk=1)


class _DummyCudaMlaRouteBank:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def route_topk(self, *_args, **_kwargs):
        raise AssertionError("stale CUDA MLA route bank was used")


def test_native_mla_route_mutation_clears_cuda_bank(tmp_path):
    """MLA P2 mirror of test_native_gqa_route_mutation_clears_cuda_bank."""
    lib = build_native(tmp_path)
    q = np.zeros(4, dtype=np.float32)

    with NativeGraftStore(
            lib, model_type="RouterMlaLifecycle", num_layers=1,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=0,
            payload_kind="mla", latent_rank=512, rope_dim=64) as store:
        node_id = store.add_node("cuda mla route stale", b"", ntok=1)
        dummy = _DummyCudaMlaRouteBank()
        store._cuda_mla_bank = dummy

        store.set_route(node_id, [1.0, 0.0, 0.0, 0.0])

        assert dummy.closed
        assert store._cuda_mla_bank is None
        with pytest.raises(RuntimeError, match="route bank is not configured"):
            store.route_mla_cuda(q, topk=1)


def test_native_mla_eligibility_mutation_clears_cuda_bank(tmp_path):
    """MLA P2 mirror of test_native_gqa_eligibility_mutation_clears_cuda_bank."""
    lib = build_native(tmp_path)
    q = np.zeros(4, dtype=np.float32)

    with NativeGraftStore(
            lib, model_type="RouterMlaLifecycle", num_layers=1,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=0,
            payload_kind="mla", latent_rank=512, rope_dim=64) as store:
        node_id = store.add_node("cuda mla eligibility stale", b"", ntok=1)
        dummy = _DummyCudaMlaRouteBank()
        store._cuda_mla_bank = dummy

        store.set_active(node_id, False)

        assert dummy.closed
        assert store._cuda_mla_bank is None
        with pytest.raises(RuntimeError, match="route bank is not configured"):
            store.route_mla_cuda(q, topk=1)


def test_native_gqa_segment_reduce_matches_python_law(tmp_path, monkeypatch):
    def raw_score(query, key):
        h, _, dh = query.shape
        kk = np.repeat(key, h // key.shape[0], axis=0)
        sc = np.einsum("hqd,hkd->hqk", query, kk) / np.sqrt(dh)
        return float(np.abs(sc).max(axis=(1, 2)).mean())

    monkeypatch.setenv("GRM_ROUTER_GQA_SEGMENT", "1")
    monkeypatch.setenv("GRM_ROUTER_GQA_ROWBLOCK", "1")
    lib = build_native(tmp_path, extra_cxxflags=("-fopenmp",))
    q = np.asarray([
        [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0],
         [0.0, 1.0, 0.0], [0.0, 0.5, 0.5]],
        [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5],
         [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
        [[0.0, 0.0, 1.0], [0.5, 0.0, 0.5],
         [0.0, 1.0, 0.0], [0.0, 0.5, 0.5]],
        [[1.0, 1.0, 0.0], [0.5, 0.0, 0.5],
         [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]],
    ], dtype=np.float32)
    weak = np.asarray([
        [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
        [[0.0, 0.0, 0.1], [0.1, 0.0, 0.0]],
    ], dtype=np.float32)
    good = np.asarray([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
    ], dtype=np.float32)
    best = np.asarray([
        [[1.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0]],
    ], dtype=np.float32)

    with NativeGraftStore(
            lib, model_type="Qwen3.5_TC", num_layers=24,
            hidden_dim=2048, vals_per_tok_layer=1024, route_layer=3,
            payload_kind="gqa", num_kv_heads=2, head_dim=3) as store:
        n_good = store.add_node("good segment GQA route", b"", ntok=1)
        n_best = store.add_node("best segment GQA route", b"", ntok=1)
        n_lex = store.add_node("lexical segment GQA route", b"", ntok=1)
        store.set_route_key_list(n_good, [good], [])
        store.set_route_key_list(n_best, [weak, best], [])
        store.set_route_key_list(n_lex, [weak], ["a17"])

        scores = {
            n_good: raw_score(q, good),
            n_best: max(raw_score(q, weak), raw_score(q, best)),
        }
        expected = sorted(scores, key=lambda node_id: -scores[node_id])
        assert store.route_gqa(q, topk=2) == expected
        assert store.route_gqa(q, lexical_keys=["a17"], topk=3)[0] == n_lex


def test_native_gqa_single_key_segment_matches_python_law(tmp_path, monkeypatch):
    def raw_score(query, key):
        h, _, dh = query.shape
        kk = np.repeat(key, h // key.shape[0], axis=0)
        sc = np.einsum("hqd,hkd->hqk", query, kk) / np.sqrt(dh)
        return float(np.abs(sc).max(axis=(1, 2)).mean())

    monkeypatch.setenv("GRM_ROUTER_GQA_SEGMENT", "1")
    monkeypatch.setenv("GRM_ROUTER_GQA_FUSED_SEGMENT", "1")
    lib = build_native(tmp_path, extra_cxxflags=("-fopenmp",))
    q = np.asarray([
        [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0],
         [0.0, 1.0, 0.0], [0.0, 0.5, 0.5]],
        [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5],
         [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
        [[0.0, 0.0, 1.0], [0.5, 0.0, 0.5],
         [0.0, 1.0, 0.0], [0.0, 0.5, 0.5]],
        [[1.0, 1.0, 0.0], [0.5, 0.0, 0.5],
         [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]],
    ], dtype=np.float32)
    weak = np.asarray([
        [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
        [[0.0, 0.0, 0.1], [0.1, 0.0, 0.0]],
    ], dtype=np.float32)
    good = np.asarray([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
    ], dtype=np.float32)
    best = np.asarray([
        [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
    ], dtype=np.float32)

    with NativeGraftStore(
            lib, model_type="Qwen3.5_TC", num_layers=24,
            hidden_dim=2048, vals_per_tok_layer=1024, route_layer=3,
            payload_kind="gqa", num_kv_heads=2, head_dim=3) as store:
        n_weak = store.add_node("weak fused segment GQA route", b"", ntok=1)
        n_good = store.add_node("good fused segment GQA route", b"", ntok=1)
        n_best = store.add_node("best fused segment GQA route", b"", ntok=1)
        store.set_route_key_list(n_weak, [weak], [])
        store.set_route_key_list(n_good, [good], [])
        store.set_route_key_list(n_best, [best], [])

        scores = {
            n_weak: raw_score(q, weak),
            n_good: raw_score(q, good),
            n_best: raw_score(q, best),
        }
        expected = sorted(scores, key=lambda node_id: -scores[node_id])
        assert store.route_gqa(q, topk=3) == expected


def test_native_gqa_rowblock_segment_matches_python_law(tmp_path, monkeypatch):
    def raw_score(query, key):
        h, _, dh = query.shape
        kk = np.repeat(key, h // key.shape[0], axis=0)
        sc = np.einsum("hqd,hkd->hqk", query, kk) / np.sqrt(dh)
        return float(np.abs(sc).max(axis=(1, 2)).mean())

    monkeypatch.setenv("GRM_ROUTER_GQA_SEGMENT", "1")
    monkeypatch.setenv("GRM_ROUTER_GQA_ROWBLOCK", "1")
    lib = build_native(tmp_path, extra_cxxflags=("-fopenmp",))
    q = np.asarray([
        [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0],
         [0.0, 1.0, 0.0], [0.0, 0.5, 0.5]],
        [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5],
         [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
        [[0.0, 0.0, 1.0], [0.5, 0.0, 0.5],
         [0.0, 1.0, 0.0], [0.0, 0.5, 0.5]],
        [[1.0, 1.0, 0.0], [0.5, 0.0, 0.5],
         [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]],
    ], dtype=np.float32)
    weak = np.asarray([
        [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
        [[0.0, 0.0, 0.1], [0.1, 0.0, 0.0]],
    ], dtype=np.float32)
    good = np.asarray([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
    ], dtype=np.float32)
    best = np.asarray([
        [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
    ], dtype=np.float32)

    with NativeGraftStore(
            lib, model_type="Qwen3.5_TC", num_layers=24,
            hidden_dim=2048, vals_per_tok_layer=1024, route_layer=3,
            payload_kind="gqa", num_kv_heads=2, head_dim=3) as store:
        n_weak = store.add_node("weak rowblock GQA route", b"", ntok=1)
        n_good = store.add_node("good rowblock GQA route", b"", ntok=1)
        n_best = store.add_node("best rowblock GQA route", b"", ntok=1)
        store.set_route_key_list(n_weak, [weak], [])
        store.set_route_key_list(n_good, [good], [])
        store.set_route_key_list(n_best, [best], [])

        scores = {
            n_weak: raw_score(q, weak),
            n_good: raw_score(q, good),
            n_best: raw_score(q, best),
        }
        expected = sorted(scores, key=lambda node_id: -scores[node_id])
        assert store.route_gqa(q, topk=3) == expected


@pytest.mark.parametrize(
    "mode", [
        "default", "avx2", "fma", "pair2", "banked", "unroll8",
        "transposed", "blas"])
def test_native_gqa_qt4_even_repeat_route_matches_python_law(
        tmp_path, monkeypatch, mode):
    def raw_score(query, key):
        h, _, dh = query.shape
        kk = np.repeat(key, h // key.shape[0], axis=0)
        sc = np.einsum("hqd,hkd->hqk", query, kk) / np.sqrt(dh)
        return float(np.abs(sc).max(axis=(1, 2)).mean())

    extra_cxxflags = ()
    extra_ldflags = ()
    if mode == "avx2":
        monkeypatch.setenv("GRM_ROUTER_GQA_AVX2", "1")
    elif mode == "fma":
        monkeypatch.setenv("GRM_ROUTER_GQA_AVX2", "1")
        monkeypatch.setenv("GRM_ROUTER_GQA_FMA", "1")
    elif mode == "pair2":
        monkeypatch.setenv("GRM_ROUTER_GQA_PAIR2_AVX2", "1")
    elif mode == "banked":
        monkeypatch.setenv("GRM_ROUTER_GQA_BANKED", "1")
    elif mode == "unroll8":
        monkeypatch.setenv("GRM_ROUTER_GQA_QT4_UNROLL8", "1")
    elif mode == "transposed":
        monkeypatch.setenv("GRM_ROUTER_GQA_TRANSPOSED", "1")
    elif mode == "blas":
        monkeypatch.setenv("GRM_ROUTER_GQA_SEGMENT", "1")
        monkeypatch.setenv("GRM_ROUTER_GQA_BLAS", "1")
        extra_cxxflags = ("-DGRM_ENABLE_CBLAS",)
        extra_ldflags = (
            os.environ.get("GRM_BLAS_LIB", "/lib/x86_64-linux-gnu/libblas.so.3"),
        )
    lib = build_native(
        tmp_path, extra_cxxflags=extra_cxxflags, extra_ldflags=extra_ldflags)
    q = np.asarray([
        [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, 0.5]],
        [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
        [[0.0, 0.0, 1.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.0, 0.5, 0.5]],
        [[1.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]],
    ], dtype=np.float32)
    weak = np.asarray([
        [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
        [[0.0, 0.0, 0.1], [0.1, 0.0, 0.0]],
    ], dtype=np.float32)
    good = np.asarray([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
    ], dtype=np.float32)

    with NativeGraftStore(
            lib, model_type="Qwen3.5_TC", num_layers=24,
            hidden_dim=2048, vals_per_tok_layer=1024, route_layer=3,
            payload_kind="gqa", num_kv_heads=2, head_dim=3) as store:
        n_weak = store.add_node("weak qt4 GQA route", b"", ntok=1)
        n_good = store.add_node("good qt4 GQA route", b"", ntok=1)
        store.set_route_key_list(n_weak, [weak], [])
        store.set_route_key_list(n_good, [good], [])

        scores = {
            n_weak: raw_score(q, weak),
            n_good: raw_score(q, good),
        }
        expected = sorted(scores, key=lambda node_id: -scores[node_id])
        assert store.route_gqa(q, topk=2) == expected


def test_native_gqa_topk_selection_preserves_tie_order(tmp_path):
    lib = build_native(tmp_path)
    q = np.asarray([[[1.0, 0.0]]], dtype=np.float32)
    key = np.asarray([[[1.0, 0.0]]], dtype=np.float32)

    with NativeGraftStore(
            lib, model_type="Qwen3_TC", num_layers=36,
            hidden_dim=2560, vals_per_tok_layer=2048, route_layer=0,
            payload_kind="gqa", num_kv_heads=1, head_dim=2) as store:
        node_ids = []
        for i in range(16):
            node_id = store.add_node(f"tie GQA route {i}", b"", ntok=1)
            store.set_route_key_list(node_id, [key], [])
            node_ids.append(node_id)

        assert store.route_gqa(q, topk=5) == node_ids[:5]


def test_native_gqa_route_drops_non_finite_scores(tmp_path):
    lib = build_native(tmp_path)
    q = np.asarray([[[1.0, 0.0]]], dtype=np.float32)
    bad_key = np.asarray([[[np.nan, 0.0]]], dtype=np.float32)
    good_key = np.asarray([[[1.0, 0.0]]], dtype=np.float32)

    with NativeGraftStore(
            lib, model_type="Qwen3_TC", num_layers=36,
            hidden_dim=2560, vals_per_tok_layer=2048, route_layer=0,
            payload_kind="gqa", num_kv_heads=1, head_dim=2) as store:
        bad = store.add_node("nan raw GQA route", b"", ntok=1)
        good = store.add_node("finite raw GQA route", b"", ntok=1)
        store.set_route_key_list(bad, [bad_key], [])
        store.set_route_key_list(good, [good_key], [])

        assert store.route_gqa(q, topk=2) == [good]


def test_native_gqa_key_bank_rebuilds_after_route_update(tmp_path):
    lib = build_native(tmp_path)
    q = np.asarray([[[1.0, 0.0]]], dtype=np.float32)
    weak = np.asarray([[[0.0, 1.0]]], dtype=np.float32)
    good = np.asarray([[[1.0, 0.0]]], dtype=np.float32)
    better = np.asarray([[[2.0, 0.0]]], dtype=np.float32)

    with NativeGraftStore(
            lib, model_type="Qwen3_TC", num_layers=36,
            hidden_dim=2560, vals_per_tok_layer=2048, route_layer=0,
            payload_kind="gqa", num_kv_heads=1, head_dim=2) as store:
        n0 = store.add_node("stale GQA arena route", b"", ntok=1)
        n1 = store.add_node("fresh GQA arena route", b"", ntok=1)
        store.set_route_key_list(n0, [weak], [])
        store.set_route_key_list(n1, [good], [])

        assert store.route_gqa(q, topk=2) == [n1, n0]

        store.set_route_key_list(n0, [better], [])
        assert store.route_gqa(q, topk=2) == [n0, n1]


def test_native_route_list_rejects_terminal_offset_mismatch(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        node_id = store.add_node("bad route offsets", b"", ntok=1)
        values = (ctypes.c_float * 2)(1.0, 0.0)
        offsets = (ctypes.c_uint64 * 3)(0, 1, 1)

        rc = store._lib.grm_store_set_route_list(
            store._handle, int(node_id), values, 2, offsets, 2, b"")

        assert rc != 0


def test_native_store_slices_host_tensor_payload(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        tensor = np.arange(2 * 6 * 3, dtype=np.float32).reshape(2, 6, 3)
        node_id = store.add_structured_node(
            "slice me", {"latent": tensor}, ntok=6)

        sliced = store.slice_tensor(node_id, "latent", axis=1,
                                    start=2, length=3)

        assert sliced.shape == (2, 3, 3)
        np.testing.assert_array_equal(sliced, tensor[:, 2:5, :])
        with pytest.raises(RuntimeError, match="slice range out of bounds"):
            store.slice_tensor(node_id, "latent", axis=1, start=5, length=2)


def test_native_route_uses_python_fractional_lexical_bonus(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        strong_semantic = store.add_node("semantic winner", b"", ntok=1)
        partial_lexical = store.add_node("partial lexical", b"", ntok=1)

        store.set_route(strong_semantic, [0.95, 0.3122499], [])
        store.set_route(partial_lexical, [0.40, 0.9165151], ["a17"])

        # Query has two identifier tokens. Python's calibrated lexical channel
        # adds 1/2 for this partial hit, not +1 per hit.
        assert store.route([1.0, 0.0], ["a17", "beta"], topk=2) == [
            strong_semantic, partial_lexical]


def test_native_route_skips_inactive_entries(tmp_path):
    lib = build_native(tmp_path)
    ckpt = tmp_path / "active_ckpt"
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        n0 = store.add_node("retired winner", b"", ntok=1)
        n1 = store.add_node("live runner-up", b"", ntok=1)
        store.set_route(n0, [1.0, 0.0], ["retired"])
        store.set_route(n1, [0.95, 0.3122499], ["live"])

        assert store.route([1.0, 0.0], topk=2) == [n0, n1]
        store.set_active(n0, False)
        assert store.route([1.0, 0.0], topk=2) == [n1]
        store.set_active(n0, True)
        assert store.route([1.0, 0.0], topk=1) == [n0]

        store.set_active(n0, False)
        store.save_checkpoint(ckpt)

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as restored:
        restored.load_checkpoint(ckpt)
        assert restored.stats().route_entries == 2
        assert restored.route([1.0, 0.0], topk=2) == [n1]
        restored.set_active(n0, True)
        assert restored.route([1.0, 0.0], topk=1) == [n0]


def test_native_checkpoint_preserves_route_keys_and_lexical_index(tmp_path):
    lib = build_native(tmp_path)
    ckpt = tmp_path / "route_ckpt"
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        multi = store.add_node("multi era route", b"", ntok=1)
        runner = store.add_node("single runner route", b"", ntok=1)
        cold = store.add_node("not routed", b"", ntok=1)

        store.set_route_keys(multi, [[1.0, 0.0], [0.0, 1.0]],
                             ["era-beta", "project"])
        store.set_route(runner, [0.95, 0.3122499], ["runner"])
        store.save_checkpoint(ckpt)

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as restored:
        restored.load_checkpoint(ckpt)
        assert restored.stats().nodes == 3
        assert restored.stats().route_entries == 2
        assert restored.route([0.0, 1.0], topk=1) == [multi]
        assert restored.route([0.0, 0.1], ["era-beta"], topk=1) == [multi]
        assert cold not in restored.route([1.0, 0.0], topk=3)


def test_native_clear_payload_preserves_metadata_route_checkpoint(tmp_path):
    lib = build_native(tmp_path)
    ckpt = tmp_path / "metadata_only_ckpt"
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        node_id = store.add_structured_node(
            "cold metadata route",
            {"payload_id": np.array([7], dtype=np.int64)},
            ntok=1)
        store.set_route(node_id, [1.0, 0.0], ["cold-route"])
        store.set_metadata(node_id, {"kind": "doc", "active": True})
        assert store.payload_stats(node_id).tensor_count == 1

        store.clear_payload(node_id)
        assert store.payload_stats(node_id).tensor_count == 0
        store.save_checkpoint(ckpt)

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as restored:
        restored.load_checkpoint(ckpt)
        assert restored.payload_stats(node_id).tensor_count == 0
        assert restored.metadata(node_id)["kind"] == "doc"
        assert restored.route([1.0, 0.0], ["cold-route"], topk=1) == [node_id]


def test_native_metadata_active_updates_route_index(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        n0 = store.add_node("metadata-controlled route", b"", ntok=1)
        store.set_route(n0, [1.0, 0.0], [])

        store.set_metadata(n0, {"active": False, "kind": "fact"})
        assert store.route([1.0, 0.0], topk=1) == []
        store.set_metadata(n0, {"active": True, "kind": "fact"})
        assert store.route([1.0, 0.0], topk=1) == [n0]


def test_native_memory_command_parser(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        assert store.parse_memory_command(
            "remember permanently: User prefers amber dashboards") == {
                "action": "remember",
                "body": "User prefers amber dashboards",
                "durability": "permanent",
                "scope": "user",
                "kind": "fact",
                "flush_immediately": True,
            }
        assert store.parse_memory_command(
            "this is temporary: staging room is Cedar") == {
                "action": "remember",
                "body": "staging room is Cedar",
                "durability": "volatile",
                "mutability": "ephemeral",
                "scope": "session",
                "kind": "task_state",
                "flush_immediately": False,
            }
        assert store.parse_memory_command("forget: stale marker") == {
            "action": "forget",
            "query": "stale marker",
            "flush_immediately": False,
        }
        assert store.parse_memory_command(
            "correct memory: stale marker => fresh marker") == {
                "action": "correct",
                "query": "stale marker",
                "replacement": "fresh marker",
                "flush_immediately": False,
            }
        assert store.parse_memory_command("update memory: missing arrow") == {
            "action": "review",
            "body": "missing arrow",
            "reason": "correction missing => separator",
            "flush_immediately": False,
        }
        assert store.parse_memory_command("approve review 5") == {
            "action": "approve_review",
            "review_id": 5,
            "flush_immediately": False,
        }
        assert store.parse_memory_command(
            "reject review 6 reason Not reliable enough") == {
                "action": "reject_review",
                "reason": "Not reliable enough",
                "review_id": 6,
                "flush_immediately": False,
            }
        assert store.parse_memory_command(
            "edit review 7 text Corrected Review Body") == {
                "action": "edit_review",
                "body": "Corrected Review Body",
                "reason": "memory command edit",
                "review_id": 7,
                "flush_immediately": False,
            }
        assert store.parse_memory_command(
            "edit review 7 text new text goes here")["body"] == (
                "new text goes here")
        with pytest.raises(RuntimeError, match="edit review text is missing"):
            store.parse_memory_command("edit review 5,text new text goes here")
        assert store.parse_memory_command(
            "change review 8 scope user durability permanent mutability stable") == {
                "action": "change_review_scope",
                "durability": "permanent",
                "mutability": "stable",
                "scope": "user",
                "review_id": 8,
                "flush_immediately": False,
            }
        assert store.parse_memory_command("flush memory now") == {
            "action": "flush",
            "flush_immediately": False,
        }
        assert store.parse_memory_command(
            "cull graft 12 into sections max tokens 128") == {
                "action": "cull_graft",
                "boundary": "section",
                "node_id": 12,
                "max_tokens": 128,
                "flush_immediately": False,
            }
        assert store.parse_memory_command("split graft 3 max tokens 64") == {
            "action": "cull_graft",
            "node_id": 3,
            "max_tokens": 64,
            "flush_immediately": False,
        }
        assert store.parse_memory_command(
            "select graft 4 span 2 7 label Key Passage") == {
                "action": "select_graft_span",
                "body": "Key Passage",
                "node_id": 4,
                "span_start": 2,
                "span_end": 7,
                "flush_immediately": False,
            }
        assert store.parse_memory_command("pin memory: live target") == {
            "action": "update_memory_metadata",
            "command": "pin_memory",
            "query": "live target",
            "metadata_key": "pinned",
            "metadata_value": "true",
            "flush_immediately": False,
        }
        assert store.parse_memory_command(
            "mark memory mutable: project state") == {
                "action": "update_memory_metadata",
                "command": "mark_mutable",
                "query": "project state",
                "metadata_key": "mutability",
                "metadata_value": "mutable",
                "flush_immediately": False,
            }
        assert store.parse_memory_command("show memory about: live target") == {
            "action": "show_memory",
            "query": "live target",
            "flush_immediately": False,
        }
        assert store.parse_memory_command(
            "why do you remember: live target") == {
                "action": "why_memory",
                "query": "live target",
                "flush_immediately": False,
            }
        assert store.parse_memory_command("switch to volatile mode") == {
            "action": "set_durability_mode",
            "durability_mode": "volatile",
            "flush_immediately": False,
        }
        assert store.parse_memory_command("switch to session-safe mode") == {
            "action": "set_durability_mode",
            "durability_mode": "session_safe",
            "flush_immediately": False,
        }
        assert store.parse_memory_command("switch to project safe mode") == {
            "action": "set_durability_mode",
            "durability_mode": "project_safe",
            "flush_immediately": False,
        }
        with pytest.raises(RuntimeError, match="unknown memory command"):
            store.parse_memory_command("remember maybe someday")


def test_native_extraction_policy_planner(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        assert store.plan_extraction_policy(
            action="write_direct", write_intent="observed",
            confidence=0.4, write_direct_threshold=0.95) == {
                "action": "review_candidate",
                "reason": "confidence below direct-write threshold",
            }
        assert store.plan_extraction_policy(
            action="write_direct", write_intent="observed",
            confidence=0.99, write_direct_threshold=0.95,
            conflict_count=1) == {
                "action": "review_candidate",
                "reason": "conflicts with active memory",
            }
        assert store.plan_extraction_policy(
            action="write_direct", write_intent="user_asserted",
            confidence=1.0, write_direct_threshold=0.95,
            conflict_count=1) == {"action": "supersede_existing"}
        assert store.plan_extraction_policy(
            action="write_direct", write_intent="observed",
            confidence=1.0, write_direct_threshold=0.95,
            equivalent_count=1) == {"action": "reinforce_existing"}
        assert store.plan_extraction_policy(
            action="expire", write_intent="inferred",
            confidence=1.0, write_direct_threshold=0.95,
            expire_target_count=1) == {
                "action": "review_candidate",
                "reason": "expire action requires authoritative intent",
            }
        assert store.plan_extraction_policy(
            action="expire", write_intent="user_asserted",
            confidence=1.0, write_direct_threshold=0.95,
            expire_target_count=1) == {"action": "expire"}


def test_native_reinforcement_planner(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        assert store.plan_reinforcement(
            old_write_intent="observed", new_write_intent="inferred",
            old_confidence=0.7, new_confidence=0.4,
            old_reinforcement_count=2) == {
                "write_intent": "observed",
                "confidence": 0.7,
                "reinforcement_count": 3,
            }
        assert store.plan_reinforcement(
            old_write_intent="observed", new_write_intent="user_asserted",
            old_confidence=0.7, new_confidence=0.95,
            old_reinforcement_count=0) == {
                "write_intent": "user_asserted",
                "confidence": 0.95,
                "reinforcement_count": 1,
            }


def test_native_remember_flush_planner(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        assert store.plan_remember_flush(
            durability_mode="session_safe", durability="project",
            scope="project") == {"force_flush": False}
        assert store.plan_remember_flush(
            durability_mode="project_safe", durability="session",
            scope="session") == {"force_flush": False}
        assert store.plan_remember_flush(
            durability_mode="project_safe", durability="project",
            scope="project") == {
                "force_flush": True,
                "reason": "project-safe mode requires project memory flush",
            }
        assert store.plan_remember_flush(
            durability_mode="volatile", durability="volatile",
            scope="session", flush_immediately=True) == {
                "force_flush": True,
                "reason": "command requested immediate flush",
            }


def test_native_runtime_event_planner(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        assert store.plan_runtime_event(
            event="memory_command", action="show_memory",
            autosave_enabled=True) == {
                "flush": False,
                "page": False,
                "read_only": True,
                "reason": "read-only runtime event",
            }
        assert store.plan_runtime_event(
            event="memory_command", action="flush",
            autosave_enabled=False) == {
                "flush": True,
                "page": True,
                "read_only": False,
                "reason": "runtime event requested flush",
            }
        assert store.plan_runtime_event(
            event="review", action="approve_review",
            autosave_enabled=True) == {
                "flush": True,
                "page": True,
                "read_only": False,
                "reason": "autosave enabled",
            }
        assert store.plan_runtime_event(
            event="extraction", action="write_direct",
            autosave_enabled=False) == {
                "flush": False,
                "page": True,
                "read_only": False,
            }


def test_native_durability_mode_planner(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        assert store.plan_durability_mode(
            requested_mode="volatile",
            current_mode="session_safe",
            old_wal_enabled=True) == {
                "durability_mode": "volatile",
                "target_wal_enabled": False,
                "final_wal_enabled": False,
                "append_config_before": True,
                "append_config_after": False,
            }
        assert store.plan_durability_mode(
            requested_mode="session safe",
            current_mode="volatile",
            old_wal_enabled=False) == {
                "durability_mode": "session_safe",
                "target_wal_enabled": True,
                "final_wal_enabled": True,
                "append_config_before": False,
                "append_config_after": True,
            }
        assert store.plan_durability_mode(
            requested_mode="project-safe",
            current_mode="volatile_fast",
            old_wal_enabled=False,
            wal_enabled_override=True) == {
                "durability_mode": "project_safe",
                "target_wal_enabled": True,
                "final_wal_enabled": False,
                "append_config_before": False,
                "append_config_after": False,
            }
        with pytest.raises(RuntimeError, match="unknown durability mode"):
            store.plan_durability_mode(
                requested_mode="made-up",
                current_mode="session_safe",
                old_wal_enabled=True)


def test_native_metadata_update_planner(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        assert store.plan_metadata_update(
            command="pin_memory", metadata_key="pinned",
            metadata_value="true") == {"metadata": {"pinned": True}}
        assert store.plan_metadata_update(
            command="unpin_memory", metadata_key="pinned",
            metadata_value="false") == {"metadata": {"pinned": False}}
        assert store.plan_metadata_update(
            command="mark_mutable", metadata_key="mutability",
            metadata_value="mutable") == {"metadata": {"mutability": "mutable"}}
        assert store.plan_metadata_update(
            command="mark_stable", metadata_key="mutability",
            metadata_value="stable") == {"metadata": {"mutability": "stable"}}
        with pytest.raises(RuntimeError, match="boolean value"):
            store.plan_metadata_update(
                command="pin_memory", metadata_key="pinned",
                metadata_value="maybe")
        with pytest.raises(RuntimeError, match="mutability"):
            store.plan_metadata_update(
                command="mark_mutable", metadata_key="mutability",
                metadata_value="temporary")


def test_native_review_transition_planner(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        assert store.plan_review_transition(
            command="edit_review", status="pending") == {
                "action": "edit_review"}
        assert store.plan_review_transition(
            command="change_review_scope", status="pending") == {
                "action": "change_review_scope"}
        assert store.plan_review_transition(
            command="edit_review", status="approved") == {
                "action": "error",
                "reason": "approved review items cannot be edited",
            }
        assert store.plan_review_transition(
            command="reject_review", status="approved") == {
                "action": "error",
                "reason": "approved review items cannot be rejected",
            }
        assert store.plan_review_transition(
            command="approve_review", status="rejected") == {
                "action": "error",
                "reason": "rejected review items cannot be approved",
            }
        assert store.plan_review_transition(
            command="approve_review", status="approved",
            has_approved_node_id=True) == {"action": "return_existing"}


def test_native_cull_span_planner(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        assert store.plan_cull_spans(ntok=5, max_tokens=2) == {
            "spans": [(0, 2), (2, 4), (4, 5)],
            "retire_parent": True,
        }
        assert store.plan_cull_spans(
            ntok=6, spans=[(0, 2), (4, 6)],
            retire_parent=False) == {
                "spans": [(0, 2), (4, 6)],
                "retire_parent": False,
            }
        with pytest.raises(RuntimeError, match="cover every token"):
            store.plan_cull_spans(
                ntok=6, spans=[(0, 2), (4, 6)],
                retire_parent=True)
        with pytest.raises(RuntimeError, match="max_tokens must be positive"):
            store.plan_cull_spans(ntok=6, max_tokens=0)
        with pytest.raises(RuntimeError, match="requires max_tokens or spans"):
            store.plan_cull_spans(ntok=6)


def test_native_apply_revision_links_edges_and_retires_routes(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        old0 = store.add_node("old fact zero", b"", ntok=1)
        old1 = store.add_node("old fact one", b"", ntok=1)
        new = store.add_node("new fact", b"", ntok=1)
        store.set_route(old0, [1.0, 0.0], ["fact"])
        store.set_route(old1, [0.98, 0.1989975], ["fact"])
        store.set_route(new, [0.95, 0.3122499], ["fact"])

        assert store.route([1.0, 0.0], ["fact"], topk=3) == [
            old0, old1, new]
        store.apply_revision(new, [old0, old1])

        assert store.route([1.0, 0.0], ["fact"], topk=3) == [new]
        assert store.graph_edges(new).supersedes == (old0, old1)
        assert store.graph_edges(old0).superseded_by == (new,)
        assert store.graph_edges(old1).superseded_by == (new,)


def test_native_apply_expire_retires_routes(tmp_path):
    lib = build_native(tmp_path)
    ckpt = tmp_path / "expire_ckpt"
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        stale0 = store.add_node("stale fact zero", b"", ntok=1)
        stale1 = store.add_node("stale fact one", b"", ntok=1)
        live = store.add_node("live fact", b"", ntok=1)
        store.set_metadata(stale0, {"active": True})
        store.set_metadata(stale1, {"active": True})
        store.set_metadata(live, {"active": True})
        store.set_route(stale0, [1.0, 0.0], ["fact"])
        store.set_route(stale1, [0.98, 0.1989975], ["fact"])
        store.set_route(live, [0.95, 0.3122499], ["fact"])

        assert store.route([1.0, 0.0], ["fact"], topk=3) == [
            stale0, stale1, live]
        store.apply_expire([stale0, stale1, stale0])

        assert store.route([1.0, 0.0], ["fact"], topk=3) == [live]
        assert store.filter_active_nodes((stale0, stale1, live)) == (live,)
        store.save_checkpoint(ckpt)

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as restored:
        restored.load_checkpoint(ckpt)
        assert restored.filter_active_nodes((stale0, stale1, live)) == (live,)
        assert restored.route([1.0, 0.0], ["fact"], topk=3) == [live]


def test_native_active_node_filter_preserves_order_and_dedupes(tmp_path):
    lib = build_native(tmp_path)
    ckpt = tmp_path / "active_filter_ckpt"
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        active0 = store.add_node("active zero", b"", ntok=1)
        inactive = store.add_node("inactive one", b"", ntok=1)
        active1 = store.add_node("active two", b"", ntok=1)
        store.set_metadata(active0, {"active": True})
        store.set_metadata(inactive, {"active": False})
        store.set_metadata(active1, {"active": True})

        assert store.filter_active_nodes(
            (active0, inactive, active0, 9999, active1)) == (
                active0, active1)
        store.save_checkpoint(ckpt)

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as restored:
        restored.load_checkpoint(ckpt)
        assert restored.filter_active_nodes(
            (inactive, active1, active0, active1)) == (
                active1, active0)


def test_native_active_text_matches_active_case_insensitive_text(tmp_path):
    lib = build_native(tmp_path)
    ckpt = tmp_path / "active_text_ckpt"
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        live0 = store.add_node("Alpha TARGET shard zero", b"", ntok=4)
        inactive = store.add_node("alpha target shard retired", b"", ntok=4)
        live1 = store.add_node("Another target shard one", b"", ntok=4)
        other = store.add_node("unrelated memory", b"", ntok=2)
        store.set_metadata(live0, {"active": True})
        store.set_metadata(inactive, {"active": False})
        store.set_metadata(live1, {"active": True})
        store.set_metadata(other, {"active": True})

        assert store.active_text_matches("TARGET shard") == (live0, live1)
        assert store.active_text_matches("  target SHARD  ") == (live0, live1)
        assert store.active_text_matches("") == ()
        store.save_checkpoint(ckpt)

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as restored:
        restored.load_checkpoint(ckpt)
        assert restored.active_text_matches("target shard") == (live0, live1)


def test_native_source_closure_walks_folded_graph(tmp_path):
    lib = build_native(tmp_path)
    ckpt = tmp_path / "source_closure_ckpt"
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        leaf0 = store.add_node("leaf zero", b"", ntok=1)
        leaf1 = store.add_node("leaf one", b"", ntok=1)
        digest = store.add_node("digest", b"", ntok=2)
        era = store.add_node("era", b"", ntok=3)

        store.set_graph_edges(digest, source_grafts=(leaf0, leaf1))
        store.set_graph_edges(era, source_grafts=(digest, leaf1))
        store.set_graph_edges(leaf0, source_grafts=(era,))

        assert store.source_closure((era,), max_depth=1) == (digest, leaf1)
        assert store.source_closure((era,), max_depth=3) == (
            digest, leaf0, leaf1)
        assert store.source_closure(
            (era,), max_depth=3, include_roots=True) == (
                era, digest, leaf0, leaf1)
        assert store.source_closure((digest, era), max_depth=3) == (
            leaf0, leaf1)
        store.save_checkpoint(ckpt)

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as restored:
        restored.load_checkpoint(ckpt)
        assert restored.source_closure((era,), max_depth=3) == (
            digest, leaf0, leaf1)


def test_native_route_filters_metadata_fields(tmp_path):
    lib = build_native(tmp_path)
    ckpt = tmp_path / "filter_ckpt"
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        project_fact = store.add_node("project fact", b"", ntok=1)
        user_fact = store.add_node("user fact", b"", ntok=1)
        task_state = store.add_node("project task state", b"", ntok=1)

        store.set_metadata(project_fact, {
            "active": True, "kind": "fact", "scope": "project",
            "durability": "project", "mutability": "stable"})
        store.set_metadata(user_fact, {
            "active": True, "kind": "fact", "scope": "user",
            "durability": "permanent", "mutability": "stable"})
        store.set_metadata(task_state, {
            "active": True, "kind": "task_state", "scope": "project",
            "durability": "session", "mutability": "ephemeral"})
        store.set_route(project_fact, [1.0, 0.0], ["same-code"])
        store.set_route(user_fact, [0.95, 0.3122499], ["same-code"])
        store.set_route(task_state, [0.9, 0.4358899], ["same-code"])

        assert store.route([1.0, 0.0], ["same-code"], topk=3,
                           kinds=("fact",),
                           scopes=("project",)) == [project_fact]
        assert store.route([1.0, 0.0], ["same-code"], topk=3,
                           kinds=("fact",),
                           scopes=("project", "user")) == [
            project_fact, user_fact]
        assert store.route([1.0, 0.0], ["same-code"], topk=3,
                           durabilities=("session",),
                           mutabilities=("ephemeral",)) == [task_state]

        store.save_checkpoint(ckpt)

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as restored:
        restored.load_checkpoint(ckpt)
        assert restored.stats().route_entries == 3
        assert restored.route([1.0, 0.0], ["same-code"], topk=3,
                              kinds=("task_state",),
                              scopes=("project",)) == [task_state]


def test_native_route_filter_bitmasks_fall_back_for_unknown_values(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        custom = store.add_node("custom metadata route", b"", ntok=1)
        known = store.add_node("known metadata route", b"", ntok=1)
        store.set_metadata(custom, {
            "active": True, "kind": "research_note",
            "scope": "lab", "durability": "archive",
            "mutability": "append_only"})
        store.set_metadata(known, {
            "active": True, "kind": "fact", "scope": "project",
            "durability": "project", "mutability": "stable"})
        store.set_route(custom, [1.0, 0.0], ["same-code"])
        store.set_route(known, [0.95, 0.3122499], ["same-code"])

        assert store.route([1.0, 0.0], ["same-code"], topk=3,
                           kinds=("fact",)) == [known]
        assert store.route([1.0, 0.0], ["same-code"], topk=3,
                           kinds=("research_note",),
                           scopes=("lab",),
                           durabilities=("archive",),
                           mutabilities=("append_only",)) == [custom]


def test_native_mla_gemv_arena_matches_reference_scan(tmp_path):
    lib = build_native(tmp_path)
    rng = np.random.default_rng(1234)
    dim = 32
    count = 257
    query = rng.normal(size=dim).astype(np.float32)
    query /= np.linalg.norm(query)
    route_keys = []
    metadata = []
    lexical = []

    def normed(v):
        v = np.asarray(v, dtype=np.float32)
        return v / np.linalg.norm(v)

    def cosine(q, k):
        dot = 0.0
        qnorm = 0.0
        knorm = 0.0
        for qv, kv in zip(q, k):
            qf = float(qv)
            kf = float(kv)
            dot += qf * kf
            qnorm += qf * qf
            knorm += kf * kf
        return dot / ((np.sqrt(qnorm) * np.sqrt(knorm)) + 1.0e-8)

    for i in range(count):
        base = normed(rng.normal(size=dim))
        keys = [base]
        if i % 7 == 0:
            keys.append(normed((0.65 * base) + (0.35 * query)))
        route_keys.append(keys)
        metadata.append({
            "active": i % 37 != 0,
            "kind": "task_state" if i % 9 == 0 else "fact",
            "scope": "user" if i % 5 == 0 else "project",
            "durability": "session" if i % 11 == 0 else "project",
            "mutability": "ephemeral" if i % 13 == 0 else "stable",
        })
        lexical.append((f"node-{i}", f"bucket-{i % 17}"))

    qlex = ("node-73", "bucket-5")
    kinds = ("fact",)
    scopes = ("project", "user")

    scored = []
    for i, keys in enumerate(route_keys):
        meta = metadata[i]
        if not meta["active"] or meta["kind"] not in kinds:
            continue
        if meta["scope"] not in scopes:
            continue
        score = max(cosine(query, key) for key in keys)
        hits = sum(1 for q in qlex if q in lexical[i])
        score += hits / len(qlex)
        if np.isfinite(score):
            scored.append((score, i))
    scored.sort(key=lambda item: -item[0])
    expected = [node_id for _, node_id in scored[:8]]

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        for i, keys in enumerate(route_keys):
            node_id = store.add_node(f"arena route {i}", b"", ntok=1)
            assert node_id == i
            store.set_metadata(node_id, metadata[i])
            store.set_route_key_list(node_id, keys, lexical[i])

        assert store.route(query, qlex, topk=8,
                           kinds=kinds, scopes=scopes) == expected


def test_native_mla_gemv_arena_openmp_matches_reference_scan(tmp_path):
    lib = build_native(tmp_path, extra_cxxflags=("-fopenmp",))
    rng = np.random.default_rng(5678)
    dim = 16
    count = 1200
    query = rng.normal(size=dim).astype(np.float32)
    query /= np.linalg.norm(query)
    route_keys = []
    lexical = []

    def normed(v):
        v = np.asarray(v, dtype=np.float32)
        return v / np.linalg.norm(v)

    def cosine(q, k):
        dot = 0.0
        qnorm = 0.0
        knorm = 0.0
        for qv, kv in zip(q, k):
            qf = float(qv)
            kf = float(kv)
            dot += qf * kf
            qnorm += qf * qf
            knorm += kf * kf
        return dot / ((np.sqrt(qnorm) * np.sqrt(knorm)) + 1.0e-8)

    for i in range(count):
        base = normed(rng.normal(size=dim))
        keys = [base]
        if i % 19 == 0:
            keys.append(normed((0.5 * base) + (0.5 * query)))
        route_keys.append(keys)
        lexical.append((f"node-{i}", f"bucket-{i % 23}"))

    qlex = ("node-731", "bucket-6")
    scored = []
    for i, keys in enumerate(route_keys):
        score = max(cosine(query, key) for key in keys)
        hits = sum(1 for q in qlex if q in lexical[i])
        score += hits / len(qlex)
        if np.isfinite(score):
            scored.append((score, i))
    scored.sort(key=lambda item: -item[0])
    expected = [node_id for _, node_id in scored[:10]]

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        for i, keys in enumerate(route_keys):
            node_id = store.add_node(f"openmp arena route {i}", b"", ntok=1)
            assert node_id == i
            store.set_route_key_list(node_id, keys, lexical[i])

        assert store.route(query, qlex, topk=10) == expected


def test_native_mla_int4_refine_all_matches_fp32_route(tmp_path, monkeypatch):
    lib = build_native(tmp_path)
    rng = np.random.default_rng(9012)
    dim = 24
    count = 96
    query = rng.normal(size=dim).astype(np.float32)
    query /= np.linalg.norm(query)

    def normed(v):
        v = np.asarray(v, dtype=np.float32)
        return v / np.linalg.norm(v)

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        for i in range(count):
            base = normed(rng.normal(size=dim))
            keys = [base]
            if i % 11 == 0:
                keys.append(normed((0.4 * base) + (0.6 * query)))
            node_id = store.add_node(f"int4 refine all route {i}", b"", ntok=1)
            store.set_route_key_list(
                node_id, keys, (f"node-{i}", f"bucket-{i % 13}"))

        lexical = ("node-42", "bucket-7")
        fp32 = store.route(query, lexical, topk=12)
        monkeypatch.setenv("GRM_ROUTER_INT4", "1")
        monkeypatch.setenv("GRM_ROUTER_INT4_REFINE_M", str(count))
        assert store.route(query, lexical, topk=12) == fp32


def test_native_mla_tied_scores_use_node_id_order(tmp_path, monkeypatch):
    lib = build_native(tmp_path)
    query = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    expected = [0, 1, 2]

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        for i in range(8):
            assert store.add_node(f"tied route {i}", b"", ntok=1) == i
        for node_id in reversed(range(8)):
            store.set_route(node_id, query, [])

        assert store.route(query, topk=3) == expected
        monkeypatch.setenv("GRM_ROUTER_INT4", "1")
        monkeypatch.setenv("GRM_ROUTER_INT4_REFINE_M", "3")
        assert store.route(query, topk=3) == expected

    monkeypatch.delenv("GRM_ROUTER_INT4", raising=False)
    monkeypatch.delenv("GRM_ROUTER_INT4_REFINE_M", raising=False)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        for i in range(8):
            assert store.add_node(f"tied scan route {i}", b"", ntok=1) == i
        for node_id in reversed(range(8)):
            key = query if node_id != 7 else np.asarray(
                [1.0, 0.0, 0.0], dtype=np.float32)
            store.set_route(node_id, key, [])

        assert store.route(query, topk=3) == expected


def test_native_mla_int4_single_row_fast_path_matches_fp32_route(
        tmp_path, monkeypatch):
    lib = build_native(tmp_path)
    rng = np.random.default_rng(2468)
    dim = 32
    count = 64
    query = rng.normal(size=dim).astype(np.float32)
    query /= np.linalg.norm(query)

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        for i in range(count):
            key = rng.normal(size=dim).astype(np.float32)
            key /= np.linalg.norm(key)
            node_id = store.add_node(f"int4 single row route {i}", b"", ntok=1)
            store.set_route(node_id, key, (f"node-{i}", f"bucket-{i % 7}"))

        lexical = ("node-11", "bucket-4")
        fp32 = store.route(query, lexical, topk=8)
        monkeypatch.setenv("GRM_ROUTER_INT4", "1")
        monkeypatch.setenv("GRM_ROUTER_INT4_REFINE_M", str(count))
        assert store.route(query, lexical, topk=8) == fp32


def test_native_router_serializes_concurrent_route_updates(tmp_path):
    lib = build_native(tmp_path)
    dim = 16
    count = 64
    query = np.zeros(dim, dtype=np.float32)
    query[0] = 1.0

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        for i in range(count):
            key = np.zeros(dim, dtype=np.float32)
            key[i % dim] = 1.0
            node_id = store.add_node(f"concurrent route {i}", b"", ntok=1)
            store.set_route(node_id, key, (f"node-{i}",))

        errors = []
        start = threading.Event()

        def reader():
            start.wait()
            try:
                for _ in range(150):
                    routed = store.route(query, ("node-3",), topk=8)
                    if any(node_id < 0 or node_id >= count for node_id in routed):
                        errors.append(("bad-route", routed))
            except Exception as exc:  # pragma: no cover - thread diagnostic
                errors.append(("reader", repr(exc)))

        def writer():
            start.wait()
            try:
                for step in range(200):
                    node_id = step % count
                    key = np.zeros(dim, dtype=np.float32)
                    key[(node_id + step + 1) % dim] = 1.0
                    store.set_route(node_id, key, (f"node-{node_id}",))
            except Exception as exc:  # pragma: no cover - thread diagnostic
                errors.append(("writer", repr(exc)))

        threads = [threading.Thread(target=reader) for _ in range(3)]
        threads.append(threading.Thread(target=writer))
        for thread in threads:
            thread.start()
        start.set()
        for thread in threads:
            thread.join(timeout=10.0)
            assert not thread.is_alive()

        assert errors == []


def test_native_gqa_router_snapshots_concurrent_route_updates(tmp_path):
    lib = build_native(tmp_path)
    count = 32
    q = np.asarray([
        [[1.0, 0.0], [0.5, 0.5]],
        [[0.0, 1.0], [0.5, -0.5]],
    ], dtype=np.float32)

    with NativeGraftStore(
            lib, model_type="Qwen3_TC", num_layers=36,
            hidden_dim=2560, vals_per_tok_layer=2048, route_layer=0,
            payload_kind="gqa", num_kv_heads=1, head_dim=2) as store:
        for i in range(count):
            key = np.asarray([[[1.0 if i % 2 == 0 else 0.0,
                                0.0 if i % 2 == 0 else 1.0]]],
                             dtype=np.float32)
            node_id = store.add_node(f"concurrent GQA route {i}", b"", ntok=1)
            store.set_route_key_list(node_id, [key], [f"gqa-{i}"])

        errors = []
        start = threading.Event()

        def reader():
            start.wait()
            try:
                for _ in range(120):
                    routed = store.route_gqa(q, topk=8)
                    if any(node_id < 0 or node_id >= count for node_id in routed):
                        errors.append(("bad-gqa-route", routed))
            except Exception as exc:  # pragma: no cover - thread diagnostic
                errors.append(("reader", repr(exc)))

        def writer():
            start.wait()
            try:
                for step in range(160):
                    node_id = step % count
                    if step % 2 == 0:
                        key = np.asarray([[[1.0, 0.0]]], dtype=np.float32)
                    else:
                        key = np.asarray([[[0.0, 1.0]]], dtype=np.float32)
                    store.set_route_key_list(node_id, [key], [f"gqa-{node_id}"])
            except Exception as exc:  # pragma: no cover - thread diagnostic
                errors.append(("writer", repr(exc)))

        threads = [threading.Thread(target=reader) for _ in range(3)]
        threads.append(threading.Thread(target=writer))
        for thread in threads:
            thread.start()
        start.set()
        for thread in threads:
            thread.join(timeout=10.0)
            assert not thread.is_alive()

        assert errors == []


def test_native_store_structured_payload_tensors_via_ctypes(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        payload = {
            "c": np.arange(6, dtype=np.float32).reshape(2, 3),
            "kpe": np.arange(4, dtype=np.float16).reshape(1, 1, 2, 2),
        }
        node_id = store.add_structured_node("structured graft", payload, ntok=2)
        pstats = store.payload_stats(node_id)
        assert pstats.tensor_count == 2
        assert pstats.payload_bytes == payload["c"].nbytes + payload["kpe"].nbytes
        c_info = store.tensor_info(node_id, "c")
        kpe_info = store.tensor_info(node_id, "kpe")
        assert c_info.shape == (2, 3)
        assert c_info.dtype == "float32"
        assert c_info.payload_bytes == payload["c"].nbytes
        assert kpe_info.shape == (1, 1, 2, 2)
        assert kpe_info.dtype == "float16"
        assert kpe_info.payload_bytes == payload["kpe"].nbytes
        np.testing.assert_array_equal(store.get_tensor(node_id, "c"),
                                      payload["c"])
        np.testing.assert_array_equal(store.get_tensor(node_id, "kpe"),
                                      payload["kpe"])

        stats = store.stats()
        assert stats.nodes == 1
        assert stats.host_payload_tensors == 2
        assert stats.host_payload_bytes == pstats.payload_bytes


def test_native_metadata_json_round_trips_through_checkpoint(tmp_path):
    lib = build_native(tmp_path)
    ckpt = tmp_path / "metadata_ckpt"
    meta = {
        "kind": "fact",
        "durability": "project",
        "scope": "project",
        "write_intent": "user_asserted",
        "confidence": 0.75,
        "subject": "native metadata",
        "active": True,
        "supersedes": [3, 4],
        "superseded_by": [9],
        "source_turns": [1, 2],
        "source_grafts": [7, 8],
    }
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        node_id = store.add_node("metadata graft", b"abc", ntok=1)
        store.set_metadata(node_id, meta)
        assert store.metadata(node_id) == meta
        assert store.graph_edges(node_id).source_turns == (1, 2)
        assert store.graph_edges(node_id).source_grafts == (7, 8)
        assert store.graph_edges(node_id).supersedes == (3, 4)
        assert store.graph_edges(node_id).superseded_by == (9,)
        store.save_checkpoint(ckpt)

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as restored:
        restored.load_checkpoint(ckpt)
        assert restored.metadata(node_id) == meta
        restored_edges = restored.graph_edges(node_id)
        assert restored_edges.source_turns == (1, 2)
        assert restored_edges.source_grafts == (7, 8)
        assert restored_edges.supersedes == (3, 4)
        assert restored_edges.superseded_by == (9,)


def test_native_node_text_round_trips_through_checkpoint(tmp_path):
    lib = build_native(tmp_path)
    ckpt = tmp_path / "text_ckpt"
    text = "native host text payload 13-1313"

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        node_id = store.add_node(text, b"abc", ntok=4)
        assert store.text(node_id) == text
        store.save_checkpoint(ckpt)

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as restored:
        restored.load_checkpoint(ckpt)
        assert restored.text(node_id) == text


def test_native_node_summary_round_trips_through_checkpoint(tmp_path):
    lib = build_native(tmp_path)
    ckpt = tmp_path / "summary_ckpt"
    text = "native summary payload 14-1414"
    meta = {
        "kind": "fact",
        "scope": "project",
        "active": True,
        "write_intent": "user_asserted",
    }

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        node_id = store.add_node(text, b"abc", ntok=4)
        store.set_metadata(node_id, meta)
        assert store.node_summary(node_id) == {
            "node_id": node_id,
            "text": text,
            "metadata": meta,
        }
        store.save_checkpoint(ckpt)

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as restored:
        restored.load_checkpoint(ckpt)
        assert restored.node_summary(node_id)["metadata"] == meta
        assert restored.node_summary(node_id)["text"] == text


def test_native_provenance_round_trips_through_checkpoint(tmp_path):
    lib = build_native(tmp_path)
    ckpt = tmp_path / "provenance_ckpt"
    provenance = [{
        "segment_id": 7,
        "node_id": 3,
        "segment_type": "selected_span",
        "created_at": 42.5,
        "source_graft": 1,
        "token_start": 2,
        "token_end": 5,
    }]

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        node_id = store.add_node("native provenance payload", b"abc", ntok=4)
        store.set_provenance(node_id, provenance)
        assert store.provenance(node_id) == provenance
        store.save_checkpoint(ckpt)

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as restored:
        restored.load_checkpoint(ckpt)
        assert restored.provenance(node_id) == provenance


def test_native_fact_identity_scan_round_trips_through_checkpoint(tmp_path):
    lib = build_native(tmp_path)
    ckpt = tmp_path / "fact_identity_ckpt"
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        old = store.add_node("GRM target is disk", b"", ntok=1)
        new = store.add_node("GRM target is RAM", b"", ntok=1)
        timed = store.add_node("GRM target is RAM in 2026", b"", ntok=1)
        user = store.add_node("User-scoped target is NVMe", b"", ntok=1)
        inactive = store.add_node("Inactive target is tape", b"", ntok=1)
        expired_temporal = store.add_node(
            "GRM temporal target used to be disk", b"", ntok=1)
        current_temporal = store.add_node(
            "GRM temporal target is RAM", b"", ntok=1)
        future_temporal = store.add_node(
            "GRM temporal target will be GPU", b"", ntok=1)
        invalid_temporal = store.add_node(
            "GRM temporal guard is strict", b"", ntok=1)
        store.set_metadata(old, {
            "active": True, "kind": "fact", "scope": "project",
            "subject": "GRM target", "predicate": "is", "value": "disk"})
        store.set_metadata(new, {
            "active": True, "kind": "fact", "scope": "project",
            "subject": "GRM target", "predicate": "is", "value": "RAM"})
        store.set_metadata(timed, {
            "active": True, "kind": "fact", "scope": "project",
            "subject": "GRM target", "predicate": "is", "value": "RAM",
            "valid_from": "2026-01-01T00:00:00+00:00",
            "expires_at": "2027-01-01T00:00:00+00:00"})
        store.set_metadata(user, {
            "active": True, "kind": "fact", "scope": "user",
            "subject": "GRM target", "predicate": "is", "value": "NVMe"})
        store.set_metadata(inactive, {
            "active": False, "kind": "fact", "scope": "project",
            "subject": "GRM target", "predicate": "is", "value": "tape"})
        store.set_metadata(expired_temporal, {
            "active": True, "kind": "fact", "scope": "project",
            "subject": "GRM temporal target", "predicate": "is",
            "value": "disk",
            "expires_at": "2000-01-01T00:00:00+00:00"})
        store.set_metadata(current_temporal, {
            "active": True, "kind": "fact", "scope": "project",
            "subject": "GRM temporal target", "predicate": "is",
            "value": "RAM"})
        store.set_metadata(future_temporal, {
            "active": True, "kind": "fact", "scope": "project",
            "subject": "GRM temporal target", "predicate": "is",
            "value": "GPU",
            "valid_from": "2999-01-01T00:00:00+00:00"})
        store.set_metadata(invalid_temporal, {
            "active": True, "kind": "fact", "scope": "project",
            "subject": "GRM temporal guard", "predicate": "is",
            "value": "strict", "valid_from": "not-a-date"})

        assert store.fact_matches(
            subject="grm target", predicate="IS", value="ram",
            scope="project", value_mode=1) == (new, timed)
        assert store.fact_matches(
            subject="grm target", predicate="IS", value="ram",
            scope="project", value_mode=1,
            valid_from="2026-01-01T00:00:00+00:00",
            expires_at="2027-01-01T00:00:00+00:00",
            temporal_mode=1) == (timed,)
        assert store.fact_matches(
            subject="GRM target", predicate="is", value="RAM",
            scope="project", value_mode=2) == (old,)
        assert store.fact_matches(
            subject="GRM target", predicate="is", value="RAM",
            scope="user", value_mode=2) == (user,)
        assert store.fact_matches(
            subject="GRM temporal target", predicate="is", value="NVMe",
            scope="project", value_mode=2, temporal_mode=2) == (
                current_temporal,)
        assert store.fact_matches(
            subject="GRM temporal target", predicate="is", value="disk",
            scope="project", value_mode=1, temporal_mode=2) == ()
        assert store.fact_matches(
            subject="GRM temporal target", predicate="is", value="RAM",
            scope="project", value_mode=1, temporal_mode=2) == (
                current_temporal,)
        assert store.fact_matches(
            subject="GRM temporal guard", predicate="is", value="lax",
            scope="project", value_mode=2, temporal_mode=2) == (
                invalid_temporal,)
        store.save_checkpoint(ckpt)

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as restored:
        restored.load_checkpoint(ckpt)
        assert restored.fact_matches(
            subject="GRM target", predicate="is", value="RAM",
            scope="project", value_mode=1) == (new, timed)
        assert restored.fact_matches(
            subject="GRM target", predicate="is", value="RAM",
            scope="project", value_mode=1,
            valid_from="2026-01-01T00:00:00+00:00",
            expires_at="2027-01-01T00:00:00+00:00",
            temporal_mode=1) == (timed,)
        assert restored.fact_matches(
            subject="GRM temporal target", predicate="is", value="NVMe",
            scope="project", value_mode=2, temporal_mode=2) == (
                current_temporal,)
        assert restored.fact_matches(
            subject="GRM target", predicate="is", value="RAM",
            scope="project", value_mode=2) == (old,)


def test_native_checkpoint_round_trips_structured_payloads(tmp_path):
    lib = build_native(tmp_path)
    ckpt = tmp_path / "native_ckpt"
    payload = {
        "c": np.arange(6, dtype=np.float32).reshape(2, 3),
        "kpe": np.arange(4, dtype=np.float16).reshape(1, 1, 2, 2),
    }
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        node_id = store.add_structured_node("checkpoint graft", payload, ntok=2)
        assert store.stats().dirty_nodes == 1
        assert store.dirty_node_ids() == (node_id,)
        store.save_checkpoint(ckpt)
        assert store.stats().dirty_nodes == 0
        assert store.stats().durable_nodes == 1
        assert store.dirty_node_ids() == ()
        store.set_metadata(node_id, {"kind": "fact", "active": True})
        assert store.dirty_node_ids() == (node_id,)

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as reloaded:
        reloaded.load_checkpoint(ckpt)
        stats = reloaded.stats()
        assert stats.nodes == 1
        assert stats.dirty_nodes == 0
        assert stats.durable_nodes == 1
        assert stats.host_payload_tensors == 2
        np.testing.assert_array_equal(reloaded.get_tensor(node_id, "c"),
                                      payload["c"])
        np.testing.assert_array_equal(reloaded.get_tensor(node_id, "kpe"),
                                      payload["kpe"])


def test_native_dirty_plan_tracks_payload_metadata_and_priority(tmp_path):
    lib = build_native(tmp_path)
    ckpt = tmp_path / "native_dirty_plan_ckpt"
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        meta_only = store.add_node("metadata-only dirty node", b"a", ntok=1)
        store.save_checkpoint(ckpt)
        store.set_metadata(meta_only, {
            "kind": "fact",
            "durability": "session",
            "scope": "project",
            "active": True,
        })

        project = store.add_node("project dirty node", b"12345", ntok=1)
        store.set_metadata(project, {
            "kind": "fact",
            "durability": "project",
            "scope": "project",
            "active": True,
        })
        permanent = store.add_node("permanent dirty node", b"xy", ntok=1)
        store.set_metadata(permanent, {
            "kind": "fact",
            "durability": "permanent",
            "scope": "project",
            "active": True,
        })

        assert store.dirty_node_ids() == (meta_only, project, permanent)
        plan = store.dirty_plan()

        assert [item.node_id for item in plan] == [
            permanent, project, meta_only]
        assert [item.durability_priority for item in plan] == [0, 1, 2]
        assert [item.payload_bytes for item in plan] == [2, 5, 0]
        assert [item.payload_dirty for item in plan] == [True, True, False]
        assert [item.metadata_dirty for item in plan] == [True, True, True]


def test_native_checkpoint_enforces_dialect_wall(tmp_path):
    lib = build_native(tmp_path)
    ckpt = tmp_path / "gqa_ckpt"
    payload = {
        "k": np.zeros((2, 8, 3, 128), np.float16),
        "v": np.ones((2, 8, 3, 128), np.float16),
    }
    with NativeGraftStore(
            lib, model_type="Qwen3_TC", num_layers=36,
            hidden_dim=2560, vals_per_tok_layer=2048, route_layer=0,
            payload_kind="gqa", num_kv_heads=8, head_dim=128) as store:
        node_id = store.add_structured_node("GQA checkpoint graft", payload,
                                            ntok=3)
        store.save_checkpoint(ckpt)

    with NativeGraftStore(
            lib, model_type="Qwen3_TC", num_layers=36,
            hidden_dim=2560, vals_per_tok_layer=2048, route_layer=0,
            payload_kind="gqa", num_kv_heads=8, head_dim=128) as reloaded:
        reloaded.load_checkpoint(ckpt)
        assert reloaded.dialect_id() == "Qwen3_TC:36x2560:g8x128"
        np.testing.assert_array_equal(reloaded.get_tensor(node_id, "v"),
                                      payload["v"])

    with NativeGraftStore(
            lib, model_type="Qwen3_TC", num_layers=36,
            hidden_dim=2560, vals_per_tok_layer=2048, route_layer=0,
            payload_kind="gqa", num_kv_heads=8, head_dim=128,
            position_law="learned_absolute",
            graftability="same_position_restore",
            remountable=False,
            composition="prefix_restore_only") as wrong_profile:
        with pytest.raises(RuntimeError, match="graft profile mismatch"):
            wrong_profile.load_checkpoint(ckpt)

    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as wrong:
        with pytest.raises(RuntimeError, match="dialect mismatch"):
            wrong.load_checkpoint(ckpt)


def test_native_device_arena_swap_plan_via_ctypes(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        store.configure_arena(sink_tokens=6, arena_width=64)

        p0 = store.plan_swap(new_mount_tokens=10, input_cache_tokens=100)
        assert p0.sink_tokens == 6
        assert p0.arena_width == 64
        assert p0.old_mount_tokens == 0
        assert p0.old_mount_end == 6
        assert p0.live_tail_start == 6
        assert p0.live_tail_tokens == 94
        assert p0.output_cache_tokens == 110
        assert p0.overflow is False

        store.commit_mount([1, 2], mount_tokens=10)
        p1 = store.plan_swap(new_mount_tokens=4, input_cache_tokens=110)
        assert p1.old_mount_tokens == 10
        assert p1.old_mount_end == 16
        assert p1.live_tail_start == 16
        assert p1.live_tail_tokens == 94
        assert p1.output_cache_tokens == 104

        p2 = store.plan_swap(new_mount_tokens=65, input_cache_tokens=110)
        assert p2.overflow is True


def test_native_device_arena_evict_plan_via_ctypes(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        store.configure_arena(sink_tokens=6, arena_width=64)
        store.commit_mount([1, 2], mount_tokens=10)

        p0 = store.plan_evict(drop_tokens=20, input_cache_tokens=110)
        assert p0.sink_tokens == 6
        assert p0.arena_width == 64
        assert p0.mount_tokens == 10
        assert p0.head_tokens == 16
        assert p0.drop_tokens == 20
        assert p0.input_cache_tokens == 110
        assert p0.output_cache_tokens == 90
        assert p0.underflow is False

        p1 = store.plan_evict(drop_tokens=95, input_cache_tokens=110)
        assert p1.underflow is True


def test_native_device_arena_applies_host_tensor_swap_via_ctypes(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        store.configure_arena(sink_tokens=2, arena_width=4)
        store.commit_mount([7], mount_tokens=3)

        old = np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3)
        mount = (
            np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3) + 1000
        )
        out = store.apply_swap_tensor(old, mount, seq_dim=1)
        expected = np.concatenate([old[:, :2, :], mount, old[:, 5:, :]],
                                  axis=1)
        assert out.shape == (2, 7, 3)
        np.testing.assert_array_equal(out, expected)


def test_native_device_arena_applies_host_tensor_evict_via_ctypes(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        store.configure_arena(sink_tokens=2, arena_width=4)
        store.commit_mount([7], mount_tokens=3)

        old = np.arange(2 * 10 * 3, dtype=np.float32).reshape(2, 10, 3)
        out = store.apply_evict_tensor(old, seq_dim=1, drop_tokens=2)
        expected = np.concatenate([old[:, :5, :], old[:, 7:, :]], axis=1)
        assert out.shape == (2, 8, 3)
        np.testing.assert_array_equal(out, expected)


def test_native_device_arena_applies_payload_swap_reference(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        store.configure_arena(sink_tokens=2, arena_width=4)
        store.commit_mount([7], mount_tokens=3)

        old = {
            "c": np.arange(1 * 8 * 4, dtype=np.float16).reshape(1, 8, 4),
            "kpe": np.arange(1 * 1 * 8 * 2, dtype=np.float16).reshape(
                1, 1, 8, 2),
        }
        mount = {
            "c": np.full((1, 2, 4), 11, dtype=np.float16),
            "kpe": np.full((1, 1, 2, 2), 22, dtype=np.float16),
        }
        out = store.apply_swap_payload(
            old, mount, payload_dims=(("c", 1), ("kpe", 2)))
        np.testing.assert_array_equal(
            out["c"], np.concatenate([old["c"][:, :2], mount["c"],
                                      old["c"][:, 5:]], axis=1))
        np.testing.assert_array_equal(
            out["kpe"], np.concatenate([old["kpe"][:, :, :2], mount["kpe"],
                                        old["kpe"][:, :, 5:]], axis=2))


def test_native_device_arena_applies_payload_evict_reference(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        store.configure_arena(sink_tokens=2, arena_width=4)
        store.commit_mount([7], mount_tokens=3)

        old = {
            "c": np.arange(1 * 10 * 4, dtype=np.float16).reshape(1, 10, 4),
            "kpe": np.arange(1 * 1 * 10 * 2, dtype=np.float16).reshape(
                1, 1, 10, 2),
        }
        out = store.apply_evict_payload(
            old, payload_dims=(("c", 1), ("kpe", 2)), drop_tokens=2)
        np.testing.assert_array_equal(
            out["c"], np.concatenate([old["c"][:, :5],
                                      old["c"][:, 7:]], axis=1))
        np.testing.assert_array_equal(
            out["kpe"], np.concatenate([old["kpe"][:, :, :5],
                                        old["kpe"][:, :, 7:]], axis=2))


def test_native_device_arena_rejects_host_tensor_swap_overflow(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        store.configure_arena(sink_tokens=2, arena_width=1)
        store.commit_mount([7], mount_tokens=3)
        old = np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3)
        mount = np.zeros((2, 2, 3), dtype=np.float32)

        with pytest.raises(RuntimeError, match="exceeds configured width"):
            store.apply_swap_tensor(old, mount, seq_dim=1)


def test_native_device_arena_rejects_host_tensor_evict_underflow(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        store.configure_arena(sink_tokens=2, arena_width=4)
        store.commit_mount([7], mount_tokens=3)
        old = np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3)

        with pytest.raises(RuntimeError, match="evict drop exceeds live tail"):
            store.apply_evict_tensor(old, seq_dim=1, drop_tokens=4)


def test_arena_commits_native_mount_state():
    class RecordingStore:
        def __init__(self):
            self.calls = []

        def commit_mount(self, node_ids, mount_tokens):
            self.calls.append((list(node_ids), int(mount_tokens)))

    arena = ArenaCache.__new__(ArenaCache)
    arena.native_store = RecordingStore()
    arena.grafts = [
        {"native_node_id": 41},
        {"native_node_id": 42},
    ]

    arena._commit_native_mount([0, 1], 9)
    assert arena.native_store.calls == [([41, 42], 9)]

    arena.grafts.append({})
    with pytest.raises(RuntimeError, match="no native_node_id"):
        arena._commit_native_mount([2], 1)


def test_arena_route_uses_native_index_for_flat_mla_nodes(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        n0 = store.add_node("semantic winner", b"", ntok=1)
        n1 = store.add_node("partial lexical", b"", ntok=1)
        n2 = store.add_node("weak lexical", b"", ntok=1)
        store.set_route(n0, [0.95, 0.3122499], [])
        store.set_route(n1, [0.40, 0.9165151], ["a17"])
        store.set_route(n2, [0.0, 1.0], ["beta"])

        arena = ArenaCache.__new__(ArenaCache)
        arena.native_store = store
        arena.last_route_backend = "python"
        arena.grafts = [
            {"native_node_id": n0, "cent": np.array([0.95, 0.3122499], np.float32),
             "rare": set(), "text": "semantic winner", "kind": "doc"},
            {"native_node_id": n1, "cent": np.array([0.40, 0.9165151], np.float32),
             "rare": {"a17"}, "text": "partial lexical", "kind": "doc"},
            {"native_node_id": n2, "cent": np.array([0.0, 1.0], np.float32),
             "rare": {"beta"}, "text": "weak lexical", "kind": "doc"},
        ]
        arena._probe_key = lambda _text: np.array([1.0, 0.0], np.float32)

        assert arena.route("What about A17 and BETA?", exclude=set()) == [0, 1, 2]
        assert arena.last_route_backend == "native"


def test_arena_route_uses_native_child_centroid_keys(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        n0 = store.add_node("digest", b"", ntok=1)
        n1 = store.add_node("near miss", b"", ntok=1)
        store.set_route_keys(n0, [[1.0, 0.0], [0.0, 1.0]], [])
        store.set_route(n1, [0.7, 0.7141428], [])

        arena = ArenaCache.__new__(ArenaCache)
        arena.native_store = store
        arena.last_route_backend = "python"
        arena.grafts = [
            {"native_node_id": n0, "cent": np.array([1.0, 0.0], np.float32),
             "child_cents": [np.array([0.0, 1.0], np.float32)],
             "rare": set(), "text": "digest", "kind": "digest"},
            {"native_node_id": n1, "cent": np.array([0.7, 0.7141428], np.float32),
             "rare": set(), "text": "near miss", "kind": "doc"},
        ]
        arena._probe_key = lambda _text: np.array([0.0, 1.0], np.float32)

        assert arena.route("plain probe", exclude=set()) == [0, 1]
        assert arena.last_route_backend == "native"


def test_gqa_arena_route_uses_native_raw_qk_backend(tmp_path):
    lib = build_native(tmp_path)
    q = np.asarray([
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 1.0], [1.0, -1.0]],
    ], dtype=np.float32)
    weak = np.asarray([[[0.1, 0.0]]], dtype=np.float32)
    good = np.asarray([[[1.0, 1.0]]], dtype=np.float32)

    with NativeGraftStore(
            lib, model_type="Qwen3_TC", num_layers=36,
            hidden_dim=2560, vals_per_tok_layer=2048, route_layer=0,
            payload_kind="gqa", num_kv_heads=1, head_dim=2) as store:
        n0 = store.add_node("weak GQA route", b"", ntok=1)
        n1 = store.add_node("good GQA route", b"", ntok=1)
        store.set_route_key_list(n0, [weak], [])
        store.set_route_key_list(n1, [good], [])

        arena = GQAArenaCache.__new__(GQAArenaCache)
        arena.native_store = store
        arena.last_route_backend = "python"
        arena.grafts = [
            {"native_node_id": n0, "cent": weak,
             "rare": set(), "text": "weak", "kind": "doc"},
            {"native_node_id": n1, "cent": good,
             "rare": set(), "text": "good", "kind": "doc"},
        ]
        arena._probe_key = lambda _text: q

        assert arena.route("plain probe", exclude=set()) == [1, 0]
        assert arena.last_route_backend == "native"


def test_gqa_arena_uses_opt_in_cuda_route_bank(monkeypatch):
    class CudaStore:
        def __init__(self):
            self.configured = None
            self.cuda_calls = []
            self._cuda_gqa_bank = None

        def configure_cuda_gqa_route_bank(self, route_bank, node_ids):
            self.configured = (
                np.asarray(route_bank).copy(),
                np.asarray(node_ids, dtype=np.uint64).copy(),
            )
            self._cuda_gqa_bank = object()

        def route_gqa_cuda(self, query, *, topk=3, **_kwargs):
            self.cuda_calls.append((np.asarray(query).shape, int(topk)))
            return [102, 101][:int(topk)]

        def route_gqa(self, *_args, **_kwargs):
            raise AssertionError("CPU GQA route should not run")

    monkeypatch.setenv("GRM_GQA_CUDA_ROUTE", "1")
    q = np.asarray([[[1.0, 0.0]]], dtype=np.float32)
    weak = np.asarray([[[0.1, 0.0]]], dtype=np.float32)
    good = np.asarray([[[1.0, 0.0]]], dtype=np.float32)

    arena = GQAArenaCache.__new__(GQAArenaCache)
    arena.native_store = CudaStore()
    arena.last_route_backend = "python"
    arena.grafts = [
        {"native_node_id": 101, "cent": weak,
         "rare": set(), "text": "weak", "kind": "doc"},
        {"native_node_id": 102, "cent": good,
         "rare": set(), "text": "good", "kind": "doc"},
    ]
    arena._probe_key = lambda _text: q

    assert arena.route("plain probe", exclude=set(), limit=2) == [1, 0]
    assert arena.last_route_backend == "cuda"
    bank, node_ids = arena.native_store.configured
    assert bank.shape == (2, 1, 1, 2)
    assert node_ids.tolist() == [101, 102]
    assert arena.native_store.cuda_calls == [((1, 1, 2), 2)]


def test_gqa_arena_rebuilds_cuda_route_bank_when_rows_change(monkeypatch):
    class RebuildStore:
        def __init__(self):
            self.configured_node_ids = []
            self.configure_calls = 0
            self._cuda_gqa_bank = None
            self._cuda_gqa_bank_signature = None

        def configure_cuda_gqa_route_bank(self, _route_bank, node_ids):
            self.configure_calls += 1
            self.configured_node_ids = [
                int(x) for x in np.asarray(node_ids, dtype=np.uint64)]
            self._cuda_gqa_bank = object()

        def route_gqa_cuda(self, _query, *, topk=3, **_kwargs):
            return list(reversed(self.configured_node_ids))[:int(topk)]

        def route_gqa(self, *_args, **_kwargs):
            raise AssertionError("CPU GQA route should not run")

    monkeypatch.setenv("GRM_GQA_CUDA_ROUTE", "1")
    q = np.asarray([[[1.0, 0.0]]], dtype=np.float32)
    weak = np.asarray([[[0.1, 0.0]]], dtype=np.float32)
    good = np.asarray([[[1.0, 0.0]]], dtype=np.float32)
    best = np.asarray([[[2.0, 0.0]]], dtype=np.float32)

    arena = GQAArenaCache.__new__(GQAArenaCache)
    arena.native_store = RebuildStore()
    arena.last_route_backend = "python"
    arena.grafts = [
        {"native_node_id": 101, "cent": weak,
         "rare": set(), "text": "weak", "kind": "doc"},
        {"native_node_id": 102, "cent": good,
         "rare": set(), "text": "good", "kind": "doc"},
    ]
    arena._probe_key = lambda _text: q

    assert arena.route("plain probe", exclude=set(), limit=1) == [1]
    first_signature = arena.native_store._cuda_gqa_bank_signature

    arena.grafts.append(
        {"native_node_id": 103, "cent": best,
         "rare": set(), "text": "best", "kind": "doc"})
    # W2 (GRM_CUDA_BRIDGE_OVERHEAD_PLAN): the epoch is now the SOLE
    # hot-path staleness gate — the O(N) signature walk only runs when
    # _cuda_gqa_epoch has moved. A raw external append (this test pokes
    # arena.grafts directly, bypassing deposit()/graft_repository.py's
    # instrumented mutation sites) must bump the epoch itself, exactly as
    # every real production mutation path now does.
    arena._bump_cuda_gqa_epoch()

    assert arena.route("plain probe", exclude=set(), limit=1) == [2]
    assert arena.native_store.configure_calls == 2
    assert arena.native_store.configured_node_ids == [101, 102, 103]
    assert arena.native_store._cuda_gqa_bank_signature != first_signature


def _gqa_epoch_battery_arena():
    """A GQAArenaCache with three eligible dense-bank nodes, wired the same
    way the existing 6-selector's arena-side tests are (no live model, no
    NativeGraftStore round trip — just the Python route-bank cache
    machinery under test)."""
    weak = np.asarray([[[0.1, 0.0]]], dtype=np.float32)
    good = np.asarray([[[1.0, 0.0]]], dtype=np.float32)
    best = np.asarray([[[2.0, 0.0]]], dtype=np.float32)
    arena = GQAArenaCache.__new__(GQAArenaCache)
    arena.native_store = None
    arena.last_route_backend = "python"
    arena.grafts = [
        {"native_node_id": 101, "cent": weak.copy(),
         "rare": set(), "text": "weak", "kind": "doc", "retired": False},
        {"native_node_id": 102, "cent": good.copy(),
         "rare": set(), "text": "good", "kind": "doc", "retired": False},
        {"native_node_id": 103, "cent": best.copy(),
         "rare": set(), "text": "best", "kind": "doc", "retired": False},
    ]
    return arena


def test_gqa_cuda_epoch_bump_covers_every_signature_changing_mutation():
    """P1 fail-closed equivalence (plan gate): for a battery of mutations
    covering the GQAArenaCache-side mutation set this task's epoch tracks
    (add, retire, cent replace, active-state/kind, revision [retire+add],
    expire [retire], clear [truncate]), any mutation that changes the arena
    signature (`_cuda_route_bank_signature`) must also bump the epoch
    (`_cuda_gqa_epoch`). Equivalently: epoch-valid implies signature-equal
    — an unchanged epoch is only ever observed together with an unchanged
    signature. Mirrors the mutation coverage of the store-level 6-selector
    (test_native_gqa_route_mutation_clears_cuda_bank /
    test_native_gqa_eligibility_mutation_clears_cuda_bank) one layer up, on
    the arena-side signature the CUDA bridge's stack cache keys on."""
    arena = _gqa_epoch_battery_arena()

    def snapshot():
        epoch = getattr(arena, "_cuda_gqa_epoch", 0)
        sig = arena._cuda_route_bank_signature()
        signature = None if sig is None else sig[1]
        return epoch, signature

    checked = 0

    def assert_mutation_is_fail_closed(mutate):
        nonlocal checked
        before_epoch, before_sig = snapshot()
        mutate()
        after_epoch, after_sig = snapshot()
        if after_sig != before_sig:
            # The load-bearing assertion: any signature change must be
            # accompanied by an epoch bump. Equivalently, an unchanged
            # epoch must never coexist with a changed signature — under-
            # invalidation would let a route call reuse a stale stacked
            # bank.
            assert after_epoch != before_epoch, (
                "signature changed but epoch did not bump — under-"
                "invalidation risk")
        checked += 1

    # add
    def do_add():
        arena.grafts.append({"native_node_id": 104,
                             "cent": np.asarray([[[3.0, 0.0]]], np.float32),
                             "rare": set(), "text": "added", "kind": "doc",
                             "retired": False})
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_add)

    # cent replace (route-key replace)
    def do_cent_replace():
        arena.grafts[0]["cent"] = np.asarray([[[9.0, 0.0]]], np.float32)
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_cent_replace)

    # active-state / kind (recall exclusion mirrors "active=False")
    def do_kind_change():
        arena.grafts[1]["kind"] = "recall"
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_kind_change)

    # metadata-only (does NOT change eligibility/signature — must NOT be
    # required to change the signature; the epoch bump is still fail-closed
    # to bump regardless, since over-invalidation is acceptable)
    def do_metadata_only():
        arena.grafts[2]["tags"] = ["noted"]
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_metadata_only)

    # revision (supersede): retire the old node, add its replacement
    def do_revision():
        arena.grafts[2]["retired"] = True
        arena.grafts.append({"native_node_id": 105,
                             "cent": np.asarray([[[4.0, 0.0]]], np.float32),
                             "rare": set(), "text": "revised", "kind": "doc",
                             "retired": False})
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_revision)

    # expire
    def do_expire():
        arena.grafts[0]["retired"] = True
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_expire)

    # retire (plain)
    def do_retire():
        arena.grafts[3]["retired"] = True
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_retire)

    # clear (truncate, as step()'s rollback does)
    def do_clear():
        del arena.grafts[:]
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_clear)

    assert checked == 8


def test_gqa_cuda_bridge_never_reuses_stale_bank_across_mutation_battery():
    """Behavior half of the P1 regression: after each mutation in the same
    battery, a bridged (CUDA) route must return the same node ordering a
    fresh attach would — no stale bank reuse. Uses a fake store that fails
    loudly if route_gqa_cuda is ever called against a bank whose configured
    node set does not match current eligibility (the stale-bank failure
    mode the 2026-07-07 lifecycle fix closed)."""
    class TrackingStore:
        def __init__(self):
            self.configured_node_ids = None
            self.configure_calls = 0
            self._cuda_gqa_bank = None
            self._cuda_gqa_bank_signature = None

        def configure_cuda_gqa_route_bank(self, _route_bank, node_ids):
            self.configure_calls += 1
            self.configured_node_ids = [
                int(x) for x in np.asarray(node_ids, dtype=np.uint64)]
            self._cuda_gqa_bank = object()

        def route_gqa_cuda(self, _query, *, topk=3, **_kwargs):
            # Route against whatever is currently attached — a stale bank
            # would answer from a node set that no longer matches the
            # arena's live eligible grafts.
            return list(self.configured_node_ids)[:int(topk)]

        def route_gqa(self, *_args, **_kwargs):
            raise AssertionError("CPU GQA route should not run")

    import os as _os
    _os.environ["GRM_GQA_CUDA_ROUTE"] = "1"
    try:
        arena = _gqa_epoch_battery_arena()
        arena.native_store = TrackingStore()
        q = np.asarray([[[1.0, 0.0]]], dtype=np.float32)
        arena._probe_key = lambda _text: q

        def eligible_native_ids():
            return sorted(
                int(g["native_node_id"]) for g in arena.grafts
                if not g.get("retired") and g.get("kind", "turn") != "recall")

        def assert_bridge_matches_fresh_attach():
            expect = eligible_native_ids()
            got_order = arena.route("plain probe", exclude=set(),
                                    limit=len(expect) or 1)
            got_native_ids = sorted(
                int(arena.grafts[i]["native_node_id"]) for i in got_order)
            assert got_native_ids == expect or not expect
            assert sorted(arena.native_store.configured_node_ids) == expect, (
                "bridged route bank does not match current eligibility — "
                "stale bank reuse")

        assert_bridge_matches_fresh_attach()

        arena.grafts.append({"native_node_id": 104,
                             "cent": np.asarray([[[3.0, 0.0]]], np.float32),
                             "rare": set(), "text": "added", "kind": "doc",
                             "retired": False})
        arena._bump_cuda_gqa_epoch()
        assert_bridge_matches_fresh_attach()

        arena.grafts[0]["cent"] = np.asarray([[[9.0, 0.0]]], np.float32)
        arena._bump_cuda_gqa_epoch()
        assert_bridge_matches_fresh_attach()

        arena.grafts[1]["kind"] = "recall"
        arena._bump_cuda_gqa_epoch()
        assert_bridge_matches_fresh_attach()

        arena.grafts[2]["retired"] = True
        arena._bump_cuda_gqa_epoch()
        assert_bridge_matches_fresh_attach()

        arena.grafts[3]["retired"] = True
        arena._bump_cuda_gqa_epoch()
        assert_bridge_matches_fresh_attach()

        del arena.grafts[:]
        arena._bump_cuda_gqa_epoch()
        result = arena.route("plain probe", exclude=set(), limit=1)
        assert result == []
    finally:
        _os.environ.pop("GRM_GQA_CUDA_ROUTE", None)


def _fake_gqa_repo_for_epoch_battery():
    """A GraftRepository wired to a GQAArenaCache with NO live model, NO
    native store — pure Python mutation-path testing (mirrors
    _gqa_epoch_battery_arena one layer up, at the GraftRepository surface
    this task's W1 scope-expansion targets: forget/correct_memory/migrate/
    cull_graft, all of which mutate arena.grafts fields the CUDA route bank
    signature reads without going through GQAArenaCache's own deposit()/
    consolidate() bump sites).

    arena.deposit() is stubbed (no harvest, no model) but preserves the
    real contract: append a graft, bump the epoch — exactly what P1 already
    covers at the GQAArenaCache layer, so this fixture isolates the NEW
    graft_repository.py-side sites from that pre-existing coverage.
    """
    arena = GQAArenaCache.__new__(GQAArenaCache)
    arena.native_store = None
    arena.node_loader = None
    arena.last_route_backend = "python"
    arena.grafts = []
    arena._cuda_gqa_epoch = 0
    arena._rare_tokens = staticmethod(ArenaCache._rare_tokens)

    counter = {"n": 0}

    def fake_deposit(text):
        counter["n"] += 1
        idx = len(arena.grafts)
        cent = np.asarray([[[float(counter["n"]), 0.0]]], dtype=np.float32)
        ntok = max(1, len(str(text).split()))
        # GQA PAYLOAD = (("k", 2), ("v", 2)) -- token axis is index 2, so a
        # cull_graft() slice on this fake payload can find it via
        # _payload_token_axis the same way a real (L, H, S, D) tensor would.
        payload = {"k": np.zeros((1, 1, ntok, 1), np.float32),
                  "v": np.zeros((1, 1, ntok, 1), np.float32)}
        arena.grafts.append({
            "cent": cent, "ntok": ntok,
            "text": text, "host_payload": payload,
            "h": None,
            # native_node_id populated (as a real sync would, absent a
            # native_store here) so _cuda_route_bank_signature is non-None
            # and the fail-closed comparison below is exercised, not
            # trivially skipped by the "no native id -> None" early-out.
            "native_node_id": idx,
        })
        arena._bump_cuda_gqa_epoch()
        return idx

    arena.deposit = fake_deposit
    arena._node_key = lambda text, h_host=None: np.asarray(
        [[[7.0, 0.0]]], dtype=np.float32)
    arena.encode = lambda text: list(range(len(str(text).split())))
    arena.decode = lambda ids: " ".join(str(i) for i in ids)

    repo = GraftRepository.__new__(GraftRepository)
    repo.arena = arena
    repo.native_store = None
    repo._native_node_ids = {}
    repo.dirty_nodes = {}
    repo._dirty_generation = 0
    repo.wal_enabled = False
    repo.review_buffer = []
    repo.fold_history = []
    repo._all_graft_text_ascii = True
    repo.path = None
    repo._segment_id = 0
    repo.vram_budget = None
    repo.autosave = False
    repo.runtime = GRMRuntime(repo)
    return repo


def test_gqa_cuda_epoch_bump_covers_graft_repository_mutation_battery():
    """W1 scope-expansion regression (ledger 2026-07-08 01:30): P1 proved
    the epoch is fail-closed for GQAArenaCache-side mutations (deposit,
    consolidate, step rollback). This proves the same equivalence for the
    graft_repository.py-side mutation battery the P1 agent identified but
    left uninstrumented: forget, correct_memory, migrate, cull_graft — every
    one of which writes arena.grafts fields the CUDA route bank signature
    reads (retired, kind, child_cents, native_node_id, cent, list
    membership) via a path that does NOT go through GQAArenaCache's own
    bump sites.

    Same fail-closed contract as the arena-side battery: any signature
    change must be accompanied by an epoch bump (epoch-valid implies
    signature-equal)."""
    repo = _fake_gqa_repo_for_epoch_battery()
    arena = repo.arena

    def snapshot():
        epoch = getattr(arena, "_cuda_gqa_epoch", 0)
        sig = arena._cuda_route_bank_signature()
        signature = None if sig is None else sig[1]
        return epoch, signature

    checked = 0

    def assert_mutation_is_fail_closed(mutate):
        nonlocal checked
        before_epoch, before_sig = snapshot()
        mutate()
        after_epoch, after_sig = snapshot()
        if after_sig != before_sig:
            assert after_epoch != before_epoch, (
                "signature changed but epoch did not bump — under-"
                "invalidation risk (graft_repository.py mutation path)")
        checked += 1

    # remember(): deposit (already-bumped) + kind overwrite afterward
    def do_remember():
        repo.remember("first fact about the routing subsystem")
    assert_mutation_is_fail_closed(do_remember)

    def do_remember2():
        repo.remember("second fact about the routing subsystem")
    assert_mutation_is_fail_closed(do_remember2)

    # forget(): direct retired=True write on a matched active target
    def do_forget():
        repo.forget("first fact")
    assert_mutation_is_fail_closed(do_forget)

    # correct_memory(): retires active targets, writes a replacement via
    # remember() internally
    def do_correct():
        repo.correct_memory("second fact", "corrected fact about routing")
    assert_mutation_is_fail_closed(do_correct)

    # add_document(): deposit + kind="doc" overwrite
    def do_add_document():
        repo.add_document("a standalone document about arenas", tags=("t",))
    assert_mutation_is_fail_closed(do_add_document)

    # cull_graft(): splits a graft into children, retires the parent,
    # rebuilds child_cents
    cull_idx = None

    def do_cull_setup():
        nonlocal cull_idx
        cull_idx = repo.arena.deposit(
            "one two three four five six seven eight")
        repo.arena.grafts[cull_idx]["metadata"] = repo._default_metadata(
            repo.arena.grafts[cull_idx])
    assert_mutation_is_fail_closed(do_cull_setup)

    def do_cull():
        repo.cull_graft(cull_idx, spans=[(0, 4), (4, 8)],
                        retire_parent=True, recompute_route=False)
    assert_mutation_is_fail_closed(do_cull)

    assert checked == 7


def test_gqa_cuda_epoch_bump_covers_migrate_mutation_battery():
    """migrate() rebuilds an EMPTY repository's graft list node-by-node from
    another repository's manifest — a distinct code path from the battery
    above (per-node deposit + kind/tags/sources/retired/rare overwrite,
    then a bulk _rebuild_child_keys()). Same fail-closed contract, isolated
    into its own test because migrate() requires an empty arena precondition
    (RuntimeError otherwise) and a source manifest.json on disk."""
    import json as _json
    import os as _os
    import tempfile as _tempfile

    repo = _fake_gqa_repo_for_epoch_battery()
    arena = repo.arena

    with _tempfile.TemporaryDirectory(prefix="grm-migrate-epoch-") as src:
        manifest = {
            "nodes": [
                {"text": "migrated node one", "kind": "doc", "tags": [],
                 "sources": [], "retired": False},
                {"text": "migrated node two", "kind": "fact", "tags": [],
                 "sources": [], "retired": True},
            ],
        }
        with open(_os.path.join(src, "manifest.json"), "w") as fh:
            _json.dump(manifest, fh)

        before_epoch = getattr(arena, "_cuda_gqa_epoch", 0)
        before_sig = arena._cuda_route_bank_signature()

        # migrate() calls flush_now()/_free_retired()/_page() at the end;
        # stub the persistence tail so this stays a pure in-memory test.
        repo.flush_now = lambda: None
        repo._free_retired = lambda: None
        repo._page = lambda: None

        migrated = repo.migrate(src)

        after_epoch = getattr(arena, "_cuda_gqa_epoch", 0)
        after_sig = arena._cuda_route_bank_signature()

        assert migrated == 2
        assert after_sig != before_sig
        assert after_epoch != before_epoch, (
            "migrate() changed the signature but did not bump the epoch — "
            "under-invalidation risk")


def test_gqa_cuda_bridge_paranoid_env_catches_epoch_under_invalidation(
        monkeypatch):
    """W1 defense-in-depth requirement: GRM_GQA_BRIDGE_PARANOID=1 re-enables
    the O(N) signature walk on every call (bypassing the epoch-only fast
    path) and asserts it agrees with what the epoch says. Off (default):
    the fast path returns the SAME cached object with no walk when the
    epoch hasn't moved. On: an artificial under-invalidation (a mutation
    that changes the signature without bumping the epoch — the exact bug
    class this whole work order exists to prevent in production) must raise
    an AssertionError instead of silently serving a stale bank."""
    weak = np.asarray([[[0.1, 0.0]]], dtype=np.float32)
    arena = GQAArenaCache.__new__(GQAArenaCache)
    arena.grafts = [
        {"native_node_id": 1, "cent": weak.copy(), "rare": set(),
         "text": "a", "kind": "doc", "retired": False},
    ]
    arena._cuda_gqa_epoch = 0

    monkeypatch.delenv("GRM_GQA_BRIDGE_PARANOID", raising=False)
    bank1 = arena._cuda_route_bank_inputs()
    assert bank1 is not None
    bank2 = arena._cuda_route_bank_inputs()
    assert bank2 is bank1, (
        "fast path (epoch unchanged, paranoid off) must return the exact "
        "same cached object with no re-walk")

    monkeypatch.setenv("GRM_GQA_BRIDGE_PARANOID", "1")
    bank3 = arena._cuda_route_bank_inputs()
    assert bank3[2] == bank1[2], (
        "paranoid mode's forced walk must agree with the epoch-only "
        "decision when nothing is actually stale")

    # Artificial under-invalidation: mutate a signature field WITHOUT
    # bumping the epoch (the exact scenario every W1 instrumentation site
    # exists to prevent). Fast path (paranoid off) would silently miss
    # this; paranoid mode must catch it.
    arena.grafts[0]["cent"] = np.asarray([[[9.0, 0.0]]], np.float32)
    with pytest.raises(AssertionError, match="under-invalidation"):
        arena._cuda_route_bank_inputs()

    monkeypatch.delenv("GRM_GQA_BRIDGE_PARANOID", raising=False)
    # Fast path recovers once queried again -- NOTE: this call also updates
    # the cache off the stale-relative-to-epoch object since the epoch
    # still hasn't moved; this documents that paranoid mode is a TEST/DEBUG
    # aid; the epoch bump is what production correctness actually depends
    # on (W1's mutation-site instrumentation), not this assertion.


def test_gqa_arena_skips_cuda_route_for_lexical_queries(monkeypatch):
    class LexicalStore:
        def configure_cuda_gqa_route_bank(self, *_args, **_kwargs):
            raise AssertionError("lexical route should not attach CUDA bank")

        def route_gqa_cuda(self, *_args, **_kwargs):
            raise AssertionError("lexical route should not use CUDA")

        def route_gqa(self, *_args, **_kwargs):
            return [201, 202]

    monkeypatch.setenv("GRM_GQA_CUDA_ROUTE", "1")
    q = np.asarray([[[1.0, 0.0]]], dtype=np.float32)
    weak = np.asarray([[[0.1, 0.0]]], dtype=np.float32)
    good = np.asarray([[[1.0, 0.0]]], dtype=np.float32)

    arena = GQAArenaCache.__new__(GQAArenaCache)
    arena.native_store = LexicalStore()
    arena.last_route_backend = "python"
    arena.grafts = [
        {"native_node_id": 201, "cent": weak,
         "rare": {"a17"}, "text": "A17 weak", "kind": "doc"},
        {"native_node_id": 202, "cent": good,
         "rare": set(), "text": "good", "kind": "doc"},
    ]
    arena._probe_key = lambda _text: q

    assert arena.route("What about A17?", exclude=set(), limit=2) == [0, 1]
    assert arena.last_route_backend == "native"


def test_arena_route_falls_back_for_child_centroid_without_native_multikey():
    class SingleKeyStore:
        supports_multi_route_keys = False

        def route(self, *_args, **_kwargs):
            raise AssertionError("single-key native route should not be used")

    arena = ArenaCache.__new__(ArenaCache)
    arena.native_store = SingleKeyStore()
    arena.last_route_backend = "native"
    arena.grafts = [
        {"native_node_id": 0, "cent": np.array([1.0, 0.0], np.float32),
         "child_cents": [np.array([0.0, 1.0], np.float32)],
         "rare": set(), "text": "digest", "kind": "digest"},
    ]
    arena._probe_key = lambda _text: np.array([0.0, 1.0], np.float32)

    assert arena.route("plain probe", exclude=set()) == [0]
    assert arena.last_route_backend == "python"


def test_arena_python_route_drops_non_finite_scores():
    arena = ArenaCache.__new__(ArenaCache)
    arena.native_store = None
    arena.last_route_backend = "native"
    arena.grafts = [
        {"cent": np.array([np.nan, 0.0], np.float32),
         "rare": set(), "text": "nan", "kind": "doc"},
        {"cent": np.array([1.0, 0.0], np.float32),
         "rare": set(), "text": "finite", "kind": "doc"},
    ]
    arena._probe_key = lambda _text: np.array([1.0, 0.0], np.float32)

    assert arena.route("plain probe", exclude=set()) == [1]
    assert arena.last_route_backend == "python"


def test_arena_python_route_vectorizes_flat_mla_scores():
    arena = ArenaCache.__new__(ArenaCache)
    arena.native_store = None
    arena.last_route_backend = "native"
    arena.grafts = [
        {"cent": np.array([0.25, 0.9682458], np.float32),
         "rare": set(), "text": "weak", "kind": "doc"},
        {"cent": np.array([0.95, 0.3122499], np.float32),
         "rare": set(), "text": "semantic winner", "kind": "doc"},
        {"cent": np.array([0.40, 0.9165151], np.float32),
         "rare": {"a17"}, "text": "lexical", "kind": "doc"},
    ]
    arena._probe_key = lambda _text: np.array([1.0, 0.0], np.float32)

    def no_scalar_route(*_args, **_kwargs):
        raise AssertionError("flat MLA route should use vector scores")

    arena._cent_score = no_scalar_route

    assert arena.route("What about A17?", exclude=set()) == [2, 1, 0]
    assert arena.last_route_backend == "python"


def test_query_lex_tokens_keep_lowercase_content_words():
    """Query-side lex extension: lowercase labels enter the channel;
    pure stopwords and punctuation do not. Rare identifiers still included."""
    qlex = ArenaCache._query_lex_tokens(
        "Recall probe. What is the current orion pin value?")
    assert "orion" in qlex
    assert "pin" in qlex
    # stop / template glue must not dilute the channel
    assert "what" not in qlex
    assert "current" not in qlex
    assert "value" not in qlex
    assert "probe" not in qlex
    # rare survivors still present when present
    qlex2 = ArenaCache._query_lex_tokens("What about A17 and cypher bridge?")
    assert "a17" in qlex2
    assert "cypher" in qlex2
    assert "bridge" in qlex2


def test_query_side_lex_ranks_short_label_node_above_long_zero_overlap(monkeypatch):
    """OPTION 1: short value-bearing node outranks a longer zero-overlap
    node when the query names its label words. Latent stand-in favors the
    long node (length-bias proxy); content-word lex against node text flips
    the ranking. Node rare keys stay empty (lowercase labels never enter
    _rare_tokens) — query-side only."""
    monkeypatch.setenv("GRM_ROUTE_QUERY_LEX", "1")
    arena = ArenaCache.__new__(ArenaCache)
    arena.native_store = None
    arena.last_route_backend = "native"
    long_text = (
        "Session continuity filler smoke alpha. Acknowledge briefly. "
        "Do not change any stored fact values. " * 8)
    arena.grafts = [
        # longer zero-overlap node — latent winner without lex
        {"cent": np.array([1.0, 0.0], np.float32),
         "rare": set(), "text": long_text, "kind": "turn", "retired": False},
        # short value-bearing fact — lowercase labels only
        {"cent": np.array([0.85, 0.0], np.float32),
         "rare": set(),
         "text": "The current orion pin value is Kestrel-9-Tango.",
         "kind": "turn", "retired": False},
    ]
    arena._probe_key = lambda _text: np.array([1.0, 0.0], np.float32)
    arena._vector_route_scores = lambda _p, _cand: {0: 1.0, 1: 0.85}
    arena._normalize_scores = lambda base: base

    order = arena.route(
        "Recall probe. What is the current orion pin value?", exclude=set())
    assert order[0] == 1, f"expected short orion node first, got {order}"
    assert order[1] == 0
    assert arena.last_route_backend == "python"

    # Disabled: rare-only lex is empty → latent order wins
    monkeypatch.setenv("GRM_ROUTE_QUERY_LEX", "0")
    order_off = arena.route(
        "Recall probe. What is the current orion pin value?", exclude=set())
    assert order_off[0] == 0, f"disabled path should keep latent order: {order_off}"


def test_query_side_lex_silent_when_no_content_hit_keeps_latent(monkeypatch):
    """Content-word channel contributes +0 when no candidate text matches;
    ranking stays latent-driven (native path when available)."""
    monkeypatch.setenv("GRM_ROUTE_QUERY_LEX", "1")
    arena = ArenaCache.__new__(ArenaCache)
    arena.native_store = None
    arena.last_route_backend = "native"
    arena.grafts = [
        {"cent": np.array([1.0, 0.0], np.float32),
         "rare": set(), "text": "alpha filler continuity", "kind": "doc",
         "retired": False},
        {"cent": np.array([0.5, 0.0], np.float32),
         "rare": set(), "text": "beta filler continuity", "kind": "doc",
         "retired": False},
    ]
    arena._probe_key = lambda _text: np.array([1.0, 0.0], np.float32)
    # "orion pin" matches neither node → lex silent
    order = arena.route("What is the current orion pin value?", exclude=set())
    assert order == [0, 1]


def test_arena_descent_expand_uses_native_source_closure():
    class ClosureStore:
        def __init__(self):
            self.calls = []

        def source_closure(self, node_ids, max_depth=1,
                           include_roots=False):
            self.calls.append((tuple(node_ids), int(max_depth),
                               bool(include_roots)))
            return (102, 101)

    store = ClosureStore()
    arena = ArenaCache.__new__(ArenaCache)
    arena.native_store = store
    arena.grafts = [
        {"native_node_id": 100, "kind": "era", "sources": [1],
         "text": "era"},
        {"native_node_id": 101, "kind": "doc", "sources": [],
         "text": "python fallback child"},
        {"native_node_id": 102, "kind": "doc", "sources": [],
         "text": "native first child"},
        {"native_node_id": 103, "kind": "doc", "sources": [],
         "text": "ordinary pick"},
    ]

    assert arena._descent_expand([0, 3], ("era",), qrare=set()) == [
        2, 1, 3]
    assert store.calls == [((100,), 1, False)]


def test_arena_descent_expand_falls_back_to_python_sources():
    class EmptyClosureStore:
        def __init__(self):
            self.calls = 0

        def source_closure(self, *_args, **_kwargs):
            self.calls += 1
            return ()

    store = EmptyClosureStore()
    arena = ArenaCache.__new__(ArenaCache)
    arena.native_store = store
    arena.grafts = [
        {"native_node_id": 100, "kind": "era", "sources": [1],
         "text": "era"},
        {"native_node_id": 101, "kind": "doc", "sources": [],
         "text": "fallback child"},
    ]

    assert arena._descent_expand([0], ("era",), qrare=set()) == [1]
    assert store.calls == 1


def test_arena_descent_expand_keeps_identifier_filter_in_python():
    class UnusedClosureStore:
        def source_closure(self, *_args, **_kwargs):
            raise AssertionError("native closure should not handle qrare")

    arena = ArenaCache.__new__(ArenaCache)
    arena.native_store = UnusedClosureStore()
    arena.grafts = [
        {"native_node_id": 100, "kind": "era", "sources": [1, 2],
         "text": "era"},
        {"native_node_id": 101, "kind": "doc", "sources": [],
         "text": "target A17"},
        {"native_node_id": 102, "kind": "doc", "sources": [],
         "text": "other child"},
    ]

    assert arena._descent_expand([0], ("era",), qrare={"a17"}) == [1]


def test_repository_native_route_sync_includes_child_centroids(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        node_id = store.add_node("digest", b"", ntok=1)
        repo = GraftRepository.__new__(GraftRepository)
        repo.native_store = store
        repo._native_node_ids = {0: node_id}
        repo.arena = SimpleNamespace(grafts=[
            {"cent": np.array([1.0, 0.0], np.float32),
             "child_cents": [np.array([0.0, 1.0], np.float32)],
             "rare": set(), "text": "digest"}
        ])

        GraftRepository._native_set_route(repo, 0)
        assert store.route([0.0, 1.0], topk=1) == [node_id]


# ============================================================================
# MLA CUDA route P1 (docs/GRM_MLA_CUDA_ROUTE_PLAN.md /
# docs/GRM_MLA_CUDA_ROUTE_LEDGER.md "P0 complete" design constraints):
# limit-aware native topk + epoch-cached native_to_idx map, mirroring the
# GQA W1 epoch-cache pattern above but on the MLA (base ArenaCache) dialect.
# ============================================================================

def _mla_epoch_battery_arena():
    """A plain ArenaCache (MLA dialect) with three eligible flat-key nodes,
    no live model / no NativeGraftStore round trip -- mirrors
    _gqa_epoch_battery_arena for the MLA _native_to_idx_map cache."""
    arena = ArenaCache.__new__(ArenaCache)
    arena.native_store = None
    arena.last_route_backend = "python"
    arena.grafts = [
        {"native_node_id": 201, "cent": np.array([0.1, 0.0], np.float32),
         "rare": set(), "text": "weak", "kind": "doc", "retired": False},
        {"native_node_id": 202, "cent": np.array([1.0, 0.0], np.float32),
         "rare": set(), "text": "good", "kind": "doc", "retired": False},
        {"native_node_id": 203, "cent": np.array([2.0, 0.0], np.float32),
         "rare": set(), "text": "best", "kind": "doc", "retired": False},
    ]
    return arena


def test_mla_native_to_idx_epoch_bump_covers_every_signature_changing_mutation():
    """P1 fail-closed equivalence gate (plan-registered): for a battery of
    mutations covering the ArenaCache-side mutation set the MLA epoch now
    tracks (add, native_node_id replace, child_cents flip, retire, kind
    flip, clear), any mutation that changes what `_native_to_idx_map()` OR
    `_route_cand_base()` (P1 follow-on: the epoch-cached eligibility base,
    which reads `retired`/`kind`) would produce must also bump
    `_cuda_gqa_epoch` (now real on the MLA dialect too, per
    _bump_cuda_gqa_epoch). Mirrors
    test_gqa_cuda_epoch_bump_covers_every_signature_changing_mutation one
    dialect over."""
    arena = _mla_epoch_battery_arena()

    def snapshot():
        epoch = getattr(arena, "_cuda_gqa_epoch", 0)
        idx_to_native, ineligible = arena._native_to_idx_map()
        cand_base = arena._route_cand_base()
        return epoch, (tuple(sorted(idx_to_native.items())),
                       tuple(sorted(ineligible)),
                       tuple(cand_base))

    checked = 0

    def assert_mutation_is_fail_closed(mutate):
        nonlocal checked
        before_epoch, before_sig = snapshot()
        mutate()
        after_epoch, after_sig = snapshot()
        if after_sig != before_sig:
            assert after_epoch != before_epoch, (
                "signature changed but epoch did not bump — under-"
                "invalidation risk (MLA native_to_idx cache)")
        checked += 1

    def do_add():
        arena.grafts.append({"native_node_id": 204,
                             "cent": np.array([3.0, 0.0], np.float32),
                             "rare": set(), "text": "added", "kind": "doc",
                             "retired": False})
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_add)

    def do_native_id_replace():
        arena.grafts[0]["native_node_id"] = 9001
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_native_id_replace)

    def do_child_cents_flip():
        arena.grafts[1]["child_cents"] = [np.array([0.0, 1.0], np.float32)]
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_child_cents_flip)

    def do_native_id_clear():
        arena.grafts[2]["native_node_id"] = None
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_native_id_clear)

    def do_metadata_only():
        arena.grafts[0]["tags"] = ["noted"]
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_metadata_only)

    def do_retire():
        # retired IS an eligibility field for `_route_cand_base()` (P1
        # follow-on) — this mutation changes the cand-base signature and
        # the battery asserts the bump accompanies it.
        arena.grafts[0]["retired"] = True
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_retire)

    def do_kind_recall():
        # kind="recall" flips cand-base eligibility (route() excludes
        # recall-kind grafts) — the other NEW field the base reads.
        arena.grafts[1]["kind"] = "recall"
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_kind_recall)

    def do_clear():
        del arena.grafts[:]
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_clear)

    assert checked == 8


def test_mla_native_to_idx_map_fast_path_returns_same_cached_object():
    """No epoch movement -> _native_to_idx_map returns the exact same
    cached tuple object, no rebuild (the O(1) fast path this whole cache
    exists to provide)."""
    arena = _mla_epoch_battery_arena()
    arena._cuda_gqa_epoch = 0

    first = arena._native_to_idx_map()
    second = arena._native_to_idx_map()
    assert second is first

    arena._bump_cuda_gqa_epoch()
    third = arena._native_to_idx_map()
    assert third is not first
    assert third[0] == first[0]     # same content, rebuilt object


def test_mla_native_to_idx_map_under_invalidation_would_be_visible():
    """Defense-in-depth companion to the epoch-bump battery: if a mutation
    is (incorrectly) made WITHOUT bumping the epoch, the fast path must
    serve the stale map -- proving the epoch is load-bearing (not simply
    always rebuilt), i.e. that the fast-path branch is real and reachable,
    not dead code shadowed by an always-miss cache."""
    arena = _mla_epoch_battery_arena()
    arena._cuda_gqa_epoch = 0

    before = arena._native_to_idx_map()
    arena.grafts[0]["native_node_id"] = 9999   # mutate WITHOUT bumping
    stale = arena._native_to_idx_map()
    assert stale is before
    assert stale[0][0] == 201                  # old id, proves staleness

    arena._bump_cuda_gqa_epoch()
    fresh = arena._native_to_idx_map()
    assert fresh[0][0] == 9999


def test_mla_route_cand_base_fast_path_and_staleness():
    """P1 follow-on: _route_cand_base() must (a) return the exact same
    cached list object while the epoch is unchanged (O(1) fast path, no
    O(N) rebuild), (b) serve the stale base if a mutation skips the bump
    (proving the epoch is load-bearing), and (c) rebuild with correct
    content once the epoch moves."""
    arena = _mla_epoch_battery_arena()
    arena._cuda_gqa_epoch = 0

    first = arena._route_cand_base()
    assert first == [0, 1, 2]
    assert arena._route_cand_base() is first          # (a) identity

    arena.grafts[0]["retired"] = True                 # mutate WITHOUT bump
    assert arena._route_cand_base() is first          # (b) stale by design

    arena._bump_cuda_gqa_epoch()
    fresh = arena._route_cand_base()
    assert fresh is not first
    assert fresh == [1, 2]                            # (c) rebuilt

    arena.grafts[1]["kind"] = "recall"
    arena._bump_cuda_gqa_epoch()
    assert arena._route_cand_base() == [2]


def test_mla_route_cand_semantics_byte_identical_to_inline_comprehension():
    """Byte-identity gate for the cand refactor: for a battery of graft
    states mixing retired/recall/eligible and per-call excludes, route()'s
    candidate set (observed through the Python-fallback ranking, which
    consumes cand directly) must equal what the OLD inline comprehension
    `[i for i in range(len(grafts)) if i not in exclude and not retired
    and kind != recall]` produces — same members, same ascending order."""
    rng = np.random.default_rng(4242)
    arena = ArenaCache.__new__(ArenaCache)
    arena.native_store = None      # force the Python fallback (consumes cand)
    arena.last_route_backend = "python"

    for trial in range(12):
        n = int(rng.integers(1, 40))
        grafts = []
        for i in range(n):
            kind = rng.choice(["turn", "doc", "recall"], p=[0.6, 0.25, 0.15])
            grafts.append({
                "native_node_id": i,
                "cent": np.asarray([float(i + 1), 1.0], np.float32),
                "rare": set(), "text": f"node {i}", "kind": str(kind),
                "retired": bool(rng.random() < 0.2),
            })
        arena.grafts = grafts
        arena._bump_cuda_gqa_epoch()
        excl_n = int(rng.integers(0, n + 1))
        exclude = set(rng.choice(n, size=excl_n, replace=False).tolist())

        old_cand = [i for i in range(len(grafts))
                    if i not in exclude and not grafts[i].get("retired")
                    and grafts[i].get("kind", "turn") != "recall"]

        probe = np.asarray([1.0, 0.0], np.float32)
        arena._probe_key = lambda _t, q=probe: q
        ranking = arena.route("plain probe", exclude=exclude, limit=None)
        assert arena.last_route_backend == "python"
        # scores are strictly increasing with index (cent = [i+1, 1.0] vs
        # probe [1,0]), so the fallback ranking is old_cand reversed —
        # membership AND order both derive from cand.
        assert ranking == sorted(old_cand, reverse=True), (
            f"trial={trial} cand semantics diverged")


def test_mla_limited_route_prefix_parity_mixed_eligibility_fuzz(tmp_path):
    """Fuzz parity with retired/recall grafts mixed in (the NEW eligibility
    fields the cand-base cache reads) plus per-call excludes: limited-path
    native results must equal the prefix of the full-path results, and no
    retired/recall/excluded index may ever appear."""
    rng = np.random.default_rng(31337)
    dim = 12
    n_nodes = 160
    with _mla_native_store(tmp_path) as store:
        node_ids = []
        cents = _random_cents(rng, n_nodes, dim)
        for i in range(n_nodes):
            node_id = store.add_node(f"node {i}", b"", ntok=1)
            store.set_route(node_id, cents[i].tolist(), [])
            node_ids.append(node_id)
        arena = _mla_arena_from_store(store, node_ids, cents, rng)
        # mark a random ~25% retired and ~10% recall
        for i in range(n_nodes):
            r = rng.random()
            if r < 0.25:
                arena.grafts[i]["retired"] = True
            elif r < 0.35:
                arena.grafts[i]["kind"] = "recall"
        arena._bump_cuda_gqa_epoch()
        ineligible = {i for i, g in enumerate(arena.grafts)
                      if g.get("retired") or g.get("kind") == "recall"}

        for trial in range(6):
            probe = _random_cents(rng, 1, dim)[0]
            arena._probe_key = lambda _t, q=probe: q
            excl_n = int(rng.integers(0, 40))
            exclude = set(rng.choice(n_nodes, size=excl_n, replace=False).tolist())
            cand_n = n_nodes - len(ineligible | exclude)

            full = arena.route("probe", exclude=exclude, limit=None)
            assert arena.last_route_backend == "native"
            assert len(full) == cand_n
            assert not (set(full) & (ineligible | exclude))

            for limit in (0, 1, 3, 9, cand_n, cand_n + 2):
                got = arena.route("probe", exclude=exclude, limit=limit)
                want_len = min(max(0, limit), cand_n)
                assert got == full[:want_len], (
                    f"trial={trial} limit={limit} diverged")
                if want_len:
                    assert arena.last_route_backend == "native"


def _mla_native_store(tmp_path):
    lib = build_native(tmp_path)
    return NativeGraftStore(
        lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
        hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
        latent_rank=512, rope_dim=64)


def _mla_arena_from_store(store, node_ids, cents, rng, n_lexical=0):
    grafts = []
    for i, (node_id, cent) in enumerate(zip(node_ids, cents)):
        g = {"native_node_id": int(node_id),
             "cent": np.asarray(cent, dtype=np.float32),
             "rare": set(), "text": f"node {i}", "kind": "turn",
             "retired": False}
        grafts.append(g)
    arena = ArenaCache.__new__(ArenaCache)
    arena.native_store = store
    arena.last_route_backend = "python"
    arena.grafts = grafts
    return arena


def _random_cents(rng, n, dim):
    rows = rng.normal(size=(n, dim)).astype(np.float32)
    rows /= (np.linalg.norm(rows, axis=1, keepdims=True) + 1e-8)
    return rows


@pytest.mark.parametrize("n_nodes", [37, 1000])
def test_mla_limited_route_prefix_parity_vs_full_path(tmp_path, n_nodes):
    """NEW prefix-parity gate (plan-registered): for a battery of
    repos/shapes, limited-path (`limit=k`) native results must equal the
    prefix of the old full-path (`limit=None`) results for the SAME
    inputs. Reuses the real NativeGraftStore native runtime (not a Python
    fake) so the C ABI's own tie-breaking/ordering is exercised, not just
    the Python-side remap logic."""
    rng = np.random.default_rng(1234 + n_nodes)
    dim = 16
    with _mla_native_store(tmp_path) as store:
        node_ids = []
        cents = _random_cents(rng, n_nodes, dim)
        for i in range(n_nodes):
            node_id = store.add_node(f"node {i}", b"", ntok=1)
            store.set_route(node_id, cents[i].tolist(), [])
            node_ids.append(node_id)
        arena = _mla_arena_from_store(store, node_ids, cents, rng)
        probe = _random_cents(rng, 1, dim)[0]
        arena._probe_key = lambda _t, q=probe: q

        full = arena.route("probe", exclude=set(), limit=None)
        assert arena.last_route_backend == "native"
        assert len(full) == n_nodes

        for limit in (0, 1, 3, 7, n_nodes, n_nodes + 5):
            got = arena.route("probe", exclude=set(), limit=limit)
            assert arena.last_route_backend == "native"
            want_len = min(max(0, limit), n_nodes)
            assert got == full[:want_len], (
                f"limit={limit} diverged from full-path prefix")


def test_mla_limited_route_prefix_parity_with_excludes_and_fuzz(tmp_path):
    """Fuzz battery over random repos with mixed eligibility (excludes),
    the core edge case the slack design exists for: excluded winners that
    would otherwise occupy native's top ranks ahead of eligible candidates.
    """
    rng = np.random.default_rng(777)
    dim = 12
    n_nodes = 240
    with _mla_native_store(tmp_path) as store:
        node_ids = []
        cents = _random_cents(rng, n_nodes, dim)
        for i in range(n_nodes):
            node_id = store.add_node(f"node {i}", b"", ntok=1)
            store.set_route(node_id, cents[i].tolist(), [])
            node_ids.append(node_id)
        arena = _mla_arena_from_store(store, node_ids, cents, rng)

        for trial in range(8):
            probe = _random_cents(rng, 1, dim)[0]
            arena._probe_key = lambda _t, q=probe: q
            # random exclude set, sized so eligible candidates still exist
            excl_n = int(rng.integers(0, n_nodes - 5))
            exclude = set(rng.choice(n_nodes, size=excl_n, replace=False).tolist())
            cand_n = n_nodes - len(exclude)

            full = arena.route("probe", exclude=exclude, limit=None)
            assert arena.last_route_backend == "native"
            assert len(full) == cand_n

            for limit in (0, 1, 2, 5, cand_n, cand_n + 3):
                got = arena.route("probe", exclude=exclude, limit=limit)
                assert arena.last_route_backend == "native"
                want_len = min(max(0, limit), cand_n)
                assert got == full[:want_len], (
                    f"trial={trial} limit={limit} exclude_n={excl_n} "
                    "diverged from full-path prefix")


def test_mla_limited_route_prefix_parity_limit_larger_than_eligible(tmp_path):
    """Edge case: limit larger than the eligible count. Must return exactly
    the full eligible ranking, not error or truncate short."""
    rng = np.random.default_rng(55)
    dim = 8
    n_nodes = 20
    with _mla_native_store(tmp_path) as store:
        node_ids = []
        cents = _random_cents(rng, n_nodes, dim)
        for i in range(n_nodes):
            node_id = store.add_node(f"node {i}", b"", ntok=1)
            store.set_route(node_id, cents[i].tolist(), [])
            node_ids.append(node_id)
        arena = _mla_arena_from_store(store, node_ids, cents, rng)
        probe = _random_cents(rng, 1, dim)[0]
        arena._probe_key = lambda _t, q=probe: q

        full = arena.route("probe", exclude=set(), limit=None)
        got = arena.route("probe", exclude=set(), limit=10_000)
        assert got == full
        assert arena.last_route_backend == "native"


def test_mla_limited_route_prefix_parity_limit_zero(tmp_path):
    """Edge case: limit=0 must short-circuit to [] without ever calling
    native (matches the pre-P1 `route()`-level `route_limit == 0` guard,
    and the `_native_route_order`-level `want <= 0` guard)."""
    rng = np.random.default_rng(9)
    dim = 8
    with _mla_native_store(tmp_path) as store:
        node_id = store.add_node("only node", b"", ntok=1)
        cent = _random_cents(rng, 1, dim)[0]
        store.set_route(node_id, cent.tolist(), [])
        arena = _mla_arena_from_store(store, [node_id], [cent], rng)
        arena._probe_key = lambda _t: cent

        assert arena.route("probe", exclude=set(), limit=0) == []


def test_mla_limited_route_prefix_parity_excluded_winner(tmp_path):
    """Edge case named explicitly in the plan: the highest-scoring node is
    EXCLUDED by the caller. The limited path must still surface the next-
    best eligible node(s), not silently drop the window short because the
    excluded winner consumed slack."""
    dim = 4
    with _mla_native_store(tmp_path) as store:
        cents = [
            np.array([1.0, 0.0, 0.0, 0.0], np.float32),   # best, excluded
            np.array([0.9, 0.1, 0.0, 0.0], np.float32),   # 2nd best
            np.array([0.1, 0.9, 0.0, 0.0], np.float32),   # 3rd
            np.array([0.0, 0.0, 1.0, 0.0], np.float32),   # weak
        ]
        node_ids = []
        for i, c in enumerate(cents):
            node_id = store.add_node(f"node {i}", b"", ntok=1)
            store.set_route(node_id, c.tolist(), [])
            node_ids.append(node_id)
        arena = _mla_arena_from_store(store, node_ids, cents, None)
        probe = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
        arena._probe_key = lambda _t: probe

        full_no_exclude = arena.route("probe", exclude=set(), limit=None)
        assert full_no_exclude[0] == 0     # node 0 is the true winner

        excluded_full = arena.route("probe", exclude={0}, limit=None)
        assert 0 not in excluded_full
        assert len(excluded_full) == 3

        limited = arena.route("probe", exclude={0}, limit=2)
        assert limited == excluded_full[:2]
        assert arena.last_route_backend == "native"


def test_mla_limited_route_non_finite_scores_present(tmp_path):
    """Edge case: a probe/route combination that produces a non-finite
    score for some node must still be handled fail-closed (M6 NaN law:
    native drops non-finite scores rather than sorting them; a legitimate
    resulting shortfall vs the requested topk must not be mistaken for the
    slack-under-covered failure mode)."""
    dim = 4
    with _mla_native_store(tmp_path) as store:
        cents = [
            np.array([1.0, 0.0, 0.0, 0.0], np.float32),
            np.array([np.nan, 0.0, 0.0, 0.0], np.float32),
            np.array([0.5, 0.5, 0.0, 0.0], np.float32),
            np.array([0.0, 1.0, 0.0, 0.0], np.float32),
        ]
        node_ids = []
        for i, c in enumerate(cents):
            node_id = store.add_node(f"node {i}", b"", ntok=1)
            store.set_route(node_id, c.tolist(), [])
            node_ids.append(node_id)
        arena = _mla_arena_from_store(store, node_ids, cents, None)
        probe = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
        arena._probe_key = lambda _t: probe

        full = arena.route("probe", exclude=set(), limit=None)
        # the NaN node must never appear (M6 law) -- 3 finite nodes survive
        assert 1 not in full
        assert len(full) == 3

        for limit in (1, 2, 3, 10):
            got = arena.route("probe", exclude=set(), limit=limit)
            want_len = min(limit, 3)
            assert got == full[:want_len]
            assert 1 not in got
            assert arena.last_route_backend == "native"


# --------------------------------------------------------------------------
# MLA CUDA route lifecycle (P2, docs/GRM_MLA_CUDA_ROUTE_PLAN.md). Mirrors the
# GQA 6-selector battery above (test_native_gqa_cuda_route_requires_explicit_
# bank / test_native_gqa_route_mutation_clears_cuda_bank / test_native_gqa_
# eligibility_mutation_clears_cuda_bank / test_gqa_arena_uses_opt_in_cuda_
# route_bank / test_gqa_arena_rebuilds_cuda_route_bank_when_rows_change /
# test_gqa_cuda_epoch_bump_covers_every_signature_changing_mutation) for the
# base ArenaCache (MLA dialect) instead of GQAArenaCache. The store-level
# 3-selector (requires-explicit-bank / mutation-clears-bank / eligibility-
# mutation-clears-bank) lives earlier in this file next to its GQA sibling;
# these are the arena-level 3 plus the epoch-fail-closed extension.


def test_mla_arena_uses_opt_in_cuda_route_bank(monkeypatch):
    class CudaStore:
        def __init__(self):
            self.configured = None
            self.cuda_calls = []
            self._cuda_mla_bank = None

        def configure_cuda_mla_route_bank(self, route_bank, node_ids):
            self.configured = (
                np.asarray(route_bank).copy(),
                np.asarray(node_ids, dtype=np.uint64).copy(),
            )
            self._cuda_mla_bank = object()

        def route_mla_cuda(self, query, *, topk=3, **_kwargs):
            self.cuda_calls.append((np.asarray(query).shape, int(topk)))
            return [102, 101][:int(topk)]

        def route(self, *_args, **_kwargs):
            raise AssertionError("CPU MLA route should not run")

    monkeypatch.setenv("GRM_MLA_CUDA_ROUTE", "1")
    q = np.asarray([1.0, 0.0], dtype=np.float32)
    weak = np.asarray([0.1, 0.0], dtype=np.float32)
    good = np.asarray([1.0, 0.0], dtype=np.float32)

    arena = ArenaCache.__new__(ArenaCache)
    arena.native_store = CudaStore()
    arena.last_route_backend = "python"
    arena.grafts = [
        {"native_node_id": 101, "cent": weak,
         "rare": set(), "text": "weak", "kind": "doc", "retired": False},
        {"native_node_id": 102, "cent": good,
         "rare": set(), "text": "good", "kind": "doc", "retired": False},
    ]
    arena._probe_key = lambda _text: q

    assert arena.route("plain probe", exclude=set(), limit=2) == [1, 0]
    assert arena.last_route_backend == "cuda"
    bank, node_ids = arena.native_store.configured
    assert bank.shape == (2, 2)
    assert node_ids.tolist() == [101, 102]
    assert arena.native_store.cuda_calls == [((2,), 2)]


def test_mla_arena_rebuilds_cuda_route_bank_when_rows_change(monkeypatch):
    class RebuildStore:
        def __init__(self):
            self.configured_node_ids = []
            self.configure_calls = 0
            self._cuda_mla_bank = None
            self._cuda_mla_bank_signature = None

        def configure_cuda_mla_route_bank(self, _route_bank, node_ids):
            self.configure_calls += 1
            self.configured_node_ids = [
                int(x) for x in np.asarray(node_ids, dtype=np.uint64)]
            self._cuda_mla_bank = object()

        def route_mla_cuda(self, _query, *, topk=3, **_kwargs):
            return list(reversed(self.configured_node_ids))[:int(topk)]

        def route(self, *_args, **_kwargs):
            raise AssertionError("CPU MLA route should not run")

    monkeypatch.setenv("GRM_MLA_CUDA_ROUTE", "1")
    q = np.asarray([1.0, 0.0], dtype=np.float32)
    weak = np.asarray([0.1, 0.0], dtype=np.float32)
    good = np.asarray([1.0, 0.0], dtype=np.float32)
    best = np.asarray([2.0, 0.0], dtype=np.float32)

    arena = ArenaCache.__new__(ArenaCache)
    arena.native_store = RebuildStore()
    arena.last_route_backend = "python"
    arena.grafts = [
        {"native_node_id": 101, "cent": weak,
         "rare": set(), "text": "weak", "kind": "doc", "retired": False},
        {"native_node_id": 102, "cent": good,
         "rare": set(), "text": "good", "kind": "doc", "retired": False},
    ]
    arena._probe_key = lambda _text: q

    assert arena.route("plain probe", exclude=set(), limit=1) == [1]
    first_signature = arena.native_store._cuda_mla_bank_signature

    arena.grafts.append(
        {"native_node_id": 103, "cent": best,
         "rare": set(), "text": "best", "kind": "doc", "retired": False})
    # The epoch is the sole hot-path staleness gate -- a raw external
    # append (this test pokes arena.grafts directly, bypassing deposit()/
    # graft_repository.py's instrumented mutation sites) must bump the
    # epoch itself, exactly as every real production mutation path does.
    arena._bump_cuda_gqa_epoch()

    assert arena.route("plain probe", exclude=set(), limit=1) == [2]
    assert arena.native_store.configure_calls == 2
    assert arena.native_store.configured_node_ids == [101, 102, 103]
    assert arena.native_store._cuda_mla_bank_signature != first_signature


def test_mla_arena_skips_cuda_route_for_lexical_queries(monkeypatch):
    """Mirrors test_gqa_arena_skips_cuda_route_for_lexical_queries: the
    dense CUDA MLA bank carries no lexical bonus, so any probe with a
    lexical channel (`qrare` non-empty) must never attach/use it."""
    class LexicalStore:
        def configure_cuda_mla_route_bank(self, *_args, **_kwargs):
            raise AssertionError("lexical route should not attach CUDA bank")

        def route_mla_cuda(self, *_args, **_kwargs):
            raise AssertionError("lexical route should not use CUDA")

        def route(self, *_args, **_kwargs):
            return [201, 202]

    monkeypatch.setenv("GRM_MLA_CUDA_ROUTE", "1")
    q = np.asarray([1.0, 0.0], dtype=np.float32)
    weak = np.asarray([0.1, 0.0], dtype=np.float32)
    good = np.asarray([1.0, 0.0], dtype=np.float32)

    arena = ArenaCache.__new__(ArenaCache)
    arena.native_store = LexicalStore()
    arena.last_route_backend = "python"
    arena.grafts = [
        {"native_node_id": 201, "cent": weak,
         "rare": {"a17"}, "text": "A17 weak", "kind": "doc", "retired": False},
        {"native_node_id": 202, "cent": good,
         "rare": set(), "text": "good", "kind": "doc", "retired": False},
    ]
    arena._probe_key = lambda _text: q

    assert arena.route("What about A17?", exclude=set(), limit=2) == [0, 1]
    assert arena.last_route_backend == "native"


def _mla_cuda_route_epoch_battery_arena():
    """An ArenaCache (MLA dialect) with three eligible dense-bank nodes,
    wired the same way GQA's `_gqa_epoch_battery_arena` is -- no live
    model, no NativeGraftStore round trip, just the Python route-bank
    cache machinery under test. Distinct from the earlier P1
    `_mla_epoch_battery_arena` fixture (node ids 201/202/203) -- same
    shape, different fixture, to avoid colliding with that pre-existing
    module-level name."""
    weak = np.asarray([0.1, 0.0], dtype=np.float32)
    good = np.asarray([1.0, 0.0], dtype=np.float32)
    best = np.asarray([2.0, 0.0], dtype=np.float32)
    arena = ArenaCache.__new__(ArenaCache)
    arena.native_store = None
    arena.last_route_backend = "python"
    arena.grafts = [
        {"native_node_id": 101, "cent": weak.copy(),
         "rare": set(), "text": "weak", "kind": "doc", "retired": False},
        {"native_node_id": 102, "cent": good.copy(),
         "rare": set(), "text": "good", "kind": "doc", "retired": False},
        {"native_node_id": 103, "cent": best.copy(),
         "rare": set(), "text": "best", "kind": "doc", "retired": False},
    ]
    return arena


def test_mla_cuda_epoch_bump_covers_every_signature_changing_mutation():
    """P2 fail-closed equivalence gate, mirrors
    test_gqa_cuda_epoch_bump_covers_every_signature_changing_mutation: for a
    battery of mutations covering the ArenaCache-side mutation set the MLA
    CUDA bank's epoch tracks (add, cent replace, kind change, metadata-only,
    revision, expire, retire, clear), any mutation that changes the arena's
    MLA CUDA bank signature (`_cuda_route_bank_signature`) must also bump
    the shared epoch (`_cuda_gqa_epoch` -- real on both dialects since P1).
    Equivalently: epoch-valid implies signature-equal."""
    arena = _mla_cuda_route_epoch_battery_arena()

    def snapshot():
        epoch = getattr(arena, "_cuda_gqa_epoch", 0)
        sig = arena._cuda_route_bank_signature()
        signature = None if sig is None else sig[1]
        return epoch, signature

    checked = 0

    def assert_mutation_is_fail_closed(mutate):
        nonlocal checked
        before_epoch, before_sig = snapshot()
        mutate()
        after_epoch, after_sig = snapshot()
        if after_sig != before_sig:
            assert after_epoch != before_epoch, (
                "signature changed but epoch did not bump — under-"
                "invalidation risk")
        checked += 1

    def do_add():
        arena.grafts.append({"native_node_id": 104,
                             "cent": np.asarray([3.0, 0.0], np.float32),
                             "rare": set(), "text": "added", "kind": "doc",
                             "retired": False})
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_add)

    def do_cent_replace():
        arena.grafts[0]["cent"] = np.asarray([9.0, 0.0], np.float32)
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_cent_replace)

    def do_kind_change():
        arena.grafts[1]["kind"] = "recall"
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_kind_change)

    def do_metadata_only():
        arena.grafts[2]["tags"] = ["noted"]
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_metadata_only)

    def do_revision():
        arena.grafts[2]["retired"] = True
        arena.grafts.append({"native_node_id": 105,
                             "cent": np.asarray([4.0, 0.0], np.float32),
                             "rare": set(), "text": "revised", "kind": "doc",
                             "retired": False})
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_revision)

    def do_expire():
        arena.grafts[0]["retired"] = True
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_expire)

    def do_retire():
        arena.grafts[3]["retired"] = True
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_retire)

    def do_clear():
        del arena.grafts[:]
        arena._bump_cuda_gqa_epoch()
    assert_mutation_is_fail_closed(do_clear)

    assert checked == 8


def test_mla_cuda_bridge_never_reuses_stale_bank_across_mutation_battery():
    """Behavior half of the P2 regression, mirrors test_gqa_cuda_bridge_
    never_reuses_stale_bank_across_mutation_battery: after each mutation in
    the same battery, a bridged (CUDA) MLA route must return the same node
    ordering a fresh attach would -- no stale bank reuse."""
    class TrackingStore:
        def __init__(self):
            self.configured_node_ids = None
            self.configure_calls = 0
            self._cuda_mla_bank = None
            self._cuda_mla_bank_signature = None

        def configure_cuda_mla_route_bank(self, _route_bank, node_ids):
            self.configure_calls += 1
            self.configured_node_ids = [
                int(x) for x in np.asarray(node_ids, dtype=np.uint64)]
            self._cuda_mla_bank = object()

        def route_mla_cuda(self, _query, *, topk=3, **_kwargs):
            return list(self.configured_node_ids)[:int(topk)]

        def route(self, *_args, **_kwargs):
            raise AssertionError("CPU MLA route should not run")

    import os as _os
    _os.environ["GRM_MLA_CUDA_ROUTE"] = "1"
    try:
        arena = _mla_cuda_route_epoch_battery_arena()
        arena.native_store = TrackingStore()
        q = np.asarray([1.0, 0.0], dtype=np.float32)
        arena._probe_key = lambda _text: q

        def eligible_native_ids():
            return sorted(
                int(g["native_node_id"]) for g in arena.grafts
                if not g.get("retired") and g.get("kind", "turn") != "recall")

        def assert_bridge_matches_fresh_attach():
            expect = eligible_native_ids()
            got_order = arena.route("plain probe", exclude=set(),
                                    limit=len(expect) or 1)
            got_native_ids = sorted(
                int(arena.grafts[i]["native_node_id"]) for i in got_order)
            assert got_native_ids == expect or not expect
            assert sorted(arena.native_store.configured_node_ids) == expect, (
                "bridged MLA route bank does not match current eligibility "
                "— stale bank reuse")

        assert_bridge_matches_fresh_attach()

        arena.grafts.append({"native_node_id": 104,
                             "cent": np.asarray([3.0, 0.0], np.float32),
                             "rare": set(), "text": "added", "kind": "doc",
                             "retired": False})
        arena._bump_cuda_gqa_epoch()
        assert_bridge_matches_fresh_attach()

        arena.grafts[0]["retired"] = True
        arena._bump_cuda_gqa_epoch()
        assert_bridge_matches_fresh_attach()

        arena.grafts[1]["kind"] = "recall"
        arena._bump_cuda_gqa_epoch()
        assert_bridge_matches_fresh_attach()

        del arena.grafts[:]
        arena._bump_cuda_gqa_epoch()
        # route() short-circuits to [] at the top when self.grafts is
        # empty (before _cuda_route_order is ever reached), so there is no
        # bank to compare against here -- mirrors the GQA precedent's
        # final step exactly.
        result = arena.route("plain probe", exclude=set(), limit=1)
        assert result == []
    finally:
        _os.environ.pop("GRM_MLA_CUDA_ROUTE", None)


def test_mla_cuda_route_falls_back_on_out_of_range_row_id(monkeypatch):
    """Sidecar/bank fail-closed contract: a degenerate query (e.g. every
    node scores non-finite, the M6 all-NaN-query edge case measured in the
    P2 fuzz battery) makes the raw CUDA top-k kernel emit UINT64_MAX
    sentinel row ids for the unfilled slots. CudaMLARouteBank.route_topk
    bounds-checks and raises (mirrors CudaGQARouteBank exactly); the arena's
    `_cuda_route_order` catches that and returns None, so `route()` falls
    through to the CPU path rather than surfacing garbage node ids."""
    class DegenerateBank:
        def route_topk(self, *_args, **_kwargs):
            raise RuntimeError("CUDA MLA route returned an out-of-range row ID")

    class DegenerateStore:
        def __init__(self):
            self._cuda_mla_bank = None
            self._cuda_mla_bank_signature = None

        def configure_cuda_mla_route_bank(self, _route_bank, node_ids):
            self._cuda_mla_bank = DegenerateBank()
            self._cuda_mla_bank_signature = tuple(
                int(x) for x in np.asarray(node_ids, dtype=np.uint64))

        def route_mla_cuda(self, _query, *, topk=3, **_kwargs):
            raise RuntimeError(
                "CUDA MLA route returned an out-of-range row ID")

        def route(self, _query, _lexical, *, topk=3, **_kwargs):
            return [101, 102][:int(topk)]

    monkeypatch.setenv("GRM_MLA_CUDA_ROUTE", "1")
    q = np.asarray([1.0, 0.0], dtype=np.float32)
    weak = np.asarray([0.1, 0.0], dtype=np.float32)
    good = np.asarray([1.0, 0.0], dtype=np.float32)

    arena = ArenaCache.__new__(ArenaCache)
    arena.native_store = DegenerateStore()
    arena.last_route_backend = "python"
    arena.grafts = [
        {"native_node_id": 101, "cent": weak,
         "rare": set(), "text": "weak", "kind": "doc", "retired": False},
        {"native_node_id": 102, "cent": good,
         "rare": set(), "text": "good", "kind": "doc", "retired": False},
    ]
    arena._probe_key = lambda _text: q

    assert arena.route("plain probe", exclude=set(), limit=2) == [0, 1]
    assert arena.last_route_backend == "native"
