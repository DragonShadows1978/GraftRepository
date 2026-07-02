import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from core.graft_arena import ArenaCache, GQAArenaCache
from core.graft_repository import GraftRepository
from core.grm_native import NativeGraftStore


ROOT = "/mnt/ForgeRealm/GraftRepository"


def build_native(tmp_path):
    lib = tmp_path / "libgrm_runtime.so"
    subprocess.run([
        "g++", "-std=c++17", "-shared", "-fPIC",
        "-I", f"{ROOT}/cpp",
        f"{ROOT}/cpp/grm_runtime.cpp",
        "-o", str(lib),
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
