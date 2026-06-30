import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from core.graft_arena import ArenaCache
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


def test_native_store_lifecycle_via_ctypes(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="DeepSeekV2Lite_TC", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as store:
        assert store.dialect_id() == "DeepSeekV2Lite_TC:27x2048:r512"

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


def test_native_store_gqa_dialect_via_ctypes(tmp_path):
    lib = build_native(tmp_path)
    with NativeGraftStore(
            lib, model_type="Qwen3_TC", num_layers=36,
            hidden_dim=2560, vals_per_tok_layer=2048, route_layer=0,
            payload_kind="gqa", num_kv_heads=8, head_dim=128) as store:
        assert store.dialect_id() == "Qwen3_TC:36x2560:g8x128"

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
        assert store.parse_memory_command("flush memory now") == {
            "action": "flush",
            "flush_immediately": False,
        }
        with pytest.raises(RuntimeError, match="unknown memory command"):
            store.parse_memory_command("remember maybe someday")


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
        store.save_checkpoint(ckpt)
        assert store.stats().dirty_nodes == 0
        assert store.stats().durable_nodes == 1

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
