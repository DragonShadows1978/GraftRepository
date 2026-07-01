import json
import os

import numpy as np
import pytest

from core.graft_repository import GraftRepository
from core.grm_runtime import GRMRuntime
from core.grm_native import NativeGraftStore
from tests.test_grm_native_runtime import build_native


class FakeConfig:
    num_layers = 27
    hidden_dim = 2048
    kv_lora_rank = 512
    qk_rope_head_dim = 64


class FakeModel:
    config = FakeConfig()

    def __init__(self):
        self.layers = [object() for _ in range(self.config.num_layers)]


class FakeGQAConfig:
    num_layers = 12
    hidden_dim = 1536
    num_kv_heads = 2
    head_dim = 128


class FakeGQAModel:
    config = FakeGQAConfig()

    def __init__(self):
        self.layers = [object() for _ in range(self.config.num_layers)]


class FakeAbsoluteConfig(FakeGQAConfig):
    position_embedding_type = "learned_absolute"


class FakeAbsoluteModel(FakeGQAModel):
    config = FakeAbsoluteConfig()


class FakeHybridConfig(FakeGQAConfig):
    num_layers = 32
    hidden_dim = 4096
    num_kv_heads = 4
    head_dim = 256
    full_attention_interval = 4
    conv_kernel = 4


class FakeHybridModel(FakeGQAModel):
    config = FakeHybridConfig()


class FakeGemmaStyleConfig(FakeGQAConfig):
    num_layers = 48
    hidden_dim = 3840
    num_kv_heads = 8
    head_dim = 256
    sliding_window = 1024
    num_global_kv_heads = 1


class FakeGemmaStyleModel(FakeGQAModel):
    config = FakeGemmaStyleConfig()


class FakeArena:
    PAYLOAD = (("c", 1), ("kpe", 2))
    VALS_PER_TOK_LAYER = 576
    ENABLE_ERA_FOLDING = True

    def __init__(self, model, encode, decode, route_layer=3, **_):
        self.m = model
        self.encode = encode
        self.decode = decode
        self.route_layer = route_layer
        self.live_segs = []
        self.grafts = []
        self.node_loader = None

    @staticmethod
    def _rare_tokens(text):
        return {w.lower() for w in text.split() if any(c.isdigit() for c in w)}

    def deposit(self, text):
        idx = len(self.grafts)
        payload = {"id": idx}
        self.grafts.append({
            "h": payload,
            "cent": np.full((512,), float(idx), dtype=np.float32),
            "ntok": len(self.encode(text)),
            "text": text,
        })
        return idx

    def feed(self, turn_text):
        idx = self.deposit(turn_text)
        self.grafts[idx]["kind"] = "turn"
        self.live_segs.append((idx, self.grafts[idx]["ntok"]))

    def step(self, user_text, ngen=64, max_trips=2):
        answer = f"Recorded {user_text}"
        self.feed(f"User: {user_text}\nAssistant: {answer}\n")
        return answer, {"ngen": ngen, "max_trips": max_trips, "mounts": []}

    def pack_node(self, h):
        return {"payload_id": np.asarray([h["id"]], dtype=np.int64)}

    def unpack_node(self, z):
        return {"id": int(z["payload_id"][0])}

    def pack_index(self):
        cents = [g["cent"] for g in self.grafts]
        return {"cents": np.stack(cents) if cents else np.zeros((0, 512),
                                                                np.float32)}

    def unpack_index(self, z, i):
        return z["cents"][i].astype(np.float32)


class FakeSliceArena(FakeArena):
    PAYLOAD = (("tok", 0),)
    VALS_PER_TOK_LAYER = 1

    def deposit(self, text):
        idx = len(self.grafts)
        toks = self.encode(text)
        payload = {"tok": np.arange(len(toks), dtype=np.int64)}
        self.grafts.append({
            "h": payload,
            "cent": self._node_key(text),
            "ntok": len(toks),
            "text": text,
        })
        return idx

    def pack_node(self, h):
        return {"tok": np.asarray(h["tok"], dtype=np.int64)}

    def unpack_node(self, z):
        return {"tok": np.asarray(z["tok"], dtype=np.int64)}

    def _node_key(self, text, h_host=None):
        return np.full((512,), float(len(self.encode(text))),
                       dtype=np.float32)


def enc(text):
    return text.split()


def dec(ids):
    return " ".join(ids)


def temp_artifacts(root):
    out = []
    for dirpath, _, names in os.walk(root):
        for name in names:
            if ".tmp." in name:
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def test_dialect_descriptor_uses_deepseek_mla_dimensions(tmp_path):
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3)

    assert repo.dialect == "FakeModel:27x2048:r512"
    assert repo.dialect_desc.payload_kind == "mla"
    assert repo.dialect_desc.latent_rank == 512
    assert repo.dialect_desc.rope_dim == 64
    assert repo.dialect_desc.vals_per_tok_layer == 576
    assert repo.dialect_desc.route_layer == 3
    assert repo.dialect_desc.position_law == "rope_partial_mla"
    assert repo.dialect_desc.state_kind == "mla_latent_plus_rope"
    assert repo.dialect_desc.graftability == "seat_remountable"
    assert repo.dialect_desc.remountable is True
    assert repo.dialect_desc.composition == "multi_mount"


def test_dialect_descriptor_records_graftability_profiles(tmp_path):
    gqa = GraftRepository(FakeGQAModel(), enc, dec, str(tmp_path / "gqa"),
                          autosave=False, arena_cls=FakeArena,
                          route_layer=0)
    assert gqa.dialect == "FakeGQAModel:12x1536:g2x128"
    assert gqa.dialect_desc.payload_kind == "gqa"
    assert gqa.dialect_desc.position_law == "rope_full_kv"
    assert gqa.dialect_desc.graftability == "seat_remountable"
    assert gqa.dialect_desc.remountable is True

    absolute = GraftRepository(
        FakeAbsoluteModel(), enc, dec, str(tmp_path / "absolute"),
        autosave=False, arena_cls=FakeArena, route_layer=0)
    assert absolute.dialect_desc.position_law == "learned_absolute"
    assert absolute.dialect_desc.graftability == "same_position_restore"
    assert absolute.dialect_desc.remountable is False
    assert absolute.dialect_desc.composition == "prefix_restore_only"

    hybrid = GraftRepository(
        FakeHybridModel(), enc, dec, str(tmp_path / "hybrid"),
        autosave=False, arena_cls=FakeArena, route_layer=0)
    assert hybrid.dialect_desc.position_law == (
        "rope_attention_plus_recurrent_state")
    assert hybrid.dialect_desc.state_kind == "hybrid_kv_recurrent"
    assert hybrid.dialect_desc.graftability == "prefix_restore_only"
    assert hybrid.dialect_desc.remountable is False
    assert hybrid.dialect_desc.composition == "single_prefix_state"

    gemma = GraftRepository(
        FakeGemmaStyleModel(), enc, dec, str(tmp_path / "gemma"),
        autosave=False, arena_cls=FakeArena, route_layer=0)
    assert gemma.dialect_desc.position_law == "rope_mixed_sliding_global"
    assert gemma.dialect_desc.state_kind == "mixed_window_kv"
    assert gemma.dialect_desc.graftability == "window_limited_remountable"
    assert gemma.dialect_desc.remountable is True
    assert gemma.dialect_desc.composition == "bounded_window_multi_mount"


def test_dialect_descriptor_accepts_arena_graftability_override(tmp_path):
    class PrefixOnlyArena(FakeArena):
        POSITION_LAW = "external_absolute"
        STATE_KIND = "kv"
        GRAFTABILITY = "same_position_restore"
        REMOUNTABLE = "false"
        COMPOSITION = "single_prefix_state"

    repo = GraftRepository(FakeGQAModel(), enc, dec, str(tmp_path),
                           autosave=False, arena_cls=PrefixOnlyArena,
                           route_layer=0)

    assert repo.dialect_desc.position_law == "external_absolute"
    assert repo.dialect_desc.graftability == "same_position_restore"
    assert repo.dialect_desc.remountable is False
    assert repo.dialect_desc.composition == "single_prefix_state"


def test_dirty_flush_and_reload_lifecycle(tmp_path):
    path = str(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                           arena_cls=FakeArena, route_layer=3)

    idx = repo.add_document("DOC code 73-4412", tags=("ops",))
    g = repo.arena.grafts[idx]

    assert g["dirty"] is True
    assert g["durable"] is False
    assert g["host_present"] is True
    assert g["device_present"] is True
    assert g["host_payload"]["payload_id"].tolist() == [0]
    queued = repo.flush_async()
    assert queued["queued_nodes"] == 1
    assert queued["dirty_bytes"] > 0
    assert repo.flush_wait(timeout=5.0) is True
    assert repo.dirty_nodes == {}

    with open(os.path.join(path, "wal", "000001.wal")) as fh:
        wal_types = [json.loads(line)["type"] for line in fh]
    assert "NODE_UPSERT" in wal_types
    assert "CHECKPOINT" in wal_types

    assert repo.dirty_nodes == {}
    assert g["dirty"] is False
    assert g["durable"] is True
    assert g["saved"] is True

    with open(os.path.join(path, "manifest.json")) as fh:
        man = json.load(fh)

    assert man["dialect"] == "FakeModel:27x2048:r512"
    assert man["dialect_descriptor"]["latent_rank"] == 512
    assert man["dialect_descriptor"]["position_law"] == "rope_partial_mla"
    assert man["dialect_descriptor"]["graftability"] == "seat_remountable"
    assert man["dialect_descriptor"]["remountable"] is True
    assert man["nodes"][0]["host_present"] is True
    assert man["nodes"][0]["durable"] is True
    assert man["nodes"][0]["dirty"] is False
    assert man["nodes"][0]["metadata"]["kind"] == "doc"
    assert man["nodes"][0]["provenance"][0]["segment_type"] == "doc_span"

    reloaded = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                               arena_cls=FakeArena, route_layer=3)

    assert reloaded.stats()["dirty_nodes"] == 0
    assert reloaded.stats()["durable_nodes"] == 1
    assert reloaded.arena.grafts[0]["host_present"] is True
    assert reloaded.arena.grafts[0]["host_payload"]["payload_id"].tolist() == [0]
    assert reloaded.arena.grafts[0]["device_present"] is True
    assert reloaded.arena.grafts[0]["h"] == {"id": 0}
    assert temp_artifacts(path) == []


def test_atomic_json_publish_preserves_prior_file_on_failure(tmp_path):
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3)
    target = os.path.join(str(tmp_path), "manifest_probe.json")

    repo._atomic_write_json(target, {"version": 1}, indent=1)
    with pytest.raises(TypeError):
        repo._atomic_write_json(target, {"bad": object()}, indent=1)

    with open(target) as fh:
        assert json.load(fh) == {"version": 1}
    assert temp_artifacts(str(tmp_path)) == []


def test_atomic_npz_publish_preserves_prior_file_on_failure(tmp_path, monkeypatch):
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3)
    target = os.path.join(str(tmp_path), "probe.npz")
    repo._atomic_savez_compressed(
        target, payload_id=np.asarray([7], dtype=np.int64))

    original = np.savez_compressed

    def fail_after_write(file, **payload):
        file.write(b"partial")
        raise RuntimeError("forced npz publish failure")

    monkeypatch.setattr(
        "core.graft_repository.np.savez_compressed", fail_after_write)
    with pytest.raises(RuntimeError, match="forced npz"):
        repo._atomic_savez_compressed(
            target, payload_id=np.asarray([8], dtype=np.int64))

    monkeypatch.setattr(
        "core.graft_repository.np.savez_compressed", original)
    with np.load(target) as z:
        assert z["payload_id"].tolist() == [7]
    assert temp_artifacts(str(tmp_path)) == []


def test_cull_graft_slices_payloads_and_preserves_lineage(tmp_path):
    lib = build_native(tmp_path)
    path = str(tmp_path / "repo")
    repo = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                           arena_cls=FakeSliceArena, route_layer=3,
                           native_lib_path=lib)
    parent = repo.add_document("A1 B2 C3 D4 E5 F6", tags=("long",))

    out = repo.cull_graft(parent, max_tokens=2)

    assert out["action"] == "cull_graft"
    assert out["parent"] == parent
    assert out["retired_parent"] is True
    assert out["children"] == [1, 2, 3]
    assert repo.arena.grafts[parent]["retired"] is True
    assert repo.arena.grafts[parent]["metadata"]["active"] is False
    assert repo.arena.grafts[parent]["metadata"]["culled_into"] == [1, 2, 3]

    expected = [
        ("A1 B2", [0, 1], 0, 2),
        ("C3 D4", [2, 3], 2, 4),
        ("E5 F6", [4, 5], 4, 6),
    ]
    for child, (text, payload, start, end) in zip(out["children"], expected):
        g = repo.arena.grafts[child]
        assert g["text"] == text
        assert g["ntok"] == 2
        assert g["sources"] == [parent]
        assert g["host_payload"]["tok"].tolist() == payload
        assert g["h"] is None
        assert g["metadata"]["source_grafts"] == [parent]
        assert g["metadata"]["culled_from"] == parent
        assert g["metadata"]["token_start"] == start
        assert g["metadata"]["token_end"] == end
        assert g["metadata"]["supersedes"] == [parent]
        assert g["provenance"][0]["segment_type"] == "cull_span"
        native_id = g["native_node_id"]
        assert repo.native_store.get_tensor(native_id, "tok").tolist() == payload

    repo.flush_now()
    repo.close()

    reloaded = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                               arena_cls=FakeSliceArena, route_layer=3,
                               native_lib_path=lib)
    assert reloaded.arena.grafts[parent]["retired"] is True
    assert reloaded.show_memory_about("A1 B2 C3") == []
    assert len(reloaded.show_memory_about("A1 B2")) == 1
    assert reloaded.arena.grafts[1]["host_payload"]["tok"].tolist() == [0, 1]
    assert reloaded.arena.grafts[2]["host_payload"]["tok"].tolist() == [2, 3]
    assert reloaded.arena.grafts[3]["host_payload"]["tok"].tolist() == [4, 5]
    assert reloaded.arena.grafts[1]["metadata"]["source_grafts"] == [parent]
    assert reloaded.stats()["native"]["nodes"] == 4
    reloaded.close()


def test_cull_graft_uses_native_slice_when_available(tmp_path, monkeypatch):
    lib = build_native(tmp_path)
    path = str(tmp_path / "repo")
    repo = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                           arena_cls=FakeSliceArena, route_layer=3,
                           native_lib_path=lib)
    parent = repo.add_document("N1 N2 N3 N4")
    calls = []
    original = repo.native_store.slice_tensor

    def traced(node_id, name, axis, start, length):
        calls.append((int(node_id), name, int(axis), int(start), int(length)))
        return original(node_id, name, axis, start, length)

    monkeypatch.setattr(repo.native_store, "slice_tensor", traced)

    out = repo.cull_graft(parent, max_tokens=2)

    assert out["children"] == [1, 2]
    assert calls == [
        (repo.arena.grafts[parent]["native_node_id"], "tok", 0, 0, 2),
        (repo.arena.grafts[parent]["native_node_id"], "tok", 0, 2, 2),
    ]
    assert repo.arena.grafts[1]["host_payload"]["tok"].tolist() == [0, 1]
    assert repo.arena.grafts[2]["host_payload"]["tok"].tolist() == [2, 3]
    repo.close()


def test_cull_graft_sections_uses_intentional_boundaries(tmp_path):
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path),
                           autosave=False, arena_cls=FakeSliceArena,
                           route_layer=3)
    parent = repo.add_document(
        "# Alpha\nA1 A2\n\n"
        "# Beta\nB1 B2 B3\n"
        "User: C1 C2\n"
        "Assistant: D1 D2")

    assert repo.plan_cull_sections(parent) == [
        (0, 4), (4, 9), (9, 12), (12, 15)]

    out = repo.cull_graft_sections(parent)

    assert out["children"] == [1, 2, 3, 4]
    assert repo.arena.grafts[parent]["retired"] is True
    expected_payloads = [
        [0, 1, 2, 3],
        [4, 5, 6, 7, 8],
        [9, 10, 11],
        [12, 13, 14],
    ]
    for child, payload in zip(out["children"], expected_payloads):
        got = repo.arena.grafts[child]["host_payload"]["tok"].tolist()
        assert got == payload
        assert repo.arena.grafts[child]["metadata"]["culled_from"] == parent


def test_plan_cull_sections_caps_large_sections(tmp_path):
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path),
                           autosave=False, arena_cls=FakeSliceArena,
                           route_layer=3)
    parent = repo.add_document("P1 P2 P3 P4\n\nQ1 Q2")

    assert repo.plan_cull_sections(
        parent, boundary="paragraph", max_tokens=2) == [
            (0, 2), (2, 4), (4, 6)]


def test_runtime_cull_graft_autosaves_and_reports(tmp_path):
    path = str(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, path, autosave=True,
                           arena_cls=FakeSliceArena, route_layer=3)
    parent = repo.add_document("R1 R2 R3 R4")
    repo.flush_now()

    out = repo.cull_graft(parent, max_tokens=2)

    assert out["action"] == "cull_graft"
    assert out["children"] == [1, 2]
    assert repo.runtime.last_result.event == "cull"
    assert repo.runtime.last_result.action == "cull_graft"
    assert repo.runtime.last_result.autosaved is True
    assert repo.runtime.last_result.new_nodes == (1, 2)
    assert repo.stats()["dirty_nodes"] == 0

    reopened = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                               arena_cls=FakeSliceArena, route_layer=3)
    assert reopened.arena.grafts[parent]["retired"] is True
    assert reopened.arena.grafts[1]["host_payload"]["tok"].tolist() == [0, 1]
    assert reopened.arena.grafts[2]["host_payload"]["tok"].tolist() == [2, 3]


def test_memory_command_culls_graft_with_token_cap(tmp_path):
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path),
                           autosave=False, arena_cls=FakeSliceArena,
                           route_layer=3)
    parent = repo.add_document("A1 B2 C3 D4")

    out = repo.apply_memory_command("cull graft 0 max tokens 2")

    assert out == {"action": "cull_graft", "parent": parent,
                   "children": [1, 2], "retired_parent": True}
    assert repo.runtime.last_result.event == "memory_command"
    assert repo.runtime.last_result.action == "cull_graft"
    assert repo.runtime.last_result.new_nodes == (1, 2)
    assert repo.arena.grafts[parent]["retired"] is True
    assert repo.arena.grafts[1]["host_payload"]["tok"].tolist() == [0, 1]
    assert repo.arena.grafts[2]["host_payload"]["tok"].tolist() == [2, 3]


def test_native_memory_command_culls_graft_by_sections(tmp_path):
    lib = build_native(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path / "repo"),
                           autosave=False, arena_cls=FakeSliceArena,
                           route_layer=3, native_lib_path=lib)
    parent = repo.add_document("# Alpha\nA1 A2\n\n# Beta\nB1 B2")

    out = repo.apply_memory_command("split graft 0 into sections")

    assert out["children"] == [1, 2]
    assert repo.runtime.last_result.event == "memory_command"
    assert repo.runtime.last_result.action == "cull_graft"
    assert repo.arena.grafts[parent]["metadata"]["culled_into"] == [1, 2]
    assert repo.arena.grafts[1]["host_payload"]["tok"].tolist() == [0, 1, 2, 3]
    assert repo.arena.grafts[2]["host_payload"]["tok"].tolist() == [4, 5, 6, 7]
    repo.close()


def test_remember_attaches_semantic_metadata(tmp_path):
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3)

    idx = repo.remember("The GRM runtime is RAM-first.",
                        durability="project",
                        mutability="stable",
                        scope="project",
                        kind="task_state",
                        metadata={"subject": "GRM runtime"})
    meta = repo.arena.grafts[idx]["metadata"]

    assert meta["kind"] == "task_state"
    assert meta["durability"] == "project"
    assert meta["mutability"] == "stable"
    assert meta["scope"] == "project"
    assert meta["write_intent"] == "user_asserted"
    assert meta["subject"] == "GRM runtime"

    repo.flush_now()
    with open(os.path.join(str(tmp_path), "manifest.json")) as fh:
        man = json.load(fh)
    assert man["nodes"][0]["metadata"]["subject"] == "GRM runtime"


def test_memory_commands_revision_and_review(tmp_path):
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3)

    out = repo.apply_memory_command(
        "remember this for the project: GRM uses RAM as the hot store")
    assert out["action"] == "remember"
    assert repo.show_memory_about("RAM as the hot store")

    corr = repo.apply_memory_command(
        "correct memory: RAM as the hot store => GRM uses RAM as the authoritative live store")
    assert corr["action"] == "correct"
    active = repo.show_memory_about("authoritative live store")
    assert len(active) == 1
    assert active[0]["metadata"]["supersedes"] == [0]
    assert repo.arena.grafts[0]["metadata"]["active"] is False
    assert repo.arena.grafts[0]["retired"] is True

    rid = repo.review_candidate("possible extracted fact", confidence=0.4)
    approved = repo.approve_review(rid)
    assert repo.arena.grafts[approved]["metadata"]["confidence"] == 0.4
    assert repo.review_buffer[rid]["status"] == "approved"

    why = repo.why_remember("authoritative live store")
    assert why[0]["write_intent"] == "user_asserted"


def test_memory_commands_autosave_all_mutations(tmp_path):
    path = str(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, path, autosave=True,
                           arena_cls=FakeArena, route_layer=3)

    remembered = repo.apply_memory_command(
        "remember this for the project: Runtime command autosave target 71-7171")
    assert remembered["action"] == "remember"
    assert repo.runtime.last_result.action == "remember"
    assert repo.runtime.last_result.autosaved is True
    assert repo.stats()["dirty_nodes"] == 0

    corrected = repo.apply_memory_command(
        "correct memory: command autosave target => Runtime command autosave replacement 72-7272")
    assert corrected["action"] == "correct"
    assert repo.runtime.last_result.action == "correct"
    assert repo.runtime.last_result.autosaved is True
    assert repo.runtime.last_result.new_nodes == (corrected["node_id"],)
    assert repo.stats()["dirty_nodes"] == 0

    review = repo.apply_memory_command(
        "update memory: runtime fallback missing separator")
    assert review["action"] == "review"
    assert repo.runtime.last_result.action == "review"
    assert repo.runtime.last_result.autosaved is True

    ignored = repo.apply_memory_command("do not remember this transient note")
    assert ignored["action"] == "ignore"
    assert repo.runtime.last_result.action == "ignore"
    assert repo.runtime.last_result.autosaved is True

    forgotten = repo.apply_memory_command("forget: replacement 72-7272")
    assert forgotten == {"action": "forget", "count": 1}
    assert repo.runtime.last_result.action == "forget"
    assert repo.runtime.last_result.autosaved is True
    assert repo.stats()["dirty_nodes"] == 0

    with open(os.path.join(path, "manifest.json")) as fh:
        man = json.load(fh)
    assert man["nodes"][remembered["node_id"]]["retired"] is True
    assert man["nodes"][corrected["node_id"]]["retired"] is True
    assert man["review_buffer"][-1]["reason"] == (
        "correction missing => separator")

    reopened = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                               arena_cls=FakeArena, route_layer=3)
    assert reopened.show_memory_about("71-7171") == []
    assert reopened.show_memory_about("72-7272") == []
    assert reopened.review_buffer[-1]["status"] == "pending"


def test_memory_control_commands_update_and_read_metadata(tmp_path):
    path = str(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, path, autosave=True,
                           arena_cls=FakeArena, route_layer=3)

    remembered = repo.apply_memory_command(
        "remember this for the project: Control target 73-7373")
    node_id = remembered["node_id"]

    pinned = repo.apply_memory_command("pin memory: Control target")
    assert pinned == {"count": 1, "node_ids": [node_id],
                      "action": "pin_memory"}
    assert repo.runtime.last_result.event == "memory_command"
    assert repo.runtime.last_result.action == "pin_memory"
    assert repo.runtime.last_result.autosaved is True
    assert repo.arena.grafts[node_id]["metadata"]["pinned"] is True
    assert repo.stats()["dirty_nodes"] == 0

    marked = repo.apply_memory_command("mark memory mutable: Control target")
    assert marked == {"count": 1, "node_ids": [node_id],
                      "action": "mark_mutable"}
    assert repo.arena.grafts[node_id]["metadata"]["mutability"] == "mutable"
    assert repo.runtime.last_result.autosaved is True

    shown = repo.apply_memory_command("show memory about: Control target")
    assert shown["action"] == "show_memory"
    assert shown["rows"][0]["node_id"] == node_id
    assert repo.runtime.last_result.action == "show_memory"
    assert repo.runtime.last_result.autosaved is False

    why = repo.apply_memory_command("why do you remember: Control target")
    assert why["action"] == "why_memory"
    assert why["rows"][0]["node_id"] == node_id
    assert why["rows"][0]["mutability"] == "mutable"
    assert repo.runtime.last_result.autosaved is False

    with open(os.path.join(path, "manifest.json")) as fh:
        man = json.load(fh)
    meta = man["nodes"][node_id]["metadata"]
    assert meta["pinned"] is True
    assert meta["mutability"] == "mutable"


def test_memory_mode_commands_change_wal_behavior(tmp_path):
    path = str(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                           arena_cls=FakeArena, route_layer=3)
    assert repo.durability_mode == "session_safe"
    assert repo.wal_enabled is True

    volatile = repo.apply_memory_command("switch to volatile mode")
    assert volatile["action"] == "set_durability_mode"
    assert volatile["durability_mode"] == "volatile"
    assert volatile["wal_enabled"] is False
    assert repo.runtime.last_result.action == "set_durability_mode"
    assert repo.stats()["wal_enabled"] is False

    with open(os.path.join(path, "wal", "000001.wal")) as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    assert records[-1]["type"] == "CONFIG"
    assert records[-1]["durability_mode"] == "volatile"
    before = len(records)

    repo.add_document("Volatile-only document 81-8181")
    with open(os.path.join(path, "wal", "000001.wal")) as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    assert len(records) == before

    safe = repo.apply_memory_command("switch to session-safe mode")
    assert safe["durability_mode"] == "session_safe"
    assert safe["wal_enabled"] is True
    repo.add_document("Session-safe document 82-8282")
    with open(os.path.join(path, "wal", "000001.wal")) as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    assert records[-2]["type"] == "CONFIG"
    assert records[-2]["durability_mode"] == "session_safe"
    assert records[-1]["type"] == "NODE_UPSERT"
    assert records[-1]["text"] == "Session-safe document 82-8282"


def test_durability_mode_recovers_from_manifest_and_wal(tmp_path):
    manifest_path = str(tmp_path / "manifest_mode")
    repo = GraftRepository(FakeModel(), enc, dec, manifest_path,
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3)
    repo.apply_memory_command("switch to volatile mode")
    repo.flush_now()

    reopened = GraftRepository(FakeModel(), enc, dec, manifest_path,
                               autosave=False, arena_cls=FakeArena,
                               route_layer=3)
    assert reopened.durability_mode == "volatile"
    assert reopened.wal_enabled is False
    assert reopened.stats()["durability_mode"] == "volatile"

    replay_path = str(tmp_path / "wal_mode")
    repo = GraftRepository(FakeModel(), enc, dec, replay_path,
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3)
    repo.flush_now()
    repo.apply_memory_command("switch to project-safe mode")

    replayed = GraftRepository(FakeModel(), enc, dec, replay_path,
                               autosave=False, arena_cls=FakeArena,
                               route_layer=3)
    assert replayed.durability_mode == "project_safe"
    assert replayed.wal_enabled is True
    assert replayed.stats()["replayed_config_records"] == 1


def test_project_safe_forces_explicit_project_memory_flush(tmp_path):
    session_path = str(tmp_path / "session_safe")
    session_repo = GraftRepository(FakeModel(), enc, dec, session_path,
                                   autosave=False, arena_cls=FakeArena,
                                   route_layer=3,
                                   durability_mode="session_safe")

    session = session_repo.apply_memory_command(
        "remember this for the project: Session-safe project memory 83-8383")

    assert session["action"] == "remember"
    assert session_repo.runtime.last_result.autosaved is False
    assert session_repo.stats()["dirty_nodes"] == 1
    assert not os.path.exists(os.path.join(session_path, "manifest.json"))

    project_path = str(tmp_path / "project_safe")
    project_repo = GraftRepository(FakeModel(), enc, dec, project_path,
                                   autosave=False, arena_cls=FakeArena,
                                   route_layer=3,
                                   durability_mode="project_safe")

    project = project_repo.apply_memory_command(
        "remember this for the project: Project-safe memory 84-8484")

    assert project["action"] == "remember"
    assert project_repo.runtime.last_result.autosaved is True
    assert project_repo.stats()["dirty_nodes"] == 0
    with open(os.path.join(project_path, "manifest.json")) as fh:
        man = json.load(fh)
    assert man["durability_mode"] == "project_safe"
    assert man["nodes"][project["node_id"]]["text"] == (
        "Project-safe memory 84-8484")

    session_note = project_repo.apply_memory_command(
        "remember this for this session: Project-safe session scratch 85-8585")
    assert session_note["action"] == "remember"
    assert project_repo.runtime.last_result.autosaved is False
    assert project_repo.stats()["dirty_nodes"] == 1


def test_runtime_coordinator_drives_chat_flush_and_extraction(tmp_path):
    class Extractor:
        def extract(self, text, **ctx):
            self.last = (text, ctx)
            return [{
                "action": "write_direct",
                "candidate_type": "fact",
                "text": "Runtime chat captured code WEST-31.",
                "subject": "runtime chat",
                "predicate": "captured",
                "value": "WEST-31",
                "scope": "project",
                "durability": "project",
                "mutability": "stable",
                "write_intent": "observed",
                "confidence": 0.99,
            }]

    extractor = Extractor()
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path),
                           autosave=True, arena_cls=FakeArena,
                           route_layer=3, extractor=extractor)

    ans, info = repo.chat("Please store code WEST-31", ngen=7, max_trips=1)

    assert ans == "Recorded Please store code WEST-31"
    assert isinstance(repo.runtime, GRMRuntime)
    assert repo.runtime.last_result.event == "chat"
    assert repo.runtime.last_result.autosaved is True
    assert repo.runtime.last_result.before_nodes == 0
    assert repo.runtime.last_result.after_nodes == 2
    assert info["extraction"][0]["action"] == "write_direct"
    assert extractor.last[1]["context"]["event"] == "chat"
    assert repo.stats()["dirty_nodes"] == 0
    assert os.path.exists(os.path.join(str(tmp_path), "manifest.json"))
    extracted_rows = repo.show_memory_about("Runtime chat captured")
    assert extracted_rows[0]["metadata"]["source_grafts"] == [0]


def test_review_buffer_edit_reject_and_manifest_wal_replay(tmp_path):
    path = str(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                           arena_cls=FakeArena, route_layer=3)

    rid = repo.review_candidate("draft extracted fact", confidence=0.2)
    rejected = repo.review_candidate("bad extracted fact", confidence=0.1)
    repo.flush_now()

    edited = repo.edit_review(
        rid,
        text="Edited extracted fact 91-2222",
        proposed_scope="user",
        proposed_durability="permanent",
        confidence=0.85,
        metadata={"subject": "review fact", "value": "91-2222"},
        reason="user edited candidate")
    assert edited["status"] == "pending"
    assert edited["text"] == "Edited extracted fact 91-2222"

    scoped = repo.change_review_scope(rid, "project", durability="project")
    assert scoped["proposed_scope"] == "project"
    assert scoped["proposed_durability"] == "project"

    rej = repo.reject_review(rejected, reason="user rejected")
    assert rej["status"] == "rejected"
    with pytest.raises(RuntimeError, match="rejected review"):
        repo.approve_review(rejected)

    reopened = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                               arena_cls=FakeArena, route_layer=3)
    assert reopened.review_buffer[rid]["text"] == "Edited extracted fact 91-2222"
    assert reopened.review_buffer[rid]["proposed_scope"] == "project"
    assert reopened.review_buffer[rid]["metadata"]["value"] == "91-2222"
    assert reopened.review_buffer[rejected]["status"] == "rejected"

    approved = reopened.approve_review(rid)
    g = reopened.arena.grafts[approved]
    assert g["text"] == "Edited extracted fact 91-2222"
    assert g["metadata"]["scope"] == "project"
    assert g["metadata"]["durability"] == "project"
    assert g["metadata"]["subject"] == "review fact"
    assert reopened.review_buffer[rid]["status"] == "approved"
    assert reopened.approve_review(rid) == approved
    assert len(reopened.arena.grafts) == 1


def test_manifest_review_replay_ignores_stale_low_lsn_records(tmp_path):
    path = str(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                           arena_cls=FakeArena, route_layer=3)
    rid = repo.review_candidate("stale review replay candidate 17-1717",
                                confidence=0.2)
    repo.flush_now()

    with open(os.path.join(path, "manifest.json")) as fh:
        man = json.load(fh)
    stale = {
        "lsn": man["wal_lsn"],
        "type": "REVIEW_REJECT",
        "time": 0,
        "review_id": rid,
        "reason": "stale low-lsn replay",
    }
    with open(os.path.join(path, "wal", "000001.wal"), "a") as fh:
        fh.write(json.dumps(stale, sort_keys=True) + "\n")

    reopened = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                               arena_cls=FakeArena, route_layer=3)

    assert reopened.review_buffer[rid]["status"] == "pending"
    assert "rejection_reason" not in reopened.review_buffer[rid]


def test_runtime_review_execution_autosaves_and_reports(tmp_path):
    path = str(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, path, autosave=True,
                           arena_cls=FakeArena, route_layer=3)

    rid = repo.review_candidate(
        "Runtime approved fact 81-8181",
        confidence=0.7,
        metadata={"subject": "review runtime", "value": "81-8181"})
    rejected = repo.review_candidate("Runtime rejected fact", confidence=0.1)

    edited = repo.edit_review(
        rid,
        proposed_scope="project",
        proposed_durability="project",
        confidence=0.92,
        reason="ready for approval")
    assert edited["status"] == "pending"
    assert repo.runtime.last_result.event == "review"
    assert repo.runtime.last_result.action == "edit_review"
    assert repo.runtime.last_result.autosaved is True
    with open(os.path.join(path, "manifest.json")) as fh:
        man = json.load(fh)
    assert man["review_buffer"][rid]["confidence"] == 0.92

    rej = repo.reject_review(rejected, reason="not useful")
    assert rej["status"] == "rejected"
    assert repo.runtime.last_result.action == "reject_review"
    with open(os.path.join(path, "manifest.json")) as fh:
        man = json.load(fh)
    assert man["review_buffer"][rejected]["status"] == "rejected"

    approved = repo.approve_review(rid)
    assert repo.runtime.last_result.action == "approve_review"
    assert repo.runtime.last_result.new_nodes == (approved,)
    assert repo.runtime.last_result.autosaved is True
    assert repo.stats()["dirty_nodes"] == 0
    rows = repo.show_memory_about("Runtime approved fact")
    assert rows[0]["node_id"] == approved
    assert rows[0]["metadata"]["subject"] == "review runtime"

    again = repo.approve_review(rid)
    assert again == approved
    assert repo.runtime.last_result.new_nodes == ()


def test_librarian_respects_arena_era_folding_capability(tmp_path):
    class NoEraArena(FakeArena):
        ENABLE_ERA_FOLDING = False

    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path),
                           autosave=False, arena_cls=NoEraArena,
                           route_layer=3)
    for i in range(6):
        idx = repo.add_document(f"DIGEST candidate {i} code {i}-0000")
        repo.arena.grafts[idx]["kind"] = "digest"
        repo.arena.grafts[idx]["metadata"]["kind"] = "digest"

    repo.TURNS_HIGH = 100
    repo.DIGESTS_HIGH = 6

    assert repo.fold_pending() == 0


def test_native_memory_command_parser_drives_repository_policy(tmp_path):
    lib = build_native(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path / "repo"),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3, native_lib_path=lib)

    out = repo.apply_memory_command(
        "remember permanently: Native parser owns command grammar")
    assert out["action"] == "remember"
    assert repo.runtime.last_result.event == "memory_command"
    assert repo.runtime.last_result.action == "remember"
    assert repo.runtime.last_result.autosaved is True
    g = repo.arena.grafts[out["node_id"]]
    assert g["metadata"]["durability"] == "permanent"
    assert g["metadata"]["scope"] == "user"
    assert g["metadata"]["kind"] == "fact"
    assert g["durable"] is True
    assert repo.stats()["native"]["durable_nodes"] == 1

    corr = repo.apply_memory_command(
        "correct memory: command grammar => native parser command grammar")
    assert corr["action"] == "correct"
    active = repo.show_memory_about("native parser command grammar")
    assert len(active) == 1
    assert active[0]["metadata"]["supersedes"] == [out["node_id"]]

    pin = repo.apply_memory_command("pin memory: native parser command grammar")
    assert pin["action"] == "pin_memory"
    assert pin["node_ids"] == [active[0]["node_id"]]
    assert repo.arena.grafts[active[0]["node_id"]]["metadata"]["pinned"] is True

    review = repo.apply_memory_command("update memory: missing arrow")
    assert review["action"] == "review"
    assert repo.review_buffer[-1]["reason"] == (
        "correction missing => separator")
    repo.close()


def test_extraction_candidate_interface_is_conservative(tmp_path):
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3)

    direct = {
        "action": "write_direct",
        "candidate_type": "fact",
        "text": "GRM hot tier is RAM-first.",
        "subject": "GRM hot tier",
        "predicate": "is",
        "value": "RAM-first",
        "scope": "project",
        "durability": "project",
        "mutability": "stable",
        "write_intent": "observed",
        "confidence": 0.97,
    }
    out = repo.apply_extraction_candidate(direct)
    assert out["action"] == "write_direct"
    meta = repo.arena.grafts[out["node_id"]]["metadata"]
    assert meta["subject"] == "GRM hot tier"
    assert meta["value"] == "RAM-first"
    assert meta["write_intent"] == "observed"

    low = dict(direct, text="Low confidence fact", subject="low",
               predicate="is", value="uncertain", confidence=0.4)
    review = repo.apply_extraction_candidate(low)
    assert review["action"] == "review_candidate"
    assert repo.review_buffer[-1]["reason"] == (
        "confidence below direct-write threshold")
    assert repo.review_buffer[-1]["metadata"]["value"] == "uncertain"

    inferred_conflict = dict(direct, text="GRM hot tier is NVMe-first.",
                             value="NVMe-first", confidence=0.99,
                             write_intent="inferred")
    conflict = repo.apply_extraction_candidate(inferred_conflict)
    assert conflict["action"] == "review_candidate"
    assert repo.review_buffer[-1]["reason"] == "conflicts with active memory"
    assert repo.arena.grafts[out["node_id"]]["metadata"]["active"] is True

    scoped_fact = dict(direct, text="For the user, GRM hot tier is manual.",
                       value="manual", scope="user", confidence=0.99)
    scoped = repo.apply_extraction_candidate(scoped_fact)
    assert scoped["action"] == "write_direct"
    assert repo.arena.grafts[scoped["node_id"]]["metadata"]["scope"] == "user"

    user_correction = dict(direct, text="GRM hot tier is GPU-mounted only.",
                           value="GPU-mounted only",
                           write_intent="user_asserted",
                           confidence=1.0)
    corr = repo.apply_extraction_candidate(user_correction)
    assert corr["action"] == "supersede_existing"
    assert corr["supersedes"] == [out["node_id"]]
    assert repo.arena.grafts[out["node_id"]]["metadata"]["active"] is False
    assert repo.arena.grafts[scoped["node_id"]]["metadata"]["active"] is True
    assert repo.arena.grafts[out["node_id"]]["retired"] is True
    assert repo.arena.grafts[corr["node_id"]]["metadata"]["supersedes"] == [
        out["node_id"]]


def test_extraction_duplicate_fact_reinforces_existing_node(tmp_path):
    path = str(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, path,
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3)

    fact = {
        "action": "write_direct",
        "candidate_type": "fact",
        "text": "GRM duplicate guard is ACTIVE-7.",
        "subject": "GRM duplicate guard",
        "predicate": "is",
        "value": "ACTIVE-7",
        "scope": "project",
        "durability": "project",
        "mutability": "stable",
        "write_intent": "observed",
        "confidence": 0.97,
        "source_grafts": [3],
    }
    out = repo.apply_extraction_candidate(fact)
    assert out["action"] == "write_direct"

    duplicate = dict(fact, text="GRM duplicate guard remains ACTIVE-7.",
                     confidence=0.99, source_grafts=[4],
                     write_intent="user_asserted")
    reinforced = repo.apply_extraction_candidate(duplicate)

    assert reinforced == {
        "action": "reinforce_existing",
        "node_id": out["node_id"],
    }
    assert len(repo.arena.grafts) == 1
    meta = repo.arena.grafts[out["node_id"]]["metadata"]
    assert meta["source_grafts"] == [3, 4]
    assert meta["confidence"] == 0.99
    assert meta["write_intent"] == "user_asserted"
    assert meta["reinforcement_count"] == 1
    assert "reinforced_at" in meta
    assert repo.runtime.last_result.action == "reinforce_existing"
    assert repo.runtime.last_result.new_nodes == ()

    reopened = GraftRepository(FakeModel(), enc, dec, path,
                               autosave=False, arena_cls=FakeArena,
                               route_layer=3)
    assert len(reopened.arena.grafts) == 1
    recovered = reopened.arena.grafts[out["node_id"]]["metadata"]
    assert recovered["source_grafts"] == [3, 4]
    assert recovered["confidence"] == 0.99
    assert recovered["write_intent"] == "user_asserted"
    assert recovered["reinforcement_count"] == 1


def test_low_confidence_duplicate_fact_still_goes_to_review(tmp_path):
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3)
    fact = {
        "action": "write_direct",
        "candidate_type": "fact",
        "text": "GRM review duplicate is SAFE-3.",
        "subject": "GRM review duplicate",
        "predicate": "is",
        "value": "SAFE-3",
        "scope": "project",
        "durability": "project",
        "mutability": "stable",
        "write_intent": "observed",
        "confidence": 0.99,
    }
    out = repo.apply_extraction_candidate(fact)
    duplicate = dict(fact, text="GRM review duplicate is still SAFE-3.",
                     confidence=0.2)

    review = repo.apply_extraction_candidate(duplicate)

    assert review["action"] == "review_candidate"
    assert repo.review_buffer[-1]["reason"] == (
        "confidence below direct-write threshold")
    assert len(repo.arena.grafts) == 1
    assert repo.arena.grafts[out["node_id"]]["metadata"].get(
        "reinforcement_count") is None


def test_approved_duplicate_review_reinforces_existing_fact(tmp_path):
    path = str(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, path,
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3)
    fact = {
        "action": "write_direct",
        "candidate_type": "fact",
        "text": "Review duplicate guard is ACTIVE-9.",
        "subject": "review duplicate guard",
        "predicate": "is",
        "value": "ACTIVE-9",
        "scope": "project",
        "durability": "project",
        "mutability": "stable",
        "write_intent": "observed",
        "confidence": 0.98,
    }
    out = repo.apply_extraction_candidate(fact)
    rid = repo.review_candidate(
        "Review duplicate guard remains ACTIVE-9.",
        confidence=0.4,
        metadata={"subject": "review duplicate guard",
                  "predicate": "is",
                  "value": "ACTIVE-9",
                  "source_grafts": [17],
                  "review_evidence": "manual approval"})

    approved = repo.approve_review(rid)

    assert approved == out["node_id"]
    assert repo.review_buffer[rid]["approved_action"] == "reinforce_existing"
    assert repo.runtime.last_result.action == "approve_review"
    assert repo.runtime.last_result.new_nodes == ()
    assert len(repo.arena.grafts) == 1
    meta = repo.arena.grafts[out["node_id"]]["metadata"]
    assert meta["source_grafts"] == [17]
    assert meta["review_evidence"] == "manual approval"
    assert meta["write_intent"] == "user_asserted"
    assert meta["reinforcement_count"] == 1

    reopened = GraftRepository(FakeModel(), enc, dec, path,
                               autosave=False, arena_cls=FakeArena,
                               route_layer=3)
    assert len(reopened.arena.grafts) == 1
    assert reopened.review_buffer[rid]["approved_node_id"] == out["node_id"]
    assert reopened.review_buffer[rid]["approved_action"] == "reinforce_existing"
    assert reopened.arena.grafts[out["node_id"]]["metadata"][
        "reinforcement_count"] == 1
    assert reopened.arena.grafts[out["node_id"]]["metadata"][
        "review_evidence"] == "manual approval"


def test_approved_conflicting_review_supersedes_existing_fact(tmp_path):
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3)
    old = repo.apply_extraction_candidate({
        "action": "write_direct",
        "candidate_type": "fact",
        "text": "Review conflict target is OLD.",
        "subject": "review conflict target",
        "predicate": "is",
        "value": "OLD",
        "scope": "project",
        "durability": "project",
        "mutability": "stable",
        "write_intent": "observed",
        "confidence": 0.98,
    })
    rid = repo.review_candidate(
        "Review conflict target is NEW.",
        confidence=0.5,
        metadata={"subject": "review conflict target",
                  "predicate": "is",
                  "value": "NEW"})

    approved = repo.approve_review(rid)

    assert approved != old["node_id"]
    assert repo.review_buffer[rid]["approved_action"] == "supersede_existing"
    assert repo.runtime.last_result.new_nodes == (approved,)
    assert repo.arena.grafts[old["node_id"]]["retired"] is True
    assert repo.arena.grafts[old["node_id"]]["metadata"]["active"] is False
    assert repo.arena.grafts[approved]["metadata"]["supersedes"] == [
        old["node_id"]]
    assert repo.arena.grafts[approved]["metadata"]["write_intent"] == (
        "user_asserted")


def test_extraction_reinforcement_maps_native_graph_refs(tmp_path):
    lib = build_native(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path / "repo"),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3, native_lib_path=lib)
    stray_native = repo.native_store.add_node("external native node", b"",
                                             ntok=1)
    assert stray_native == 0

    source0 = repo.remember("Native extraction source zero 10-0000")
    source0_native = repo.arena.grafts[source0]["native_node_id"]
    assert source0 == 0
    assert source0_native != source0

    fact = {
        "action": "write_direct",
        "candidate_type": "fact",
        "text": "Native duplicate guard is READY.",
        "subject": "Native duplicate guard",
        "predicate": "is",
        "value": "READY",
        "scope": "project",
        "durability": "project",
        "mutability": "stable",
        "write_intent": "observed",
        "confidence": 0.98,
        "source_grafts": [source0],
    }
    out = repo.apply_extraction_candidate(fact)
    fact_native = repo.arena.grafts[out["node_id"]]["native_node_id"]
    assert repo.native_store.graph_edges(fact_native).source_grafts == (
        source0_native,)

    source1 = repo.remember("Native extraction source one 11-1111")
    source1_native = repo.arena.grafts[source1]["native_node_id"]
    duplicate = dict(fact, text="Native duplicate guard remains READY.",
                     source_grafts=[source1], confidence=0.99)
    reinforced = repo.apply_extraction_candidate(duplicate)

    assert reinforced["action"] == "reinforce_existing"
    assert reinforced["node_id"] == out["node_id"]
    assert repo.arena.grafts[out["node_id"]]["metadata"]["source_grafts"] == [
        source0, source1]
    assert repo.native_store.graph_edges(fact_native).source_grafts == (
        source0_native, source1_native)
    repo.close()


def test_extraction_explicit_supersede_requires_authority(tmp_path):
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3)

    old = repo.apply_extraction_candidate({
        "action": "write_direct",
        "candidate_type": "fact",
        "text": "GRM manual target is OLD.",
        "subject": "GRM manual target",
        "predicate": "is",
        "value": "OLD",
        "scope": "project",
        "durability": "project",
        "mutability": "stable",
        "write_intent": "observed",
        "confidence": 0.99,
    })
    assert old["action"] == "write_direct"

    inferred = {
        "action": "write_direct",
        "candidate_type": "fact",
        "text": "Unrelated automatic claim wants to retire OLD.",
        "subject": "unrelated automatic claim",
        "predicate": "is",
        "value": "NEW",
        "scope": "project",
        "durability": "project",
        "mutability": "stable",
        "write_intent": "observed",
        "confidence": 0.99,
        "supersedes": [old["node_id"]],
    }
    review = repo.apply_extraction_candidate(inferred)
    assert review["action"] == "review_candidate"
    assert repo.review_buffer[-1]["reason"] == (
        "supersede action requires authoritative intent")
    assert len(repo.arena.grafts) == 1
    assert repo.arena.grafts[old["node_id"]]["metadata"]["active"] is True

    authoritative = dict(inferred, text="User confirms replacement.",
                         write_intent="user_asserted", confidence=1.0)
    replaced = repo.apply_extraction_candidate(authoritative)

    assert replaced["action"] == "supersede_existing"
    assert replaced["supersedes"] == [old["node_id"]]
    assert repo.arena.grafts[old["node_id"]]["metadata"]["active"] is False
    assert repo.arena.grafts[old["node_id"]]["retired"] is True


def test_extraction_conflicts_respect_temporal_validity(tmp_path):
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3)

    def candidate(**updates):
        base = {
            "action": "write_direct",
            "candidate_type": "fact",
            "text": "GRM route target is RAM.",
            "subject": "GRM route target",
            "predicate": "is",
            "value": "RAM",
            "scope": "project",
            "durability": "project",
            "mutability": "mutable",
            "write_intent": "observed",
            "confidence": 0.99,
        }
        base.update(updates)
        return base

    expired = repo.apply_extraction_candidate(candidate(
        text="GRM route target used to be disk.",
        value="disk",
        expires_at="2000-01-01T00:00:00+00:00"))
    assert expired["action"] == "write_direct"

    current = repo.apply_extraction_candidate(candidate(
        text="GRM route target is RAM.",
        value="RAM"))
    assert current["action"] == "write_direct"
    assert repo.arena.grafts[expired["node_id"]]["metadata"]["active"] is True

    future = repo.apply_extraction_candidate(candidate(
        text="GRM route target will be GPU.",
        value="GPU",
        valid_from="2999-01-01T00:00:00+00:00"))
    assert future["action"] == "write_direct"

    conflict = repo.apply_extraction_candidate(candidate(
        text="GRM route target is NVMe.",
        value="NVMe"))
    assert conflict["action"] == "review_candidate"
    assert repo.review_buffer[-1]["reason"] == "conflicts with active memory"

    malformed = repo.apply_extraction_candidate(candidate(
        text="GRM temporal guard is strict.",
        subject="GRM temporal guard",
        value="strict",
        valid_from="not-a-date"))
    assert malformed["action"] == "write_direct"

    guarded = repo.apply_extraction_candidate(candidate(
        text="GRM temporal guard is lax.",
        subject="GRM temporal guard",
        value="lax"))
    assert guarded["action"] == "review_candidate"


def test_extraction_expire_action_retires_and_recovers_from_wal(tmp_path):
    path = str(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                           arena_cls=FakeArena, route_layer=3)

    fact = {
        "action": "write_direct",
        "candidate_type": "fact",
        "text": "GRM stale target is ACTIVE-1.",
        "subject": "GRM stale target",
        "predicate": "is",
        "value": "ACTIVE-1",
        "scope": "project",
        "durability": "project",
        "mutability": "mutable",
        "write_intent": "observed",
        "confidence": 0.99,
    }
    out = repo.apply_extraction_candidate(fact)
    assert out["action"] == "write_direct"

    inferred = dict(fact, action="expire",
                    text="Expire GRM stale target ACTIVE-1.",
                    write_intent="inferred")
    review = repo.apply_extraction_candidate(inferred)
    assert review["action"] == "review_candidate"
    assert repo.review_buffer[-1]["reason"] == (
        "expire action requires authoritative intent")
    assert repo.arena.grafts[out["node_id"]]["metadata"]["active"] is True

    expire = dict(fact, action="expire",
                  text="Expire GRM stale target ACTIVE-1.",
                  write_intent="user_asserted",
                  confidence=1.0)
    expired = repo.apply_extraction_candidate(expire)
    assert expired["action"] == "expire"
    assert expired["expired"] == [out["node_id"]]
    retired = repo.arena.grafts[out["node_id"]]
    assert retired["retired"] is True
    assert retired["metadata"]["active"] is False
    assert retired["metadata"]["expired_by"] == expire["text"]
    assert "expired_at" in retired["metadata"]
    assert repo.show_memory_about("ACTIVE-1") == []
    assert repo.runtime.last_result.action == "expire"
    assert repo.runtime.last_result.new_nodes == ()
    assert not os.path.exists(os.path.join(path, "manifest.json"))

    reopened = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                               arena_cls=FakeArena, route_layer=3)
    recovered = reopened.arena.grafts[out["node_id"]]
    assert recovered["retired"] is True
    assert recovered["metadata"]["active"] is False
    assert recovered["metadata"]["expired_by"] == expire["text"]
    assert reopened.show_memory_about("ACTIVE-1") == []


def test_runtime_extraction_candidate_autosaves_and_reports(tmp_path):
    path = str(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, path, autosave=True,
                           arena_cls=FakeArena, route_layer=3)

    direct = {
        "action": "write_direct",
        "candidate_type": "fact",
        "text": "Runtime extraction target is ALPHA-1.",
        "subject": "runtime extraction target",
        "predicate": "is",
        "value": "ALPHA-1",
        "scope": "project",
        "durability": "project",
        "mutability": "stable",
        "write_intent": "observed",
        "confidence": 0.99,
    }
    out = repo.apply_extraction_candidate(direct)
    assert out["action"] == "write_direct"
    assert repo.runtime.last_result.event == "extraction"
    assert repo.runtime.last_result.action == "write_direct"
    assert repo.runtime.last_result.autosaved is True
    assert repo.runtime.last_result.new_nodes == (out["node_id"],)
    assert repo.stats()["dirty_nodes"] == 0

    low = dict(direct, text="Runtime extraction uncertain.",
               value="uncertain", confidence=0.2)
    review = repo.apply_extraction_candidate(low)
    assert review["action"] == "review_candidate"
    assert repo.runtime.last_result.event == "extraction"
    assert repo.runtime.last_result.action == "review_candidate"
    with open(os.path.join(path, "manifest.json")) as fh:
        man = json.load(fh)
    assert man["review_buffer"][-1]["reason"] == (
        "conflicts with active memory")

    correction = dict(direct, text="Runtime extraction target is BETA-2.",
                      value="BETA-2", write_intent="user_asserted",
                      confidence=1.0)
    corr = repo.apply_extraction_candidates([correction])[0]
    assert corr["action"] == "supersede_existing"
    assert repo.runtime.last_result.event == "extraction"
    assert repo.runtime.last_result.action == "supersede_existing"
    assert repo.runtime.last_result.new_nodes == (corr["node_id"],)
    assert repo.stats()["dirty_nodes"] == 0

    reopened = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                               arena_cls=FakeArena, route_layer=3)
    assert reopened.show_memory_about("ALPHA-1") == []
    rows = reopened.show_memory_about("BETA-2")
    assert len(rows) == 1
    assert rows[0]["metadata"]["supersedes"] == [out["node_id"]]
    assert reopened.review_buffer[-1]["status"] == "pending"


def test_runtime_extractor_runs_on_completed_turns(tmp_path):
    class Extractor:
        def __init__(self):
            self.calls = []

        def extract(self, text, **ctx):
            self.calls.append((text, ctx))
            return [{
                "action": "write_direct",
                "candidate_type": "fact",
                "text": "Runtime extractor captured code EAST-77.",
                "subject": "runtime extractor",
                "predicate": "captured",
                "value": "EAST-77",
                "scope": "project",
                "durability": "project",
                "mutability": "stable",
                "write_intent": "observed",
                "confidence": 0.99,
            }]

    extractor = Extractor()
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3, extractor=extractor)

    repo.add_turn("Please remember code EAST-77", "Recorded EAST-77.")

    assert len(extractor.calls) == 1
    assert extractor.calls[0][1]["source_grafts"] == [0]
    assert extractor.calls[0][1]["context"]["event"] == "add_turn"
    assert repo.last_extraction_results[0]["action"] == "write_direct"
    assert repo.runtime.last_result.event == "add_turn"
    assert repo.runtime.last_result.extraction
    rows = repo.show_memory_about("Runtime extractor captured")
    assert len(rows) == 1
    meta = rows[0]["metadata"]
    assert meta["source_grafts"] == [0]
    assert meta["source_turns"] == [0]
    assert meta["value"] == "EAST-77"


def test_runtime_extractor_review_and_error_policy(tmp_path):
    class LowConfidenceExtractor:
        def extract(self, text, **_ctx):
            return {
                "action": "write_direct",
                "candidate_type": "fact",
                "text": "Maybe extracted weak fact.",
                "subject": "weak",
                "predicate": "is",
                "value": "maybe",
                "confidence": 0.3,
            }

    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path / "review"),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3, extractor=LowConfidenceExtractor())
    repo.add_turn("weak fact", "maybe")
    assert repo.last_extraction_results[0]["action"] == "review_candidate"
    assert repo.review_buffer[-1]["metadata"]["source_grafts"] == [0]

    class FailingExtractor:
        def extract(self, text, **_ctx):
            raise RuntimeError("extractor offline")

    err_repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path / "error"),
                               autosave=False, arena_cls=FakeArena,
                               route_layer=3, extractor=FailingExtractor())
    err_repo.add_turn("trigger extractor", "recorded")
    assert err_repo.last_extraction_error == "extractor offline"
    assert err_repo.last_extraction_results == [{
        "action": "extract_error",
        "error": "extractor offline",
    }]


def test_runtime_extractor_malformed_candidates_are_isolated(tmp_path):
    class MixedExtractor:
        def extract(self, text, **_ctx):
            return [
                {
                    "action": "write_direct",
                    "candidate_type": "fact",
                    "text": "Extractor batch kept GOOD-42.",
                    "subject": "extractor batch",
                    "predicate": "kept",
                    "value": "GOOD-42",
                    "confidence": 0.99,
                },
                "not a candidate dictionary",
                {"action": "write_direct", "confidence": "not-a-number"},
            ]

    path = str(tmp_path / "mixed")
    repo = GraftRepository(FakeModel(), enc, dec, path,
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3, extractor=MixedExtractor())

    repo.add_turn("run mixed extractor", "recorded GOOD-42")

    actions = [r["action"] for r in repo.last_extraction_results]
    assert actions == ["write_direct", "extract_error", "extract_error"]
    assert repo.last_extraction_error == (
        "memory candidate has no text or semantic fields")
    assert repo.runtime.last_result.event == "add_turn"
    assert len(repo.runtime.last_result.extraction) == 3
    facts = [r for r in repo.show_memory_about("GOOD-42")
             if r["metadata"]["kind"] == "fact"]
    assert len(facts) == 1
    with open(os.path.join(path, "wal", "000001.wal")) as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    errors = [r for r in records if r["type"] == "EXTRACTION_ERROR"]
    assert len(errors) == 2
    assert errors[0]["source_grafts"] == [0]
    assert "not a candidate dictionary" in errors[0]["candidate"]


def test_runtime_extractor_malformed_candidate_raise_policy(tmp_path):
    class BadCandidateExtractor:
        def extract(self, text, **_ctx):
            return "not a candidate dictionary"

    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3, extractor=BadCandidateExtractor(),
                           extraction_error_policy="raise")

    with pytest.raises(TypeError, match="candidate must be a dictionary"):
        repo.add_turn("run strict extractor", "recorded")


def test_dirty_node_can_leave_vram_before_disk_flush(tmp_path):
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3, vram_budget_mb=0.000001)

    idx = repo.add_document("DOC code 99-1234")
    g = repo.arena.grafts[idx]

    assert g["host_present"] is True
    assert g["host_payload"]["payload_id"].tolist() == [0]
    assert g["device_present"] is False
    assert g["h"] is None
    assert g["dirty"] is True
    assert g["durable"] is False

    repo.flush_now()
    assert g["durable"] is True
    assert os.path.exists(os.path.join(str(tmp_path), "nodes", "0000.npz"))

    restored = repo._load_node(idx)
    assert restored == {"id": 0}
    assert g["device_present"] is True


def test_load_enforces_vram_budget(tmp_path):
    path = str(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                           arena_cls=FakeArena, route_layer=3)

    for i in range(4):
        repo.add_document(f"DOC reload budget code {i}-1234")
    repo.flush_now()
    assert repo.stats()["active_device"] == 4

    reloaded = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                               arena_cls=FakeArena, route_layer=3,
                               vram_budget_mb=0.000001)
    st = reloaded.stats()

    assert st["nodes"] == 4
    assert st["durable_nodes"] == 4
    assert st["active_device"] == 0
    for g in reloaded.arena.grafts:
        assert g["host_present"] is True
        assert g["host_payload"] is not None
        assert g["device_present"] is False
        assert g["h"] is None


def test_native_load_budget_eviction_keeps_durable_payload_clean(tmp_path):
    lib = build_native(tmp_path)
    repo_path = tmp_path / "repo"
    repo = GraftRepository(FakeModel(), enc, dec, str(repo_path),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3, native_lib_path=lib)

    for i in range(3):
        repo.add_document(f"DOC native reload budget code {i}-5678")
    repo.flush_now()
    repo.close()

    reloaded = GraftRepository(FakeModel(), enc, dec, str(repo_path),
                               autosave=False, arena_cls=FakeArena,
                               route_layer=3, native_lib_path=lib,
                               vram_budget_mb=0.000001)
    st = reloaded.stats()

    assert st["active_device"] == 0
    assert st["dirty_nodes"] == 0
    assert st["durable_nodes"] == 3
    assert st["native"]["dirty_nodes"] == 0
    assert st["native"]["durable_nodes"] == 3
    assert st["native"]["host_payload_tensors"] == 3
    reloaded.close()


def test_native_eviction_refreshes_stale_metadata_only_node(tmp_path):
    lib = build_native(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path / "repo"),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3, native_lib_path=lib,
                           vram_budget_mb=0.000001)

    idx = repo.arena.deposit("DOC stale native payload code 88-4400")
    g = repo.arena.grafts[idx]
    g["kind"] = "doc"
    g["tags"] = []
    g["sources"] = []
    g["retired"] = False
    repo._ensure_lifecycle(idx, g)

    node_id = repo.native_store.add_node(g["text"], b"", ntok=g["ntok"])
    repo._native_node_ids[idx] = node_id
    g["native_node_id"] = node_id
    repo._native_set_route(idx)
    repo._native_set_metadata(idx)
    assert repo.native_store.payload_stats(node_id).tensor_count == 0

    repo._page()

    assert g["h"] is None
    assert g["host_payload"]["payload_id"].tolist() == [idx]
    assert repo.native_store.payload_stats(node_id).tensor_count == 1
    repo.close()


def test_native_reload_mirrors_retired_cold_nodes_as_metadata_only(tmp_path):
    lib = build_native(tmp_path)
    repo_path = tmp_path / "repo"
    repo = GraftRepository(FakeModel(), enc, dec, str(repo_path),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3, native_lib_path=lib)

    retired = repo.add_document("DOC retired native cold code 12-1200")
    live = repo.add_document("DOC live native hot code 34-3400")
    before = repo._snapshot_state()
    repo.arena.grafts[retired]["retired"] = True
    repo._mark_mutations(before)
    repo.flush_now()
    repo.close()

    reloaded = GraftRepository(FakeModel(), enc, dec, str(repo_path),
                               autosave=False, arena_cls=FakeArena,
                               route_layer=3, native_lib_path=lib)
    st = reloaded.stats()

    assert st["nodes"] == 2
    assert st["native"]["nodes"] == 2
    assert st["native"]["durable_nodes"] == 2
    assert st["native"]["host_payload_tensors"] == 1
    assert reloaded.arena.grafts[retired]["cold_only"] is True
    assert reloaded.arena.grafts[retired]["host_payload"] is None
    assert retired not in reloaded.native_route(
        [1.0] * 512, lexical_keys=["12-1200"], topk=4)
    assert live in reloaded.native_route(
        [1.0] * 512, lexical_keys=["34-3400"], topk=4)
    reloaded.close()


def test_wal_recovers_text_metadata_without_manifest(tmp_path):
    path = str(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                           arena_cls=FakeArena, route_layer=3)

    repo.add_document("DOC crash-only code 44-9000", tags=("crash",))
    assert os.path.exists(os.path.join(path, "wal", "000001.wal"))
    assert not os.path.exists(os.path.join(path, "manifest.json"))

    reopened = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                               arena_cls=FakeArena, route_layer=3)

    # Reporting surface still populated.
    assert reopened.stats()["recovered_wal_records"] > 0
    assert reopened.recovered_nodes[0]["text"] == "DOC crash-only code 44-9000"
    assert reopened.recovered_nodes[0]["kind"] == "doc"
    assert reopened.recovered_nodes[0]["payload_pending"] is True

    # WAL recovery now REHYDRATES the live repository, not just a report.
    assert len(reopened.arena.grafts) == 1
    g = reopened.arena.grafts[0]
    assert g["text"] == "DOC crash-only code 44-9000"
    assert g["kind"] == "doc"
    assert g["recovered"] is True
    assert g["payload_pending"] is True
    # text-authoritative but not yet routable / durable: no payload, no device
    # tensor, a zero placeholder centroid, and dialect-correct centroid width.
    assert g["host_payload"] is None
    assert g["h"] is None
    assert g["durable"] is False
    assert g["host_present"] is False
    assert g["cent"].shape == (512,)
    assert not g["cent"].any()
    # the recovered node is inspectable through the memory-query surface.
    assert reopened.show_memory_about("44-9000")
    assert g["metadata"]["active"] is True  # active (not a forgotten node)


def test_wal_recovery_keeps_forgotten_nodes_inert(tmp_path):
    path = str(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                           arena_cls=FakeArena, route_layer=3)
    repo.add_document("DOC keep-me code 11-1111", tags=("keep",))
    repo.add_document("DOC drop-me code 22-2222", tags=("drop",))
    repo.forget("drop-me")
    assert not os.path.exists(os.path.join(path, "manifest.json"))

    reopened = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                               arena_cls=FakeArena, route_layer=3)

    # both rehydrated, but the forgotten one is inert (retired / inactive)
    assert len(reopened.arena.grafts) == 2
    by_text = {g["text"]: g for g in reopened.arena.grafts}
    kept = by_text["DOC keep-me code 11-1111"]
    dropped = by_text["DOC drop-me code 22-2222"]
    assert kept["metadata"]["active"] is True
    assert dropped["retired"] is True
    assert dropped["metadata"]["active"] is False
    # forgotten memory does not surface as active answer authority
    assert reopened.show_memory_about("drop-me") == []
    assert reopened.show_memory_about("keep-me")


def test_wal_recovery_replays_extraction_supersession(tmp_path):
    path = str(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                           arena_cls=FakeArena, route_layer=3)

    old = repo.apply_extraction_candidate({
        "action": "write_direct",
        "candidate_type": "fact",
        "text": "GRM hot tier is RAM-first.",
        "subject": "GRM hot tier",
        "predicate": "is",
        "value": "RAM-first",
        "scope": "project",
        "durability": "project",
        "mutability": "stable",
        "write_intent": "observed",
        "confidence": 0.97,
    })
    assert old["action"] == "write_direct"

    new = repo.apply_extraction_candidate({
        "action": "write_direct",
        "candidate_type": "fact",
        "text": "GRM hot tier is RAM-authoritative.",
        "subject": "GRM hot tier",
        "predicate": "is",
        "value": "RAM-authoritative",
        "scope": "project",
        "durability": "project",
        "mutability": "stable",
        "write_intent": "user_asserted",
        "confidence": 1.0,
    })
    assert new["action"] == "supersede_existing"
    assert not os.path.exists(os.path.join(path, "manifest.json"))

    reopened = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                               arena_cls=FakeArena, route_layer=3)

    old_g = reopened.arena.grafts[old["node_id"]]
    new_g = reopened.arena.grafts[new["node_id"]]
    assert old_g["retired"] is True
    assert old_g["metadata"]["active"] is False
    assert old_g["metadata"]["superseded_by"] == [new["node_id"]]
    assert new_g["retired"] is False
    assert new_g["metadata"]["active"] is True
    assert new_g["metadata"]["supersedes"] == [old["node_id"]]
    assert reopened.show_memory_about("RAM-first") == []
    assert len(reopened.show_memory_about("RAM-authoritative")) == 1


def test_wal_recovery_preserves_fold_abort_exemption(tmp_path):
    path = str(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                           arena_cls=FakeArena, route_layer=3)

    idx = repo.add_document("DOC fold-abort code 77-7717")
    before = repo._snapshot_state()
    repo.arena.grafts[idx]["no_fold"] = True
    repo._mark_mutations(before)
    assert not os.path.exists(os.path.join(path, "manifest.json"))

    reopened = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                               arena_cls=FakeArena, route_layer=3)

    assert reopened.arena.grafts[idx]["no_fold"] is True
    assert reopened.arena.grafts[idx]["metadata"]["no_fold"] is True


def test_manifest_load_replays_post_checkpoint_forget(tmp_path):
    path = str(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                           arena_cls=FakeArena, route_layer=3)
    keep = repo.add_document("DOC keep checkpoint code 41-4100")
    drop = repo.add_document("DOC drop checkpoint code 42-4200")
    repo.flush_now()
    assert repo.forget("drop checkpoint") == 1

    with open(os.path.join(path, "manifest.json")) as fh:
        man = json.load(fh)
    assert man["nodes"][drop]["metadata"]["active"] is True

    reopened = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                               arena_cls=FakeArena, route_layer=3)

    assert reopened.replayed_wal_nodes == (drop,)
    assert reopened.stats()["replayed_wal_nodes"] == 1
    assert reopened.arena.grafts[keep]["metadata"]["active"] is True
    assert reopened.arena.grafts[drop]["retired"] is True
    assert reopened.arena.grafts[drop]["metadata"]["active"] is False
    assert reopened.show_memory_about("42-4200") == []
    assert len(reopened.show_memory_about("41-4100")) == 1


def test_manifest_load_replays_post_checkpoint_correction(tmp_path):
    path = str(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                           arena_cls=FakeArena, route_layer=3)
    old = repo.remember("GRM recovery target is OLD-100.",
                        kind="fact", scope="project")
    repo.flush_now()
    new = repo.correct_memory(
        "OLD-100", "GRM recovery target is NEW-200.")

    reopened = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                               arena_cls=FakeArena, route_layer=3)

    assert reopened.replayed_wal_nodes == (old, new)
    old_g = reopened.arena.grafts[old]
    new_g = reopened.arena.grafts[new]
    assert old_g["retired"] is True
    assert old_g["metadata"]["active"] is False
    assert old_g["metadata"]["superseded_by"] == [new]
    assert new_g["metadata"]["active"] is True
    assert new_g["metadata"]["supersedes"] == [old]
    assert new_g["payload_pending"] is True
    assert new_g["host_payload"] is None
    assert new_g["durable"] is False
    assert reopened.show_memory_about("OLD-100") == []
    assert len(reopened.show_memory_about("NEW-200")) == 1


def test_native_manifest_load_replays_wal_into_route_state(tmp_path):
    lib = build_native(tmp_path)
    path = str(tmp_path / "repo")
    repo = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                           arena_cls=FakeArena, route_layer=3,
                           native_lib_path=lib)
    idx = repo.add_document("DOC native post-checkpoint drop 66-6601")
    repo.flush_now()
    assert idx in repo.native_route([1.0] * 512,
                                    lexical_keys=["66-6601"], topk=2)
    assert repo.forget("66-6601") == 1
    repo.close()

    reopened = GraftRepository(FakeModel(), enc, dec, path, autosave=False,
                               arena_cls=FakeArena, route_layer=3,
                               native_lib_path=lib)

    assert reopened.replayed_wal_nodes == (idx,)
    assert reopened.arena.grafts[idx]["metadata"]["active"] is False
    assert idx not in reopened.native_route([1.0] * 512,
                                            lexical_keys=["66-6601"], topk=2)
    assert reopened.native_store.dirty_node_ids() == (
        reopened.arena.grafts[idx]["native_node_id"],)
    reopened.close()


def test_native_store_mirrors_repository_lifecycle(tmp_path):
    lib = build_native(tmp_path)
    repo_path = tmp_path / "repo"
    repo = GraftRepository(FakeModel(), enc, dec, str(repo_path),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3, native_lib_path=lib)

    idx0 = repo.add_document("DOC native mirror 22-1450")
    idx1 = repo.add_document("DOC native mirror 77-9999")
    assert repo.arena.grafts[idx0]["native_node_id"] == 0
    assert repo.arena.grafts[idx1]["native_node_id"] == 1

    st = repo.stats()
    assert st["native"]["nodes"] == 2
    assert st["native"]["dirty_nodes"] == 2
    assert st["native"]["durable_nodes"] == 0
    assert st["native"]["host_payload_bytes"] == 16
    assert st["native"]["host_payload_tensors"] == 2
    assert st["native"]["route_entries"] == 2
    assert repo.native_store.payload_stats(
        repo.arena.grafts[idx0]["native_node_id"]).tensor_count == 1
    assert repo.native_store.tensor_info(
        repo.arena.grafts[idx0]["native_node_id"], "payload_id").shape == (1,)
    assert repo.native_store.get_tensor(
        repo.arena.grafts[idx0]["native_node_id"],
        "payload_id").tolist() == [0]
    assert repo.native_route([1.0] * 512, lexical_keys=["77-9999"],
                             topk=1) == [idx1]

    repo.flush_now()
    st = repo.stats()
    assert st["native"]["dirty_nodes"] == 0
    assert st["native"]["durable_nodes"] == 2
    assert os.path.exists(repo_path / "native" / "grm_store.bin")
    with open(repo_path / "manifest.json") as fh:
        man = json.load(fh)
    assert man["native_checkpoint"] == "native/grm_store.bin"
    assert man["nodes"][idx0]["native_node_id"] == 0
    assert man["nodes"][idx1]["native_node_id"] == 1

    assert repo.forget("22-1450") == 1
    assert repo.stats()["native"]["dirty_nodes"] == 1
    repo.flush_now()
    st = repo.stats()
    assert st["native"]["dirty_nodes"] == 0
    assert st["native"]["durable_nodes"] == 2
    assert idx0 not in repo.native_route([1.0] * 512,
                                         lexical_keys=["22-1450"], topk=2)

    repo.close()
    reloaded = GraftRepository(FakeModel(), enc, dec, str(repo_path),
                               autosave=False, arena_cls=FakeArena,
                               route_layer=3, native_lib_path=lib)
    st = reloaded.stats()
    assert st["native"]["checkpoint_loaded"] is True
    assert st["native"]["nodes"] == 2
    assert st["native"]["dirty_nodes"] == 0
    assert st["native"]["durable_nodes"] == 2
    assert st["native"]["route_entries"] == 2
    assert st["native"]["host_payload_tensors"] == 1
    assert reloaded.arena.grafts[idx0]["native_node_id"] == 0
    assert reloaded.arena.grafts[idx1]["native_node_id"] == 1
    assert idx0 not in reloaded.native_route([1.0] * 512,
                                             lexical_keys=["22-1450"],
                                             topk=2)
    assert reloaded.native_route([1.0] * 512, lexical_keys=["77-9999"],
                                 topk=1) == [idx1]
    reloaded.close()


def test_repository_native_store_supports_gqa_dialect(tmp_path):
    lib = build_native(tmp_path)
    repo_path = tmp_path / "gqa_repo"
    repo = GraftRepository(FakeGQAModel(), enc, dec, str(repo_path),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=0, native_lib_path=lib)

    assert repo.native_store.dialect_id() == "FakeGQAModel:12x1536:g2x128"
    assert repo.native_store.dialect_profile() == (
        "rope_full_kv|kv|seat_remountable|1|multi_mount")
    idx = repo.add_document("DOC native GQA mirror 66-6600")
    native_id = repo.arena.grafts[idx]["native_node_id"]
    st = repo.stats()

    assert st["native"]["nodes"] == 1
    assert st["native"]["route_entries"] == 1
    assert repo.native_store.payload_stats(native_id).tensor_count == 1
    assert repo.native_route([1.0] * 512, lexical_keys=["66-6600"],
                             topk=1) == [idx]

    repo.flush_now()
    repo.close()
    reloaded = GraftRepository(FakeGQAModel(), enc, dec, str(repo_path),
                               autosave=False, arena_cls=FakeArena,
                               route_layer=0, native_lib_path=lib)
    assert reloaded.native_store.dialect_id() == "FakeGQAModel:12x1536:g2x128"
    assert reloaded.stats()["native"]["durable_nodes"] == 1
    assert reloaded.native_route([1.0] * 512, lexical_keys=["66-6600"],
                                 topk=1) == [idx]
    reloaded.close()


def test_repository_native_route_respects_memory_lifecycle(tmp_path):
    lib = build_native(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path / "repo"),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3, native_lib_path=lib)

    stale = repo.remember("Native lifecycle stale 11-1111")
    live = repo.remember("Native lifecycle live 22-2222")
    assert stale in repo.native_route([1.0] * 512,
                                      lexical_keys=["11-1111"], topk=4)

    assert repo.forget("stale 11-1111") == 1
    routed = repo.native_route([1.0] * 512, lexical_keys=["11-1111"], topk=4)
    assert stale not in routed
    assert live in repo.native_route([1.0] * 512,
                                     lexical_keys=["22-2222"], topk=4)

    old = repo.remember("Native correction target 33-3333")
    new = repo.correct_memory("correction target 33-3333",
                              "Native correction replacement 44-4444")
    assert repo.arena.grafts[old]["metadata"]["active"] is False
    assert repo.arena.grafts[old]["retired"] is True
    old_native = repo.arena.grafts[old]["native_node_id"]
    new_native = repo.arena.grafts[new]["native_node_id"]
    assert repo.native_store.graph_edges(new_native).supersedes == (old_native,)
    assert repo.native_store.graph_edges(old_native).superseded_by == (new_native,)
    routed_old = repo.native_route([1.0] * 512,
                                   lexical_keys=["33-3333"], topk=8)
    assert old not in routed_old
    assert new in repo.native_route([1.0] * 512,
                                    lexical_keys=["44-4444"], topk=8)
    repo.close()


def test_repository_native_route_filters_metadata_policy(tmp_path):
    lib = build_native(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path / "repo"),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3, native_lib_path=lib)

    fact = repo.remember("Filtered native fact 55-5555",
                         kind="fact", scope="project",
                         durability="project", mutability="stable")
    task = repo.remember("Filtered native task 55-5555",
                         kind="task_state", scope="session",
                         durability="session", mutability="ephemeral")
    doc = repo.add_document("DOC filtered native 55-5555")

    assert repo.native_route([1.0] * 512, lexical_keys=["55-5555"],
                             topk=8, kinds="fact") == [fact]
    assert repo.native_route([1.0] * 512, lexical_keys=["55-5555"],
                             topk=8, kinds=("task_state",),
                             scopes=("session",)) == [task]
    assert repo.native_route([1.0] * 512, lexical_keys=["55-5555"],
                             topk=8, kinds=("doc",),
                             scopes=("project",)) == [doc]
    filtered = repo.native_route([1.0] * 512, lexical_keys=["55-5555"],
                                 topk=8, durabilities=("project",),
                                 mutabilities=("stable",))
    assert set(filtered) == {fact, doc}
    repo.close()


def test_repository_native_mirror_can_checkpoint_payloads(tmp_path):
    lib = build_native(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path / "repo"),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3, native_lib_path=lib)
    idx = repo.add_document("DOC native checkpoint 18-2048")
    native_id = repo.arena.grafts[idx]["native_node_id"]
    ckpt = tmp_path / "native_ckpt"

    repo.native_store.save_checkpoint(ckpt)
    with NativeGraftStore(
            lib, model_type="FakeModel", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as restored:
        restored.load_checkpoint(ckpt)
        assert restored.stats().nodes == 1
        assert restored.tensor_info(native_id, "payload_id").shape == (1,)
        assert restored.get_tensor(native_id, "payload_id").tolist() == [0]
    repo.close()


def test_repository_skips_clean_native_checkpoint_rewrite(tmp_path, monkeypatch):
    lib = build_native(tmp_path)
    repo_path = tmp_path / "repo"
    repo = GraftRepository(FakeModel(), enc, dec, str(repo_path),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3, native_lib_path=lib)
    idx = repo.add_document("DOC native clean-checkpoint code 31-3100")
    repo.flush_now()
    assert repo.native_store.dirty_node_ids() == ()

    calls = []
    original = repo.native_store.save_checkpoint

    def traced(root):
        calls.append(root)
        return original(root)

    monkeypatch.setattr(repo.native_store, "save_checkpoint", traced)
    repo.flush_now()
    assert calls == []

    assert repo.forget("31-3100") == 1
    assert repo.native_store.dirty_node_ids() == (
        repo.arena.grafts[idx]["native_node_id"],)
    repo.flush_now()
    assert len(calls) == 1
    assert repo.native_store.dirty_node_ids() == ()
    repo.close()


def test_repository_native_mirror_persists_semantic_metadata(tmp_path):
    lib = build_native(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path / "repo"),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3, native_lib_path=lib)
    idx = repo.remember("GRM metadata lives in native RAM too.",
                        durability="project",
                        mutability="stable",
                        scope="project",
                        kind="fact",
                        metadata={"subject": "native metadata mirror"})
    native_id = repo.arena.grafts[idx]["native_node_id"]
    meta = repo.native_store.metadata(native_id)
    assert meta["kind"] == "fact"
    assert meta["durability"] == "project"
    assert meta["subject"] == "native metadata mirror"

    ckpt = tmp_path / "metadata_ckpt"
    repo.native_store.save_checkpoint(ckpt)
    with NativeGraftStore(
            lib, model_type="FakeModel", num_layers=27,
            hidden_dim=2048, vals_per_tok_layer=576, route_layer=3,
            latent_rank=512, rope_dim=64) as restored:
        restored.load_checkpoint(ckpt)
        restored_meta = restored.metadata(native_id)
        assert restored_meta["write_intent"] == "user_asserted"
        assert restored_meta["subject"] == "native metadata mirror"
    repo.close()


def test_repository_attaches_native_store_to_arena(tmp_path):
    lib = build_native(tmp_path)
    repo = GraftRepository(FakeModel(), enc, dec, str(tmp_path / "repo"),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=3, native_lib_path=lib)
    assert repo.arena.native_store is repo.native_store
    repo.close()
    assert repo.arena.native_store is None
