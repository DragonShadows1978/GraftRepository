import json
import os

import numpy as np

from core.graft_repository import GraftRepository
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


def enc(text):
    return text.split()


def dec(ids):
    return " ".join(ids)


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

    why = repo.why_remember("authoritative live store")
    assert why[0]["write_intent"] == "user_asserted"


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

    user_correction = dict(direct, text="GRM hot tier is GPU-mounted only.",
                           value="GPU-mounted only",
                           write_intent="user_asserted",
                           confidence=1.0)
    corr = repo.apply_extraction_candidate(user_correction)
    assert corr["action"] == "supersede_existing"
    assert corr["supersedes"] == [out["node_id"]]
    assert repo.arena.grafts[out["node_id"]]["metadata"]["active"] is False
    assert repo.arena.grafts[out["node_id"]]["retired"] is True
    assert repo.arena.grafts[corr["node_id"]]["metadata"]["supersedes"] == [
        out["node_id"]]


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

    repo.close()
    reloaded = GraftRepository(FakeModel(), enc, dec, str(repo_path),
                               autosave=False, arena_cls=FakeArena,
                               route_layer=3, native_lib_path=lib)
    st = reloaded.stats()
    assert st["native"]["nodes"] == 2
    assert st["native"]["dirty_nodes"] == 0
    assert st["native"]["durable_nodes"] == 2
    assert st["native"]["route_entries"] == 2
    assert st["native"]["host_payload_tensors"] == 2
    assert reloaded.native_route([1.0] * 512, lexical_keys=["77-9999"],
                                 topk=1) == [idx1]
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
