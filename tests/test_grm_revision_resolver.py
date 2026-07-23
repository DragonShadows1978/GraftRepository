import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.graft_arena import ArenaCache
from core.graft_repository import GraftRepository
from core.grm_native import NativeGraftStore


ROOT = Path(__file__).resolve().parents[1]


def _graft(text, *, kind="turn", active=True, retired=False,
           subject="", predicate="", value="", scope="project"):
    return {
        "text": text,
        "kind": kind,
        "retired": retired,
        "ntok": 4,
        "cent": np.asarray([1.0, 0.0], dtype=np.float32),
        "metadata": {
            "kind": kind,
            "active": active,
            "subject": subject,
            "predicate": predicate,
            "value": value,
            "scope": scope,
        },
    }


def _repo(grafts, native_store=None, native_ids=None):
    repo = GraftRepository.__new__(GraftRepository)
    repo.arena = SimpleNamespace(grafts=grafts)
    repo.native_store = native_store
    repo._native_node_ids = dict(native_ids or {})
    repo._last_route_resolution = None
    repo.revision_aware_resolver = True
    return repo


def test_python_resolver_pins_unique_active_fact_and_preserves_stable_order():
    grafts = [
        _graft("derived orion pin answer"),
        _graft("unrelated"),
        _graft("current", kind="fact", subject="orion pin",
               predicate="value", value="Kestrel-9"),
    ]
    repo = _repo(grafts)
    result = repo._resolve_route_family(
        bare_text="What is the current orion pin value?",
        query_keys={"orion", "pin"}, candidate_ids=[0, 1, 2],
        semantic_ranking=[0, 1, 2], limit=3)

    assert result["state"] == "exact"
    assert result["backend"] == "python"
    assert result["pinned_node_ids"] == [2]
    assert result["ranking"] == [2, 0, 1]
    assert result["matched_subject"] == "orion pin"


@pytest.mark.parametrize("grafts", [
    [
        _graft("leaf a", kind="fact", subject="orion pin",
               predicate="value", value="A"),
        _graft("leaf b", kind="fact", subject="orion pin",
               predicate="value", value="B"),
    ],
    [
        _graft("first", kind="fact", subject="orion pin",
               predicate="value", value="A"),
        _graft("second", kind="fact", subject="pin orion",
               predicate="status", value="B"),
    ],
])
def test_python_resolver_fails_closed_on_ambiguity(grafts):
    repo = _repo(grafts)
    result = repo._resolve_route_family(
        bare_text="orion pin", query_keys={"orion", "pin"},
        candidate_ids=[0, 1], semantic_ranking=[1, 0], limit=2)

    assert result["state"] == "ambiguous"
    assert result["pinned_node_ids"] == []
    assert result["ranking"] == [1, 0]


def test_python_resolver_prefers_most_specific_fully_addressed_subject():
    grafts = [
        _graft("broad", kind="fact", subject="orion",
               predicate="value", value="Broad"),
        _graft("specific", kind="fact", subject="orion pin",
               predicate="value", value="Specific"),
    ]
    repo = _repo(grafts)
    result = repo._resolve_route_family(
        bare_text="orion pin", query_keys={"orion", "pin"},
        candidate_ids=[0, 1], semantic_ranking=[0, 1], limit=2)

    assert result["state"] == "exact"
    assert result["pinned_node_ids"] == [1]
    assert result["ranking"] == [1, 0]


def test_python_resolver_never_pins_inactive_or_expired_fact():
    grafts = [
        _graft("old", kind="fact", active=False, subject="orion pin",
               predicate="value", value="Old"),
        _graft("expired", kind="fact", subject="orion pin",
               predicate="value", value="Expired"),
    ]
    grafts[1]["metadata"]["expires_at"] = "2000-01-01T00:00:00Z"
    repo = _repo(grafts)
    result = repo._resolve_route_family(
        bare_text="orion pin", query_keys={"orion", "pin"},
        candidate_ids=[0, 1], semantic_ranking=[1, 0], limit=2)

    assert result["state"] == "no_match"
    assert result["pinned_node_ids"] == []
    assert result["ranking"] == [1, 0]
    assert result["inactive_rejected"] == 2


def test_unicode_identity_uses_python_policy():
    grafts = [_graft(
        "Die aktuelle München brücke value is Blau.", kind="fact",
        subject="münchen brücke", predicate="value", value="Blau")]
    repo = _repo(grafts)
    result = repo._resolve_route_family(
        bare_text="Was ist die münchen brücke?",
        query_keys={"münchen", "brücke"}, candidate_ids=[0],
        semantic_ranking=[0], limit=1)

    assert result["state"] == "exact"
    assert result["backend"] == "python"
    assert result["pinned_node_ids"] == [0]


def test_correction_identity_derivation_is_grounded_and_fail_closed():
    inherited = _graft(
        "The current orion pin value is Old.", kind="fact",
        subject="orion pin", predicate="value", value="Old")
    repo = _repo([inherited])

    got = repo._derive_correction_fact_identity(
        "orion pin", "The current orion pin value is New.", [0], {})
    assert got["subject"] == "orion pin"
    assert got["predicate"] == "value"
    assert got["value"] == "New"
    assert got["fact_identity_source"] == "inherited_supersession"

    grounded = repo._derive_correction_fact_identity(
        "The current cypher bridge value is Old.",
        "The current cypher bridge value is New.", [], {})
    assert grounded["subject"] == "cypher bridge"
    assert grounded["predicate"] == "value"
    assert grounded["fact_identity_source"] == "grounded_correction_grammar"

    rejected = repo._derive_correction_fact_identity(
        "Please correct my note", "A different unsupported claim", [], {})
    assert "subject" not in rejected
    assert "predicate" not in rejected


def test_correct_memory_wires_grounded_identity_only_when_opted_in():
    old = _graft(
        "The current orion pin value is Old.", kind="fact", value="Old")
    repo = _repo([old])
    repo._snapshot_state = lambda: []
    repo._native_active_text_matches = lambda _query: [0]
    repo._native_memory_mutation_plan = lambda *_args, **_kwargs: None
    repo._mark_dirty = lambda *_args, **_kwargs: None
    repo._native_apply_revision = lambda *_args, **_kwargs: None
    repo._append_wal = lambda *_args, **_kwargs: None
    repo._mark_mutations = lambda *_args, **_kwargs: None
    captured = {}

    def remember(text, **kwargs):
        captured["text"] = text
        captured.update(kwargs)
        return 1

    repo.remember = remember
    replacement = "The current orion pin value is New."
    assert repo.correct_memory(old["text"], replacement) == 1
    assert captured["metadata"]["subject"] == "orion pin"
    assert captured["metadata"]["predicate"] == "value"
    assert captured["metadata"]["value"] == "New"
    assert captured["metadata"]["supersedes"] == [0]
    assert old["retired"] is True
    assert old["metadata"]["active"] is False


def test_step_uses_exact_resolution_as_first_single_mount():
    arena = ArenaCache.__new__(ArenaCache)
    arena.m = SimpleNamespace(layers=[])
    arena.stop_sequences = ()
    arena.ephemeral = False
    arena.live_segs = []
    arena.grafts = [_graft("derived"), _graft(
        "fact", kind="fact", subject="orion pin", predicate="value")]
    arena.topk = 2
    arena.width = 32
    arena.caches = None
    arena.pos = 0
    arena.cur_mounts = []
    arena.cur_mount_n = 0
    arena.last_route_resolution = {
        "state": "exact", "pinned_node_ids": [1], "backend": "python"}
    arena.route = lambda *_args, **_kwargs: [1, 0]
    arena._descent_expand = lambda picks, *_args, **_kwargs: list(picks)
    mounted = []

    def attempt(_text, picks, *_args):
        mounted.append(list(picks))
        return "grounded", {}

    arena._attempt = attempt
    arena._grounded = lambda *_args: True

    _answer, info = arena.step("ordinary lowercase query", deposit=False)
    assert mounted == [[1]]
    assert info["route_resolution"]["pinned_node_ids"] == [1]


@pytest.fixture(scope="module")
def native_lib(tmp_path_factory):
    build = tmp_path_factory.mktemp("revision_resolver_native")
    lib = build / "libgrm_runtime.so"
    subprocess.run([
        "g++", "-std=c++17", "-O2", "-shared", "-fPIC",
        "-I", str(ROOT / "cpp"), str(ROOT / "cpp" / "grm_runtime.cpp"),
        "-o", str(lib),
    ], check=True)
    return str(lib)


def _open_native(lib):
    return NativeGraftStore(
        lib, model_type="ResolverTest", num_layers=2, hidden_dim=8,
        vals_per_tok_layer=4, route_layer=0, payload_kind="mla",
        latent_rank=4, rope_dim=0)


def test_native_resolver_parity_lifecycle_and_checkpoint(native_lib, tmp_path):
    ckpt = tmp_path / "resolver_ckpt"
    with _open_native(native_lib) as store:
        old = store.add_node("old", b"", ntok=1)
        current = store.add_node("current", b"", ntok=1)
        turn = store.add_node("derived", b"", ntok=1)
        store.set_metadata(old, {
            "kind": "fact", "active": False, "subject": "orion pin",
            "predicate": "value", "value": "Old", "scope": "project"})
        store.set_metadata(current, {
            "kind": "fact", "active": True, "subject": "orion pin",
            "predicate": "value", "value": "New", "scope": "project"})
        store.set_metadata(turn, {"kind": "turn", "active": True})

        native = store.resolve_fact_query(
            ["orion", "pin"], [old, current, turn])
        assert native.state == "exact"
        assert native.candidate_node_ids == (current,)
        assert native.inactive_rejected == 1
        store.save_checkpoint(ckpt)

    with _open_native(native_lib) as restored:
        restored.load_checkpoint(ckpt)
        after = restored.resolve_fact_query(
            ["orion", "pin"], [old, current, turn])
        assert after == native

        grafts = [
            _graft("old", kind="fact", active=False, subject="orion pin",
                   predicate="value", value="Old"),
            _graft("current", kind="fact", subject="orion pin",
                   predicate="value", value="New"),
            _graft("derived"),
        ]
        native_repo = _repo(
            grafts, native_store=restored,
            native_ids={0: old, 1: current, 2: turn})
        native_result = native_repo._resolve_route_family(
            bare_text="What is the current orion pin value?",
            query_keys={"orion", "pin"}, candidate_ids=[0, 1, 2],
            semantic_ranking=[2, 1, 0], limit=3)
        python_repo = _repo(grafts)
        python_result = python_repo._resolve_route_family(
            bare_text="What is the current orion pin value?",
            query_keys={"orion", "pin"}, candidate_ids=[0, 1, 2],
            semantic_ranking=[2, 1, 0], limit=3)

        assert native_result["backend"] == "native"
        assert native_result["state"] == python_result["state"] == "exact"
        assert native_result["ranking"] == python_result["ranking"] == [1, 2, 0]
        assert native_result["pinned_node_ids"] == [1]
