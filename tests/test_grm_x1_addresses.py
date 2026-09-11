"""CPU invariant gates. Prior art: HOUSE_RULES invariant/mutation testing
(2026); author baseline only, not independent blind verification.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

if os.environ.get("GRM_X1_TEST_MODULE"):
    spec = importlib.util.spec_from_file_location("x1_mutant", os.environ["GRM_X1_TEST_MODULE"])
    x1 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = x1
    spec.loader.exec_module(x1)
else:
    from core import grm_x1_addresses as x1

DIGEST = "a" * 64


class Untouchable:
    def __getattribute__(self, name):
        raise AssertionError(f"OFF touched address dependency: {name}")


@pytest.mark.parametrize("environment", [{}, {x1.ENV_NAME: "off"}, {x1.ENV_NAME: "banana"}])
def test_grm_x1_default_off_passthrough_identity(environment):
    expected = (object(), {"unchanged": object()})
    calls = []
    def fallback(*args, **kwargs):
        calls.append((args, kwargs))
        return expected
    adapter = x1.AddressedRecall(Untouchable(), fallback, Untouchable(), Untouchable(), environ=environment)
    assert adapter.serve("original question", address=Untouchable(), page_fault=True, ngen=7) is expected
    assert calls == [(("original question",), {"ngen": 7})]
    assert adapter.record_deposit(Untouchable(), -1, [], "invalid") is None


def test_default_off_environment(monkeypatch):
    monkeypatch.delenv(x1.ENV_NAME, raising=False)
    assert x1.enabled() is False
    for token in ("1", "true", "YES", " on "):
        assert x1.enabled({x1.ENV_NAME: token}) is True


def test_current_version_not_arrival_rank_or_page_order():
    idx = x1.AddressIndex("repo-a")
    key = x1.Address("entity", "relation")
    idx.publish(key, 1, [900], DIGEST)
    idx.publish(key, 10, [3, 2], DIGEST)
    assert idx.resolve(key).page_ids == (3, 2)
    assert idx.resolve(key).version == 10
    with pytest.raises(ValueError, match="stale"):
        idx.publish(key, 2, [9999], DIGEST)
    with pytest.raises(ValueError, match="conflict"):
        idx.publish(key, 10, [4], DIGEST)
    assert idx.publish(key, 10, [3, 2], DIGEST) == idx.resolve(key)
    assert idx.resolve(x1.Address("entity", "other relation")) is None


@pytest.mark.parametrize("version,pages,digest", [(0, [1], DIGEST), (True, [1], DIGEST),
    (1, [], DIGEST), (1, [-1], DIGEST), (1, [True], DIGEST), (1, [1, 1], DIGEST),
    (1, [1.0], DIGEST), (1, [1], "not a hash")])
def test_invalid_publication_is_atomic(version, pages, digest):
    idx = x1.AddressIndex("repo")
    with pytest.raises(ValueError):
        idx.publish(x1.Address("e", "r"), version, pages, digest)
    assert idx.dump()["versions"] == []


def test_key_normalization_and_relation_isolation():
    idx = x1.AddressIndex("repo")
    first, second = x1.Address("ＥＮＴＩＴＹ", " key "), x1.Address("entity", "seal")
    idx.publish(first, 1, [2], DIGEST)
    idx.publish(second, 1, [3], DIGEST)
    assert idx.resolve(x1.Address("entity", "KEY")).page_ids == (2,)
    assert idx.resolve(second).page_ids == (3,)
    with pytest.raises(ValueError):
        x1.Address("", "key")


def test_repository_identity_roundtrip_and_create_only(tmp_path):
    idx = x1.AddressIndex("repo-a")
    idx.publish(x1.Address("e", "r"), 1, [4, 5], DIGEST)
    path = tmp_path / "index.json"
    idx.save(path)
    assert x1.AddressIndex.restore(json.loads(path.read_text()), repository_id="repo-a").dump() == idx.dump()
    with pytest.raises(ValueError, match="different repository"):
        x1.AddressIndex.restore(idx.dump(), repository_id="repo-b")
    with pytest.raises(FileExistsError):
        idx.save(path)


def make_adapter(available, mounts=None):
    idx = x1.AddressIndex("repo")
    idx.publish(x1.Address("e", "r"), 1, [2, 3], DIGEST)
    calls = []
    def fallback(question, **kw):
        calls.append("router")
        return "router answer", {}
    def mount_read(question, pages, guard, **kw):
        calls.append("mount")
        guard(pages if mounts is None else mounts)
        calls.append("generate")
        return "read answer", {}
    adapter = x1.AddressedRecall(idx, fallback, mount_read, lambda: available, environ={x1.ENV_NAME: "1"})
    return adapter, calls


@pytest.mark.parametrize("available,query_address,reason", [([], x1.Address("e", "r"), "pages_absent"),
    ([2], x1.Address("e", "r"), "pages_absent"), ([2, 3], x1.Address("e", "other"), "unresolved_address"),
    ([2, 3], None, "unresolved_address")])
def test_c_missing_partial_unknown_never_generates(available, query_address, reason):
    adapter, calls = make_adapter(available)
    answer, info = adapter.serve("question", address=query_address, point_lookup=True, page_fault=True)
    assert calls == []
    assert answer == x1.ABSTENTION
    assert info["x1_fault_reason"] == reason
    assert info["x1_generation_calls"] == 0


def test_b_absence_falls_back_and_c_faults():
    adapter, calls = make_adapter([])
    answer, info = adapter.serve("q", address=x1.Address("e", "r"))
    assert answer == "router answer" and calls == ["router"]
    assert info["x1_structural_demand_fired"] is True
    assert info["x1_page_fault"] is False


@pytest.mark.parametrize("multiplicity", [1, 10, 100])
def test_point_lookup_never_consults_duplicate_ranking(multiplicity):
    adapter, calls = make_adapter([2, 3, *range(10, 10 + multiplicity)])
    answer, info = adapter.serve("q", address=x1.Address("e", "r"), page_fault=True)
    assert answer == "read answer"
    assert calls == ["mount", "generate"]
    assert info["x1_address_coverage"] is True


@pytest.mark.parametrize("actual", [(2,), (2, 3, 900), (3, 2)])
def test_c_checks_actual_mounts_before_forward(actual):
    adapter, calls = make_adapter([2, 3], actual)
    answer, info = adapter.serve("q", address=x1.Address("e", "r"), page_fault=True)
    assert calls == ["mount"]
    assert answer == x1.ABSTENTION and info["x1_generation_calls"] == 0


def test_topical_discovery_is_original_router_even_when_on():
    adapter, calls = make_adapter([])
    assert adapter.serve("Discuss the history of shipping")[0] == "router answer"
    assert calls == ["router"]


def test_alias_resolver_ambiguity_boundaries_and_no_gold_lookup():
    aliases = {"entities": {"rhea": ["rhea"], "io": ["io"]}, "relations": {"key": ["key", "access code"]}}
    assert x1.resolve_alias("Current RHEA access code?", aliases) == x1.Address("rhea", "key")
    assert x1.resolve_alias("Rhea and Io key", aliases) is None
    assert x1.resolve_alias("Rheas key", aliases) is None
    assert x1.resolve_alias("Rhea taxonomy", aliases) is None


def test_native_adapter_guard_order_and_restoration():
    events = []
    class FakeArena:
        def __init__(self):
            self.m = SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(live_shift=0))])
            self.live_shift = 115
            self.stop_sequences = ()
            self.cur_mounts = []
            self.width = 96
            self.grafts = [{"ntok": 20}] * 3
        def eb1_begin_turn(self):
            events.append("clear_boat")
        def _forward(self):
            events.append("forward")
        def _attempt(self, question, pages, ngen, deposit, stops, defer_memory):
            self.cur_mounts = [999]  # Simulate L2/native mount drift.
            events.append("mounted")
            self._forward()
            return "bad", {}
        def _eb1_frame_info(self):
            return {"frame_ephemeral": True}
    arena = FakeArena()
    original = arena._forward
    def guard(mounts):
        assert mounts == (999,)
        raise x1.PageFault("mismatch")
    with pytest.raises(x1.PageFault):
        x1.arena_mount_read(arena, "q", (2,), guard)
    assert events == ["clear_boat", "mounted"]
    assert arena._forward == original
    assert "_forward" not in arena.__dict__


def test_unseatable_address_never_enters_native_forward():
    arena = SimpleNamespace(width=96, grafts=[{"ntok": 97}])
    with pytest.raises(x1.PageFault, match="unseatable"):
        x1.arena_mount_read(arena, "q", (0,), lambda actual: None)


def test_b_unseatable_falls_back_without_false_c_fault():
    adapter, calls = make_adapter([2, 3])
    def read(*_args, **_kwargs):
        raise x1.PageFault("unseatable_current_version")
    adapter.mount_read = read
    answer, info = adapter.serve("q", address=x1.Address("e", "r"))
    assert calls == ["router"] and answer == "router answer"
    assert info["x1_fallback_reason"] == "unseatable_current_version"
    assert info["x1_page_fault"] is False
