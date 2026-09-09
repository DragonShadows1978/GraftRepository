"""Author baseline, not a blind red team.

Prior art: house mutation/logic discipline (2026); DeMillo, Lipton & Sayward
(1978) mutation testing, unverified — lead to check: hints on test data
selection 1978. Ours: these concrete provenance and relation attacks.
"""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import grm_x2_witnesses as x
from scripts.grm_x2_common import OUT, contract, read, frozen_mounts, today_guard, score_answer


def mount(text, sid="a"):
    return {"text": text, "metadata": {x.META: x.extract(text, sid)}}


def test_default_off_preserves_deposit_payload_and_answer_identity(monkeypatch):
    monkeypatch.delenv(x.FLAG, raising=False)
    class Arena:
        def __init__(self):
            self.grafts = []
        def deposit(self, text, **kwargs):
            self.grafts.append({"text": text, "h": b"native-kv"})
            return len(self.grafts) - 1
        deposit_from_cache = deposit
    arena = Arena()
    original = dict(arena.__dict__)
    assert x.install(arena) is None
    assert arena.__dict__ == original
    arena.deposit("The owner of Orion is Mira.")
    assert x.deposit_metadata(arena.grafts[0], "a") is None
    assert arena.grafts == [{"text": "The owner of Orion is Mira.", "h": b"native-kv"}]
    info = {"grounded": False, "metadata": object()}
    answer = "a deliberately unsupported assertion"
    out, receipt = x.gate_answer(answer, "opaque", [], info)
    assert out is answer and receipt is info


def test_default_off_real_arena_grounding_identity(monkeypatch):
    from core.graft_arena import ArenaCache
    monkeypatch.delenv(x.FLAG, raising=False)
    arena = ArenaCache.__new__(ArenaCache)
    arena.grafts = [{"text": "The owner of Orion is Mira.", "h": b"native-kv"}]
    before = dict(arena.__dict__)
    expected = arena._grounding_verdict("Mira", [0], "What is the owner of Orion?", normalized=True)
    assert x.install(arena) is None
    assert arena.__dict__ == before
    assert arena._grounding_verdict("Mira", [0], "What is the owner of Orion?", normalized=True) == expected


def test_witness_metadata_passes_existing_native_sync_seam():
    from core.graft_repository import GraftRepository
    from types import SimpleNamespace
    calls = []
    source = mount("The owner of Orion is Mira.")
    native = SimpleNamespace(set_metadata=lambda node_id, metadata: calls.append((node_id, metadata)))
    stub = SimpleNamespace(native_store=native, _native_node_ids={0: 7}, arena=SimpleNamespace(grafts=[source]))
    GraftRepository._native_set_metadata(stub, 0)
    assert calls == [(7, source["metadata"])]
    assert json.loads(json.dumps(calls[0][1])) == source["metadata"]


@pytest.mark.parametrize("token", [None, "", "0", "false", "off", "nonsense", "enable"])
def test_unknown_flag_fails_off(token):
    assert not x.enabled({} if token is None else {x.FLAG: token})


@pytest.mark.parametrize("token", ["1", "true", "YES", " on "])
def test_explicit_flag(token):
    assert x.enabled({x.FLAG: token})


def test_adapter_both_deposit_paths_metadata_and_prompt_isolation(monkeypatch):
    monkeypatch.setenv(x.FLAG, "1")
    encoded = []
    class Arena:
        def __init__(self):
            self.grafts = []
        def deposit(self, text, *args, **kwargs):
            encoded.append(text)
            self.grafts.append({"text": text, "h": b"native-kv"})
            return len(self.grafts) - 1
        deposit_from_cache = deposit
    arena = Arena()
    x.install(arena)
    x.install(arena)
    for fn in (arena.deposit, arena.deposit_from_cache):
        idx = fn("The owner of Orion is Mira.")
        assert x.META in arena.grafts[idx]["metadata"]
        assert arena.grafts[idx]["h"] == b"native-kv"
    assert encoded == ["The owner of Orion is Mira."] * 2
    assert not any("witness" in s for s in encoded)
    info = {"grounded": True}
    answer, info2 = x.gate_answer("Mira", "What is the owner of Orion?", [arena.grafts[0]], info)
    assert answer == "Mira" and info2[x.META]["accepted"]
    assert info == {"grounded": True}


def test_frozen_contract_counts_sources_and_sha():
    reg = contract()
    cpu, live = read(OUT / "fixtures/cpu.json"), read(OUT / "fixtures/live.json")
    assert len(cpu) == 40
    assert {k: sum(r["kind"] == k for r in cpu) for k in {r["kind"] for r in cpu}} == {
        "role_swapped": 10, "negated": 10, "scattered_token": 10, "correct_paraphrase": 10}
    assert {k: sum(r["mount_class"] == k for r in live) for k in {r["mount_class"] for r in live}} == {"correct": 12, "decoy": 6, "unavailable": 6}
    assert len({r["id"] for r in live}) == 24
    assert len(reg["gpu"]["cells"]) * reg["gpu"]["worker_seconds"] <= reg["gpu"]["budget_seconds"] <= 2700
    for case in cpu + live:
        for source in case["sources"]:
            assert x.sha(source["text"]) == source["source_version"]
    for case in cpu:
        if case["kind"] == "scattered_token":
            words = lambda s: set(x.canonical(s).split())
            assert words(case["answer"]) <= words(case["sources"][0]["text"])


def test_original_guard_accepts_known_wrong_relation():
    assert today_guard("The owner of Orion is Neri.", "What is the owner of Orion?",
                       [{"text": "The owner of Orion is Mira. The owner of Lyra is Neri."}])["accepted"]


@pytest.mark.parametrize("answer", ["The owner of Orion is Neri.", "Orion owns Mira.", "The owner of Orion is not Mira.",
                                   "Mira does not own Orion.", "Orion is not owned by Mira.", "The color of Orion is Mira.",
                                   "Mira owns Orion and Neri owns Lyra.", "Mira owns Orion. Penguins can fly.",
                                   "Mira owns Orion but is imaginary.", "Mira owns Orion, allegedly.",
                                   "Perhaps Mira owns Orion.", "If Mira owns Orion.", "Neri says Mira owns Orion.",
                                   "Orion is never owned by Mira.", "", "Mira or Neri"])
def test_unwitnessed_binding_or_extra_assertion_rejects(answer):
    assert not x.verify(answer, "What is the owner of Orion?", [mount("The owner of Orion is Mira.")])["accepted"]


def test_matching_negative_is_evidence_not_global_ban():
    result = x.verify("Mira does not own Orion.", "What is the owner of Orion?", [mount("The owner of Orion is not Mira.")])
    assert result["accepted"]


@pytest.mark.parametrize("text", ["Mira says the owner of Orion is Neri.", "If the owner of Orion is Neri.",
                                 "The owner of Orion might be Neri.", "The owner of Orion is not known.",
                                 "If it rains and the owner of Orion is Neri.",
                                 "Mira says the code of Atlas is blue and the owner of Orion is Neri."])
def test_reported_or_conditional_source_is_not_affirmative_evidence(text):
    assert not x.extract(text, "a")["witnesses"]


def test_source_offsets_and_unicode_preserved():
    text = "User: The code of Atlas is **Cobalt‑1‑India**.\nAssistant: Recorded: Atlas code is Cobalt-1-India."
    meta = x.extract(text, "versioned")
    assert len(meta["witnesses"]) == 2
    for w in meta["witnesses"]:
        a, b = w["span"]
        assert text[a:b] == w["text"]
        for field in w["fields"].values():
            a, b = field["span"]
            assert text[a:b] == field["text"]
    assert x.verify("Cobalt-1-India", "What is the Atlas code value?", [{"text": text, "metadata": {x.META: meta}}])["accepted"]


def test_no_scattered_value_token_union():
    text = "The code of Atlas is Cobalt. The code of Lyra is India. The key of Nova is 1."
    assert not x.verify("Cobalt 1 India", "What is the code of Atlas?", [mount(text)])["accepted"]


def test_source_version_tampering_rejects():
    source = mount("The owner of Orion is Mira.")
    source["metadata"][x.META]["source_version"] = "0" * 64
    assert not x.verify("Mira", "What is the owner of Orion?", [source])["accepted"]


def test_metadata_binding_tampering_rejects():
    source = mount("The owner of Orion is Mira.")
    source["metadata"][x.META]["witnesses"][0]["binding"]["value"] = "neri"
    assert not x.verify("Neri", "What is the owner of Orion?", [source])["accepted"]


def test_stale_metadata_no_answer_time_backfill():
    assert not x.verify("Mira", "What is the owner of Orion?", [{"text": "The owner of Orion is Mira."}])["accepted"]


def test_metadata_immutable(monkeypatch):
    monkeypatch.setenv(x.FLAG, "1")
    g = {"text": "The owner of Orion is Mira."}
    x.deposit_metadata(g, "a")
    g["text"] = "The owner of Orion is Neri."
    with pytest.raises(ValueError, match="immutable"):
        x.deposit_metadata(g, "a")


def test_two_source_conjunction_has_explicit_join():
    mounted = [mount("The code of Atlas is Cobalt-1-India.", "a"), mount("The key of Nova is Quartz-5-Bravo.", "b")]
    result = x.verify("Atlas code is Cobalt-1-India and Nova key is Quartz-5-Bravo.",
                      "What are the current Atlas code and Nova key values?", mounted)
    assert result["accepted"] and len(result["joins"]) == 1
    join = result["joins"][0]
    assert join["kind"] == "conjunction" and len(join["inputs"]) == 2
    assert {i["source_id"] for i in join["inputs"]} == {"a", "b"}
    assert len({i["source_version"] for i in join["inputs"]}) == 2


def test_multi_answer_cannot_omit_binding_or_mix_versions():
    sources = [mount("The code of Atlas is Cobalt-1-India.", "same"), mount("The key of Nova is Quartz-5-Bravo.", "same")]
    q = "What are the current Atlas code and Nova key values?"
    assert not x.verify("Atlas code is Cobalt-1-India and Nova key is Quartz-5-Bravo.", q, sources)["accepted"]
    assert not x.verify("Atlas code is Cobalt-1-India.", q, sources[:1])["accepted"]


def test_conflicting_affirmative_sources_fail_closed():
    sources = [mount("The owner of Orion is Mira.", "a"), mount("The owner of Orion is Neri.", "b")]
    assert not x.verify("Mira", "What is the owner of Orion?", sources)["accepted"]


def test_no_inferred_transitive_relation():
    sources = [mount("The owner of Orion is Mira.", "a"), mount("The manager of Mira is Neri.", "b")]
    assert not x.verify("The manager of Orion is Neri.", "What is the manager of Orion?", sources)["accepted"]


def test_existing_guard_failure_cannot_be_rescued_by_B(monkeypatch):
    monkeypatch.setenv(x.FLAG, "1")
    out, info = x.gate_answer("Mira", "What is the owner of Orion?", [mount("The owner of Orion is Mira.")], {"grounded": False})
    assert out == x.ABSTENTION and info["abstained"]


def test_scorer_does_not_call_witness_parser(monkeypatch):
    monkeypatch.setattr(x, "verify", lambda *a: pytest.fail("oracle must be independent"))
    fixture = {"mount_class": "correct", "expected_bindings": [{"entity": "Atlas", "relation": "code", "value": "Cobalt-1-India"}]}
    assert score_answer("Cobalt‑1‑India", fixture)["exact_answer"]
    assert score_answer("The Atlas code is not Cobalt-1-India", fixture)["false_answer"]
    assert not score_answer("Cobalt-1-India. Also Neri owns Orion", fixture)["exact_answer"]
    assert score_answer(x.ABSTENTION, fixture)["abstention_incorrect"]
    fixture = {"mount_class": "unavailable", "expected_bindings": [], "withheld_truth": "Cobalt-1-India"}
    assert score_answer(x.ABSTENTION, fixture)["abstention_correct"]
    assert score_answer("Cobalt-1-India", fixture)["false_answer"]
    assert score_answer("Cobalt-1-India", fixture)["unsupported_lucky_truth"]
