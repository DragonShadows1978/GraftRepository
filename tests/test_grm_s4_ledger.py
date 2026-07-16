"""GRM S4 grounding-hit ledger: CPU contract tests and lead-run GPU gates.

Default pytest collection is CPU-only.  Model/tensor imports live exclusively
inside ``--run-gpu`` functions.  The lead executes one process/model at a time:

    python3 tests/test_grm_s4_ledger.py --run-gpu --g0-smoke
    python3 tests/test_grm_s4_ledger.py --run-gpu --convo N   # N=1..4
    python3 tests/test_grm_s4_ledger.py --analyze
"""
import argparse
import glob
import importlib.util
import inspect
import json
import os
import sys
from types import MethodType, SimpleNamespace

import numpy as np
import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.graft_arena import ArenaCache
from tests.test_grm_importance_counterfactual import dependence, minus_node_picks
from tests.test_grm_importance_g1g2 import (
    ELIGIBILITY_CUTOFF,
    G1_MIN_MEDIAN_SPEARMAN,
    G1_MIN_TOP1_AGREEMENT,
    classify_fixture_turns,
    next_probe_answer_turn,
    next_scripted_deposit_turn,
    probe_is_eligible,
    scripted_deposit_text,
    spearman,
    top1_agreement,
    wipe_workspace,
)


FIXTURES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures",
    "importance_convos_repeat")
ARTIFACTS_DIR = os.path.join(FIXTURES_DIR, "artifacts")
CONVO_ARTIFACT_SCHEMA = "grm_s4_grounding_ledger_convo_v1"
VERDICT_SCHEMA = "grm_s4_grounding_ledger_verdict_v1"
G0_SMOKE_SCHEMA = "grm_s4_grounding_ledger_g0_v1"


# ---------------------------------------------------------------------------
# CPU: grounding split and transaction discipline
# ---------------------------------------------------------------------------

def _bare_arena(texts):
    arena = object.__new__(ArenaCache)
    arena.grafts = [
        {"text": text, "ntok": 1, "kind": "turn", "metadata": {
            "importance": {}}}
        for text in texts
    ]
    arena._s4_turn = 0
    arena.s4_metadata_callback = None
    return arena


def _legacy_grounded(arena, ans, mount_idxs, question):
    """The pre-S4 pooled implementation, frozen here for parity tests."""
    a = ans.lower()
    if any(h in a for h in arena.HEDGES):
        return False
    qrare = arena._rare_tokens(question)
    if qrare:
        mounted = set()
        for i in mount_idxs:
            mounted |= arena._rare_tokens(arena.grafts[i]["text"])
        if not (qrare & mounted):
            return False
    content = ((arena._rare_tokens(ans) | arena._caps_tokens(ans))
               - arena._rare_tokens(question)
               - arena._caps_tokens(question,
                                    skip_sentence_initial=False))
    have = set()
    words = set()
    for i in mount_idxs:
        text = arena.grafts[i]["text"]
        have |= arena._rare_tokens(text) | arena._caps_tokens(text, False)
        words |= {word.lower().rstrip(".,:;") for word in text.split()}
    if not content:
        question_words = {
            word.lower().rstrip(".,:;?") for word in question.split()}
        substantive = ({
            word.lower().rstrip(".,:;") for word in ans.split()
            if len(word.rstrip(".,:;")) >= 4
        } - question_words - arena.SCAFFOLD)
        return bool(substantive) and bool(substantive & words)
    return content <= have


@pytest.mark.parametrize(
    "texts,answer,mounts,question",
    [
        (["The deploy token is AXIOM-731."], "AXIOM-731.", [0],
         "What is the deploy token?"),
        (["Codes AXIOM-731 and CR-842 are active."],
         "AXIOM-731 and CR-842.", [0], "Which codes are active?"),
        (["The header parser crashed during import."],
         "The header parser crashed.", [0], "What failed during import?"),
        (["The deploy token is AXIOM-731."], "CR-842.", [0],
         "What is the deploy token?"),
        (["The deploy token is AXIOM-731."], "I don't know.", [0],
         "What is the deploy token?"),
        (["The deploy token is AXIOM-731."], "AXIOM-731.", [0],
         "Which record uses OMEGA-990?"),
        (["Nothing relevant is mounted."], "same place as last time", [0],
         "Where is it?"),
        ([], "AXIOM-731.", [], "What is the deploy token?"),
    ],
)
def test_grounding_split_preserves_pooled_verdict_bit_for_bit(
        texts, answer, mounts, question):
    arena = _bare_arena(texts)
    expected = _legacy_grounded(arena, answer, mounts, question)
    pooled, _contributors = arena._grounding_attribution(
        answer, mounts, question)
    assert pooled is expected
    assert arena._grounded(answer, mounts, question) is expected


def test_identifier_coverage_assigns_only_contributing_mounts():
    arena = _bare_arena([
        "The release token is AXIOM-731.",
        "The backup rack is CR-842.",
        "The lunch menu has tomato soup.",
    ])
    pooled, contributors = arena._grounding_attribution(
        "The values are AXIOM-731 and CR-842.", [0, 1, 2],
        "What are the two values?")
    assert pooled is True
    assert contributors == {0, 1}


def test_substantive_fallback_assigns_the_source_not_a_decoy():
    arena = _bare_arena([
        "The header parser crashed during import.",
        "The garden path stayed dry after lunch.",
    ])
    pooled, contributors = arena._grounding_attribution(
        "The header parser crashed.", [0, 1], "What failed during import?")
    assert pooled is True
    assert contributors == {0}


def test_hedge_has_no_grounding_hits_even_when_source_is_mounted():
    arena = _bare_arena(["The release token is AXIOM-731."])
    assert arena._grounding_attribution(
        "I am not sure; I can't find it.", [0],
        "What is the release token?") == (False, set())


def test_grounding_attribution_is_structurally_forward_free():
    from core.graft_repository import GraftRepository

    sources = [
        inspect.getsource(ArenaCache._grounding_attribution),
        inspect.getsource(ArenaCache._commit_s4_attempt),
        inspect.getsource(GraftRepository._persist_s4_metadata),
    ]
    for source in sources:
        assert "_forward(" not in source
        assert "_harvest(" not in source
        assert "self.m(" not in source


def test_s4_commit_updates_only_its_subkey_and_notifies_once():
    arena = _bare_arena(["source AXIOM-731", "decoy CR-842"])
    arena.grafts[0]["metadata"]["importance"] = {
        "s1_mass": 0.25, "s2_salience": 3}
    notifications = []
    arena.s4_metadata_callback = notifications.append

    arena._commit_s4_attempt([0, 1, 1], [0], [0])
    arena._commit_s4_attempt([0, 1], [0], [])

    importance = arena.grafts[0]["metadata"]["importance"]
    assert importance["s1_mass"] == 0.25
    assert importance["s2_salience"] == 3
    assert importance["s4"] == {
        "n_routed": 2,
        "n_mounted": 2,
        "n_grounded": 1,
        "last_grounded_turn": 1,
    }
    assert arena.grafts[1]["metadata"]["importance"]["s4"] == {
        "n_routed": 2,
        "n_mounted": 0,
        "n_grounded": 0,
        "last_grounded_turn": None,
    }
    assert notifications == [(0, 1), (0, 1)]


def test_s4_turn_ordinal_uses_persisted_turn_nodes_as_restart_floor():
    from core.graft_repository import GraftRepository

    arena = _bare_arena(["turn one", "turn two", "turn three"])
    repo = object.__new__(GraftRepository)
    repo.arena = arena
    repo._bind_s4_ledger()
    assert arena._s4_turn == 3
    assert arena._next_s4_turn() == 4
    arena._s4_turn = 0
    assert arena._next_s4_turn() == 1
    arena._s4_turn = 8
    assert arena._next_s4_turn() == 9


def test_scripted_feed_advances_turn_ordinal_without_touching_counters():
    arena = _bare_arena([])
    arena.ephemeral = True
    arena.deposit = lambda _text: 7
    assert arena.feed("User: scripted\n", deposit=True) == 7
    assert arena._s4_turn == 1
    assert arena.feed("User: retrieval-only\n", deposit=False) is None
    assert arena._s4_turn == 2
    assert arena.grafts == []


def _scripted_step_arena():
    arena = _bare_arena([
        "An unrelated record carries DECOY-100.",
        "The component assigned TARGET-200 is the header parser.",
    ])
    arena._s4_turn = 2       # two persisted conversation nodes precede probe
    arena.m = SimpleNamespace(layers=[])
    arena.ephemeral = False
    arena.live_segs = []
    arena.cur_mounts = []
    arena.cur_mount_n = 0
    arena.caches = object()
    arena.pos = 0
    arena.topk = 1
    arena.width = 64
    arena.stop_sequences = ()
    arena.route = lambda _text, exclude, limit=None: [0, 1]
    calls = []

    def attempt(self, user_text, picks, ngen, deposit, stops):
        calls.append(tuple(picks))
        if picks == [1]:
            answer = "The component is the header parser."
        else:
            answer = "I don't know."
        return answer, {
            "mounts": [i + 1 for i in picks],
            "resident": 0,
            "evicted": [],
            "live_tokens": 0,
        }

    arena._attempt = MethodType(attempt, arena)
    return arena, calls


def test_rolled_back_trips_never_increment_mount_or_grounded_counts():
    arena, calls = _scripted_step_arena()
    answer, info = arena.step(
        "Which component uses TARGET-200?", ngen=2, deposit=False,
        max_trips=2)

    assert answer == "The component is the header parser."
    assert info["trip"] == 2
    assert calls == [(0,), (0,), (1,)]
    decoy = arena.grafts[0]["metadata"]["importance"]["s4"]
    source = arena.grafts[1]["metadata"]["importance"]["s4"]
    # Both entered the one routed candidate ranking.  The decoy's two
    # discarded seating epochs do not become mounted or grounded events.
    assert decoy == {
        "n_routed": 1, "n_mounted": 0, "n_grounded": 0,
        "last_grounded_turn": None}
    assert source == {
        "n_routed": 1, "n_mounted": 1, "n_grounded": 1,
        "last_grounded_turn": 3}


def test_repository_metadata_path_persists_s4_and_restores_turn(tmp_path):
    # Reuse the predecessor's CPU-only repository/model double rather than
    # growing a second lifecycle fake in this harness.
    from tests.test_grm_importance_salience import make_repo

    repo = make_repo(tmp_path)
    repo.add_turn("Remember token AXIOM-731", "Recorded AXIOM-731")
    idx = repo.runtime.last_result.new_nodes[0]
    importance = repo.arena.grafts[idx]["metadata"]["importance"]
    importance["s2_salience"] = 2
    importance["s4"] = {
        "n_routed": 4, "n_mounted": 3, "n_grounded": 2,
        "last_grounded_turn": 7}
    repo._persist_s4_metadata((idx,))
    repo.flush_now()

    with open(tmp_path / "manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    persisted = manifest["nodes"][idx]["metadata"]["importance"]
    assert persisted == {
        "s2_salience": 2,
        "s4": {"n_routed": 4, "n_mounted": 3, "n_grounded": 2,
               "last_grounded_turn": 7},
    }

    reopened = make_repo(tmp_path)
    assert reopened.arena._s4_turn == 7
    assert (reopened.arena.grafts[idx]["metadata"]["importance"]["s4"]
            == persisted["s4"])
    reopened.arena._s4_turn = 0
    reopened.load()
    assert reopened.arena._s4_turn == 7


# ---------------------------------------------------------------------------
# CPU: repeat fixtures, sealed artifact math, inherited bars, lift
# ---------------------------------------------------------------------------

def _fixture_paths():
    return sorted(glob.glob(os.path.join(FIXTURES_DIR, "convo_*.json")))


def _load_fixture_path(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_repeat_validator():
    path = os.path.join(FIXTURES_DIR, "validate_fixtures.py")
    spec = importlib.util.spec_from_file_location(
        "grm_s4_repeat_fixture_validator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repeat_fixtures_validate_and_exercise_driver_solo_path():
    validator = _load_repeat_validator()
    paths = _fixture_paths()
    assert len(paths) == 4
    summaries = [validator.validate_file(path) for path in paths]
    assert sum(summary["n_probes"] for summary in summaries) == 32
    assert all(len(summary["targets"]) == 4 for summary in summaries)
    assert all(len(summary["controls"]) >= 2 for summary in summaries)

    collisions = {}
    for summary in summaries:
        for token in summary["identifiers"]:
            collisions.setdefault(token, set()).add(
                summary["conversation_id"])
    assert not {token: owners for token, owners in collisions.items()
                if len(owners) > 1}

    classified = {
        os.path.basename(path): classify_fixture_turns(
            _load_fixture_path(path)["turns"])
        for path in paths
    }
    solo_ids = [turn_id for rows in classified.values()
                for turn_id, shape in rows if shape == "solo"]
    assert solo_ids == [21, 22, 23, 24]


def compute_s4_g1(records):
    """Inherited G1 bars over S4-at-early versus late-probe S3 ranks."""
    correlations = []
    agreements = []
    n_eligible = 0
    for record in records:
        candidates = record["candidates"]
        ids = [str(candidate["candidate_id"]) for candidate in candidates]
        s4 = {str(candidate["candidate_id"]):
              int(candidate["s4_grounded_through_early"])
              for candidate in candidates}
        s3 = {str(candidate["candidate_id"]):
              float(candidate["s3_dep_dlogit"])
              for candidate in candidates}
        correlation = spearman([s4[cid] for cid in ids],
                               [s3[cid] for cid in ids])
        if not np.isnan(correlation):
            correlations.append(correlation)
        if probe_is_eligible(s3):
            n_eligible += 1
            agreements.append(top1_agreement(s4, s3, ids))
    median_spearman = (float(np.median(correlations))
                       if correlations else float("nan"))
    top1 = (float(sum(agreements) / len(agreements))
            if agreements else float("nan"))
    gate_pass = bool(
        correlations and agreements
        and median_spearman >= G1_MIN_MEDIAN_SPEARMAN
        and top1 >= G1_MIN_TOP1_AGREEMENT)
    return {
        "median_spearman": median_spearman,
        "top1_agreement": top1,
        "n_probes_ranked": len(correlations),
        "n_eligible": n_eligible,
        "n_probes_total": len(records),
        "pass": gate_pass,
    }


def compute_repeat_lift(records):
    """Registered secondary metric over the repeated target facts."""
    early = [record for record in records if bool(record["early_hit"])]
    no_early = [record for record in records if not bool(record["early_hit"])]
    p_late_early = (sum(bool(record["late_hit"]) for record in early)
                    / len(early) if early else None)
    p_late_no_early = (
        sum(bool(record["late_hit"]) for record in no_early) / len(no_early)
        if no_early else None)
    if p_late_early is None or p_late_no_early is None:
        lift = None
        status = "undefined_missing_condition"
    elif p_late_no_early == 0.0:
        lift = None
        status = "undefined_zero_baseline"
    else:
        lift = p_late_early / p_late_no_early
        status = "defined"
    return {
        "p_late_hit_given_early_hit": p_late_early,
        "p_late_hit_given_no_early_hit": p_late_no_early,
        "lift": lift,
        "status": status,
        "n_early_hit": len(early),
        "n_no_early_hit": len(no_early),
        "gate": None,
    }


def _candidate(candidate_id, s4, s3):
    return {
        "candidate_id": str(candidate_id),
        "turn_id": int(candidate_id),
        "node_class": "PROBED_LATER",
        "ground_truth_grade": 0,
        "s4_grounded_through_early": s4,
        "s3_dep_dlogit": s3,
        "s3_dep_kl": 0.0,
    }


def _record(name, s4_values, s3_values, early_hit=True, late_hit=True):
    return {
        "probe_id": name,
        "target_turn_id": 1,
        "early_probe_turn_id": 13,
        "late_probe_turn_id": 29,
        "early_hit": early_hit,
        "late_hit": late_hit,
        "candidates": [_candidate(i, s4, s3) for i, (s4, s3) in enumerate(
            zip(s4_values, s3_values), start=1)],
    }


def test_compute_s4_g1_applies_inherited_and_eligibility_bars():
    records = [
        _record("a", [0, 1, 2, 3], [0.1, 1.0, 2.0, 5.0]),
        _record("b", [1, 2, 3, 4], [0.2, 1.2, 2.2, 6.0]),
        # Ranked for Spearman, excluded only from top-1: max S3 < 3.762.
        _record("c", [0, 1, 2, 3], [0.1, 1.0, 2.0, 3.7]),
    ]
    verdict = compute_s4_g1(records)
    assert verdict["median_spearman"] == pytest.approx(1.0)
    assert verdict["top1_agreement"] == pytest.approx(1.0)
    assert verdict["n_probes_ranked"] == 3
    assert verdict["n_eligible"] == 2
    assert verdict["pass"] is True
    assert ELIGIBILITY_CUTOFF == pytest.approx(3.762)


def test_compute_s4_g1_red_when_spearman_bar_fails():
    records = [
        _record("a", [3, 2, 1, 0], [0.1, 1.0, 2.0, 5.0]),
        _record("b", [4, 3, 2, 1], [0.2, 1.2, 2.2, 6.0]),
    ]
    verdict = compute_s4_g1(records)
    assert verdict["median_spearman"] == pytest.approx(-1.0)
    assert verdict["pass"] is False


def test_repeat_lift_computes_registered_secondary_ratio():
    records = [
        _record("a", [0, 1], [1, 4], True, True),
        _record("b", [0, 1], [1, 4], True, True),
        _record("c", [0, 1], [1, 4], False, True),
        _record("d", [0, 1], [1, 4], False, False),
    ]
    lift = compute_repeat_lift(records)
    assert lift["p_late_hit_given_early_hit"] == 1.0
    assert lift["p_late_hit_given_no_early_hit"] == 0.5
    assert lift["lift"] == 2.0
    assert lift["status"] == "defined"
    assert lift["gate"] is None


def make_convo_artifact(conversation_id, records):
    return {"schema": CONVO_ARTIFACT_SCHEMA,
            "conversation_id": conversation_id,
            "late_probes": records}


def load_convo_artifacts(artifacts_dir=ARTIFACTS_DIR):
    paths = sorted(glob.glob(os.path.join(artifacts_dir, "convo_*.json")))
    artifacts = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            artifact = json.load(handle)
        if artifact.get("schema") != CONVO_ARTIFACT_SCHEMA:
            raise ValueError(
                f"{path}: schema {artifact.get('schema')!r}, expected "
                f"{CONVO_ARTIFACT_SCHEMA!r}")
        if not isinstance(artifact.get("late_probes"), list):
            raise ValueError(f"{path}: missing late_probes list")
        artifacts.append(artifact)
    return artifacts


def test_s4_artifact_round_trip_and_wrong_schema_rejection(tmp_path):
    artifact = make_convo_artifact(
        "convo_99_synthetic",
        [_record("synthetic#29", [0, 1], [1.0, 4.0])])
    path = tmp_path / "convo_99_synthetic.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert load_convo_artifacts(str(tmp_path)) == [artifact]
    artifact["schema"] = "wrong"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_convo_artifacts(str(tmp_path))


def test_analysis_mode_reads_four_sealed_artifacts(tmp_path, capsys):
    for fixture_path in _fixture_paths():
        conversation_id = _load_fixture_path(fixture_path)["conversation_id"]
        records = [
            _record(f"{conversation_id}#{29 + 2 * offset}",
                    [0, 1, 2, 3], [0.1, 1.0, 2.0, 5.0],
                    early_hit=(offset % 2 == 0), late_hit=True)
            for offset in range(4)
        ]
        artifact = make_convo_artifact(conversation_id, records)
        (tmp_path / f"{conversation_id}.json").write_text(
            json.dumps(artifact), encoding="utf-8")

    verdict = run_analysis(str(tmp_path))
    output = capsys.readouterr().out
    assert verdict["n_convos"] == 4
    assert verdict["n_late_probes"] == 16
    assert verdict["g1_s4"]["pass"] is True
    assert "Secondary lift (report-only)" in output
    assert VERDICT_SCHEMA in output


def analyze_artifacts(artifacts):
    records = [record for artifact in artifacts
               for record in artifact["late_probes"]]
    return {
        "schema": VERDICT_SCHEMA,
        "n_convos": len(artifacts),
        "n_late_probes": len(records),
        "eligibility_cutoff": ELIGIBILITY_CUTOFF,
        "bars": {
            "median_spearman": G1_MIN_MEDIAN_SPEARMAN,
            "top1_agreement": G1_MIN_TOP1_AGREEMENT,
        },
        "g1_s4": compute_s4_g1(records),
        "secondary_lift": compute_repeat_lift(records),
    }


def run_analysis(artifacts_dir=ARTIFACTS_DIR):
    artifacts = load_convo_artifacts(artifacts_dir)
    if len(artifacts) != 4:
        raise RuntimeError(
            f"expected four sealed S4 artifacts under {artifacts_dir}, "
            f"found {len(artifacts)}")
    conversation_ids = {artifact["conversation_id"] for artifact in artifacts}
    expected_ids = {
        _load_fixture_path(path)["conversation_id"] for path in _fixture_paths()
    }
    if conversation_ids != expected_ids:
        raise RuntimeError(
            f"sealed artifact set mismatch: observed={sorted(conversation_ids)}, "
            f"expected={sorted(expected_ids)}")
    if any(len(artifact["late_probes"]) != 4 for artifact in artifacts):
        raise RuntimeError("each sealed conversation artifact must contain "
                           "exactly four late probes")
    verdict = analyze_artifacts(artifacts)
    result = verdict["g1_s4"]
    lift = verdict["secondary_lift"]
    print(f"Loaded {verdict['n_convos']} sealed artifacts, "
          f"{verdict['n_late_probes']} late probes", flush=True)
    print(f"G1-S4 median Spearman={result['median_spearman']:.4f} "
          f"(>= {G1_MIN_MEDIAN_SPEARMAN}); top-1="
          f"{result['top1_agreement']:.4f} "
          f"(>= {G1_MIN_TOP1_AGREEMENT}, eligible={result['n_eligible']}); "
          f"PASS={result['pass']}", flush=True)
    print("Secondary lift (report-only): "
          f"P(late|early)={lift['p_late_hit_given_early_hit']}, "
          f"P(late|no-early)={lift['p_late_hit_given_no_early_hit']}, "
          f"lift={lift['lift']}, status={lift['status']}", flush=True)
    print(json.dumps(verdict, sort_keys=True), flush=True)
    return verdict


# ---------------------------------------------------------------------------
# Lead-run GPU gates.  No imports from this section execute under pytest.
# ---------------------------------------------------------------------------

def _load_gpu_repository(workspace, *, topk=3):
    tensor_path = "/mnt/ForgeRealm/Project-Tensor/tensor_cuda"
    if tensor_path not in sys.path:
        sys.path.insert(0, tensor_path)
    import tensor_cuda as tc
    from tokenizers import Tokenizer as HFTok
    from core.graft_repository import GraftRepository
    from core.minicpm3_tc import MiniCPM3_TC, _snap

    tokenizer = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
    model, info = MiniCPM3_TC.from_pretrained()
    print(f"loaded: {info}", flush=True)
    with tc.no_grad():
        model(np.array([[1, 2, 3]], dtype=np.int64), kv_caches=None,
              position_offset=0, last_token_only=True)

    wipe_workspace(workspace)
    repo = GraftRepository(
        model, lambda text: tokenizer.encode(text).ids,
        lambda ids: tokenizer.decode(ids), workspace,
        autosave=False, librarian_mode="deferred",
        sink_text="<conversation>\n", arena_width=512, route_layer=44,
        topk=topk, live_turns=2, ephemeral=False, recency_mounts=2)
    assert len(repo.arena.grafts) == 0, (
        f"workspace hygiene failure: {workspace} did not boot empty")
    for layer in model.layers:
        layer.self_attn.live_shift = repo.arena.live_shift
        layer.self_attn.absorbed_decode = True
    return repo, tc


def _s4_counts(arena, idx):
    value = (arena.grafts[idx].get("metadata", {})
             .get("importance", {}).get("s4"))
    return dict(value) if isinstance(value, dict) else {
        "n_routed": 0, "n_mounted": 0, "n_grounded": 0,
        "last_grounded_turn": None}


def _run_g0_gpu_smoke():
    workspace = "/tmp/graftrepo_s4_g0_smoke"
    repo, _tc = _load_gpu_repository(workspace, topk=2)
    arena = repo.arena
    repo.add_turn(
        "The coolant manifold identifier is OSPREY-771.",
        "Recorded: coolant manifold identifier OSPREY-771.")
    source = repo.runtime.last_result.new_nodes[0]
    repo.add_turn(
        "The break-room filter identifier is RAVEN-663.",
        "Recorded: break-room filter identifier RAVEN-663.")
    decoy = repo.runtime.last_result.new_nodes[0]
    repo.add_turn("The loading dock is quiet today.",
                  "Good time to move the spare crates.")
    repo.add_turn("The office fern has a new leaf.",
                  "It seems to like the brighter window.")

    original_route = arena.route
    original_attempt = arena._attempt
    try:
        arena.topk = 2
        arena.route = lambda _text, exclude, limit=None: [source, decoy]
        before_source = _s4_counts(arena, source)
        before_decoy = _s4_counts(arena, decoy)
        answer, info = arena.step(
            "What is the coolant manifold identifier?", ngen=32,
            deposit=False, max_trips=0)
        after_source = _s4_counts(arena, source)
        after_decoy = _s4_counts(arena, decoy)
        assert "osprey-771" in answer.lower(), answer
        assert after_source["n_grounded"] == before_source["n_grounded"] + 1
        assert after_decoy["n_grounded"] == before_decoy["n_grounded"]

        # Deterministic hedge overlay on a real generated/cache-mutating
        # attempt: grounding sees a hedge, and no mounted node gets a hit.
        def hedged_attempt(user_text, picks, ngen, deposit, stops):
            _answer, attempt_info = original_attempt(
                user_text, picks, ngen, deposit, stops)
            return "I don't know.", attempt_info

        arena._attempt = hedged_attempt
        arena.topk = 1
        arena.route = lambda _text, exclude, limit=None: [source]
        before_hedge = _s4_counts(arena, source)
        hedged, _ = arena.step(
            "Repeat the coolant identifier.", ngen=16,
            deposit=False, max_trips=0)
        after_hedge = _s4_counts(arena, source)
        assert "don't know" in hedged.lower()
        assert after_hedge["n_grounded"] == before_hedge["n_grounded"]
        arena._attempt = original_attempt

        # First two epochs are the same decoy (normal + clean-room ladder),
        # final epoch mounts the source.  Only the accepted source mount may
        # reach n_mounted/n_grounded.
        arena.topk = 1
        arena.route = lambda _text, exclude, limit=None: [decoy, source]
        before_source = _s4_counts(arena, source)
        before_decoy = _s4_counts(arena, decoy)
        rollback_answer, rollback_info = arena.step(
            "Which component is assigned OSPREY-771?", ngen=32,
            deposit=False, max_trips=2)
        after_source = _s4_counts(arena, source)
        after_decoy = _s4_counts(arena, decoy)
        assert rollback_info["trip"] == 2, (rollback_answer, rollback_info)
        assert after_decoy["n_mounted"] == before_decoy["n_mounted"]
        assert after_decoy["n_grounded"] == before_decoy["n_grounded"]
        assert after_source["n_mounted"] == before_source["n_mounted"] + 1
        assert after_source["n_grounded"] == before_source["n_grounded"] + 1
    finally:
        arena.route = original_route
        arena._attempt = original_attempt

    receipt = {
        "schema": G0_SMOKE_SCHEMA,
        "source_node": source,
        "decoy_node": decoy,
        "source_s4": _s4_counts(arena, source),
        "decoy_s4": _s4_counts(arena, decoy),
        "co_mount_answer": answer,
        "rollback_answer": rollback_answer,
        "rollback_trip": rollback_info["trip"],
        "pass": True,
    }
    print(json.dumps(receipt, sort_keys=True), flush=True)
    print("G0-S4 PASS", flush=True)
    return receipt


def _fixture_for_convo(convo_n):
    matches = [path for path in _fixture_paths()
               if os.path.basename(path).startswith(f"convo_{convo_n:02d}_")]
    if len(matches) != 1:
        raise RuntimeError(f"convo {convo_n}: expected one fixture, "
                           f"found {matches}")
    return _load_fixture_path(matches[0]), matches[0]


def _target_turn_id(probe):
    targets = [int(tid) for tid, grade in probe["relevance"].items()
               if grade == 3]
    if len(targets) != 1:
        raise ValueError(f"probe {probe['probe_turn_id']} has targets {targets}")
    return targets[0]


def _teacher_forced_logits(arena, tc, prompt_ids, reply_ids, picks):
    arena.caches, arena.pos, arena.live_segs = None, 0, []
    arena.cur_mounts, arena.cur_mount_n = [], 0
    arena._ensure_h(picks)
    from core import kv_graft

    mounts = [{"h": arena.sink_h}] + [arena.grafts[i] for i in picks]
    to_numpy = lambda tensor: (
        tensor if isinstance(tensor, np.ndarray) else tensor.numpy())
    injection = []
    for layer_index in range(len(arena.m.layers)):
        injection.append({
            key: np.concatenate(
                [to_numpy(graft["h"][layer_index][key]) for graft in mounts],
                axis=dim)
            for key, dim in arena.PAYLOAD
        })
    arena._set_injection_host(injection)
    arena.cur_mounts = list(picks)
    arena.cur_mount_n = sum(arena.grafts[i]["ntok"] for i in picks)
    arena._commit_native_mount(picks, arena.cur_mount_n)
    for layer in arena.m.layers:
        layer.self_attn.live_shift = arena.live_shift
    full_ids = prompt_ids + reply_ids
    with tc.no_grad():
        logits, _ = arena.m(
            np.array([full_ids], dtype=np.int64), kv_caches=None,
            position_offset=arena.live_shift, last_token_only=False)
    kv_graft.clear_injection(arena.m)
    rows = logits.numpy()[0].astype(np.float32)
    start = len(prompt_ids) - 1
    return rows[start:start + len(reply_ids)]


def _run_g1_gpu_leg(convo_n):
    fixture, fixture_path = _fixture_for_convo(convo_n)
    print(f"driving {fixture_path} ({fixture['conversation_id']})", flush=True)
    workspace = f"/tmp/graftrepo_s4_convo_{convo_n:02d}"
    repo, tc = _load_gpu_repository(workspace, topk=3)
    arena = repo.arena
    turns = fixture["turns"]
    probes_by_turn = {probe["probe_turn_id"]: probe
                      for probe in fixture["probes"]}
    turn_by_id = {turn["turn_id"]: turn for turn in turns}
    turn_id_to_graft = {}

    def deposit_scripted(user_turn, assistant_turn=None):
        before_len = len(arena.grafts)
        if assistant_turn is None:
            before = repo._snapshot_state()
            arena.feed(scripted_deposit_text(user_turn))
            repo._set_new_node_provenance(before, "exchange_span")
            extracted = repo._extract_from_new_turns(
                before, context={"event": "add_turn",
                                 "user_text": user_turn["text"],
                                 "assistant_text": None})
            result = repo.runtime._finish_turn_event(
                "add_turn", before, extraction=extracted, autosave=False)
            repo._queue_s2_pending()
            new_nodes = result.new_nodes
        else:
            repo.add_turn(user_turn["text"], assistant_turn["text"])
            new_nodes = repo.runtime.last_result.new_nodes
        assert len(new_nodes) == 1 and new_nodes[0] == before_len, (
            f"expected one node at {before_len}, got {new_nodes}")
        graft_idx = new_nodes[0]
        turn_id_to_graft[user_turn["turn_id"]] = graft_idx
        if assistant_turn is not None:
            turn_id_to_graft[assistant_turn["turn_id"]] = graft_idx
        return graft_idx

    early = {}
    late_records = []
    index = 0
    while index < len(turns):
        turn = turns[index]
        if turn["node_class"] != "PROBE":
            assistant, next_index = next_scripted_deposit_turn(turns, index)
            deposit_scripted(turn, assistant)
            index = next_index
            continue

        probe = probes_by_turn[turn["turn_id"]]
        target_tid = _target_turn_id(probe)
        candidate_tids = sorted(int(tid) for tid in probe["relevance"])
        candidate_nodes = {tid: turn_id_to_graft[tid]
                           for tid in candidate_tids}
        target_node = candidate_nodes[target_tid]
        before_hit = _s4_counts(arena, target_node)["n_grounded"]
        reply, attempt_info = arena.step(
            probe["question"], ngen=48, deposit=False, max_trips=2)
        after_hit = _s4_counts(arena, target_node)["n_grounded"]
        target_hit = after_hit > before_hit
        answer_turn, next_index = next_probe_answer_turn(turns, index)
        scripted_answer = answer_turn["text"] if answer_turn else None
        print(f"  probe={turn['turn_id']} target={target_tid} "
              f"reply={reply[:80]!r} mounts={attempt_info['mounts']} "
              f"hit={target_hit}", flush=True)

        if target_tid not in early:
            early[target_tid] = {
                "probe_turn_id": turn["turn_id"],
                "answer": reply,
                "hit": target_hit,
                "counts": {tid: _s4_counts(arena, node)["n_grounded"]
                           for tid, node in candidate_nodes.items()},
            }
            index = next_index
            continue

        early_record = early[target_tid]
        resume = (arena.caches, arena.pos, list(arena.live_segs),
                  list(arena.cur_mounts), arena.cur_mount_n,
                  list(arena.grafts))
        prompt_ids = arena.encode(arena._format_step_prompt(probe["question"]))
        reply_ids = arena.encode(reply)
        if not reply_ids:
            raise RuntimeError(f"late probe {turn['turn_id']} produced no tokens")
        picks = sorted(set(candidate_nodes.values()))
        try:
            _teacher_forced_logits(arena, tc, prompt_ids, reply_ids, picks)
            full_logits = _teacher_forced_logits(
                arena, tc, prompt_ids, reply_ids, picks)
            candidates = []
            for candidate_tid in candidate_tids:
                node = candidate_nodes[candidate_tid]
                minus_logits = _teacher_forced_logits(
                    arena, tc, prompt_ids, reply_ids,
                    minus_node_picks(picks, node))
                dep = dependence(full_logits, minus_logits)
                candidates.append({
                    "candidate_id": str(candidate_tid),
                    "turn_id": candidate_tid,
                    "node_class": turn_by_id[candidate_tid]["node_class"],
                    "ground_truth_grade": probe["relevance"][
                        str(candidate_tid)],
                    "s4_grounded_through_early": early_record["counts"][
                        candidate_tid],
                    "s3_dep_dlogit": dep["mean_abs_dlogit"],
                    "s3_dep_kl": dep["mean_kl"],
                })
                print(f"    candidate={candidate_tid} node={node} "
                      f"s4_early={early_record['counts'][candidate_tid]} "
                      f"s3_dlogit={dep['mean_abs_dlogit']:.4f} "
                      f"s3_kl={dep['mean_kl']:.4f}", flush=True)
        finally:
            (arena.caches, arena.pos, arena.live_segs, arena.cur_mounts,
             arena.cur_mount_n) = (resume[0], resume[1], list(resume[2]),
                                   list(resume[3]), resume[4])
            arena.grafts[:] = resume[5]
            from core import kv_graft
            kv_graft.clear_injection(arena.m)

        late_records.append({
            "probe_id": f"{fixture['conversation_id']}#{turn['turn_id']}",
            "target_turn_id": target_tid,
            "early_probe_turn_id": early_record["probe_turn_id"],
            "late_probe_turn_id": turn["turn_id"],
            "early_answer": early_record["answer"],
            "late_answer": reply,
            "expected_answer_scripted": scripted_answer,
            "early_hit": early_record["hit"],
            "late_hit": target_hit,
            "candidates": candidates,
        })
        index = next_index

    if len(early) != 4 or len(late_records) != 4:
        raise RuntimeError(
            f"fixture pairing failed: early={len(early)} late={len(late_records)}")
    artifact = make_convo_artifact(fixture["conversation_id"], late_records)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    out_path = os.path.join(
        ARTIFACTS_DIR, f"{fixture['conversation_id']}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
    print(f"wrote {out_path}", flush=True)
    print(json.dumps({"schema": CONVO_ARTIFACT_SCHEMA,
                      "conversation_id": fixture["conversation_id"],
                      "n_late_probes": len(late_records)},
                     sort_keys=True), flush=True)
    print("DONE", flush=True)
    return artifact


def _main(argv=None):
    parser = argparse.ArgumentParser(
        description="GRM S4 grounding ledger gates and analysis")
    parser.add_argument("--run-gpu", action="store_true",
                        help="authorize one lead-run GPU gate")
    parser.add_argument("--g0-smoke", action="store_true",
                        help="run G0-S4 attribution/rollback GPU smoke")
    parser.add_argument("--convo", type=int, choices=range(1, 5),
                        help="run one G1-S4 repeat fixture leg")
    parser.add_argument("--analyze", action="store_true",
                        help="CPU analysis of four sealed G1 artifacts")
    args = parser.parse_args(argv)

    if args.run_gpu:
        if args.analyze:
            parser.error("--analyze is CPU-only; omit --run-gpu")
        if bool(args.g0_smoke) == bool(args.convo is not None):
            parser.error("--run-gpu requires exactly one of --g0-smoke or "
                         "--convo N")
        if args.g0_smoke:
            _run_g0_gpu_smoke()
        else:
            _run_g1_gpu_leg(args.convo)
        return 0
    if args.g0_smoke or args.convo is not None:
        parser.error("--g0-smoke/--convo require --run-gpu")
    if args.analyze:
        verdict = run_analysis()
        return 0 if verdict["g1_s4"]["pass"] else 1
    print("No action flag: run CPU units with `python3 -m pytest "
          "tests/test_grm_s4_ledger.py -q`. GPU: --run-gpu --g0-smoke "
          "or --run-gpu --convo N (N=1..4). Analysis: --analyze.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
