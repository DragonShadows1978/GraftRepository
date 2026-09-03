"""GRM-LSR-P2C.1 G1 — CPU tests for the census scoring driver's pure parts.

ORDER: ``orders/GRM_LSR_P2C_1_E2E_CENSUS_SCORING.md`` (gate G1).
REGISTRATION: ``artifacts/lsr_p2c_1/registration.json``.

These cover the three things the order names -- probe selection, N derivation
for the three new turns pinned against the frozen instrumentation, and table
assembly -- plus the reachability rule and the reproduction comparator, which
are the two places a wrong answer would silently become a passing gate.

Nothing here loads a model or touches the card.  The frozen artifacts are read
only; no test writes anywhere under the frozen run tree.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_lsr_p2c_1_census as p2c1  # noqa: E402

pytestmark = pytest.mark.skipif(
    not p2c1.CENSUS.is_file(),
    reason="the frozen DET1 race tree is not present in this checkout",
)


# ---------------------------------------------------------------------------
# Fixtures over the frozen evidence
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def families() -> list[str]:
    return sorted(set(p2c1.PROBE_SESSIONS.values()))


@pytest.fixture(scope="module")
def node_counts(families) -> dict[str, dict[int, int]]:
    return {family: p2c1.family_probe_node_counts(family)
            for family in families}


@pytest.fixture(scope="module")
def flushes(families) -> dict[str, tuple]:
    return {family: p2c1.no_fold_flushes(family) for family in families}


@pytest.fixture(scope="module")
def probes(node_counts, flushes) -> list[dict]:
    return p2c1.census_probes(node_counts=node_counts, flushes=flushes)


# ---------------------------------------------------------------------------
# Probe selection
# ---------------------------------------------------------------------------
def test_ten_census_probes_are_selected_in_turn_order(probes):
    assert [p["probe_id"] for p in probes] == list(p2c1.CENSUS_PROBE_IDS)
    turns = [p["conversation_turn"] for p in probes]
    assert turns == sorted(turns), "probes must come out in session-turn order"


def test_every_e2e_row_of_the_census_is_selected():
    """The denominator is the census's OWN e2e rows, not a hand-kept list."""
    rows = p2c1.census_rows()
    from_census = sorted(k for k in rows if k.startswith("e2e_"))
    assert from_census == sorted(p2c1.CENSUS_PROBE_IDS)


def test_candidate_observations_are_one_to_one_with_the_census():
    """The 1:1 binding is the premise that lets one row carry each probe."""
    census = p2c1.census_rows()
    candidates = p2c1.candidate_rows()
    assert set(census) == set(candidates)
    for probe_id, row in census.items():
        candidate = candidates[probe_id]
        assert candidate["served_answer"] == row["served_answer"]
        assert bool(candidate["served_answer_correct"]) == bool(
            row["served_answer_correct"])


def test_lived_rows_carry_the_census_text_and_verdict(probes):
    census = p2c1.census_rows()
    for probe in probes:
        row = census[probe["probe_id"]]
        assert probe["lived"]["served_answer"] == row["served_answer"]
        assert probe["lived"]["served_correct"] == bool(
            row["served_answer_correct"])


def test_nine_of_ten_are_lived_correct_and_t33_is_the_refusal(probes):
    """The PASS RULE's denominator, read off the artifact rather than assumed."""
    correct = [p["probe_id"] for p in probes if p["lived"]["served_correct"]]
    incorrect = [
        p["probe_id"] for p in probes if not p["lived"]["served_correct"]]
    assert len(correct) == 9
    assert incorrect == ["e2e_t33_polaris_mark"]
    t33 = next(p for p in probes if p["probe_id"] == "e2e_t33_polaris_mark")
    assert t33["census_class"] == "REFUSAL"


def test_census_probes_refuses_to_run_without_derived_node_counts(flushes):
    """N is EVIDENCE. A default would let a probe id be read as a node index."""
    with pytest.raises(p2c1.P2C1Error, match="node_counts"):
        p2c1.census_probes(node_counts=None, flushes=flushes)


def test_census_probes_refuses_to_run_without_flushes(node_counts):
    with pytest.raises(p2c1.P2C1Error, match="flushes"):
        p2c1.census_probes(node_counts=node_counts, flushes=None)


def test_census_probes_fails_closed_on_a_broken_one_to_one_binding(
    node_counts, flushes,
):
    census = {k: dict(v) for k, v in p2c1.census_rows().items()}
    census["e2e_t05_orion_pin"]["served_answer"] = "a different answer"
    with pytest.raises(p2c1.P2C1Error, match="1:1 binding is broken"):
        p2c1.census_probes(
            census=census, node_counts=node_counts, flushes=flushes)


def test_candidate_rows_fails_closed_on_a_duplicate(tmp_path):
    line = (
        '{"fixture_id": "x", "served_answer": "a", '
        '"effective_fixture": {}, "source_snapshot": {}}'
    )
    path = tmp_path / "dupe.jsonl"
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(p2c1.P2C1Error, match="not 1:1"):
        p2c1.candidate_rows(path)


# ---------------------------------------------------------------------------
# N derivation -- the three new turns, pinned against the frozen evidence
# ---------------------------------------------------------------------------
def test_probe_id_is_a_zero_based_conversation_turn_not_a_node_index():
    assert p2c1.fixture_conversation_turn("e2e_t09_cypher_bridge") == 10
    assert p2c1.fixture_conversation_turn("e2e_t16_lyra_dock") == 17
    assert p2c1.fixture_conversation_turn("e2e_t33_polaris_mark") == 34
    with pytest.raises(p2c1.P2C1Error):
        p2c1.fixture_conversation_turn("sup_harbor_restatement")


def test_the_three_new_turns_derive_the_N_the_registration_pinned(node_counts):
    """t09 -> 10, t16 -> 18, t33 -> 36, from the frozen instrumentation.

    These are the values ``artifacts/lsr_p2c_1/registration.json`` registered
    BEFORE any gate ran.  If the backward walk ever stops producing them the
    reconstruction is describing a different index than the one registered.
    """
    assert node_counts["calibration"][10] == 10
    assert node_counts["calibration"][17] == 18
    assert node_counts["polaris"][34] == 36


def test_the_seven_carried_N_are_unchanged_from_sc1_2(node_counts):
    """SC1.2 registered 5/15/22/25/27/29/33. Carrying means not drifting."""
    eval_counts = node_counts["eval"]
    assert [eval_counts[t] for t in (6, 14, 20, 23, 25, 27, 31)] == [
        5, 15, 22, 25, 27, 29, 33]


def test_node_index_and_conversation_turn_diverge_after_a_supersede(
    node_counts,
):
    """The trap the derivation exists to avoid, asserted rather than described.

    Turn 14 (``e2e_t13_orion_pin``) deposits node 15, not node 13: a supersede
    turn deposits TWO nodes.  Reading N off the probe id would reconstruct a
    13-node index and hand the turn a prefix missing the node it mounts.
    """
    assert node_counts["eval"][14] == 15
    assert node_counts["eval"][14] != 13


def test_derived_N_matches_every_probes_selection(probes, node_counts):
    for probe in probes:
        family = probe["session_family"]
        turn = probe["conversation_turn"]
        assert probe["probe_node_index"] == node_counts[family][turn]


def test_every_probe_uses_its_own_sessions_derivation(probes):
    """t09/t16 are calibration; t33 is the polaris chain; the rest are eval."""
    by_family: dict[str, list[str]] = {}
    for probe in probes:
        by_family.setdefault(probe["session_family"], []).append(
            probe["probe_id"])
    assert by_family["calibration"] == [
        "e2e_t09_cypher_bridge", "e2e_t16_lyra_dock"]
    assert by_family["polaris"] == ["e2e_t33_polaris_mark"]
    assert len(by_family["eval"]) == 7


def test_the_calibration_session_is_not_the_eval_session():
    """The measured reason t09/t16 need their own source.

    If these two repositories ever agreed on every deposit-time field the
    separate calibration family would be unnecessary -- but they do not, and
    reconstructing a calibration probe from the eval superset would fabricate
    an index.  Asserted so the divergence cannot silently disappear.
    """
    from scripts.grm_sc1_2_session import prefix_agreement

    eval_manifest = p2c1.read_json(
        p2c1.session_dir("eval") / "repository" / "manifest.json")
    cal_manifest = p2c1.read_json(
        p2c1.session_dir("calibration") / "repository" / "manifest.json")
    agreement = prefix_agreement(
        eval_manifest["nodes"], cal_manifest["nodes"])
    assert not agreement["agree"]
    assert {d["node"] for d in agreement["diffs"]} == {14, 18}


def test_the_polaris_session_reaches_the_turn_the_eval_chain_cannot(
    node_counts,
):
    """t33 exists only in the 34-turn plant-registration chain."""
    assert 34 not in node_counts["eval"]
    assert 34 in node_counts["polaris"]


def test_polaris_prefix_includes_the_polaris_fact_node():
    """N=36 must contain node 35, the fact the probe is asked to recall.

    A prefix of 35 would exclude it and the probe could not answer from
    memory at all, which would make a refusal meaningless as evidence.
    """
    manifest = p2c1.read_json(
        p2c1.session_dir("polaris") / "repository" / "manifest.json")
    assert len(manifest["nodes"]) == 37
    assert "polaris mark" in manifest["nodes"][35]["text"].lower()


# ---------------------------------------------------------------------------
# no_fold bracketing
# ---------------------------------------------------------------------------
def test_no_fold_flushes_are_read_from_the_frozen_manifests(flushes):
    assert flushes["eval"] == (
        (11, ()), (19, ()), (28, (1, 4, 6, 8)), (34, (1, 4, 6, 8)))
    assert flushes["calibration"] == ((18, ()), (19, ()))
    assert flushes["polaris"] == (
        (11, ()), (19, ()), (28, (1, 4, 6, 8)), (37, (1, 4, 6, 8)))


def test_agreeing_brackets_pin_and_disagreeing_brackets_do_not():
    rows = ((11, ()), (19, ()), (28, (1, 4)), (34, (1, 4)))
    # Below the first flush: nothing above disagrees with the empty default.
    assert p2c1.no_fold_at_prefix(10, rows) == (frozenset(), True)
    # Strictly between two DISAGREEING flushes: unpinned, lower bracket used.
    assert p2c1.no_fold_at_prefix(22, rows) == (frozenset(), False)
    # Between two AGREEING flushes: pinned.
    assert p2c1.no_fold_at_prefix(30, rows) == (frozenset({1, 4}), True)
    # At or beyond the last flush: pinned.
    assert p2c1.no_fold_at_prefix(34, rows) == (frozenset({1, 4}), True)


def test_the_three_new_probes_are_all_no_fold_pinned(probes):
    by_id = {p["probe_id"]: p for p in probes}
    for probe_id in (
        "e2e_t09_cypher_bridge", "e2e_t16_lyra_dock", "e2e_t33_polaris_mark",
    ):
        assert by_id[probe_id]["no_fold_pinned"] is True


def test_the_unpinned_probes_are_exactly_sc1_2s_three(probes):
    unpinned = [p["probe_id"] for p in probes if not p["no_fold_pinned"]]
    assert unpinned == [
        "e2e_t19_nova_key", "e2e_t22_mira_seal", "e2e_t24_terra_port"]


def test_disagreeing_flushes_at_the_same_node_count_fail_closed(monkeypatch):
    """A family whose two flushes disagree at one node count cannot bracket."""
    monkeypatch.setattr(
        p2c1, "family_flush_manifests",
        lambda family: {
            "a": {"nodes": [{"no_fold": False}, {"no_fold": True}]},
            "b": {"nodes": [{"no_fold": True}, {"no_fold": False}]},
        })
    with pytest.raises(p2c1.P2C1Error, match="disagreeing no_fold flushes"):
        p2c1.no_fold_flushes("eval")


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------
def test_all_ten_probes_are_reachable(probes):
    for probe in probes:
        reach = p2c1.snapshot_reachability(probe)
        assert reach["reachable"], (probe["probe_id"], reach["reasons"])


def test_every_snapshot_is_a_finalized_lived_rung_zero_capture(probes):
    """The measured fact that licenses this instrument on the census leg."""
    for probe in probes:
        reach = p2c1.snapshot_reachability(probe)
        assert reach["schema"] == "grm.det1_3.model_visible_snapshot.v2"
        assert reach["label"] == "lived"
        assert reach["provenance_arm"] == "lived"
        assert reach["phase"] == "before_probe_prefill"
        assert reach["capture_finalized"] is True
        assert reach["probe_selected_attempt"] == 0
        assert reach["captured_attempt_ordinal"] == 0
        assert reach["linked_probe_answer_matches_census"] is True


def test_all_ten_snapshots_share_one_frame_identity(probes):
    """A differing frame would make the fork target arena the wrong model."""
    frames = {
        p2c1.snapshot_reachability(probe)["frame_sha256"] for probe in probes}
    assert len(frames) == 1


def test_lived_ranking_lies_inside_every_reconstructed_prefix(probes):
    for probe in probes:
        reach = p2c1.snapshot_reachability(probe)
        assert reach["ranking_outside_prefix"] == []
        assert max(reach["lived_admission_ranking"]) < probe[
            "probe_node_index"]


def test_a_missing_snapshot_is_unreachable_not_an_exception(probes):
    probe = dict(probes[0])
    probe["snapshot_manifest"] = "artifacts/does/not/exist.json"
    reach = p2c1.snapshot_reachability(probe)
    assert reach["reachable"] is False
    assert reach["reasons"] == [
        "frozen lived snapshot is absent from the tree"]


def test_a_ranking_outside_the_prefix_is_unreachable(probes):
    """The prefix rule failing is a RESULT, never something to truncate away."""
    probe = dict(probes[-1])
    probe["probe_node_index"] = 1
    reach = p2c1.snapshot_reachability(probe)
    assert reach["reachable"] is False
    assert any("escapes the derived prefix" in r for r in reach["reasons"])


# ---------------------------------------------------------------------------
# Arm 0 reproduction comparator
# ---------------------------------------------------------------------------
def test_text_leg_is_the_det1_semantic_comparator_not_bytes():
    """``Cobalt 1 India`` vs ``Cobalt-1-India`` is the census's own t30 case."""
    verdict = p2c1.reproduction_verdict(
        lived_answer="Cobalt 1 India",
        fork_answer="Cobalt‑1‑India",
        lived_min_mass=None, fork_min_mass=None,
        mass_leg_available=False, byte_exact_fork=True)
    assert verdict["text_bytes_match"] is False
    # U+2011 collapses to "-" under the DET1 projection; the space does not.
    assert verdict["text_normalized_match"] is False

    same = p2c1.reproduction_verdict(
        lived_answer="**Auric-4-Alpha**", fork_answer="Auric‑4‑Alpha",
        lived_min_mass=None, fork_min_mass=None,
        mass_leg_available=False, byte_exact_fork=True)
    assert same["text_bytes_match"] is False
    assert same["text_normalized_match"] is True
    assert same["reproduced"] is True


def test_an_absent_mass_leg_is_never_counted_as_a_match():
    verdict = p2c1.reproduction_verdict(
        lived_answer="x", fork_answer="x",
        lived_min_mass=None, fork_min_mass=0.5,
        mass_leg_available=False, byte_exact_fork=True)
    assert verdict["mass_leg_available"] is False
    assert verdict["min_mass_exact_match"] is None
    assert verdict["min_mass_delta"] is None
    assert "NO MASS LEG" in verdict["min_mass_comparison_basis"]
    assert verdict["reproduced"] is True


def test_an_available_mass_leg_must_match_exactly():
    match = p2c1.reproduction_verdict(
        lived_answer="x", fork_answer="x",
        lived_min_mass=0.25, fork_min_mass=0.25,
        mass_leg_available=True, byte_exact_fork=True)
    assert match["min_mass_delta"] == 0.0
    assert match["min_mass_exact_match"] is True
    assert match["reproduced"] is True

    off = p2c1.reproduction_verdict(
        lived_answer="x", fork_answer="x",
        lived_min_mass=0.25, fork_min_mass=0.2500001,
        mass_leg_available=True, byte_exact_fork=True)
    assert off["min_mass_exact_match"] is False
    assert off["reproduced"] is False, "no tolerance may rescue a mass miss"


def test_a_text_miss_cannot_be_rescued_by_the_mass_leg():
    verdict = p2c1.reproduction_verdict(
        lived_answer="Marble-4-Juliet", fork_answer="Polaris-1-Alpha",
        lived_min_mass=0.25, fork_min_mass=0.25,
        mass_leg_available=True, byte_exact_fork=True)
    assert verdict["min_mass_exact_match"] is True
    assert verdict["reproduced"] is False


def test_a_refusal_reproduces_only_as_the_same_refusal():
    """t33's lived row is a refusal; a confabulated value is NOT-REPRODUCED."""
    refusal = "I’m sorry, but I don’t have that information."
    same = p2c1.reproduction_verdict(
        lived_answer=refusal, fork_answer=refusal,
        lived_min_mass=None, fork_min_mass=None,
        mass_leg_available=False, byte_exact_fork=True)
    assert same["reproduced"] is True

    confabulated = p2c1.reproduction_verdict(
        lived_answer=refusal,
        fork_answer="The current polaris mark value is Polaris-1-Alpha.",
        lived_min_mass=None, fork_min_mass=None,
        mass_leg_available=False, byte_exact_fork=True)
    assert confabulated["reproduced"] is False


def test_only_the_seven_eval_probes_have_a_mass_leg(probes):
    with_mass = [p["probe_id"] for p in probes if p["mass_leg_available"]]
    without = [p["probe_id"] for p in probes if not p["mass_leg_available"]]
    assert len(with_mass) == 7
    assert without == [
        "e2e_t09_cypher_bridge", "e2e_t16_lyra_dock", "e2e_t33_polaris_mark"]


def test_the_seven_lived_minima_are_the_ones_sc1_2_registered(probes):
    by_id = {p["probe_id"]: p["lived"]["served_min_mass"] for p in probes}
    assert by_id["e2e_t05_orion_pin"] == 0.3670685812830925
    assert by_id["e2e_t13_orion_pin"] == 0.38534095510840416
    assert by_id["e2e_t19_nova_key"] == 0.33923041572173435
    assert by_id["e2e_t22_mira_seal"] == 0.44133302321036655
    assert by_id["e2e_t24_terra_port"] == 0.3830605794986089
    assert by_id["e2e_t26_ember_code"] == 0.417310930788517
    assert by_id["e2e_t30_atlas_tone"] == 0.3625904594858487


# ---------------------------------------------------------------------------
# Table assembly
# ---------------------------------------------------------------------------
def _result(
    probe_id, *, reachable=True, reproduced=True, lived_correct=True,
    arm1_correct=True, lived="v", arm1_value="v", turn=1, node_index=5,
):
    return {
        "probe_id": probe_id,
        "session_family": "eval",
        "conversation_turn": turn,
        "probe_node_index": node_index,
        "no_fold_pinned": True,
        "reachability": {
            "reachable": reachable,
            "reasons": [] if reachable else ["no snapshot"],
        },
        "index_fidelity": {"rank1_match": True, "prefix_match": True},
        "lived": {
            "served_answer": lived,
            "served_correct": lived_correct,
            "census_class": "LAWFUL" if lived_correct else "REFUSAL",
        },
        "arm0": {
            "reproduced": reproduced,
            "fork_answer": lived if reproduced else "something else",
            "text_normalized_match": reproduced,
            "text_bytes_match": reproduced,
            "mass_leg_available": True,
            "lived_min_mass": 0.25,
            "fork_min_mass": 0.25,
            "min_mass_delta": 0.0,
        } if reachable else {},
        "arm1": {
            "served_answer": arm1_value,
            "served_verdict": {
                "correct": arm1_correct,
                "expected_hits": ["v"] if arm1_correct else [],
                "rejected_hits": [],
            },
            "mounted_ids": [0],
            "fit_receipt": {"fit_planned": [0]},
            "grounding_receipt": {"grounded": True},
        } if reachable else {},
    }


def test_g2_counts_reachable_and_reproduced_separately():
    rows = [
        _result("a"),
        _result("b", reachable=False),
        _result("c", reproduced=False),
    ]
    table = p2c1.g2_table(rows)
    assert table["probe_count"] == 3
    assert table["reachable_count"] == 2
    assert table["reproduced_count"] == 1
    assert [u["probe_id"] for u in table["unreachable"]] == ["b"]
    assert [n["probe_id"] for n in table["not_reproduced"]] == ["c"]
    assert table["gate_pass"] is False


def test_g2_passes_at_nine_of_ten_as_registered():
    rows = [_result(f"p{i}") for i in range(9)] + [
        _result("p9", reproduced=False)]
    assert p2c1.g2_table(rows)["gate_pass"] is True

    rows = [_result(f"p{i}") for i in range(8)] + [
        _result("p8", reproduced=False), _result("p9", reachable=False)]
    assert p2c1.g2_table(rows)["gate_pass"] is False


def test_change_class_covers_every_direction():
    assert p2c1.change_class(True, True) == "SAME"
    assert p2c1.change_class(False, False) == "SAME"
    assert p2c1.change_class(True, False) == "REGRESSION"
    assert p2c1.change_class(False, True) == "FIX"
    assert p2c1.change_class(True, None) == "EXCLUDED"
    assert p2c1.change_class(False, None) == "EXCLUDED"


def test_g3_pass_rule_is_zero_regressions_on_lived_correct_probes():
    clean = p2c1.g3_table([_result("a"), _result("b")])
    assert clean["regression_count"] == 0
    assert clean["gate_pass"] is True

    regressed = p2c1.g3_table(
        [_result("a"), _result("b", arm1_correct=False)])
    assert regressed["regression_count"] == 1
    assert regressed["regressions"] == ["b"]
    assert regressed["gate_pass"] is False


def test_g3_counts_a_fix_without_letting_it_offset_a_regression():
    table = p2c1.g3_table([
        _result("fixed", lived_correct=False, arm1_correct=True),
        _result("broken", lived_correct=True, arm1_correct=False),
    ])
    assert table["fix_count"] == 1
    assert table["regression_count"] == 1
    assert table["gate_pass"] is False


def test_g3_excludes_a_non_reproducing_probe_from_both_directions():
    """An excluded probe is neither a regression nor a silent pass."""
    table = p2c1.g3_table([
        _result("ok"),
        _result("skipped", reproduced=False, arm1_correct=False),
    ])
    assert table["arm1_measured_count"] == 1
    assert table["regression_count"] == 0
    assert table["gate_pass"] is True
    assert [e["probe_id"] for e in table["excluded_from_arm1"]] == ["skipped"]
    row = next(r for r in table["table"] if r["probe_id"] == "skipped")
    assert row["change"] == "EXCLUDED"
    assert row["arm1_correct"] is None
    assert row["arm1_value"] is None


def test_g3_names_why_a_probe_was_excluded():
    table = p2c1.g3_table([
        _result("no_repro", reproduced=False),
        _result("no_snapshot", reachable=False),
    ])
    reasons = {
        e["probe_id"]: e["reason"] for e in table["excluded_from_arm1"]}
    assert "did not reproduce" in reasons["no_repro"]
    assert "UNREACHABLE" in reasons["no_snapshot"]


def test_g3_denominator_is_the_achieved_count_never_a_padded_ten():
    table = p2c1.g3_table([
        _result("a"),
        _result("b", reproduced=False),
        _result("c", reachable=False),
    ])
    assert table["probe_count"] == 3
    assert table["arm1_measured_count"] == 1
    assert table["lived_correct_measured_count"] == 1
    assert "measured=1" in table["denominator_note"]


def test_a_lived_incorrect_probe_that_stays_incorrect_is_not_a_regression():
    """t33's shape: the PASS RULE is scoped to lived-CORRECT probes."""
    table = p2c1.g3_table(
        [_result("t33", lived_correct=False, arm1_correct=False)])
    row = table["table"][0]
    assert row["change"] == "SAME"
    assert table["regression_count"] == 0
    assert table["gate_pass"] is True
    assert table["lived_correct_measured_count"] == 0


def test_g3_carries_the_fit_and_grounding_receipts_onto_the_row():
    row = p2c1.g3_table([_result("a")])["table"][0]
    assert row["fit_receipt"] == {"fit_planned": [0]}
    assert row["grounding_receipt"] == {"grounded": True}
    assert row["arm1_mounted_ids"] == [0]
