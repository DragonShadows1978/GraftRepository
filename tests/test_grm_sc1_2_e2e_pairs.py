"""GRM-SC1.2 — CPU tests for the E2E session-resume driver's pure parts.

ORDER: ``orders/GRM_SC1_2_E2E_PAIRS_SESSION_RESUME.md`` (gate G1).

WHAT THESE PIN.  Everything the gate's MEANING rests on that does not need a
GPU: pair selection, the probe-turn node-count derivation (the one place a
plausible-looking shortcut silently drops the node the demand trip has to
fetch), the lineage recomputation, row matching, and the 10-pair table's
counting rules -- especially its floor rule, that rates are reported over the
ACHIEVED count and never padded back to ten.

Several tests assert against the FROZEN race artifacts directly.  That is
deliberate: the derivations here are only trustworthy because they reproduce
what the frozen evidence already records, so the evidence is the oracle.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_sc1_2_session as sc12  # noqa: E402


def _shard_session(shard: str) -> Path:
    return sc12.SHARDS / shard / "attempt_001" / "session"


@pytest.fixture(scope="module")
def shard_manifests() -> dict:
    return {
        shard: sc12.read_json(
            _shard_session(shard) / "repository" / "manifest.json")
        for shard in ("e2e-1", "e2e-2", "e2e-3", "e2e-4")
    }


@pytest.fixture(scope="module")
def source_manifest(shard_manifests) -> dict:
    return shard_manifests[sc12.SOURCE_SHARD]


@pytest.fixture(scope="module")
def instrumentation() -> list:
    return sc12.read_jsonl(
        _shard_session(sc12.INSTRUMENTATION_SHARD) / "instrumentation.jsonl")


@pytest.fixture(scope="module")
def node_counts(instrumentation, source_manifest) -> dict:
    return sc12.probe_node_counts(
        instrumentation, len(source_manifest["nodes"]))


@pytest.fixture(scope="module")
def pairs(node_counts) -> list:
    return sc12.e2e_pairs(node_counts=node_counts)


# --------------------------------------------------------------------------
# Pair selection
# --------------------------------------------------------------------------
def test_seven_e2e_pairs_are_selected_and_all_certified(pairs):
    assert [p["fixture_id"] for p in pairs] == list(sc12.E2E_FIXTURE_IDS)
    assert len(pairs) == 7
    assert all(p["source_family"] == "certified_34_turn" for p in pairs)


def test_every_pair_has_both_arms_in_the_frozen_rows():
    rows = sc12.race_rows()
    for fixture_id in sc12.E2E_FIXTURE_IDS:
        assert f"{fixture_id}:served" in rows
        assert f"{fixture_id}:planted_miss" in rows


def test_pair_selection_refuses_a_missing_arm(node_counts):
    rows = sc12.race_rows()
    del rows["e2e_t19_nova_key:planted_miss"]
    with pytest.raises(sc12.SC12Error, match="lack both arms"):
        sc12.e2e_pairs(rows, node_counts=node_counts)


def test_pair_selection_refuses_a_non_certified_family(node_counts):
    rows = sc12.race_rows()
    rows["e2e_t05_orion_pin:served"] = {
        **rows["e2e_t05_orion_pin:served"],
        "source_family": "standalone",
    }
    with pytest.raises(sc12.SC12Error, match="not a certified_34_turn"):
        sc12.e2e_pairs(rows, node_counts=node_counts)


def test_pair_selection_refuses_a_planted_miss_that_is_not_a_positive(
        node_counts):
    rows = sc12.race_rows()
    rows["e2e_t05_orion_pin:planted_miss"] = {
        **rows["e2e_t05_orion_pin:planted_miss"], "label": 0}
    with pytest.raises(sc12.SC12Error, match="not a positive"):
        sc12.e2e_pairs(rows, node_counts=node_counts)


def test_pair_selection_requires_node_counts_and_never_guesses_them():
    """The whole point: the count is EVIDENCE, not arithmetic on a fixture id."""
    with pytest.raises(sc12.SC12Error, match="NOT a node index"):
        sc12.e2e_pairs()


def test_pair_selection_refuses_a_plant_alias_outside_its_prefix(node_counts):
    counts = dict(node_counts)
    counts[31] = 5  # e2e_t30_atlas_tone's plant alias is node 30
    with pytest.raises(sc12.SC12Error, match="outside its reconstructed"):
        sc12.e2e_pairs(node_counts=counts)


# --------------------------------------------------------------------------
# Probe-turn node counts
# --------------------------------------------------------------------------
def test_probe_node_counts_match_the_hand_derived_values(node_counts):
    """Conversation turn -> nodes present when that probe routed.

    These are the values the frozen node-to-turn map yields, derived
    independently of the backward walk that produces them here.
    """
    assert node_counts == {
        6: 5, 10: 10, 14: 15, 17: 18, 20: 22, 23: 25, 25: 27, 27: 29, 31: 33}


def test_fixture_ids_map_to_the_expected_conversation_turns():
    assert sc12.fixture_conversation_turn("e2e_t05_orion_pin") == 6
    assert sc12.fixture_conversation_turn("e2e_t13_orion_pin") == 14
    assert sc12.fixture_conversation_turn("e2e_t30_atlas_tone") == 31


def test_fixture_id_number_is_NOT_the_node_index(pairs):
    """The trap this order nearly fell into.

    Reading the node count off the fixture id gives 13 for t13 against a true
    15, and 30 for t30 against a true 33 -- and 30 IS t30's plant alias, so
    the wrong rule would have excluded the very node the demand trip must
    fetch.
    """
    by_id = {p["fixture_id"]: p for p in pairs}
    assert by_id["e2e_t13_orion_pin"]["probe_node_index"] == 15
    assert by_id["e2e_t30_atlas_tone"]["probe_node_index"] == 33
    assert by_id["e2e_t30_atlas_tone"]["plant_alias_ids"] == [30]


def test_every_plant_alias_lies_inside_its_own_prefix(pairs):
    for pair in pairs:
        for alias in pair["plant_alias_ids"]:
            assert 0 <= alias < pair["probe_node_index"], pair["fixture_id"]


def test_every_lived_ranking_lies_inside_its_own_prefix(pairs):
    for pair in pairs:
        check = sc12.ranking_within_prefix(pair)
        assert check["within"], (pair["fixture_id"], check["outside_prefix"])


def test_probe_node_counts_fail_closed_when_a_ranking_escapes():
    rows = [
        {"kind": "fact", "info": {"nodes_before": 0, "nodes_after": 1}},
        {"kind": "probe", "info": {"ranking_ids": [7]}},
    ]
    with pytest.raises(sc12.SC12Error, match="outside a derived prefix"):
        sc12.probe_node_counts(rows, 2)


def test_probe_node_counts_walk_backward_not_forward():
    """An unanchored deposit turn is where a FORWARD fill goes wrong.

    Anchors: row 0 says 0 before and 2 after; row 3 says 4 before.  Rows 1
    and 2 each deposit one node, so the probe at row 2 saw 3 nodes -- not 2,
    which is what a forward count of one-per-turn would give.
    """
    rows = [
        {"kind": "fact", "info": {"nodes_before": 0, "nodes_after": 2}},
        {"kind": "filler", "info": {}},
        {"kind": "probe", "info": {"ranking_ids": [0, 1, 2]}},
        {"kind": "fact", "info": {"nodes_before": 4, "nodes_after": 5}},
    ]
    assert sc12.probe_node_counts(rows, 5) == {3: 3}


# --------------------------------------------------------------------------
# Lineage recomputation
# --------------------------------------------------------------------------
def test_lineage_derivation_reproduces_every_frozen_manifest(shard_manifests):
    """The oracle.

    Each frozen flush is a real observed lineage state at its own node count.
    Zero mismatches is what licenses using the rule on a prefix that no flush
    ever recorded.
    """
    check = sc12.lineage_selfcheck(shard_manifests)
    assert check["all_match"], check["shards"]
    assert [row["node_count"] for row in check["shards"]] == [11, 19, 28, 34]


def test_lineage_derivation_is_prefix_sensitive(source_manifest):
    """Node 0 is superseded by node 7.

    At a 5-node prefix (t05) node 7 does not exist yet, so node 0 must be
    ACTIVE -- which is exactly what the lived t05 probe required, and exactly
    what the end-of-shard flush (which says retired) would have got wrong.
    """
    nodes = source_manifest["nodes"]
    at_five = sc12.derive_lineage(nodes[:5])
    assert at_five[0] == {"superseded_by": [], "active": True, "retired": False}
    at_fifteen = sc12.derive_lineage(nodes[:15])
    assert at_fifteen[0]["superseded_by"] == [7]
    assert at_fifteen[0]["active"] is False
    assert at_fifteen[0]["retired"] is True


def test_lineage_fails_closed_on_a_forward_edge_outside_the_prefix():
    nodes = [{"metadata": {"supersedes": [9]}}]
    with pytest.raises(sc12.SC12Error, match="outside prefix"):
        sc12.derive_lineage(nodes)


def test_supersedes_is_deposit_final_across_every_frozen_shard(
        shard_manifests, source_manifest):
    """The property the whole reconstruction rests on."""
    for shard, manifest in shard_manifests.items():
        agreement = sc12.prefix_agreement(
            source_manifest["nodes"], manifest["nodes"])
        assert agreement["agree"], (shard, agreement["diffs"])


def test_prefix_agreement_reports_a_deposit_time_disagreement():
    left = [{"text": "a", "ntok": 1, "kind": "turn", "sources": [],
             "tags": [], "rare": [], "metadata": {"supersedes": []}}]
    right = [{**left[0], "ntok": 2}]
    agreement = sc12.prefix_agreement(left, right)
    assert not agreement["agree"]
    assert agreement["diffs"] == [{"node": 0, "field": "ntok"}]


# --------------------------------------------------------------------------
# Truncated manifest
# --------------------------------------------------------------------------
def test_truncated_manifest_keeps_the_prefix_and_rebuilds_the_back_edges(
        source_manifest):
    manifest = sc12.truncated_manifest(source_manifest, 5)
    assert len(manifest["nodes"]) == 5
    assert manifest["nodes"][0]["text"] == source_manifest["nodes"][0]["text"]
    assert manifest["nodes"][0]["ntok"] == source_manifest["nodes"][0]["ntok"]
    # The end-of-shard flush said retired; the probe-turn truth is not.
    assert source_manifest["nodes"][0]["retired"] is True
    assert manifest["nodes"][0]["retired"] is False
    assert manifest["nodes"][0]["metadata"]["active"] is True
    assert manifest["nodes"][0]["metadata"]["superseded_by"] == []


def test_truncated_manifest_drops_the_native_checkpoint(source_manifest):
    """A 34-node checkpoint under a 5-node manifest would be a silent lie."""
    assert source_manifest["native_checkpoint"]
    manifest = sc12.truncated_manifest(source_manifest, 5)
    assert manifest["native_checkpoint"] is None
    assert all("native_node_id" not in n for n in manifest["nodes"])


def test_truncated_manifest_refuses_a_prefix_longer_than_its_source(
        source_manifest):
    with pytest.raises(sc12.SC12Error, match="exceeds source manifest"):
        sc12.truncated_manifest(source_manifest, 99)


def test_truncated_manifest_applies_the_no_fold_pins(source_manifest):
    manifest = sc12.truncated_manifest(
        source_manifest, 10, no_fold_at_probe={1: True, 4: True})
    assert manifest["nodes"][1]["no_fold"] is True
    assert manifest["nodes"][4]["no_fold"] is True
    assert manifest["nodes"][0]["no_fold"] is False
    assert manifest["nodes"][1]["metadata"]["no_fold"] is True


def test_no_fold_pinning_brackets_and_flags_the_ambiguity():
    """Agreeing brackets -> pinned.  Straddling the 19/28 flushes -> not."""
    assert sc12.no_fold_at_prefix(5) == (frozenset(), True)
    assert sc12.no_fold_at_prefix(15) == (frozenset(), True)
    assert sc12.no_fold_at_prefix(22) == (frozenset(), False)
    assert sc12.no_fold_at_prefix(27) == (frozenset(), False)
    assert sc12.no_fold_at_prefix(29) == (frozenset({1, 4, 6, 8}), True)
    assert sc12.no_fold_at_prefix(33) == (frozenset({1, 4, 6, 8}), True)


def test_unpinned_no_fold_pairs_are_named(pairs):
    unpinned = sorted(
        p["fixture_id"] for p in pairs if not p["no_fold_pinned"])
    assert unpinned == [
        "e2e_t19_nova_key", "e2e_t22_mira_seal", "e2e_t24_terra_port"]


# --------------------------------------------------------------------------
# Reproduction comparator (Arm 0)
# --------------------------------------------------------------------------
def test_reproduction_needs_both_text_and_signal():
    good = sc12.reproduction_verdict(
        lived_answer="Harbor-8-Golf", fork_answer="Harbor-8-Golf",
        lived_min_mass=0.5, fork_min_mass=0.5, byte_exact_fork=True)
    assert good["reproduced"] is True

    text_only = sc12.reproduction_verdict(
        lived_answer="Harbor-8-Golf", fork_answer="Harbor-8-Golf",
        lived_min_mass=0.5, fork_min_mass=0.4, byte_exact_fork=True)
    assert text_only["text_normalized_match"] is True
    assert text_only["reproduced"] is False

    signal_only = sc12.reproduction_verdict(
        lived_answer="Harbor-8-Golf", fork_answer="Zenith-2-Echo",
        lived_min_mass=0.5, fork_min_mass=0.5, byte_exact_fork=True)
    assert signal_only["min_mass_exact_match"] is True
    assert signal_only["reproduced"] is False


def test_reproduction_uses_the_det1_semantic_comparator_for_text():
    """"Value comparison is semantics, not glyphs."

    U+2011 must not break a reproduction, and the byte leg is reported
    ALONGSIDE the semantic one, never instead of it.
    """
    verdict = sc12.reproduction_verdict(
        lived_answer="Quartz-8-Jade",
        fork_answer="Quartz‑8‑Jade",
        lived_min_mass=0.25, fork_min_mass=0.25, byte_exact_fork=True)
    assert verdict["text_normalized_match"] is True
    assert verdict["text_bytes_match"] is False
    assert verdict["reproduced"] is True


def test_reproduction_min_mass_uses_exact_equality_with_no_tolerance():
    verdict = sc12.reproduction_verdict(
        lived_answer="x", fork_answer="x",
        lived_min_mass=0.3670685812830925,
        fork_min_mass=0.3670685812830926,
        byte_exact_fork=True)
    assert verdict["min_mass_exact_match"] is False
    assert verdict["reproduced"] is False
    assert verdict["min_mass_delta"] != 0.0


def test_reproduction_records_the_delta_when_the_fork_is_not_byte_exact():
    verdict = sc12.reproduction_verdict(
        lived_answer="0.0", fork_answer="0.0",
        lived_min_mass=0.0, fork_min_mass=0.0, byte_exact_fork=False)
    assert verdict["reproduced"] is True
    assert "NOT byte-exact" in verdict["min_mass_comparison_basis"]


def test_reproduction_fails_closed_on_a_missing_signal():
    verdict = sc12.reproduction_verdict(
        lived_answer="x", fork_answer="x",
        lived_min_mass=None, fork_min_mass=0.5, byte_exact_fork=True)
    assert verdict["reproduced"] is False


def test_pair_reproduces_only_when_both_arms_do():
    both = {"served": {"arm0": {"reproduced": True}},
            "planted_miss": {"arm0": {"reproduced": True}}}
    assert sc12.pair_reproduced(both) is True
    for variant in ("served", "planted_miss"):
        broken = {k: {"arm0": {"reproduced": k != variant}} for k in both}
        assert sc12.pair_reproduced(broken) is False


# --------------------------------------------------------------------------
# Ten-pair table
# --------------------------------------------------------------------------
def _sc1_1_summary() -> dict:
    return sc12.read_json(sc12.SC1_1_G2_SUMMARY)


def _e2e_result(fixture_id, *, reproduced, fired, recovered, control_fired):
    return {
        "fixture_id": fixture_id,
        "arm0_reproduced": reproduced,
        "arms": {
            "served": {"arm1": {
                "demand_fired": control_fired,
                "attempt_min_mass": 0.4,
                "served_verdict": {"correct": True},
            }},
            "planted_miss": {"arm1": {
                "demand_fired": fired,
                "demand_token_index": 0 if fired else None,
                "attempt_min_mass": 0.0,
                "demand_trip": {"demand_fetched": [1]},
                "demand_served": "demand_trip" if recovered else "original",
                "served_answer": "value",
                "recovered": recovered,
            }},
        },
    }


def test_sc1_1_rows_carry_into_the_table_unchanged():
    summary = _sc1_1_summary()
    rows = sc12.sup_rows_from_sc1_1(summary)
    assert [r["pair"] for r in rows] == list(sc12.SUP_FIXTURE_IDS)
    assert all(r["instrument"] == "same_process_fork" for r in rows)
    # SC1.1's own numbers, carried not recomputed.
    assert summary["measured_positives"] == 2
    assert summary["measured_negatives"] == 3
    assert sum(1 for r in rows if r["planted_measured"]) == 2
    assert sum(1 for r in rows if r["recovered"]) == 2


def test_sc1_1_unmeasurable_arm_is_excluded_not_counted_as_a_miss():
    rows = sc12.sup_rows_from_sc1_1(_sc1_1_summary())
    solace = next(r for r in rows if r["pair"] == "sup_solace_fresh")
    assert solace["planted_measured"] is False
    assert solace["fired"] is None
    table = sc12.ten_pair_table([], _sc1_1_summary())
    assert table["measured_positives"] == 2
    assert any(e["pair"] == "sup_solace_fresh"
               for e in table["excluded_from_arm1"])


def test_ten_pair_table_has_ten_rows_when_all_seven_are_measured():
    results = [
        _e2e_result(f, reproduced=True, fired=True, recovered=True,
                    control_fired=False)
        for f in sc12.E2E_FIXTURE_IDS
    ]
    table = sc12.ten_pair_table(results, _sc1_1_summary())
    assert table["rows_present"] == 10
    assert table["registered_pair_count"] == 10
    assert table["measured_positives"] == 9  # 7 E2E + 2 SC1.1
    assert table["detection_recall"] == 1.0
    assert table["recovered"] == 9


def test_an_unreproduced_pair_is_excluded_from_numerator_and_denominator():
    results = [
        _e2e_result(f, reproduced=(f != "e2e_t19_nova_key"), fired=True,
                    recovered=True, control_fired=False)
        for f in sc12.E2E_FIXTURE_IDS
    ]
    table = sc12.ten_pair_table(results, _sc1_1_summary())
    assert table["rows_present"] == 10
    # 6 reproduced E2E + 2 SC1.1 -- NOT 9, and NOT padded back to 10.
    assert table["measured_positives"] == 8
    assert table["detection_recall"] == 1.0
    excluded = {e["pair"] for e in table["excluded_from_arm1"]}
    assert "e2e_t19_nova_key" in excluded
    row = next(r for r in table["table"] if r["pair"] == "e2e_t19_nova_key")
    assert row["fired"] is None and row["recovered"] is None


def test_the_floor_rule_never_pads_the_denominator_to_ten():
    table = sc12.ten_pair_table([], _sc1_1_summary())
    assert table["measured_positives"] == 2
    assert table["rows_present"] == 3
    assert "ACHIEVED count" in table["denominator_note"]
    assert table["registered_pair_count"] == 10


def test_pass_rule_is_recall_ge_0_9_and_fp_le_1():
    # One missed detection out of nine positives -> recall 8/9 < 0.9 -> FAIL.
    results = [
        _e2e_result(f, reproduced=True,
                    fired=(f != "e2e_t05_orion_pin"),
                    recovered=(f != "e2e_t05_orion_pin"),
                    control_fired=False)
        for f in sc12.E2E_FIXTURE_IDS
    ]
    table = sc12.ten_pair_table(results, _sc1_1_summary())
    assert table["measured_positives"] == 9
    assert table["detection_recall"] == pytest.approx(8 / 9)
    assert table["gate_pass"] is False

    # Two false fires -> FAIL even at perfect recall.
    results = [
        _e2e_result(f, reproduced=True, fired=True, recovered=True,
                    control_fired=(f in sc12.E2E_FIXTURE_IDS[:2]))
        for f in sc12.E2E_FIXTURE_IDS
    ]
    table = sc12.ten_pair_table(results, _sc1_1_summary())
    assert table["detection_recall"] == 1.0
    # 2 E2E false fires + SC1.1's own 1 (sup_solace_fresh) = 3.
    assert table["false_fires_on_served_controls"] == 3
    assert table["gate_pass"] is False


def test_recovery_is_reported_never_gated():
    results = [
        _e2e_result(f, reproduced=True, fired=True, recovered=False,
                    control_fired=False)
        for f in sc12.E2E_FIXTURE_IDS
    ]
    table = sc12.ten_pair_table(results, _sc1_1_summary())
    assert table["detection_recall"] == 1.0
    assert table["false_fires_on_served_controls"] == 1  # SC1.1's own
    assert table["recovered"] == 2  # only SC1.1's two
    # Zero E2E recovery does NOT fail the mechanism gate.
    assert table["gate_pass"] is True
    assert table["recovery_is_reported_not_gated"] is True


def test_table_names_the_two_instruments_rather_than_blending_them():
    results = [
        _e2e_result(f, reproduced=True, fired=True, recovered=True,
                    control_fired=False)
        for f in sc12.E2E_FIXTURE_IDS
    ]
    table = sc12.ten_pair_table(results, _sc1_1_summary())
    instruments = {r["instrument"] for r in table["table"]}
    assert instruments == {
        "same_process_fork", "cross_process_fork_from_frozen_snapshot"}
    assert "NOT the same instrument" in table["instrument_note"]


# --------------------------------------------------------------------------
# Registration + frozen-tree hygiene
# --------------------------------------------------------------------------
def test_registration_exists_and_carries_the_race_threshold():
    registration = sc12.read_json(sc12.REGISTRATION)
    assert registration["written_before_any_gate_run"] is True
    carried = registration["carried_threshold"]
    assert carried["carried_not_refit"] is True
    assert carried["value"] == 0.3380523274342219
    assert carried["fit_turn_count"] == 2
    # The same number the production config carries -- never refit here.
    config = sc12.read_json(ROOT / "config" / "grm_demand_registered.json")
    assert config["threshold"] == carried["value"]


def test_registration_names_the_cross_process_weakening():
    registration = sc12.read_json(sc12.REGISTRATION)
    fork = registration["cross_process_fork_REGISTERED_BEFORE_THE_RUN"]
    assert fork["require_same_process_index"] is False
    assert "NOT_VERIFIED" in fork["what_is_therefore_NOT_verified"]


def test_driver_declares_no_writes_under_the_frozen_run():
    """A cheap structural guard against a frozen-tree write.

    Any write verb applied on the same line as a frozen-run path is a
    failure; the driver must only ever read there.
    """
    source = (ROOT / "scripts" / "grm_sc1_2_e2e_recovery_gpu.py").read_text(
        encoding="utf-8")
    banned = ("write_text(", "write_bytes(", "mkdir(", "unlink(", "rmtree(")
    for line in source.splitlines():
        if not any(verb in line for verb in banned):
            continue
        if any(token in line
               for token in ("FROZEN_RUN", "SHARDS", "_shard_session")):
            raise AssertionError(f"frozen-tree write risk: {line.strip()}")


def test_snapshot_manifests_for_all_seven_pairs_exist_and_are_lived(pairs):
    for pair in pairs:
        manifest_path = ROOT / pair["snapshot_manifest"]
        assert manifest_path.is_file(), pair["fixture_id"]
        snapshot = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert snapshot["label"] == "lived"
        assert snapshot["phase"] == "before_probe_prefill"
        assert snapshot["capture_finalized"] is True
        assert snapshot["schema"] == "grm.det1_3.model_visible_snapshot.v2"


def test_lived_rows_match_the_snapshot_the_pair_forks_from(pairs):
    """The lived answer to reproduce is the one linked to THAT snapshot.

    Not a same-named row from somewhere else in the campaign.
    """
    for pair in pairs:
        snapshot = json.loads(
            (ROOT / pair["snapshot_manifest"]).read_text(encoding="utf-8"))
        linked = snapshot["linked_answer"]
        assert linked["probe_answer"] == pair["lived"]["served_answer"], (
            pair["fixture_id"])
        assert sorted(int(v) for v in linked["probe_mounts"]) == sorted(
            pair["lived"]["served_mounted_ids"]), pair["fixture_id"]


def test_sc1_recovery_summarize10_delegates_and_does_not_duplicate_the_rule():
    """The one permitted edit to ``grm_sc1_recovery_gpu`` is a DELEGATION.

    If that module ever grows its own copy of the counting rule, the two
    implementations can drift and the 10-pair number stops being one number.
    """
    from scripts import grm_sc1_recovery_gpu as sc1

    source = (ROOT / "scripts" / "grm_sc1_recovery_gpu.py").read_text(
        encoding="utf-8")
    assert "ten_pair_table" in source
    assert "grm_sc1_2_session" in source
    assert hasattr(sc1, "summarize_ten_pairs")
    # summarize10 must not re-derive recall / FP / recovery itself.
    body = source.split("def summarize_ten_pairs")[1].split("\ndef ")[0]
    for reinvented in ("detection_recall =", "false_fires", "recovered ="):
        assert reinvented not in body, reinvented


def test_summarize10_receipts_route_to_the_sc1_2_artifact_dir():
    """SC1.2 receipts must not land among SC1's frozen evidence."""
    source = (ROOT / "scripts" / "grm_sc1_recovery_gpu.py").read_text(
        encoding="utf-8")
    assert 'stem.startswith("sc1_2_")' in source
    assert '"artifacts" / "grm_sc1_2"' in source


def test_planted_miss_arms_are_all_single_mount_withholdings(pairs):
    """Why a planted-miss min mass of exactly 0.0 is the EXPECTED reproduction.

    Every one of the seven mounts exactly one node, and the plant alias IS
    that node, so the withholding empties the mount set entirely.
    """
    for pair in pairs:
        assert pair["lived"]["served_mounted_ids"] == pair["plant_alias_ids"]
        assert pair["lived"]["planted_mounted_ids"] == []
        assert pair["lived"]["planted_min_mass"] == 0.0
