"""GRM-RS2 — CPU tests for the mount-read-deficit probe's pure parts.

ORDER: ``orders/GRM_RS2_MOUNT_READ_DEFICIT.md`` (gate G1: "new CPU tests for
the probe script's pure parts (arm plan, capture-text derivation, per-layer-type
partition arithmetic sums to one, table assembly)").

Every test here is CPU-only.  Nothing loads a model, nothing takes the GPU
lease, and nothing writes into ``artifacts/``.  The four things pinned are
exactly the four G1 names:

  * ARM PLAN.  Every registered probe appears once per arm, the arms are the
    registered set, the plan carries the probe's ids forward unchanged, and the
    solace probe gets the extra ``solace_fact`` variant of EVERY arm that the
    order's "ALSO run every arm with the solace FACT node mounted" requires.

  * CAPTURE-TEXT DERIVATION.  The registered capture text must be the text the
    LIVED INSTALLER captured from, so these tests re-derive it from the fixture
    through the same two frozen functions the installer uses and require
    equality — a registration built against an edited fixture fails here rather
    than silently steering a GPU arm.  The B2 scaffold text is required to
    DIFFER from it (otherwise B2 is B1 and the arm measures nothing), and a
    fit-time split child is required to be recorded as having no fixture text
    rather than being handed another node's.

  * PER-LAYER-TYPE PARTITION ARITHMETIC SUMS TO ONE.  ``partition_sum_ok``
    accepts a partition inside the tolerance ``grm_demand._summarize_mass``
    itself enforces, rejects one outside it, and rejects an INCOMPLETE
    partition instead of treating a missing band as zero.
    ``_summarize_layer_type`` is checked to average layers then positions, to
    carry the partition law through to its own output, and to refuse a capture
    whose layer count drifted between answer positions.  ``_window_reach`` is
    checked to report a STRUCTURAL zero (the window cannot reach the mount
    band) distinguishably from a measured near-zero, which is exactly what the
    registered partition definition promises.

  * TABLE ASSEMBLY.  The G3 column set is fixed, both layer types appear as
    their own columns, the ordering is probe-then-variant-then-arm in the
    registered arm order, and a row with no mass block still produces a row
    (with empty mass cells) rather than vanishing.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_rs2_mount_read_gpu as rs2  # noqa: E402
from scripts import grm_rs2_registration as reg  # noqa: E402

REGISTRATION = ROOT / "artifacts" / "grm_rs2" / "registration.json"
RS1_REGISTRATION = ROOT / "artifacts" / "grm_rs1" / "registration.json"

pytestmark = pytest.mark.skipif(
    not REGISTRATION.is_file(),
    reason="RS2 registration not written yet (scripts/grm_rs2_registration.py)",
)


@pytest.fixture(scope="module")
def registration() -> dict:
    return rs2.read_registration(REGISTRATION)


def _bands(mounted: float, live: float, sink: float, learned: float) -> dict:
    return {
        "physical_sink_mass": float(sink),
        "mounted_mass": float(mounted),
        "live_mass": float(live),
        "learned_sink_mass": float(learned),
    }


# ----------------------------------------------------------------------
# arm plan
# ----------------------------------------------------------------------
def test_arm_plan_covers_every_registered_probe_once_per_arm(registration):
    for session_id in rs2.sessions_in_play(registration):
        probes = rs2.probes_for_session(registration, session_id)
        plan = rs2.arm_plan(registration, session_id)
        assert {str(item["arm"]) for item in plan} == set(rs2.ARMS)
        for arm in rs2.ARMS:
            registered = [
                item for item in plan
                if item["arm"] == arm and item["variant"] == "registered"]
            assert [str(item["probe_id"]) for item in registered] == probes


def test_arm_plan_carries_registered_ids_forward_unchanged(registration):
    rows = rs2.registered_probes(registration)
    for session_id in rs2.sessions_in_play(registration):
        for item in rs2.arm_plan(registration, session_id):
            if item["variant"] != "registered":
                continue
            row = rows[str(item["probe_id"])]
            assert item["mount_ids"] == [int(v) for v in row["a0_mounted_ids"]]
            assert item["live_ids"] == [int(v) for v in row["live_ids"]]
            assert item["live_node_ids"] == [
                str(v) for v in row["live_node_ids"]]
            assert item["role"] == str(row["role"])


def test_arm_plan_adds_the_solace_fact_variant_of_every_arm(registration):
    variant = registration["solace_fact_variant_REGISTERED_BEFORE_ANY_GATE"]
    probe_id = str(variant["probe_id"])
    plan = rs2.arm_plan(registration, str(variant["session_id"]))
    extra = [item for item in plan if item["variant"] == "solace_fact"]
    # Every arm, exactly once, and only for the solace probe.
    assert [str(item["arm"]) for item in extra] == list(rs2.ARMS)
    assert {str(item["probe_id"]) for item in extra} == {probe_id}
    # And it mounts the registered FACT node, not RS1's split child.
    assert {tuple(item["mount_ids"]) for item in extra} == {
        (int(variant["graft_id"]),)}
    assert int(variant["graft_id"]) not in {
        int(v) for v in variant["replaces_a0_mounted_ids"]}


def test_arm_plan_rejects_a_fixture_with_no_registered_probe(registration):
    with pytest.raises(reg.RS2Error):
        rs2.arm_plan(registration, "not_a_fixture")


def test_sessions_in_play_are_deduplicated_and_complete(registration):
    sessions = rs2.sessions_in_play(registration)
    assert len(sessions) == len(set(sessions))
    probes = rs2.registered_probes(registration)
    assert set(sessions) == {row["session_id"] for row in probes.values()}


# ----------------------------------------------------------------------
# capture-text derivation
# ----------------------------------------------------------------------
def test_registered_capture_text_is_the_lived_installers_own_text(registration):
    """Re-derived from the fixture through the installer's own two functions."""
    for row in rs2.registered_probes(registration).values():
        texts = reg.fixture_nodes(str(row["session_id"]))
        for node in row["capture_nodes"]:
            if not node.get("is_fixture_node"):
                continue
            node_id = str(node["node_id"])
            assert node["fixture_text"] == texts[node_id]
            assert node["capture_text_B0_B1"] == reg.capture_text_for_node(
                texts[node_id])
            assert node["capture_text_B2_scaffold"] == (
                reg.question_scaffold_text(texts[node_id]))


def test_b2_scaffold_text_actually_differs_from_the_installed_capture(
    registration,
):
    """If it did not, B2 would be B1 and the H-SCAFFOLD arm would measure
    nothing.  The registration records the comparison; this pins it."""
    seen = 0
    for row in rs2.registered_probes(registration).values():
        for node in row["capture_nodes"]:
            if not node.get("is_fixture_node"):
                continue
            seen += 1
            assert node["capture_texts_identical"] is False
            assert (node["capture_text_B2_scaffold"]
                    != node["capture_text_B0_B1"])
    assert seen, "no fixture-node capture text was registered at all"


def test_a_fit_split_child_is_recorded_as_having_no_fixture_text(registration):
    """RS1's wrong-graft finding: some registered mount ids are fit-time split
    CHILDREN.  They must be recorded as such, never handed another node's
    text."""
    children = [
        node
        for row in rs2.registered_probes(registration).values()
        for node in row["capture_nodes"]
        if not node.get("is_fixture_node")
    ]
    assert children, "the battery is expected to carry at least one split child"
    for node in children:
        assert node["node_id"] is None
        assert "capture_text_B0_B1" not in node
        assert "split" in str(node["note"]).lower()


def test_capture_text_for_selects_the_registered_text_by_arm(registration):
    for probe_id, row in rs2.registered_probes(registration).items():
        for node in row["capture_nodes"]:
            graft_id = int(node["graft_id"])
            if not node.get("is_fixture_node"):
                # A split child has no registered capture text; the arm falls
                # back to the graft's own text, which only the GPU run sees.
                assert rs2.capture_text_for(
                    registration, probe_id, graft_id, "B1") is None
                continue
            for arm, key in (
                ("B1", "capture_text_B0_B1"),
                ("B1b", "capture_text_B0_B1"),
                ("B2", "capture_text_B2_scaffold"),
            ):
                picked = rs2.capture_text_for(
                    registration, probe_id, graft_id, arm)
                assert picked is not None
                assert picked["text"] == node[key]
                assert picked["node_id"] == node["node_id"]


def test_solace_fact_variant_capture_text_is_the_fact_nodes(registration):
    variant = registration["solace_fact_variant_REGISTERED_BEFORE_ANY_GATE"]
    texts = reg.fixture_nodes(str(variant["session_id"]))
    node_id = str(variant["node_id"])
    assert variant["capture_text"] == reg.capture_text_for_node(texts[node_id])
    assert variant["scaffold_text"] == reg.question_scaffold_text(
        texts[node_id])
    picked = rs2.capture_text_for(
        registration, str(variant["probe_id"]), int(variant["graft_id"]), "B1")
    assert picked is not None
    assert picked["text"] == variant["capture_text"]
    scaffold = rs2.capture_text_for(
        registration, str(variant["probe_id"]), int(variant["graft_id"]), "B2")
    assert scaffold is not None
    assert scaffold["text"] == variant["scaffold_text"]


def test_registered_ids_are_the_ids_rs1_measured(registration):
    """Nothing in RS2's registration may drift from RS1's receipts."""
    rs1_rows = json.loads(RS1_REGISTRATION.read_text(encoding="utf-8"))[
        "probes_REGISTERED_BEFORE_ANY_GATE"]
    for probe_id, row in rs2.registered_probes(registration).items():
        source = rs1_rows[probe_id]
        assert row["a0_mounted_ids"] == [
            int(v) for v in source["a0_mounted_ids"]]
        assert row["live_ids"] == [int(v) for v in source["live_ids"]]
        assert row["session_id"] == str(source["session_id"])
        assert row["role"] == str(source["role"])
        assert row["rs1_A0"]["mounted_ids"] == [
            int(v) for v in source["a0_mounted_ids"]]


# ----------------------------------------------------------------------
# partition arithmetic sums to one
# ----------------------------------------------------------------------
def test_partition_sum_ok_accepts_inside_the_production_tolerance():
    assert rs2.partition_sum_ok(_bands(0.18, 0.43, 0.06, 0.33)) is True
    assert rs2.partition_sum_ok(_bands(0.0, 0.0, 0.0, 1.0)) is True
    # The tolerance is grm_demand's own [0.98, 1.02], edges included.
    assert rs2.partition_sum_ok(_bands(0.48, 0.25, 0.15, 0.10)) is True


def test_partition_sum_ok_rejects_outside_the_tolerance():
    assert rs2.partition_sum_ok(_bands(0.10, 0.20, 0.05, 0.10)) is False
    assert rs2.partition_sum_ok(_bands(0.50, 0.50, 0.50, 0.50)) is False


def test_partition_sum_ok_rejects_an_incomplete_partition():
    """A missing band is not a zero band — treating it as one would turn a
    capture failure into a plausible-looking measurement."""
    partial = _bands(0.18, 0.43, 0.06, 0.33)
    for name in rs2.MASS_NAMES:
        broken = dict(partial)
        broken[name] = None
        assert rs2.partition_sum_ok(broken) is False
        missing = {k: v for k, v in partial.items() if k != name}
        assert rs2.partition_sum_ok(missing) is False


def test_summarize_layer_type_averages_layers_then_positions():
    """Two answer positions, two layers, hand-computable means."""
    steps = [
        [_bands(0.2, 0.4, 0.1, 0.3), _bands(0.4, 0.2, 0.1, 0.3)],
        [_bands(0.6, 0.0, 0.1, 0.3), _bands(0.0, 0.6, 0.1, 0.3)],
    ]
    out = rs2._summarize_layer_type(rs2.FULL, steps)
    assert out["layers_observed"] == 2
    assert out["answer_positions"] == 2
    mean = out["mean_over_answer_positions"]
    # position means: (0.2+0.4)/2 = 0.3 and (0.6+0.0)/2 = 0.3 -> 0.3
    assert mean["mounted_mass"] == pytest.approx(0.3)
    assert mean["live_mass"] == pytest.approx(0.3)
    assert mean["physical_sink_mass"] == pytest.approx(0.1)
    assert mean["learned_sink_mass"] == pytest.approx(0.3)
    assert out["partition_sums_to_one"] is True
    assert out["partition_sum"] == pytest.approx(1.0)
    # per-layer means: layer 0 mounted (0.2+0.6)/2 = 0.4; layer 1 (0.4+0.0)/2
    per_layer = {row["layer_ordinal"]: row for row in out["per_layer_mean"]}
    assert per_layer[0]["mounted_mass"] == pytest.approx(0.4)
    assert per_layer[1]["mounted_mass"] == pytest.approx(0.2)
    assert out["top_contributing_layer"]["layer_ordinal"] == 0
    assert out["top_contributing_layer"]["mounted_mass"] == pytest.approx(0.4)
    assert all(row["layer_type"] == rs2.FULL for row in out["per_layer_mean"])


def test_summarize_layer_type_reports_a_broken_partition_rather_than_hiding_it():
    out = rs2._summarize_layer_type(rs2.SLIDING, [[_bands(0.1, 0.1, 0.1, 0.1)]])
    assert out["partition_sums_to_one"] is False
    assert out["partition_sum"] == pytest.approx(0.4)


def test_summarize_layer_type_refuses_a_drifting_layer_count():
    steps = [
        [_bands(0.2, 0.4, 0.1, 0.3), _bands(0.4, 0.2, 0.1, 0.3)],
        [_bands(0.2, 0.4, 0.1, 0.3)],
    ]
    with pytest.raises(reg.RS2Error):
        rs2._summarize_layer_type(rs2.SLIDING, steps)


def test_summarize_layer_type_on_no_capture_reports_none_not_zero():
    out = rs2._summarize_layer_type(rs2.SLIDING, [])
    assert out["layers_observed"] == 0
    assert out["answer_positions"] == 0
    assert out["top_contributing_layer"] is None
    assert out["partition_sums_to_one"] is None
    assert all(
        value is None for value in out["mean_over_answer_positions"].values())


# ----------------------------------------------------------------------
# the sliding window's reach (structural zero vs measured near-zero)
# ----------------------------------------------------------------------
def _sliding_row(*, s: int, k0: int, mount_lo: int, mount_hi: int,
                 window: int = 128) -> dict:
    mount_rows = max(0, min(mount_hi, s) - max(mount_lo, k0))
    return {
        "physical_sink_mass": 0.05,
        "mounted_mass": 0.2 if mount_rows else 0.0,
        "live_mass": 0.6,
        "learned_sink_mass": 0.15,
        "S": int(s),
        "q_abs": int(s - 1),
        "k0": int(k0),
        "k1": int(s),
        "sliding_window": int(window),
        "mount_band": [int(mount_lo), int(mount_hi)],
        "mount_rows_in_window": int(mount_rows),
        "window_reaches_mount": bool(mount_rows > 0),
    }


def test_window_reach_flags_a_structural_zero():
    """The window never reaches the mount band: the zero is STRUCTURAL and the
    reach block must say so, so it is not read as a measured near-zero."""
    steps = [
        [_sliding_row(s=300, k0=173, mount_lo=19, mount_hi=85)],
        [_sliding_row(s=301, k0=174, mount_lo=19, mount_hi=85)],
    ]
    out = rs2._window_reach(steps)
    assert out["observed_rows"] == 2
    assert out["never_reaches_mount"] is True
    assert out["always_reaches_mount"] is False
    assert out["reaching_rows"] == 0
    assert out["mount_rows_in_window_max"] == 0
    assert out["fraction_of_mount_band_visible_max"] == pytest.approx(0.0)


def test_window_reach_reports_full_and_partial_visibility():
    steps = [
        [_sliding_row(s=141, k0=13, mount_lo=19, mount_hi=85)],   # fully seen
        [_sliding_row(s=200, k0=72, mount_lo=19, mount_hi=85)],   # partly seen
    ]
    out = rs2._window_reach(steps)
    assert out["always_reaches_mount"] is True
    assert out["never_reaches_mount"] is False
    assert out["reaching_rows"] == 2
    assert out["mount_rows_in_window_max"] == 66     # the whole 19..85 band
    assert out["mount_rows_in_window_min"] == 13     # rows 72..85
    assert out["fraction_of_mount_band_visible_max"] == pytest.approx(1.0)
    assert out["fraction_of_mount_band_visible_min"] == pytest.approx(13 / 66)
    assert out["S_min"] == 141
    assert out["S_max"] == 200


def test_window_reach_on_no_capture_is_none_not_false():
    out = rs2._window_reach([])
    assert out["observed_rows"] == 0
    assert out["always_reaches_mount"] is None
    assert out["never_reaches_mount"] is None


# ----------------------------------------------------------------------
# table assembly
# ----------------------------------------------------------------------
def _measured_row(probe_id: str, arm: str, *, variant: str = "registered",
                  correct: bool = False, mounted_full: float = 0.18,
                  mounted_sliding: float = 0.05) -> dict:
    return {
        "probe_id": probe_id,
        "arm": arm,
        "variant": variant,
        "served_answer": f"{arm} answer",
        "correct": correct,
        "mounted_ids": [1],
        "mass": {
            "by_layer_type": {
                rs2.FULL: {
                    "mean_over_answer_positions": _bands(
                        mounted_full, 0.43, 0.06, 1.0 - mounted_full - 0.49),
                    "top_contributing_layer": {
                        "layer_ordinal": 5, "mounted_mass": 0.31},
                },
                rs2.SLIDING: {
                    "mean_over_answer_positions": _bands(
                        mounted_sliding, 0.70, 0.10,
                        1.0 - mounted_sliding - 0.80),
                    "top_contributing_layer": {
                        "layer_ordinal": 2, "mounted_mass": 0.09},
                },
            },
            "sliding_window_reach": {"always_reaches_mount": True},
        },
    }


def test_assemble_table_column_set_is_fixed():
    table = rs2.assemble_table([_measured_row("p", "B0")])
    assert set(table[0]) == {
        "probe_id", "variant", "arm", "served", "correct", "mounted_ids",
        "mounted_mass_full_layers", "mounted_mass_sliding_layers",
        "live_mass_full_layers", "live_mass_sliding_layers",
        "sink_mass_full_layers", "sink_mass_sliding_layers",
        "learned_sink_mass_full_layers",
        "top_layer_full", "top_layer_full_mounted_mass",
        "top_layer_sliding", "top_layer_sliding_mounted_mass",
        "sliding_window_reaches_mount",
    }


def test_assemble_table_carries_both_layer_types_separately():
    row = rs2.assemble_table([
        _measured_row("p", "B0", mounted_full=0.18, mounted_sliding=0.05)])[0]
    assert row["mounted_mass_full_layers"] == pytest.approx(0.18)
    assert row["mounted_mass_sliding_layers"] == pytest.approx(0.05)
    assert row["live_mass_full_layers"] == pytest.approx(0.43)
    assert row["live_mass_sliding_layers"] == pytest.approx(0.70)
    assert row["top_layer_full"] == 5
    assert row["top_layer_sliding"] == 2
    assert row["sliding_window_reaches_mount"] is True


def test_assemble_table_orders_probe_then_variant_then_registered_arm_order():
    rows = [
        _measured_row("b_probe", "B5"),
        _measured_row("a_probe", "B5"),
        _measured_row("a_probe", "B0"),
        _measured_row("a_probe", "B0", variant="solace_fact"),
        _measured_row("a_probe", "B3a"),
    ]
    table = rs2.assemble_table(rows)
    assert [(r["probe_id"], r["variant"], r["arm"]) for r in table] == [
        ("a_probe", "registered", "B0"),
        ("a_probe", "registered", "B3a"),
        ("a_probe", "registered", "B5"),
        ("a_probe", "solace_fact", "B0"),
        ("b_probe", "registered", "B5"),
    ]


def test_assemble_table_keeps_a_row_with_no_mass_block():
    """A serve that captured nothing must still appear — a vanished row would
    read as an arm that was never run."""
    table = rs2.assemble_table([{
        "probe_id": "p", "arm": "B0", "variant": "registered",
        "served_answer": "", "correct": False, "mounted_ids": [],
    }])
    assert len(table) == 1
    assert table[0]["mounted_mass_full_layers"] is None
    assert table[0]["mounted_mass_sliding_layers"] is None
    assert table[0]["sliding_window_reaches_mount"] is None


# ----------------------------------------------------------------------
# the registration's own invariants
# ----------------------------------------------------------------------
def test_registration_declares_the_sliding_partition_before_any_gate(
    registration,
):
    block = registration["sliding_partition_REGISTERED_BEFORE_ANY_GATE"]
    for key in (
        "cache_row_layout", "full_layer_partition", "sliding_layer_partition",
        "window_indexing_law", "mounted_band_when_unreachable",
        "partition_law", "layer_type_source", "reported_statistic",
    ):
        assert isinstance(block[key], str) and block[key].strip()


def test_registration_vocabulary_is_the_orders_three_verdicts(registration):
    assert set(registration["vocabulary_REGISTERED_BEFORE_ANY_GATE"]) == {
        "SUPPORTED", "PARTIAL", "NOT DETECTED"}


def test_registration_names_every_arm_the_harness_serves(registration):
    declared = registration["arms_REGISTERED_BEFORE_ANY_GATE"]
    assert set(rs2.ARMS) <= set(declared)
    # B4 is registered as a measurement partition, not a serve, and must NOT
    # be in the serving tuple.
    assert "B4" in declared
    assert "B4" not in rs2.ARMS
    assert registration["arm_B4_is_a_measurement_partition_not_a_serve"] is True
    assert list(registration["arm_order"]) == list(rs2.ARMS)


def test_registration_records_the_structural_capture_finding(registration):
    block = registration["structural_finding_REGISTERED_BEFORE_ANY_GATE"]
    assert "deposit" in block["what"]
    assert "ephemeral" in block["what"]
    assert block["read_from"]


def test_registration_carries_every_lead_prediction(registration):
    predictions = registration["predictions_REGISTERED_BEFORE_ANY_GATE"]
    assert {row["hypothesis"] for row in predictions} == {
        "H-CAPTURE", "H-SCAFFOLD", "H-BAND-POSITION", "H-SLIDING"}
    for row in predictions:
        assert row["prediction"].strip()
        assert set(row["arms"]) <= set(rs2.ARMS) | {"B4"}
    # Every one of the ORDER's four hypotheses carries a LEAD prediction.
    lead = [
        row for row in predictions
        if row["source"] == "lead, stated in the order"]
    assert {row["hypothesis"] for row in lead} == {
        "H-CAPTURE", "H-SCAFFOLD", "H-BAND-POSITION", "H-SLIDING"}
    # A seat-added prediction must SAY it is the seat's, so it is never scored
    # as if the lead had made it.
    for row in predictions:
        if row["source"] == "lead, stated in the order":
            continue
        assert row["source"].startswith("SEAT")
        assert "not a lead prediction" in row["source"].casefold()


def test_the_b1p_amendment_is_an_addition_and_is_declared(registration):
    """An arm added mid-run must leave a trail, not appear silently."""
    amended = registration["amended_by"]
    assert amended["adds"] == ["arms_REGISTERED_BEFORE_ANY_GATE.B1p"]
    assert "ADDITION ONLY" in amended["why"]
    path = ROOT / str(amended["path"])
    if not path.is_file():
        pytest.skip("the RS2 amendment has not been written yet")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["amends_field"] == amended["adds"]
    assert payload["amends"]["sha256_before_amendment"]
    # B1 must be kept EXACTLY as registered, so B1/B1p vary one thing.
    assert payload["new_arm"]["B1p"]["B1_unchanged"]
    assert payload["new_arm"]["B1p"][
        "prediction_is_the_seats_not_the_leads"] is True


def test_b1_and_b1p_share_one_capture_text(registration):
    """B1/B1p differ in capture POSITION only; a text difference would make the
    pair a two-variable contrast and the verdict unreadable."""
    for probe_id, row in rs2.registered_probes(registration).items():
        for node in row["capture_nodes"]:
            if not node.get("is_fixture_node"):
                continue
            graft_id = int(node["graft_id"])
            b1 = rs2.capture_text_for(registration, probe_id, graft_id, "B1")
            b1p = rs2.capture_text_for(registration, probe_id, graft_id, "B1p")
            assert b1 is not None and b1p is not None
            assert b1["text"] == b1p["text"]
