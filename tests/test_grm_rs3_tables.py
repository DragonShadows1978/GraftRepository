"""GRM-RS3 — CPU tests for the harness's pure parts.

ORDER: ``orders/GRM_RS3_CAPTURE_PIN_SEAT_NEAR_LIVE.md`` (gate G1).

CPU only: nothing loads a model, nothing takes the GPU lease, nothing writes
into ``artifacts/``.  These pin the shaping and scoring code that turns arm
receipts into the gate tables — the parts that decide what the report SAYS,
which is exactly where a silent error would be most expensive.

WHAT IS PINNED

  * ARM PLAN.  Every registered probe appears once per arm, and the solace
    probe additionally gets the ``solace_fact`` variant of EVERY arm (RS1's
    wrong-graft finding is why that variant exists, and an arm run only on the
    confounded mount would measure the confound).

  * ARM FLAGS COME FROM THE REGISTRATION.  ``arm_flags`` reads each arm's
    lever values off the registration rather than a table in the harness, so a
    row can never claim a geometry the run did not use.  C0 and C5 must both
    resolve to BOTH LEVERS OFF — C0 because it is the byte-identity
    reproduction, C5 because the ceiling is a live-band row that must not be
    perturbed by either lever.

  * TABLE ASSEMBLY reads the SAME keys off ``LayerTypeMassObserver.partition``
    that RS2's assembler read.  That is what makes a C0 cell and a B0 cell the
    same number rather than two computations that happen to agree, which is
    the whole basis of gate G2.  A row with no mass block still produces a row
    (with empty cells) rather than vanishing.

  * THE VERDICT RULE is applied as REGISTERED: either the mass line or the
    flip line is sufficient for CLOSES, a broken control disqualifies CLOSES,
    and an immeasurable change is NOTHING rather than MOVES.

  * THE PART-4 TRIGGER refuses to nominate an arm that broke a control, and
    names the number that triggered (or failed to trigger) it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_rs3_capture_seat_gpu as rs3  # noqa: E402
from scripts import grm_rs3_results as results  # noqa: E402

REGISTRATION = ROOT / "artifacts" / "grm_rs3" / "registration.json"

pytestmark = pytest.mark.skipif(
    not REGISTRATION.is_file(),
    reason="RS3 registration not written yet (it is written before any gate)")


@pytest.fixture(scope="module")
def registration() -> dict:
    return json.loads(REGISTRATION.read_text(encoding="utf-8"))


# ======================================================================
# Arm plan
# ======================================================================
def test_sessions_cover_every_registered_probe(registration):
    covered = set()
    for session in rs3.sessions_in_play(registration):
        covered.update(rs3.probes_for_session(registration, session))
    assert covered == set(rs3.registered_probes(registration))


def test_every_probe_appears_once_per_arm(registration):
    for session in rs3.sessions_in_play(registration):
        plan = rs3.arm_plan(registration, session)
        assert {str(item["arm"]) for item in plan} == set(rs3.ARMS)
        for probe_id in rs3.probes_for_session(registration, session):
            for arm in rs3.ARMS:
                hits = [item for item in plan
                        if item["probe_id"] == probe_id
                        and item["arm"] == arm
                        and item["variant"] == "registered"]
                assert len(hits) == 1, (probe_id, arm)


def test_solace_probe_gets_the_fact_variant_of_every_arm(registration):
    variant = registration["solace_fact_variant_REGISTERED_BEFORE_ANY_GATE"]
    probe_id = str(variant["probe_id"])
    plan = rs3.arm_plan(registration, str(variant["session_id"]))
    for arm in rs3.ARMS:
        hits = [item for item in plan
                if item["probe_id"] == probe_id and item["arm"] == arm
                and item["variant"] == "solace_fact"]
        assert len(hits) == 1, arm
        assert hits[0]["mount_ids"] == [int(variant["graft_id"])]


def test_plan_carries_the_registered_mount_ids_unchanged(registration):
    probes = rs3.registered_probes(registration)
    for session in rs3.sessions_in_play(registration):
        for item in rs3.arm_plan(registration, session):
            if item["variant"] != "registered":
                continue
            expected = [int(v)
                        for v in probes[item["probe_id"]]["a0_mounted_ids"]]
            assert item["mount_ids"] == expected


# ======================================================================
# Arm flags
# ======================================================================
def test_arm_flags_come_from_the_registration(registration):
    arms = registration["arms_REGISTERED_BEFORE_ANY_GATE"]
    for arm in rs3.ARMS:
        flags = rs3.arm_flags(registration, arm)
        assert flags["capture_pin"] == arms[arm]["capture_pin"]
        assert flags["seat_near_live"] == arms[arm]["seat_near_live"]


def test_baseline_and_ceiling_arms_have_both_levers_off(registration):
    """C0 is the byte-identity reproduction and C5 is the live-band ceiling.

    Either one running with a lever on would silently invalidate gate G2, or
    the reference row every delta is measured against.
    """
    for arm in ("C0", "C5"):
        flags = rs3.arm_flags(registration, arm)
        assert flags["capture_pin"] == "off", arm
        assert flags["seat_near_live"] is False, arm


def test_the_arm_matrix_covers_each_lever_alone_and_together(registration):
    seen = {(rs3.arm_flags(registration, arm)["capture_pin"],
             rs3.arm_flags(registration, arm)["seat_near_live"])
            for arm in rs3.ARMS}
    assert ("off", False) in seen        # C0 / C5
    assert ("mount", False) in seen      # C1m: pin alone
    assert ("live", False) in seen       # C1l: pin alone
    assert ("off", True) in seen         # C2: seating alone
    assert ("mount", True) in seen       # C3m: pair
    assert ("live", True) in seen        # C3l: pair


def test_unknown_arm_raises(registration):
    with pytest.raises(rs3.RS3Error):
        rs3.arm_flags(registration, "C99")


# ======================================================================
# Capture-text derivation
# ======================================================================
def test_fixture_capture_text_matches_its_registered_digest(registration):
    found = 0
    for probe_id, probe in rs3.registered_probes(registration).items():
        for node in probe.get("capture_nodes", ()):
            if not node.get("is_fixture_node", False):
                continue
            got = rs3.capture_text_for(
                registration, probe_id, int(node["graft_id"]))
            assert got is not None
            digest = hashlib.sha256(got["text"].encode("utf-8")).hexdigest()
            assert digest == got["registered_sha256"]
            found += 1
    assert found, "no fixture capture nodes found to check"


def test_split_child_has_no_registered_text(registration):
    """A FIT-TIME SPLIT CHILD must return None, not another node's text.

    The registration records it as ``is_fixture_node: false`` precisely so the
    re-capture falls back to the child's own stored text; handing it a fixture
    text would re-capture a DIFFERENT node.
    """
    checked = 0
    for probe_id, probe in rs3.registered_probes(registration).items():
        for node in probe.get("capture_nodes", ()):
            if node.get("is_fixture_node", False):
                continue
            assert rs3.capture_text_for(
                registration, probe_id, int(node["graft_id"])) is None
            checked += 1
    assert checked, "no split-child nodes found to check"


# ======================================================================
# Table assembly
# ======================================================================
def _row(**over):
    base = {
        "probe_id": "p", "variant": "registered", "arm": "C0",
        "levers": {}, "served_answer": "hello", "correct": True,
        "mounted_ids": [1],
        "mass": {
            "answer_positions": 3,
            "by_layer_type": {
                rs3.FULL: {
                    "mean_over_answer_positions": {
                        "mounted_mass": 0.25, "live_mass": 0.5,
                        "physical_sink_mass": 0.05,
                        "learned_sink_mass": 0.2},
                    "top_contributing_layer": {
                        "layer_ordinal": 5, "mounted_mass": 0.4}},
                rs3.SLIDING: {
                    "mean_over_answer_positions": {
                        "mounted_mass": 0.03, "live_mass": 0.6,
                        "physical_sink_mass": 0.01},
                    "top_contributing_layer": {
                        "layer_ordinal": 7, "mounted_mass": 0.06}},
            },
            "sliding_window_reach": {"always_reaches_mount": True},
        },
        "info": {},
    }
    base.update(over)
    return base


def test_table_reads_the_partition_keys_rs2_read():
    row = rs3.assemble_table([_row()])[0]
    assert row["mounted_mass_full_layers"] == 0.25
    assert row["mounted_mass_sliding_layers"] == 0.03
    assert row["live_mass_full_layers"] == 0.5
    assert row["live_mass_sliding_layers"] == 0.6
    assert row["sink_mass_full_layers"] == 0.05
    assert row["learned_sink_mass_full_layers"] == 0.2
    assert row["top_layer_full"] == 5
    assert row["top_layer_sliding"] == 7
    assert row["sliding_window_reaches_mount"] is True
    assert row["answer_positions"] == 3


def test_table_row_with_no_mass_block_still_produces_a_row():
    row = rs3.assemble_table([_row(mass={})])[0]
    assert row["probe_id"] == "p"
    assert row["mounted_mass_full_layers"] is None
    assert row["live_mass_full_layers"] is None


def test_table_orders_by_probe_then_variant_then_registered_arm_order():
    rows = [
        _row(probe_id="b", arm="C5"),
        _row(probe_id="a", arm="C3l"),
        _row(probe_id="a", arm="C0"),
        _row(probe_id="a", arm="C0", variant="solace_fact"),
    ]
    got = [(r["probe_id"], r["variant"], r["arm"])
           for r in rs3.assemble_table(rows)]
    assert got == [
        ("a", "registered", "C0"),
        ("a", "registered", "C3l"),
        ("a", "solace_fact", "C0"),
        ("b", "registered", "C5"),
    ]


def test_table_surfaces_the_capture_and_seating_receipts():
    row = rs3.assemble_table([_row(info={
        "rs3_capture": {
            "capture_pin": "live",
            "per_node": [{"recaptured_payload": {"sha256": "deadbeef"}}]},
        "rs3_seating": {
            "seat_offset_plan_head": 51, "seat_order": [1, 0],
            "mount_pos0": 51, "delta_positions": 32,
            "seat_near_live": True},
    })])[0]
    assert row["graft_digest"] == ["deadbeef"]
    assert row["capture_pin_applied"] == "live"
    assert row["seat_offset_plan_head"] == 51
    assert row["seat_order"] == [1, 0]
    assert row["delta_positions"] == 32
    assert row["seat_near_live_applied"] is True


# ======================================================================
# The verdict rule
# ======================================================================
#: The registered Part-4 gate, in the shape the registration writes it.
GATE = {
    "rule": (
        "Part 4 runs ONLY if Part 3's best arm meets the mass >= 0.50 OR "
        "flips >= 2/3 line; otherwise SKIP and say so, naming the number "
        "that triggered the skip."),
    "mass_line": 0.50,
    "flip_line": "2/3 of the refusers, with the correct value",
    "measured_on": "the refusers, mounted-band mass on FULL layers",
}


def test_the_test_gate_matches_the_registered_gates_shape(registration):
    """The fixture above must not drift from the real registration's shape,
    or these tests would be exercising a gate the harness never sees."""
    real = registration["part4_gate_REGISTERED_BEFORE_ANY_GATE"]
    assert set(GATE) == set(real)
    assert GATE["mass_line"] == real["mass_line"]


def _stats(mean, flipped, broken=(), delta=0.2):
    return {
        "mean_mounted_mass_full_layers": mean,
        "refusers_flipped": f"{flipped}/3",
        "controls_broken": list(broken),
        "delta_vs_c0_mean": delta,
    }


def test_mass_line_alone_is_enough_for_closes():
    got = results.verdict_for(_stats(0.55, 0), GATE)
    assert got["verdict"] == results.CLOSES
    assert got["meets_mass_line"] is True
    assert got["meets_flip_line"] is False


def test_flip_line_alone_is_enough_for_closes():
    got = results.verdict_for(_stats(0.40, 2), GATE)
    assert got["verdict"] == results.CLOSES
    assert got["meets_mass_line"] is False
    assert got["meets_flip_line"] is True


def test_a_broken_control_disqualifies_closes():
    got = results.verdict_for(
        _stats(0.90, 3, broken=["sup_reserve_tundra_ledger"]), GATE)
    assert got["verdict"] == results.MOVES
    assert got["controls_broken_disqualifies_closes"] is True


def test_measurable_movement_below_both_lines_is_moves():
    assert results.verdict_for(
        _stats(0.30, 1, delta=0.13), GATE)["verdict"] == results.MOVES


def test_immeasurable_movement_is_nothing_not_moves():
    got = results.verdict_for(
        _stats(0.1697, 0, delta=results.MEASURABLE / 10), GATE)
    assert got["verdict"] == results.NOTHING


def test_a_negative_movement_is_still_measurable():
    assert results.verdict_for(
        _stats(0.05, 0, delta=-0.12), GATE)["verdict"] == results.MOVES


# ======================================================================
# The Part-4 trigger
# ======================================================================
def _panels(**arms):
    return {"registered": {
        name: {"stats": stats, "verdict": results.verdict_for(stats, GATE)}
        for name, stats in arms.items()}}


def test_part4_trigger_names_the_flip_number_that_fired_it():
    registration = {"part4_gate_REGISTERED_BEFORE_ANY_GATE": GATE}
    panels = _panels(
        C1m=_stats(0.22, 1), C1l=_stats(0.22, 1), C2=_stats(0.37, 2),
        C3m=_stats(0.40, 2), C3l=_stats(0.41, 2))
    got = results.part4_trigger(registration, panels)
    assert got["triggered"] is True
    assert got["best_arm"] == "C3l"
    assert "2/3" in got["triggering_number"]
    assert "MASS line was NOT met" in got["triggering_number"]


def test_part4_trigger_will_not_nominate_an_arm_that_broke_a_control():
    registration = {"part4_gate_REGISTERED_BEFORE_ANY_GATE": GATE}
    panels = _panels(
        C1m=_stats(0.99, 3, broken=["c"]), C1l=_stats(0.99, 3, broken=["c"]),
        C2=_stats(0.37, 2), C3m=_stats(0.38, 2), C3l=_stats(0.39, 2))
    got = results.part4_trigger(registration, panels)
    assert got["best_arm"] in ("C2", "C3m", "C3l")
    assert got["triggered"] is True


def test_part4_trigger_reports_a_miss_with_its_number():
    registration = {"part4_gate_REGISTERED_BEFORE_ANY_GATE": GATE}
    panels = _panels(
        C1m=_stats(0.20, 0), C1l=_stats(0.20, 0), C2=_stats(0.21, 1),
        C3m=_stats(0.22, 1), C3l=_stats(0.23, 1))
    got = results.part4_trigger(registration, panels)
    assert got["triggered"] is False
    assert "0.23" in got["triggering_number"]
    assert "1/3" in got["triggering_number"]
