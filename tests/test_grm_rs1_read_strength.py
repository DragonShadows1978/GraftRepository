"""GRM-RS1 — CPU tests for the read-strength probe's pure parts.

ORDER: ``orders/GRM_RS1_READ_STRENGTH_PROBE.md`` (gate G1: "new CPU tests for
the probe script's pure parts (arm plan, id extraction from EB1 receipts,
table assembly, partition arithmetic sums to one)").

Every test here is CPU-only.  Nothing loads a model, nothing takes the GPU
lease, and nothing writes into ``artifacts/``.  The four things pinned are
exactly the four G1 names:

  * ID EXTRACTION FROM THE EB1 RECEIPTS.  The registration's ids must be the
    ids ON DISK, not ids the harness typed.  These tests re-derive every
    ``a0_mounted_ids`` and ``live_ids`` from the receipts the registration
    cites and require equality — so a registration built against a different
    (or edited) receipt fails here rather than silently steering a GPU arm.
    The FRAME of each cited receipt is checked too: the ``a0_mounted_ids``
    source must be an EPHEMERAL-frame receipt and the ``live_ids`` source a
    PERSISTENT-frame one, because reading the live window off the spec frame
    would give the empty list that hides the whole finding.

  * ARM PLAN.  Every registered probe appears once per arm, the arms are the
    registered set, and the plan carries the probe's ids forward unchanged.

  * TABLE ASSEMBLY.  The G3 column set is fixed, the ordering is
    probe-then-arm in the registered arm order, and a row with no mass block
    still produces a row (with empty mass cells) rather than vanishing.

  * PARTITION ARITHMETIC SUMS TO ONE.  ``partition_sum_ok`` accepts a
    partition inside the tolerance ``grm_demand._summarize_mass`` itself
    enforces, rejects one outside it, and rejects an INCOMPLETE partition
    instead of treating a missing band as zero.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_rs1_read_strength_gpu as rs1  # noqa: E402
from scripts import grm_rs1_registration as reg  # noqa: E402

REGISTRATION = ROOT / "artifacts" / "grm_rs1" / "registration.json"

pytestmark = pytest.mark.skipif(
    not REGISTRATION.is_file(),
    reason="RS1 registration not written yet (scripts/grm_rs1_registration.py)",
)


@pytest.fixture(scope="module")
def registration() -> dict:
    return rs1.read_registration(REGISTRATION)


def _receipt(record: dict) -> dict:
    return json.loads((ROOT / record["path"]).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# id extraction from the EB1 receipts
# ----------------------------------------------------------------------
def test_registration_covers_the_ordered_probes(registration):
    """The three refusers and the two contrast controls, and nothing else."""
    probes = rs1.registered_probes(registration)
    assert set(probes) == set(reg.REFUSERS) | set(reg.CONTROLS)
    roles = {pid: row["role"] for pid, row in probes.items()}
    assert {pid for pid, r in roles.items() if r == "refuser"} == set(
        reg.REFUSERS)
    assert {pid for pid, r in roles.items() if r == "control"} == set(
        reg.CONTROLS)


def test_a0_mounted_ids_come_from_an_ephemeral_frame_receipt(registration):
    """``a0_mounted_ids`` is the SPEC-frame mount, re-derived from disk."""
    for probe_id, row in rs1.registered_probes(registration).items():
        receipt = _receipt(row["eb1_receipt"])
        assert receipt["frame"]["ephemeral"] is True, probe_id
        # The spec frame's live window after install is EMPTY. That emptiness
        # is the whole premise of the order, so it is pinned, not assumed.
        assert receipt["live_graft_ids_after_install"] == [], probe_id
        served = {
            str(p["probe_id"]): [int(v) for v in p["mounted_ids"]]
            for p in receipt["probes"]
        }
        assert served[probe_id] == [int(v) for v in row["a0_mounted_ids"]]


def test_live_ids_come_from_a_persistent_frame_receipt(registration):
    """``live_ids`` is the window the LIVED run carried, re-derived from disk."""
    for probe_id, row in rs1.registered_probes(registration).items():
        receipt = _receipt(row["persistent_receipt"])
        # A persistent-frame receipt either predates the frame field or
        # records ephemeral=False; what it must never be is ephemeral=True.
        frame = receipt.get("frame")
        assert not (isinstance(frame, dict) and frame.get("ephemeral")), (
            probe_id)
        assert [int(v) for v in receipt["live_graft_ids_after_install"]] == [
            int(v) for v in row["live_ids"]], probe_id
        assert row["live_ids"], (
            f"{probe_id}: an EMPTY live window would make A1/A2 vacuous")


def test_live_node_ids_resolve_through_the_receipt_mapping(registration):
    """The node NAMES are the receipt's own index mapping, inverted."""
    for probe_id, row in rs1.registered_probes(registration).items():
        node_to_idx = {
            str(k): int(v) for k, v in row["fixture_node_to_idx"].items()}
        resolved = [node_to_idx[name] for name in row["live_node_ids"]]
        assert resolved == [int(v) for v in row["live_ids"]], probe_id


def test_registered_refusers_refused_and_controls_did_not(registration):
    """The order's premise, read off the registration rather than restated."""
    probes = rs1.registered_probes(registration)
    for probe_id in reg.REFUSERS:
        assert probes[probe_id]["a0_correct"] is False, probe_id
        assert probes[probe_id]["persistent_frame_correct"] is True, probe_id
    for probe_id in reg.CONTROLS:
        assert probes[probe_id]["a0_correct"] is True, probe_id
        assert probes[probe_id]["persistent_frame_correct"] is True, probe_id


# ----------------------------------------------------------------------
# arm plan
# ----------------------------------------------------------------------
def test_arm_plan_is_every_arm_times_every_probe_of_the_fixture(registration):
    for session in rs1.sessions_in_play(registration):
        probes = rs1.probes_for_session(registration, session)
        plan = rs1.arm_plan(registration, session)
        assert len(plan) == len(rs1.ARMS) * len(probes)
        assert [item["arm"] for item in plan[:len(probes)]] == (
            [rs1.ARMS[0]] * len(probes))
        for arm in rs1.ARMS:
            got = [item["probe_id"] for item in plan if item["arm"] == arm]
            assert got == probes, (session, arm)


def test_arm_plan_carries_the_registered_ids_unchanged(registration):
    probes = rs1.registered_probes(registration)
    for session in rs1.sessions_in_play(registration):
        for item in rs1.arm_plan(registration, session):
            row = probes[item["probe_id"]]
            assert item["a0_mounted_ids"] == [
                int(v) for v in row["a0_mounted_ids"]]
            assert item["live_ids"] == [int(v) for v in row["live_ids"]]
            assert item["live_node_ids"] == [
                str(v) for v in row["live_node_ids"]]


def test_arm_plan_rejects_a_fixture_with_no_registered_probe(registration):
    with pytest.raises(reg.RS1Error):
        rs1.arm_plan(registration, "not_a_fixture")


def test_sessions_in_play_are_deduplicated_and_ordered(registration):
    sessions = rs1.sessions_in_play(registration)
    assert len(sessions) == len(set(sessions))
    probes = rs1.registered_probes(registration)
    assert set(sessions) == {row["session_id"] for row in probes.values()}


# ----------------------------------------------------------------------
# table assembly
# ----------------------------------------------------------------------
def _row(probe_id: str, arm: str, *, correct: bool = False,
         mounted: float | None = 0.30, live: float = 0.20) -> dict:
    mass = None
    if mounted is not None:
        mass = {
            "mean_over_answer_positions": {
                "physical_sink_mass": 0.30,
                "mounted_mass": mounted,
                "live_mass": live,
                "learned_sink_mass": 1.0 - 0.30 - mounted - live,
            },
            "top_contributing_layer": {
                "full_layer_ordinal": 7, "mounted_mass": mounted + 0.1},
        }
    return {
        "probe_id": probe_id,
        "arm": arm,
        "served_answer": f"{probe_id}/{arm}",
        "correct": correct,
        "mounted_ids": [1],
        "mass": mass,
    }


def test_table_is_probe_then_arm_in_the_registered_arm_order():
    rows = [
        _row("b_probe", "A2b"), _row("a_probe", "A4"),
        _row("a_probe", "A0"), _row("b_probe", "A0"),
    ]
    table = rs1.assemble_table(rows)
    assert [(r["probe_id"], r["arm"]) for r in table] == [
        ("a_probe", "A0"), ("a_probe", "A4"),
        ("b_probe", "A0"), ("b_probe", "A2b"),
    ]


def test_table_columns_are_the_registered_g3_set():
    table = rs1.assemble_table([_row("p", "A0", correct=True)])
    assert set(table[0]) == {
        "probe_id", "arm", "served", "correct", "mounted_ids",
        "mounted_band_mass", "live_band_mass", "sink_mass",
        "learned_sink_mass", "top_contributing_layer",
        "top_layer_mounted_mass",
    }
    assert table[0]["correct"] is True
    assert table[0]["mounted_band_mass"] == pytest.approx(0.30)
    assert table[0]["live_band_mass"] == pytest.approx(0.20)
    assert table[0]["top_contributing_layer"] == 7


def test_table_keeps_a_row_whose_mass_block_is_missing():
    """A serve with no captured mass is still a serve; it must not vanish."""
    table = rs1.assemble_table([_row("p", "A1", mounted=None)])
    assert len(table) == 1
    assert table[0]["mounted_band_mass"] is None
    assert table[0]["top_contributing_layer"] is None
    assert table[0]["served"] == "p/A1"


# ----------------------------------------------------------------------
# partition arithmetic sums to one
# ----------------------------------------------------------------------
def test_partition_sums_to_one_is_accepted():
    assert rs1.partition_sum_ok({
        "physical_sink_mass": 0.31,
        "mounted_mass": 0.29,
        "live_mass": 0.22,
        "learned_sink_mass": 0.18,
    })


@pytest.mark.parametrize("total", (0.5, 1.5, 0.97, 1.03))
def test_partition_outside_the_tolerance_is_rejected(total):
    assert not rs1.partition_sum_ok({
        "physical_sink_mass": total,
        "mounted_mass": 0.0,
        "live_mass": 0.0,
        "learned_sink_mass": 0.0,
    })


@pytest.mark.parametrize("total", (0.98, 1.02))
def test_partition_edges_of_the_tolerance_are_accepted(total):
    """The bound is the one grm_demand enforces: [0.98, 1.02], inclusive."""
    assert rs1.partition_sum_ok({
        "physical_sink_mass": total,
        "mounted_mass": 0.0,
        "live_mass": 0.0,
        "learned_sink_mass": 0.0,
    })


def test_incomplete_partition_is_rejected_not_treated_as_zero():
    assert not rs1.partition_sum_ok({
        "physical_sink_mass": 0.5, "mounted_mass": 0.5, "live_mass": 0.0,
    })
    assert not rs1.partition_sum_ok({
        "physical_sink_mass": None, "mounted_mass": 0.5,
        "live_mass": 0.3, "learned_sink_mass": 0.2,
    })


def test_partition_tolerance_is_the_production_observer_s_own():
    """RS1 must not invent its own band names or its own tolerance."""
    from core import grm_demand

    body = inspect.getsource(grm_demand.DemandObserver._full_mass)
    for name in rs1.MASS_NAMES:
        assert f'"{name}"' in body, name
    summarize = inspect.getsource(grm_demand.DemandObserver._summarize_mass)
    assert "0.98 <= total <= 1.02" in summarize
    ours = inspect.getsource(rs1.partition_sum_ok)
    assert "0.98 <=" in ours and "<= 1.02" in ours


# ----------------------------------------------------------------------
# the mechanism verdict vocabulary
# ----------------------------------------------------------------------
def test_mechanism_verdict_follows_the_registered_precedence():
    assert rs1.mechanism_verdict({"A3": True, "A4": True, "A2": True,
                                  "A1": True}) == "QUANTIZATION"
    assert rs1.mechanism_verdict({"A4": True, "A2": True,
                                  "A1": True}) == "PRIOR"
    assert rs1.mechanism_verdict({"A2": True, "A1": True}) == "CONTENT"
    assert rs1.mechanism_verdict({"A1": True}) == "POSITION"
    assert rs1.mechanism_verdict({}) == "UNRESOLVED"
    # A2 alone is CONTENT, not POSITION: POSITION requires A1 to rescue and
    # A2 to fail, which is what "the live band, not the text" means.
    assert rs1.mechanism_verdict({"A2": True}) == "CONTENT"


@pytest.mark.parametrize("sub", rs1.A2_ARMS)
def test_either_a2_sub_arm_counts_as_a2_rescuing(sub):
    """The width-forced sub-arm split must not change the registered rule."""
    assert rs1.a2_rescued({sub: True})
    assert rs1.mechanism_verdict({sub: True, "A1": True}) == "CONTENT"
    assert rs1.mechanism_verdict({sub: True}) == "CONTENT"


def test_a2_rescued_is_false_when_no_mount_arm_served():
    assert not rs1.a2_rescued({})
    assert not rs1.a2_rescued({"A1": True, "A3": False})
    assert rs1.mechanism_verdict({"A1": True, "A2a": False,
                                  "A2b": False}) == "POSITION"


def test_a2_sub_arms_are_in_the_arm_set():
    assert set(rs1.A2_ARMS) <= set(rs1.ARMS)
    assert set(rs1.DIRECT_ARMS) <= set(rs1.ARMS)
    assert "A1" in rs1.DIRECT_ARMS


def test_every_verdict_is_in_the_registered_vocabulary(registration):
    vocabulary = set(registration["vocabulary_REGISTERED_BEFORE_ANY_GATE"])
    for arms in (
        {"A3": True}, {"A4": True}, {"A2": True}, {"A1": True}, {},
    ):
        assert rs1.mechanism_verdict(arms) in vocabulary


# ----------------------------------------------------------------------
# the A4 wording
# ----------------------------------------------------------------------
@pytest.mark.parametrize("question,subject", (
    ("What is the current Harbor token value?", "current Harbor token value"),
    ("What is the current Praxis dock value?", "current Praxis dock value"),
    ("What is the current Solace key value?", "current Solace key value"),
))
def test_strict_wording_preserves_the_probe_subject(question, subject):
    """A4 changes the FRAME around the question, never its identifiers."""
    asked = rs1.strict_wording(question)
    assert subject in asked
    assert asked.startswith(rs1.STRICT_PREFIX)
    assert asked.endswith(rs1.STRICT_SUFFIX)
    assert "?" not in asked


def test_strict_wording_passes_an_unfamiliar_question_through_the_frame():
    asked = rs1.strict_wording("Name the vault keyword.")
    assert "Name the vault keyword." in asked
    assert asked.startswith(rs1.STRICT_PREFIX)


def test_strict_wording_is_the_repo_registered_variant():
    """The wording is quoted from the sweep script, not invented here."""
    sweep = (ROOT / "scripts" / "gpt_oss20b_turn_prompt_sweep.py").read_text(
        encoding="utf-8")
    assert rs1.STRICT_PREFIX in sweep
    assert rs1.STRICT_SUFFIX.strip(". ") in sweep
    assert rs1.STRICT_TURN_PREFIX in sweep


def test_strict_wording_plants_no_numeral():
    """The turn label is DROPPED: its numeral is an unbindable identifier.

    Measured: with ``Turn 50.`` in the query, ``_probe_identifier_tokens``
    reads ``50`` as an identifier, nothing binds it, and P2A's
    ``identifier_unbound`` abstention fires before any mount work — which
    would measure the abstention rule, not the instruct prior.
    """
    for question in (
        "What is the current Harbor token value?",
        "What is the current Solace key value?",
    ):
        asked = rs1.strict_wording(question)
        assert rs1.STRICT_TURN_PREFIX not in asked
        assert not any(ch.isdigit() for ch in asked), asked


# ----------------------------------------------------------------------
# the answer comparator
# ----------------------------------------------------------------------
def test_answer_verdict_is_the_frozen_det1_comparator():
    good = rs1.answer_verdict(
        "The current Harbor token value is **Nacre-6-Blue**.",
        expected_values=["nacre-6-blue"], rejected_values=["opal-7-green"])
    assert good["correct"] is True
    bad = rs1.answer_verdict(
        "I'm sorry, but I don't have that information.",
        expected_values=["nacre-6-blue"], rejected_values=[])
    assert bad["correct"] is False
    guarded = rs1.answer_verdict(
        "Nacre-6-Blue, formerly Opal-7-Green.",
        expected_values=["nacre-6-blue"], rejected_values=["opal-7-green"])
    assert guarded["correct"] is False
