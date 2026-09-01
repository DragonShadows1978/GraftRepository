"""GRM-DET1.11: no stage consumer hard-pins a superseded envelope.

The DET1.9 precollection envelope is re-authored whenever the declared source
set or its bytes move, writing a new append-only file (_r6, _r8 ... _r12).
Artifacts frozen in earlier campaign rounds name the envelope that governed
THAT round.  Campaign r10 fail-closed because every readback compared those
persisted records for equality against the CURRENT envelope, so each advance
retroactively invalidated all prior markers, receipts, and shards.

The invariant these tests pin: a persisted authorization identity is checked
for membership of the validated lineage, never for equality against the
newest member.  A record naming a file outside the series is still rejected —
lineage tolerance is not permissiveness.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_det1_5_analyze as analyze  # noqa: E402
from scripts import grm_det1_5_gpu as campaign  # noqa: E402
from scripts.grm_det1_common import DETError, file_record  # noqa: E402


# --- the lineage resolver ------------------------------------------------


def test_lineage_contains_every_authored_envelope():
    lineage = campaign.det1_7_source_auth_lineage()
    assert len(lineage) >= 2, "expected an append-only series of envelopes"
    names = [path.name for path in lineage]
    assert len(set(names)) == len(names)
    for name in names:
        assert name.startswith(campaign.DET1_9_ENVELOPE_STEM)


def test_lineage_is_ordered_by_authorship_not_filename():
    """``_r10`` must follow ``_r9``; lexicographic order gets this wrong."""
    lineage = campaign.det1_7_source_auth_lineage()
    created = [
        str(campaign.read_json(path).get("created_utc")) for path in lineage
    ]
    assert created == sorted(created)

    suffixes = [
        path.stem[len(campaign.DET1_9_ENVELOPE_STEM):] for path in lineage
    ]
    rounds = [int(s[2:]) for s in suffixes if s.startswith("_r")]
    assert rounds == sorted(rounds)


def test_current_envelope_is_the_newest_lineage_member():
    lineage = campaign.det1_7_source_auth_lineage()
    assert lineage[-1].name == campaign.DET1_7_SOURCE_AUTH.name


def test_every_superseded_envelope_is_still_accepted():
    """The r10 regression, stated directly."""
    lineage = campaign.det1_7_source_auth_lineage()
    assert len(lineage) >= 2
    for path in lineage[:-1]:
        record = file_record(path)
        assert campaign.is_lineage_source_authorization(record) is True, (
            f"superseded envelope {path.name} is not accepted; a marker "
            f"frozen under it would fail closed"
        )


def test_records_outside_the_series_are_rejected():
    assert campaign.is_lineage_source_authorization(None) is False
    assert campaign.is_lineage_source_authorization({}) is False
    assert campaign.is_lineage_source_authorization(
        {"path": "nowhere.json", "bytes": 1, "sha256": "f" * 64}
    ) is False
    # The DET1.8 predecessor is a real file but is NOT a DET1.9 member.
    det1_8 = campaign.FROZEN_RUN / campaign.DET1_8_SOURCE_AUTH.name
    if det1_8.is_file():
        assert campaign.is_lineage_source_authorization(
            file_record(det1_8)
        ) is False


def test_a_tampered_record_for_a_real_member_is_rejected():
    """Right filename, wrong bytes: membership is by full record."""
    lineage = campaign.det1_7_source_auth_lineage()
    forged = dict(file_record(lineage[0]))
    forged["sha256"] = "0" * 64
    assert campaign.is_lineage_source_authorization(forged) is False


def test_require_helper_names_the_offending_record():
    with pytest.raises(DETError, match="outside the validated DET1.9"):
        campaign._require_lineage_source_authorization(
            {"path": "nowhere.json", "bytes": 1, "sha256": "f" * 64},
            "test consumer",
        )


# --- the sweep invariant -------------------------------------------------
#
# Every VERIFY-EQUALITY site the r10 class produced, enumerated.  If a future
# change reintroduces a hard pin at one of these functions, the source scan
# below fails and names it.

# (module, function, artifact it reads back)
ENVELOPE_CONSUMERS = [
    (campaign, "_validate_stage_marker", "stage marker + stage receipt"),
    (campaign, "validate_det1_7_terminal_amendment", "terminal amendment"),
    (campaign, "_calibration_thresholds", "thresholds receipt"),
    (analyze, "_validate_common_stage_bindings", "stage receipts"),
    (analyze, "_validate_marker_chain", "stage markers + prerequisites"),
    (analyze, "_worker_shard", "worker shard"),
]

# The shape that caused r10: comparing a persisted value against a freshly
# computed record of the CURRENT envelope.
FORBIDDEN_PINS = (
    "DET1_7_SOURCE_AUTH.name)",
    "file_record(precollection_authorization_path)",
    'source_auth["record"]',
)


@pytest.mark.parametrize(
    "module,function,artifact",
    ENVELOPE_CONSUMERS,
    ids=[f for _m, f, _a in ENVELOPE_CONSUMERS],
)
def test_no_stage_consumer_hard_pins_a_superseded_envelope(
    module, function, artifact
):
    """No readback of a persisted envelope identity may pin the current one.

    Reading the source is the only way to state this invariant without a
    live campaign: these functions validate artifacts that a CPU test cannot
    manufacture, but the forbidden construct is visible statically.
    """
    source = inspect.getsource(getattr(module, function))
    for pin in FORBIDDEN_PINS:
        assert pin not in source, (
            f"{module.__name__}.{function} hard-pins the current envelope "
            f"via {pin!r} while validating {artifact}; a marker frozen under "
            f"an earlier envelope will fail closed (the r10 regression). "
            f"Use the lineage-membership check instead."
        )


@pytest.mark.parametrize(
    "module,function,artifact",
    ENVELOPE_CONSUMERS,
    ids=[f for _m, f, _a in ENVELOPE_CONSUMERS],
)
def test_every_consumer_checks_lineage_membership(module, function, artifact):
    """The positive half: each consumer must actually run the check."""
    source = inspect.getsource(getattr(module, function))
    assert (
        "_require_lineage_source_authorization" in source
        or "_require_lineage_authorization" in source
    ), (
        f"{module.__name__}.{function} validates {artifact} but never checks "
        f"lineage membership of its authorization record"
    )


@pytest.mark.parametrize(
    "function",
    ["_write_stage_marker", "_threshold_receipt_payload"],
)
def test_persisted_sites_still_stamp_the_current_envelope(function):
    """Writers must NOT become lineage-tolerant — they pin the newest.

    Lineage tolerance on the write side would let a stale envelope be
    stamped into a fresh artifact, which is the opposite defect.
    """
    source = inspect.getsource(getattr(campaign, function))
    assert "DET1_7_SOURCE_AUTH.name)" in source, (
        f"{function} must stamp the CURRENT envelope; lineage tolerance "
        f"belongs on the readback side only"
    )
    assert "_require_lineage_source_authorization" not in source, (
        f"{function} is a writer and must not perform a readback check"
    )


# --- stage-shard binding is marker-referenced ----------------------------
#
# Campaign r11 ran every stage to completion and only the analyzer refused:
# eleven append-only rounds had left completed shard receipts from earlier
# attempts in the same shard directories, and the binding globbed the
# directory and demanded exactly one.  The authority is the marker chain,
# which already names this round's receipt; historical attempts are inert.


def test_stage_shard_binding_resolves_the_attempt_from_the_marker():
    """The binding must derive its attempt dir from the named receipt."""
    source = inspect.getsource(analyze._validate_stage_shards)
    assert "bound_attempt_dir = receipt_path.parent.resolve()" in source, (
        "the bound attempt must be resolved from the marker-referenced "
        "receipt, not searched for in the shard directory"
    )
    assert "if output_path.parent.resolve() != bound_attempt_dir:" in source, (
        "completed attempts outside the bound attempt directory must be "
        "skipped; globbing them into the binding is the r11 regression"
    )


def test_stray_completed_receipts_in_shard_dirs_change_nothing():
    """The live run-dir carries many completed attempts per shard.

    This is the r11 condition itself: if the binding still globbed, the
    presence of more than one completed attempt in a spec directory would
    make the analyzer refuse.  The analyzer's own receipt on disk is the
    proof it does not.
    """
    campaign_dir = campaign.campaign_root(campaign.FROZEN_RUN)
    shard_root = campaign_dir / "det1_7/plant_registration/shards/e2e-cal"
    if not shard_root.is_dir():
        pytest.skip("plant_registration shards not present in this tree")

    completed = []
    for output_path in sorted(shard_root.glob("attempt_*/worker_output.json")):
        payload = campaign.read_json(output_path)
        if payload.get("status") in ("PASS", "COMPLETE"):
            completed.append(output_path.parent.name)
    assert len(completed) > 1, (
        "expected several completed attempts from append-only rounds; "
        "without them this test cannot witness the r11 condition"
    )


def test_analysis_receipt_exists_and_names_a_lineage_envelope():
    """End-to-end: the analyzer ran to completion over the real evidence."""
    analysis_dir = campaign.campaign_root(campaign.FROZEN_RUN) / "analysis"
    receipts = sorted(analysis_dir.glob("analysis_receipt_*.json"))
    if not receipts:
        pytest.skip("no analysis receipt in this tree")
    assert len(receipts) == 1, "analysis directory holds more than one receipt"
    payload = campaign.read_json(receipts[0])
    assert payload.get("prediction_verdict") in (
        "SUPPORTED", "PARTIAL", "REFUTED",
    )
    campaign._require_lineage_source_authorization(
        payload.get("precollection_authorization"), "analysis receipt",
    )


def test_marker_prerequisite_chain_tolerates_lineage_positionally():
    """The subtlest pin: the envelope sits anonymously in a list."""
    source = inspect.getsource(analyze._validate_marker_chain)
    assert "authorization_slots" in source, (
        "the prerequisite chain embeds the envelope positionally; it needs "
        "an explicit lineage-tolerant slot, not a whole-list equality"
    )
    assert 'marker.get("prerequisites") == expected' not in source, (
        "whole-list equality on prerequisites re-pins the envelope"
    )
