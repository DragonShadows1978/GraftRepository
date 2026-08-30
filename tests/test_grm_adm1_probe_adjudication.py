from pathlib import Path

from scripts.grm_adm1_probe_adjudication import (
    completed_capture_available,
    project_l2_mounts,
    validate_frame_receipts,
)


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "artifacts" / "grm_adm1" / "run_20260830T093538Z_3defaa"


def test_l2_projection_filters_only_comounted_ancestor():
    ancestors = {0: set(), 1: {0}, 2: set()}
    assert project_l2_mounts([0, 2], ancestors) == [0, 2]
    assert project_l2_mounts([0, 1, 2], ancestors) == [1, 2]
    assert project_l2_mounts([1, 2], ancestors) == [1, 2]


def test_completed_diag_capture_is_exact_anchor_and_l2_noop():
    assert completed_capture_available(CAPTURE, "diag")
    receipt = validate_frame_receipts(CAPTURE, "diag")
    assert receipt["status"] == "PASS"
    assert receipt["raw_probe_score"] == [7, 9]
    assert receipt["registered_anchor_exact"] is True
    assert receipt["all_a_k3_canonical_replays_equal"] is True
    assert receipt["capture_mode"] == "legacy_l2_off_migrated_by_noop_projection"
    assert receipt["l2_projection"]["l2_projection_noop"] is True
    assert receipt["l2_projection"]["changed_candidate_sets"] == []
