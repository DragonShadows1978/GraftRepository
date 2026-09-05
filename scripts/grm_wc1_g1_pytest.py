#!/usr/bin/env python3
"""GRM-WC1 G1 — record the sharded pytest battery and its comparison to RS4.

The numbers here are TRANSCRIBED FROM THE LOGS IN ``logs/``, and each shard
row names the log it came from so the claim is checkable rather than asserted.
The comparison baseline is RS4's own G1 receipt in the MAIN checkout, which is
the last G1 run on this branch.

THE WORKTREE-ARTIFACT-ABSENCE REGION, stated plainly rather than hidden.
``artifacts/`` is gitignored, so a fresh worktree has no artifacts tree.  A
family of tests reads FROZEN RECEIPTS from ``artifacts/grm_det1/
run_20260831T160525Z_2/`` (8.6 GB, the DET1.5 race tree) and from other
campaign artifact trees.  Those tests FAIL IN THIS WORKTREE and PASS IN THE
MAIN CHECKOUT at the same HEAD, with FileNotFoundError on the missing
receipt — every one of them.  That is an absence of read-only input data in a
worktree, NOT a regression: WC1 changed no core file and no test that reads
those trees.  The region is reported as INCOMPLETE, never as a pass.

One receipt WAS copied in (read-only, unchanged, sha256 recorded):
``artifacts/grm_rs3/registration.json``, because RS4's own CPU test file reads
it and the order authorises writing under ``artifacts/``.  With it present
that file goes 20 passed / 5 skipped, matching the main checkout.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_cmc1_mechanism import (  # noqa: E402
    canonical_json_bytes, sha256_bytes,
)
from scripts.grm_det1_common import file_record  # noqa: E402
from scripts.grm_wc1_common import ARTIFACT_DIR, SCHEMA_PREFIX  # noqa: E402
from scripts.grm_wc1_sweep_gpu import registration_record  # noqa: E402

RECEIPT = ARTIFACT_DIR / "grm_wc1_g1_pytest.json"
RS4_G1 = Path(
    "/mnt/ForgeRealm/GraftRepository/artifacts/grm_rs4/grm_rs4_g1_pytest.json")

SHARDS: tuple[dict[str, Any], ...] = (
    {
        "shard": "A (WC1 + RS3/RS4 + LSR P2A/P2B/P2C + RT1 + admission + EB1)",
        "line": "354 passed, 10 skipped, 2 warnings in 6.95s",
        "rc": 0,
        "log": "logs/wc1_pytest_G1_A.log",
        "note": (
            "Contains the 39 NEW WC1 CPU tests. First run of this shard had "
            "2 failures in tests/test_grm_rs4_baseline_and_gates.py, both "
            "FileNotFoundError on artifacts/grm_rs3/registration.json — an "
            "absent read-only receipt in a fresh worktree, not a code fault. "
            "The receipt was copied in unchanged and the shard went green."),
    },
    {
        "shard": "C1 (importance, S4, SC1.1, three-pass, supersession, arena)",
        "line": "374 passed, 2 warnings in 7.37s",
        "rc": 0,
        "log": "logs/wc1_pytest_G1_C1.log",
    },
    {
        "shard": "D2 (router baseline, route seams, RS1, RS2, runtime "
                 "lifecycle, deepseek hooks, apamq)",
        "line": "1 failed, 133 passed, 66 skipped, 2 warnings in 291.68s",
        "rc": 1,
        "log": "logs/wc1_pytest_G1_D2.log",
        "note": (
            "The single failure is "
            "tests/test_apamq_fc_ppl.py::test_summary_is_paired_and_contains"
            "_no_verdict — ALREADY IN THE PRE-EXISTING FAILING SET per RS4's "
            "G1 receipt ('STILL FAILS — reproduced alone. In the set.'), and "
            "re-confirmed failing in the MAIN checkout during this gate "
            "(1 failed, 5 passed in 0.25s)."),
    },
    {
        "shard": "E (native runtime, probe ladder)",
        "line": "136 passed, 2 warnings in 410.70s",
        "rc": 0,
        "log": "logs/wc1_pytest_G1_E.log",
        "note": (
            "Both files are in RS4's 'not completed under the per-call "
            "budget' region; under WC1 they completed and passed."),
    },
    {
        "shard": "quant_format alone",
        "line": "22 passed, 2 warnings in 0.38s",
        "rc": 0,
        "log": "(run inline; matches RS4's '22 passed alone')",
    },
)

WORKTREE_ABSENCE_REGION: dict[str, Any] = {
    "verdict": "INCOMPLETE IN THIS WORKTREE — reported, never counted as pass",
    "cause": (
        "artifacts/ is gitignored, so a fresh worktree has no artifacts tree. "
        "These files read FROZEN receipts (the 8.6 GB DET1.5 race tree at "
        "artifacts/grm_det1/run_20260831T160525Z_2/ and sibling campaign "
        "trees). Every failure and error in this region is a "
        "FileNotFoundError on a missing read-only receipt."),
    "not_a_regression_because": (
        "WC1 modified NO core file, NO existing script and NO existing test. "
        "The identical file lists PASS in the main checkout at the same HEAD, "
        "measured during this gate."),
    "measured_both_ways": [
        {
            "files": (
                "test_grm_det1_11_achieved, test_grm_det1_11_census, "
                "test_grm_det1_11_envelope_lineage, test_grm_det1_3_snapshot, "
                "test_grm_det1_5_campaign, test_grm_det1_7_registry, "
                "test_grm_det1_7_source_auth, test_grm_det1_8_source_auth, "
                "test_grm_det1_9_findings, test_grm_det1_9_source_auth, "
                "test_grm_det1_baseline_registry, test_grm_adm1_3_dual_frame, "
                "test_grm_adm1_probe_adjudication, test_grm_adm1_snapshot, "
                "test_lsr_p2c_1_census"),
            "worktree": "25 failed, 249 passed, 2 skipped, 23 errors",
            "main_checkout": "299 passed",
            "worktree_log": "logs/wc1_pytest_G1_B.log",
        },
        {
            "files": (
                "test_grm_sc1_2_e2e_pairs, test_grm_sc1_demand_loop, "
                "test_grm_sc2_calibration_early_abort"),
            "worktree": "23 failed, 118 passed, 20 errors",
            "main_checkout": "161 passed",
            "worktree_log": "logs/wc1_pytest_G1_C.log",
        },
    ],
}

GPU_CONTENTION_REGION: dict[str, Any] = {
    "verdict": "NOT COMPLETED — the card was held by the parallel seat",
    "files": [
        "tests/test_scribe_floor_gates.py", "tests/test_scribe_resume_kill.py",
        "tests/test_scribe_g0.py", "tests/test_scribe_fit_probe.py",
        "tests/test_graft_migrate.py", "tests/test_graft_descent.py",
        "tests/test_graft_corpus100.py",
    ],
    "symptom": "RuntimeError: cudaMalloc failed: out of memory, AT COLLECTION",
    "log": "logs/wc1_pytest_G1_D.log, logs/wc1_pytest_G1_D1.log",
    "precedent": (
        "RS4's own G1 receipt records the same class ('GPU CONTENTION — "
        "QuantLinearTC INT4 init failed is the OOM path. Another seat, in a "
        "SEPARATE worktree, held the card'). ANOTHER SEAT IS RUNNING THIS "
        "SAME ORDER in a parallel worktree throughout this gate. Nothing was "
        "killed, signalled or waited on to clear it."),
    "honesty_note": (
        "Reported as an INCOMPLETE region, not as a pass. None of these files "
        "imports any WC1 module."),
    "also_not_completed": {
        "files": [
            "tests/test_graft_gqa_arena.py",
            "tests/test_graft_gqa_features.py",
        ],
        "line": "2 warnings in 183.07s (rc=5, no tests collected)",
        "log": "logs/wc1_pytest_G1_F.log",
        "note": (
            "Both are in RS4's own not-completed region; here they collected "
            "zero tests rather than failing."),
    },
}


def build() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}.g1.v1",
        "program": "GRM",
        "phase": "WC1",
        "gate": "G1",
        "order": "orders/GRM_WC1_ARENA_WIDTH_CURVE.md",
        "branch": "wc1-opus",
        "rule": (
            "pytest sharded under 10 min per call; the pre-existing failing "
            "set unchanged vs RS4's G1; new CPU tests for the sweep driver's "
            "pure parts."),
        "clean_shards": [dict(s) for s in SHARDS],
        "wc1_new_tests": {
            "file": "tests/test_grm_wc1_sweep.py",
            "line": "39 passed in 0.09s",
            "covers": [
                "the registered width grid and battery names",
                "frame derivation moves exactly one field",
                "the single-variable check CATCHES a second change "
                "(both a flag change and a new top-level key)",
                "a non-positive width and a width-less frame are refused",
                "a battery that did not run reads as None, never as 0",
                "regression vs 96 distinguishes regression from recovery",
                "a differing probe set is non-comparable, not a fake "
                "regression",
                "split census reads only the existing width-guard flags",
                "an empty mean is None, never 0.0",
                "every registered prediction's HIT/MISS logic, and that an "
                "unknown or unrunnable check is UNSCORED not a silent HIT",
            ],
        },
        "wc1_touches": {
            "core_changed": False,
            "existing_files_modified": [],
            "files_added": [
                "scripts/grm_wc1_common.py",
                "scripts/grm_wc1_registration.py",
                "scripts/grm_wc1_amend.py",
                "scripts/grm_wc1_sweep_gpu.py",
                "scripts/grm_wc1_results.py",
                "scripts/grm_wc1_g1_pytest.py",
                "tests/test_grm_wc1_sweep.py",
            ],
            "note": (
                "WC1 added seven files and modified none. No existing test "
                "imports any of them."),
        },
        "preexisting_set_comparison": {
            "rs4_named_failure_still_failing": (
                "tests/test_apamq_fc_ppl.py::"
                "test_summary_is_paired_and_contains_no_verdict — re-measured "
                "FAILING in the main checkout during this gate, so it is "
                "unchanged, in the set."),
            "rs4_named_now_passing_still_passing": (
                "tests/test_graft_quant_format.py — 22 passed alone, as RS4 "
                "recorded."),
            "new_failures_attributable_to_wc1": [],
            "verdict": (
                "NO REGRESSION ATTRIBUTABLE TO WC1. Every failure in this "
                "battery is one of: (a) the RS4-documented pre-existing "
                "apamq failure, (b) the worktree-artifact-absence region "
                "(passes in the main checkout at the same HEAD), or (c) the "
                "GPU-contention region RS4 documented, with a second seat on "
                "the card throughout."),
        },
        "worktree_artifact_absence_region": WORKTREE_ABSENCE_REGION,
        "gpu_contention_region": GPU_CONTENTION_REGION,
        "read_only_receipts_copied_into_this_worktree": [
            file_record(ROOT / "artifacts" / "grm_rs3" / "registration.json"),
            file_record(
                ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2"
                / "runtime_frame_28b3196f8fb04a41.json"),
            file_record(
                ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2"
                / "det1_11" / "census"
                / "lived_serving_census_98ef71e88dec17a1.json"),
        ],
        "rs4_g1_baseline": {
            "path": str(RS4_G1),
            "sha256": sha256_bytes(RS4_G1.read_bytes()),
            "collection_at_rs4": 1624,
        },
        "registration": registration_record(),
    }


def main(argv: list[str] | None = None) -> int:
    body = canonical_json_bytes(build())
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_bytes(body)
    print(f"receipt={RECEIPT}")
    print(f"sha256={sha256_bytes(body)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
