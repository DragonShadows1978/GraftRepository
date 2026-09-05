#!/usr/bin/env python3
"""GRM-WC1 — write the IMMUTABLE registration before any gate runs.

``artifacts/grm_wc1/registration.json`` is written ONCE and never rewritten.
A second invocation refuses rather than overwriting; anything that has to
change afterwards goes in a separate amendment file citing this one's sha256.
"""
from __future__ import annotations

import json
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
from scripts.grm_wc1_common import (  # noqa: E402
    ARTIFACT_DIR, BATTERIES, LEVERS, REFERENCE_WIDTH, SCHEMA_PREFIX,
    WIDTH_GRID, WC1Error,
)

REGISTRATION = ARTIFACT_DIR / "registration.json"

#: The frozen DET1 runtime frame every battery derives its flags from, and the
#: lived census both the sup and the census batteries score against.  Copied
#: read-only into this worktree from the main checkout (artifacts/ is
#: gitignored, so a worktree has no artifacts tree of its own); the copies'
#: sha256 are recorded here so the binding is checkable.
FROZEN_FRAME = (
    ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2"
    / "runtime_frame_28b3196f8fb04a41.json"
)
FROZEN_CENSUS = (
    ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2" / "det1_11"
    / "census" / "lived_serving_census_98ef71e88dec17a1.json"
)

#: The RS3 Part 4 receipt in the MAIN checkout — read-only reference material,
#: and G2's reproduction target.
RS3_PART4 = Path(
    "/mnt/ForgeRealm/GraftRepository/artifacts/grm_rs3/grm_rs3_part4.json")

#: The lead's registered predictions, verbatim from the order's
#: "Registered predictions (lead, before any run)" section, each bound to the
#: machine check that scores it in ``grm_wc1_common.prediction_verdicts``.
PREDICTIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "P1",
        "check": "96_not_optimum",
        "claim": (
            "96 is NOT the optimum: 128 and 192 match or beat 96 on all "
            "three batteries (fewer splits, no co-mount blending because "
            "admission is k=1 on these probes)."),
    },
    {
        "id": "P2",
        "check": "64_loses_two_sup",
        "claim": (
            "64 loses at least 2 sup probes (unseatable-after-split class)."),
    },
    {
        "id": "P3",
        "check": "256_yarn_wall",
        "claim": (
            "256 shows the first sign of the YaRN wall: wall ms per turn "
            "rises and at least one census probe degrades; if it does not, "
            "say so — the 384 collapse may be a cliff, not a slope."),
    },
    {
        "id": "P4",
        "check": "longhorizon_4_of_4",
        "claim": "Long-horizon 4/4 at every width >= 96.",
    },
)


def build_registration() -> dict[str, Any]:
    part4 = json.loads(RS3_PART4.read_text(encoding="utf-8"))
    return {
        "schema": f"{SCHEMA_PREFIX}.registration.v1",
        "program": "GRM",
        "phase": "WC1",
        "order": "orders/GRM_WC1_ARENA_WIDTH_CURVE.md",
        "branch": "wc1-opus",
        "question": (
            "Primer open question 2: how wide should the arena be? Measured "
            "under the spec frame (EB1) with seat-near-live + capture-pin "
            "(RS3) ON. Measurement only; no production change."),
        "independent_variable": {
            "name": "arena_width",
            "grid": list(WIDTH_GRID),
            "reference_width": REFERENCE_WIDTH,
            "note": (
                "width also governs the deposit-time split budget "
                "(mountable_budget = arena.width - recency_reserve), so nodes "
                "split differently per width. That is part of the "
                "measurement, not a confound; split counts are receipted."),
        },
        "batteries": {
            "sup": {
                "probes": 9,
                "harness": "scripts/lsr_p2c_replay_gpu.py (arm 1)",
                "fixtures": [
                    "correction_then_restatement", "fresh_fact_controls",
                    "multi_hop_a_b_c", "short_correction_long_competitor"],
            },
            "census": {
                "probes": 10,
                "harness": "scripts/lsr_p2c_e2e_gpu.py (arm 1, sharded)",
                "shards": ["e2e-1", "e2e-2", "e2e-3", "e2e-4"],
            },
            "longhorizon": {
                "probes": 4,
                "harness": "scripts/grm_eb1_longhorizon_gpu.py",
                "shards": ["lh-1", "lh-2", "lh-3", "lh-4", "lh-5"],
                "scored": (
                    "the 4 spot probes at distances 36-60 from EB1 G5, same "
                    "registered generator and seed (GENERATOR_SEED=20260903, "
                    "TARGET_TURNS=104, PROBE_EVERY=10, MIN_DISTANCE=30)"),
            },
        },
        "batteries_order": list(BATTERIES),
        "levers": dict(LEVERS),
        "env": {
            "GRM_LSR_FIXES": "1",
            "GRM_CAPTURE_PIN": "live",
            "GRM_SEAT_NEAR_LIVE": "1",
            "GRM_PERSISTENT_BOAT": "(unset — the EB1 spec/ephemeral frame)",
            "demand": "OFF (core/grm_demand.py not engaged)",
        },
        "seeds": {
            "longhorizon_generator_seed": 20260903,
            "note": (
                "Every battery is greedy/deterministic; the only seed in the "
                "system is the long-horizon script generator's, reused "
                "unchanged from EB1 so the probe set is identical per width."),
        },
        "method": {
            "width_injection": (
                "A derived copy of the frozen DET1 runtime frame with exactly "
                "one field changed (resolved_flags.arena_width), written to "
                "artifacts/grm_wc1/frames/. The read-only harnesses' "
                "module-level RUNTIME_FRAME is rebound to it for the duration "
                "of one battery run; the census and long-horizon drivers pass "
                "it straight through as --arena-width. The sup replay's arena "
                "is built by grm_det1_2_gpu._load_model_repo, which "
                "hard-codes arena_width=96, so that function is WRAPPED (not "
                "edited) to rewrite the one keyword."),
            "single_variable_check": (
                "grm_wc1_common.unchanged_fields machine-checks that the "
                "derived frame differs from the frozen frame in the width "
                "field alone; the check result rides on every receipt."),
            "no_production_change": (
                "core/ and config/ are untouched. Only scripts/grm_wc1_*.py, "
                "tests/, artifacts/grm_wc1/ and logs/ are written."),
        },
        "predictions": [dict(p) for p in PREDICTIONS],
        "gates": {
            "G1": (
                "python3 -m pytest -q, sharded under 10 min per call; the "
                "pre-existing set unchanged vs the RS4 G1 log in the main "
                "checkout, plus new CPU tests for the sweep driver's pure "
                "parts (frame derivation, table assembly, regression-vs-96)."),
            "G2": (
                "Width 96 reproduces RS3 Part 4 + RT1: the sup, census and "
                "long-horizon columns and the served text (semantic "
                "comparator). A non-reproducing 96 STOPS the sweep."),
            "G3": (
                "The 5-widths x 3-batteries sweep table, predictions "
                "hit/miss, and a one-paragraph reading of the curve."),
        },
        "g2_reproduction_targets": {
            "source_receipt": {
                "path": str(RS3_PART4),
                "sha256": sha256_bytes(RS3_PART4.read_bytes()),
                "bytes": RS3_PART4.stat().st_size,
            },
            "measured_on_disk": {
                "sup": f"{part4['G4_sup']['correct']}/9",
                "census": (f"{part4['G4_census']['correct']}/"
                           f"{part4['G4_census']['total']}"),
                "longhorizon_spot_checks":
                    part4["G4_longhorizon"]["spot_check_stayed_correct"],
            },
            "order_text": {
                "sup": "9/9", "census": "9/10", "longhorizon": "4/4"},
            "declared_ambiguity": (
                "The order's G2 line asks for sup 9/9. RS3 Part 4 ON DISK "
                "records sup 8/9 (its own honest_note explains the residual: "
                "sup_solace_fresh, the routing/split confound RS1 named). "
                "There is no 9/9 sup receipt at width 96 to reproduce, so "
                "G2's sup criterion is REPRODUCTION OF THE RECEIPT (8/9, same "
                "probe-by-probe verdicts, same served text), not the order's "
                "9/9. Registered before the gate ran so this is a declared "
                "reading of the order, not a threshold moved after seeing "
                "results. A width-96 run that lands 8/9 with the same probe "
                "identities REPRODUCES; anything else is reported RED."),
        },
        "columns_reported_per_width": [
            "sup correct / 9",
            "census correct / 10",
            "longhorizon spot probes correct / 4",
            "regressions vs width 96 (per battery, by probe id)",
            "mean mounted mass at readout (optional column; if the "
            "LayerTypeMassObserver does not attach cleanly it is reported "
            "null and does NOT block the gate)",
            "number of split nodes (width-guard parents + children)",
            "mean resident seats per turn",
            "mean wall ms per turn",
        ],
        "gpu_conventions": {
            "lock": "/tmp/forge-gpu.lock via scripts.grm_cmc1_gpu_arms.gpu_lease",
            "lease_seconds": 580,
            "hard_cap_seconds": 590,
            "inter_run_gap_seconds": 30,
            "single_gpu": True,
            "contention": (
                "ANOTHER SEAT runs this same order in a parallel worktree. "
                "The lock is WAITED ON, never cleared. No process this seat "
                "did not start is ever killed, signalled or interrupted."),
        },
        "immutability": (
            "This file is written once and never rewritten. Amendments go to "
            "artifacts/grm_wc1/amendment_*.json citing this file's sha256."),
        "sources": {
            "grm_wc1_common": file_record(
                ROOT / "scripts" / "grm_wc1_common.py"),
            "grm_wc1_registration": file_record(Path(__file__).resolve()),
            "graft_arena": file_record(ROOT / "core" / "graft_arena.py"),
            "grm_admission": file_record(ROOT / "core" / "grm_admission.py"),
            "graft_repository": file_record(
                ROOT / "core" / "graft_repository.py"),
            "grm_frame": file_record(ROOT / "core" / "grm_frame.py"),
            "lsr_p2c_replay_gpu": file_record(
                ROOT / "scripts" / "lsr_p2c_replay_gpu.py"),
            "lsr_p2c_e2e_gpu": file_record(
                ROOT / "scripts" / "lsr_p2c_e2e_gpu.py"),
            "grm_eb1_longhorizon_gpu": file_record(
                ROOT / "scripts" / "grm_eb1_longhorizon_gpu.py"),
            "grm_det1_2_gpu": file_record(
                ROOT / "scripts" / "grm_det1_2_gpu.py"),
            "native_library": file_record(
                ROOT / "cpp" / "build" / "libgrm_runtime.so"),
            "frozen_runtime_frame": file_record(FROZEN_FRAME),
            "frozen_census": file_record(FROZEN_CENSUS),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if REGISTRATION.exists():
        body = REGISTRATION.read_bytes()
        raise WC1Error(
            "registration already written and is IMMUTABLE: "
            f"{REGISTRATION} sha256={sha256_bytes(body)} — amend instead")
    body = canonical_json_bytes(build_registration())
    REGISTRATION.write_bytes(body)
    print(f"registration={REGISTRATION}")
    print(f"sha256={sha256_bytes(body)}")
    print(f"bytes={len(body)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
