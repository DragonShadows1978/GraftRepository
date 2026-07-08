#!/usr/bin/env python3
"""P3 receipt (b): recall equality — packed format-2 banks vs the sweep's
already-recorded dequantized-dir margins (sweep_depth8/sweep_depth6). The
noise band is 0.0 (deterministic engine, SWEEP_RESULTS.md), so this is a
HARD equality gate, not a tolerance check: every mount margin from the gate
run against a REAL packed bank must equal the corresponding
sweep_depth<bits>/<gate>_gate.json margin bit-for-bit.

Usage:
    grm_graft_quant_p3_recall_receipt.py --gate multifact --bits 8 \
        --packed-json artifacts/grm_graft_quant/P3/multifact_packed_8bit_gate.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
QUANT_DIR = REPO_ROOT / "artifacts" / "grm_graft_quant"

from scripts.grm_graft_quant_sweep import extract_margins  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate", required=True, choices=("multifact", "preference", "supersession", "addressing_fact_local"))
    p.add_argument("--bits", type=int, required=True, choices=(8, 6, 4, 3))
    p.add_argument("--packed-json", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sweep_json = QUANT_DIR / f"sweep_depth{args.bits}" / f"{args.gate}_gate.json"
    if not sweep_json.exists():
        raise SystemExit(f"missing sweep reference: {sweep_json}")

    sweep_margins = extract_margins(sweep_json)
    packed_margins = extract_margins(args.packed_json)

    sweep_rows = {r["id"]: r for r in sweep_margins["rows"]}
    packed_rows = {r["id"]: r for r in packed_margins["rows"]}

    all_equal = True
    comparisons = []
    for item_id, sweep_row in sweep_rows.items():
        packed_row = packed_rows.get(item_id)
        eq = (packed_row is not None
              and packed_row.get("mount_margin") == sweep_row.get("mount_margin")
              and packed_row.get("mount_hit") == sweep_row.get("mount_hit"))
        if not eq:
            all_equal = False
        comparisons.append({
            "id": item_id,
            "sweep_mount_margin": sweep_row.get("mount_margin"),
            "packed_mount_margin": (packed_row or {}).get("mount_margin"),
            "sweep_mount_hit": sweep_row.get("mount_hit"),
            "packed_mount_hit": (packed_row or {}).get("mount_hit"),
            "equal": eq,
        })

    report = {
        "gate": args.gate,
        "bits": args.bits,
        "sweep_reference": str(sweep_json),
        "packed_artifact": str(args.packed_json),
        "sweep_classification": sweep_margins.get("classification"),
        "packed_classification": packed_margins.get("classification"),
        "classification_equal": sweep_margins.get("classification") == packed_margins.get("classification"),
        "all_margins_equal": all_equal,
        "comparisons": comparisons,
    }
    print(json.dumps(report, indent=2))
    return 0 if all_equal else 1


if __name__ == "__main__":
    sys.exit(main())
