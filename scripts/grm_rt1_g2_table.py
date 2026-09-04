"""GRM-RT1 G2 — assemble the sup-battery table from the replay receipts.

The replay writes one receipt per session per run, named by session and by the
RS3 lever pair, but NOT by the RT1 switch (``GRM_RT1_RULE`` is this order's
measurement escape and the replay is read-only under the file boundary).  So
the arm each receipt belongs to is supplied HERE, explicitly, by path — the
run log is the provenance, not a guess from the filename.

Writes ``artifacts/grm_rt1/grm_rt1_g2_results.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACT_DIR = ROOT / "artifacts" / "grm_rt1"
RECEIPTS = ROOT / "artifacts" / "lsr_p2c"

#: The registered probe order of the sup battery.
PROBE_ORDER = (
    "sup_harbor_restatement",
    "sup_reserve_juniper_pass",
    "sup_praxis_fresh",
    "sup_solace_fresh",
    "sup_reserve_tundra_ledger",
    "sup_lumen_head",
    "sup_reserve_meridian_docket",
    "sup_orion_current",
    "sup_reserve_falcon_registry",
)

#: Receipts by arm, named explicitly from the run log.  Both arms ran with
#: GRM_LSR_FIXES=1 and the RS3 pair ON (capture_pin=live, seat_near_live=1);
#: the ONLY difference is GRM_RT1_RULE, the single-variable control recorded
#: in artifacts/grm_rt1/amendment_a1_rt1_only_switch.json.
ARMS: dict[str, dict[str, Any]] = {
    "rule_on": {
        "levers": {
            "GRM_LSR_FIXES": "1",
            "GRM_RT1_RULE": "(unset - the rule follows the family switch)",
            "capture_pin": "live",
            "seat_near_live": True,
        },
        "receipts": [
            "lsr_p2c_arm1_correction_then_restatement_pin-live_seat-1"
            "_0d437009d0b80b06.json",
            "lsr_p2c_arm1_fresh_fact_controls_pin-live_seat-1"
            "_23d9412706944b8c.json",
            "lsr_p2c_arm1_multi_hop_a_b_c_pin-live_seat-1"
            "_21b6c4f7d82bf11e.json",
            "lsr_p2c_arm1_short_correction_long_competitor_pin-live_seat-1"
            "_56d55caee1977cf3.json",
        ],
    },
    "rule_off": {
        "levers": {
            "GRM_LSR_FIXES": "1",
            "GRM_RT1_RULE": "0",
            "capture_pin": "live",
            "seat_near_live": True,
        },
        "receipts": [
            "lsr_p2c_arm1_correction_then_restatement_pin-live_seat-1"
            "_c86ef93b688d49ae.json",
            "lsr_p2c_arm1_fresh_fact_controls_pin-live_seat-1"
            "_f2948d64622cd3b7.json",
            "lsr_p2c_arm1_multi_hop_a_b_c_pin-live_seat-1"
            "_3e99b6093210b95d.json",
            "lsr_p2c_arm1_short_correction_long_competitor_pin-live_seat-1"
            "_c9088b4c80bb814a.json",
        ],
    },
}


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_record(path: Path) -> dict[str, Any]:
    data = Path(path).read_bytes()
    return {"path": str(Path(path).relative_to(ROOT)), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def collect(arm: str) -> dict[str, Any]:
    spec = ARMS[arm]
    rows: dict[str, dict[str, Any]] = {}
    records = []
    for name in spec["receipts"]:
        path = RECEIPTS / name
        if not path.is_file():
            raise SystemExit(f"{arm}: receipt missing on disk: {path}")
        records.append(_file_record(path))
        for probe in _read(path)["probes"]:
            info = probe["info"]
            rows[probe["probe_id"]] = {
                "probe_id": probe["probe_id"],
                "correct": bool(probe["verdict"]["correct"]),
                "served_answer": probe["served_answer"],
                "ranking_ids": [int(v) for v in info.get("ranking_ids", ())],
                "admission_rank_plan": [
                    int(v) for v in info.get("admission_rank_plan", ())],
                "admission_policy_branch": info.get(
                    "admission_policy_branch"),
                "fit_planned": [int(v) for v in info.get("fit_planned", ())],
                "fit_seated": [int(v) for v in info.get("fit_seated", ())],
                "fit_split_parent": info.get("fit_split_parent"),
                "fit_split_children": [
                    int(v) for v in info.get("fit_split_children", ())],
            }
    missing = [p for p in PROBE_ORDER if p not in rows]
    if missing:
        raise SystemExit(f"{arm}: probes missing from the receipts: {missing}")
    table = [rows[p] for p in PROBE_ORDER]
    return {
        "arm": arm,
        "levers": spec["levers"],
        "receipts": records,
        "total": len(table),
        "correct": sum(1 for r in table if r["correct"]),
        "incorrect_probes": [
            r["probe_id"] for r in table if not r["correct"]],
        "table": table,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=ARTIFACT_DIR / "grm_rt1_g2_results.json")
    args = parser.parse_args()

    on = collect("rule_on")
    off = collect("rule_off")
    deltas = []
    for row_on, row_off in zip(on["table"], off["table"]):
        if row_on["correct"] != row_off["correct"]:
            deltas.append({
                "probe_id": row_on["probe_id"],
                "off_correct": row_off["correct"],
                "on_correct": row_on["correct"],
                "direction": (
                    "RECOVERED" if row_on["correct"] else "REGRESSED"),
                "ranking_off": row_off["ranking_ids"],
                "ranking_on": row_on["ranking_ids"],
                "plan_off": row_off["admission_rank_plan"],
                "plan_on": row_on["admission_rank_plan"],
                "branch_off": row_off["admission_policy_branch"],
                "branch_on": row_on["admission_policy_branch"],
            })

    out = {
        "schema": "grm.rt1.g2.v1",
        "program": "GRM",
        "phase": "RT1",
        "gate": "G2-sup",
        "order": "orders/GRM_RT1_SPLIT_CHILD_ROUTING.md",
        "registration": _file_record(ARTIFACT_DIR / "registration.json"),
        "amendment": _file_record(
            ARTIFACT_DIR / "amendment_a1_rt1_only_switch.json"),
        "registered_prediction": (
            "rule ON 9/9 (solace recovers, nothing regresses); "
            "rule OFF reproduces RS3's 8/9"),
        "prediction_hit": bool(
            on["correct"] == 9 and off["correct"] == 8
            and off["incorrect_probes"] == ["sup_solace_fresh"]
            and all(d["direction"] == "RECOVERED" for d in deltas)),
        "rule_on": on,
        "rule_off": off,
        "deltas": deltas,
        "regressions": [
            d for d in deltas if d["direction"] == "REGRESSED"],
        "control_note": (
            "Both arms ran GRM_LSR_FIXES=1 with the RS3 pair ON, so the "
            "fit-time split is present in BOTH.  The only difference is "
            "GRM_RT1_RULE - a single-variable control.  Turning off "
            "GRM_LSR_FIXES instead would have removed the split as well and "
            "the comparison would have carried two changes at once."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")

    print(f"G2 rule ON : {on['correct']}/{on['total']}"
          f"  misses={on['incorrect_probes']}")
    print(f"G2 rule OFF: {off['correct']}/{off['total']}"
          f"  misses={off['incorrect_probes']}")
    print(f"prediction_hit={out['prediction_hit']}  "
          f"regressions={len(out['regressions'])}")
    print()
    print(f"{'probe_id':<30} {'OFF':<6} {'ON':<6} "
          f"{'ranking OFF':<18} {'ranking ON':<18} "
          f"{'plan OFF':<12} {'plan ON':<10}")
    for row_on, row_off in zip(on["table"], off["table"]):
        print(f"{row_on['probe_id']:<30} "
              f"{str(row_off['correct']):<6} {str(row_on['correct']):<6} "
              f"{str(row_off['ranking_ids']):<18} "
              f"{str(row_on['ranking_ids']):<18} "
              f"{str(row_off['admission_rank_plan']):<12} "
              f"{str(row_on['admission_rank_plan']):<10}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
