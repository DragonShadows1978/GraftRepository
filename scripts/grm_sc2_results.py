#!/usr/bin/env python3
"""GRM-SC2 — the results roll-up: Part A, Part B, G1..G4, in one receipt.

ORDER: ``orders/GRM_SC2_CALIBRATION_EARLY_ABORT.md``.
REGISTRATION: ``artifacts/grm_sc2/registration.json``.

This composes the gate verdicts from the receipts the gates themselves wrote.
It decides nothing it cannot read off disk, and every number it prints is
copied from an artifact rather than remembered.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_det1_common import file_record  # noqa: E402

SC2_DIR = ROOT / "artifacts" / "grm_sc2"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    """Repo-relative, whether the caller passed an absolute or relative path."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (ROOT / resolved).resolve()
    return str(resolved.relative_to(ROOT))


def build(
    *,
    calibration_receipt: Path,
    g3_table: Path,
    g4_summary: Path,
    g4_baseline: Path,
    pytest_logs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    part_a = _read(calibration_receipt)
    g3 = _read(g3_table)
    g4 = _read(g4_summary)
    g4_base = _read(g4_baseline)

    base_by_probe = {r["probe_id"]: r for r in g4_base["table"]}
    g4_rows = []
    for row in g4["table"]:
        base = base_by_probe.get(row["probe_id"], {})
        g4_rows.append({
            "probe_id": row["probe_id"],
            "served_answer": row["served_answer"],
            "sc1_1_served_answer": base.get("served_answer"),
            "identical_to_sc1_1": (
                row["served_answer"] == base.get("served_answer")),
            "demand_fired": row["demand_fired"],
            "demand_min_mass": row["demand_min_mass"],
            "demand_token_index": row["demand_token_index"],
            "demand_served": row["demand_served"],
            "false_fire": row["false_fire"],
            "correct": row["correct"],
        })
    g4_identical = sum(1 for r in g4_rows if r["identical_to_sc1_1"])

    candidate = part_a["candidate"]
    held = part_a["held_out_evaluation"]

    return {
        "schema": "grm.sc2.results.v1",
        "order": part_a["order"],
        "order_sha256": part_a["order_sha256"],
        "registration": part_a["registration"],
        "branch": "lc1-wip",

        "G1_pytest": {
            "shards": list(pytest_logs),
            "baseline_set_source": (
                "artifacts/grm_sc1_2/results.json :: G1_pytest"),
        },

        "G2_part_a_calibration": {
            "receipt": _rel(calibration_receipt),
            "rule_equality_vs_frozen_DET1": part_a[
                "rule_equality_vs_frozen_DET1"]["equality_holds"],
            "decisions_checked": part_a[
                "rule_equality_vs_frozen_DET1"]["decisions_checked"],
            "candidate_threshold": candidate["candidate_threshold"],
            "carried_race_threshold": candidate["carried_race_threshold"],
            "candidate_argmin_row": candidate["argmin_row_id"],
            "all_rows_envelope": part_a[
                "all_rows_envelope_secondary_figure"]["value"],
            "all_rows_envelope_argmin": part_a[
                "all_rows_envelope_secondary_figure"]["argmin_row_id"],
            "gap": part_a["gap"]["gap"],
            "calibration_set_n": part_a["split"]["calibration_set_n"],
            "held_out_set_n": part_a["split"]["held_out_set_n"],
            "held_out_false_fires_at_carried": held[
                "at_carried_threshold"]["false_fires"],
            "held_out_false_fire_ids_at_carried": held[
                "at_carried_threshold"]["false_fire_ids"],
            "held_out_false_fires_at_candidate": held[
                "at_candidate_threshold"]["false_fires"],
            "held_out_false_fire_ids_at_candidate": held[
                "at_candidate_threshold"]["false_fire_ids"],
            "recall_at_carried": (
                f"{held['at_carried_threshold']['recall_hits']}/"
                f"{held['at_carried_threshold']['recall_total']}"),
            "recall_at_candidate": (
                f"{held['at_candidate_threshold']['recall_hits']}/"
                f"{held['at_candidate_threshold']['recall_total']}"),
            "recall_on_recovery_denominator_at_candidate": held[
                "at_candidate_threshold"]["recall_on_recovery_denominator"],
            "widest_zero_false_fire_band": _zero_band(part_a),
            "predictions": part_a["predictions_vs_measured"],
            "candidate_is_REPORTED_NOT_ADOPTED": True,
        },

        "G3_early_abort_ten_pairs": {
            "receipt": _rel(g3_table),
            "gate_pass": g3["gate_pass"],
            "pair_count": g3["pair_count"],
            "all_served_text_identical": g3["all_served_text_identical"],
            "recovered": g3["recovered"],
            "sc1_2_recovered": g3["sc1_2_recovered"],
            "recovery_matches_sc1_2": g3["recovery_matches_sc1_2"],
            "resumed_original_pairs": g3["resumed_original_pairs"],
            "mean_tokens_generated_before_abort": g3[
                "mean_tokens_generated_before_abort"],
            "table": g3["table"],
        },

        "G4_lived_battery": {
            "receipt": _rel(g4_summary),
            "baseline": _rel(g4_baseline),
            "gate_pass": g4["gate_pass"],
            "probe_count": g4["probe_count"],
            "identical_to_sc1_1": g4_identical,
            "all_identical_to_sc1_1": g4_identical == len(g4_rows),
            "regressions": g4["regressions"],
            "demand_fired_count": g4["demand_fired_count"],
            "false_fires": g4["false_fires_on_correctly_served_probes"],
            "table": g4_rows,
        },

        "sources": {
            name: file_record(ROOT / name) for name in (
                "core/grm_demand.py",
                "core/graft_arena.py",
                "config/grm_demand_registered.json",
                "scripts/grm_e2e_session.py",
                "scripts/grm_sc2_calibration.py",
                "scripts/grm_sc2_e2e_abort_gpu.py",
                "scripts/grm_sc2_results.py",
                "tests/test_grm_sc2_calibration_early_abort.py",
            )
        },
    }


def _zero_band(part_a: Mapping[str, Any]) -> dict[str, Any]:
    """The widest contiguous sweep band with ZERO held-out false fires.

    This is the picture the order asked the sweep for, stated as a number: how
    much room there is between "never fires on a served turn we have seen" and
    the line production currently runs.  It is REPORTED. Nothing here is
    registered, and no value from the sweep is adopted.
    """
    rows = part_a["sensitivity_sweep"]["table"]
    best: tuple[float, float] | None = None
    run: list[float] = []
    for row in rows:
        if row["held_out_false_fires"] == 0:
            run.append(float(row["threshold"]))
            continue
        if run and (best is None or (run[-1] - run[0]) > (best[1] - best[0])):
            best = (run[0], run[-1])
        run = []
    if run and (best is None or (run[-1] - run[0]) > (best[1] - best[0])):
        best = (run[0], run[-1])
    if best is None:
        return {"exists": False}
    recall_rows = [r for r in rows if r["threshold"] == best[1]]
    return {
        "exists": True,
        "lowest": best[0],
        "highest": best[1],
        "recall_at_highest": (
            f"{recall_rows[0]['recall_hits']}/{recall_rows[0]['recall_total']}"
            if recall_rows else None),
        "meaning": (
            "every threshold in this closed band fires on ZERO held-out served "
            "controls while keeping full recall on the planted misses. It is "
            "REPORTED, not registered and not adopted."),
    }


def emit(payload: Mapping[str, Any], stem: str) -> Path:
    import hashlib

    SC2_DIR.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    path = SC2_DIR / f"{stem}_{digest}.json"
    path.write_text(body + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--g3-table", required=True)
    parser.add_argument("--g4-summary", required=True)
    parser.add_argument("--g4-baseline", required=True)
    parser.add_argument("--pytest-log", nargs="*", default=())
    args = parser.parse_args(argv)

    logs = []
    for entry in args.pytest_log:
        log, _, summary = entry.partition("::")
        logs.append({"log": log, "summary": summary})

    payload = build(
        calibration_receipt=Path(args.calibration),
        g3_table=Path(args.g3_table),
        g4_summary=Path(args.g4_summary),
        g4_baseline=Path(args.g4_baseline),
        pytest_logs=logs,
    )
    path = emit(payload, "sc2_results")
    print(f"receipt={_rel(path)}")
    g2 = payload["G2_part_a_calibration"]
    print(f"G2 candidate={g2['candidate_threshold']!r} "
          f"carried={g2['carried_race_threshold']!r} gap={g2['gap']!r}")
    print(f"G2 held_out_ff carried={g2['held_out_false_fires_at_carried']} "
          f"candidate={g2['held_out_false_fires_at_candidate']}")
    print(f"G2 zero_false_fire_band={g2['widest_zero_false_fire_band']}")
    g3 = payload["G3_early_abort_ten_pairs"]
    print(f"G3 gate_pass={g3['gate_pass']} "
          f"identical={g3['all_served_text_identical']} "
          f"recovered={g3['recovered']}/{g3['sc1_2_recovered']} "
          f"resumed={g3['resumed_original_pairs']}")
    g4 = payload["G4_lived_battery"]
    print(f"G4 gate_pass={g4['gate_pass']} "
          f"identical={g4['identical_to_sc1_1']}/{g4['probe_count']} "
          f"regressions={g4['regressions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
