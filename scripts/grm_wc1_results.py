#!/usr/bin/env python3
"""GRM-WC1 G3 — assemble the sweep table from the per-cell receipts on disk.

CPU only.  Reads the receipts ``grm_wc1_sweep_gpu`` wrote, computes the
regressions vs width 96, scores the registered predictions, and emits
``artifacts/grm_wc1/grm_wc1_results.json``.

EVERY COLUMN NAMES ITS SOURCE.  Nothing here is retyped from a log:

  * correct counts        — the per-cell score receipts;
  * regressions vs 96     — ``grm_wc1_common.regressions_vs_reference`` over
                            the probe-id sets, so a differing probe set is
                            reported non-comparable rather than scored;
  * split nodes           — the repository manifests the runs persisted, read
                            through ``split_census``, which looks only at the
                            flags the existing width-guard writers set;
  * resident grafts / turn — length of ``info.mount_fitted`` (legacy key
                            ``mean_resident_seats_per_turn`` retained);
  * token seats / turn    — ``route_receipt.fit.cur_mount_n``, the serving
                            sum of mounted graft ntok, over the same turns;
                            unavailable measurements remain null;
  * wall ms per turn      — ``turn_wall_ms`` from the driver's own
                            instrumentation.jsonl;
  * mounted mass          — the OPTIONAL column. RS2's LayerTypeMassObserver
                            is not attached by any of the three registered
                            batteries, and attaching it would change the
                            serve path being measured, so this column is
                            reported ``null`` with the reason stated. Per the
                            order it does not block the gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_cmc1_mechanism import (  # noqa: E402
    canonical_json_bytes, sha256_bytes,
)
from scripts.grm_det1_common import file_record  # noqa: E402
from scripts.grm_wc1_common import (  # noqa: E402
    ARTIFACT_DIR, BATTERIES, REFERENCE_WIDTH, SCHEMA_PREFIX, WIDTH_GRID,
    WC1Error, battery_result, mean_or_none, prediction_verdicts,
    read_json, regressions_vs_reference, split_census, sweep_table,
)
from scripts.grm_wc1_registration import PREDICTIONS  # noqa: E402
from scripts.grm_wc1_sweep_gpu import (  # noqa: E402
    CENSUS_SHARDS, LONGHORIZON_SHARDS, SUP_SESSIONS,
    registration_record, run_dir_for,
)

RESULTS = ARTIFACT_DIR / "grm_wc1_results.json"

#: Why the optional mounted-mass column is null. Stated once, carried on the
#: receipt, so an absent number is never mistaken for a measured zero.
MASS_COLUMN_NOTE = (
    "NOT MEASURED. RS2's LayerTypeMassObserver wraps the arena's forward and "
    "the sliding-attention kernel; none of the three registered batteries "
    "attaches it, and attaching it inside them would change the serve path "
    "this sweep is measuring (RS3 used a SEPARATE build path, "
    "grm_rs1_read_strength_gpu._load_lived_repo, for exactly that reason). "
    "The order registers this as an optional column that must not block the "
    "gate, so it is reported null with the reason rather than approximated.")

TOKEN_SEATS_NOTE = (
    "Mean of recorded route_receipt.fit.cur_mount_n (sum of mounted ntok), "
    "over the same observations as the legacy graft-count mean. Null if any "
    "observation lacks a token sum; coverage is reported separately. "
    "Sup fixture receipts may lack this evidence. Counts exclude sink/live rows.")


def _token_seats(row: dict[str, Any]) -> int | None:
    # Prior art: WC1 turn means and ArenaCache.cur_mount_n's ntok sum
    # (project contributors, 2026). Reuse the measured sum, not list length;
    # this is a reporting repair, with no new residency estimator.
    info = row.get("info") or {}
    receipt = row.get("route_receipt") or info.get("route_receipt_record") or {}
    value = (receipt.get("fit") or {}).get("cur_mount_n")
    return None if value is None else int(value)


def _token_seat_stats(values: Sequence[int | None]) -> dict[str, Any]:
    measured = [v for v in values if v is not None]
    return {
        "mean_resident_token_seats_per_turn": (
            mean_or_none(measured) if len(measured) == len(values) else None),
        "turns_with_token_seat_measurement": len(measured),
        "token_seats_note": TOKEN_SEATS_NOTE,
    }


def _one(run_dir: Path, pattern: str) -> Path | None:
    hits = sorted(run_dir.glob(pattern))
    if not hits:
        return None
    if len(hits) > 1:
        # Content-addressed names: more than one means the cell was re-run and
        # produced a DIFFERENT payload. Refuse to pick, rather than pick the
        # newest and silently report a run nobody chose.
        raise WC1Error(
            f"{run_dir}/{pattern} matches {len(hits)} receipts: "
            + ", ".join(p.name for p in hits))
    return hits[0]


# ---------------------------------------------------------------------------
# Per-battery cell assembly
# ---------------------------------------------------------------------------

def sup_cell(width: int) -> dict[str, Any] | None:
    """The 9-probe sup column, stitched from the four fixture receipts."""
    run_dir = run_dir_for(width)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    elapsed: list[float] = []
    for session in SUP_SESSIONS:
        path = _one(run_dir, f"sup_{session}_*.json")
        if path is None:
            return None
        payload = read_json(path)
        if int(payload["arena_width_on_receipt"]) != int(width):
            raise WC1Error(
                f"{path}: arena_width_on_receipt "
                f"{payload['arena_width_on_receipt']} != width {width}")
        if not payload["frame_binding"]["single_variable_check"]:
            raise WC1Error(f"{path}: single-variable check FAILED")
        rows.extend(payload["rows"])
        sources.append(file_record(path))
        elapsed.append(float(payload["elapsed_seconds"]))
    cell = battery_result(width, "sup", rows)
    cell["sources"] = sources
    cell["fixture_elapsed_seconds"] = elapsed
    cell["reproduced_lived_count"] = sum(
        1 for r in rows if r.get("reproduced_lived"))
    # Grafts actually fitted at readout; retain the legacy field name.
    seats = [len(r.get("fit", {}).get("mount_fitted") or ()) for r in rows]
    cell["mean_resident_seats_per_turn"] = mean_or_none(seats)
    cell.update(_token_seat_stats([_token_seats(r) for r in rows]))
    cell["mean_wall_ms_per_turn"] = mean_or_none(
        [float(r.get("elapsed_ns", 0)) / 1e6 for r in rows])
    cell["mounted_mass_at_readout"] = None
    cell["mounted_mass_note"] = MASS_COLUMN_NOTE
    return cell


def _session_dirs(run_dir: Path, battery: str) -> list[Path]:
    if battery == "census":
        return [run_dir / "census_session" / "arm1" / spec / "session"
                for spec in CENSUS_SHARDS]
    return [run_dir / "lh_session" / spec / "session"
            for spec in LONGHORIZON_SHARDS]


def _turn_stats(session_dirs: Sequence[Path]) -> dict[str, Any]:
    """Wall ms, fitted grafts and token seats, from the driver's own log.

    Shards RESUME one another, so a turn a later shard replays would be
    counted twice.  Turn numbers are therefore de-duplicated across the chain:
    the first shard that logged a turn owns it.
    """
    wall: list[float] = []
    seats: list[int] = []
    token_seats: list[int | None] = []
    stages: dict[str, list[float]] = {
        "route_wall_ms": [], "deposit_wall_ms": [], "mount_wall_ms": []}
    dropped = 0
    turns_seen: set[int] = set()
    node_counts: list[int] = []
    for session in session_dirs:
        log = session / "instrumentation.jsonl"
        if not log.is_file():
            continue
        for line in log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            turn = int(row.get("turn", -1))
            if turn in turns_seen:
                continue
            turns_seen.add(turn)
            if row.get("turn_wall_ms") is not None:
                wall.append(float(row["turn_wall_ms"]))
            for key, acc in stages.items():
                if row.get(key) is not None:
                    acc.append(float(row[key]))
            info = row.get("info") or {}
            fitted = info.get("mount_fitted")
            if fitted is not None:
                seats.append(len(fitted))
                token_seats.append(_token_seats(row))
            dropped += len(info.get("mount_dropped_for_width") or ())
            if row.get("repo_node_count") is not None:
                node_counts.append(int(row["repo_node_count"]))
    return {
        "turns_counted": len(turns_seen),
        "mean_wall_ms_per_turn": mean_or_none(wall),
        # The STAGE decomposition matters because total turn wall time is
        # confounded by GPU-lock contention with the parallel seat running the
        # same order. These three are in-process GPU work, so they carry the
        # width signal the total cannot carry cleanly.
        "mean_route_ms_per_turn": mean_or_none(stages["route_wall_ms"]),
        "mean_deposit_ms_per_turn": mean_or_none(stages["deposit_wall_ms"]),
        "mean_mount_ms_per_turn": mean_or_none(stages["mount_wall_ms"]),
        "wall_ms_confound_note": (
            "mean_wall_ms_per_turn includes any stall while another seat held "
            "/tmp/forge-gpu.lock. The route/deposit/mount means are in-process "
            "GPU work and are the cleaner width signal."),
        "mean_resident_seats_per_turn": mean_or_none(seats),
        **_token_seat_stats(token_seats),
        "turns_with_a_mount_decision": len(seats),
        "mounts_dropped_for_width": dropped,
        "final_repo_node_count": max(node_counts) if node_counts else None,
    }


def _split_counts(session_dirs: Sequence[Path]) -> dict[str, Any]:
    """Width-guard split counts from the LAST shard's persisted repository."""
    for session in reversed(list(session_dirs)):
        manifest = session / "repository" / "manifest.json"
        if manifest.is_file():
            payload = read_json(manifest)
            out = split_census(payload.get("nodes") or ())
            out["manifest"] = file_record(manifest)
            return out
    return {"nodes": 0, "split_parents": 0, "split_children": 0,
            "ephemeral_split_members": 0, "split_nodes": 0,
            "manifest": None}


def census_cell(width: int) -> dict[str, Any] | None:
    run_dir = run_dir_for(width)
    path = _one(run_dir, "census_score_*.json")
    if path is None:
        return None
    payload = read_json(path)
    if not payload["frame_binding"]["single_variable_check"]:
        raise WC1Error(f"{path}: single-variable check FAILED")
    if int(payload["frame_binding"]["derived_width"]) != int(width):
        raise WC1Error(f"{path}: derived width is not {width}")
    cell = battery_result(width, "census", payload["rows"])
    cell["sources"] = [file_record(path)]
    sessions = _session_dirs(run_dir, "census")
    cell.update(_turn_stats(sessions))
    cell["splits"] = _split_counts(sessions)
    cell["mounted_mass_at_readout"] = None
    cell["mounted_mass_note"] = MASS_COLUMN_NOTE
    shard_elapsed = {}
    for spec in CENSUS_SHARDS:
        shard_path = _one(run_dir, f"census_{spec}_*.json")
        if shard_path is not None:
            shard_elapsed[spec] = float(
                read_json(shard_path)["elapsed_seconds"])
    cell["shard_elapsed_seconds"] = shard_elapsed
    return cell


def longhorizon_cell(width: int) -> dict[str, Any] | None:
    run_dir = run_dir_for(width)
    path = _one(run_dir, "lh_score_*.json")
    if path is None:
        return None
    payload = read_json(path)
    if not payload["frame_binding"]["single_variable_check"]:
        raise WC1Error(f"{path}: single-variable check FAILED")
    cell = battery_result(width, "longhorizon", payload["rows"])
    cell["sources"] = [file_record(path)]
    cell["all_probes_correct"] = int(payload["correct_count_all"])
    cell["all_probes_measured"] = int(payload["measured_count"])
    cell["recall_by_distance_bucket"] = payload["recall_by_distance_bucket"]
    sessions = _session_dirs(run_dir, "longhorizon")
    cell.update(_turn_stats(sessions))
    cell["splits"] = _split_counts(sessions)
    cell["mounted_mass_at_readout"] = None
    cell["mounted_mass_note"] = MASS_COLUMN_NOTE
    shard_elapsed = {}
    for spec in LONGHORIZON_SHARDS:
        shard_path = _one(run_dir, f"lh_{spec}_*.json")
        if shard_path is not None:
            shard_elapsed[spec] = float(
                read_json(shard_path)["elapsed_seconds"])
    cell["shard_elapsed_seconds"] = shard_elapsed
    return cell


def collect_cells() -> dict[int, dict[str, Any]]:
    cells: dict[int, dict[str, Any]] = {}
    for width in WIDTH_GRID:
        row = {
            "sup": sup_cell(width),
            "census": census_cell(width),
            "longhorizon": longhorizon_cell(width),
        }
        if any(v is not None for v in row.values()):
            cells[width] = row
    return cells


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def build_results() -> dict[str, Any]:
    cells = collect_cells()
    if REFERENCE_WIDTH not in cells:
        raise WC1Error(
            f"width {REFERENCE_WIDTH} has no receipts; there is nothing to "
            "compare the sweep against")
    table = sweep_table(cells)
    detail: dict[str, Any] = {}
    for width in sorted(cells):
        per_battery: dict[str, Any] = {}
        for battery in BATTERIES:
            cell = cells[width].get(battery)
            ref = cells[REFERENCE_WIDTH].get(battery)
            if cell is None:
                per_battery[battery] = None
                continue
            entry: dict[str, Any] = {
                "score": f"{cell['correct']}/{cell['total']}",
                "correct": cell["correct"],
                "total": cell["total"],
                "mean_wall_ms_per_turn": cell.get("mean_wall_ms_per_turn"),
                "mean_route_ms_per_turn": cell.get("mean_route_ms_per_turn"),
                "mean_deposit_ms_per_turn": cell.get(
                    "mean_deposit_ms_per_turn"),
                "mean_mount_ms_per_turn": cell.get("mean_mount_ms_per_turn"),
                "wall_ms_confound_note": cell.get("wall_ms_confound_note"),
                "mean_resident_seats_per_turn": cell.get(
                    "mean_resident_seats_per_turn"),
                "mean_resident_token_seats_per_turn": cell.get(
                    "mean_resident_token_seats_per_turn"),
                "turns_with_token_seat_measurement": cell.get(
                    "turns_with_token_seat_measurement"),
                "token_seats_note": cell.get("token_seats_note"),
                "mounts_dropped_for_width": cell.get(
                    "mounts_dropped_for_width"),
                "split_nodes": (cell.get("splits") or {}).get("split_nodes"),
                "split_parents": (cell.get("splits") or {}).get(
                    "split_parents"),
                "split_children": (cell.get("splits") or {}).get(
                    "split_children"),
                "final_repo_node_count": cell.get("final_repo_node_count"),
                "mounted_mass_at_readout": None,
                "sources": cell.get("sources", []),
            }
            if int(width) == REFERENCE_WIDTH:
                entry["vs_96"] = "reference"
            elif ref is not None:
                entry["vs_96"] = regressions_vs_reference(cell, ref)
            else:
                entry["vs_96"] = None
            if battery == "longhorizon":
                entry["all_probes"] = (
                    f"{cell['all_probes_correct']}/"
                    f"{cell['all_probes_measured']}")
                entry["recall_by_distance_bucket"] = cell[
                    "recall_by_distance_bucket"]
            per_battery[battery] = entry
        detail[str(width)] = per_battery

    predictions = prediction_verdicts(table, PREDICTIONS, cells)
    return {
        "schema": f"{SCHEMA_PREFIX}.results.v1",
        "program": "GRM",
        "phase": "WC1",
        "gate": "G3",
        "order": "orders/GRM_WC1_ARENA_WIDTH_CURVE.md",
        "reference_width": REFERENCE_WIDTH,
        "widths_measured": sorted(cells),
        "table": table,
        "detail": detail,
        "predictions": predictions,
        "mounted_mass_column": {"measured": False, "note": MASS_COLUMN_NOTE},
        "registration": registration_record(),
        "sources": {
            "grm_wc1_common": file_record(
                ROOT / "scripts" / "grm_wc1_common.py"),
            "grm_wc1_sweep_gpu": file_record(
                ROOT / "scripts" / "grm_wc1_sweep_gpu.py"),
            "grm_wc1_results": file_record(Path(__file__).resolve()),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-table", action="store_true")
    return parser.parse_args(argv)


def _fmt(value: Any, width: int = 9) -> str:
    if value is None:
        return "-".rjust(width)
    if isinstance(value, float):
        return f"{value:,.1f}".rjust(width)
    return str(value).rjust(width)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_results()
    body = canonical_json_bytes(payload)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.write_bytes(body)
    print(f"results={RESULTS}")
    print(f"sha256={sha256_bytes(body)}")

    if args.print_table:
        print()
        header = (f"{'width':>6} {'sup':>6} {'census':>8} {'lh':>6} "
                  f"{'lh_all':>7} {'splits':>7} {'grafts':>7} "
                  f"{'token_seats':>11} {'wall_ms':>9}  "
                  f"regressions vs 96")
        print(header)
        print("-" * len(header))
        for row in payload["table"]:
            width = int(row["width"])
            d = payload["detail"][str(width)]
            regs = []
            for battery in BATTERIES:
                entry = d.get(battery)
                if not entry or entry["vs_96"] in (None, "reference"):
                    continue
                for probe in entry["vs_96"]["regressions"]:
                    regs.append(f"{battery}:{probe}")
            census = d.get("census") or {}
            sup = d.get("sup") or {}
            lh = d.get("longhorizon") or {}
            print(
                f"{width:>6} {sup.get('score', '-'):>6} "
                f"{census.get('score', '-'):>8} {lh.get('score', '-'):>6} "
                f"{lh.get('all_probes', '-'):>7} "
                f"{_fmt(census.get('split_nodes'), 7)} "
                f"{_fmt(census.get('mean_resident_seats_per_turn'), 7)} "
                f"{_fmt(census.get('mean_resident_token_seats_per_turn'), 11)} "
                f"{_fmt(census.get('mean_wall_ms_per_turn'), 9)}  "
                f"{', '.join(regs) if regs else '(none)'}")
        print()
        for pred in payload["predictions"]:
            print(f"{pred['id']} {pred['verdict']:>8}  {pred['claim'][:96]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
