#!/usr/bin/env python3
"""GRM-WC1 — shared, PURE parts of the arena-width sweep driver.

Program: GRM (Graft Repository Memory), order
``orders/GRM_WC1_ARENA_WIDTH_CURVE.md``.

WHY A WIDTH-OVERRIDDEN FRAME AND NOT A NEW HARNESS.
All three registered batteries read their run flags from ONE place — the
frozen DET1 runtime frame ``resolved_flags``:

  * ``lsr_p2c_replay_gpu.serve_fixture``   -> ``flags`` (sup battery), and the
    arena width it constructs comes from ``grm_det1_2_gpu._load_model_repo``,
    which HARD-CODES ``arena_width=96``;
  * ``lsr_p2c_e2e_gpu.run_shard``          -> ``--arena-width flags[...]``;
  * ``grm_eb1_longhorizon_gpu.run_shard``  -> ``--arena-width flags[...]``.

Those modules are READ-ONLY under this order's file boundary, so WC1 does not
edit them.  It builds a COPY of the frozen frame with exactly one field
changed — ``resolved_flags.arena_width`` — writes it under
``artifacts/grm_wc1/frames/``, and rebinds the module-level ``RUNTIME_FRAME``
constant to it for the duration of one battery run.  Everything else about
the run is byte-identical to the certified frame, and the receipt carries both
frames' sha256 so the single-variable claim is checkable rather than asserted.

For the sup battery the frame is not enough — ``_load_model_repo`` ignores the
frame's width entirely — so the sweep driver wraps that function and rewrites
the one keyword.  It is a wrapper, not an edit: the read-only module is
imported unchanged and the wrapper is installed on the importing module's
namespace for the duration of the call.

NO TUNING LITERALS LIVE HERE beyond the registered width grid, which IS the
measurement's independent variable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACT_DIR = ROOT / "artifacts" / "grm_wc1"
FRAME_DIR = ARTIFACT_DIR / "frames"
SCHEMA_PREFIX = "grm.wc1"

#: The registered width grid — the ONLY numeric constants this order adds.
#: ``orders/GRM_WC1_ARENA_WIDTH_CURVE.md``: "Sweep arena_width in
#: {64, 96, 128, 192, 256}".
WIDTH_GRID: tuple[int, ...] = (64, 96, 128, 192, 256)

#: The reference width every other width is compared against: the reliability
#: harness default, and the width RS3 Part 4 / RT1 were measured at.
REFERENCE_WIDTH = 96

#: The three batteries, in the order the order names them.
BATTERIES: tuple[str, ...] = ("sup", "census", "longhorizon")

#: The registered lever pair (RS3's winning column) plus the spec frame.
LEVERS: Mapping[str, Any] = {
    "capture_pin": "live",
    "seat_near_live": True,
    "lsr_fixes": True,
    "frame": "ephemeral (EB1 spec frame, constructor default)",
    "demand": "off",
}


class WC1Error(RuntimeError):
    """The WC1 sweep could not be performed as registered."""


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The width-overridden frame
# ---------------------------------------------------------------------------

def width_frame_payload(base_frame: Mapping[str, Any], width: int
                        ) -> dict[str, Any]:
    """The frozen frame with ``resolved_flags.arena_width`` set to ``width``.

    EXACTLY ONE FIELD MOVES.  A ``wc1_override`` block is added beside it that
    names the field, the frozen value and the new one, so a reader of the
    derived frame can never mistake it for the certified frame itself.
    """
    width = int(width)
    if width <= 0:
        raise WC1Error(f"arena width must be positive, got {width}")
    payload = json.loads(json.dumps(base_frame))  # deep copy, JSON-only data
    flags = payload.get("resolved_flags")
    if not isinstance(flags, dict) or "arena_width" not in flags:
        raise WC1Error("frozen frame has no resolved_flags.arena_width")
    frozen_width = int(flags["arena_width"])
    flags["arena_width"] = width
    payload["wc1_override"] = {
        "order": "orders/GRM_WC1_ARENA_WIDTH_CURVE.md",
        "field": "resolved_flags.arena_width",
        "frozen_value": frozen_width,
        "override_value": width,
        "note": (
            "DERIVED frame, not the certified DET1 frame. Exactly one field "
            "differs; everything else is copied byte-for-byte from the "
            "frozen frame this was derived from."),
    }
    return payload


def unchanged_fields(base_frame: Mapping[str, Any],
                     derived: Mapping[str, Any]) -> bool:
    """True when ``derived`` differs from ``base_frame`` in exactly the width.

    The single-variable claim, MACHINE-CHECKED rather than asserted: strip the
    override block and reset the width, and the two payloads must be equal.
    """
    probe = json.loads(json.dumps(derived))
    probe.pop("wc1_override", None)
    flags = probe.get("resolved_flags")
    if not isinstance(flags, dict):
        return False
    flags["arena_width"] = int(base_frame["resolved_flags"]["arena_width"])
    return probe == json.loads(json.dumps(base_frame))


# ---------------------------------------------------------------------------
# Table assembly and the regression-vs-96 computation
# ---------------------------------------------------------------------------

def probe_row(probe_id: str, correct: bool, *, served: str = "",
              expected: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "probe_id": str(probe_id),
        "correct": bool(correct),
        "served_answer": str(served),
        "expected_values": [str(v) for v in expected],
    }


def battery_result(width: int, battery: str,
                   rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """One (width, battery) cell: the rows plus their correct count."""
    if battery not in BATTERIES:
        raise WC1Error(f"unknown battery {battery!r}")
    ordered = [dict(r) for r in rows]
    return {
        "width": int(width),
        "battery": str(battery),
        "rows": ordered,
        "correct": sum(1 for r in ordered if r.get("correct")),
        "total": len(ordered),
    }


def regressions_vs_reference(
    cell: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Which probes this width loses, and which it recovers, vs width 96.

    A REGRESSION is a probe the reference width serves correctly and this
    width does not.  A RECOVERY is the converse.  Probes present in one cell
    and not the other are reported as ``missing`` rather than scored — a
    silent set difference would fake a regression count.
    """
    ref_rows = {str(r["probe_id"]): bool(r.get("correct"))
                for r in reference.get("rows", ())}
    this_rows = {str(r["probe_id"]): bool(r.get("correct"))
                 for r in cell.get("rows", ())}
    shared = sorted(set(ref_rows) & set(this_rows))
    regressions = [p for p in shared if ref_rows[p] and not this_rows[p]]
    recoveries = [p for p in shared if not ref_rows[p] and this_rows[p]]
    missing_here = sorted(set(ref_rows) - set(this_rows))
    missing_there = sorted(set(this_rows) - set(ref_rows))
    return {
        "battery": str(cell.get("battery")),
        "width": int(cell.get("width", 0)),
        "reference_width": int(reference.get("width", 0)),
        "regressions": regressions,
        "recoveries": recoveries,
        "regression_count": len(regressions),
        "recovery_count": len(recoveries),
        "delta_correct": int(cell.get("correct", 0)) - int(
            reference.get("correct", 0)),
        "missing_from_this_width": missing_here,
        "absent_from_reference": missing_there,
        "comparable": not missing_here and not missing_there,
    }


def sweep_table(cells: Mapping[int, Mapping[str, Mapping[str, Any]]],
                *, reference_width: int = REFERENCE_WIDTH
                ) -> list[dict[str, Any]]:
    """The 5-widths x 3-batteries table, one row per width.

    ``cells`` is ``{width: {battery: battery_result(...)}}``.  A width or a
    battery that was not run appears as ``None`` rather than as a zero, so a
    missing run can never read as a failing run.
    """
    rows: list[dict[str, Any]] = []
    for width in sorted(cells):
        row: dict[str, Any] = {"width": int(width)}
        for battery in BATTERIES:
            cell = cells[width].get(battery)
            key = f"{battery}_regressions_vs_{reference_width}"
            if cell is None:
                row[battery] = None
                row[key] = None
                continue
            row[battery] = {
                "correct": int(cell["correct"]),
                "total": int(cell["total"]),
                "score": f"{int(cell['correct'])}/{int(cell['total'])}",
            }
            ref_cell = cells.get(reference_width, {}).get(battery)
            if int(width) == int(reference_width):
                row[key] = []
            elif ref_cell is None:
                row[key] = None
            else:
                row[key] = regressions_vs_reference(cell, ref_cell)[
                    "regressions"]
        rows.append(row)
    return rows


def split_census(nodes: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count width-guard split parents and children in a node list.

    Reads ONLY the flags the existing split writers set
    (``graft_repository._guard_deposit_width`` and
    ``graft_arena._split_unseatable``); WC1 adds no metadata of its own.
    """
    parents = children = ephemeral = 0
    total = 0
    for node in nodes:
        total += 1
        meta = node.get("metadata") or {}
        if meta.get("width_guard_parent") or node.get("ephemeral_split"):
            parents += 1
        if (meta.get("width_guard_child")
                or node.get("ephemeral_split_of") is not None):
            children += 1
        if (node.get("ephemeral_split")
                or node.get("ephemeral_split_of") is not None):
            ephemeral += 1
    return {
        "nodes": total,
        "split_parents": parents,
        "split_children": children,
        "ephemeral_split_members": ephemeral,
        "split_nodes": parents + children,
    }


def mean_or_none(values: Sequence[Any]) -> float | None:
    """The mean, or ``None`` for an empty sample — never 0.0 for 'no data'."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def prediction_verdicts(
    table: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    cells: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Score each registered prediction against the measured table.

    Every prediction carries its own ``check`` name; unknown checks are
    recorded UNSCORED rather than silently passed.
    """
    by_width = {int(r["width"]): r for r in table}
    out: list[dict[str, Any]] = []
    for pred in predictions:
        check = str(pred.get("check", ""))
        if check == "96_not_optimum":
            verdict, evidence = _check_96_not_optimum(by_width)
        elif check == "64_loses_two_sup":
            verdict, evidence = _check_64_sup(by_width)
        elif check == "256_yarn_wall":
            verdict, evidence = _check_256_wall(by_width, cells)
        elif check == "longhorizon_4_of_4":
            verdict, evidence = _check_longhorizon(by_width)
        else:
            verdict = "UNSCORED"
            evidence = {"reason": f"unknown check {check!r}"}
        out.append({
            "id": str(pred.get("id", "")),
            "check": check,
            "claim": str(pred.get("claim", "")),
            "verdict": verdict,
            "evidence": evidence,
        })
    return out


def _check_96_not_optimum(by_width: Mapping[int, Mapping[str, Any]]):
    ref = by_width.get(REFERENCE_WIDTH)
    ok = ref is not None
    ev: dict[int, Any] = {}
    for width in (128, 192):
        row = by_width.get(width)
        if row is None or ref is None:
            ok = False
            ev[width] = None
            continue
        per: dict[str, Any] = {}
        for battery in BATTERIES:
            cell = row.get(battery)
            rcell = ref.get(battery)
            if cell is None or rcell is None:
                ok = False
                per[battery] = None
                continue
            matched = int(cell["correct"]) >= int(rcell["correct"])
            per[battery] = {
                "width": cell["score"],
                "ref96": rcell["score"],
                "matched_or_beat": matched,
            }
            if not matched:
                ok = False
        ev[width] = per
    return ("HIT" if ok else "MISS"), ev


def _check_64_sup(by_width: Mapping[int, Mapping[str, Any]]):
    row = by_width.get(64)
    ref = by_width.get(REFERENCE_WIDTH)
    if row is None or ref is None or row.get("sup") is None or ref.get(
            "sup") is None:
        return "UNSCORED", {"reason": "width 64 or 96 sup absent"}
    regs = row.get(f"sup_regressions_vs_{REFERENCE_WIDTH}") or []
    return ("HIT" if len(regs) >= 2 else "MISS"), {
        "sup_64": row["sup"]["score"],
        "sup_96": ref["sup"]["score"],
        "correct_lost": int(ref["sup"]["correct"]) - int(row["sup"]["correct"]),
        "regressions_vs_96": regs,
    }


def _check_256_wall(by_width: Mapping[int, Mapping[str, Any]],
                    cells: Mapping[int, Mapping[str, Mapping[str, Any]]]):
    row = by_width.get(256)
    ref = by_width.get(REFERENCE_WIDTH)
    if row is None or ref is None:
        return "UNSCORED", {"reason": "width 256 or 96 absent"}
    regs = row.get(f"census_regressions_vs_{REFERENCE_WIDTH}") or []
    wall_256 = (cells.get(256, {}).get("census") or {}).get(
        "mean_wall_ms_per_turn")
    wall_96 = (cells.get(REFERENCE_WIDTH, {}).get("census") or {}).get(
        "mean_wall_ms_per_turn")
    wall_rose = (None if wall_256 is None or wall_96 is None
                 else float(wall_256) > float(wall_96))
    return ("HIT" if (regs and wall_rose) else "MISS"), {
        "census_regressions_vs_96": list(regs),
        "mean_wall_ms_per_turn_256": wall_256,
        "mean_wall_ms_per_turn_96": wall_96,
        "wall_rose": wall_rose,
    }


def _check_longhorizon(by_width: Mapping[int, Mapping[str, Any]]):
    ev: dict[int, Any] = {}
    ok = True
    for width in sorted(by_width):
        if int(width) < REFERENCE_WIDTH:
            continue
        cell = by_width[width].get("longhorizon")
        ev[int(width)] = None if cell is None else cell["score"]
        if cell is None or int(cell["correct"]) != int(cell["total"]):
            ok = False
    return ("HIT" if ok else "MISS"), ev
