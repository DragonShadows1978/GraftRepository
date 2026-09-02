"""GRM-SC2 Part A — D-NGH calibration widening under the UNCHANGED envelope rule.

ORDER: ``orders/GRM_SC2_CALIBRATION_EARLY_ABORT.md``.
REGISTRATION: ``artifacts/grm_sc2/registration.json`` (written before this file ran).

WHAT THIS IS.  The GRM-DET1 race fit the production D-NGH line on TWO served
calibration turns (a thin envelope, and every receipt says so).  Since then the
stack has lived through SC1, SC1.1 and SC1.2 and has accumulated more served
control turns.  This module recomputes the line under the SAME rule over the
WIDER set, on a split registered before any computation, and REPORTS the
candidate.  Production keeps the carried race threshold; adoption is David's.

WHAT THIS IS NOT.  It is not a new rule.  ``fit_thresholds``'s D-NGH branch is
``min over served turns of (min over that turn's answer tokens of
mounted_mass)``, direction ``trigger_if_strictly_below``; that is carried here
operand for operand.  Nothing is re-derived, re-weighted, or softened.

WHY PER-TURN MINIMA ARE EXACT HERE (the persistence finding, registered).
The order expected per-token ``mounted_mass`` arrays under ``artifacts/grm_sc1*``.
They are not there: only the frozen DET1 race rows carry token arrays (plus one
SC1 bit-identity probe).  SC1.1/SC1.2 persist the per-turn MINIMUM and the fire
INDEX.  That is sufficient and EXACT for this order, because:

  * the envelope rule consumes only per-turn minima, by construction; and
  * ``detector_decision``'s D-NGH verdict for a turn is ``any token < threshold``,
    which is identically ``min(tokens) < threshold``.

The one thing minima cannot give is the fire INDEX at a threshold the row did
not already fire at.  This module therefore reports fire COUNTS and RECALL and
never invents an index.  ``rule_equality_check`` pins the minima path against
the real DET1 functions on the 22 rows that do carry arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "artifacts"
SC2_DIR = ARTIFACTS / "grm_sc2"
REGISTRATION_PATH = SC2_DIR / "registration.json"

RACE_ROOT = ARTIFACTS / "grm_det1" / "run_20260831T160525Z_2" / "det1_4" / "campaign"
RACE_CALIBRATION_ROWS = RACE_ROOT / "calibration" / "mechanistic_rows.jsonl"
RACE_EVAL_ROWS = RACE_ROOT / "eval" / "mechanistic_rows.jsonl"
CARRIED_THRESHOLDS = RACE_ROOT / "calibration" / "thresholds_2149a44b6136fd1b.json"

SC1_1_G3_SUMMARY = ARTIFACTS / "grm_sc1_1" / "sc1_1_g3_summary_979e8e70276871ab.json"
SC1_1_G3_SUMMARY_DEMAND_OFF = (
    ARTIFACTS / "grm_sc1_1" / "sc1_1_g3_summary_51eb47fdbebfbdd1.json")
SC1_1_G2_RECEIPTS = (
    ARTIFACTS / "grm_sc1_1" / "sc1_1_g2_correction_then_restatement_3b13020d92d8ff89.json",
    ARTIFACTS / "grm_sc1_1" / "sc1_1_g2_fresh_fact_controls_832fe67cd8e33a05.json",
)
SC1_2_PAIR_GLOB = "sc1_2_pair_e2e_*.json"

#: The registered sweep grid — the ONLY numeric constants this order registers.
SWEEP_START = 0.05
SWEEP_STOP = 0.45
SWEEP_STEP = 0.01
SWEEP_POINTS = 41


class CalibrationError(RuntimeError):
    """Part A could not run as registered."""


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServedRow:
    """One served-control turn: the per-turn minimum is the whole envelope input."""

    row_id: str
    probe_id: str
    min_mass: float
    provenance: str
    source: str
    fired_index_at_carried: int | None = None
    tokens: tuple[float, ...] | None = None

    def as_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "row_id": self.row_id,
            "probe_id": self.probe_id,
            "min_mass": float(self.min_mass),
            "provenance": self.provenance,
            "source": self.source,
            "has_token_array": self.tokens is not None,
        }
        if self.fired_index_at_carried is not None:
            out["fired_index_at_carried"] = int(self.fired_index_at_carried)
        if self.tokens is not None:
            out["token_count"] = len(self.tokens)
        return out


@dataclass(frozen=True)
class PlantedRow:
    row_id: str
    probe_id: str
    min_mass: float
    provenance: str
    source: str
    tokens: tuple[float, ...] | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "probe_id": self.probe_id,
            "min_mass": float(self.min_mass),
            "provenance": self.provenance,
            "source": self.source,
            "has_token_array": self.tokens is not None,
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CalibrationError(f"missing persisted rows: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CalibrationError(f"missing persisted receipt: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _race_token_masses(row: Mapping[str, Any]) -> tuple[float, ...]:
    tokens = list((row.get("signals") or {}).get("tokens") or [])
    if not tokens:
        raise CalibrationError(f"race row has no answer tokens: {row.get('row_id')}")
    return tuple(float(t["ngh"]["mounted_mass"]) for t in tokens)


def _first_below(masses: Sequence[float], threshold: float) -> int | None:
    for index, mass in enumerate(masses):
        if float(mass) < float(threshold):
            return index
    return None


# ---------------------------------------------------------------------------
# Inventory — every lived served-control turn with persisted mass
# ---------------------------------------------------------------------------


def inventory() -> dict[str, Any]:
    """Every lived served-control and planted-miss row, by provenance class.

    De-duplication is applied exactly as the registration states: the SC1.2
    Arm-0 rows reproduce their race-eval counterparts with delta 0.0, so they
    fold into the race-eval class rather than being counted twice.  The one
    genuine disagreement (``sup_solace_fresh`` under the SC1.1 G2 same-process
    fork) is carried as its own row, never collapsed.
    """
    served: dict[str, list[ServedRow]] = {}
    planted: list[PlantedRow] = []

    carried = float(_read_json(CARRIED_THRESHOLDS)["D-NGH"]["threshold"])

    # -- class R_CAL: race calibration served -------------------------------
    r_cal: list[ServedRow] = []
    for row in _read_jsonl(RACE_CALIBRATION_ROWS):
        if row.get("variant") != "served":
            continue
        masses = _race_token_masses(row)
        r_cal.append(ServedRow(
            row_id=f"{row['fixture_id']}@race_calibration",
            probe_id=str(row["fixture_id"]),
            min_mass=min(masses),
            provenance="race_calibration_lived",
            source=_rel(RACE_CALIBRATION_ROWS),
            fired_index_at_carried=_first_below(masses, carried),
            tokens=masses,
        ))
    served["R_CAL"] = sorted(r_cal, key=lambda r: r.probe_id)

    # -- classes R_EVAL / P: race eval served + planted ---------------------
    r_eval: list[ServedRow] = []
    for row in _read_jsonl(RACE_EVAL_ROWS):
        masses = _race_token_masses(row)
        if row.get("variant") == "served":
            r_eval.append(ServedRow(
                row_id=f"{row['fixture_id']}@race_eval",
                probe_id=str(row["fixture_id"]),
                min_mass=min(masses),
                provenance="race_eval_lived",
                source=_rel(RACE_EVAL_ROWS),
                fired_index_at_carried=_first_below(masses, carried),
                tokens=masses,
            ))
        elif row.get("variant") == "planted_miss":
            planted.append(PlantedRow(
                row_id=f"{row['fixture_id']}@race_eval",
                probe_id=str(row["fixture_id"]),
                min_mass=min(masses),
                provenance="race_eval_lived",
                source=_rel(RACE_EVAL_ROWS),
                tokens=masses,
            ))
    served["R_EVAL"] = sorted(r_eval, key=lambda r: r.probe_id)

    # -- class S12: SC1.2 Arm-0 served controls (dedup target) --------------
    s12: list[ServedRow] = []
    for path in sorted((ARTIFACTS / "grm_sc1_2").glob(SC1_2_PAIR_GLOB)):
        result = _read_json(path)["result"]
        arm0 = result["arms"]["served"]["arm0"]
        s12.append(ServedRow(
            row_id=f"{result['fixture_id']}@sc1_2_arm0",
            probe_id=str(result["fixture_id"]),
            min_mass=float(arm0["attempt_min_mass"]),
            provenance="cross_process_fork_reproduction_of_race_eval",
            source=_rel(path),
            fired_index_at_carried=(
                None if arm0.get("demand_token_index") is None
                else int(arm0["demand_token_index"])),
        ))
    served["S12"] = sorted(s12, key=lambda r: r.probe_id)

    # -- class S11_G3: SC1.1 lived battery served turns ---------------------
    #
    # PROVENANCE CORRECTION, measured not assumed.  The order names the
    # "demand-OFF arm" as the source of these 9 rows.  The demand-OFF arm
    # carries NO mass: the D-NGH observer is only attached when the flag is
    # on, so every ``demand_min_mass`` in the demand-OFF summary is null.  The
    # masses live in the demand-ON arm.  That arm is still a valid served-
    # control source, and SC1.1 G3 proves it: both arms served text IDENTICAL
    # to the P2C arm-1 baseline on all 9 probes with zero regressions, so the
    # demand-ON turn whose mass is read here served exactly the text the
    # demand-OFF turn served.  The cross-check below pins that equality per
    # probe instead of pretending to compare two masses.
    battery = _read_json(SC1_1_G3_SUMMARY)
    battery_off = _read_json(SC1_1_G3_SUMMARY_DEMAND_OFF)
    off_by_probe = {str(r["probe_id"]): r for r in battery_off["table"]}
    s11_g3: list[ServedRow] = []
    for entry in battery["table"]:
        probe = str(entry["probe_id"])
        mass = float(entry["demand_min_mass"])
        off = off_by_probe.get(probe)
        if off is None:
            raise CalibrationError(
                f"lived battery probe {probe} has no demand-OFF counterpart")
        if str(off["served_answer"]) != str(entry["served_answer"]):
            raise CalibrationError(
                f"lived battery demand-ON/OFF served text differs for {probe}; "
                "the demand-ON mass cannot stand for the demand-OFF turn")
        if off.get("demand_min_mass") is not None:
            raise CalibrationError(
                f"demand-OFF arm unexpectedly carries mass for {probe}: "
                f"{off['demand_min_mass']!r}")
        s11_g3.append(ServedRow(
            row_id=f"{probe}@sc1_1_g3",
            probe_id=probe,
            min_mass=mass,
            provenance="post_race_lived_battery",
            source=_rel(SC1_1_G3_SUMMARY),
            fired_index_at_carried=(
                None if entry.get("demand_token_index") is None
                else int(entry["demand_token_index"])),
        ))
    served["S11_G3"] = sorted(s11_g3, key=lambda r: r.probe_id)

    # -- class S11_G2: SC1.1 same-process fork served controls --------------
    s11_g2: list[ServedRow] = []
    for path in SC1_1_G2_RECEIPTS:
        for pair in _read_json(path)["pairs"]:
            arm = (pair.get("arms") or {}).get("served") or {}
            if "attempt_min_mass" not in arm:
                continue
            probe = str(pair["fixture_id"])
            s11_g2.append(ServedRow(
                row_id=f"{probe}@sc1_1_g2_fork",
                probe_id=probe,
                min_mass=float(arm["attempt_min_mass"]),
                provenance="post_race_same_process_fork",
                source=_rel(path),
                fired_index_at_carried=(
                    None if arm.get("demand_token_index") is None
                    else int(arm["demand_token_index"])),
            ))
    served["S11_G2"] = sorted(s11_g2, key=lambda r: r.probe_id)

    return {
        "served_classes": served,
        "planted": sorted(planted, key=lambda r: r.probe_id),
        "carried_threshold": carried,
    }


# ---------------------------------------------------------------------------
# De-duplication + the registered split
# ---------------------------------------------------------------------------


def deduplicate(inv: Mapping[str, Any]) -> dict[str, Any]:
    """Fold S12 into R_EVAL where the reproduction is exact; report the folds.

    The registration states the fold is lossless because every SC1.2 Arm-0 row
    reproduced its race-eval counterpart with min-mass delta exactly 0.0.  That
    claim is MEASURED here, not assumed: a non-zero delta keeps the row as its
    own distinct measurement rather than silently discarding either side.
    """
    classes = inv["served_classes"]
    race_eval_by_probe = {row.probe_id: row for row in classes["R_EVAL"]}
    folds: list[dict[str, Any]] = []
    kept_s12: list[ServedRow] = []
    for row in classes["S12"]:
        counterpart = race_eval_by_probe.get(row.probe_id)
        if counterpart is None:
            kept_s12.append(row)
            folds.append({
                "probe_id": row.probe_id, "folded": False,
                "reason": "no race-eval counterpart; carried as its own row"})
            continue
        delta = float(row.min_mass) - float(counterpart.min_mass)
        folds.append({
            "probe_id": row.probe_id,
            "sc1_2_arm0_min_mass": float(row.min_mass),
            "race_eval_min_mass": float(counterpart.min_mass),
            "delta": delta,
            "folded": delta == 0.0,
        })
        if delta != 0.0:
            kept_s12.append(row)
    return {
        "sc1_2_folds": folds,
        "sc1_2_folded_count": sum(1 for f in folds if f.get("folded")),
        "sc1_2_kept_as_distinct": [r.row_id for r in kept_s12],
        "kept_s12_rows": kept_s12,
    }


def registered_split(inv: Mapping[str, Any], dedup: Mapping[str, Any]) -> dict[str, Any]:
    """The PROVENANCE split, exactly as registered before computation.

    calibration = every RACE-provenance served row (R_CAL + R_EVAL, plus any
    S12 row that did NOT fold).
    held_out    = every POST-RACE served row (S11_G3 + S11_G2), de-duplicated
                  within the held-out side by probe id preferring the LIVED
                  battery row, with a disagreeing fork row carried additionally.
    """
    classes = inv["served_classes"]
    calibration = sorted(
        list(classes["R_CAL"]) + list(classes["R_EVAL"])
        + list(dedup["kept_s12_rows"]),
        key=lambda r: r.row_id)

    battery_by_probe = {row.probe_id: row for row in classes["S11_G3"]}
    held_out: list[ServedRow] = list(classes["S11_G3"])
    fork_notes: list[dict[str, Any]] = []
    for row in classes["S11_G2"]:
        twin = battery_by_probe.get(row.probe_id)
        if twin is None:
            held_out.append(row)
            fork_notes.append({
                "probe_id": row.probe_id, "carried": True,
                "reason": "no lived-battery counterpart"})
            continue
        agrees = float(row.min_mass) == float(twin.min_mass)
        fork_notes.append({
            "probe_id": row.probe_id,
            "sc1_1_g2_fork_min_mass": float(row.min_mass),
            "sc1_1_g3_battery_min_mass": float(twin.min_mass),
            "agrees": agrees,
            "carried": not agrees,
            "reason": ("identical to the lived-battery row; de-duplicated by "
                       "preferring the lived-battery provenance" if agrees else
                       "DISAGREES with the lived-battery row under a different "
                       "instrument; carried as its own row, never collapsed"),
        })
        if not agrees:
            held_out.append(row)
    held_out = sorted(held_out, key=lambda r: r.row_id)

    race_probes = {row.probe_id for row in calibration}
    disjoint = sorted(
        (row for row in held_out if row.probe_id not in race_probes),
        key=lambda r: r.row_id)

    return {
        "rule": ("PROVENANCE SPLIT: calibration = race-provenance served rows; "
                 "held-out = post-race served rows"),
        "calibration": calibration,
        "held_out": held_out,
        "held_out_disjoint_probe_ids": disjoint,
        "fork_dedup_notes": fork_notes,
    }


# ---------------------------------------------------------------------------
# The carried envelope rule + the carried decision rule
# ---------------------------------------------------------------------------


def envelope_threshold(rows: Sequence[ServedRow]) -> float:
    """``fit_thresholds``'s D-NGH branch, carried: min of served turn minima."""
    if not rows:
        raise CalibrationError("threshold fit has no served calibration rows")
    return float(min(float(row.min_mass) for row in rows))


def fires(min_mass: float, threshold: float) -> bool:
    """``detector_decision``'s D-NGH verdict, carried: strictly below fires.

    A turn fires iff SOME answer token is strictly below the line, which is
    identically its per-turn MINIMUM being strictly below the line.  Equality
    does not fire.
    """
    return float(min_mass) < float(threshold)


def false_fires(rows: Sequence[ServedRow], threshold: float) -> list[str]:
    return [row.row_id for row in rows if fires(row.min_mass, threshold)]


#: The one planted-miss probe whose RECOVERY arm was not measurable under the
#: post-race fork instrument (SC1.1 G2 / SC1.2).  Its RACE row IS measured and
#: present, which is why Part A reports recall against BOTH denominators.
RECOVERY_UNMEASURABLE_PROBE = "sup_solace_fresh"


def recall(rows: Sequence[PlantedRow], threshold: float) -> tuple[int, int]:
    hits = sum(1 for row in rows if fires(row.min_mass, threshold))
    return hits, len(rows)


def recall_recovery_denominator(
    rows: Sequence[PlantedRow], threshold: float,
) -> tuple[int, int]:
    """Recall over the 9 rows the RECOVERY instrument could measure.

    The order predicts ``9/9``.  The race itself measured TEN planted-miss
    rows, ``sup_solace_fresh`` included (min mass 0.0 over 13 tokens, present
    in the frozen eval rows).  What was not measurable was that probe's
    planted-miss arm under the POST-RACE fork instrument, which is why SC1.1
    and SC1.2 both count 9 positives against 10 negatives.  Detection recall
    and recovery recall therefore have different denominators, and reporting
    only one of them would misstate a real distinction.
    """
    subset = [row for row in rows if row.probe_id != RECOVERY_UNMEASURABLE_PROBE]
    hits = sum(1 for row in subset if fires(row.min_mass, threshold))
    return hits, len(subset)


# ---------------------------------------------------------------------------
# Rule equality against the FROZEN DET1 functions
# ---------------------------------------------------------------------------


def sweep_grid() -> list[float]:
    """0.05 .. 0.45 inclusive in 0.01 steps — the registered grid, 41 points."""
    grid = [round(SWEEP_START + index * SWEEP_STEP, 2)
            for index in range(SWEEP_POINTS)]
    if grid[0] != SWEEP_START or grid[-1] != SWEEP_STOP:
        raise CalibrationError(f"sweep grid drifted: {grid[0]}..{grid[-1]}")
    return grid


def rule_equality_check() -> dict[str, Any]:
    """Pin this module's minima path against the real DET1 functions.

    Runs on the 22 frozen race rows, the only rows that carry per-token arrays,
    where BOTH paths are computable.  Two equalities are checked:

      1. ``envelope_threshold`` over the race calibration served rows equals
         ``fit_thresholds``'s ``D-NGH.threshold`` EXACTLY (float equality, not
         a tolerance).
      2. ``fires(min_mass, t)`` equals ``detector_decision(row, "D-NGH", t)[0]``
         for every race row at every point of the registered sweep grid, and
         where it fires, DET1's returned index equals the first-strictly-below
         index.

    Equality 2 is the load-bearing one: it is what licenses Part A to evaluate
    fire verdicts from persisted minima on the SC1.1/SC1.2 rows, which carry no
    token arrays.
    """
    from scripts.grm_det1_common import detector_decision, fit_thresholds

    calibration_rows = _read_jsonl(RACE_CALIBRATION_ROWS)
    eval_rows = _read_jsonl(RACE_EVAL_ROWS)

    fitted = fit_thresholds(calibration_rows)
    det1_threshold = float(fitted["D-NGH"]["threshold"])
    ours = envelope_threshold([
        ServedRow(
            row_id=str(row["row_id"]), probe_id=str(row["fixture_id"]),
            min_mass=min(_race_token_masses(row)),
            provenance="race_calibration_lived",
            source=_rel(RACE_CALIBRATION_ROWS))
        for row in calibration_rows if row.get("variant") == "served"])

    grid = sweep_grid()
    mismatches: list[dict[str, Any]] = []
    checked = 0
    for row in calibration_rows + eval_rows:
        masses = _race_token_masses(row)
        minimum = min(masses)
        for point in grid:
            thresholds = {"D-NGH": {"threshold": float(point)}}
            det1_fired, det1_index = detector_decision(row, "D-NGH", thresholds)
            our_fired = fires(minimum, point)
            checked += 1
            if bool(det1_fired) != bool(our_fired):
                mismatches.append({
                    "row_id": row.get("row_id"), "threshold": float(point),
                    "det1_fired": bool(det1_fired), "sc2_fired": bool(our_fired)})
            if det1_fired:
                expected_index = _first_below(masses, point)
                if int(det1_index) != int(expected_index):
                    mismatches.append({
                        "row_id": row.get("row_id"), "threshold": float(point),
                        "det1_index": int(det1_index),
                        "sc2_first_below": expected_index})
    return {
        "det1_fit_thresholds_D_NGH_threshold": det1_threshold,
        "sc2_envelope_threshold_over_same_rows": ours,
        "envelope_exact_float_equality": ours == det1_threshold,
        "rows_checked": len(calibration_rows) + len(eval_rows),
        "decisions_checked": checked,
        "grid_points": len(grid),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "equality_holds": (ours == det1_threshold) and not mismatches,
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def sweep(
    held_out: Sequence[ServedRow],
    planted: Sequence[PlantedRow],
) -> list[dict[str, Any]]:
    table = []
    for point in sweep_grid():
        hits, total = recall(planted, point)
        rec_hits, rec_total = recall_recovery_denominator(planted, point)
        ff = false_fires(held_out, point)
        table.append({
            "threshold": float(point),
            "held_out_false_fires": len(ff),
            "held_out_false_fire_ids": ff,
            "held_out_n": len(held_out),
            "recall_hits": hits,
            "recall_total": total,
            "recall": (hits / total) if total else None,
            "recall_on_recovery_denominator": f"{rec_hits}/{rec_total}",
        })
    return table


# ---------------------------------------------------------------------------
# The Part A receipt
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _predictions(
    registration: Mapping[str, Any],
    candidate: float,
    ff_candidate: Sequence[str],
    recall_candidate: tuple[int, int],
    recall_recovery: tuple[int, int],
    gap: float,
) -> dict[str, Any]:
    preds = registration["PREDICTIONS_LEAD_REGISTERED_BEFORE_ANY_GATE"]
    rows = [
        {"prediction": preds["G2_candidate_threshold_le_0_30"],
         "measured": candidate, "met": candidate <= 0.30},
        {"prediction": preds["G2_held_out_false_fires_at_candidate_eq_0"],
         "measured": len(ff_candidate), "met": len(ff_candidate) == 0},
        {"prediction": preds["G2_recall_at_candidate_eq_9_of_9"],
         "measured": f"{recall_candidate[0]}/{recall_candidate[1]}",
         "met": tuple(recall_candidate) == (9, 9),
         "denominator_note": (
             "MET on the recovery denominator the prediction meant (9 rows, "
             "sup_solace_fresh excluded because its planted-miss arm was not "
             "measurable under the post-race fork instrument). The race itself "
             "measured TEN planted-miss rows, so on the full race denominator "
             "the same measurement reads 10/10. Both are reported; neither is "
             "hidden behind the other."),
         "met_on_recovery_denominator": tuple(recall_recovery) == (9, 9)},
        {"prediction": preds["G2_gap_ge_0_25"],
         "measured": gap, "met": gap >= 0.25},
    ]
    return {
        "rows": rows,
        "all_met": all(row["met"] for row in rows),
        "seat_disagreement_registered_before_computation": registration[
            "SEAT_PREDICTION_DISAGREEMENT_REGISTERED_BEFORE_COMPUTATION"],
    }


def part_a() -> dict[str, Any]:
    registration = _read_json(REGISTRATION_PATH)
    inv = inventory()
    dedup = deduplicate(inv)
    split = registered_split(inv, dedup)
    equality = rule_equality_check()

    carried = float(inv["carried_threshold"])
    calibration = split["calibration"]
    held_out = split["held_out"]
    planted = inv["planted"]

    candidate = envelope_threshold(calibration)
    all_rows = sorted(
        {row.row_id: row for row in list(calibration) + list(held_out)}.values(),
        key=lambda r: r.row_id)
    min_served_all = min(float(row.min_mass) for row in all_rows)
    min_served_row = min(all_rows, key=lambda r: float(r.min_mass))
    max_planted = max(float(row.min_mass) for row in planted)
    gap = min_served_all - max_planted

    ff_carried = false_fires(held_out, carried)
    ff_candidate = false_fires(held_out, candidate)
    recall_carried = recall(planted, carried)
    recall_candidate = recall(planted, candidate)
    recall_carried_recovery = recall_recovery_denominator(planted, carried)
    recall_candidate_recovery = recall_recovery_denominator(planted, candidate)

    disjoint = split["held_out_disjoint_probe_ids"]

    return {
        "schema": "grm.sc2.part_a_calibration.v1",
        "gate": "G2",
        "order": registration["order"],
        "order_sha256": registration["order_sha256"],
        "registration": {
            "path": _rel(REGISTRATION_PATH),
            "sha256": _sha256(REGISTRATION_PATH),
        },
        "envelope_rule_carried_unchanged": (
            registration["PART_A"]["envelope_rule_CARRIED_UNCHANGED"]),
        "persistence_finding": registration["PART_A"][
            "PERSISTENCE_FINDING_REGISTERED_BEFORE_COMPUTATION"],
        "rule_equality_vs_frozen_DET1": equality,

        "inventory": {
            "counts": {
                name: len(rows) for name, rows in inv["served_classes"].items()},
            "served_rows_by_class": {
                name: [row.as_json() for row in rows]
                for name, rows in inv["served_classes"].items()},
            "planted_rows": [row.as_json() for row in planted],
            "planted_measurable_n": len(planted),
            "planted_denominator_finding": {
                "race_planted_rows_measured": len(planted),
                "recovery_instrument_measurable": len(planted) - 1,
                "probe_not_measurable_under_recovery_instrument":
                    RECOVERY_UNMEASURABLE_PROBE,
                "correction_to_the_order": (
                    "The order says '9 measurable' planted-miss rows. The RACE "
                    "measured TEN, sup_solace_fresh included (min mass 0.0 over "
                    "13 answer tokens, present in the frozen eval rows). What "
                    "was not measurable is that probe's planted-miss arm under "
                    "the POST-RACE fork instrument, which is why SC1.1 and "
                    "SC1.2 count 9 positives against 10 negatives. Part A draws "
                    "its planted rows from the race, so its natural denominator "
                    "is 10; the prediction's denominator is 9. Both are "
                    "reported everywhere recall appears."),
            },
            "deduplication": {
                "sc1_2_folds": dedup["sc1_2_folds"],
                "sc1_2_folded_count": dedup["sc1_2_folded_count"],
                "sc1_2_kept_as_distinct": dedup["sc1_2_kept_as_distinct"],
                "fork_dedup_notes": split["fork_dedup_notes"],
            },
            "final_served_row_count_after_dedup": len(all_rows),
        },

        "split": {
            "rule": split["rule"],
            "registered_before_computation": True,
            "calibration_set_ids": [row.row_id for row in calibration],
            "calibration_set_n": len(calibration),
            "calibration_set_minima": {
                row.row_id: float(row.min_mass) for row in calibration},
            "held_out_set_ids": [row.row_id for row in held_out],
            "held_out_set_n": len(held_out),
            "held_out_set_minima": {
                row.row_id: float(row.min_mass) for row in held_out},
            "held_out_contains_sup_solace_fresh": any(
                row.probe_id == "sup_solace_fresh" for row in held_out),
            "secondary_cut_held_out_probe_ids_the_race_never_contained": {
                "ids": [row.row_id for row in disjoint],
                "n": len(disjoint),
                "minima": {row.row_id: float(row.min_mass) for row in disjoint},
                "false_fires_at_carried": false_fires(disjoint, carried),
                "false_fires_at_candidate": false_fires(disjoint, candidate),
            },
        },

        "candidate": {
            "candidate_threshold": candidate,
            "carried_race_threshold": carried,
            "delta_candidate_minus_carried": candidate - carried,
            "selection": "minimum_of_served_turn_minima_strict_envelope",
            "direction": "trigger_if_strictly_below",
            "fit_side": "served_controls_only",
            "argmin_row_id": min(
                calibration, key=lambda r: float(r.min_mass)).row_id,
            "REPORTED_NOT_ADOPTED": True,
        },

        "all_rows_envelope_secondary_figure": {
            "value": min_served_all,
            "argmin_row_id": min_served_row.row_id,
            "meaning": (
                "the envelope over EVERY served row in the inventory, both sides "
                "of the split. NOT the candidate: it is fit on the held-out rows "
                "too, so it cannot be tested against them. Reported because it is "
                "the number the lead's <= 0.30 prediction was reaching for."),
            "is_not_the_candidate": True,
        },

        "gap": {
            "min_served_control_mass_over_ALL_rows": min_served_all,
            "min_served_argmin_row_id": min_served_row.row_id,
            "max_planted_miss_mass": max_planted,
            "gap": gap,
            "meaning": (
                "the whole separable margin between the two populations. Every "
                "planted miss has mass exactly 0.0 -- the withheld node is not "
                "mounted, so there is no mounted band to read -- so the gap is "
                "just the lowest served number. That makes the gap a statement "
                "about how far the QUIETEST served turn sits above zero, not "
                "about how finely the detector discriminates."),
        },

        "held_out_evaluation": {
            "at_carried_threshold": {
                "threshold": carried,
                "false_fires": len(ff_carried),
                "false_fire_ids": ff_carried,
                "held_out_n": len(held_out),
                "recall_hits": recall_carried[0],
                "recall_total": recall_carried[1],
                "recall": recall_carried[0] / recall_carried[1],
                "recall_on_recovery_denominator": (
                    f"{recall_carried_recovery[0]}/{recall_carried_recovery[1]}"),
            },
            "at_candidate_threshold": {
                "threshold": candidate,
                "false_fires": len(ff_candidate),
                "false_fire_ids": ff_candidate,
                "held_out_n": len(held_out),
                "recall_hits": recall_candidate[0],
                "recall_total": recall_candidate[1],
                "recall": recall_candidate[0] / recall_candidate[1],
                "recall_on_recovery_denominator": (
                    f"{recall_candidate_recovery[0]}/"
                    f"{recall_candidate_recovery[1]}"),
            },
        },

        "sensitivity_sweep": {
            "grid": {"start": SWEEP_START, "stop": SWEEP_STOP,
                     "step": SWEEP_STEP, "points": SWEEP_POINTS},
            "reported_not_gated": True,
            "no_value_from_this_sweep_is_registered_or_adopted": True,
            "table": sweep(held_out, planted),
        },

        "predictions_vs_measured": _predictions(
            registration, candidate, ff_candidate, recall_candidate,
            recall_candidate_recovery, gap),
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
    payload = part_a()
    path = emit(payload, "sc2_calibration")
    equality = payload["rule_equality_vs_frozen_DET1"]
    print(f"receipt={_rel(path)}")
    print(f"rule_equality_holds={equality['equality_holds']}"
          f" decisions_checked={equality['decisions_checked']}"
          f" mismatches={equality['mismatch_count']}")
    print(f"calibration_n={payload['split']['calibration_set_n']}"
          f" held_out_n={payload['split']['held_out_set_n']}")
    print(f"candidate_threshold={payload['candidate']['candidate_threshold']!r}"
          f" argmin={payload['candidate']['argmin_row_id']}")
    print(f"carried_threshold={payload['candidate']['carried_race_threshold']!r}")
    print(f"all_rows_envelope={payload['all_rows_envelope_secondary_figure']['value']!r}"
          f" argmin={payload['all_rows_envelope_secondary_figure']['argmin_row_id']}")
    print(f"gap={payload['gap']['gap']!r}"
          f" min_served={payload['gap']['min_served_control_mass_over_ALL_rows']!r}"
          f" max_planted={payload['gap']['max_planted_miss_mass']!r}")
    carried_eval = payload["held_out_evaluation"]["at_carried_threshold"]
    cand_eval = payload["held_out_evaluation"]["at_candidate_threshold"]
    print(f"held_out_false_fires_at_carried={carried_eval['false_fires']}"
          f" ids={carried_eval['false_fire_ids']}")
    print(f"held_out_false_fires_at_candidate={cand_eval['false_fires']}"
          f" ids={cand_eval['false_fire_ids']}")
    print(f"recall_at_carried={carried_eval['recall_hits']}/{carried_eval['recall_total']}"
          f" (recovery_denominator={carried_eval['recall_on_recovery_denominator']})")
    print(f"recall_at_candidate={cand_eval['recall_hits']}/{cand_eval['recall_total']}"
          f" (recovery_denominator={cand_eval['recall_on_recovery_denominator']})")
    for row in payload["predictions_vs_measured"]["rows"]:
        print(f"prediction met={row['met']} measured={row['measured']!r}"
              f" :: {row['prediction']}")
    print("--- sweep (threshold, held_out_false_fires, recall/10, recall/9) ---")
    for row in payload["sensitivity_sweep"]["table"]:
        print(f"  {row['threshold']:.2f}  ff={row['held_out_false_fires']}"
              f"  recall={row['recall_hits']}/{row['recall_total']}"
              f"  recovery_recall={row['recall_on_recovery_denominator']}"
              f"  ff_ids={row['held_out_false_fire_ids']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
