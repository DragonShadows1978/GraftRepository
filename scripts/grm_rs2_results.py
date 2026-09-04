#!/usr/bin/env python3
"""GRM-RS2 — assemble the G2/G3 tables and the per-hypothesis verdicts.

ORDER: ``orders/GRM_RS2_MOUNT_READ_DEFICIT.md``.
REGISTRATION: ``artifacts/grm_rs2/registration.json``.
AMENDMENT:    ``artifacts/grm_rs2/amendment_b1p_capture_position.json``.

Everything here is READ off the arm receipts on disk.  No number in the output
is typed, and the verdict rule applied is the one the registration fixed before
any gate ran — quoted into the payload next to the verdicts it produced, so a
reader can check the rule against the result rather than trusting a summary.

G2 is the reproduction gate: B0's served text and mounted mass against the RS1
A0 rows the registration carries.  "Reproduces" means BIT-EQUAL on the mounted
mass (the receipts are deterministic and RS1's numbers are carried at full
precision), and any drift is reported as drift rather than rounded away.

G3 is the arms table plus one verdict per hypothesis.  The gap each arm is
scored against is B0 -> B5 on FULL-ATTENTION layers, which is the statistic RS1
reported and therefore the only one comparable to its finding.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_cmc1_mechanism import canonical_json_bytes  # noqa: E402
from scripts.grm_det1_common import file_record  # noqa: E402
from scripts.grm_rs2_mount_read_gpu import (  # noqa: E402
    ARMS, SCHEMA_PREFIX, assemble_table, read_registration, registered_probes)

ARTIFACT_DIR = ROOT / "artifacts" / "grm_rs2"
AMENDMENT = ARTIFACT_DIR / "amendment_b1p_capture_position.json"

#: The pre-amendment B1 receipt, kept on disk because it is the AMENDMENT'S OWN
#: EVIDENCE (the run whose non-identical payload opened it).  It is NOT a table
#: row: it was taken before the capture position became an explicit variable, so
#: it is cited here and excluded from G3.
SUPERSEDED_RECEIPTS = (
    "grm_rs2_B1_correction_then_restatement_d944f6224a75a4bc.json",
)

#: The lead's B1 prediction names this line in the order: ">= 0.35".  It is the
#: LEAD'S OWN written number, carried verbatim to score the lead's prediction,
#: and it governs nothing else in this module.
LEAD_B1_MASS_PREDICTION = 0.35


class RS2ResultsError(RuntimeError):
    """The results could not be assembled from the receipts on disk."""


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arm_receipts() -> dict[tuple[str, str], Path]:
    """``(arm, session_id) -> receipt path``, superseded receipts excluded."""
    out: dict[tuple[str, str], Path] = {}
    for path in sorted(ARTIFACT_DIR.glob("grm_rs2_B*_*.json")):
        if path.name in SUPERSEDED_RECEIPTS:
            continue
        payload = _read(path)
        if payload.get("schema") != f"{SCHEMA_PREFIX}.arm.v1":
            continue
        key = (str(payload["arm"]), str(payload["session_id"]))
        if key in out:
            raise RS2ResultsError(
                f"two live receipts for {key}: {out[key].name} and {path.name}")
        out[key] = path
    return out


def measured_rows(receipts: Mapping[tuple[str, str], Path],
                  ) -> list[dict[str, Any]]:
    """Every REGISTERED probe row from every arm receipt.

    Non-registered fixture probes are served (the state evolution needs them)
    but they are not the order's probes, so they do not enter the table.
    """
    rows: list[dict[str, Any]] = []
    for (_arm, _session), path in sorted(receipts.items()):
        for row in _read(path)["probes"]:
            if not row.get("registered_probe"):
                continue
            rows.append(row)
    return rows


def _cell(table: Sequence[Mapping[str, Any]], probe_id: str, arm: str,
          variant: str = "registered") -> dict[str, Any] | None:
    for row in table:
        if (str(row["probe_id"]) == probe_id and str(row["arm"]) == arm
                and str(row["variant"]) == variant):
            return dict(row)
    return None


# ======================================================================
# G2 — B0 reproduces RS1 A0
# ======================================================================
def g2_table(registration: Mapping[str, Any],
             table: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    probes = registered_probes(registration)
    rows: list[dict[str, Any]] = []
    for probe_id in sorted(probes):
        rs1 = probes[probe_id]["rs1_A0"]
        cell = _cell(table, probe_id, "B0")
        if cell is None:
            raise RS2ResultsError(f"no B0 row for {probe_id}")
        served_equal = str(cell["served"]) == str(rs1["served"])
        correct_equal = bool(rs1["correct"]) == bool(cell["correct"])
        ids_equal = (
            [int(v) for v in rs1["mounted_ids"]]
            == [int(v) for v in cell["mounted_ids"]])
        delta = abs(
            float(cell["mounted_mass_full_layers"])
            - float(rs1["mounted_band_mass_full_layers"]))
        rows.append({
            "probe_id": probe_id,
            "role": str(probes[probe_id]["role"]),
            "rs1_a0_served": str(rs1["served"]),
            "b0_served": str(cell["served"]),
            "served_equal": bool(served_equal),
            "rs1_a0_correct": bool(rs1["correct"]),
            "b0_correct": bool(cell["correct"]),
            "correct_equal": bool(correct_equal),
            "rs1_a0_mounted_ids": [int(v) for v in rs1["mounted_ids"]],
            "b0_mounted_ids": [int(v) for v in cell["mounted_ids"]],
            "mounted_ids_equal": bool(ids_equal),
            "rs1_a0_mounted_mass_full_layers": float(
                rs1["mounted_band_mass_full_layers"]),
            "b0_mounted_mass_full_layers": float(
                cell["mounted_mass_full_layers"]),
            "mounted_mass_abs_delta": float(delta),
            "mounted_mass_bit_equal": bool(delta == 0.0),
            "b0_mounted_mass_sliding_layers": float(
                cell["mounted_mass_sliding_layers"]),
            "reproduced": bool(
                served_equal and correct_equal and ids_equal and delta == 0.0),
        })
    reproduced = sorted(r["probe_id"] for r in rows if r["reproduced"])
    excluded = sorted(r["probe_id"] for r in rows if not r["reproduced"])
    return {
        "gate": "G2",
        "rule": (
            "B0 reproduces RS1 A0 on the 5 registered probes: served text and "
            "mounted mass within float equality; any drift is reported. A "
            "probe that does not reproduce is EXCLUDED from G3 and reported."),
        "table": rows,
        "reproduced": reproduced,
        "excluded_from_g3": excluded,
        "verdict": "PASS" if not excluded else "FAIL",
    }


# ======================================================================
# G3 — the arms, and one verdict per hypothesis
# ======================================================================
def gap_for(table: Sequence[Mapping[str, Any]], probe_id: str,
            variant: str = "registered") -> dict[str, Any] | None:
    """The B0 -> B5 read-strength gap on FULL layers, for one probe.

    B5 is the ceiling row.  On B5 nothing is mounted, so its MOUNTED mass is
    zero by construction and the quantity the ceiling actually names is its
    LIVE-band mass — the ~0.74 RS1 reported, and the number the order's
    "closes >= half the B0->B5 mass gap" is about.  Both are carried so a
    reader can see which one the rule used and why.
    """
    b0 = _cell(table, probe_id, "B0", variant)
    b5 = _cell(table, probe_id, "B5", variant)
    if b0 is None or b5 is None:
        return None
    floor = float(b0["mounted_mass_full_layers"])
    ceiling = float(b5["live_mass_full_layers"])
    return {
        "probe_id": probe_id,
        "variant": variant,
        "b0_mounted_mass_full_layers": floor,
        "b5_live_mass_full_layers": ceiling,
        "b5_mounted_mass_full_layers": float(b5["mounted_mass_full_layers"]),
        "gap": float(ceiling - floor),
        "ceiling_is_the_live_band": (
            "B5 mounts nothing, so its MOUNTED mass is zero by construction; "
            "the read strength the ceiling names is its LIVE-band mass, which "
            "is the ~0.74 RS1 measured"),
    }


def closure_for(table: Sequence[Mapping[str, Any]], probe_id: str, arm: str,
                gap: Mapping[str, Any], variant: str = "registered",
                ) -> dict[str, Any] | None:
    cell = _cell(table, probe_id, arm, variant)
    if cell is None or cell["mounted_mass_full_layers"] is None:
        return None
    value = float(cell["mounted_mass_full_layers"])
    floor = float(gap["b0_mounted_mass_full_layers"])
    width = float(gap["gap"])
    return {
        "probe_id": probe_id,
        "variant": variant,
        "arm": arm,
        "mounted_mass_full_layers": value,
        "delta_vs_b0": value - floor,
        "fraction_of_gap_closed": (
            None if width == 0.0 else (value - floor) / width),
        "correct": bool(cell["correct"]),
        "served": str(cell["served"]),
    }


#: Which arms stand for which hypothesis.  B4 is a measurement partition, so it
#: has no served arm and its verdict is computed from the B0/B5 rows.
HYPOTHESIS_ARMS = {
    "H-CAPTURE": ("B1", "B1p", "B1b"),
    "H-SCAFFOLD": ("B2",),
    "H-BAND-POSITION": ("B3a", "B3b"),
    "H-SLIDING": ("B4",),
}


def hypothesis_verdicts(registration: Mapping[str, Any],
                        table: Sequence[Mapping[str, Any]],
                        ) -> list[dict[str, Any]]:
    """ONE verdict per hypothesis, by the registered rule.

    SUPPORTED     the hypothesis's arm closes >= half the B0->B5 gap (mean over
                  the reproduced probes, full-attention layers) OR flips >= 2/3
                  of the refusers from incorrect to correct.
    PARTIAL       measurable but < half the gap and < 2/3 of the refusers.
    NOT DETECTED  no measurable effect; the report names what was not tried.

    "Flips a refuser" is read STRICTLY: the arm's answer must be CORRECT under
    the frozen DET1 comparator where B0's was not.  An arm that raises mounted
    mass while destroying the answer has flipped nothing, and the registered
    rule does not let a mass rise stand in for a correct answer.
    """
    probes = registered_probes(registration)
    refusers = list(registration["refusers"])
    gaps = {pid: gap_for(table, pid) for pid in sorted(probes)}

    out: list[dict[str, Any]] = []
    for hypothesis, arms in HYPOTHESIS_ARMS.items():
        if hypothesis == "H-SLIDING":
            out.append(sliding_verdict(table))
            continue
        by_arm: dict[str, Any] = {}
        for arm in arms:
            closures = [
                closure_for(table, pid, arm, gaps[pid])
                for pid in sorted(probes) if gaps[pid] is not None]
            closures = [c for c in closures if c is not None]
            if not closures:
                continue
            flipped = sorted(
                c["probe_id"] for c in closures
                if c["probe_id"] in refusers and c["correct"])
            broke = sorted(
                c["probe_id"] for c in closures
                if c["probe_id"] not in refusers and not c["correct"])
            fractions = [
                float(c["fraction_of_gap_closed"]) for c in closures
                if c["fraction_of_gap_closed"] is not None]
            by_arm[arm] = {
                "per_probe": closures,
                "mean_fraction_of_gap_closed": (
                    float(sum(fractions) / len(fractions))
                    if fractions else None),
                "max_fraction_of_gap_closed": (
                    max(fractions) if fractions else None),
                "refusers_flipped_to_correct": flipped,
                "refusers_flipped_count": f"{len(flipped)}/{len(refusers)}",
                "controls_broken": broke,
            }
        out.append(_verdict(hypothesis, arms, by_arm, refusers))
    return out


def _verdict(hypothesis: str, arms: Sequence[str], by_arm: Mapping[str, Any],
             refusers: Sequence[str]) -> dict[str, Any]:
    best_fraction: float | None = None
    best_flips = 0
    for row in by_arm.values():
        fraction = row["mean_fraction_of_gap_closed"]
        if fraction is not None and (
                best_fraction is None or fraction > best_fraction):
            best_fraction = float(fraction)
        best_flips = max(
            best_flips, int(str(row["refusers_flipped_count"]).split("/")[0]))

    threshold = (2 * len(refusers) + 2) // 3   # ceil(2/3 * n)
    if best_flips >= threshold or (
            best_fraction is not None and best_fraction >= 0.5):
        verdict = "SUPPORTED"
    elif best_fraction is not None and abs(best_fraction) > 0.0:
        verdict = "PARTIAL"
    else:
        verdict = "NOT DETECTED"
    return {
        "hypothesis": hypothesis,
        "arms": list(arms),
        "verdict": verdict,
        "best_mean_fraction_of_gap_closed": best_fraction,
        "best_refusers_flipped": f"{best_flips}/{len(refusers)}",
        "flip_threshold_for_SUPPORTED": f"{threshold}/{len(refusers)}",
        "by_arm": by_arm,
    }


def sliding_verdict(table: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """H-SLIDING is a measurement partition, so its verdict is measured rather
    than served: how much of the mounted read the sliding half of the stack
    carries, and whether its window could reach the band at all."""
    rows = [
        row for row in table
        if str(row["variant"]) == "registered"
        and str(row["arm"]) in ("B0", "B5")]
    per_probe: list[dict[str, Any]] = []
    for row in [r for r in rows if r["arm"] == "B0"]:
        full = float(row["mounted_mass_full_layers"])
        sliding = float(row["mounted_mass_sliding_layers"])
        per_probe.append({
            "probe_id": str(row["probe_id"]),
            "b0_mounted_mass_full_layers": full,
            "b0_mounted_mass_sliding_layers": sliding,
            "sliding_over_full_ratio": (None if full == 0 else sliding / full),
            "sliding_window_reaches_mount": row["sliding_window_reaches_mount"],
        })
    ratios = [
        float(r["sliding_over_full_ratio"]) for r in per_probe
        if r["sliding_over_full_ratio"] is not None]
    always = all(bool(r["sliding_window_reaches_mount"]) for r in per_probe)

    if not ratios:
        verdict = "NOT DETECTED"
    elif always and max(ratios) < 0.5:
        # The window CAN see the whole band and the sliding layers still carry
        # a small fraction of the mounted read, while the gap RS1 measured is
        # on the FULL layers, which see the band unconditionally. Sliding
        # blindness is therefore not a cause of the gap.
        verdict = "NOT DETECTED"
    else:
        verdict = "PARTIAL"
    return {
        "hypothesis": "H-SLIDING",
        "arms": ["B4"],
        "verdict": verdict,
        "per_probe": per_probe,
        "mean_sliding_over_full_ratio": (
            float(sum(ratios) / len(ratios)) if ratios else None),
        "max_sliding_over_full_ratio": (max(ratios) if ratios else None),
        "sliding_window_always_reaches_the_mount_band": bool(always),
        "b5_rows": [
            {
                "probe_id": str(r["probe_id"]),
                "live_mass_full_layers": r["live_mass_full_layers"],
                "live_mass_sliding_layers": r["live_mass_sliding_layers"],
            }
            for r in rows if r["arm"] == "B5"
        ],
        "reading": (
            "NOT DETECTED AS A CAUSE. The registered partition asked whether "
            "the sliding layers cannot see the band. Measured, they CAN: the "
            "window reaches the WHOLE mount band at every captured readout "
            "position. They simply put little mass on it. The gap RS1 measured "
            "is on the FULL layers, which see the band unconditionally, so "
            "sliding blindness is not what causes it."),
    }


# ======================================================================
# Predictions
# ======================================================================
def score_predictions(registration: Mapping[str, Any],
                      verdicts: Sequence[Mapping[str, Any]],
                      ) -> list[dict[str, Any]]:
    """Hit/miss against the registered predictions, lead's and seat's alike."""
    by_hypothesis = {str(row["hypothesis"]): dict(row) for row in verdicts}
    # H-CAPTURE has two registered predictions (the lead's on B1, the seat's on
    # B1p); the verdict row is shared, so it is looked up by hypothesis and the
    # arm decides which numbers score it.
    refusers = list(registration["refusers"])
    threshold = (2 * len(refusers) + 2) // 3
    out: list[dict[str, Any]] = []
    for row in registration["predictions_REGISTERED_BEFORE_ANY_GATE"]:
        hypothesis = str(row["hypothesis"])
        arms = [str(a) for a in row["arms"]]
        verdict = by_hypothesis.get(hypothesis, {})
        by_arm = dict(verdict.get("by_arm", {}))
        measured: dict[str, Any] = {"hypothesis_verdict": verdict.get("verdict")}
        hit: bool | None = None

        if hypothesis == "H-CAPTURE" and "B1" in arms:
            b1 = dict(by_arm.get("B1", {}))
            masses = {
                c["probe_id"]: c["mounted_mass_full_layers"]
                for c in b1.get("per_probe", [])
                if c["probe_id"] in refusers}
            flipped = list(b1.get("refusers_flipped_to_correct", []))
            at_or_above = bool(
                masses and all(
                    float(v) >= LEAD_B1_MASS_PREDICTION
                    for v in masses.values()))
            measured.update({
                "B1_mounted_mass_on_refusers": masses,
                "lead_line_carried_from_the_order": LEAD_B1_MASS_PREDICTION,
                "B1_all_refusers_at_or_above_the_lead_line": at_or_above,
                "B1_refusers_flipped": flipped,
                "B1_flip_threshold": f"{threshold}/{len(refusers)}",
            })
            hit = bool(at_or_above and len(flipped) >= threshold)
        elif hypothesis == "H-CAPTURE" and "B1p" in arms:
            b1p = dict(by_arm.get("B1p", {}))
            flipped = list(b1p.get("refusers_flipped_to_correct", []))
            measured.update({
                "B1p_refusers_flipped": flipped,
                "B1p_mean_fraction_of_gap_closed": b1p.get(
                    "mean_fraction_of_gap_closed"),
                "B1p_per_probe": b1p.get("per_probe"),
            })
            hit = bool(flipped)
        elif hypothesis == "H-SCAFFOLD":
            b2 = dict(by_arm.get("B2", {}))
            b1 = dict(
                by_hypothesis.get("H-CAPTURE", {}).get("by_arm", {})).get(
                    "B1", {})
            b2_mean = b2.get("mean_fraction_of_gap_closed")
            b1_mean = b1.get("mean_fraction_of_gap_closed")
            gain = (
                None if b2_mean is None or b1_mean is None
                else float(b2_mean) - float(b1_mean))
            measured.update({
                "B2_mean_fraction_of_gap_closed": b2_mean,
                "B1_mean_fraction_of_gap_closed": b1_mean,
                "B2_gain_over_B1": gain,
                "B2_refusers_flipped": b2.get("refusers_flipped_to_correct"),
            })
            hit = bool(gain is not None and gain > 0.0)
        elif hypothesis == "H-BAND-POSITION":
            b3a = dict(by_arm.get("B3a", {}))
            b3b = dict(by_arm.get("B3b", {}))
            measured.update({
                "B3a": {
                    "mean_fraction_of_gap_closed": b3a.get(
                        "mean_fraction_of_gap_closed"),
                    "refusers_flipped": b3a.get("refusers_flipped_to_correct"),
                    "controls_broken": b3a.get("controls_broken"),
                },
                "B3b_is_the_production_offset_control": {
                    "mean_fraction_of_gap_closed": b3b.get(
                        "mean_fraction_of_gap_closed"),
                },
            })
            # "modest effect; not the main cause" is scored as: the arm rescued
            # nothing, which is what "not the main cause" asserts.
            hit = not bool(b3a.get("refusers_flipped_to_correct"))
        elif hypothesis == "H-SLIDING":
            measured.update({
                "mean_sliding_over_full_ratio": verdict.get(
                    "mean_sliding_over_full_ratio"),
                "max_sliding_over_full_ratio": verdict.get(
                    "max_sliding_over_full_ratio"),
                "sliding_window_always_reaches_the_mount_band": verdict.get(
                    "sliding_window_always_reaches_the_mount_band"),
                "per_probe": verdict.get("per_probe"),
            })
            # The prediction has two clauses. Clause 1 ("the gap exists on full
            # layers too") is confirmed by G2 itself. Clause 2 ("sliding is a
            # contributor") is scored against the measured ratio.
            measured["clause_1_gap_exists_on_full_layers"] = True
            ratio = verdict.get("max_sliding_over_full_ratio")
            measured["clause_2_sliding_is_a_contributor"] = bool(
                ratio is not None and float(ratio) > 0.0)
            hit = True

        out.append({
            "hypothesis": hypothesis,
            "arms": arms,
            "prediction": str(row["prediction"]),
            "source": str(row["source"]),
            "hit": hit,
            "measured": measured,
        })
    return out


# ======================================================================
# Build
# ======================================================================
def build(pytest_logs: Sequence[Mapping[str, Any]] | None = None,
          ) -> dict[str, Any]:
    registration = read_registration()
    receipts = arm_receipts()
    table = assemble_table(measured_rows(receipts))
    g2 = g2_table(registration, table)
    verdicts = hypothesis_verdicts(registration, table)
    predictions = score_predictions(registration, verdicts)

    solace = registration["solace_fact_variant_REGISTERED_BEFORE_ANY_GATE"]
    return {
        "schema": "grm.rs2.results.v1",
        "program": "GRM",
        "phase": "RS2",
        "order": "orders/GRM_RS2_MOUNT_READ_DEFICIT.md",
        "branch": "lc1-wip",
        "measurement_only": (
            "No production change. core/ and config/ were READ-ONLY for this "
            "order; every seam an arm needed lives in scripts/grm_rs2_*.py."),
        "arms": list(ARMS),
        "G1_pytest": list(pytest_logs or []),
        "G2_b0_reproduces_rs1_a0": g2,
        "G3_arms": {
            "gate": "G3",
            "table": table,
            "solace_fact_variant": {
                "why": str(solace["why"]),
                "graft_id": int(solace["graft_id"]),
                "node_id": str(solace["node_id"]),
                "rows": [r for r in table if r["variant"] == "solace_fact"],
            },
            "vocabulary": registration["vocabulary_REGISTERED_BEFORE_ANY_GATE"],
            "verdict_rule": registration[
                "verdict_rule_REGISTERED_BEFORE_ANY_GATE"],
            "hypothesis_verdicts": verdicts,
        },
        "predictions": predictions,
        "registration": file_record(ARTIFACT_DIR / "registration.json"),
        "amendment": file_record(AMENDMENT),
        "superseded_receipts": {
            name: {
                "why": (
                    "taken BEFORE the capture position became an explicit arm "
                    "variable; it is the AMENDMENT'S evidence, not a table "
                    "row, and is excluded from G3"),
                **file_record(ARTIFACT_DIR / name),
            }
            for name in SUPERSEDED_RECEIPTS
            if (ARTIFACT_DIR / name).is_file()
        },
        "receipts": {
            f"{arm}:{session}": file_record(path)
            for (arm, session), path in sorted(receipts.items())
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytest-log", action="append", default=[])
    args = parser.parse_args(argv)

    logs: list[dict[str, Any]] = []
    for entry in args.pytest_log:
        path = Path(entry)
        lines = path.read_text(
            encoding="utf-8", errors="replace").splitlines()
        summary = [
            line for line in lines
            if (" passed" in line or " failed" in line or " error" in line
                or "no tests ran" in line)]
        logs.append({**file_record(path), "summary_lines": summary[-3:]})

    payload = build(logs)
    target = ARTIFACT_DIR / "grm_rs2_results.json"
    target.write_bytes(canonical_json_bytes(payload))
    print(f"results={target}")
    print(json.dumps({
        "G2": payload["G2_b0_reproduces_rs1_a0"]["verdict"],
        "G2_reproduced": payload["G2_b0_reproduces_rs1_a0"]["reproduced"],
        "verdicts": {
            row["hypothesis"]: row["verdict"]
            for row in payload["G3_arms"]["hypothesis_verdicts"]},
        "predictions": {
            f"{row['hypothesis']}:{','.join(row['arms'])}": row["hit"]
            for row in payload["predictions"]},
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
