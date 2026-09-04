#!/usr/bin/env python3
"""GRM-RS1 — assemble G2 and G3 from the arm receipts.

ORDER: ``orders/GRM_RS1_READ_STRENGTH_PROBE.md`` (gates G2 and G3).

This module DECIDES NOTHING BY ITSELF.  It reads the per-arm receipts written
by ``scripts/grm_rs1_read_strength_gpu.py``, applies the verdict rule that was
written into ``artifacts/grm_rs1/registration.json`` BEFORE any gate ran, and
emits one summary.  Every number in the output is copied out of a receipt; no
value is recomputed from memory and none is typed.

G2 — A0 REPRODUCES EB1.  For each registered probe the A0 row's served answer
is compared to the EB1 G2 row the registration carries, under the same frozen
semantic comparator the battery uses (``lsr_p2c_replay_gpu.reproduction_
verdict``: a refusal reproduces a refusal, a value reproduces when the EB1
value is present).  A probe that does not reproduce is EXCLUDED from G3 and
named as excluded, per the order.

G3 — THE ARMS.  One table row per (probe, arm), then the mechanism verdict per
refuser in the registered vocabulary, then the lead's predictions scored
hit/miss against what was measured.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_cmc1_mechanism import canonical_json_bytes  # noqa: E402
from scripts.grm_det1_common import file_record  # noqa: E402
from scripts import grm_rs1_read_strength_gpu as rs1  # noqa: E402
from scripts import grm_rs1_registration as reg  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts" / "grm_rs1"
REGISTRATION = ARTIFACT_DIR / "registration.json"
AMENDMENT = ARTIFACT_DIR / "amendment_a2_width.json"


def latest_receipts() -> dict[tuple[str, str], Path]:
    """The NEWEST receipt per (arm, session).

    Content-addressed names mean a re-run writes a new file rather than
    overwriting, and this order re-ran two arms after a MEASURED harness fault
    (A2's arena overflow, A4's abstention).  The newest file for a pair is the
    one that ran under the corrected harness; the superseded ones stay on disk
    and the amendment names them.
    """
    out: dict[tuple[str, str], Path] = {}
    for path in sorted(ARTIFACT_DIR.glob("grm_rs1_A*_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != f"{rs1.SCHEMA_PREFIX}.arm.v1":
            continue
        arm = str(payload["arm"])
        if arm not in rs1.ARMS:
            # A receipt from a SUPERSEDED arm name. The bare ``"A2"`` runs
            # predate the width amendment (see artifacts/grm_rs1/
            # amendment_a2_width.json): one overflowed the arena and one
            # collapsed onto A0. They stay on disk because the amendment cites
            # them as the evidence that forced the split, and they are kept
            # OUT of the table because an arm the amendment retired must not
            # sit in it as if it were a result.
            continue
        key = (arm, str(payload["session_id"]))
        current = out.get(key)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            out[key] = path
    return out


def arm_rows() -> list[dict[str, Any]]:
    """Every REGISTERED probe row from the newest receipt of each arm."""
    rows: list[dict[str, Any]] = []
    for (arm, session), path in sorted(latest_receipts().items()):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for probe in payload["probes"]:
            if not probe.get("registered_probe"):
                continue
            rows.append({
                **probe,
                "arm": arm,
                "session_id": session,
                "receipt": file_record(path),
            })
    return rows


def g2_table(registration: Mapping[str, Any],
             rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A0 vs the EB1 G2 rows the registration carries."""
    from scripts.lsr_p2c_replay_gpu import reproduction_verdict

    probes = rs1.registered_probes(registration)
    table: list[dict[str, Any]] = []
    for probe_id, row in probes.items():
        a0 = next(
            (r for r in rows
             if r["probe_id"] == probe_id and r["arm"] == "A0"), None)
        if a0 is None:
            table.append({
                "probe_id": probe_id,
                "role": row["role"],
                "reproduced": False,
                "reason": "no A0 receipt for this probe",
            })
            continue
        verdict = reproduction_verdict(
            str(a0["served_answer"]), str(row["a0_served_answer"]))
        table.append({
            "probe_id": probe_id,
            "role": row["role"],
            "eb1_served_answer": str(row["a0_served_answer"]),
            "eb1_correct": bool(row["a0_correct"]),
            "eb1_mounted_ids": [int(v) for v in row["a0_mounted_ids"]],
            "rs1_a0_served_answer": str(a0["served_answer"]),
            "rs1_a0_correct": bool(a0["correct"]),
            "rs1_a0_mounted_ids": [int(v) for v in a0["mounted_ids"]],
            "mounted_ids_match": (
                [int(v) for v in a0["mounted_ids"]]
                == [int(v) for v in row["a0_mounted_ids"]]),
            "reproduction_mode": verdict["mode"],
            "reproduced": bool(verdict["reproduced"]),
        })
    reproduced = [r["probe_id"] for r in table if r["reproduced"]]
    excluded = [r["probe_id"] for r in table if not r["reproduced"]]
    return {
        "gate": "G2",
        "rule": (
            "A0 reproduces EB1 G2 on the 5 registered probes (3 refusals + 2 "
            "correct, served text semantically equal). A probe that does not "
            "reproduce is EXCLUDED from the other arms and reported."),
        "table": table,
        "reproduced": reproduced,
        "excluded_from_g3": excluded,
        "verdict": "PASS" if not excluded else "PARTIAL",
    }


def g3_table(registration: Mapping[str, Any], rows: list[dict[str, Any]],
             reproduced: list[str]) -> dict[str, Any]:
    """The arm table, the mechanism verdicts, and the vocabulary they use."""
    eligible = [r for r in rows if r["probe_id"] in set(reproduced)]
    table = rs1.assemble_table(eligible)

    correct_by: dict[str, dict[str, bool]] = {}
    for row in eligible:
        correct_by.setdefault(str(row["probe_id"]), {})[
            str(row["arm"])] = bool(row["correct"])

    verdicts: list[dict[str, Any]] = []
    for probe_id in reg.REFUSERS:
        if probe_id not in correct_by:
            verdicts.append({
                "probe_id": probe_id,
                "verdict": None,
                "reason": "excluded at G2 (A0 did not reproduce)",
            })
            continue
        arms = dict(correct_by[probe_id])
        if arms.get("A0"):
            verdicts.append({
                "probe_id": probe_id,
                "verdict": None,
                "reason": "A0 served correctly, so this is not a refuser in "
                          "this run and has no mechanism to name",
            })
            continue
        verdicts.append({
            "probe_id": probe_id,
            "arms_correct": arms,
            "rescued_by": sorted(
                arm for arm, ok in arms.items() if ok and arm != "A0"),
            "verdict": rs1.mechanism_verdict(arms),
        })

    return {
        "gate": "G3",
        "table": table,
        "arms_correct_by_probe": correct_by,
        "mechanism_verdicts": verdicts,
        "vocabulary": dict(
            registration["vocabulary_REGISTERED_BEFORE_ANY_GATE"]),
        "verdict_rule": registration[
            "verdict_rule_REGISTERED_BEFORE_ANY_GATE"],
    }


def score_predictions(correct_by: Mapping[str, Mapping[str, bool]],
                      table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The lead's registered predictions, scored against what was measured."""
    def served(probe_id: str, arm: str) -> bool | None:
        return dict(correct_by.get(probe_id) or {}).get(arm)

    def mass(probe_id: str, arm: str, column: str) -> Any:
        for row in table:
            if row["probe_id"] == probe_id and row["arm"] == arm:
                return row.get(column)
        return None

    refusers = [p for p in reg.REFUSERS if p in correct_by]
    a1_all = [served(p, "A1") for p in refusers]
    a2_harbor = any(
        served("sup_harbor_restatement", arm) for arm in rs1.A2_ARMS)
    a3_changed = [
        p for p in correct_by
        if served(p, "A3") is not None and served(p, "A3") != served(p, "A0")
    ]
    a4_flipped = [
        p for p in refusers if served(p, "A4") and not served(p, "A0")]

    return [
        {
            "arm": "A1",
            "prediction": "correct on all three refusers",
            "measured": {p: served(p, "A1") for p in refusers},
            "hit": bool(a1_all) and all(bool(v) for v in a1_all),
        },
        {
            "arm": "A2",
            "prediction": "correct on harbor (restatement is content); "
                          "UNKNOWN on praxis/solace — report",
            "measured": {
                p: {arm: served(p, arm) for arm in rs1.A2_ARMS}
                for p in refusers},
            "hit": bool(a2_harbor),
        },
        {
            "arm": "A3",
            "prediction": "no change (quantization is not the cause)",
            "measured": {
                "probes_whose_correctness_changed_vs_A0": a3_changed},
            "hit": not a3_changed,
        },
        {
            "arm": "A4",
            "prediction": "partial — if refusals flip here, the model READ "
                          "the graft and the refusal is the instruct prior",
            "measured": {"refusers_flipped_to_correct": a4_flipped},
            "hit": None,
            "hit_note": (
                "the order registered A4 as 'partial' with no pass rule, so "
                "it is REPORTED rather than scored: "
                f"{len(a4_flipped)} of {len(refusers)} refusers flipped"),
        },
        {
            "arm": "A5",
            "prediction": "A0 mounted mass on the refusers < the "
                          "persistent-frame served-control masses "
                          "(0.29-0.44); A1 shows the mass on the LIVE band; "
                          "A2 moves it onto the mounted band",
            "measured": {
                "A0_mounted_band_mass_on_refusers": {
                    p: mass(p, "A0", "mounted_band_mass") for p in refusers},
                "A0_mounted_band_mass_on_controls": {
                    p: mass(p, "A0", "mounted_band_mass")
                    for p in reg.CONTROLS if p in correct_by},
                "A1_live_band_mass_on_refusers": {
                    p: mass(p, "A1", "live_band_mass") for p in refusers},
                "A1_mounted_band_mass_on_refusers": {
                    p: mass(p, "A1", "mounted_band_mass") for p in refusers},
                "A2b_mounted_band_mass_on_refusers": {
                    p: mass(p, "A2b", "mounted_band_mass") for p in refusers},
            },
            "hit": None,
            "hit_note": (
                "three clauses, scored separately in the report; the "
                "0.29-0.44 figure is the lead's own written prediction and "
                "was never a threshold this order applies"),
        },
    ]


def build() -> dict[str, Any]:
    registration = rs1.read_registration(REGISTRATION)
    rows = arm_rows()
    g2 = g2_table(registration, rows)
    g3 = g3_table(registration, rows, g2["reproduced"])
    payload: dict[str, Any] = {
        "schema": "grm.rs1.results.v1",
        "program": "GRM",
        "phase": "RS1",
        "order": "orders/GRM_RS1_READ_STRENGTH_PROBE.md",
        "branch": "lc1-wip",
        "measurement_only": (
            "No production change. core/ and config/ were READ-ONLY for this "
            "order; every seam an arm needed lives in scripts/grm_rs1_*.py."),
        "registration": file_record(REGISTRATION),
        "arms": list(rs1.ARMS),
        "G2_a0_reproduces_eb1": g2,
        "G3_arms": g3,
        "predictions": score_predictions(
            g3["arms_correct_by_probe"], g3["table"]),
        "receipts": {
            f"{arm}:{session}": file_record(path)
            for (arm, session), path in sorted(latest_receipts().items())
        },
    }
    if AMENDMENT.is_file():
        payload["amendment"] = file_record(AMENDMENT)
    return payload


def main() -> int:
    payload = build()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / "grm_rs1_results.json"
    path.write_bytes(canonical_json_bytes(payload))
    print(f"results={path}")
    print(f"G2={payload['G2_a0_reproduces_eb1']['verdict']}")
    for row in payload["G3_arms"]["table"]:
        print(json.dumps(row))
    for row in payload["G3_arms"]["mechanism_verdicts"]:
        print(json.dumps(row))
    for row in payload["predictions"]:
        print(json.dumps({"arm": row["arm"], "hit": row["hit"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
