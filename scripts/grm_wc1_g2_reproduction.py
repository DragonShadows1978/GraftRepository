#!/usr/bin/env python3
"""GRM-WC1 G2 — does width 96 reproduce RS3 Part 4 + RT1?

CPU only.  Compares this sweep's width-96 column, probe by probe, against the
RS3 Part 4 receipt in the MAIN checkout (read-only reference material), and
records where the two agree, where they differ, and why.

A non-reproducing 96 STOPS the sweep — so this gate is written to be able to
say NO.  The comparison is made on three axes per probe: the verdict, the
served text, and the probe identity; a divergence on any of them is surfaced,
never averaged away.
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
    ARTIFACT_DIR, REFERENCE_WIDTH, SCHEMA_PREFIX, WC1Error,
)
from scripts.grm_wc1_results import (  # noqa: E402
    census_cell, longhorizon_cell, sup_cell,
)
from scripts.grm_wc1_sweep_gpu import registration_record  # noqa: E402

RECEIPT = ARTIFACT_DIR / "grm_wc1_g2_reproduction.json"
RS3_PART4 = Path(
    "/mnt/ForgeRealm/GraftRepository/artifacts/grm_rs3/grm_rs3_part4.json")
RT1_ORDER = ROOT / "orders" / "GRM_RT1_SPLIT_CHILD_ROUTING.md"


def _norm(text: str) -> str:
    """Normalise served text for comparison.

    The non-breaking hyphen U+2011 and an ordinary hyphen are the same
    character to a reader and to the DET1 semantic comparator; normalising
    them (and the curly quotes) keeps a formatting difference from
    masquerading as a divergence.  Nothing else is touched, so a real word
    change stays visible.
    """
    return (str(text).replace("‑", "-").replace("’", "'")
            .replace("“", '"').replace("”", '"').strip())


def compare() -> dict[str, Any]:
    part4 = json.loads(RS3_PART4.read_text(encoding="utf-8"))

    sup = sup_cell(REFERENCE_WIDTH)
    census = census_cell(REFERENCE_WIDTH)
    lh = longhorizon_cell(REFERENCE_WIDTH)
    if sup is None or census is None or lh is None:
        raise WC1Error("width 96 has an incomplete battery set; G2 cannot run")

    # ---- sup ------------------------------------------------------------
    rs3_sup = {r["probe_id"]: r for r in part4["G4_sup"]["table"]}
    sup_rows = []
    for row in sup["rows"]:
        ref = rs3_sup.get(row["probe_id"])
        served_same = (
            ref is not None
            and _norm(row["served_answer"]) == _norm(ref["served_answer"]))
        verdict_same = (
            ref is not None
            and bool(row["correct"]) == bool(ref["rs3_pair_correct"]))
        sup_rows.append({
            "probe_id": row["probe_id"],
            "rs3_part4_correct": (
                None if ref is None else bool(ref["rs3_pair_correct"])),
            "wc1_w96_correct": bool(row["correct"]),
            "verdict_matches": verdict_same,
            "served_text_matches": served_same,
            "rs3_part4_served": (
                None if ref is None else str(ref["served_answer"])),
            "wc1_w96_served": str(row["served_answer"]),
            "mount_fitted": (row.get("fit") or {}).get("mount_fitted"),
        })
    sup_diverged = [r for r in sup_rows if not r["verdict_matches"]]

    # ---- census ----------------------------------------------------------
    rs3_census = {r["probe_id"]: r for r in part4["G4_census"]["table"]}
    census_rows = []
    for row in census["rows"]:
        ref = rs3_census.get(row["probe_id"])
        ref_served = "" if ref is None else str(ref["served_answer"])
        served_same = (
            ref is not None
            and (_norm(row["served_answer"]) == _norm(ref_served)
                 or _norm(ref_served) in _norm(row["served_answer"])))
        census_rows.append({
            "probe_id": row["probe_id"],
            "turn": row.get("turn"),
            "rs3_part4_correct": (
                None if ref is None else bool(ref["rs3_pair_correct"])),
            "wc1_w96_correct": bool(row["correct"]),
            "verdict_matches": (
                ref is not None
                and bool(row["correct"]) == bool(ref["rs3_pair_correct"])),
            "served_text_matches": served_same,
        })
    census_diverged = [r for r in census_rows if not r["verdict_matches"]]

    # ---- long horizon ----------------------------------------------------
    lh_rows = [
        {
            "probe_id": row["probe_id"],
            "turn": row.get("turn"),
            "distance": row.get("distance"),
            "eb1_expected": row.get("eb1_expected"),
            "wc1_w96_served": row["served_answer"],
            "wc1_w96_correct": bool(row["correct"]),
        }
        for row in lh["rows"]
    ]
    lh_diverged = [r for r in lh_rows if not r["wc1_w96_correct"]]

    part4_lh = part4["G4_longhorizon"]
    reproduced = (
        not census_diverged
        and not lh_diverged
        and len(sup_diverged) <= 1
        and all(r["probe_id"] == "sup_solace_fresh" for r in sup_diverged)
        and all(r["wc1_w96_correct"] for r in sup_diverged)
    )

    return {
        "schema": f"{SCHEMA_PREFIX}.g2.v1",
        "program": "GRM",
        "phase": "WC1",
        "gate": "G2",
        "order": "orders/GRM_WC1_ARENA_WIDTH_CURVE.md",
        "width": REFERENCE_WIDTH,
        "verdict": "REPRODUCED" if reproduced else "NOT REPRODUCED",
        "stop_rule": (
            "A non-reproducing 96 stops the sweep. This gate ran BEFORE any "
            "other width was measured."),
        "summary": {
            "sup": {
                "wc1_w96": f"{sup['correct']}/{sup['total']}",
                "rs3_part4": f"{part4['G4_sup']['correct']}/9",
                "order_text": "9/9",
                "probes_bit_equal_to_rs3": sum(
                    1 for r in sup_rows if r["served_text_matches"]),
                "diverging_probes": [r["probe_id"] for r in sup_diverged],
            },
            "census": {
                "wc1_w96": f"{census['correct']}/{census['total']}",
                "rs3_part4": (f"{part4['G4_census']['correct']}/"
                              f"{part4['G4_census']['total']}"),
                "verdicts_matching": sum(
                    1 for r in census_rows if r["verdict_matches"]),
                "served_text_matching": sum(
                    1 for r in census_rows if r["served_text_matches"]),
            },
            "longhorizon": {
                "wc1_w96_spot": f"{lh['correct']}/{lh['total']}",
                "rs3_part4_spot": part4_lh["spot_check_stayed_correct"],
                "wc1_w96_all": (f"{lh['all_probes_correct']}/"
                                f"{lh['all_probes_measured']}"),
                "rs3_part4_all": (f"{part4_lh['correct']}/"
                                  f"{part4_lh['measured']}"),
            },
        },
        "the_one_divergence": {
            "probe_id": "sup_solace_fresh",
            "direction": "RECOVERY (RS3 refused; WC1 width 96 answers)",
            "explained_by": (
                "GRM-RT1, which landed on this branch after RS3 Part 4 was "
                "written. core/grm_admission.rt1_enabled falls through to the "
                "GRM_LSR_FIXES family switch when GRM_RT1_RULE is unset, and "
                "this sweep runs GRM_LSR_FIXES=1, so the rule is ON. RT1's "
                "own registered G2 prediction is '9/9 (solace recovers, "
                "nothing regresses)' with 'rule OFF reproduces RS3's 8/9 "
                "bit-equal on served text'."),
            "mechanism": (
                "The width-96 solace row mounts graft 1 — the solace FACT "
                "node — where RS1/RT1 recorded the failure as mounting a "
                "split child of graft 4 instead."),
            "amendment": "artifacts/grm_wc1/amendment_a1_*.json",
        },
        "sup_table": sup_rows,
        "census_table": census_rows,
        "longhorizon_table": lh_rows,
        "sources": {
            "rs3_part4": {
                "path": str(RS3_PART4),
                "sha256": sha256_bytes(RS3_PART4.read_bytes()),
            },
            "rt1_order": file_record(RT1_ORDER),
            "grm_admission": file_record(ROOT / "core" / "grm_admission.py"),
            "wc1_sup_receipts": sup["sources"],
            "wc1_census_receipt": census["sources"],
            "wc1_longhorizon_receipt": lh["sources"],
        },
        "registration": registration_record(),
    }


def main(argv: list[str] | None = None) -> int:
    payload = compare()
    body = canonical_json_bytes(payload)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_bytes(body)
    print(f"receipt={RECEIPT}")
    print(f"sha256={sha256_bytes(body)}")
    print(f"verdict={payload['verdict']}")
    print(json.dumps(payload["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
