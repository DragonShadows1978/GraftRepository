#!/usr/bin/env python3
"""GRM-RS4 — assemble the gate tables and the verdicts from the receipts.

ORDER: ``orders/GRM_RS4_CEILING_REPARTITION.md`` (gates G2 and G3).

PURE ASSEMBLY.  This module runs no arm, takes no GPU lease and decides nothing
the registration did not already decide: it reads the arm receipts off disk,
reads the REGISTERED predictions and the frozen RS3 rows off the registration,
and applies the one to the other.  Every number it prints comes from a receipt
file it names.

G2 — THE BIT-IDENTITY GATE.  RS4 re-ran the arms, so its C0 rows must reproduce
RS3's C0 rows exactly: served text, correctness and every mass band at FLOAT
equality.  The bands compared are the ones off RS4's
``base_partition_UNMODIFIED`` block — the untouched
``LayerTypeMassObserver.partition()`` — so a pass says the extended instrument
perturbed nothing, which is the only basis on which RS4's new numbers can be
read beside RS3's.

G3 — THE RE-PARTITION.  For every probe x arm: the mount band, the four live
sub-bands, sink and learned sink; then the two comparable numbers side by side
and the residual between them.

THE RESIDUAL, stated carefully because the order's framing and RS3's differ.

  RS3's framing:  C5 WHOLE LIVE BAND (0.74)  minus  C3l MOUNT (0.41)  = 0.33
  RS4's framing:  C5 FED-TEXT ROWS           minus  C3l MOUNT        = ?

Both are reported on every row.  The registered prediction is judged on the
second.  ``refusers`` — the probes RS3 registered as the ones C0 refuses — are
the rows the headline mean is taken over, carried off RS3's registration rather
than chosen here.
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
from scripts.grm_rs4_ceiling_gpu import (  # noqa: E402
    ARTIFACT_DIR, REGISTRATION, assemble_table, read_registration,
)
from scripts.grm_rs4_registration import RS4Error  # noqa: E402

RS3_REGISTRATION = ROOT / "artifacts" / "grm_rs3" / "registration.json"

#: The mount arm the residual is taken against, the ceiling arm it is taken
#: from, and the reproduction arm.  All three named by the order.
MOUNT_ARM = "C3l"
CEILING_ARM = "C5"
BASELINE_ARM = "C0"

#: The bands G2 compares — the ones RS3 published on its C0 rows, all taken off
#: RS4's UNMODIFIED base partition.
G2_BANDS = ("mounted_mass_full_layers", "live_mass_full_layers",
            "sink_mass_full_layers", "learned_sink_mass_full_layers",
            "mounted_mass_sliding_layers", "live_mass_sliding_layers")


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def receipts() -> dict[str, Path]:
    """Every RS4 arm receipt on disk, keyed ``arm:session``."""
    out: dict[str, Path] = {}
    for path in sorted(ARTIFACT_DIR.glob("grm_rs4_C*_*.json")):
        payload = _read(path)
        out[f"{payload['arm']}:{payload['session_id']}"] = path
    return out


def all_rows() -> list[dict[str, Any]]:
    """EVERY probe row from every receipt, tagged with whether it is registered.

    Both kinds are kept, deliberately, and the two gates use different slices.

    A fixture probe with no registered work item is still SERVED, through the
    C0 path, so that the repository state the registered probes inherit matches
    C0's.  Those rows carry no arm contrast — every arm reports the identical
    numbers for them — so G3 ignores them.  But they ARE C0 rows, RS3 published
    them in its own G3 table, and RS4's registration therefore carries them as
    rows to reproduce.  Dropping them here would have made G2 report them as
    missing; keeping them makes G2 STRICTER than RS3's own G2 (which checked
    only the five registered probes) rather than weaker.
    """
    rows: list[dict[str, Any]] = []
    for _key, path in sorted(receipts().items()):
        payload = _read(path)
        for probe in payload["probes"]:
            entry = assemble_table([probe])[0]
            entry["session_id"] = str(payload["session_id"])
            entry["receipt"] = path.name
            entry["registered_probe"] = bool(probe.get("registered_probe"))
            rows.append(entry)
    return rows


def refusers() -> list[str]:
    """The refuser probe ids, carried off RS3's registration rather than chosen."""
    return [str(v) for v in _read(RS3_REGISTRATION)["refusers"]]


# ======================================================================
# G2 — C0 reproduces RS3 bit-equal
# ======================================================================
def gate_g2(registration: Mapping[str, Any],
            rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = dict(registration["rs3_c0_rows_RS4_MUST_REPRODUCE"])
    observed = {
        f"{row['probe_id']}:{row['variant']}": row
        for row in rows if str(row["arm"]) == BASELINE_ARM
    }
    table: list[dict[str, Any]] = []
    drift: list[str] = []
    for key in sorted(expected):
        want = expected[key]
        got = observed.get(key)
        if got is None:
            drift.append(f"{key}: no RS4 C0 row")
            table.append({"key": key, "reproduced": False,
                          "why": "no RS4 C0 row for this registered probe"})
            continue
        bands = {}
        for band in G2_BANDS:
            a, b = want.get(band), got.get(band)
            bands[band] = bool(
                a is not None and b is not None and float(a) == float(b))
        served_equal = str(want["served"]) == str(got["served"])
        correct_equal = bool(want["correct"]) == bool(got["correct"])
        reproduced = served_equal and correct_equal and all(bands.values())
        if not reproduced:
            drift.append(key)
        table.append({
            "key": key,
            "probe_id": str(want["probe_id"]),
            "variant": str(want["variant"]),
            "registered_probe": bool(got.get("registered_probe")),
            "reproduced": bool(reproduced),
            "served_equal": bool(served_equal),
            "correct_equal": bool(correct_equal),
            "bands_bit_equal": bands,
            "rs3_served": str(want["served"]),
            "rs4_served": str(got["served"]),
            "rs3_mounted_mass_full_layers": want["mounted_mass_full_layers"],
            "rs4_mounted_mass_full_layers": got["mounted_mass_full_layers"],
            "rs3_live_mass_full_layers": want["live_mass_full_layers"],
            "rs4_live_mass_full_layers": got["live_mass_full_layers"],
        })
    return {
        "gate": "G2",
        "rule": (
            "RS4's C0 reproduces RS3's C0 on every registered row: served "
            "text, correctness and every mass band at FLOAT equality. The "
            "bands come off RS4's base_partition_UNMODIFIED, so a pass says "
            "the extended instrument perturbed nothing."),
        "scope_note": (
            "RS3's own G2 checked its five REGISTERED probes. RS4's "
            "registration carried every C0 row off RS3's G3 table, which also "
            "holds the fixture probes served through the C0 path without a "
            "registered work item. Both kinds are checked here, so this gate "
            "is a superset of RS3's, and each row says which kind it is."),
        "rows_checked": len(table),
        "registered_rows_checked": int(sum(
            1 for r in table if r.get("registered_probe"))),
        "reproduced": [r["key"] for r in table if r.get("reproduced")],
        "drift": drift,
        "verdict": "PASS" if not drift else "FAIL",
        "table": table,
    }


# ======================================================================
# G3 — the re-partition table and the residual
# ======================================================================
def _pairs(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """One entry per ``probe:variant``, holding each arm's row."""
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row['probe_id']}:{row['variant']}"
        out.setdefault(key, {})[str(row["arm"])] = row
    return out


def residual_table(registration: Mapping[str, Any],
                   rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The two framings, side by side, one row per probe x variant.

    Every cell is read off a receipt.  The RS3 framing column is recomputed
    from RS4's own bit-equal numbers rather than copied from RS3, and the
    registration's carried RS3 pair is reported beside it so the two can be
    seen to agree.
    """
    carried = dict(registration["rs3_ceiling_pair_AS_RS3_FRAMED_IT"])
    refuser_ids = set(refusers())
    # G3 is an ARM CONTRAST, so it uses only the registered rows: on a
    # non-registered probe every arm serves the identical C0 row and a
    # "residual" between two copies of the same number would be a fiction.
    registered = [r for r in rows if r.get("registered_probe")]
    out: list[dict[str, Any]] = []
    for key, arms in sorted(_pairs(registered).items()):
        ceiling = arms.get(CEILING_ARM)
        mount = arms.get(MOUNT_ARM)
        if ceiling is None or mount is None:
            continue
        c5_live = float(ceiling["live_mass_full_layers"])
        c5_fed = float(ceiling["fed_text_mass_full_layers"])
        c5_question = float(ceiling["question_mass_full_layers"])
        c5_answer = float(ceiling["answer_mass_full_layers"])
        c3l_mount = float(mount["mounted_mass_full_layers"])
        c3l_live = float(mount["live_mass_full_layers"])
        c3l_question = float(mount["question_mass_full_layers"])
        c3l_answer = float(mount["answer_mass_full_layers"])
        out.append({
            "key": key,
            "probe_id": str(ceiling["probe_id"]),
            "variant": str(ceiling["variant"]),
            "session_id": str(ceiling.get("session_id", "")),
            "is_refuser": bool(str(ceiling["probe_id"]) in refuser_ids),
            # RS3's framing, recomputed from RS4's bit-equal numbers.
            "rs3_framing_c5_whole_live_band": c5_live,
            "rs3_framing_c3l_mount": c3l_mount,
            "rs3_framing_residual": c5_live - c3l_mount,
            "rs3_carried_residual": (
                carried.get(key, {}).get("rs3_framing_residual")),
            # RS4's framing.
            "c5_fed_text_mass": c5_fed,
            "c5_question_mass": c5_question,
            "c5_answer_mass": c5_answer,
            "c5_sink_mass": ceiling.get("sink_mass_full_layers"),
            "c5_learned_sink_mass": ceiling.get(
                "learned_sink_mass_full_layers"),
            "c3l_mount_mass": c3l_mount,
            "c3l_live_band_mass": c3l_live,
            "c3l_question_mass": c3l_question,
            "c3l_answer_mass": c3l_answer,
            "c3l_sink_mass": mount.get("sink_mass_full_layers"),
            "c3l_learned_sink_mass": mount.get(
                "learned_sink_mass_full_layers"),
            "rs4_residual_fed_minus_mount": c5_fed - c3l_mount,
            # The lead's hypothesis, checked directly and then checked again
            # in the shape the measurement says it should have had.
            "c5_question_minus_c3l_live_band": c5_question - c3l_live,
            "c5_question_plus_answer": c5_question + c5_answer,
            "c5_question_plus_answer_minus_c3l_live_band": (
                c5_question + c5_answer - c3l_live),
            "c5_correct": bool(ceiling["correct"]),
            "c3l_correct": bool(mount["correct"]),
        })
    return out


def _mean(values: Sequence[Any]) -> float | None:
    kept = [float(v) for v in values if v is not None]
    return (sum(kept) / len(kept)) if kept else None


def gate_g3(registration: Mapping[str, Any],
            rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    table = residual_table(registration, rows)
    refuser_rows = [r for r in table if r["is_refuser"]]
    predictions = registration["predictions_REGISTERED_BEFORE_ANY_GATE"]

    means = {
        name: _mean([r[name] for r in refuser_rows])
        for name in (
            "c5_fed_text_mass", "c5_question_mass", "c5_answer_mass",
            "rs3_framing_c5_whole_live_band", "c3l_mount_mass",
            "c3l_live_band_mass", "c3l_question_mass", "c3l_answer_mass",
            "rs3_framing_residual", "rs4_residual_fed_minus_mount",
            "c5_question_plus_answer",
            "c5_question_plus_answer_minus_c3l_live_band",
        )
    }

    p1 = predictions["P1_c5_question_row_mass_matches_mounted_live_band"]
    p2 = predictions["P2_c5_fed_text_mass"]
    p3 = predictions["P3_residual"]

    def _within(value: Any, centre: Any, tol: Any) -> bool:
        return bool(value is not None
                    and abs(float(value) - float(centre)) <= float(tol))

    residual = means["rs4_residual_fed_minus_mount"]
    if residual is None:
        p3_verdict = "NO DATA"
    elif residual <= float(p3["hit_if_at_most"]):
        p3_verdict = "HIT"
    elif residual >= float(p3["mechanism_stands_if_at_least"]):
        p3_verdict = "MISS — mechanism question stands as RS3 left it"
    else:
        p3_verdict = "INCONCLUSIVE — between the registered branches"

    verdicts = {
        "P1_c5_question_row_mass_matches_mounted_live_band": {
            "claim": p1["claim"],
            "registered_centre": p1["centre"],
            "registered_tolerance": p1["tolerance"],
            "measured": means["c5_question_mass"],
            "measured_c3l_live_band_mass": means["c3l_live_band_mass"],
            "verdict": (
                "HIT" if _within(means["c5_question_mass"],
                                 p1["centre"], p1["tolerance"]) else "MISS"),
            "why": (
                "The prediction read the mounted arms' ~0.29 live band as the "
                "question reading itself. MEASURED: that band is the question "
                "AND the answer together, because _clear_boat leaves the "
                "mounted arms with exactly those two things in the live band. "
                "C5's question rows alone are smaller. The comparison the "
                "prediction was reaching for is "
                "c5_question_plus_answer vs c3l_live_band_mass, reported as "
                "c5_question_plus_answer_minus_c3l_live_band."),
            "corrected_comparison": {
                "c5_question_plus_answer": means["c5_question_plus_answer"],
                "c3l_live_band_mass": means["c3l_live_band_mass"],
                "difference": means[
                    "c5_question_plus_answer_minus_c3l_live_band"],
            },
        },
        "P2_c5_fed_text_mass": {
            "claim": p2["claim"],
            "registered_centre": p2["centre"],
            "registered_tolerance": p2["tolerance"],
            "measured": means["c5_fed_text_mass"],
            "verdict": (
                "HIT" if _within(means["c5_fed_text_mass"],
                                 p2["centre"], p2["tolerance"]) else "MISS"),
        },
        "P3_residual": {
            "claim": p3["claim"],
            "hit_if_at_most": p3["hit_if_at_most"],
            "mechanism_stands_if_at_least": p3["mechanism_stands_if_at_least"],
            "measured": residual,
            "verdict": p3_verdict,
        },
    }
    return {
        "gate": "G3",
        "refuser_probe_ids": refusers(),
        "refuser_rows_counted": len(refuser_rows),
        "means_over_refusers_full_layers": means,
        "predictions": verdicts,
        "framings_side_by_side": {
            "rs3": (
                f"C5 whole live band {means['rs3_framing_c5_whole_live_band']} "
                f"minus C3l mount {means['c3l_mount_mass']} = "
                f"{means['rs3_framing_residual']}"),
            "rs4": (
                f"C5 FED-TEXT rows {means['c5_fed_text_mass']} minus "
                f"C3l mount {means['c3l_mount_mass']} = "
                f"{means['rs4_residual_fed_minus_mount']}"),
        },
        "table": table,
    }


# ======================================================================
# The partition-law audit
# ======================================================================
def partition_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every measured row's own partition checks, collected.

    The instrument checks itself on every row it produces; this reports the
    collection so a reader does not have to take "the arithmetic held" on
    trust.
    """
    subband_ok = [bool(r.get("subbands_sum_to_live_full")) for r in rows]
    partition_ok = [bool(r.get("partition_sums_to_one_full")) for r in rows]
    deltas: list[float] = []
    for row in rows:
        block = dict(row.get("recut_vs_base_abs_delta") or {})
        for layer in block.values():
            for value in dict(layer).values():
                if value is not None:
                    deltas.append(float(value))
    return {
        "rows": len(rows),
        "subbands_sum_to_live_all_rows": bool(all(subband_ok)),
        "partition_sums_to_one_all_rows": bool(all(partition_ok)),
        "recut_vs_base_max_abs_delta": (max(deltas) if deltas else None),
        "recut_vs_base_comparisons": len(deltas),
        "recut_vs_base_note": (
            "the maximum absolute difference between a band RS4 recomputed "
            "and the same band off the UNMODIFIED instrument, over every "
            "layer type and every row. Zero means the recut reproduced the "
            "production numbers exactly rather than approximately."),
    }


def build() -> dict[str, Any]:
    registration = read_registration()
    rows = all_rows()
    if not rows:
        raise RS4Error(f"no RS4 arm receipts found in {ARTIFACT_DIR}")
    return {
        "schema": "grm.rs4.results.v1",
        "program": "GRM",
        "phase": "RS4",
        "branch": "lc1-wip",
        "order": "orders/GRM_RS4_CEILING_REPARTITION.md",
        "why_a_rerun": (
            "RS3's receipts do not persist per-key-row attention: "
            "core/grm_demand.py::_full_mass softmaxes over the key rows and "
            "immediately sums them into four scalars per layer, and "
            "LayerTypeMassObserver keeps only those scalars. No RS3 artifact "
            "carries the per-row probabilities, so the live band could not be "
            "re-cut from disk at any granularity."),
        "G2_c0_reproduces_rs3": gate_g2(registration, rows),
        "G3_repartition": gate_g3(registration, rows),
        "partition_audit": partition_audit(rows),
        "table": rows,
        "registration": file_record(REGISTRATION),
        "receipts": {key: file_record(path)
                     for key, path in sorted(receipts().items())},
        "sources": {
            str(Path(p).relative_to(ROOT)): file_record(Path(p))
            for p in (
                ROOT / "scripts" / "grm_rs4_registration.py",
                ROOT / "scripts" / "grm_rs4_row_split.py",
                ROOT / "scripts" / "grm_rs4_ceiling_gpu.py",
                ROOT / "scripts" / "grm_rs4_results.py",
                ROOT / "tests" / "test_grm_rs4_row_split.py",
            )
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    payload = build()
    target = ARTIFACT_DIR / "grm_rs4_results.json"
    target.write_bytes(canonical_json_bytes(payload))
    print(f"results={target}")
    print(json.dumps({
        "G2": {
            "verdict": payload["G2_c0_reproduces_rs3"]["verdict"],
            "rows_checked": payload["G2_c0_reproduces_rs3"]["rows_checked"],
            "drift": payload["G2_c0_reproduces_rs3"]["drift"],
        },
        "G3_means": payload["G3_repartition"][
            "means_over_refusers_full_layers"],
        "G3_predictions": {
            name: {"measured": block["measured"], "verdict": block["verdict"]}
            for name, block in payload["G3_repartition"][
                "predictions"].items()
        },
        "framings": payload["G3_repartition"]["framings_side_by_side"],
        "partition_audit": payload["partition_audit"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
