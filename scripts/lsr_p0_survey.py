#!/usr/bin/env python3
"""LSR Phase 0 — CPU survey of the frozen lived-serving snapshots.

Program: LSR (Lived-Serving Reliability).  Spec: docs/LSR_LIVED_SERVING_PLAN.md

WHY THIS EXISTS.  Phase 0 asks where the wrong value physically comes from.
Before spending a GPU lease re-serving eight probes, this script reads the
window state the DET1 campaign ALREADY froze for every one of them
(grm.det1_3.model_visible_snapshot.v2 manifests captured at
phase="before_probe_prefill") and answers the parts of the question that are
decidable from text and layout alone:

  * the derived window layout per probe (anchor / arena / recent / live);
  * whether the RECENT-TURNS region exists at all at the probe;
  * which session node carries the SERVED value and which carries the
    EXPECTED value, and which of them the arena actually mounted.

It does NOT emit a value-provenance verdict.  The plan's verdicts require the
attention-mass leg (plurality readout mass), which needs the engine's own
softmax operands and therefore the GPU.  Anything this script cannot decide is
reported NOT_MEASURED with the lead script path, per the order.

Everything here is CPU-only, read-only, and append-only.  The DET1 snapshots
are INBOUND REFERENCES: nothing is written back into any DET envelope.

Run:  PYTHONPATH=/mnt/ForgeRealm/GraftRepository \
        python3 scripts/lsr_p0_survey.py --emit
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_cmc1_mechanism import canonical_json_bytes, sha256_bytes  # noqa: E402
from scripts.grm_det1_common import file_record  # noqa: E402
from scripts.lsr_p0_core import (  # noqa: E402
    ARTIFACT_DIR,
    CENSUS,
    DET1_RUN,
    PLAN,
    REGION_ARENA,
    REGION_LIVE,
    REGION_RECENT,
    SCHEMA_PREFIX,
    VERDICT_NOT_MEASURED,
    LSRError,
    enumerate_probes,
    layout_from_snapshot_state,
    probe_session_map,
    session_node_inventory,
    value_presence_across_session,
)

LEAD_SCRIPT = ROOT / "scripts" / "lsr_p0_lead_gpu.sh"


def _snapshot_manifests(probe_id: str, run_dir: Path = DET1_RUN) -> list[Path]:
    pattern = str(run_dir / "**" / "snapshots" / probe_id / "rung_*" / "manifest.json")
    return sorted(Path(value) for value in glob.glob(pattern, recursive=True))


def _state_scalar(state: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = state.get(key, default)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return value


def _text_level_class(
    recent_present: bool, in_mounted_node: bool, anywhere_in_session: bool,
) -> dict[str, Any]:
    """A TEXT-LEVEL observation, deliberately NOT one of the plan's verdicts.

    The plan's vocabulary (RECENT-TURNS / ARENA / ABSENT) is reserved for the
    full witness.  This field records only what the frozen layout and the
    session text establish, in its own words, so the receipt never launders a
    text observation into an adjudicated verdict.
    """
    return {
        "recent_turns_region_exists_at_probe": recent_present,
        "served_value_text_is_in_a_mounted_arena_node": in_mounted_node,
        "served_value_text_exists_anywhere_in_the_session": anywhere_in_session,
        "note": (
            "text-level only; not a plan verdict.  RECENT-TURNS is "
            "unreachable for any probe whose recent region is empty, because "
            "the region has no tokens to carry the value or the mass."
        ),
    }


def survey_probe(probe_id: str, served_value: str | None,
                 expected_value: str | None) -> dict[str, Any]:
    """Survey every frozen snapshot of one probe."""
    manifests = _snapshot_manifests(probe_id)
    if not manifests:
        return {
            "probe_id": probe_id,
            "snapshots_found": 0,
            "window_shapes": [],
            "status": VERDICT_NOT_MEASURED,
            "reason": "no frozen DET1.3 snapshot exists for this probe",
            "value_provenance_verdict": VERDICT_NOT_MEASURED,
            "lead_script": str(LEAD_SCRIPT.relative_to(ROOT)),
        }

    distinct: dict[str, dict[str, Any]] = {}
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        state = manifest["state"]
        layout = layout_from_snapshot_state(state)
        key = canonical_json_bytes({
            "regions": layout["regions"],
            "mounts": _state_scalar(state, "arena.cur_mounts"),
        }).decode("utf-8")
        row = distinct.setdefault(key, {
            "window_layout": layout,
            "phase": str(manifest.get("phase", "")),
            "cur_mounts": _state_scalar(state, "arena.cur_mounts"),
            "arena_pos": _state_scalar(state, "arena.pos"),
            "live_segments": _state_scalar(state, "live.segments"),
            "prompt_ntok": _state_scalar(state, "tokens.prompt_count"),
            "observed_in": [],
        })
        row["observed_in"].append(str(path.relative_to(ROOT)))

    shapes = list(distinct.values())
    recent_present = any(
        shape["window_layout"]["regions"][REGION_RECENT]["ntok"] > 0
        for shape in shapes
    )

    served_presence = (
        value_presence_across_session(probe_id, served_value)
        if served_value else None
    )
    expected_presence = (
        value_presence_across_session(probe_id, expected_value)
        if expected_value else None
    )
    inventory = session_node_inventory(probe_id)

    mounted: set[int] = set()
    for shape in shapes:
        for value in (shape["cur_mounts"] or []):
            mounted.add(int(value))

    served_carriers = set(
        served_presence["carrying_graft_indices"]) if served_presence else set()
    expected_carriers = set(
        expected_presence["carrying_graft_indices"]) if expected_presence else set()

    return {
        "probe_id": probe_id,
        "snapshots_found": len(manifests),
        "distinct_window_shapes": len(shapes),
        "phase": shapes[0]["phase"],
        "window_shapes": shapes,
        "recent_turns_region_present": recent_present,
        "mounted_graft_indices": sorted(mounted),
        "session_id": inventory["session_id"],
        "session_nodes": inventory["nodes"],
        "served_value": served_value,
        "expected_value": expected_value,
        "served_value_carrying_nodes": sorted(served_carriers),
        "expected_value_carrying_nodes": sorted(expected_carriers),
        "served_value_in_a_mounted_node": bool(served_carriers & mounted),
        "expected_value_in_a_mounted_node": bool(expected_carriers & mounted),
        "served_value_present_anywhere_in_session_text": bool(
            served_presence and served_presence["present_in_any_node"]),
        "text_level_provenance": _text_level_class(
            recent_present, bool(served_carriers & mounted),
            bool(served_presence and served_presence["present_in_any_node"])),
        "value_provenance_verdict": VERDICT_NOT_MEASURED,
        "value_provenance_verdict_reason": (
            "the plan's verdict requires the plurality-readout-mass leg, "
            "which needs the engine's own softmax operands (GPU)"
        ),
        "lead_script": str(LEAD_SCRIPT.relative_to(ROOT)),
    }


def survey(census_path: Path = CENSUS) -> dict[str, Any]:
    enumerated = enumerate_probes(census_path)
    bindings = probe_session_map()
    rows: list[dict[str, Any]] = []
    for group in ("wrong_value_probes", "lawful_controls"):
        for probe in enumerated[group]:
            probe_id = str(probe["probe_id"])
            expected = (probe["expected_values"] or [None])[0]
            row = survey_probe(probe_id, probe.get("served_value"), expected)
            row["lsr_role"] = str(probe["lsr_role"])
            row["census_class"] = str(probe["census_class"])
            row["served_answer"] = str(probe["served_answer"])
            rows.append(row)

    wrong = [row for row in rows if row["lsr_role"] == "wrong_value"]
    controls = [row for row in rows if row["lsr_role"] == "lawful_control"]
    recent_bearing = [row for row in wrong if row["recent_turns_region_present"]]

    return {
        "schema": f"{SCHEMA_PREFIX}.snapshot_survey.v1",
        "program": "LSR",
        "phase": "0",
        "scope_note": (
            "Reads the DET1 campaign's already-frozen before_probe_prefill "
            "snapshots as inbound references.  Establishes window layout and "
            "text-level value location.  Emits NO value-provenance verdict: "
            "the plan's verdicts need the attention-mass leg (GPU)."
        ),
        "plan": file_record(PLAN),
        "census": file_record(census_path),
        "sources": {
            "lsr_p0_core": file_record(ROOT / "scripts" / "lsr_p0_core.py"),
            "lsr_p0_survey": file_record(Path(__file__).resolve()),
        },
        "declares_in_det_envelope": False,
        "probe_enumeration": enumerated,
        "session_bindings": {
            key: {k: v for k, v in value.items() if k != "fixture_file"}
            for key, value in bindings.items()
        },
        "probes": rows,
        "summary": {
            "wrong_value_probes": len(wrong),
            "lawful_controls": len(controls),
            "wrong_value_probes_with_a_nonempty_recent_region":
                len(recent_bearing),
            "wrong_value_probes_whose_served_value_is_in_a_mounted_node":
                sum(1 for row in wrong if row["served_value_in_a_mounted_node"]),
            "wrong_value_probes_whose_served_value_is_absent_from_the_session":
                sum(1 for row in wrong
                    if not row["served_value_present_anywhere_in_session_text"]),
            "wrong_value_probes_whose_expected_value_was_never_mounted":
                sum(1 for row in wrong
                    if not row["expected_value_in_a_mounted_node"]),
        },
        "h_lsr_1_reachability": {
            "leg": (
                "H-LSR-1 requires the served value present in RECENT TURNS "
                "with plurality readout mass on those tokens"
            ),
            "wrong_value_probes_where_that_leg_is_structurally_reachable":
                len(recent_bearing),
            "note": (
                "A probe whose recent-turns region holds zero tokens cannot "
                "satisfy either leg: there are no recent-turn tokens to carry "
                "the value or to receive mass.  This is a statement about the "
                "frozen capture, not an adjudication."
            ),
        },
        "value_provenance_verdicts": VERDICT_NOT_MEASURED,
        "value_provenance_verdicts_reason": (
            "the plurality-readout-mass leg needs the engine's softmax "
            "operands; run the lead GPU script"
        ),
        "lead_script": str(LEAD_SCRIPT.relative_to(ROOT)),
    }


def render(result: Mapping[str, Any]) -> str:
    lines = ["LSR Phase 0 - frozen-snapshot survey", ""]
    header = (
        f"{'probe':32s} {'role':14s} {'anch':>4s} {'aren':>4s} "
        f"{'rcnt':>4s} {'live':>4s} {'mounts':>8s}  served/expected nodes"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in result["probes"]:
        if not row.get("window_shapes"):
            lines.append(f"{row['probe_id']:32s} NO SNAPSHOT")
            continue
        regions = row["window_shapes"][0]["window_layout"]["regions"]
        lines.append(
            f"{row['probe_id']:32s} {row['lsr_role']:14s} "
            f"{regions['anchor']['ntok']:4d} {regions[REGION_ARENA]['ntok']:4d} "
            f"{regions[REGION_RECENT]['ntok']:4d} {regions[REGION_LIVE]['ntok']:4d} "
            f"{str(row['mounted_graft_indices']):>8s}  "
            f"served={row['served_value_carrying_nodes']} "
            f"expected={row['expected_value_carrying_nodes']}"
        )
    summary = result["summary"]
    lines += [
        "",
        f"wrong-value probes: {summary['wrong_value_probes']}",
        "  with a non-empty recent-turns region: "
        f"{summary['wrong_value_probes_with_a_nonempty_recent_region']}",
        "  served value inside a MOUNTED arena node: "
        f"{summary['wrong_value_probes_whose_served_value_is_in_a_mounted_node']}",
        "  served value absent from the whole session: "
        f"{summary['wrong_value_probes_whose_served_value_is_absent_from_the_session']}",
        "  expected value never mounted: "
        f"{summary['wrong_value_probes_whose_expected_value_was_never_mounted']}",
        "",
        f"value-provenance verdicts: {result['value_provenance_verdicts']}"
        f" ({result['value_provenance_verdicts_reason']})",
        f"lead script: {result['lead_script']}",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSR Phase 0 snapshot survey")
    parser.add_argument("--emit", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = survey()
    blob = canonical_json_bytes(result)
    if args.json:
        print(blob.decode("utf-8"))
    else:
        print(render(result))
    if args.emit:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = ARTIFACT_DIR / f"lsr_p0_survey_{sha256_bytes(blob)[:16]}.json"
        if path.exists():
            if path.read_bytes() != blob:
                raise LSRError(f"content-address collision: {path}")
        else:
            with path.open("xb") as handle:
                handle.write(blob)
        print(f"\nreceipt {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
