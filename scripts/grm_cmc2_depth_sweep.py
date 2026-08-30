#!/usr/bin/env python3
"""CPU rails, adjudication, and report renderer for GRM-CMC2.

The registered predictions are deliberately strict and frozen before the
live sweep:

* P1 holds only when both INT8 and INT6 repair the bf16 Praxis wrong-read
  and both reduce decisive-head sibling-outlier mass concentration.
* P2 holds only when both INT3 and INT2 raise mounted-graft entropy above
  bf16 and both degrade the production-L2 supersession read.

One-depth and mixed signatures remain in the raw table but do not get
promoted into P1/P2.  Final vocabulary is exactly LAW-FUSED / HALF-LAW /
REFUTED.  This module is CPU-only; ``grm_cmc2_gpu_sweep.py`` produces the
live depth receipts consumed by the ``adjudicate`` subcommand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from grm_cmc1_mechanism import (  # noqa: E402
    assert_finite,
    canonical_json_bytes,
    append_report_once,
    sha256_file,
    write_content_addressed,
)
from core.graft_quant import (  # noqa: E402
    GROUP_SIZE,
    pack_kv_arrays,
    quantize_dequantize_symmetric_group32,
    unpack_kv_arrays,
)


DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "grm_cmc2"
REPORT = DEFAULT_ARTIFACT_DIR / "GRM_CMC2_REPORT.md"
ORDER = ROOT / "orders" / "GRM_CMC2_DEPTH_SWEEP.md"
ADDENDUM = ROOT / "orders" / "GRM_CMC2_ADDENDUM.md"
QUANT_LEDGER = ROOT / "docs" / "GRM_GRAFT_QUANT_LEDGER.md"

DEPTHS: tuple[tuple[str, int], ...] = (
    ("bf16", 16),
    ("int8", 8),
    ("int6", 6),
    ("int4", 4),
    ("int3", 3),
    ("int2", 2),
)
DEPTH_LABELS = tuple(label for label, _bits in DEPTHS)
P1_DEPTHS = ("int8", "int6")
P2_DEPTHS = ("int3", "int2")

PRODUCTION_FRAME: dict[str, Any] = {
    "name": "F-PROD-LADDER-ON-L2-ON",
    "probe_ladder": True,
    "supersession_l2": True,
    "probe_ladder_pin": "explicit_on",
    "supersession_l2_pin": "explicit_on",
    "alignment": "GRM-ADM1.2 production-default alignment",
}


class CMC2Error(RuntimeError):
    """A receipt or registered CMC2 rule is invalid."""


def _ndarray_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finite_array(name: str, value: np.ndarray) -> None:
    if not np.issubdtype(value.dtype, np.number):
        raise CMC2Error(f"{name}: expected numeric dtype, got {value.dtype}")
    if not np.isfinite(value.astype(np.float32)).all():
        raise CMC2Error(f"{name}: non-finite payload")


def roundtrip_arrays(
    arrays: Mapping[str, np.ndarray], storage_bits: int,
    *, group_size: int = GROUP_SIZE,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Round-trip one format-1 payload and content-validate it.

    bf16 is represented by the repository's plain fp16-at-rest identity path
    (``storage_bits=16``).  Other depths exercise the exact packed format's
    pack/unpack path and are compared bit-for-bit with the frozen quant sweep
    transform.  The function returns reconstructed arrays plus a JSON-safe
    receipt and never imports or touches CUDA.
    """
    source = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
    if not source:
        raise CMC2Error("round-trip payload is empty")
    for name, value in source.items():
        if value.dtype != np.float16:
            raise CMC2Error(f"{name}: expected fp16 at rest, got {value.dtype}")
        if value.shape[-1] % int(group_size):
            raise CMC2Error(
                f"{name}: last axis {value.shape[-1]} is not group-{group_size}")
        _finite_array(name, value)

    if int(storage_bits) == 16:
        reconstructed = {name: value.copy() for name, value in source.items()}
        packed_format = "plain_fp16_identity"
    else:
        packed = pack_kv_arrays(source, int(storage_bits), int(group_size))
        reconstructed = unpack_kv_arrays(packed, list(source))
        packed_format = "grm_graft_quant_format_v1"

    tensors: dict[str, Any] = {}
    all_valid = True
    for name, original in source.items():
        rebuilt = np.ascontiguousarray(reconstructed[name])
        _finite_array(name, rebuilt)
        expected = quantize_dequantize_symmetric_group32(
            original, int(storage_bits), int(group_size))
        diff = original.astype(np.float32) - rebuilt.astype(np.float32)
        content_equal = bool(np.array_equal(rebuilt, expected))
        identity_equal = bool(np.array_equal(rebuilt, original))
        valid = bool(
            rebuilt.dtype == original.dtype
            and rebuilt.shape == original.shape
            and content_equal
        )
        all_valid = all_valid and valid
        tensors[name] = {
            "shape": list(original.shape),
            "dtype": str(original.dtype),
            "source_sha256": _ndarray_sha256(original),
            "reconstructed_sha256": _ndarray_sha256(rebuilt),
            "reference_transform_sha256": _ndarray_sha256(expected),
            "content_equal_reference": content_equal,
            "identity_equal_source": identity_equal,
            "rmse": float(np.sqrt(np.mean(diff * diff))),
            "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
            "pass": valid,
        }

    receipt = {
        "schema": "grm.cmc2.roundtrip.v1",
        "frame": dict(PRODUCTION_FRAME),
        "storage_bits": int(storage_bits),
        "group_size": int(group_size),
        "packed_format": packed_format,
        "reference": "core.graft_quant.quantize_dequantize_symmetric_group32",
        "tensors": tensors,
        "pass": bool(all_valid),
    }
    assert_finite(receipt)
    return reconstructed, receipt


def _entropy(probabilities: Sequence[float]) -> tuple[float, float]:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1 or probs.size < 2 or not np.isfinite(probs).all():
        raise CMC2Error("entropy expects at least two finite probabilities")
    if np.any(probs < 0.0) or float(probs.sum()) <= 0.0:
        raise CMC2Error("entropy probabilities must be nonnegative with positive mass")
    probs = probs / probs.sum()
    nz = probs[probs > 0.0]
    nats = float(-np.sum(nz * np.log(nz)))
    normalized = float(nats / math.log(float(probs.size)))
    return nats, normalized


def summarize_mass_split(
    attention: Mapping[str, Any], *, target: str, sibling: str,
) -> dict[str, Any]:
    """Add registered entropy statistics to the trued CMC1.2 witness.

    ``target_vs_sibling_entropy_normalized`` is the binary split requested by
    the order.  ``mounted_graft_entropy_normalized`` is also reported because
    P2 explicitly names flattening toward uniform *across grafts*.
    """
    heads = list(attention.get("decisive_heads") or ())
    if not heads:
        raise CMC2Error("attention receipt has no decisive heads")
    names = sorted({
        str(name)
        for head in heads
        for name in (head.get("graft_mass") or {}).keys()
    })
    if target not in names or sibling not in names or len(names) < 2:
        raise CMC2Error(
            f"mass split lacks target/sibling: target={target} sibling={sibling} names={names}")

    per_head = []
    mass_samples = {name: [] for name in names}
    binary_entropy = []
    graft_entropy = []
    concentrations = []
    for head in heads:
        graft_mass = {
            name: float((head.get("graft_mass") or {}).get(name, 0.0))
            for name in names
        }
        for name, value in graft_mass.items():
            if not math.isfinite(value) or value < 0.0:
                raise CMC2Error(f"invalid graft mass {name}={value}")
            mass_samples[name].append(value)
        binary_nats, binary_norm = _entropy(
            [graft_mass[target], graft_mass[sibling]])
        graft_nats, graft_norm = _entropy([graft_mass[name] for name in names])
        concentration = float(head["sibling_mass_above_own_p99_fraction"])
        binary_entropy.append(binary_norm)
        graft_entropy.append(graft_norm)
        concentrations.append(concentration)
        per_head.append({
            "layer": int(head["layer"]),
            "head": int(head["head"]),
            "graft_mass": graft_mass,
            "target_vs_sibling_entropy_nats": binary_nats,
            "target_vs_sibling_entropy_normalized": binary_norm,
            "mounted_graft_entropy_nats": graft_nats,
            "mounted_graft_entropy_normalized": graft_norm,
            "sibling_mass_above_own_p99_fraction": concentration,
        })

    result = {
        "decisive_head_count": len(heads),
        "graft_names": names,
        "mean_captured_mass": {
            name: float(np.mean(values)) for name, values in mass_samples.items()
        },
        "target": target,
        "sibling": sibling,
        "target_vs_sibling_entropy_normalized_mean": float(np.mean(binary_entropy)),
        "target_vs_sibling_entropy_normalized_min": float(np.min(binary_entropy)),
        "target_vs_sibling_entropy_normalized_max": float(np.max(binary_entropy)),
        "mounted_graft_entropy_normalized_mean": float(np.mean(graft_entropy)),
        "mounted_graft_entropy_normalized_min": float(np.min(graft_entropy)),
        "mounted_graft_entropy_normalized_max": float(np.max(graft_entropy)),
        "sibling_outlier_mass_concentration_mean": float(np.mean(concentrations)),
        "sibling_outlier_mass_concentration_max": float(np.max(concentrations)),
        "concentrated_decisive_heads": int(sum(value >= 0.5 for value in concentrations)),
        "per_head": per_head,
    }
    assert_finite(result)
    return result


def read_outcome(classification: str) -> str:
    """Map the registered CMC answer classes to the CMC2 three-way row."""
    if classification == "correct":
        return "repaired"
    if classification == "wrong-fact":
        return "confused"
    return "collapsed"


def _supersession_quality(row: Mapping[str, Any]) -> int:
    classification = str(row.get("classification", ""))
    return {"correct": 2, "stale": 1, "wrong-fact": 1}.get(classification, 0)


def adjudicate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply only the registered P1/P2 logic; never smooth mixed rows."""
    by_depth = {str(row.get("depth")): row for row in rows}
    if tuple(by_depth) != DEPTH_LABELS or len(rows) != len(DEPTH_LABELS):
        raise CMC2Error(
            f"depth rows must be exactly {DEPTH_LABELS} in order, got {tuple(by_depth)}")
    baseline = by_depth["bf16"]
    baseline_concentration = float(
        baseline["mass_split"]["sibling_outlier_mass_concentration_mean"])
    baseline_entropy = float(
        baseline["mass_split"]["mounted_graft_entropy_normalized_mean"])
    baseline_sup_quality = _supersession_quality(baseline["supersession"])

    p1_rows: dict[str, Any] = {}
    for depth in P1_DEPTHS:
        row = by_depth[depth]
        concentration = float(
            row["mass_split"]["sibling_outlier_mass_concentration_mean"])
        repaired = row["read"]["outcome"] == "repaired"
        reduced = concentration < baseline_concentration
        p1_rows[depth] = {
            "repaired": repaired,
            "sibling_outlier_concentration": concentration,
            "bf16_sibling_outlier_concentration": baseline_concentration,
            "concentration_reduced": reduced,
            "signature": bool(repaired and reduced),
        }
    p1_holds = all(value["signature"] for value in p1_rows.values())

    p2_rows: dict[str, Any] = {}
    for depth in P2_DEPTHS:
        row = by_depth[depth]
        entropy = float(
            row["mass_split"]["mounted_graft_entropy_normalized_mean"])
        quality = _supersession_quality(row["supersession"])
        entropy_rises = entropy > baseline_entropy
        supersession_degrades = quality < baseline_sup_quality
        p2_rows[depth] = {
            "mounted_graft_entropy": entropy,
            "bf16_mounted_graft_entropy": baseline_entropy,
            "entropy_rises": entropy_rises,
            "supersession_quality": quality,
            "bf16_supersession_quality": baseline_sup_quality,
            "supersession_degrades": supersession_degrades,
            "signature": bool(entropy_rises and supersession_degrades),
        }
    p2_holds = all(value["signature"] for value in p2_rows.values())

    if p1_holds and p2_holds:
        verdict = "LAW-FUSED"
    elif p1_holds or p2_holds:
        verdict = "HALF-LAW"
    else:
        verdict = "REFUTED"

    raw_signatures = {}
    for depth, row in by_depth.items():
        concentration = float(
            row["mass_split"]["sibling_outlier_mass_concentration_mean"])
        entropy = float(
            row["mass_split"]["mounted_graft_entropy_normalized_mean"])
        raw_signatures[depth] = {
            "read_repaired": row["read"]["outcome"] == "repaired",
            "outlier_concentration_reduced_vs_bf16": concentration < baseline_concentration,
            "entropy_rises_vs_bf16": entropy > baseline_entropy,
            "supersession_degrades_vs_bf16": (
                _supersession_quality(row["supersession"]) < baseline_sup_quality),
        }

    result = {
        "P1": {
            "holds": p1_holds,
            "required_depths": list(P1_DEPTHS),
            "rows": p1_rows,
            "rule": (
                "both INT8 and INT6 repair the bf16 wrong-read and strictly "
                "reduce mean decisive-head sibling mass above sibling p99"
            ),
        },
        "P2": {
            "holds": p2_holds,
            "required_depths": list(P2_DEPTHS),
            "rows": p2_rows,
            "rule": (
                "both INT3 and INT2 strictly raise normalized mounted-graft "
                "entropy above bf16 and worsen the production-L2 answer class"
            ),
        },
        "verdict": verdict,
        "half_law_component": (
            "P1" if p1_holds and not p2_holds
            else "P2" if p2_holds and not p1_holds
            else None
        ),
        "addendum_interpretation": (
            "P1 appeared; reopen outlier contribution at the named passing depths."
            if p1_holds
            else "P1 failed to appear; independent confirmation of value blending."
        ),
        "raw_depth_signatures": raw_signatures,
    }
    assert_finite(result)
    return result


def _read_unique(run_dir: Path, stem: str) -> tuple[Path, dict[str, Any]]:
    matches = sorted(run_dir.glob(f"{stem}_*.json"))
    if len(matches) != 1:
        raise CMC2Error(
            f"expected exactly one {stem}_*.json under {run_dir}, found {len(matches)}")
    try:
        value = json.loads(matches[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CMC2Error(f"cannot read {matches[0]}: {exc}") from exc
    if not isinstance(value, dict):
        raise CMC2Error(f"receipt is not an object: {matches[0]}")
    assert_finite(value)
    return matches[0], value


def _frame_pass(receipt: Mapping[str, Any]) -> bool:
    frame = receipt.get("frame") or {}
    return bool(
        frame.get("probe_ladder") is True
        and frame.get("supersession_l2") is True
        and frame.get("name") == PRODUCTION_FRAME["name"]
    )


def adjudicate_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    g0_path, g0 = _read_unique(run_dir, "g0")
    g1_path, g1 = _read_unique(run_dir, "g1")
    paths: dict[str, Path] = {"g0": g0_path, "g1": g1_path}
    rows = []
    all_frames = _frame_pass(g0) and _frame_pass(g1)
    for label in DEPTH_LABELS:
        depth_path, depth = _read_unique(run_dir, f"depth_{label}")
        sup_path, supersession = _read_unique(run_dir, f"sup_{label}")
        paths[f"depth_{label}"] = depth_path
        paths[f"sup_{label}"] = sup_path
        if str(depth.get("depth")) != label or str(supersession.get("depth")) != label:
            raise CMC2Error(f"depth label mismatch for {label}")
        all_frames = all_frames and _frame_pass(depth) and _frame_pass(supersession)
        merged = dict(depth)
        merged["supersession"] = supersession["supersession"]
        merged["supersession_roundtrip"] = supersession["roundtrip"]
        rows.append(merged)

    g0_pass = bool(
        g0.get("fixture_status") == "REPRODUCES"
        and g0.get("fresh_control_correct") == 1
        and g0.get("fresh_control_total") == 2
    )
    g1_pass = bool(g1.get("byte_identity"))
    g2_pass = all(
        bool(row.get("roundtrip", {}).get("pass"))
        and bool(row.get("supersession_roundtrip", {}).get("pass"))
        for row in rows
    )
    witness_pass = all(
        bool(row.get("engine_operand_witness", {}).get("pass")) for row in rows)
    if not (g0_pass and g1_pass and g2_pass and witness_pass and all_frames):
        raise CMC2Error(
            "cannot adjudicate failed rail(s): "
            f"G0={g0_pass} G1={g1_pass} G2={g2_pass} "
            f"witness={witness_pass} frame={all_frames}")

    predictions = adjudicate_rows(rows)
    provenance = {
        name: {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }
    result = {
        "schema": "grm.cmc2.adjudication.v1",
        "order": "GRM-CMC2",
        "run_dir": str(run_dir),
        "frame": dict(PRODUCTION_FRAME),
        "gates": {
            "CMC2-G0": "PASS",
            "CMC2-G1": "PASS",
            "CMC2-G2": "PASS",
            "CMC2-G3": "PASS",
        },
        "engine_operand_witness_all_depths": "PASS",
        "rows": rows,
        "predictions": predictions,
        "adjudication": predictions["verdict"],
        "quant_floor_context": (
            "The prior GRM graft-quant ledger places the last green depth at "
            "INT6, a shoulder at INT4, and the un-supersedes floor at INT3. "
            "CMC2 measures whether the same boundary carries rising graft-mass "
            "entropy and a degraded production-L2 supersession read on the CMC "
            "machinery; it does not upgrade the earlier finite battery into a law."
        ),
        "provenance": provenance,
    }
    assert_finite(result)
    return result


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def render_report(result: Mapping[str, Any]) -> str:
    rows = result["rows"]
    predictions = result["predictions"]
    lines = [
        f"<!-- GRM-CMC2 RUN {Path(str(result['run_dir'])).name} -->",
        "",
        "# GRM-CMC2 quantization-depth sweep",
        "",
        f"Run: `{result['run_dir']}`  ",
        "Frame: **production defaults, ladder ON, L2 ON** (`F-PROD-LADDER-ON-L2-ON`).  ",
        "Witness: **CMC1.2 engine-operand tap**, unchanged tolerance `0.002`.",
        "",
        "## Gates",
        "",
        "| Gate | Status |",
        "|---|---|",
    ]
    for gate in ("CMC2-G0", "CMC2-G1", "CMC2-G2", "CMC2-G3"):
        lines.append(f"| {gate} | **{result['gates'][gate]}** |")
    lines += [
        "",
        "## Full sweep table (verbatim)",
        "",
        "Target/sibling/other are mean captured masses across the frozen decisive heads. Entropies are normalized to `[0,1]`; graft entropy spans every mounted graft. Sup quality is `2=correct`, `1=stale/wrong-fact`, `0=collapsed`.",
        "",
        "| Depth | Read | Praxis class | Solace class | Target mass | Sibling mass | Other mass | T/S entropy | Graft entropy | Sibling >p99 mass | Sup class | Sup quality | v2 outranks v1 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        split = row["mass_split"]
        masses = split["mean_captured_mass"]
        target = str(split["target"])
        sibling = str(split["sibling"])
        other = sum(
            float(value) for name, value in masses.items()
            if name not in (target, sibling)
        )
        sup = row["supersession"]
        lines.append(
            f"| {row['depth'].upper()} | {row['read']['outcome']} | "
            f"{row['read']['praxis']['classification']} | "
            f"{row['read']['solace']['classification']} | "
            f"{_fmt(float(masses[target]))} | {_fmt(float(masses[sibling]))} | "
            f"{_fmt(float(other))} | "
            f"{_fmt(float(split['target_vs_sibling_entropy_normalized_mean']))} | "
            f"{_fmt(float(split['mounted_graft_entropy_normalized_mean']))} | "
            f"{_fmt(float(split['sibling_outlier_mass_concentration_mean']))} | "
            f"{sup['classification']} | {_supersession_quality(sup)} | "
            f"{sup.get('v2_outranks_v1')} |"
        )

    lines += [
        "",
        "## Key-norm statistics",
        "",
        "| Depth | Graft | p50 | p99 | max | vectors |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        for graft, stats in row["key_norms"].items():
            lines.append(
                f"| {row['depth'].upper()} | {graft} | {_fmt(float(stats['p50']))} | "
                f"{_fmt(float(stats['p99']))} | {_fmt(float(stats['max']))} | "
                f"{int(stats['count'])} |"
            )

    lines += [
        "",
        "## Pack/unpack and trued-witness receipts",
        "",
        "| Depth | CMC round-trip | Sup round-trip | Witness max abs | Tolerance | Witness |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        witness = row["engine_operand_witness"]
        lines.append(
            f"| {row['depth'].upper()} | "
            f"{'PASS' if row['roundtrip']['pass'] else 'RED'} | "
            f"{'PASS' if row['supersession_roundtrip']['pass'] else 'RED'} | "
            f"{_fmt(float(witness['max_abs']))} | "
            f"{_fmt(float(witness['tolerance']))} | "
            f"{'PASS' if witness['pass'] else 'RED'} |"
        )

    p1 = predictions["P1"]
    p2 = predictions["P2"]
    lines += [
        "",
        "## Registered predictions and adjudication",
        "",
        f"- P1 repair band: **{'HOLDS' if p1['holds'] else 'DOES NOT HOLD'}**. {p1['rule']}.",
        f"- P2 identity collapse: **{'HOLDS' if p2['holds'] else 'DOES NOT HOLD'}**. {p2['rule']}.",
        "",
        f"**ADJUDICATION: {predictions['verdict']}**"
        + (f" — {predictions['half_law_component']} only." if predictions['verdict'] == 'HALF-LAW' else "."),
        "",
        predictions["addendum_interpretation"],
        "",
        "Mixed and mid-band signatures are not harmonized; the complete row flags are preserved in the machine adjudication receipt.",
        "",
        "## Relation to the un-supersedes floor",
        "",
        result["quant_floor_context"],
        "",
        "## Files",
        "",
    ]
    for name, item in result["provenance"].items():
        lines.append(
            f"- `{item['path']}` ({name}, sha256 `{item['sha256']}`)"
        )
    lines += ["", "Anything not done: **none for the registered sweep.**", ""]
    return "\n".join(lines)


def cpu_self_test() -> dict[str, Any]:
    rng = np.random.default_rng(0xC0C2)
    arrays = {
        "c": rng.normal(size=(2, 3, 4, 64)).astype(np.float16),
        "kpe": rng.normal(size=(2, 3, 4, 32)).astype(np.float16),
    }
    roundtrips = {}
    for label, bits in DEPTHS:
        rebuilt, receipt = roundtrip_arrays(arrays, bits)
        if not receipt["pass"]:
            raise CMC2Error(f"synthetic round-trip failed at {label}")
        if bits == 16 and not all(np.array_equal(arrays[k], rebuilt[k]) for k in arrays):
            raise CMC2Error("bf16 identity path changed content")
        roundtrips[label] = {
            "pass": receipt["pass"],
            "max_abs": max(v["max_abs"] for v in receipt["tensors"].values()),
        }

    attention = {
        "decisive_heads": [
            {
                "layer": 0,
                "head": 0,
                "graft_mass": {"target": 0.6, "sibling": 0.3, "other": 0.1},
                "sibling_mass_above_own_p99_fraction": 0.2,
            },
            {
                "layer": 1,
                "head": 1,
                "graft_mass": {"target": 0.4, "sibling": 0.4, "other": 0.2},
                "sibling_mass_above_own_p99_fraction": 0.1,
            },
        ]
    }
    mass = summarize_mass_split(attention, target="target", sibling="sibling")
    if not (0.0 <= mass["mounted_graft_entropy_normalized_mean"] <= 1.0):
        raise CMC2Error("normalized entropy left [0,1]")

    rows = []
    for label in DEPTH_LABELS:
        is_band = label in P1_DEPTHS
        is_low = label in P2_DEPTHS
        praxis_class = "correct" if is_band else "wrong-fact"
        sup_class = "collapsed" if is_low else "correct"
        rows.append({
            "depth": label,
            "read": {
                "outcome": "repaired" if is_band else "confused",
                "praxis": {"classification": praxis_class},
                "solace": {"classification": "correct"},
            },
            "mass_split": {
                "sibling_outlier_mass_concentration_mean": 0.1 if is_band else 0.2,
                "mounted_graft_entropy_normalized_mean": 0.9 if is_low else 0.5,
                "target_vs_sibling_entropy_normalized_mean": 0.8,
                "target": "target",
                "sibling": "sibling",
                "mean_captured_mass": {
                    "target": 0.5,
                    "sibling": 0.3,
                    "other": 0.2,
                },
            },
            "supersession": {
                "classification": sup_class,
                "v2_outranks_v1": not is_low,
            },
            "key_norms": {
                "target": {"p50": 1.0, "p99": 2.0, "max": 3.0, "count": 4},
            },
            "roundtrip": {"pass": True},
            "supersession_roundtrip": {"pass": True},
            "engine_operand_witness": {
                "max_abs": 0.001,
                "tolerance": 0.002,
                "pass": True,
            },
        })
    adjudication = adjudicate_rows(rows)
    if adjudication["verdict"] != "LAW-FUSED":
        raise CMC2Error("synthetic LAW-FUSED fixture did not fuse")
    rendered = render_report({
        "run_dir": "/tmp/grm_cmc2_synthetic",
        "rows": rows,
        "predictions": adjudication,
        "gates": {
            "CMC2-G0": "PASS",
            "CMC2-G1": "PASS",
            "CMC2-G2": "PASS",
            "CMC2-G3": "PASS",
        },
        "quant_floor_context": "synthetic floor context",
        "provenance": {},
    })
    if "ADJUDICATION: LAW-FUSED" not in rendered or "| INT2 |" not in rendered:
        raise CMC2Error("report renderer dropped the adjudication or a depth")

    nan_guard = False
    try:
        canonical_json_bytes({"x": float("nan")})
    except Exception:
        nan_guard = True
    if not nan_guard:
        raise CMC2Error("non-finite writer guard failed")

    result = {
        "schema": "grm.cmc2.cpu_selftest.v1",
        "frame": dict(PRODUCTION_FRAME),
        "status": "PASS",
        "depth_registration": list(DEPTH_LABELS),
        "roundtrips": roundtrips,
        "entropy_fixture": "PASS",
        "adjudication_fixture": "LAW-FUSED",
        "report_renderer_fixture": "PASS",
        "non_finite_writer_guard": "PASS",
        "gpu_imported_or_used": False,
    }
    assert_finite(result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    self_test = sub.add_parser("self-test", help="run CPU-only rails")
    self_test.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    adjudicate = sub.add_parser("adjudicate", help="adjudicate a complete live run")
    adjudicate.add_argument("--run-dir", type=Path, required=True)
    adjudicate.add_argument("--append-report", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "self-test":
        result = cpu_self_test()
        path = write_content_addressed(
            args.artifact_dir.resolve(), "cpu_selftest", result)
        print(canonical_json_bytes(result).decode("utf-8"), end="")
        print(f"receipt={path}")
        return 0

    result = adjudicate_run(args.run_dir)
    result_path = write_content_addressed(args.run_dir, "adjudication", result)
    report = render_report(result)
    markdown_path = write_content_addressed(
        args.run_dir, "GRM_CMC2_RESULTS", report, suffix=".md")
    appended = False
    if args.append_report:
        marker = report.splitlines()[0]
        appended = append_report_once(REPORT, report, marker)
    print(canonical_json_bytes({
        "result": str(result_path),
        "markdown": str(markdown_path),
        "report_appended": appended,
        "gates": result["gates"],
        "adjudication": result["adjudication"],
    }).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
