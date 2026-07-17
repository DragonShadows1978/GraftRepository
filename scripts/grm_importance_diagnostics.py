#!/usr/bin/env python3
"""CPU-only exploratory 7a/7b importance diagnostics.

The report deliberately applies no eligibility cutoff or decision threshold.
It compares headline S1/S2 ranks with both recorded S3 reductions and, when
v2 receipts carry it, compares raw per-layer S1 ranks too.  Spearman and
top-1 tie handling come directly from the G1/G2 harness.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from statistics import median
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_grm_importance_g1g2 import (  # noqa: E402
    ARTIFACTS_DIR,
    compute_g1,
    load_convo_artifacts,
    spearman,
    top1_agreement,
)


DIAGNOSTIC_SCHEMA = "grm_importance_diag_v1"


def _finite_number(value: Any) -> float | None:
    """Return a finite JSON-number-like value, preserving missingness."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _layer_order_key(layer: str) -> tuple[int, int | str]:
    try:
        return (0, int(layer))
    except ValueError:
        return (1, layer)


def _layer_value(candidate: Mapping[str, Any], layer: str) -> float | None:
    values = candidate.get("s1_mass_per_layer")
    if not isinstance(values, Mapping):
        return None
    if layer in values:
        return _finite_number(values[layer])
    # In-memory callers may still use integer layer keys.  JSON artifacts
    # always use strings, but accepting both makes the pure builder robust.
    try:
        numeric_layer = int(layer)
    except ValueError:
        return None
    return _finite_number(values.get(numeric_layer))


def _rank_metric(
        signal_scores: Mapping[str, Any],
        reference_scores: Mapping[str, Any],
        candidate_ids: Sequence[str],
) -> dict[str, Any]:
    """One probe's tie-aware rank comparison, using harness functions."""
    if len(candidate_ids) < 2:
        return {
            "ranked": False,
            "reason": "fewer_than_two_candidates",
            "n_candidates": len(candidate_ids),
            "spearman": None,
            "top1_agreement": None,
        }

    signal = {candidate_id: _finite_number(signal_scores.get(candidate_id))
              for candidate_id in candidate_ids}
    reference = {candidate_id: _finite_number(reference_scores.get(candidate_id))
                 for candidate_id in candidate_ids}
    if any(value is None for value in signal.values()) or any(
            value is None for value in reference.values()):
        return {
            "ranked": False,
            "reason": "missing_or_nonfinite_score",
            "n_candidates": len(candidate_ids),
            "spearman": None,
            "top1_agreement": None,
        }

    # Both functions are imported from the harness rather than recreated
    # here: average-rank Spearman and top-winner-set tie handling therefore
    # exactly match the registered analysis implementation.
    rho = spearman(
        [signal[candidate_id] for candidate_id in candidate_ids],
        [reference[candidate_id] for candidate_id in candidate_ids],
    )
    rho_value = float(rho) if math.isfinite(float(rho)) else None
    return {
        "ranked": True,
        "reason": None,
        "n_candidates": len(candidate_ids),
        "spearman": rho_value,
        "top1_agreement": int(top1_agreement(
            signal, reference, candidate_ids)),
    }


def _aggregate(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ranked = [metric for metric in metrics if metric["ranked"]]
    correlations = [metric["spearman"] for metric in ranked
                    if metric["spearman"] is not None]
    top1 = [metric["top1_agreement"] for metric in ranked
            if metric["top1_agreement"] is not None]
    return {
        "n_probes_total": len(metrics),
        "n_probes_ranked": len(ranked),
        "n_spearman_defined": len(correlations),
        "median_spearman": (float(median(correlations))
                            if correlations else None),
        "n_top1_matches": int(sum(top1)),
        "top1_agreement": (float(sum(top1) / len(top1)) if top1 else None),
    }


def _registered_dlogit_crosscheck(
        records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Expose the published G1 quantities without emitting a decision.

    The historic harness's Spearman is signal versus fixture grade, while
    its top-1 relation is signal versus dlogit S3 with the historic
    eligibility rule.  That is distinct from this report's direct
    signal-versus-S3 rank tables, so retain it as an explicit cross-check
    rather than silently calling the two quantities equivalent.
    """
    g1 = compute_g1(list(records))
    signals = {}
    for signal in ("s1_mass", "s2_salience"):
        result = g1[signal]
        signals[signal] = {
            "n_probes_ranked": int(result["n_probes_ranked"]),
            "n_eligible_top1": int(result["n_eligible"]),
            "median_spearman_vs_fixture_grade": _finite_number(
                result["median_spearman"]),
            "top1_agreement_vs_s3_dlogit": _finite_number(
                result["top1_agreement"]),
        }
    return {
        "semantics": (
            "legacy G1: Spearman uses fixture ground_truth_grade; top-1 "
            "uses S3 dlogit and the historic eligibility rule"
        ),
        "signals": signals,
    }


def _probe_diagnostics(
        conversation_id: str,
        artifact_schema: str | None,
        probe: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = probe.get("candidates", [])
    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    headline = {
        key: {str(candidate["candidate_id"]): candidate.get(key)
              for candidate in candidates}
        for key in ("s1_mass", "s2_salience", "s3_dep_dlogit", "s3_dep_kl")
    }
    dlogit = headline["s3_dep_dlogit"]
    kl = headline["s3_dep_kl"]

    layer_ids = sorted({str(layer)
                        for candidate in candidates
                        if isinstance(candidate.get("s1_mass_per_layer"), Mapping)
                        for layer in candidate["s1_mass_per_layer"]},
                       key=_layer_order_key)
    per_layer = {}
    for layer in layer_ids:
        layer_signal = {
            str(candidate["candidate_id"]): _layer_value(candidate, layer)
            for candidate in candidates
        }
        per_layer[layer] = {
            "s1_mass_vs_s3_dlogit": _rank_metric(
                layer_signal, dlogit, candidate_ids),
            "s1_mass_vs_s3_kl": _rank_metric(
                layer_signal, kl, candidate_ids),
        }

    return {
        "conversation_id": conversation_id,
        "artifact_schema": artifact_schema,
        "probe_id": probe.get("probe_id"),
        "n_candidates": len(candidate_ids),
        "s1_mass_vs_s3_dlogit": _rank_metric(
            headline["s1_mass"], dlogit, candidate_ids),
        "s2_salience_vs_s3_dlogit": _rank_metric(
            headline["s2_salience"], dlogit, candidate_ids),
        "s1_mass_vs_s3_kl": _rank_metric(
            headline["s1_mass"], kl, candidate_ids),
        "s2_salience_vs_s3_kl": _rank_metric(
            headline["s2_salience"], kl, candidate_ids),
        "s3_dlogit_vs_s3_kl": _rank_metric(dlogit, kl, candidate_ids),
        "s1_mass_per_layer": per_layer,
    }


def _post_hoc_best_layer(layers: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    """Select only for labeling; this is never a decision criterion."""
    candidates = []
    for layer, report in layers.items():
        summary = report["s1_mass_vs_s3_dlogit"]
        rho = summary["median_spearman"]
        if rho is not None:
            candidates.append((float(rho), layer))
    if not candidates:
        return None
    best_rho, best_layer = sorted(
        candidates, key=lambda item: (-item[0], _layer_order_key(item[1]))) [0]
    summary = layers[best_layer]["s1_mass_vs_s3_dlogit"]
    return {
        "label": "post-hoc best-single-layer by median Spearman vs S3 dlogit",
        "layer": best_layer,
        "median_spearman": best_rho,
        "n_probes_ranked": summary["n_probes_ranked"],
        "n_spearman_defined": summary["n_spearman_defined"],
    }


def build_diagnostics(artifacts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a JSON-serializable exploratory report from v1 and/or v2 data."""
    per_probe = []
    registered_records = []
    schemas = Counter()
    for artifact in artifacts:
        conversation_id = str(artifact.get("conversation_id", "unknown"))
        artifact_schema = artifact.get("schema")
        schemas[str(artifact_schema)] += 1
        for probe in artifact.get("probes", []):
            registered_records.append(probe)
            per_probe.append(_probe_diagnostics(
                conversation_id, artifact_schema, probe))

    comparisons = (
        "s1_mass_vs_s3_dlogit",
        "s2_salience_vs_s3_dlogit",
        "s1_mass_vs_s3_kl",
        "s2_salience_vs_s3_kl",
        "s3_dlogit_vs_s3_kl",
    )
    aggregate = {
        comparison: _aggregate([probe[comparison] for probe in per_probe])
        for comparison in comparisons
    }

    layer_ids = sorted({layer for probe in per_probe
                        for layer in probe["s1_mass_per_layer"]},
                       key=_layer_order_key)
    per_layer = {
        layer: {
            "s1_mass_vs_s3_dlogit": _aggregate([
                probe["s1_mass_per_layer"].get(layer, {}).get(
                    "s1_mass_vs_s3_dlogit", {
                        "ranked": False,
                        "reason": "layer_not_recorded",
                        "n_candidates": probe["n_candidates"],
                        "spearman": None,
                        "top1_agreement": None,
                    })
                for probe in per_probe
            ]),
            "s1_mass_vs_s3_kl": _aggregate([
                probe["s1_mass_per_layer"].get(layer, {}).get(
                    "s1_mass_vs_s3_kl", {
                        "ranked": False,
                        "reason": "layer_not_recorded",
                        "n_candidates": probe["n_candidates"],
                        "spearman": None,
                        "top1_agreement": None,
                    })
                for probe in per_probe
            ]),
        }
        for layer in layer_ids
    }

    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "exploratory": True,
        "n_convos": len(artifacts),
        "n_probes": len(per_probe),
        "artifact_schema_counts": dict(sorted(schemas.items())),
        "tie_handling": {
            "spearman": "imported harness average-rank ties",
            "top1": "imported harness winner-set overlap",
        },
        "aggregate": aggregate,
        "registered_dlogit_crosscheck": _registered_dlogit_crosscheck(
            registered_records),
        "per_probe": per_probe,
        "per_layer_s1": {
            "available": bool(per_layer),
            "value_kind": "raw mass from arena.s1_mass(per_layer=True)",
            "layers": per_layer,
            "post_hoc_best_single_layer": _post_hoc_best_layer(per_layer),
        },
    }


def _format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.4f}"


def _format_metric(metric: Mapping[str, Any]) -> str:
    if not metric["ranked"]:
        return "n/a"
    rho = _format_float(metric["spearman"])
    return f"{rho}/{metric['top1_agreement']}"


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    print(" | ".join(header.ljust(widths[idx])
                     for idx, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[idx])
                         for idx, value in enumerate(row)))


def print_human_report(diagnostics: Mapping[str, Any]) -> None:
    """Print the requested table before the one-line machine report."""
    schema_counts = ", ".join(
        f"{schema}={count}" for schema, count in
        diagnostics["artifact_schema_counts"].items())
    print("EXPLORATORY GRM importance diagnostics (7a/7b; no decision thresholds)")
    print(f"Artifacts: {diagnostics['n_convos']} convos, "
          f"{diagnostics['n_probes']} probes ({schema_counts})")
    print("Tie handling: harness average-rank Spearman; harness top-winner-set overlap.")
    print("No eligibility cutoff is applied in this diagnostic report.")
    print()
    print("Per-probe rank agreement (rho/top1)")
    probe_rows = []
    for probe in diagnostics["per_probe"]:
        probe_rows.append([
            str(probe["probe_id"]),
            str(probe["n_candidates"]),
            _format_metric(probe["s1_mass_vs_s3_dlogit"]),
            _format_metric(probe["s2_salience_vs_s3_dlogit"]),
            _format_metric(probe["s1_mass_vs_s3_kl"]),
            _format_metric(probe["s2_salience_vs_s3_kl"]),
            _format_metric(probe["s3_dlogit_vs_s3_kl"]),
        ])
    _print_table(
        ("probe", "n", "S1~dlogit", "S2~dlogit", "S1~KL", "S2~KL",
         "dlogit~KL"),
        probe_rows,
    )

    print()
    print("Aggregated rank agreement")
    aggregate_rows = []
    display_names = (
        ("s1_mass_vs_s3_dlogit", "S1 vs S3 dlogit"),
        ("s2_salience_vs_s3_dlogit", "S2 vs S3 dlogit"),
        ("s1_mass_vs_s3_kl", "S1 vs S3 KL"),
        ("s2_salience_vs_s3_kl", "S2 vs S3 KL"),
        ("s3_dlogit_vs_s3_kl", "S3 dlogit vs KL"),
    )
    for key, name in display_names:
        summary = diagnostics["aggregate"][key]
        agreement = ("n/a" if summary["top1_agreement"] is None
                     else f"{summary['top1_agreement']:.4f}")
        aggregate_rows.append([
            name,
            f"{summary['n_probes_ranked']}/{summary['n_probes_total']}",
            str(summary["n_spearman_defined"]),
            _format_float(summary["median_spearman"]),
            f"{summary['n_top1_matches']}/{summary['n_probes_ranked']}",
            agreement,
        ])
    _print_table(
        ("comparison", "ranked", "rho n", "median rho", "top1 matches",
         "top1 agreement"),
        aggregate_rows,
    )

    print()
    print("Published dlogit G1 cross-check (legacy harness semantics)")
    print("Spearman here uses fixture grade; top-1 uses dlogit S3 with the "
          "historic eligibility rule. This is shown separately from the "
          "direct S3-rank tables above.")
    crosscheck_rows = []
    for signal in ("s1_mass", "s2_salience"):
        summary = diagnostics["registered_dlogit_crosscheck"]["signals"][signal]
        agreement = summary["top1_agreement_vs_s3_dlogit"]
        crosscheck_rows.append([
            signal,
            str(summary["n_probes_ranked"]),
            str(summary["n_eligible_top1"]),
            _format_float(summary["median_spearman_vs_fixture_grade"]),
            "n/a" if agreement is None else f"{agreement:.4f}",
        ])
    _print_table(
        ("signal", "ranked", "eligible top1", "median grade rho",
         "dlogit top1 agreement"),
        crosscheck_rows,
    )

    print()
    print("Per-layer S1 rank agreement (exploratory raw diagnostics)")
    per_layer = diagnostics["per_layer_s1"]
    if not per_layer["available"]:
        print("No s1_mass_per_layer values are present; headline v1 analysis remains shown above.")
        return

    layer_rows = []
    for layer, report in per_layer["layers"].items():
        dlogit = report["s1_mass_vs_s3_dlogit"]
        kl = report["s1_mass_vs_s3_kl"]
        layer_rows.append([
            layer,
            f"{dlogit['n_probes_ranked']}/{dlogit['n_probes_total']}",
            _format_float(dlogit["median_spearman"]),
            ("n/a" if dlogit["top1_agreement"] is None
             else f"{dlogit['top1_agreement']:.4f}"),
            _format_float(kl["median_spearman"]),
            ("n/a" if kl["top1_agreement"] is None
             else f"{kl['top1_agreement']:.4f}"),
        ])
    _print_table(
        ("layer", "ranked", "dlogit median rho", "dlogit top1",
         "KL median rho", "KL top1"),
        layer_rows,
    )
    best = per_layer["post_hoc_best_single_layer"]
    if best is not None:
        print("POST-HOC best-single-layer (selection after observing this data; "
              "not a registered metric): "
              f"layer={best['layer']} median_rho={best['median_spearman']:+.4f} "
              f"over {best['n_spearman_defined']} defined probe correlations.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(ARTIFACTS_DIR),
        help="directory containing convo_*.json v1 and/or v2 artifacts",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = load_convo_artifacts(str(args.artifacts_dir))
    if not artifacts:
        raise SystemExit(f"no convo_*.json artifacts found under {args.artifacts_dir}")
    diagnostics = build_diagnostics(artifacts)
    print_human_report(diagnostics)
    print()
    print(json.dumps(diagnostics, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
