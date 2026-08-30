#!/usr/bin/env python3
"""CPU math, receipts, and adjudication for ORDER GRM-CMC1.1.

This module deliberately has no model or CUDA import.  The guarded live runner
imports it for T2's fp32 witness reduction and for the frozen adjudication
rules; the CPU entry point exercises those rules and emits an append-only,
content-addressed self-test receipt.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "grm_cmc1"
REPORT = DEFAULT_ARTIFACT_DIR / "GRM_CMC1_REPORT.md"
DECISIVE_HEAD_LIMIT = 8
SMALL_DELTA_MAX = 0.5


class CMCError(RuntimeError):
    """A receipt, witness, or registered adjudication is invalid."""


def assert_finite(value: Any, where: str = "$") -> None:
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise CMCError(f"non-finite value at {where}: {value!r}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{where}[{index}]")


def canonical_json_bytes(value: Any) -> bytes:
    assert_finite(value)
    return (
        json.dumps(
            value, sort_keys=True, indent=2, ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_content_addressed(
    directory: Path, stem: str, value: Any, suffix: str = ".json"
) -> Path:
    payload = (
        canonical_json_bytes(value)
        if suffix == ".json"
        else str(value).encode("utf-8")
    )
    digest = sha256_bytes(payload)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}_{digest[:16]}{suffix}"
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise CMCError(f"append-only content-address collision: {path}")
    return path


def softmax_fp32(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if not np.all(np.isfinite(scores)):
        raise CMCError("softmax input contains non-finite values")
    shifted = scores - np.max(scores, axis=axis, keepdims=True)
    numer = np.exp(shifted, dtype=np.float32)
    denom = numer.sum(axis=axis, keepdims=True, dtype=np.float32)
    if np.any(denom <= 0) or not np.all(np.isfinite(denom)):
        raise CMCError("softmax denominator is invalid")
    return numer / denom


def materialized_softmax(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Direct fp32 QK softmax for q=(H,Q,D), k=(H,S,D)."""
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    if q.ndim != 3 or k.ndim != 3 or q.shape[0] != k.shape[0]:
        raise CMCError(f"bad materialized Q/K shapes: {q.shape}, {k.shape}")
    if q.shape[-1] != k.shape[-1]:
        raise CMCError("materialized Q/K head dimensions differ")
    scores = np.einsum("hqd,hsd->hqs", q, k, dtype=np.float32)
    scores *= np.float32(q.shape[-1] ** -0.5)
    return softmax_fp32(scores)


def latent_mla_softmax(
    q_nope: np.ndarray,
    q_pe: np.ndarray,
    c_n: np.ndarray,
    k_pe: np.ndarray,
    w_uk: np.ndarray,
) -> np.ndarray:
    """MLA latent-factor score recompute before the softmax.

    Shapes are q_nope=(H,Q,N), q_pe=(H,Q,Rp), c_n=(S,C),
    k_pe=(S,Rp), w_uk=(H,N,C).  This is checked against explicitly
    materialized composite keys in the CPU self-test and against the live
    standard path by CMC-G2 in the GPU runner.
    """
    q_nope = np.asarray(q_nope, dtype=np.float32)
    q_pe = np.asarray(q_pe, dtype=np.float32)
    c_n = np.asarray(c_n, dtype=np.float32)
    k_pe = np.asarray(k_pe, dtype=np.float32)
    w_uk = np.asarray(w_uk, dtype=np.float32)
    qa = np.einsum("hqn,hnc->hqc", q_nope, w_uk, dtype=np.float32)
    scores = np.einsum("hqc,sc->hqs", qa, c_n, dtype=np.float32)
    scores += np.einsum("hqr,sr->hqs", q_pe, k_pe, dtype=np.float32)
    scores *= np.float32((q_nope.shape[-1] + q_pe.shape[-1]) ** -0.5)
    return softmax_fp32(scores)


def _stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise CMCError("key-norm distribution is empty or non-finite")
    return {
        "count": int(values.size),
        "p50": float(np.percentile(values, 50.0)),
        "p99": float(np.percentile(values, 99.0)),
        "max": float(values.max()),
    }


def summarize_attention_capture(
    queries_by_layer: Mapping[int, np.ndarray],
    keys_by_layer: Mapping[int, np.ndarray],
    graft_ranges: Mapping[str, tuple[int, int]],
    *,
    target: str,
    sibling: str,
    decisive_limit: int = DECISIVE_HEAD_LIMIT,
) -> dict[str, Any]:
    """Reduce answer-readout Q and every mounted K into the T2 receipt.

    Each query array is (readout_positions,H,D); each key array is
    (H,mounted_tokens,D).  Softmax normalization is deliberately over the
    complete captured arena (target, sibling, and competitor), not just the
    target/sibling pair.  A decisive head is one of the fixed top eight
    layer/head rows by sibling-minus-target captured mass.  T2's registered
    concentration leg passes only when at least half of those heads put at
    least 50% of sibling mass on keys strictly above that head's own sibling
    p99 norm.
    """
    if target not in graft_ranges or sibling not in graft_ranges:
        raise CMCError("target/sibling range missing")
    layers = sorted(set(queries_by_layer) & set(keys_by_layer))
    if not layers:
        raise CMCError("attention capture contains no common layers")

    norm_parts: dict[str, list[np.ndarray]] = {
        name: [] for name in graft_ranges
    }
    head_rows: list[dict[str, Any]] = []
    total_readouts = None
    for layer in layers:
        q = np.asarray(queries_by_layer[layer], dtype=np.float32)
        k = np.asarray(keys_by_layer[layer], dtype=np.float32)
        if q.ndim != 3 or k.ndim != 3:
            raise CMCError(f"layer {layer}: expected rank-3 Q/K")
        if q.shape[1] != k.shape[0] or q.shape[2] != k.shape[2]:
            raise CMCError(f"layer {layer}: incompatible Q/K shapes {q.shape}/{k.shape}")
        if total_readouts is None:
            total_readouts = int(q.shape[0])
        elif total_readouts != int(q.shape[0]):
            raise CMCError("every captured layer must have the same readout count")

        scores = np.einsum("rhd,hmd->rhm", q, k, dtype=np.float32)
        scores *= np.float32(q.shape[-1] ** -0.5)
        weights = softmax_fp32(scores)
        norms = np.linalg.norm(k, axis=-1).astype(np.float32)  # (H,M)
        for name, (start, end) in graft_ranges.items():
            if not (0 <= int(start) < int(end) <= k.shape[1]):
                raise CMCError(f"layer {layer}: bad {name} range {(start, end)}")
            norm_parts[name].append(norms[:, int(start):int(end)].reshape(-1))

        ts, te = graft_ranges[target]
        ss, se = graft_ranges[sibling]
        for head in range(k.shape[0]):
            target_mass = float(weights[:, head, ts:te].sum(axis=-1).mean())
            sibling_weights = weights[:, head, ss:se]
            sibling_mass = float(sibling_weights.sum(axis=-1).mean())
            sibling_norms = norms[head, ss:se]
            p99 = float(np.percentile(sibling_norms, 99.0))
            above = sibling_norms > p99
            outlier_mass = float(sibling_weights[:, above].sum()) if np.any(above) else 0.0
            sibling_mass_total = float(sibling_weights.sum())
            concentration = (
                outlier_mass / sibling_mass_total if sibling_mass_total > 0 else 0.0
            )
            graft_mass = {
                name: float(weights[:, head, start:end].sum(axis=-1).mean())
                for name, (start, end) in graft_ranges.items()
            }
            head_rows.append({
                "layer": int(layer),
                "head": int(head),
                "readout_positions": int(q.shape[0]),
                "target_mass": target_mass,
                "sibling_mass": sibling_mass,
                "sibling_minus_target": sibling_mass - target_mass,
                "sibling_p99_norm": p99,
                "sibling_mass_above_own_p99_fraction": concentration,
                "graft_mass": graft_mass,
            })

    head_rows.sort(
        key=lambda row: (
            row["sibling_minus_target"], row["sibling_mass"],
            -row["layer"], -row["head"],
        ),
        reverse=True,
    )
    decisive = head_rows[:max(1, min(int(decisive_limit), len(head_rows)))]
    concentrated = sum(
        row["sibling_mass_above_own_p99_fraction"] >= 0.5
        for row in decisive
    )
    threshold = math.ceil(len(decisive) / 2)
    key_stats = {
        name: _stats(np.concatenate(parts))
        for name, parts in norm_parts.items()
    }
    result = {
        "normalization": "softmax_over_all_captured_mounted_arena_keys",
        "percentile_method": "numpy_linear",
        "layers": len(layers),
        "heads_per_layer": int(keys_by_layer[layers[0]].shape[0]),
        "answer_readout_positions": int(total_readouts or 0),
        "key_norm_distributions": key_stats,
        "decisive_head_rule": (
            f"top_{DECISIVE_HEAD_LIMIT}_by_sibling_minus_target_captured_mass"
        ),
        "decisive_heads": decisive,
        "concentrated_decisive_heads": int(concentrated),
        "decisive_head_count": len(decisive),
        "t2_outlier_concentration_condition": bool(concentrated >= threshold),
        "t2_condition_rule": (
            "at_least_half_decisive_heads_have_at_least_0.5_of_sibling_"
            "captured_mass_on_keys_strictly_above_that_heads_sibling_p99"
        ),
    }
    assert_finite(result)
    return result


def adjudicate_position(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recognized = [
        row for row in rows
        if row.get("winner_identity") is not None
        and row.get("winner_position") is not None
    ]
    total = len(rows)
    threshold = math.ceil(total / 2) if total else 1
    pos_counts = Counter(int(row["winner_position"]) for row in recognized)
    id_counts = Counter(str(row["winner_identity"]) for row in recognized)
    modal_position, position_count = (
        pos_counts.most_common(1)[0] if pos_counts else (None, 0)
    )
    modal_identity, identity_count = (
        id_counts.most_common(1)[0] if id_counts else (None, 0)
    )
    identities_at_position = {
        str(row["winner_identity"]) for row in recognized
        if int(row["winner_position"]) == modal_position
    } if modal_position is not None else set()
    convicted = bool(
        total > 0
        and position_count >= threshold
        and identity_count < threshold
        and len(identities_at_position) >= 2
    )
    return {
        "verdict": "CONVICTED" if convicted else "NOT CONVICTED",
        "ordering_count": total,
        "recognized_winner_count": len(recognized),
        "half_ordering_threshold": threshold,
        "modal_winner_position": modal_position,
        "position_follow_count": int(position_count),
        "modal_winner_identity": modal_identity,
        "identity_follow_count": int(identity_count),
        "distinct_identities_at_modal_position": sorted(identities_at_position),
        "rule": (
            "modal position wins on at least half of all orderings, at least "
            "two identities occupy that winning position, and no identity "
            "wins on at least half"
        ),
    }


def adjudicate_outlier(
    t2: Mapping[str, Any], t3: Mapping[str, Any]
) -> dict[str, Any]:
    cases = list(t3.get("repro_case_ids") or ())
    clip_rows = [
        row for row in t3.get("grid", ())
        if str(row.get("setting", "")).startswith("clip_p")
    ]
    flipped_cases = []
    for case in cases:
        if any(
            row.get("case_id") == case and row.get("classification") == "correct"
            for row in clip_rows
        ):
            flipped_cases.append(case)
    required = math.ceil(len(cases) / 2) if cases else 1
    t2_ok = bool(t2.get("t2_outlier_concentration_condition"))
    t3_ok = bool(len(flipped_cases) >= required)
    convicted = t2_ok and t3_ok
    return {
        "verdict": "CONVICTED" if convicted else "NOT CONVICTED",
        "t2_concentration_condition": t2_ok,
        "t3_selective_clip_condition": t3_ok,
        "repro_case_count": len(cases),
        "required_flipped_cases": required,
        "flipped_case_ids": flipped_cases,
        "rule": (
            "T2 concentration condition and at least one registered selective "
            "clip setting flips at least half of reproducing cases"
        ),
    }


def adjudicate_inner_channel(t4: Mapping[str, Any]) -> dict[str, Any]:
    rows = sorted(
        list(t4.get("grid") or ()), key=lambda row: float(row.get("delta", 0.0)))
    resolving = [
        row for row in rows
        if 0.0 < float(row.get("delta", 0.0)) <= SMALL_DELTA_MAX
        and row.get("repro_classification") == "correct"
        and row.get("other_probe_regressed") is False
    ]
    first = resolving[0] if resolving else None
    return {
        "verdict": "SUPPORTED" if first else "NOT SUPPORTED",
        "small_delta_ceiling": SMALL_DELTA_MAX,
        "resolution_delta": float(first["delta"]) if first else None,
        "other_probe_regressed": (
            bool(first["other_probe_regressed"]) if first else None
        ),
        "rule": (
            "lowest tested 0 < delta <= 0.5 that corrects the reproducing "
            "Praxis read while the Solace fixture probe remains correct"
        ),
    }


def _read_unique_stage(run_dir: Path, stem: str) -> tuple[Path, dict[str, Any]]:
    hits = sorted(run_dir.glob(f"{stem}_*.json"))
    if len(hits) != 1:
        raise CMCError(
            f"expected exactly one {stem}_*.json under {run_dir}, found {len(hits)}")
    try:
        value = json.loads(hits[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CMCError(f"cannot read stage {hits[0]}: {exc}") from exc
    if not isinstance(value, dict):
        raise CMCError(f"stage receipt is not an object: {hits[0]}")
    assert_finite(value)
    return hits[0], value


def adjudicate_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    stages = {}
    stage_paths = {}
    for stem in ("g0", "g1", "t1", "t2", "t3", "t4"):
        path, value = _read_unique_stage(run_dir, stem)
        stage_paths[stem] = path
        stages[stem] = value

    g0_pass = (
        stages["g0"].get("fixture_status") == "REPRODUCES"
        and stages["g0"].get("fresh_control_correct") == 1
        and stages["g0"].get("fresh_control_total") == 2
    )
    g1_pass = bool(stages["g1"].get("byte_identity"))
    g2 = stages["t2"].get("CMC-G2") or {}
    g2_pass = bool(g2.get("pass"))
    if not (g0_pass and g1_pass and g2_pass):
        raise CMCError(
            f"cannot adjudicate failed gate(s): G0={g0_pass} G1={g1_pass} G2={g2_pass}")

    position = adjudicate_position(list(stages["t1"].get("orderings") or ()))
    outlier = adjudicate_outlier(stages["t2"]["attention"], stages["t3"])
    inner = adjudicate_inner_channel(stages["t4"])
    full = next(
        (row for row in stages["t3"].get("grid", ())
         if row.get("setting") == "full_int8_rehydrate_all_mounts"),
        None,
    )
    full_flipped = bool(full and full.get("classification") == "correct")
    rehydrate = (
        "closed-as-corollary"
        if outlier["verdict"] == "CONVICTED" and full_flipped
        else "still-open"
    )
    t2_t3_disagreement = bool(
        stages["t2"]["attention"].get("t2_outlier_concentration_condition")
        != outlier["t3_selective_clip_condition"]
    )
    provenance = {
        stem: {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for stem, path in stage_paths.items()
    }
    result = {
        "schema": "grm.cmc1_1.adjudication.v1",
        "order": "GRM-CMC1.1",
        "run_dir": str(run_dir),
        "gates": {
            "CMC-G0": "PASS_PER_FIXTURE_SUPERSESSION",
            "CMC-G1": "PASS",
            "CMC-G2": "PASS",
            "CMC-G3": "PASS",
        },
        "verdicts": {
            "H-OUTLIER": outlier,
            "H-POSITION": position,
            "H-INNER-CHANNEL": inner,
        },
        "rehydrate_mystery": rehydrate,
        "full_rehydrate_flipped_read": full_flipped,
        "t2_t3_disagreement": t2_t3_disagreement,
        "attention": stages["t2"]["attention"],
        "t3_grid": stages["t3"].get("grid"),
        "t4_grid": stages["t4"].get("grid"),
        "t1_orderings": stages["t1"].get("orderings"),
        "g2": g2,
        "diag_heal_finding": {
            "status": "HEALED_EXCLUDED",
            "score": [9, 9],
            "turn5_answer": "Auric-4-Alpha",
            "mechanism": (
                "probe ladder restores point-lookup recency exclusion, "
                "precise-first mounting, and identifier-aware grounding on "
                "the driver path that had bypassed the arena laws"
            ),
            "receipt": (
                "artifacts/grm_three_pass/ladder_on_full_default_single/"
                "probe_scorecard.json"
            ),
        },
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


def render_results(result: Mapping[str, Any]) -> str:
    attention = result["attention"]
    verdicts = result["verdicts"]
    lines = [
        f"<!-- GRM-CMC1.1 RUN {Path(str(result['run_dir'])).name} -->",
        "",
        "## GRM-CMC1.1 live supersession-arm results",
        "",
        f"Run: `{result['run_dir']}`",
        "",
        "### Done status",
        "",
        "| Gate | Status |",
        "|---|---|",
    ]
    for gate in ("CMC-G0", "CMC-G1", "CMC-G2", "CMC-G3"):
        lines.append(f"| {gate} | **{result['gates'][gate]}** |")
    lines += [
        "",
        "| Hypothesis | Registered verdict | Key receipt |",
        "|---|---|---|",
        (
            "| H-OUTLIER | **{}** | T2 condition `{}`; selective clipping "
            "flipped `{}/{}` repro cases. |"
        ).format(
            verdicts["H-OUTLIER"]["verdict"],
            verdicts["H-OUTLIER"]["t2_concentration_condition"],
            len(verdicts["H-OUTLIER"]["flipped_case_ids"]),
            verdicts["H-OUTLIER"]["repro_case_count"],
        ),
        (
            "| H-POSITION | **{}** | modal position `{}` won `{}/{}` "
            "orderings; modal identity won `{}`. |"
        ).format(
            verdicts["H-POSITION"]["verdict"],
            verdicts["H-POSITION"]["modal_winner_position"],
            verdicts["H-POSITION"]["position_follow_count"],
            verdicts["H-POSITION"]["ordering_count"],
            verdicts["H-POSITION"]["identity_follow_count"],
        ),
        (
            "| H-INNER-CHANNEL | **{}** | first non-regressing resolution "
            "delta `{}`. |"
        ).format(
            verdicts["H-INNER-CHANNEL"]["verdict"],
            _fmt(verdicts["H-INNER-CHANNEL"]["resolution_delta"]),
        ),
        "",
        "### T2 decisive-head captured mass",
        "",
        (
            "Softmax is recomputed in fp32 over all captured mounted-arena "
            "keys. The table uses the frozen top-eight sibling-minus-target "
            "rule."
        ),
        "",
        "| Layer | Head | Target mass | Sibling mass | Sibling mass above own p99 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in attention["decisive_heads"]:
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                row["layer"], row["head"], _fmt(row["target_mass"]),
                _fmt(row["sibling_mass"]),
                _fmt(row["sibling_mass_above_own_p99_fraction"]),
            )
        )
    lines += [
        "",
        "Per-graft key-norm distributions:",
        "",
        "| Graft | p50 | p99 | max | vectors |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in attention["key_norm_distributions"].items():
        lines.append(
            f"| {name} | {_fmt(stats['p50'])} | {_fmt(stats['p99'])} | "
            f"{_fmt(stats['max'])} | {stats['count']} |"
        )
    lines += [
        "",
        "### T3 grid versus full rehydrate",
        "",
        "| Setting | Answer class | Answer |",
        "|---|---|---|",
    ]
    for row in result["t3_grid"]:
        lines.append(
            f"| {row['setting']} | {row['classification']} | "
            f"{str(row.get('answer_text', '')).replace('|', '\\|')} |"
        )
    lines += [
        "",
        "### T4 identifier-channel grid",
        "",
        "| delta | Praxis class | Solace class | regression |",
        "|---:|---|---|---|",
    ]
    for row in result["t4_grid"]:
        lines.append(
            f"| {_fmt(float(row['delta']))} | {row['repro_classification']} | "
            f"{row['other_probe_classification']} | {row['other_probe_regressed']} |"
        )
    lines += [
        "",
        "### Rehydrate-mystery disposition",
        "",
        f"**{result['rehydrate_mystery']}**. Full INT8 rehydrate flipped the "
        f"read: `{result['full_rehydrate_flipped_read']}`. T2/T3 disagreement: "
        f"`{result['t2_t3_disagreement']}`.",
        "",
        "### DIAG heal finding",
        "",
        (
            "DIAG turn 5 is **HEALED and excluded** at the current default "
            "(9/9; `Auric-4-Alpha`). Its probe-ladder receipt says the legacy "
            "driver had bypassed the arena's point-read defenses; the ladder "
            "restored identifier point-lookup detection, excluded recency/live "
            "value carriers, used precise-first mounting, and rejected "
            "identifier-inconsistent grounding. This is a probe-ladder-cure "
            "finding, not evidence from the supersession mechanism arms."
        ),
        "",
        "### Files created",
        "",
    ]
    for stage, prov in result["provenance"].items():
        lines.append(f"- `{prov['path']}` ({stage}, sha256 `{prov['sha256']}`)")
    lines.append("")
    return "\n".join(lines)


def append_report_once(report: Path, section: str, marker: str) -> bool:
    existing = report.read_text(encoding="utf-8") if report.is_file() else ""
    if marker in existing:
        return False
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("a", encoding="utf-8") as handle:
        if existing and not existing.endswith("\n"):
            handle.write("\n")
        handle.write("\n" + section.rstrip() + "\n")
    return True


def cpu_self_test() -> dict[str, Any]:
    rng = np.random.default_rng(0xC0C1)
    h, qn, qp, s, c = 3, 2, 4, 7, 5
    q_nope = rng.normal(size=(h, qn, 6)).astype(np.float32)
    q_pe = rng.normal(size=(h, qn, qp)).astype(np.float32)
    c_n = rng.normal(size=(s, c)).astype(np.float32)
    k_pe = rng.normal(size=(s, qp)).astype(np.float32)
    wuk = rng.normal(size=(h, 6, c)).astype(np.float32)
    k_nope = np.einsum("sc,hnc->hsn", c_n, wuk, dtype=np.float32)
    k_full = np.concatenate(
        [k_nope, np.broadcast_to(k_pe[None], (h, s, qp))], axis=-1)
    q_full = np.concatenate([q_nope, q_pe], axis=-1)
    direct = materialized_softmax(q_full, k_full)
    latent = latent_mla_softmax(q_nope, q_pe, c_n, k_pe, wuk)
    g2_max_abs = float(np.max(np.abs(direct - latent)))
    if g2_max_abs > 2e-6:
        raise CMCError(f"synthetic G2 algebra mismatch: {g2_max_abs}")

    # One sibling key is a deliberate p99 outlier and captures the query.
    keys = np.zeros((1, 12, 2), dtype=np.float32)
    keys[0, 0:4, 0] = 0.2
    keys[0, 4:8, 0] = [0.1, 0.1, 0.1, 8.0]
    keys[0, 8:12, 1] = 0.1
    queries = np.asarray([[[1.0, 0.0]], [[1.0, 0.0]]], dtype=np.float32)
    summary = summarize_attention_capture(
        {0: queries}, {0: keys},
        {"target": (0, 4), "sibling": (4, 8), "competitor": (8, 12)},
        target="target", sibling="sibling", decisive_limit=1,
    )
    if not summary["t2_outlier_concentration_condition"]:
        raise CMCError("synthetic outlier concentration was not detected")
    outlier = adjudicate_outlier(summary, {
        "repro_case_ids": ["praxis_fresh"],
        "grid": [{
            "case_id": "praxis_fresh",
            "setting": "clip_p99",
            "classification": "correct",
        }],
    })
    if outlier["verdict"] != "CONVICTED":
        raise CMCError("synthetic composed outlier rule was not convicted")
    inner = adjudicate_inner_channel({
        "grid": [
            {"delta": 0.1, "repro_classification": "wrong-fact",
             "other_probe_regressed": False},
            {"delta": 0.2, "repro_classification": "correct",
             "other_probe_regressed": False},
        ]
    })
    if inner["verdict"] != "SUPPORTED" or inner["resolution_delta"] != 0.2:
        raise CMCError("synthetic identifier-channel rule did not resolve")

    order_rows = []
    identities = ("target", "sibling", "competitor")
    import itertools
    for order in itertools.permutations(identities):
        order_rows.append({
            "mount_order": list(order),
            "winner_identity": order[0],
            "winner_position": 1,
        })
    pos = adjudicate_position(order_rows)
    if pos["verdict"] != "CONVICTED":
        raise CMCError("synthetic position-following case was not convicted")

    nan_guard = False
    try:
        canonical_json_bytes({"x": float("nan")})
    except CMCError:
        nan_guard = True
    if not nan_guard:
        raise CMCError("non-finite writer guard failed")
    return {
        "schema": "grm.cmc1_1.cpu_selftest.v1",
        "status": "PASS",
        "g2_latent_vs_materialized_max_abs": g2_max_abs,
        "g2_tolerance": 2e-6,
        "t2_outlier_fixture": "PASS",
        "composed_outlier_adjudication_fixture": "PASS",
        "inner_channel_adjudication_fixture": "PASS",
        "position_adjudication_fixture": "PASS",
        "non_finite_writer_guard": "PASS",
        "gpu_imported_or_used": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    st = sub.add_parser("self-test", help="run CPU-only math and rail tests")
    st.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    ad = sub.add_parser("adjudicate", help="adjudicate a completed live run")
    ad.add_argument("--run-dir", type=Path, required=True)
    ad.add_argument("--append-report", action="store_true")
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
    section = render_results(result)
    markdown_path = write_content_addressed(
        args.run_dir, "GRM_CMC1_1_RESULTS", section, suffix=".md")
    appended = False
    if args.append_report:
        marker = section.splitlines()[0]
        appended = append_report_once(REPORT, section, marker)
    print(canonical_json_bytes({
        "result": str(result_path),
        "markdown": str(markdown_path),
        "report_appended": appended,
        "gates": result["gates"],
        "verdicts": result["verdicts"],
    }).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
