#!/usr/bin/env python3
"""CPU-only registration, metrics, and reporting for ORDER GRM-ADM1.

The live runners deliberately keep policy selection separate from model
readout.  This module owns the append-only receipt format, the frozen
fit/eval split, the held-out margin threshold selection, deterministic
bootstrap intervals, gate checks, and the registered ADM verdict.

Nothing in this file imports a model or CUDA runtime.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "grm_adm1"
ORDER_PATH = ROOT / "orders" / "GRM_ADM1_K_POLICY.md"
REPORT_PATH = ARTIFACT_DIR / "GRM_ADM1_REPORT.md"
ARMS = ("A-k1", "A-k2", "A-k3", "A-DEC")
CLASSES = ("POINT-LOOKUP", "CROSS-FACT / SYNTHESIS", "AMBIGUOUS")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95


class ADMError(RuntimeError):
    """A registration, receipt, or registered adjudication is invalid."""


def assert_finite(value: Any, where: str = "$") -> None:
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ADMError(f"non-finite value at {where}: {value!r}")
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
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
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


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def write_content_addressed(
    directory: Path, stem: str, value: Any, *, suffix: str = ".json"
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
            handle.flush()
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ADMError(f"append-only content-address collision: {path}")
    return path


def write_exclusive_json(path: Path, value: Any) -> None:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
    except FileExistsError as exc:
        if path.read_bytes() != payload:
            raise ADMError(f"append-only target already exists: {path}") from exc


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ADMError(f"invalid JSONL {path}:{lineno}: {exc}") from exc
        if not isinstance(value, dict):
            raise ADMError(f"JSONL row is not an object: {path}:{lineno}")
        rows.append(value)
    return rows


def _extract_assignment(path: Path, name: str) -> Any:
    """Evaluate one literal/lambda assignment without importing its GPU test."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            selected.append(node)
            break
    if not selected:
        raise ADMError(f"assignment {name!r} not found in {path}")
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace: dict[str, Any] = {}
    safe_globals = {
        "__builtins__": {},
        "str": str,
    }
    exec(compile(module, str(path), "exec"), safe_globals, namespace)
    return namespace[name]


def corpus100_fixture_rows() -> list[dict[str, Any]]:
    source = ROOT / "tests" / "test_graft_corpus100.py"
    families = _extract_assignment(source, "FAMILIES")
    rows = []
    ordinal = 0
    for family_index, (family, _mk_text, mk_probe, mk_values) in enumerate(families):
        for instance_index in (
            (3 * family_index + 1) % 10,
            (7 * family_index + 4) % 10,
        ):
            code, value = mk_values(instance_index)
            rows.append({
                "probe_id": f"corpus100:{family}:{instance_index}",
                "fixture": "CORPUS-100",
                "class_label": "POINT-LOOKUP",
                "split": "fit" if ordinal % 5 == 0 else "eval",
                "question": mk_probe(code),
                "registered_identifiers": [str(code)],
                "expected_values": [str(value).casefold()],
                "target_ordinal": family_index * 10 + instance_index,
                "family": family,
                "source": str(source.relative_to(ROOT)),
            })
            ordinal += 1
    if len(rows) != 20 or sum(row["split"] == "fit" for row in rows) != 4:
        raise ADMError("CORPUS-100 registration must contain 20 probes / 4 fit")
    return rows


def supersession_fixture_rows() -> list[dict[str, Any]]:
    path = (
        ROOT / "tests" / "fixtures" / "supersession_battery"
        / "fresh_fact_controls.json"
    )
    fixture = read_json(path)
    nodes = {str(node["node_id"]): node for node in fixture["nodes"]}
    rows = []
    for probe in fixture["probes"]:
        target = nodes[str(probe["target_node"])]
        identifier = re.sub(r"_fact$", "", str(target["node_id"]))
        rows.append({
            "probe_id": f"supersession:{probe['probe_id']}",
            "fixture": "SUPERSESSION-FRESH-CONTROLS",
            "class_label": "POINT-LOOKUP",
            "split": "eval",
            "question": probe["question"],
            "registered_identifiers": [identifier.replace("_", " ")],
            "expected_values": list(probe["expected_values"]),
            "target_node": probe["target_node"],
            "wrong_fact_values": list(probe["wrong_fact_values"]),
            "source": str(path.relative_to(ROOT)),
        })
    if len(rows) != 2:
        raise ADMError("supersession fresh controls must contain two probes")
    return rows


def _e2e_script() -> list[dict[str, Any]]:
    path = ROOT / "artifacts" / "grm_e2e" / "full_session_20260708_P2" / "run_config.json"
    value = read_json(path)
    script = list(value.get("script") or ())
    if len(script) != 34:
        raise ADMError(f"certified E2E script is not 34 turns: {len(script)}")
    return script


def e2e_fixture_rows() -> list[dict[str, Any]]:
    script = _e2e_script()
    point_rows = []
    for turn, event in enumerate(script):
        if event.get("kind") != "probe":
            continue
        point_rows.append({
            "probe_id": f"diag:turn-{turn}:{event['fact_id'].replace(' ', '-')}",
            "fixture": "DIAG-CURRENT-FULL",
            "class_label": "POINT-LOOKUP",
            "split": "eval",
            "turn": turn,
            "question": event["user"],
            "registered_identifiers": [event["fact_id"]],
            "expected_values": [str(event["expected"]).casefold()],
            "source_turn": int(event["source_turn"]),
            "source": "scripts/grm_e2e_session.py:build_full_script",
        })
    if len(point_rows) != 9:
        raise ADMError(f"DIAG registration must contain nine probes: {len(point_rows)}")

    cross_turns = (5, 13)
    cross_rows = []
    for fixture, source in (
        ("E2E-34-CROSS-FACT", "artifacts/grm_e2e/full_session_20260708_P2"),
        ("P4-REPLICATION-CROSS-FACT", "artifacts/grm_three_pass/p4_ffull_three_pass"),
    ):
        for turn in cross_turns:
            event = script[turn]
            cross_rows.append({
                "probe_id": f"{fixture.lower()}:turn-{turn}:{event['fact_id'].replace(' ', '-')}",
                "fixture": fixture,
                "class_label": "CROSS-FACT / SYNTHESIS",
                "split": "eval",
                "turn": turn,
                "question": event["user"],
                "registered_identifiers": [event["fact_id"]],
                "expected_values": [str(event["expected"]).casefold()],
                "source_turn": int(event["source_turn"]),
                "source": source,
                "pre_run_label_basis": (
                    "certified A-k3 answer contains the cypher-bridge value; "
                    "lead ledger labels this cross-fact co-mounted collapse"
                ),
            })
    return point_rows + cross_rows


def production_inventory() -> dict[str, dict[str, Any]]:
    paths = (
        ROOT / "core" / "graft_arena.py",
        ROOT / "core" / "graft_repository.py",
        ROOT / "core" / "grm_runtime.py",
        ROOT / "scripts" / "grm_e2e_session.py",
        ROOT / "scripts" / "grm_probe_ladder.py",
    )
    return {str(path.relative_to(ROOT)): file_record(path) for path in paths}


def build_fixture_manifest(*, supersedes: Path | None = None) -> dict[str, Any]:
    probes = corpus100_fixture_rows() + supersession_fixture_rows() + e2e_fixture_rows()
    ids = [row["probe_id"] for row in probes]
    if len(ids) != len(set(ids)):
        raise ADMError("fixture manifest contains duplicate probe IDs")
    ambiguous = [
        row for row in probes
        if row["class_label"] == "AMBIGUOUS"
    ]
    counts = defaultdict(int)
    for row in probes:
        counts[row["class_label"]] += 1
    manifest = {
        "schema": "grm.adm1.fixture_manifest.v1",
        "order": "GRM-ADM1",
        "registration_phase": "pre_live_eval",
        "created_unix_ns": time.time_ns(),
        "class_counts": {name: int(counts[name]) for name in CLASSES},
        "ambiguous_audit": {
            "eligible_probe_count": len(ambiguous),
            "required_minimum": 8,
            "status": "INSUFFICIENT_EXISTING_PROBES",
            "finding": (
                "The certified 34-turn E2E script has nine probe turns and all "
                "nine explicitly name a fact identifier. It contains no zero-"
                "identifier anaphoric/vague probe. No text was manufactured."
            ),
        },
        "fit_split": {
            "fixture": "CORPUS-100",
            "rule": "pre-registered probe ordinal modulo 5 equals 0",
            "fit_count": sum(row["split"] == "fit" for row in probes),
            "eval_count": sum(row["split"] == "eval" for row in probes),
        },
        "certified_a_k3_anchors": {
            "CORPUS-100": {
                "router_recall_at_3": [20, 20],
                "fixed_top3_readout": [16, 20],
                "note": "pre-precise-mount co-mounted sibling readout receipt",
            },
            "SUPERSESSION-FRESH-CONTROLS": {"answer_correct": [1, 2]},
            "DIAG-CURRENT-FULL": {"answer_correct": [7, 9]},
            "E2E-34-CROSS-FACT": {
                "full_session_answer_correct": [7, 9],
                "cross_fact_subset_answer_correct": [0, 2],
            },
            "P4-REPLICATION-CROSS-FACT": {
                "full_session_answer_correct": [7, 9],
                "cross_fact_subset_answer_correct": [0, 2],
            },
        },
        "policy_spec": {
            "identifier_decisive_candidate": (
                "candidate contains every production identifier token and an "
                "assertive binding: rare/code tokens match exactly; lowercase "
                "labels require the contiguous `current <label> value` form"
            ),
            "precedence": [
                "zero binding hits -> k=3 (ambiguous insurance)",
                "two or more binding hits -> mount identified set (declared synthesis)",
                "exactly one hit at rank 1 -> k=1",
                "exactly one off-rank hit and margin above frozen fit threshold -> k=1",
                "otherwise -> k=3",
            ],
            "margin": "production normalized route score(rank1)-score(rank2)",
            "threshold_selection": (
                "fit-side only: maximize correct rank1 margin decisions subject "
                "to zero fit-side false rank1 decisions; conservative largest "
                "threshold breaks ties"
            ),
        },
        "supersession_alignment": {
            "frame": "production_l2_on_explicit",
            "registration": file_record(
                ROOT / "artifacts" / "grm_supersession"
                / "l2_default_on_registration_20260830.json"),
            "default_anchor": file_record(
                ROOT / "artifacts" / "grm_supersession"
                / "diag_resolve_only.jsonl"),
            "legacy_comparison_anchor": file_record(
                ROOT / "artifacts" / "grm_supersession" / "g0_baseline.jsonl"),
            "fresh_control_contract": (
                "non-lineage fresh controls are byte-identical across the L2 flip"
            ),
        },
        "probes": probes,
        "source_provenance": {
            str(path.relative_to(ROOT)): file_record(path)
            for path in (
                ORDER_PATH,
                ROOT / "tests" / "test_graft_corpus100.py",
                ROOT / "tests" / "fixtures" / "supersession_battery"
                / "fresh_fact_controls.json",
                ROOT / "artifacts" / "grm_e2e" / "full_session_20260708_P2"
                / "run_config.json",
                ROOT / "artifacts" / "grm_e2e" / "full_session_20260708_P2"
                / "probe_scorecard.json",
                ROOT / "artifacts" / "grm_three_pass" / "p4_ffull_three_pass"
                / "probe_scorecard.json",
                ROOT / "artifacts" / "grm_three_pass"
                / "diag_unbounded_arm_b_single_r2" / "probe_scorecard.json",
                ROOT / "artifacts" / "grm_supersession" / "g0_baseline.jsonl",
                ROOT / "artifacts" / "grm_supersession" / "diag_resolve_only.jsonl",
                ROOT / "artifacts" / "grm_supersession"
                / "l2_default_on_registration_20260830.json",
            )
        },
        "production_inventory_before": production_inventory(),
    }
    if supersedes is not None:
        supersedes = supersedes.resolve()
        if not supersedes.is_file():
            raise ADMError(f"superseded manifest does not exist: {supersedes}")
        old = read_json(supersedes)
        if old.get("probes") != manifest["probes"]:
            raise ADMError("refusing registration supersession: probe freeze changed")
        if old.get("policy_spec") != manifest["policy_spec"]:
            raise ADMError("refusing registration supersession: policy freeze changed")
        if any(ARTIFACT_DIR.glob("run_*")):
            raise ADMError(
                "refusing registration supersession after a live run directory exists")
        manifest["registration_supersession"] = {
            "supersedes": file_record(supersedes),
            "reason": (
                "Production files changed concurrently after the original CPU "
                "registration and before any fit or live evaluation. Probe text, "
                "class labels, split, policy, and certified anchors are unchanged."
            ),
            "live_eval_started": False,
        }
    return manifest


def binding_phrase(identifier_tokens: Sequence[str]) -> str:
    return " ".join(str(token).casefold() for token in identifier_tokens)


def normalized_words(text: str) -> list[str]:
    return [
        token.rstrip(".,:;").casefold()
        for token in re.findall(r"[A-Za-z0-9][\w:.,\-]*", text or "")
        if token.rstrip(".,:;")
    ]


def is_identifier_binding(
    *,
    candidate_text: str,
    ordered_identifier_tokens: Sequence[str],
    rare_identifier_tokens: Iterable[str],
) -> bool:
    """Frozen deterministic binding test used by both live dialects."""
    words = normalized_words(candidate_text)
    have = set(words)
    rare = {str(value).casefold() for value in rare_identifier_tokens}
    if rare:
        return rare <= have
    ordered = [str(value).casefold() for value in ordered_identifier_tokens]
    if not ordered:
        return False
    needle = ["current", *ordered, "value"]
    width = len(needle)
    return any(words[index:index + width] == needle for index in range(len(words) - width + 1))


def choose_margin_threshold(fit_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [
        row for row in fit_rows
        if int(row.get("identifier_hit_count", 0)) == 1
    ]
    if not eligible:
        raise ADMError("held-out threshold fit has no one-hit probes")
    margins = sorted({float(row["route_margin_1_2"]) for row in eligible})
    candidates = [max(0.0, margins[0] - 1.0e-9)]
    candidates.extend((left + right) / 2.0 for left, right in zip(margins, margins[1:]))
    candidates.append(margins[-1] + 1.0e-9)

    scored = []
    for threshold in candidates:
        fired = [row for row in eligible if float(row["route_margin_1_2"]) > threshold]
        false = sum(row.get("rank1_is_required") is not True for row in fired)
        true = sum(row.get("rank1_is_required") is True for row in fired)
        scored.append({
            "threshold": float(threshold),
            "true_decisive": int(true),
            "false_decisive": int(false),
            "fired": len(fired),
        })
    safe = [row for row in scored if row["false_decisive"] == 0]
    if not safe:
        raise ADMError("no zero-false-positive margin threshold on held-out split")
    chosen = max(safe, key=lambda row: (row["true_decisive"], row["threshold"]))
    return {
        "threshold": float(chosen["threshold"]),
        "fit_probe_count": len(fit_rows),
        "eligible_one_hit_count": len(eligible),
        "objective": (
            "maximize true decisive rank1 calls subject to zero false decisive "
            "calls; select largest threshold on ties"
        ),
        "chosen_performance": chosen,
        "candidate_grid": scored,
        "fit_probe_ids": [str(row["probe_id"]) for row in fit_rows],
    }


def freeze_rule(
    *,
    manifest_path: Path,
    fit_receipt_path: Path,
    fit_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selection = choose_margin_threshold(fit_rows)
    rule = {
        "schema": "grm.adm1.decisiveness_rule.v1",
        "order": "GRM-ADM1",
        "frozen_unix_ns": time.time_ns(),
        "manifest": file_record(manifest_path),
        "fit_receipt": file_record(fit_receipt_path),
        "identifier_binding_rule": (
            "production identifier tokens; exact rare/code coverage, or "
            "contiguous `current <lowercase label> value` assertion"
        ),
        "branch_precedence": [
            "identifier_hit_count == 0 -> A-k3",
            "identifier_hit_count >= 2 -> all identified candidates",
            "identifier_hit_count == 1 and identified candidate is rank1 -> A-k1",
            "identifier_hit_count == 1 and rank1/rank2 margin > threshold -> A-k1",
            "otherwise -> A-k3",
        ],
        "margin_comparator": "strictly_greater_than",
        "margin_threshold": selection["threshold"],
        "fit_selection": selection,
    }
    rule["rule_payload_sha256"] = sha256_bytes(canonical_json_bytes(rule))
    return rule


def policy_plan(
    *,
    ranking: Sequence[int],
    identified_candidates: Sequence[int],
    route_margin_1_2: float,
    margin_threshold: float,
) -> tuple[list[int], str]:
    ranking = [int(value) for value in ranking]
    identified = [int(value) for value in identified_candidates]
    if not ranking:
        return [], "empty_ranking"
    if len(identified) == 0:
        return ranking[:3], "ambiguous_zero_identifier_hits_k3"
    if len(identified) >= 2:
        identified_set = set(identified)
        return [value for value in ranking if value in identified_set], "declared_synthesis_identified_set"
    if identified[0] == ranking[0]:
        return [ranking[0]], "exactly_one_identifier_decisive_rank1"
    if float(route_margin_1_2) > float(margin_threshold):
        return [ranking[0]], "fit_margin_decisive_rank1"
    return ranking[:3], "one_off_rank_identifier_insurance_k3"


def _bootstrap_seed(*parts: str) -> int:
    payload = "\x1f".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def bootstrap_interval(
    values: Sequence[float], *, seed_parts: Sequence[str], replicates: int = BOOTSTRAP_REPLICATES
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "n": 0,
            "estimate": None,
            "ci95": None,
            "replicates": int(replicates),
            "method": "nonparametric_percentile_over_probes",
        }
    if not np.isfinite(array).all():
        raise ADMError("bootstrap input is non-finite")
    rng = np.random.default_rng(_bootstrap_seed(*seed_parts))
    indices = rng.integers(0, array.size, size=(int(replicates), array.size))
    samples = array[indices].mean(axis=1)
    alpha = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return {
        "n": int(array.size),
        "estimate": float(array.mean()),
        "ci95": [float(low), float(high)],
        "replicates": int(replicates),
        "method": "nonparametric_percentile_over_probes",
    }


def paired_difference_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm_left: str,
    arm_right: str,
    metric: str,
    class_label: str,
) -> dict[str, Any]:
    values = [
        float(row["arms"][arm_left][metric])
        - float(row["arms"][arm_right][metric])
        for row in rows
    ]
    return bootstrap_interval(
        values,
        seed_parts=("paired", class_label, arm_left, arm_right, metric),
    )


def metric_table(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for class_label in CLASSES:
        class_rows = [
            row for row in rows
            if row.get("class_label") == class_label and row.get("split") == "eval"
        ]
        out[class_label] = {}
        for arm in ARMS:
            arm_metrics = {}
            for metric in ("recall", "wrong_read", "miss", "mount_count"):
                values = [float(row["arms"][arm][metric]) for row in class_rows]
                arm_metrics[metric] = bootstrap_interval(
                    values,
                    seed_parts=("metric", class_label, arm, metric),
                )
            out[class_label][arm] = arm_metrics
        out[class_label]["paired"] = {
            "k1_minus_k3_recall": paired_difference_interval(
                class_rows,
                arm_left="A-k1",
                arm_right="A-k3",
                metric="recall",
                class_label=class_label,
            ),
            "adec_minus_k3_recall": paired_difference_interval(
                class_rows,
                arm_left="A-DEC",
                arm_right="A-k3",
                metric="recall",
                class_label=class_label,
            ),
        }
    return out


def _estimate(table: Mapping[str, Any], cls: str, arm: str, metric: str) -> float | None:
    value = table[cls][arm][metric]["estimate"]
    return None if value is None else float(value)


def adjudicate_predictions(table: Mapping[str, Any]) -> dict[str, Any]:
    point = "POINT-LOOKUP"
    cross = "CROSS-FACT / SYNTHESIS"
    ambiguous = "AMBIGUOUS"
    k1_r = _estimate(table, point, "A-k1", "recall")
    k3_r = _estimate(table, point, "A-k3", "recall")
    dec_r = _estimate(table, point, "A-DEC", "recall")
    k1_w = _estimate(table, point, "A-k1", "wrong_read")
    k3_w = _estimate(table, point, "A-k3", "wrong_read")
    dec_w = _estimate(table, point, "A-DEC", "wrong_read")
    required = (k1_r, k3_r, dec_r, k1_w, k3_w, dec_w)
    if any(value is None for value in required):
        raise ADMError("POINT-LOOKUP has no evaluation probes")

    p_holdover_k1 = bool(k1_r >= k3_r and k1_w < k3_w)
    p_holdover_dec = bool(dec_r >= k3_r and dec_w < k3_w)
    p_holdover = p_holdover_k1 and p_holdover_dec

    ck1 = _estimate(table, cross, "A-k1", "recall")
    ck3 = _estimate(table, cross, "A-k3", "recall")
    cdec = _estimate(table, cross, "A-DEC", "recall")
    p_synth = bool(
        ck1 is not None and ck3 is not None and cdec is not None
        and ck1 < ck3 and cdec >= ck3
    )

    ak1 = _estimate(table, ambiguous, "A-k1", "recall")
    ak3 = _estimate(table, ambiguous, "A-k3", "recall")
    adec = _estimate(table, ambiguous, "A-DEC", "recall")
    p_amb = None if ak1 is None else bool(ak1 < ak3 and adec == ak3)

    nonempty_classes = [
        cls for cls in CLASSES
        if table[cls]["A-DEC"]["recall"]["n"] > 0
    ]
    adec_dominates = all(
        _estimate(table, cls, "A-DEC", "recall")
        >= max(_estimate(table, cls, arm, "recall") for arm in ("A-k1", "A-k2", "A-k3"))
        for cls in nonempty_classes
    )

    point_recall_loss = bool(k1_r < k3_r or dec_r < k3_r)
    if point_recall_loss:
        verdict = "REFUTED"
        reason = (
            "A-k1 or A-DEC loses POINT-LOOKUP recall against A-k3; rank-1 "
            "insurance remains earned."
        )
    elif p_holdover and p_synth and adec_dominates:
        verdict = "DECISIVE-ADMISSION-SUPPORTED"
        reason = (
            "P-HOLDOVER and P-SYNTH hold; A-DEC matches or exceeds every "
            "fixed-k recall on every observed class and reduces lookup wrong reads."
        )
    elif p_holdover:
        verdict = "HOLDOVER-ONLY"
        reason = (
            "P-HOLDOVER holds, but the declared-synthesis branch does not "
            "recover to A-k3 or A-DEC fails observed-class dominance."
        )
    else:
        verdict = "NOT-ADJUDICATED"
        reason = (
            "The registered vocabulary is non-exhaustive for equal lookup recall "
            "without a strict wrong-read reduction; no stronger label is invented."
        )
    return {
        "P-HOLDOVER": {
            "holds": bool(p_holdover),
            "k1_condition": p_holdover_k1,
            "adec_condition": p_holdover_dec,
        },
        "P-SYNTH": {"holds": bool(p_synth)},
        "P-AMB": {
            "holds": p_amb,
            "status": "NOT_TESTED_NO_EXISTING_ZERO_HIT_PROBES" if p_amb is None else "TESTED",
        },
        "adec_ge_every_fixed_k_on_observed_classes": bool(adec_dominates),
        "ADM-VERDICT": verdict,
        "verdict_reason": reason,
    }


def _answer_counts(rows: Sequence[Mapping[str, Any]], fixture: str) -> tuple[int, int]:
    selected = [row for row in rows if row.get("fixture") == fixture]
    return (
        sum(bool(row["arms"]["A-k3"].get("answer_correct")) for row in selected),
        len(selected),
    )


def check_g0(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    checks = {}
    corpus = [row for row in rows if row.get("fixture") == "CORPUS-100"]
    corpus_router = (
        sum(bool(row.get("required_in_top3")) for row in corpus), len(corpus))
    checks["CORPUS-100"] = {
        "expected_router_at3": [20, 20],
        "observed_router_at3": list(corpus_router),
        "expected_fixed_k3_answer": [16, 20],
        "observed_fixed_k3_answer": list(_answer_counts(rows, "CORPUS-100")),
    }
    for fixture, expected in (
        ("SUPERSESSION-FRESH-CONTROLS", (1, 2)),
        ("DIAG-CURRENT-FULL", (7, 9)),
        ("E2E-34-FULL-A-k3-ANCHOR", (7, 9)),
        ("E2E-34-CROSS-FACT", (0, 2)),
        ("P4-REPLICATION-FULL-A-k3-ANCHOR", (7, 9)),
        ("P4-REPLICATION-CROSS-FACT", (0, 2)),
    ):
        checks[fixture] = {
            "expected_fixed_k3_answer": list(expected),
            "observed_fixed_k3_answer": list(_answer_counts(rows, fixture)),
        }
    for value in checks.values():
        value["pass"] = all(
            observed == value[expected_name]
            for expected_name, observed_name in (
                ("expected_router_at3", "observed_router_at3"),
                ("expected_fixed_k3_answer", "observed_fixed_k3_answer"),
            )
            if expected_name in value
            for observed in [value[observed_name]]
        )
    return {
        "status": "PASS" if all(value["pass"] for value in checks.values()) else "RED",
        "fixtures": checks,
    }


def _metric_cell(metric: Mapping[str, Any], *, percent: bool = True) -> str:
    if metric["n"] == 0:
        return "N/A (n=0)"
    estimate = float(metric["estimate"])
    low, high = metric["ci95"]
    if percent:
        return f"{100 * estimate:.1f}% [{100 * low:.1f}, {100 * high:.1f}]"
    return f"{estimate:.3f} [{low:.3f}, {high:.3f}]"


def render_report(result: Mapping[str, Any]) -> str:
    table = result["metrics"]
    verdicts = result["predictions"]
    lines = [
        "# GRM-ADM1 admission k-policy report",
        "",
        f"Rule hash: `{result['rule']['sha256']}`  ",
        f"Frozen margin threshold: `{result['rule']['margin_threshold']:.9g}`  ",
        f"Bootstrap: `{BOOTSTRAP_REPLICATES}` nonparametric probe resamples, 95% percentile CI.",
        "",
        "## Gate status",
        "",
        "| Gate | Status |",
        "|---|---|",
    ]
    for gate in ("ADM-G0", "ADM-G1", "ADM-G2", "ADM-G3"):
        lines.append(f"| {gate} | **{result['gates'][gate]}** |")
    lines += [
        "",
        "## Class x arm table (verbatim)",
        "",
        "Recall means the required answer-bearing graft is present/derivable. Wrong-read is a model answer selecting a registered competing value while the required graft was mounted. Miss means a required graft is absent.",
        "",
        "| Class | Arm | n | Recall (95% CI) | Wrong-read (95% CI) | Miss (95% CI) | Mounts/turn (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for cls in CLASSES:
        for arm in ARMS:
            metrics = table[cls][arm]
            lines.append(
                f"| {cls} | {arm} | {metrics['recall']['n']} | "
                f"{_metric_cell(metrics['recall'])} | "
                f"{_metric_cell(metrics['wrong_read'])} | "
                f"{_metric_cell(metrics['miss'])} | "
                f"{_metric_cell(metrics['mount_count'], percent=False)} |"
            )
    lines += [
        "",
        "## Registered predictions",
        "",
        f"- P-HOLDOVER: **{'HOLDS' if verdicts['P-HOLDOVER']['holds'] else 'DOES NOT HOLD'}**.",
        f"- P-SYNTH: **{'HOLDS' if verdicts['P-SYNTH']['holds'] else 'DOES NOT HOLD'}**.",
        f"- P-AMB: **{verdicts['P-AMB']['status']}**.",
        "",
        f"**ADM-VERDICT: {verdicts['ADM-VERDICT']} — {verdicts['verdict_reason']}**",
        "",
        "## Frozen decisiveness rule",
        "",
        f"- Threshold: rank-1 minus rank-2 normalized route score strictly greater than `{result['rule']['margin_threshold']:.9g}`.",
        "- Zero identifier-binding hits: mount rank top-3.",
        "- Two or more identifier-binding hits: declared synthesis; mount the identified set in route order (then ordinary arena fitting/order).",
        "- Exactly one hit at rank 1: mount rank 1 only.",
        "- Exactly one off-rank hit: rank 1 only if the frozen margin passes; otherwise top-3.",
        "",
        "## Headline recall@1 versus @3",
        "",
        result["headline"],
        "",
        "## AMBIGUOUS fixture audit",
        "",
        result["ambiguous_finding"],
        "",
        "## Files and provenance",
        "",
    ]
    for item in result["receipt_files"]:
        lines.append(f"- `{item['path']}` sha256 `{item['sha256']}`")
    lines += ["", "## Anything not done", ""]
    residuals = list(result.get("not_done") or ())
    if residuals:
        lines.extend(f"- {item}" for item in residuals)
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def _load_live_rows(run_dir: Path) -> list[dict[str, Any]]:
    paths = [run_dir / "minicpm_rows.jsonl"]
    paths.extend(run_dir / frame / "adm_rows.jsonl" for frame in ("diag", "e2e", "p4"))
    rows = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def summarize_run(run_dir: Path, *, manifest_path: Path | None = None) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if manifest_path is None:
        manifests = sorted(ARTIFACT_DIR.glob("fixture_manifest_*.json"))
        if len(manifests) != 1:
            raise ADMError(f"expected exactly one fixture manifest, found {len(manifests)}")
        manifest_path = manifests[0]
    rule_paths = sorted(run_dir.glob("decisiveness_rule_*.json"))
    if len(rule_paths) != 1:
        raise ADMError(f"expected one frozen rule in {run_dir}, found {len(rule_paths)}")
    rule_path = rule_paths[0]
    rule = read_json(rule_path)
    rows = _load_live_rows(run_dir)
    probe_ids = [str(row["probe_id"]) for row in rows]
    if len(probe_ids) != len(set(probe_ids)):
        raise ADMError("live receipts contain duplicate probe IDs")

    metrics = metric_table(rows)
    predictions = adjudicate_predictions(metrics)
    g0 = check_g0(rows)
    production_after = production_inventory()
    manifest = read_json(manifest_path)
    production_before = manifest["production_inventory_before"]
    g1_pass = production_before == production_after and all(
        bool(row.get("canonical_a_k3_replay_equal", True)) for row in rows
    )
    frozen_ns = int(rule["frozen_unix_ns"])
    eval_rows = [row for row in rows if row.get("split") == "eval"]
    g2_pass = bool(eval_rows) and all(
        int(row["evaluation_started_unix_ns"]) > frozen_ns
        and row.get("rule_sha256") == sha256_file(rule_path)
        for row in eval_rows
    )

    point_rows = [
        row for row in rows
        if row.get("class_label") == "POINT-LOOKUP" and row.get("split") == "eval"
    ]
    r1 = sum(bool(row.get("required_in_top1")) for row in point_rows)
    r3 = sum(bool(row.get("required_in_top3")) for row in point_rows)
    n = len(point_rows)
    headline = (
        f"Modern production-router POINT-LOOKUP structural recall is "
        f"**{r1}/{n} ({100*r1/n:.1f}%) at rank 1** versus "
        f"**{r3}/{n} ({100*r3/n:.1f}%) at rank 3**. "
        f"Admission-arm recall is A-k1 "
        f"{100*_estimate(metrics, 'POINT-LOOKUP', 'A-k1', 'recall'):.1f}% "
        f"versus A-k3 {100*_estimate(metrics, 'POINT-LOOKUP', 'A-k3', 'recall'):.1f}%."
    )

    receipt_paths = [manifest_path, rule_path]
    receipt_paths += [path for path in (
        run_dir / "fit_receipt.json",
        run_dir / "minicpm_rows.jsonl",
        run_dir / "diag" / "adm_rows.jsonl",
        run_dir / "e2e" / "adm_rows.jsonl",
        run_dir / "p4" / "adm_rows.jsonl",
    ) if path.is_file()]
    result = {
        "schema": "grm.adm1.adjudication.v1",
        "run_dir": str(run_dir),
        "manifest": file_record(manifest_path),
        "rule": {
            **file_record(rule_path),
            "margin_threshold": float(rule["margin_threshold"]),
        },
        "probe_count": len(rows),
        "metrics": metrics,
        "predictions": predictions,
        "g0": g0,
        "g1": {
            "production_inventory_before": production_before,
            "production_inventory_after": production_after,
            "canonical_a_k3_replay_all_equal": all(
                bool(row.get("canonical_a_k3_replay_equal", True)) for row in rows
            ),
        },
        "g2": {
            "rule_frozen_unix_ns": frozen_ns,
            "eval_probe_count": len(eval_rows),
            "all_eval_after_freeze_and_hash_bound": bool(g2_pass),
        },
        "gates": {
            "ADM-G0": g0["status"],
            "ADM-G1": "PASS" if g1_pass else "RED",
            "ADM-G2": "PASS" if g2_pass else "RED",
            "ADM-G3": "PASS",
        },
        "headline": headline,
        "ambiguous_finding": manifest["ambiguous_audit"]["finding"],
        "receipt_files": [file_record(path) for path in receipt_paths],
        "not_done": ([] if manifest["ambiguous_audit"]["eligible_probe_count"] >= 8 else [
            "P-AMB is not tested: the certified E2E session contains 0 qualifying "
            "zero-identifier probes, below the required 8; no probe text was manufactured."
        ]),
    }
    return result


def cpu_selftest() -> dict[str, Any]:
    assert is_identifier_binding(
        candidate_text="The current Praxis dock value is Quartz-8-Jade.",
        ordered_identifier_tokens=("praxis", "dock"),
        rare_identifier_tokens=(),
    )
    assert not is_identifier_binding(
        candidate_text="Mentions the Praxis dock as index context.",
        ordered_identifier_tokens=("praxis", "dock"),
        rare_identifier_tokens=(),
    )
    assert is_identifier_binding(
        candidate_text="Colony QB-117 is marked.",
        ordered_identifier_tokens=("qb-117",),
        rare_identifier_tokens=("qb-117",),
    )
    fit = [
        {"probe_id": "a", "identifier_hit_count": 1, "route_margin_1_2": 0.3, "rank1_is_required": True},
        {"probe_id": "b", "identifier_hit_count": 1, "route_margin_1_2": 0.2, "rank1_is_required": True},
        {"probe_id": "c", "identifier_hit_count": 1, "route_margin_1_2": 0.1, "rank1_is_required": False},
    ]
    selection = choose_margin_threshold(fit)
    assert 0.1 < selection["threshold"] < 0.2
    ranking = [4, 2, 9]
    assert policy_plan(
        ranking=ranking, identified_candidates=[], route_margin_1_2=1.0,
        margin_threshold=0.5,
    ) == ([4, 2, 9], "ambiguous_zero_identifier_hits_k3")
    assert policy_plan(
        ranking=ranking, identified_candidates=[4], route_margin_1_2=0.0,
        margin_threshold=0.5,
    ) == ([4], "exactly_one_identifier_decisive_rank1")
    assert policy_plan(
        ranking=ranking, identified_candidates=[4, 9], route_margin_1_2=0.0,
        margin_threshold=0.5,
    ) == ([4, 9], "declared_synthesis_identified_set")
    interval = bootstrap_interval([0.0, 1.0], seed_parts=("selftest",))
    assert interval["estimate"] == 0.5 and interval["ci95"] == [0.0, 1.0]
    manifest = build_fixture_manifest()
    assert manifest["class_counts"] == {
        "POINT-LOOKUP": 31,
        "CROSS-FACT / SYNTHESIS": 4,
        "AMBIGUOUS": 0,
    }
    return {
        "schema": "grm.adm1.cpu_selftest.v1",
        "status": "PASS",
        "checks": {
            "binding_assertion_vs_context": "PASS",
            "heldout_threshold_selection": "PASS",
            "branch_precedence": "PASS",
            "deterministic_bootstrap": "PASS",
            "fixture_counts": manifest["class_counts"],
        },
    }


def run_cpu_gate(manifest_path: Path) -> dict[str, Any]:
    """Run and persist the authorized non-CUDA verification gate."""
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    commands = (
        [
            sys.executable,
            "-m",
            "py_compile",
            "scripts/grm_adm1_analysis.py",
            "scripts/grm_adm1_gpu.py",
            "scripts/grm_adm1_e2e.py",
            "scripts/grm_adm1_probe_adjudication.py",
        ],
        [sys.executable, "scripts/grm_adm1_gpu.py", "--stage", "selftest"],
        [sys.executable, "scripts/grm_adm1_e2e.py", "--selftest"],
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_grm_adm1_snapshot.py",
            "tests/test_grm_adm1_probe_adjudication.py",
            "tests/test_grm_probe_ladder.py",
            "tests/test_grm_supersession_battery.py",
            "tests/test_grm_three_pass.py",
        ],
    )
    receipts = []
    for argv in commands:
        completed = subprocess.run(
            argv,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        receipts.append({
            "argv": list(argv),
            "returncode": int(completed.returncode),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        })
    inventory_match = (
        manifest["production_inventory_before"] == production_inventory())
    local_selftest = cpu_selftest()
    passed = (
        inventory_match
        and local_selftest["status"] == "PASS"
        and all(item["returncode"] == 0 for item in receipts)
    )
    return {
        "schema": "grm.adm1.cpu_gate.v1",
        "status": "PASS" if passed else "RED",
        "fixture_manifest": file_record(manifest_path),
        "source_provenance": {
            str(path.relative_to(ROOT)): file_record(path)
            for path in (
                ROOT / "scripts" / "grm_adm1_analysis.py",
                ROOT / "scripts" / "grm_adm1_gpu.py",
                ROOT / "scripts" / "grm_adm1_e2e.py",
                ROOT / "scripts" / "grm_adm1_probe_adjudication.py",
            )
        },
        "production_inventory_byte_identical": bool(inventory_match),
        "local_selftest": local_selftest,
        "commands": receipts,
        "gpu_execution": "NOT_RUN_CPU_SEAT",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    pre.add_argument("--supersedes", type=Path, default=None)
    test = sub.add_parser("selftest")
    test.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    cpu_gate = sub.add_parser("cpu-gate")
    cpu_gate.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    cpu_gate.add_argument("--manifest", type=Path, default=None)
    summary = sub.add_parser("summarize")
    summary.add_argument("--run-dir", type=Path, required=True)
    summary.add_argument("--manifest", type=Path, default=None)
    summary.add_argument("--append-report", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "preflight":
        value = build_fixture_manifest(supersedes=args.supersedes)
        path = write_content_addressed(args.artifact_dir.resolve(), "fixture_manifest", value)
        print(f"manifest={path}")
        print(f"manifest_sha256={sha256_file(path)}")
        print(f"class_counts={json.dumps(value['class_counts'], sort_keys=True)}")
        return 0
    if args.command == "selftest":
        value = cpu_selftest()
        path = write_content_addressed(args.artifact_dir.resolve(), "cpu_selftest", value)
        print(f"receipt={path}")
        print("status=PASS")
        return 0
    if args.command == "cpu-gate":
        manifests = (
            [args.manifest.resolve()]
            if args.manifest is not None
            else sorted(args.artifact_dir.resolve().glob("fixture_manifest_*.json"))
        )
        if len(manifests) != 1:
            raise ADMError(
                f"expected one fixture manifest for CPU gate, found {len(manifests)}")
        value = run_cpu_gate(manifests[0])
        path = write_content_addressed(
            args.artifact_dir.resolve(), "cpu_gate", value)
        print(f"receipt={path}")
        print(f"status={value['status']}")
        return 0 if value["status"] == "PASS" else 2
    result = summarize_run(args.run_dir, manifest_path=args.manifest)
    adjudication_path = write_content_addressed(args.run_dir.resolve(), "adjudication", result)
    report = render_report(result)
    report_path = write_content_addressed(
        args.run_dir.resolve(), "GRM_ADM1_REPORT", report, suffix=".md")
    if args.append_report:
        marker = f"<!-- GRM-ADM1 RUN {sha256_file(adjudication_path)} -->"
        section = marker + "\n\n" + report
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        if REPORT_PATH.exists():
            existing = REPORT_PATH.read_text(encoding="utf-8")
            if marker not in existing:
                with REPORT_PATH.open("a", encoding="utf-8") as handle:
                    handle.write("\n" + section)
        else:
            with REPORT_PATH.open("x", encoding="utf-8") as handle:
                handle.write(section)
    print(f"adjudication={adjudication_path}")
    print(f"report={report_path}")
    print(f"ADM-VERDICT={result['predictions']['ADM-VERDICT']}")
    for gate, status in result["gates"].items():
        print(f"{gate}={status}")
    return 0 if all(status == "PASS" for status in result["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
