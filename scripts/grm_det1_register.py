#!/usr/bin/env python3
"""Freeze GRM-DET1 fixtures, split, detector semantics, and provenance pre-data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_det1_common import (  # noqa: E402
    ARTIFACT_ROOT,
    CALIBRATION_IDS,
    EVAL_E2E_IDS,
    ORDER_PATH,
    VERBAL_QUESTION,
    DETError,
    file_record,
    source_inventory,
    utc_now,
    write_content_addressed,
    write_json_exclusive,
)


CERTIFIED_SESSION = (
    ROOT / "artifacts" / "grm_adm1"
    / "adm1_3_prod_run_20260830T161236Z_3eaccb"
    / "e2e" / "session"
)
SUP_FIXTURE_DIR = ROOT / "tests" / "fixtures" / "supersession_battery"
MODEL_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
MODEL_DIR = Path(
    "/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/"
    f"snapshots/{MODEL_REVISION}"
)
NATIVE_LIB = ROOT / "cpp" / "build" / "libgrm_runtime.so"
TENSOR_CUDA_ROOT = Path("/mnt/ForgeRealm/Project-Tensor/tensor_cuda/tensor_cuda")
E2E_TURNS = {
    5: "e2e_t05_orion_pin",
    9: "e2e_t09_cypher_bridge",
    13: "e2e_t13_orion_pin",
    16: "e2e_t16_lyra_dock",
    19: "e2e_t19_nova_key",
    22: "e2e_t22_mira_seal",
    24: "e2e_t24_terra_port",
    26: "e2e_t26_ember_code",
    30: "e2e_t30_atlas_tone",
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fixtures() -> list[dict[str, Any]]:
    config_path = CERTIFIED_SESSION / "run_config.json"
    scorecard_path = CERTIFIED_SESSION / "probe_scorecard.json"
    config = _read(config_path)
    scorecard = _read(scorecard_path)
    if len(config.get("script") or ()) != 34:
        raise DETError("certified source is not a 34-turn session")
    if not (
        scorecard.get("all_passed") is True
        and int(scorecard.get("passed", -1)) == 9
        and int(scorecard.get("total", -1)) == 9
    ):
        raise DETError("selected certified receipt is not ladder-on 9/9")
    fixtures = []
    for turn, fixture_id in E2E_TURNS.items():
        event = config["script"][turn]
        if event.get("kind") != "probe":
            raise DETError(f"certified turn {turn} is not a probe")
        split = "calibration" if fixture_id in CALIBRATION_IDS else "eval"
        fixtures.append({
            "fixture_id": fixture_id,
            "source_family": "certified_34_turn",
            "split": split,
            "turn": int(turn),
            "question": str(event["user"]),
            "expected_values": [str(event["expected"])],
            "old_values": ([str(event["old_value"])]
                           if event.get("old_value") is not None else []),
            "source_turn": int(event["source_turn"]),
            "fact_id": str(event["fact_id"]),
            "source": file_record(config_path),
        })

    for path in sorted(SUP_FIXTURE_DIR.glob("*.json")):
        fixture = _read(path)
        for probe in fixture["probes"]:
            fixture_id = f"sup_{probe['probe_id']}"
            fixtures.append({
                "fixture_id": fixture_id,
                "source_family": "supersession_battery_on_gpt_oss",
                "split": "eval",
                "session_id": str(fixture["session_id"]),
                "scenario": str(fixture["scenario"]),
                "probe_id": str(probe["probe_id"]),
                "question": str(probe["question"]),
                "expected_values": [str(value) for value in probe["expected_values"]],
                "stale_values": [str(value) for value in probe["stale_values"]],
                "wrong_fact_values": [str(value) for value in probe["wrong_fact_values"]],
                "target_node": str(probe["target_node"]),
                "source": file_record(path),
            })
    ids = [row["fixture_id"] for row in fixtures]
    if len(fixtures) != 14 or len(set(ids)) != 14:
        raise DETError(f"expected 14 unique source probes, observed {len(fixtures)}")
    eval_ids = [row["fixture_id"] for row in fixtures if row["split"] == "eval"]
    if len(eval_ids) != 12 or set(EVAL_E2E_IDS) - set(eval_ids):
        raise DETError(f"registered eval split is not exactly 12: {eval_ids}")
    calibration = [row for row in fixtures if row["split"] == "calibration"]
    evaluation = [row for row in fixtures if row["split"] == "eval"]
    if (
        {row["question"].casefold() for row in calibration}
        & {row["question"].casefold() for row in evaluation}
    ):
        raise DETError("calibration and eval share a query")
    if (
        {row["fact_id"].casefold() for row in calibration}
        & {row["fact_id"].casefold() for row in evaluation if "fact_id" in row}
    ):
        raise DETError("calibration and eval share a certified-session fact")
    return fixtures


def build_registration() -> dict[str, Any]:
    detector_paths = sorted((ROOT / "scripts").glob("grm_det1_*.py"))
    fixture_paths = sorted(SUP_FIXTURE_DIR.glob("*.json"))
    production_paths = [
        ORDER_PATH,
        ROOT / "scripts" / "grm_e2e_session.py",
        ROOT / "scripts" / "grm_probe_ladder.py",
        ROOT / "core" / "graft_arena.py",
        ROOT / "core" / "graft_repository.py",
        ROOT / "core" / "grm_admission.py",
        ROOT / "core" / "gpt_oss20b_tc.py",
        ROOT / "core" / "kv_graft.py",
        ROOT / "core" / "grm_supersession.py",
        NATIVE_LIB,
        TENSOR_CUDA_ROOT / "__init__.py",
        TENSOR_CUDA_ROOT / "_tensor_cuda.cpython-312-x86_64-linux-gnu.so",
        CERTIFIED_SESSION / "run_config.json",
        CERTIFIED_SESSION / "probe_scorecard.json",
        *fixture_paths,
        *detector_paths,
    ]
    model_paths = [
        MODEL_DIR / "config.json",
        MODEL_DIR / "generation_config.json",
        MODEL_DIR / "tokenizer.json",
        MODEL_DIR / "tokenizer_config.json",
        MODEL_DIR / "special_tokens_map.json",
        MODEL_DIR / "model.safetensors.index.json",
        *sorted(MODEL_DIR.glob("model-*.safetensors")),
    ]
    model_inventory = {}
    for path in model_paths:
        if not path.is_file():
            raise DETError(f"model snapshot member is absent: {path}")
        model_inventory[str(path)] = {
            "bytes": path.stat().st_size,
            "resolved_blob": path.resolve().name,
            "verification": "huggingface_content_addressed_blob_identity",
        }
    return {
        "schema": "grm.det1.registration.v1",
        "order": "GRM-DET1",
        "registered_pre_data": True,
        "registered_utc": utc_now(),
        "registered_unix_ns": __import__("time").time_ns(),
        "lead_prediction": (
            "The S1/S2 split reproduces for need detection: mechanistic "
            "detectors beat the verbal ask and D-LQR wins. If D-VERB wins, "
            "the prediction is REFUTED."
        ),
        "verbal_question_exact": VERBAL_QUESTION,
        "verbal_pass_construction": (
            "original probe text + two newline bytes + verbal_question_exact"
        ),
        "fixtures": _fixtures(),
        "split_rule": {
            "unit": "underlying probe; paired variants never cross splits",
            "query_and_certified_fact_disjoint": True,
            "calibration": list(CALIBRATION_IDS),
            "eval_e2e": list(EVAL_E2E_IDS),
            "eval_supersession": [
                row["fixture_id"] for row in _fixtures()
                if row["source_family"] == "supersession_battery_on_gpt_oss"
            ],
            "eval_pair_count": 12,
            "calibration_pair_count": 2,
        },
        "runtime_frame_rule": {
            "model": "GPT-OSS-20B revision 6cee5e81ee83917806bbde320786a8fb61efebee",
            "one_model_for_both_source_families": True,
            "probe_ladder": "required ON; verify live default ON at first GPU lease, then pin and receipt",
            "l2_revision_resolution": "required ON; verify live default ON at first GPU lease, then pin and receipt",
            "admission": "resolve production default at GPU start, then pin and receipt",
            "l1_length_debias": "production default",
            "turn_pipeline": "production CLI default",
        },
        "planted_miss_rule": {
            "raw_physical_rank1_withheld": True,
            "logical_target": (
                "the raw router rank-1's surviving logical head after the "
                "required production L2 pass"
            ),
            "physical_aliases": (
                "the complete supersession-connected revision component plus, "
                "when the head contains the expected value, every eligible "
                "physical mirror containing that exact whole value"
            ),
            "intervention": (
                "compute the unmodified production full-index routing and, when "
                "active, A-DEC profile/branch; then remove only the logical "
                "rank-1 unit from A-DEC rank_plan and from every ladder attempt "
                "at the common pre-budget-fit admission boundary; other mounts, "
                "ladder, L2, and detector index are unchanged"
            ),
            "g0_checks": [
                "every withheld alias absent from fitted/current mounts",
                "registered expected value absent from every mounted graft text",
                "served control answer contains expected and no stale value",
            ],
            "rationale": (
                "L2 heads and complete-turn/correction mirrors can represent "
                "one logical graft with several physical rows; physical-rank1-only "
                "withholding can be a no-op and would violate DET-G0"
            ),
        },
        "detectors": {
            "D-LQR": {
                "signal": (
                    "generation-time route-layer pre-RoPE last-row query, "
                    "scored by existing GQA _cent_score/_normalize_scores "
                    "against the full active non-recall index"
                ),
                "lexical_channel": "none; the contextual model query is the live direction",
                "logical_index": "post-L2; planted physical aliases collapse to one logical graft",
                "trigger": "strict best_unmounted_score > best_mounted_score",
                "threshold_fit": "none",
            },
            "D-NGH": {
                "signal": (
                    "absolute mounted-seat probability from actual standard "
                    "GPT-OSS q/k/sink softmax operands, mean over heads and 12 "
                    "full-attention layers; sliding layers excluded"
                ),
                "trigger": "mounted_mass strictly below frozen fit-side threshold",
                "threshold_fit": "minimum of served calibration turn minima",
            },
            "D-ENT": {
                "signal": "full-vocabulary entropy and top1-top2 logit margin at answer positions",
                "trigger": "entropy strictly above fit max OR margin strictly below fit min",
                "threshold_fit": "served calibration per-turn extrema envelope",
            },
            "D-VERB": {
                "signal": "separate greedy pass with exact appended neutral question",
                "trigger": "whole-word NO in the answer; YES and all other output do not trigger",
            },
        },
        "latency": {
            "indexing": "zero-based generated-token prediction index",
            "headline_median_population": "triggered planted-miss true positives",
        },
        "metrics": {
            "positive": "planted_miss",
            "negative": "served",
            "recall": "TP / 12 planted misses",
            "false_positive_rate": "FP / 12 served controls",
            "f1": "harmonic mean of precision and recall",
            "viable": "recall >= 0.80 AND false_positive_rate <= 0.10",
            "race_winner": "all viable detectors tied at the highest exact F1",
        },
        "prediction_adjudication": {
            "REFUTED": (
                "D-VERB is a highest-F1 viable race winner OR its raw F1 "
                "ties the best mechanistic raw F1"
            ),
            "SUPPORTED": (
                "D-LQR is among highest-F1 viable winner(s) and every viable "
                "mechanistic detector has F1 > D-VERB"
            ),
            "PARTIAL": (
                "best mechanistic raw F1 > D-VERB and D-LQR is not a "
                "highest-F1 viable race winner"
            ),
            "UNADJUDICATED": (
                "the measurements satisfy none of the order's three iff "
                "clauses; report that registered-rule gap without relabeling"
            ),
        },
        "production_source_inventory": source_inventory(production_paths),
        "model_snapshot_inventory": model_inventory,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    registration = build_registration()
    if args.selftest:
        print(json.dumps({
            "schema": "grm.det1.registration_selftest.v1",
            "status": "PASS",
            "fixture_count": len(registration["fixtures"]),
            "calibration_count": sum(
                row["split"] == "calibration" for row in registration["fixtures"]),
            "eval_count": sum(row["split"] == "eval" for row in registration["fixtures"]),
        }, sort_keys=True))
        return 0
    if args.run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = args.artifact_root / f"run_{stamp}_{os.getpid()}"
    else:
        run_dir = args.run_dir
    run_dir = run_dir.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    required_root = ARTIFACT_ROOT.resolve()
    if artifact_root != required_root or not run_dir.is_relative_to(required_root):
        raise DETError(
            f"ORDER GRM-DET1 confines artifacts to {required_root}: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    registration_path = write_content_addressed(run_dir, "registration", registration)
    write_json_exclusive(run_dir / "run_manifest.json", {
        "schema": "grm.det1.run_manifest.v1",
        "created_utc": utc_now(),
        "run_dir": str(run_dir),
        "registration": file_record(registration_path),
        "status": "REGISTERED_PRE_DATA",
    })
    print(str(run_dir))
    print(str(registration_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
