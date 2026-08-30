#!/usr/bin/env python3
"""ADM1.3 production-ladder wrapper for the certified E2E driver.

Bare inner invocation is performed only by ``grm_adm1_3_dual_frame.py``,
which owns ``/tmp/forge-gpu.lock`` and pins repo-root ``PYTHONPATH``.  With
``GRM_ADM1_3_ENABLED`` false/unset this file installs no patch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from grm_adm1_analysis import ARMS, read_json, sha256_file  # noqa: E402
from grm_adm1_gpu import (  # noqa: E402
    FRAME_CONFIG,
    LiveError,
    append_jsonl_once,
    arm_plans,
    build_probe_row,
    route_profile,
)
from grm_adm1_3_dual_frame import (  # noqa: E402
    anchor_matches,
    evaluate_driver_arm,
    fresh_anchor,
)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().casefold() in ("1", "true", "yes", "on")


def _known_values(script: Sequence[Mapping[str, Any]]) -> list[str]:
    out = []
    for event in script:
        for key in ("value", "expected", "old_value"):
            value = event.get(key)
            if value is not None and str(value).casefold() not in out:
                out.append(str(value).casefold())
    return out


def _required_nodes(arena, ranking: Sequence[int], expected: str) -> list[int]:
    needle = str(expected).casefold()
    return [
        int(index) for index in ranking
        if needle in str(arena.grafts[int(index)].get("text", "") or "").casefold()
    ]


def _metric_probe_id(frame: str, turn: int, fact_id: str) -> str:
    fixture = str(FRAME_CONFIG[frame]["metric_fixture"]).lower()
    return f"{fixture}:turn-{turn}:{fact_id.replace(' ', '-')}"


def make_probe_wrapper(
    e2e,
    original: Callable[..., tuple[str, dict[str, Any]]],
    *,
    frame: str,
    rule_path: Path,
    rows_path: Path,
    anchors_path: Path,
) -> Callable[..., tuple[str, dict[str, Any]]]:
    if frame not in FRAME_CONFIG:
        raise LiveError(f"unknown ADM1.3 E2E frame: {frame}")
    rule = read_json(rule_path)
    rule_sha = sha256_file(rule_path)
    script = e2e.build_full_script()
    known_values = _known_values(script)
    config = FRAME_CONFIG[frame]

    def wrapped(
        repo,
        user_text: str,
        *,
        topk: int,
        ngen: int,
        defer_memory: bool = False,
        turn_idx: int | None = None,
        probe_ladder: bool = True,
        max_trips: int = 1,
    ) -> tuple[str, dict[str, Any]]:
        del topk
        if turn_idx is None:
            raise LiveError("ADM1.3 E2E wrapper requires turn_idx")
        if probe_ladder is not True:
            raise LiveError("ADM1.3 F-PROD requires probe ladder ON")
        turn = int(turn_idx)
        event = script[turn]
        if event.get("kind") != "probe" or event.get("user") != user_text:
            raise LiveError(f"turn {turn} is not the registered probe")
        arena = repo.arena
        if not bool(arena.revision_resolution):
            raise LiveError("ADM1.3 F-PROD requires L2 ON")
        started_ns = time.time_ns()
        live_idx = {int(graft) for graft, _tokens in arena.live_segs if graft is not None}
        profile = route_profile(arena, user_text, exclude=live_idx)
        if not profile["ranking"]:
            raise LiveError(f"empty route ranking at {frame} turn {turn}")
        required = _required_nodes(arena, profile["ranking"], event["expected"])
        plans = arm_plans(profile, rule)
        competing = [
            value for value in known_values
            if value != str(event["expected"]).casefold()
        ]

        anchor_result = evaluate_driver_arm(
            repo,
            e2e,
            original,
            arm="A-k3",
            profile=profile,
            plans=plans,
            question=user_text,
            ngen=int(ngen),
            max_trips=int(max_trips),
            turn_idx=turn,
            accepted_values=[event["expected"]],
            competing_values=competing,
            required_nodes=required,
        )
        anchor_id = f"{frame}:turn-{turn}:{event['fact_id'].replace(' ', '-')}"
        prod_config = {
            "probe_ladder": True,
            "ladder_implementation": "grm_e2e_session.probe_multimount_chat",
            "max_trips": int(max_trips),
            "revision_resolution": True,
            "turn_pipeline": "three_pass" if frame == "p4" else "single",
        }
        anchor = fresh_anchor(
            anchor_id=anchor_id,
            fixture=(
                str(config["metric_fixture"])
                if frame == "diag" else str(config["anchor_fixture"])
            ),
            class_label=str(config["metric_class"]) if frame == "diag" else "G0-ONLY",
            split="eval" if frame == "diag" else "anchor",
            question=user_text,
            accepted_values=[event["expected"]],
            required_nodes=required,
            profile=profile,
            result=anchor_result,
            rule_path=rule_path,
            config=prod_config,
            extra={
                "session_frame": frame,
                "turn": turn,
                "fact_id": event["fact_id"],
                "expected": event["expected"],
                "source_turn": int(event["source_turn"]),
            },
        )
        # Registration boundary precedes every independent arm attempt.
        append_jsonl_once(anchors_path, anchor, key="anchor_id")

        results = {
            arm: evaluate_driver_arm(
                repo,
                e2e,
                original,
                arm=arm,
                profile=profile,
                plans=plans,
                question=user_text,
                ngen=int(ngen),
                max_trips=int(max_trips),
                turn_idx=turn,
                accepted_values=[event["expected"]],
                competing_values=competing,
                required_nodes=required,
            )
            for arm in ARMS
        }
        arm_match = anchor_matches(anchor, results["A-k3"])

        # Advance the certified session only after all counterfactuals restore.
        answer, info = original(
            repo,
            user_text,
            topk=3,
            ngen=int(ngen),
            defer_memory=bool(defer_memory),
            turn_idx=turn,
            probe_ladder=True,
            max_trips=int(max_trips),
        )
        actual = [int(value) for value in (info.get("mount_fitted") or arena.cur_mounts)]
        canonical = {
            **results["A-k3"],
            "answer": str(answer),
            "mounted_nodes": actual,
            "answer_correct": str(event["expected"]).casefold() in str(answer).casefold(),
            "recall": int(bool(set(actual) & set(required))),
        }
        canonical["miss"] = int(not canonical["recall"])
        canonical["wrong_read"] = int(
            canonical["recall"]
            and not canonical["answer_correct"]
            and any(value in str(answer).casefold() for value in competing)
        )
        canonical_match = anchor_matches(anchor, canonical)

        if frame == "diag":
            base_id = f"diag:turn-{turn}:{event['fact_id'].replace(' ', '-')}"
            base_fixture = str(config["metric_fixture"])
            base_class = str(config["metric_class"])
            base_split = "eval"
        else:
            base_id = f"{frame}-anchor:turn-{turn}:{event['fact_id'].replace(' ', '-')}"
            base_fixture = str(config["anchor_fixture"])
            base_class = "G0-ONLY"
            base_split = "anchor"
        base = build_probe_row(
            probe_id=base_id,
            fixture=base_fixture,
            class_label=base_class,
            split=base_split,
            question=user_text,
            profile=profile,
            plans=plans,
            results=results,
            required_nodes=required,
            accepted_values=[event["expected"]],
            rule_path=rule_path,
            started_ns=started_ns,
            extra={
                "frame": "F-PROD",
                "session_frame": frame,
                "turn": turn,
                "fact_id": event["fact_id"],
                "expected": event["expected"],
                "source_turn": int(event["source_turn"]),
                "fresh_anchor_id": anchor_id,
                "fresh_anchor_registered_before_arms": True,
                "a_k3_fresh_anchor_match": bool(arm_match),
                "canonical_a_k3_fresh_anchor_match": bool(canonical_match),
                "canonical_a_k3_answer": str(answer),
                "canonical_a_k3_mounts": actual,
                "rule_payload_sha256": rule.get("rule_payload_sha256"),
                "wrapper_rule_sha256": rule_sha,
                "production_config": prod_config,
            },
        )
        append_jsonl_once(rows_path, base)
        if frame != "diag" and turn in set(config["metric_turns"]):
            metric = json.loads(json.dumps(base))
            metric.update({
                "probe_id": _metric_probe_id(frame, turn, event["fact_id"]),
                "fixture": str(config["metric_fixture"]),
                "class_label": str(config["metric_class"]),
                "split": "eval",
            })
            append_jsonl_once(rows_path, metric)
        print(json.dumps({
            "adm1_3_frame": "F-PROD",
            "session_frame": frame,
            "turn": turn,
            "answers_correct": {arm: results[arm]["answer_correct"] for arm in ARMS},
            "recall": {arm: results[arm]["recall"] for arm in ARMS},
            "fresh_anchor_match": bool(arm_match and canonical_match),
            "branches": {arm: plans[arm]["policy_branch"] for arm in ARMS},
        }, sort_keys=True), flush=True)
        return answer, info

    return wrapped


def install_patch(
    e2e,
    *,
    enabled: bool,
    frame: str = "diag",
    rule_path: Path | None = None,
    rows_path: Path | None = None,
    anchors_path: Path | None = None,
) -> bool:
    if not enabled:
        return False
    if rule_path is None or rows_path is None or anchors_path is None:
        raise LiveError("enabled ADM1.3 wrapper requires rule, row, and anchor paths")
    original = e2e.probe_multimount_chat
    e2e.probe_multimount_chat = make_probe_wrapper(
        e2e,
        original,
        frame=frame,
        rule_path=rule_path,
        rows_path=rows_path,
        anchors_path=anchors_path,
    )
    return True


def cpu_selftest() -> dict[str, Any]:
    original = object()
    fake = SimpleNamespace(probe_multimount_chat=original)
    installed = install_patch(fake, enabled=False)
    if installed or fake.probe_multimount_chat is not original:
        raise LiveError("flags-off ADM1.3 wrapper changed the driver callable")
    return {
        "schema": "grm.adm1_3.prod_e2e_selftest.v1",
        "status": "PASS",
        "checks": {"flags_off_no_patch_identity": "PASS"},
    }


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values == ["--selftest"]:
        print(json.dumps(cpu_selftest(), sort_keys=True))
        return 0
    from scripts import grm_e2e_session as e2e

    enabled = _truthy(os.environ.get("GRM_ADM1_3_ENABLED"))
    if enabled:
        run_dir_raw = os.environ.get("GRM_ADM1_3_RUN_DIR")
        frame = os.environ.get("GRM_ADM1_3_FRAME", "")
        rule_raw = os.environ.get("GRM_ADM1_3_RULE")
        if not run_dir_raw or not rule_raw:
            raise LiveError("GRM_ADM1_3_RUN_DIR and GRM_ADM1_3_RULE are required")
        run_dir = Path(run_dir_raw).resolve()
        rule_path = Path(rule_raw).resolve()
        frame_dir = run_dir / frame
        install_patch(
            e2e,
            enabled=True,
            frame=frame,
            rule_path=rule_path,
            rows_path=frame_dir / "adm_rows.jsonl",
            anchors_path=frame_dir / "fresh_anchors.jsonl",
        )
        e2e.__file__ = str(Path(__file__).resolve())
    return int(e2e.main(values))


if __name__ == "__main__":
    raise SystemExit(main())
