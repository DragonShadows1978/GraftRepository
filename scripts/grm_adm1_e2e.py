#!/usr/bin/env python3
"""GRM-ADM1 multi-arm wrapper around the certified 34-turn E2E driver.

The wrapper is selected only by the emitted ADM GPU runner.  It patches the
driver-level probe admission call in-process, evaluates all four admissions
from one identical pre-probe cache snapshot, restores that snapshot after each
comparison, and finally calls the unmodified legacy A-k3 path once so the
certified session continues normally.  On checkpoint restart the E2E driver's
re-exec target is this wrapper, so the same isolated patch is restored.

With ``GRM_ADM1_ENABLED`` false/unset, no function is replaced.
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

from grm_adm1_analysis import (  # noqa: E402
    ARMS,
    ARTIFACT_DIR,
    read_json,
    sha256_file,
    write_content_addressed,
)
from grm_adm1_gpu import (  # noqa: E402
    FRAME_CONFIG,
    LiveError,
    append_jsonl_once,
    arm_plans,
    build_probe_row,
    evaluate_attempt,
    fit_plan,
    route_profile,
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
) -> Callable[..., tuple[str, dict[str, Any]]]:
    if frame not in FRAME_CONFIG:
        raise LiveError(f"unknown ADM E2E frame: {frame}")
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
        if turn_idx is None:
            raise LiveError("ADM E2E probe wrapper requires turn_idx")
        turn = int(turn_idx)
        event = script[turn]
        if event.get("kind") != "probe" or event.get("user") != user_text:
            raise LiveError(f"turn {turn} is not the registered probe")
        arena = repo.arena
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
        results = {}
        for arm in ARMS:
            fitted = fit_plan(arena, plans[arm]["rank_plan"])
            results[arm] = evaluate_attempt(
                arena,
                question=user_text,
                picks=fitted,
                ngen=int(ngen),
                accepted_values=[event["expected"]],
                competing_values=competing,
                required_nodes=required,
            )

        # The comparison attempts have all restored the same pre-probe state.
        # Continue the certified fixture through the original fixed top-3 path.
        answer, info = original(
            repo,
            user_text,
            topk=3,
            ngen=int(ngen),
            defer_memory=bool(defer_memory),
            turn_idx=turn,
            probe_ladder=False,
            max_trips=int(max_trips),
        )
        actual = [int(value) for value in (info.get("mount_fitted") or arena.cur_mounts)]
        replay_equal = bool(
            str(answer) == results["A-k3"]["answer"]
            and actual == results["A-k3"]["mounted_nodes"]
        )
        if not replay_equal:
            raise LiveError(
                f"{frame} turn {turn}: diagnostic A-k3 differs from canonical replay")

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
                "frame": frame,
                "turn": turn,
                "fact_id": event["fact_id"],
                "expected": event["expected"],
                "source_turn": int(event["source_turn"]),
                "canonical_a_k3_replay_equal": replay_equal,
                "canonical_a_k3_answer": str(answer),
                "canonical_a_k3_mounts": actual,
                "rule_payload_sha256": rule.get("rule_payload_sha256"),
                "wrapper_rule_sha256": rule_sha,
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
        print(
            json.dumps({
                "adm1_frame": frame,
                "turn": turn,
                "identifier_hits": profile["identifier_hit_count"],
                "margin": profile["route_margin_1_2"],
                "branches": {
                    arm: plans[arm]["policy_branch"] for arm in ARMS
                },
                "answers_correct": {
                    arm: results[arm]["answer_correct"] for arm in ARMS
                },
                "recall": {arm: results[arm]["recall"] for arm in ARMS},
                "canonical_replay_equal": replay_equal,
            }, sort_keys=True),
            flush=True,
        )
        return answer, info

    return wrapped


def install_patch(e2e, *, enabled: bool, frame: str = "diag", rule_path: Path | None = None,
                  rows_path: Path | None = None) -> bool:
    """Return False and leave the driver untouched when the flag is off."""
    if not enabled:
        return False
    if rule_path is None or rows_path is None:
        raise LiveError("enabled ADM wrapper requires rule and row paths")
    original = e2e.probe_multimount_chat
    e2e.probe_multimount_chat = make_probe_wrapper(
        e2e,
        original,
        frame=frame,
        rule_path=rule_path,
        rows_path=rows_path,
    )
    return True


def cpu_selftest() -> dict[str, Any]:
    original = object()
    fake = SimpleNamespace(probe_multimount_chat=original)
    installed = install_patch(fake, enabled=False)
    if installed or fake.probe_multimount_chat is not original:
        raise LiveError("flags-off wrapper changed the driver callable")
    return {
        "schema": "grm.adm1.e2e_wrapper_cpu_selftest.v1",
        "status": "PASS",
        "checks": {
            "flags_off_no_patch_identity": "PASS",
            "frame_names": sorted(FRAME_CONFIG),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values == ["--selftest"]:
        value = cpu_selftest()
        path = write_content_addressed(
            ARTIFACT_DIR, "e2e_wrapper_cpu_selftest", value)
        print(json.dumps(value, sort_keys=True))
        print(f"receipt={path}")
        return 0

    from scripts import grm_e2e_session as e2e

    enabled = _truthy(os.environ.get("GRM_ADM1_ENABLED"))
    if enabled:
        run_dir_raw = os.environ.get("GRM_ADM1_RUN_DIR")
        frame = os.environ.get("GRM_ADM1_FRAME", "")
        rule_raw = os.environ.get("GRM_ADM1_RULE")
        if not run_dir_raw or not rule_raw:
            raise LiveError("GRM_ADM1_RUN_DIR and GRM_ADM1_RULE are required")
        run_dir = Path(run_dir_raw).resolve()
        rule_path = Path(rule_raw).resolve()
        rows_path = run_dir / frame / "adm_rows.jsonl"
        install_patch(
            e2e,
            enabled=True,
            frame=frame,
            rule_path=rule_path,
            rows_path=rows_path,
        )
        # maybe_restart() uses its module-global __file__ for os.execvpe.
        # Point it back to this wrapper so the isolated patch survives restart.
        e2e.__file__ = str(Path(__file__).resolve())
    return int(e2e.main(values))


if __name__ == "__main__":
    raise SystemExit(main())
