#!/usr/bin/env python3
"""GRM-ADM1.3 production-frame runner and dual-frame adjudicator.

Bare repo-root invocation (the runner self-leases every GPU stage):

    PYTHONPATH=/mnt/ForgeRealm/GraftRepository \
      python3 scripts/grm_adm1_3_dual_frame.py --stage all

The production frame pins the probe ladder and L2 ON.  It reuses the frozen
ADM1 decisiveness rule, writes each fresh A-k3 expectation before evaluating
the four arms, and never mutates the captured ADM1 isolation run.  CORPUS-100
uses its registered production ``Arena.step(max_trips=2)`` path; fresh
supersession controls use ``Arena.step(max_trips=0)``; DIAG/E2E/P4 use the
production driver ladder with one retry.

This file and its companion wrapper are experiment-only.  They do not wire
the admission policy into production.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from grm_adm1_analysis import (  # noqa: E402
    ADMError,
    ARMS,
    BOOTSTRAP_REPLICATES,
    CLASSES,
    ARTIFACT_DIR,
    adjudicate_predictions,
    canonical_json_bytes,
    corpus100_fixture_rows,
    file_record,
    metric_table,
    production_inventory,
    read_json,
    read_jsonl,
    sha256_file,
    supersession_fixture_rows,
    write_content_addressed,
    write_exclusive_json,
)
from grm_adm1_gpu import (  # noqa: E402
    DEFAULT_LEASE_SECONDS,
    DEFAULT_LOCK_WAIT_SECONDS,
    FRAME_CONFIG,
    GAP_SECONDS,
    MAX_LEASE_SECONDS,
    _corpus_bank,
    _load_minicpm,
    append_jsonl_once,
    arm_plans,
    build_probe_row,
    fit_plan,
    gpu_lease,
    restore_arena,
    route_profile,
    snapshot_arena,
)
from grm_probe_ladder import build_probe_ladder_attempts  # noqa: E402


ORDER_PATH = ROOT / "orders" / "GRM_ADM1_3_DUAL_FRAME.md"
ISO_RUN_DEFAULT = (
    ARTIFACT_DIR / "run_20260830T093538Z_3defaa"
)
RULE_DEFAULT = (
    ISO_RUN_DEFAULT / "decisiveness_rule_c304609f81475bd2.json"
)
FROZEN_RULE_SHA256 = (
    "c304609f81475bd2ae3399ad180d3bb00cc5d2b44b1a49db570387c810defb91"
)
PROD_STAGES = ("lookup", "diag", "e2e", "p4")
E2E_WRAPPER = ROOT / "scripts" / "grm_adm1_3_prod_e2e.py"
ISO_ROW_PATHS = (
    ISO_RUN_DEFAULT / "minicpm_rows.jsonl",
    ISO_RUN_DEFAULT / "diag" / "adm_rows.jsonl",
    ISO_RUN_DEFAULT / "e2e" / "adm_rows.jsonl",
    ISO_RUN_DEFAULT / "p4" / "adm_rows.jsonl",
)
ISO_EXPECTED = {
    "minicpm_rows.jsonl": (22, "5da8d42c8314ed4a6c3af38f3456d1b179252c8e55bf96fff1f51aca765069c3"),
    "diag/adm_rows.jsonl": (9, "8585bba8fc3732d7c1220703d64e3091d30030e7cfbb9ee2e0a943b27d72d5cf"),
    "e2e/adm_rows.jsonl": (11, "aca9250e2928660a7bb9868ac072373e3f423f81c28f29fdf1fa5c6278f303e7"),
    "p4/adm_rows.jsonl": (11, "dcbc71b6c8790116309809a4a5d56a12f881855a76433c0128b512856ad90177"),
}


class DualFrameError(ADMError):
    pass


def _contains(answer: str, values: Sequence[str]) -> bool:
    folded = str(answer).casefold()
    return any(str(value).casefold() in folded for value in values)


def _json_info(info: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep only compact, deterministic ladder/readout receipt fields."""
    info = dict(info or {})
    keys = (
        "trip",
        "clean_room",
        "no_mount_fit",
        "driver_probe_multimount",
        "driver_probe_ladder",
        "driver_topk",
        "point_lookup",
        "precise_first",
        "mount_plan",
        "mount_fitted",
        "mount_dropped_for_width",
        "ranking_ids",
        "ungrounded_kept_first",
        "evicted",
        "resident",
        "live_tokens",
        "mounts",
    )
    return {key: info[key] for key in keys if key in info}


def production_frame_inventory() -> dict[str, dict[str, Any]]:
    """Hash the current production frame, including the L2 default helper."""
    inventory = dict(production_inventory())
    extra = ROOT / "core" / "grm_supersession.py"
    if not extra.is_file():
        raise DualFrameError(f"current production L2 helper is absent: {extra}")
    inventory[str(extra.relative_to(ROOT))] = file_record(extra)
    return inventory


def assert_run_inventory(run_dir: Path) -> dict[str, Any]:
    """Fail closed if production or experiment sources drift between stages."""
    manifest_path = run_dir.resolve() / "run_manifest.json"
    manifest = read_json(manifest_path)
    expected_production = manifest["production_inventory_before"]
    current_production = production_frame_inventory()
    source_checks = {}
    for relative, expected in manifest["source_provenance"].items():
        path = ROOT / relative
        current = file_record(path) if path.is_file() else None
        source_checks[relative] = {
            "expected": expected,
            "current": current,
            "match": current == expected,
        }
    value = {
        "production_match": current_production == expected_production,
        "source_match": all(row["match"] for row in source_checks.values()),
        "source_checks": source_checks,
    }
    if not value["production_match"] or not value["source_match"]:
        raise DualFrameError(
            "ADM1.3 run inventory drifted between stage boundaries; refuse "
            "a mixed F-PROD capture"
        )
    return value


def validate_frozen_rule(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise DualFrameError(f"frozen rule does not exist: {path}")
    digest = sha256_file(path)
    if digest != FROZEN_RULE_SHA256:
        raise DualFrameError(
            "ADM1.3 requires the unchanged ADM1 frozen rule; "
            f"expected {FROZEN_RULE_SHA256}, observed {digest}"
        )
    value = read_json(path)
    if value.get("schema") != "grm.adm1.decisiveness_rule.v1":
        raise DualFrameError("unexpected decisiveness-rule schema")
    return value


def _snapshot_counterfactual(arena) -> dict[str, Any]:
    attrs = {}
    for name in (
        "_s4_turn",
        "_deferred_route_key_token",
        "_deferred_route_keys",
    ):
        if hasattr(arena, name):
            attrs[name] = (True, copy.deepcopy(getattr(arena, name)))
        else:
            attrs[name] = (False, None)
    return {
        "arena": snapshot_arena(arena),
        "topk": int(arena.topk),
        "metadata": [copy.deepcopy(graft.get("metadata")) for graft in arena.grafts],
        "metadata_present": ["metadata" in graft for graft in arena.grafts],
        "attrs": attrs,
    }


def _restore_counterfactual(arena, snapshot: Mapping[str, Any]) -> None:
    restore_arena(arena, snapshot["arena"])
    arena.topk = int(snapshot["topk"])
    for graft, present, metadata in zip(
        arena.grafts,
        snapshot["metadata_present"],
        snapshot["metadata"],
    ):
        if present:
            graft["metadata"] = copy.deepcopy(metadata)
        else:
            graft.pop("metadata", None)
    for name, (present, value) in snapshot["attrs"].items():
        if present:
            setattr(arena, name, copy.deepcopy(value))
        elif hasattr(arena, name):
            delattr(arena, name)


def _result(
    *,
    arena,
    answer: str,
    info: Mapping[str, Any] | None,
    accepted_values: Sequence[str],
    competing_values: Sequence[str],
    required_nodes: Sequence[int],
    initial_plan: Sequence[int],
    branch: str,
) -> dict[str, Any]:
    actual = [int(value) for value in arena.cur_mounts]
    recall = bool(set(actual) & {int(value) for value in required_nodes})
    correct = _contains(answer, accepted_values)
    competing = _contains(answer, competing_values)
    compact_info = _json_info(info)
    return {
        "rank_plan_fitted": [int(value) for value in actual],
        "mounted_nodes": actual,
        "answer": str(answer),
        "answer_correct": bool(correct),
        "recall": int(recall),
        "wrong_read": int(recall and not correct and competing),
        "miss": int(not recall),
        "mount_count": len(actual),
        "arena_info": compact_info,
        "ladder_initial_policy_plan": [int(value) for value in initial_plan],
        "ladder_policy_branch": str(branch),
    }


def _fixed_topk(arm: str, plan: Sequence[int], ranking: Sequence[int]) -> int:
    fixed = {"A-k1": 1, "A-k2": 2, "A-k3": 3}
    if arm in fixed:
        return fixed[arm]
    normalized = [int(value) for value in plan]
    ranked = [int(value) for value in ranking]
    for topk in (1, 2, 3):
        if normalized == ranked[:topk]:
            return topk
    raise DualFrameError(
        "Arena.step production evaluator cannot encode a non-prefix A-DEC "
        f"plan: plan={normalized}, ranking={ranked[:6]}"
    )


def evaluate_core_arm(
    arena,
    *,
    arm: str,
    profile: Mapping[str, Any],
    plans: Mapping[str, Mapping[str, Any]],
    question: str,
    ngen: int,
    max_trips: int,
    accepted_values: Sequence[str],
    competing_values: Sequence[str],
    required_nodes: Sequence[int],
    restore_after: bool = True,
) -> dict[str, Any]:
    """Evaluate one arm through the fixture's real ``Arena.step`` ladder."""
    if not bool(arena.revision_resolution):
        raise DualFrameError("F-PROD core attempt is not explicitly L2-on")
    plan = [int(value) for value in plans[arm]["rank_plan"]]
    branch = str(plans[arm]["policy_branch"])
    topk = _fixed_topk(arm, plan, profile["ranking"])
    before = _snapshot_counterfactual(arena)
    arena.topk = int(topk)
    try:
        answer, info = arena.step(
            question,
            ngen=int(ngen),
            deposit=False,
            max_trips=int(max_trips),
        )
        result = _result(
            arena=arena,
            answer=answer,
            info=info,
            accepted_values=accepted_values,
            competing_values=competing_values,
            required_nodes=required_nodes,
            initial_plan=plan,
            branch=branch,
        )
        result["production_ladder"] = {
            "implementation": "ArenaCache.step",
            "topk": int(topk),
            "max_trips": int(max_trips),
            "revision_resolution": True,
        }
        return result
    finally:
        if restore_after:
            _restore_counterfactual(arena, before)
        else:
            arena.topk = int(before["topk"])


def build_dec_driver_attempts(
    *,
    ranking: list[int],
    topk: int,
    precise: list[int] | None,
    point_lookup: bool,
    max_trips: int,
    policy_plan: Sequence[int],
    baseline_builder: Callable[..., list[tuple[list[int], bool]]] = build_probe_ladder_attempts,
) -> list[tuple[list[int], bool]]:
    """Put the frozen A-DEC plan at trip 0, retaining production retry law."""
    del topk
    clean = bool(point_lookup)
    first = ([int(value) for value in policy_plan], clean)
    baseline = baseline_builder(
        ranking=[int(value) for value in ranking],
        topk=3,
        precise=None if precise is None else [int(value) for value in precise],
        point_lookup=bool(point_lookup),
        max_trips=int(max_trips),
    )
    attempts = [first]
    # Replacing production trip 0 must retain production's registered retry
    # (the tail), not retry the baseline primary and accidentally discard its
    # wider/next-slice attempt under the max_trips cap.
    retry_order = [*baseline[1:], *baseline[:1]]
    for planned, is_clean in retry_order:
        candidate = ([int(value) for value in planned], bool(is_clean))
        if candidate not in attempts:
            attempts.append(candidate)
    return attempts[:max(1, int(max_trips) + 1)]


@contextmanager
def _dec_driver_planner(e2e, policy_plan: Sequence[int]) -> Iterator[None]:
    original = e2e.build_probe_ladder_attempts

    def override(**kwargs):
        return build_dec_driver_attempts(
            **kwargs,
            policy_plan=policy_plan,
            baseline_builder=original,
        )

    e2e.build_probe_ladder_attempts = override
    try:
        yield
    finally:
        e2e.build_probe_ladder_attempts = original


def evaluate_driver_arm(
    repo,
    e2e,
    original,
    *,
    arm: str,
    profile: Mapping[str, Any],
    plans: Mapping[str, Mapping[str, Any]],
    question: str,
    ngen: int,
    max_trips: int,
    turn_idx: int | None,
    accepted_values: Sequence[str],
    competing_values: Sequence[str],
    required_nodes: Sequence[int],
) -> dict[str, Any]:
    """Evaluate one counterfactual via the unmodified production driver ladder."""
    arena = repo.arena
    if not bool(arena.revision_resolution):
        raise DualFrameError("F-PROD driver attempt is not explicitly L2-on")
    plan = [int(value) for value in plans[arm]["rank_plan"]]
    branch = str(plans[arm]["policy_branch"])
    if arm == "A-DEC" and branch != "declared_synthesis_identified_set":
        topk = _fixed_topk(arm, plan, profile["ranking"])
    else:
        topk = {"A-k1": 1, "A-k2": 2, "A-k3": 3, "A-DEC": 3}[arm]
    before = _snapshot_counterfactual(arena)
    declared_override = bool(
        arm == "A-DEC" and branch == "declared_synthesis_identified_set"
    )
    planner = (
        _dec_driver_planner(e2e, plan) if declared_override else nullcontext()
    )
    try:
        with planner:
            answer, info = original(
                repo,
                question,
                topk=int(topk),
                ngen=int(ngen),
                defer_memory=True,
                turn_idx=turn_idx,
                probe_ladder=True,
                max_trips=int(max_trips),
            )
        if not bool((info or {}).get("driver_probe_ladder")):
            raise DualFrameError(f"{arm} escaped the production probe ladder")
        result = _result(
            arena=arena,
            answer=answer,
            info=info,
            accepted_values=accepted_values,
            competing_values=competing_values,
            required_nodes=required_nodes,
            initial_plan=plan,
            branch=branch,
        )
        result["production_ladder"] = {
            "implementation": "grm_e2e_session.probe_multimount_chat",
            "topk": int(topk),
            "max_trips": int(max_trips),
            "probe_ladder": True,
            "revision_resolution": True,
            "a_dec_declared_synthesis_trip0_override_only": declared_override,
        }
        return result
    finally:
        _restore_counterfactual(arena, before)


def fresh_anchor(
    *,
    anchor_id: str,
    fixture: str,
    class_label: str,
    split: str,
    question: str,
    accepted_values: Sequence[str],
    required_nodes: Sequence[int],
    profile: Mapping[str, Any],
    result: Mapping[str, Any],
    rule_path: Path,
    config: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = {
        "schema": "grm.adm1_3.fresh_a_k3_anchor.v1",
        "frame": "F-PROD",
        "anchor_id": str(anchor_id),
        "fixture": str(fixture),
        "class_label": str(class_label),
        "split": str(split),
        "question": str(question),
        "accepted_values": list(accepted_values),
        "required_node_ids": [int(value) for value in required_nodes],
        "router_rank_order": [int(value) for value in profile["ranking"]],
        "identifier_hit_count": int(profile["identifier_hit_count"]),
        "rule": file_record(rule_path),
        "production_config": dict(config),
        "generated_unix_ns": time.time_ns(),
        "a_k3_expectation": dict(result),
    }
    if extra:
        value.update(dict(extra))
    return value


def anchor_matches(anchor: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    expected = anchor["a_k3_expectation"]
    return bool(
        str(result["answer"]) == str(expected["answer"])
        and list(result["mounted_nodes"]) == list(expected["mounted_nodes"])
        and bool(result["answer_correct"]) == bool(expected["answer_correct"])
        and int(result["recall"]) == int(expected["recall"])
        and int(result["wrong_read"]) == int(expected["wrong_read"])
    )


def _thin_repo(arena):
    return SimpleNamespace(arena=arena, _snapshot_state=lambda: [])


def _append_core_probe(
    *,
    arena,
    rule: Mapping[str, Any],
    rule_path: Path,
    rows_path: Path,
    anchors_path: Path,
    probe: Mapping[str, Any],
    target: int,
    competing_values: Sequence[str],
    max_trips: int,
) -> None:
    started_ns = time.time_ns()
    profile = route_profile(arena, str(probe["question"]), exclude=set())
    plans = arm_plans(profile, rule)
    config = {
        "probe_ladder": True,
        "ladder_implementation": "ArenaCache.step",
        "max_trips": int(max_trips),
        "revision_resolution": True,
    }
    anchor_result = evaluate_core_arm(
        arena,
        arm="A-k3",
        profile=profile,
        plans=plans,
        question=str(probe["question"]),
        ngen=40 if probe["fixture"] == "CORPUS-100" else 32,
        max_trips=int(max_trips),
        accepted_values=probe["expected_values"],
        competing_values=competing_values,
        required_nodes=[target],
    )
    anchor_id = str(probe["probe_id"])
    anchor = fresh_anchor(
        anchor_id=anchor_id,
        fixture=str(probe["fixture"]),
        class_label=str(probe["class_label"]),
        split=str(probe["split"]),
        question=str(probe["question"]),
        accepted_values=probe["expected_values"],
        required_nodes=[target],
        profile=profile,
        result=anchor_result,
        rule_path=rule_path,
        config=config,
    )
    # Registration boundary: the same-frame expectation is closed before any
    # comparison arm begins.
    append_jsonl_once(anchors_path, anchor, key="anchor_id")

    results = {
        arm: evaluate_core_arm(
            arena,
            arm=arm,
            profile=profile,
            plans=plans,
            question=str(probe["question"]),
            ngen=40 if probe["fixture"] == "CORPUS-100" else 32,
            max_trips=int(max_trips),
            accepted_values=probe["expected_values"],
            competing_values=competing_values,
            required_nodes=[target],
        )
        for arm in ARMS
    }
    arm_match = anchor_matches(anchor, results["A-k3"])
    canonical = evaluate_core_arm(
        arena,
        arm="A-k3",
        profile=profile,
        plans=plans,
        question=str(probe["question"]),
        ngen=40 if probe["fixture"] == "CORPUS-100" else 32,
        max_trips=int(max_trips),
        accepted_values=probe["expected_values"],
        competing_values=competing_values,
        required_nodes=[target],
        restore_after=False,
    )
    canonical_match = anchor_matches(anchor, canonical)
    row = build_probe_row(
        probe_id=str(probe["probe_id"]),
        fixture=str(probe["fixture"]),
        class_label=str(probe["class_label"]),
        split=str(probe["split"]),
        question=str(probe["question"]),
        profile=profile,
        plans=plans,
        results=results,
        required_nodes=[target],
        accepted_values=probe["expected_values"],
        rule_path=rule_path,
        started_ns=started_ns,
        extra={
            "frame": "F-PROD",
            "fresh_anchor_id": anchor_id,
            "fresh_anchor_registered_before_arms": True,
            "a_k3_fresh_anchor_match": bool(arm_match),
            "canonical_a_k3_fresh_anchor_match": bool(canonical_match),
            "production_config": config,
        },
    )
    append_jsonl_once(rows_path, row)
    print(
        f"prod fixture={probe['fixture']} probe={probe['probe_id']} "
        f"anchor={anchor_result['answer_correct']} "
        f"k1={results['A-k1']['answer_correct']} "
        f"k2={results['A-k2']['answer_correct']} "
        f"k3={results['A-k3']['answer_correct']} "
        f"dec={results['A-DEC']['answer_correct']} "
        f"anchor_match={arm_match and canonical_match}",
        flush=True,
    )


def _supersession_arena(model, tokenizer):
    from core.graft_arena import ArenaCache

    fixture_path = (
        ROOT / "tests" / "fixtures" / "supersession_battery"
        / "fresh_fact_controls.json"
    )
    fixture = read_json(fixture_path)
    arena = ArenaCache(
        model,
        encode=lambda text: tokenizer.encode(text).ids,
        decode=lambda ids: tokenizer.decode(ids),
        sink_text="<conversation>\n",
        arena_width=256,
        route_layer=44,
        topk=3,
        live_turns=0,
        cache_deposits=False,
        length_debias=False,
        revision_resolution=True,
        # This harness supplies each A-k1/k2/k3/A-DEC plan itself.
        decisive_admission=False,
    )
    node_to_idx = {}
    values = {}
    for node in fixture["nodes"]:
        index = int(arena.deposit(node["text"]))
        node_to_idx[node["node_id"]] = index
        values[index] = str(node["value"]).casefold()
        arena.grafts[index]["adm_fixture_role"] = node["role"]
    registered = {
        row["probe_id"].split(":", 1)[1]: row
        for row in supersession_fixture_rows()
    }
    return arena, fixture, node_to_idx, values, registered


def _write_anchor_receipt(stage_dir: Path, stage: str) -> Path:
    anchors_path = stage_dir / "fresh_anchors.jsonl"
    anchors = read_jsonl(anchors_path)
    if not anchors:
        raise DualFrameError(f"no fresh anchors captured for {stage}")
    receipt = {
        "schema": "grm.adm1_3.fresh_anchor_receipt.v1",
        "frame": "F-PROD",
        "stage": stage,
        "created_unix_ns": time.time_ns(),
        "registration_order": "each anchor JSONL row was closed before its arm attempts",
        "anchor_count": len(anchors),
        "anchors": file_record(anchors_path),
        "all_ladder_on": all(
            bool(row["production_config"].get("probe_ladder")) for row in anchors
        ),
        "all_l2_on": all(
            bool(row["production_config"].get("revision_resolution")) for row in anchors
        ),
        "a_k3_answer_correct": {
            fixture: [
                sum(
                    bool(row["a_k3_expectation"]["answer_correct"])
                    for row in anchors if row["fixture"] == fixture
                ),
                sum(row["fixture"] == fixture for row in anchors),
            ]
            for fixture in sorted({row["fixture"] for row in anchors})
        },
    }
    return write_content_addressed(stage_dir, "fresh_anchor_receipt", receipt)


def run_lookup_stage(run_dir: Path, rule_path: Path) -> None:
    stage_dir = run_dir / "lookup"
    marker = stage_dir / "stage_complete.json"
    if marker.is_file():
        print("stage=lookup status=already_complete", flush=True)
        return
    if stage_dir.exists() and any(stage_dir.iterdir()):
        raise DualFrameError(
            "partial lookup stage cannot be state-faithfully resumed; use a fresh run dir"
        )
    stage_dir.mkdir(parents=True, exist_ok=True)
    boundary_before = assert_run_inventory(run_dir)
    rows_path = stage_dir / "adm_rows.jsonl"
    anchors_path = stage_dir / "fresh_anchors.jsonl"
    rule = validate_frozen_rule(rule_path)
    model, tokenizer, model_info, tc = _load_minicpm()
    print(f"loaded={model_info}", flush=True)
    try:
        arena, meta = _corpus_bank(model, tokenizer)
        for probe in corpus100_fixture_rows():
            target = int(probe["target_ordinal"])
            competing = [
                str(row["value"])
                for row in meta
                if row["family"] == probe["family"]
                and int(row["node_id"]) != target
            ]
            _append_core_probe(
                arena=arena,
                rule=rule,
                rule_path=rule_path,
                rows_path=rows_path,
                anchors_path=anchors_path,
                probe=probe,
                target=target,
                competing_values=competing,
                max_trips=2,
            )
        del arena
        gc.collect()
        if hasattr(tc, "empty_cache"):
            tc.empty_cache()

        arena, fixture, node_to_idx, values, registered = _supersession_arena(
            model, tokenizer
        )
        for probe in fixture["probes"]:
            reg = dict(registered[probe["probe_id"]])
            reg["fixture"] = "SUPERSESSION-FRESH-CONTROLS"
            target = int(node_to_idx[probe["target_node"]])
            competing = [value for index, value in values.items() if index != target]
            _append_core_probe(
                arena=arena,
                rule=rule,
                rule_path=rule_path,
                rows_path=rows_path,
                anchors_path=anchors_path,
                probe=reg,
                target=target,
                competing_values=competing,
                max_trips=0,
            )
        del arena
        boundary_after = assert_run_inventory(run_dir)
        receipt_path = _write_anchor_receipt(stage_dir, "lookup")
        write_exclusive_json(marker, {
            "schema": "grm.adm1_3.prod_stage_complete.v1",
            "frame": "F-PROD",
            "stage": "lookup",
            "completed_unix_ns": time.time_ns(),
            "row_count": len(read_jsonl(rows_path)),
            "anchor_count": len(read_jsonl(anchors_path)),
            "rows": file_record(rows_path),
            "anchors": file_record(anchors_path),
            "anchor_receipt": file_record(receipt_path),
            "rule": file_record(rule_path),
            "inventory_boundary_checks": {
                "before": boundary_before,
                "after": boundary_after,
            },
        })
    finally:
        del model
        gc.collect()
        if hasattr(tc, "empty_cache"):
            tc.empty_cache()


def _kill_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_e2e_stage(
    run_dir: Path,
    frame: str,
    rule_path: Path,
    *,
    lease_seconds: int,
) -> None:
    frame_dir = run_dir / frame
    marker = frame_dir / "stage_complete.json"
    if marker.is_file():
        print(f"stage={frame} status=already_complete", flush=True)
        return
    boundary_before = assert_run_inventory(run_dir)
    session_dir = frame_dir / "session"
    command = [
        sys.executable,
        str(E2E_WRAPPER),
        "--mode", "full",
        "--session-dir", str(session_dir),
        "--turn-pipeline", "three_pass" if frame == "p4" else "single",
        "--topk", "3",
        "--ngen", "32",
        "--max-trips", "1",
        "--live-turns", "2",
        "--restart-after", "17",
        "--probe-ladder",
        "--sup-resolve",
        "--no-adm-decisive",
        "--skip-gpu-idle-check",
    ]
    if session_dir.exists():
        # A fresh anchor is intentionally persisted before the arm attempts
        # and before the driver commits the turn.  Replaying a killed partial
        # turn would collide on its timestamped immutable row and, worse,
        # could pair a newer transcript with an older repository checkpoint.
        # The driver's registered turn-17 restart is an in-process re-exec and
        # does not return through this branch.
        raise DualFrameError(
            f"partial {frame} session cannot be state-faithfully resumed; "
            "use a fresh ADM1.3 run directory"
        )
    env = os.environ.copy()
    prior_path = env.get("PYTHONPATH", "")
    env.update({
        "GRM_ADM1_3_RUN_DIR": str(run_dir),
        "GRM_ADM1_3_FRAME": frame,
        "GRM_ADM1_3_RULE": str(rule_path),
        "GRM_ADM1_3_ENABLED": "1",
        "GRM_PROBE_LADDER": "1",
        "GRM_SUP_RESOLVE": "1",
        "GRM_ADM_DECISIVE": "0",
        "PYTHONPATH": str(ROOT) + (os.pathsep + prior_path if prior_path else ""),
        "PYTHONUNBUFFERED": "1",
    })
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        start_new_session=True,
    )
    try:
        returncode = process.wait(timeout=max(1, int(lease_seconds) - 8))
    except BaseException:
        _kill_process_group(process)
        raise
    if returncode not in (0, 2):
        raise DualFrameError(f"F-PROD {frame} replay failed with status {returncode}")

    rows_path = frame_dir / "adm_rows.jsonl"
    anchors_path = frame_dir / "fresh_anchors.jsonl"
    rows = read_jsonl(rows_path)
    anchors = read_jsonl(anchors_path)
    expected_rows = 9 if frame == "diag" else 11
    if len(rows) != expected_rows or len(anchors) != 9:
        raise DualFrameError(
            f"{frame}: expected rows/anchors {expected_rows}/9, "
            f"observed {len(rows)}/{len(anchors)}"
        )
    config = read_json(session_dir / "run_config.json")
    if config.get("probe_ladder") is not True or config.get("sup_resolve") is not True:
        raise DualFrameError(f"{frame}: session escaped production ladder/L2 pins")
    boundary_after = assert_run_inventory(run_dir)
    receipt_path = _write_anchor_receipt(frame_dir, frame)
    write_exclusive_json(marker, {
        "schema": "grm.adm1_3.prod_stage_complete.v1",
        "frame": "F-PROD",
        "stage": frame,
        "completed_unix_ns": time.time_ns(),
        "row_count": len(rows),
        "anchor_count": len(anchors),
        "rows": file_record(rows_path),
        "anchors": file_record(anchors_path),
        "anchor_receipt": file_record(receipt_path),
        "scorecard": file_record(session_dir / "probe_scorecard.json"),
        "run_config": file_record(session_dir / "run_config.json"),
        "replay_process_returncode": int(returncode),
        "rule": file_record(rule_path),
        "production_pins": {"probe_ladder": True, "sup_resolve": True},
        "inventory_boundary_checks": {
            "before": boundary_before,
            "after": boundary_after,
        },
    })


def _load_prod_rows(run_dir: Path) -> list[dict[str, Any]]:
    paths = [run_dir / "lookup" / "adm_rows.jsonl"]
    paths += [run_dir / frame / "adm_rows.jsonl" for frame in ("diag", "e2e", "p4")]
    rows = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def _load_iso_rows(iso_run: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    iso_run = iso_run.resolve()
    rows = []
    receipts = []
    for relative, (expected_count, expected_sha) in ISO_EXPECTED.items():
        path = iso_run / relative
        values = read_jsonl(path)
        observed_sha = sha256_file(path)
        match = len(values) == expected_count and observed_sha == expected_sha
        receipts.append({
            "path": str(path.relative_to(ROOT)),
            "expected_count": int(expected_count),
            "observed_count": len(values),
            "expected_sha256": expected_sha,
            "observed_sha256": observed_sha,
            "match": bool(match),
        })
        rows.extend(values)
    audit_paths = sorted(iso_run.glob("adm1_2_probe_adjudication_*.json"))
    audited = any(
        bool(
            ((read_json(path).get("frame_alignment") or {}).get("l2_projection") or {})
            .get("l2_projection_noop")
        )
        and len(
            ((read_json(path).get("frame_alignment") or {}).get("l2_projection") or {})
            .get("changed_candidate_sets") or ()
        ) == 0
        and int(
            ((read_json(path).get("frame_alignment") or {}).get("l2_projection") or {})
            .get("candidate_sets_examined", 0)
        ) == 63
        for path in audit_paths
    )
    return rows, {
        "schema": "grm.adm1_3.iso_reuse_validation.v1",
        "run_dir": str(iso_run),
        "recomputed": False,
        "files": receipts,
        "all_hashes_and_counts_match": all(row["match"] for row in receipts),
        "adm1_2_zero_changed_candidate_sets_audit_present": bool(audited),
    }


def _metric_cell(metric: Mapping[str, Any], *, percent: bool = True) -> str:
    if int(metric["n"]) == 0:
        return "N/A (n=0)"
    estimate = float(metric["estimate"])
    low, high = metric["ci95"]
    if percent:
        return f"{100 * estimate:.1f}% [{100 * low:.1f}, {100 * high:.1f}]"
    return f"{estimate:.3f} [{low:.3f}, {high:.3f}]"


def _estimate(table: Mapping[str, Any], cls: str, arm: str, metric: str) -> float | None:
    value = table[cls][arm][metric]["estimate"]
    return None if value is None else float(value)


def _answer_counts(rows: Sequence[Mapping[str, Any]], fixture: str, arm: str) -> list[int]:
    selected = [row for row in rows if row.get("fixture") == fixture]
    return [sum(bool(row["arms"][arm]["answer_correct"]) for row in selected), len(selected)]


def _fresh_g0(run_dir: Path, prod_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    anchors = []
    for stage in PROD_STAGES:
        anchors.extend(read_jsonl(run_dir / stage / "fresh_anchors.jsonl"))
    anchor_map = {str(row["anchor_id"]): row for row in anchors}
    if len(anchor_map) != len(anchors):
        raise DualFrameError("fresh production anchors contain duplicate IDs")
    comparisons = []
    for row in prod_rows:
        anchor_id = str(row["fresh_anchor_id"])
        anchor = anchor_map[anchor_id]
        comparisons.append({
            "probe_id": str(row["probe_id"]),
            "anchor_id": anchor_id,
            "arm_a_k3_matches": bool(anchor_matches(anchor, row["arms"]["A-k3"])),
            "recorded_arm_match": bool(row.get("a_k3_fresh_anchor_match")),
            "canonical_match": bool(row.get("canonical_a_k3_fresh_anchor_match")),
        })
    fixture_specs = (
        ("CORPUS-100", lambda row: row["fixture"] == "CORPUS-100"),
        ("SUPERSESSION-FRESH-CONTROLS", lambda row: row["fixture"] == "SUPERSESSION-FRESH-CONTROLS"),
        ("DIAG-CURRENT-FULL", lambda row: row["fixture"] == "DIAG-CURRENT-FULL"),
        ("E2E-34-FULL-A-k3-ANCHOR", lambda row: row["fixture"] == "E2E-34-FULL-A-k3-ANCHOR"),
        ("E2E-34-CROSS-FACT", lambda row: row["fixture"] == "E2E-34-CROSS-FACT"),
        ("P4-REPLICATION-FULL-A-k3-ANCHOR", lambda row: row["fixture"] == "P4-REPLICATION-FULL-A-k3-ANCHOR"),
        ("P4-REPLICATION-CROSS-FACT", lambda row: row["fixture"] == "P4-REPLICATION-CROSS-FACT"),
    )
    fixtures = {}
    for name, predicate in fixture_specs:
        selected = [row for row in prod_rows if predicate(row)]
        expected = [
            sum(
                bool(anchor_map[str(row["fresh_anchor_id"])]["a_k3_expectation"]["answer_correct"])
                for row in selected
            ),
            len(selected),
        ]
        observed = [
            sum(bool(row["arms"]["A-k3"]["answer_correct"]) for row in selected),
            len(selected),
        ]
        fixtures[name] = {
            "fresh_expected_a_k3_answer_correct": expected,
            "observed_arm_a_k3_answer_correct": observed,
            "pass": expected == observed and all(
                item["arm_a_k3_matches"]
                and item["recorded_arm_match"]
                and item["canonical_match"]
                for item in comparisons
                if item["probe_id"] in {str(row["probe_id"]) for row in selected}
            ),
        }
    all_config = all(
        row["production_config"].get("probe_ladder") is True
        and row["production_config"].get("revision_resolution") is True
        for row in anchors
    )
    return {
        "schema": "grm.adm1_3.fresh_g0.v1",
        "frame": "F-PROD",
        "anchor_count": len(anchors),
        "comparison_count": len(comparisons),
        "all_anchor_configs_ladder_on_l2_on": bool(all_config),
        "fixtures": fixtures,
        "comparisons": comparisons,
        "status": "PASS" if all_config and all(v["pass"] for v in fixtures.values()) else "RED",
    }


def adoption_rule(
    metrics: Mapping[str, Any], prod_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    measured = [
        cls for cls in CLASSES
        if int(metrics[cls]["A-k3"]["recall"]["n"]) > 0
    ]
    recall_checks = []
    wrong_checks = []
    failures = []
    for cls in measured:
        dec_recall = _estimate(metrics, cls, "A-DEC", "recall")
        k3_recall = _estimate(metrics, cls, "A-k3", "recall")
        recall_pass = bool(dec_recall >= k3_recall)
        recall_checks.append({
            "class": cls,
            "a_dec": dec_recall,
            "a_k3": k3_recall,
            "pass": recall_pass,
        })
        if not recall_pass:
            failures.append(
                f"F-PROD × {cls} × recall: A-DEC {dec_recall:.6g} < A-k3 {k3_recall:.6g}"
            )
        dec_wrong = _estimate(metrics, cls, "A-DEC", "wrong_read")
        k3_wrong = _estimate(metrics, cls, "A-k3", "wrong_read")
        wrong_pass = bool(dec_wrong <= k3_wrong)
        wrong_checks.append({
            "class": cls,
            "a_dec": dec_wrong,
            "a_k3": k3_wrong,
            "pass": wrong_pass,
        })
        if not wrong_pass:
            failures.append(
                f"F-PROD × {cls} × wrong-read: A-DEC {dec_wrong:.6g} > A-k3 {k3_wrong:.6g}"
            )

    cross = [
        row for row in prod_rows
        if row.get("class_label") == "CROSS-FACT / SYNTHESIS"
        and row.get("split") == "eval"
        and row["arms"]["A-DEC"].get("policy_branch")
        == "declared_synthesis_identified_set"
    ]
    synth_checks = []
    for row in cross:
        dec = row["arms"]["A-DEC"]
        k3 = row["arms"]["A-k3"]
        plan_preserved = list(dec.get("ladder_initial_policy_plan") or ()) == list(
            row.get("identified_candidates") or ()
        )
        required = {int(value) for value in (row.get("required_node_ids") or ())}
        dec_receipts = required & {
            int(value) for value in (dec.get("mounted_nodes") or ())
        }
        k3_receipts = required & {
            int(value) for value in (k3.get("mounted_nodes") or ())
        }
        final_mount_receipt_preserved = bool(
            k3_receipts and k3_receipts <= dec_receipts
        )
        passed = bool(
            int(k3["recall"]) == 1
            and int(k3["miss"]) == 0
            and int(dec["recall"]) == 1
            and int(dec["miss"]) == 0
            and plan_preserved
            and final_mount_receipt_preserved
            and dec.get("production_ladder", {}).get("probe_ladder") is True
        )
        synth_checks.append({
            "probe_id": row["probe_id"],
            "fixture": row["fixture"],
            "a_dec_recall": int(dec["recall"]),
            "a_k3_recall": int(k3["recall"]),
            "a_dec_miss": int(dec["miss"]),
            "a_k3_miss": int(k3["miss"]),
            "declared_plan_preserved": bool(plan_preserved),
            "a_dec_final_mounts": list(dec.get("mounted_nodes") or ()),
            "a_k3_final_mounts": list(k3.get("mounted_nodes") or ()),
            "registered_required_nodes": sorted(required),
            "a_dec_preserved_receipt_nodes": sorted(dec_receipts),
            "a_k3_receipt_nodes": sorted(k3_receipts),
            "final_mount_receipt_preserved": bool(final_mount_receipt_preserved),
            "pass": passed,
        })
        if not passed:
            failures.append(
                f"F-PROD × {row['fixture']} × {row['probe_id']} × declared-synthesis receipt"
            )
    fixtures = {row["fixture"] for row in cross}
    synth_coverage = fixtures == {
        "E2E-34-CROSS-FACT", "P4-REPLICATION-CROSS-FACT"
    }
    if not synth_coverage:
        failures.append(
            "F-PROD × CROSS-FACT / SYNTHESIS × declared-synthesis coverage "
            f"(observed fixtures: {sorted(fixtures)})"
        )
    recommended = not failures
    return {
        "registered_rule": (
            "A-DEC is ADOPT-RECOMMENDED iff F-PROD A-DEC recall is at least "
            "A-k3 on every measured class, A-DEC wrong-read rate is no greater "
            "than A-k3, and the declared-synthesis branch preserves cross-fact receipts."
        ),
        "measured_classes": measured,
        "recall_checks": recall_checks,
        "wrong_read_checks": wrong_checks,
        "declared_synthesis_checks": synth_checks,
        "declared_synthesis_fixture_coverage": bool(synth_coverage),
        "failures": failures,
        "recommendation": "ADOPT-RECOMMENDED" if recommended else "NOT-RECOMMENDED",
        "operator_decision_required": True,
        "wired": False,
    }


def gate_adoption(
    adoption: Mapping[str, Any], gates: Mapping[str, str]
) -> dict[str, Any]:
    """Data-validity gates are prerequisites to applying the adoption rule."""
    value = copy.deepcopy(dict(adoption))
    prerequisite_names = (
        "ADM1.3-G0-FRESH-ANCHORS",
        "F-PROD-LADDER-ON-L2-ON",
        "F-ISO-REUSE",
        "FROZEN-RULE-UNCHANGED",
        "PRODUCTION-FILES-UNCHANGED-DURING-RUN",
    )
    red = [name for name in prerequisite_names if gates.get(name) != "PASS"]
    value["data_validity_gate_failures"] = red
    value["data_validity_gates_pass"] = not red
    if red:
        value["failures"] = [
            *list(value.get("failures") or ()),
            *(f"F-PROD data-validity gate {name}=RED" for name in red),
        ]
        value["recommendation"] = "NOT-RECOMMENDED"
    return value


def _frame_differences(
    prod: Mapping[str, Any], iso: Mapping[str, Any]
) -> list[dict[str, Any]]:
    differences = []
    for cls in CLASSES:
        for arm in ARMS:
            for metric in ("recall", "wrong_read", "miss", "mount_count"):
                left = _estimate(prod, cls, arm, metric)
                right = _estimate(iso, cls, arm, metric)
                if left is None or right is None:
                    continue
                if abs(left - right) > 1.0e-12:
                    differences.append({
                        "class": cls,
                        "arm": arm,
                        "metric": metric,
                        "F-PROD": left,
                        "F-ISO": right,
                        "delta_prod_minus_iso": left - right,
                    })
    return differences


def corpus_frame_direction(
    prod_counts: Mapping[str, Sequence[int]],
    iso_counts: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    deltas = {
        arm: int(prod_counts[arm][0]) - int(iso_counts[arm][0]) for arm in ARMS
    }
    prod_rates = [
        int(prod_counts[arm][0]) / int(prod_counts[arm][1]) for arm in ARMS
    ]
    iso_rates = [
        int(iso_counts[arm][0]) / int(iso_counts[arm][1]) for arm in ARMS
    ]
    prod_spread = max(prod_rates) - min(prod_rates)
    iso_spread = max(iso_rates) - min(iso_rates)
    return {
        "answer_correct_deltas_prod_minus_iso": deltas,
        "rescued_arms": [arm for arm, delta in deltas.items() if delta > 0],
        "regressed_arms": [arm for arm, delta in deltas.items() if delta < 0],
        "unchanged_arms": [arm for arm, delta in deltas.items() if delta == 0],
        "F-PROD_inter_arm_spread": prod_spread,
        "F-ISO_inter_arm_spread": iso_spread,
        "arm_differences_compressed": prod_spread < iso_spread,
    }


def frame_disagreement_text(
    direction: Mapping[str, Any],
    prod_counts: Mapping[str, Sequence[int]],
    iso_counts: Mapping[str, Sequence[int]],
    metric_difference_count: int,
) -> str:
    changed = [
        arm for arm in ARMS if list(prod_counts[arm]) != list(iso_counts[arm])
    ]
    if not changed:
        return (
            "The frames do not disagree on CORPUS-100 answer counts. The expected "
            "near-duplicate ladder rescue was not observed; "
            f"{metric_difference_count} nonzero structural metric cells remain in "
            "the machine receipt."
        )
    details = "; ".join(
        f"{arm} {iso_counts[arm][0]}/{iso_counts[arm][1]} → "
        f"{prod_counts[arm][0]}/{prod_counts[arm][1]}"
        for arm in changed
    )
    rescued = list(direction["rescued_arms"])
    regressed = list(direction["regressed_arms"])
    compressed = bool(direction["arm_differences_compressed"])
    observations = []
    if rescued:
        observations.append(f"higher production-frame correctness for {', '.join(rescued)}")
    if regressed:
        observations.append(f"lower production-frame correctness for {', '.join(regressed)}")
    observations.append(
        "compressed inter-arm differences"
        if compressed else "no compression of inter-arm differences"
    )
    interpretation = (
        "This is consistent with the production ladder rescuing near-duplicate "
        "answers and compressing arm differences."
        if rescued and not regressed and compressed
        else "The data do not support the full expected rescue-and-compression account."
    )
    return (
        "The frames **DISAGREE**. On all CORPUS-100 probes, answer correctness "
        f"moves {details} (F-ISO → F-PROD), showing {'; '.join(observations)}. "
        f"{interpretation} The {metric_difference_count} nonzero class/arm metric "
        "deltas are retained in the machine receipt; F-ISO remains only the "
        "mechanism-isolation annex."
    )


def render_dual_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# GRM-ADM1.3 dual-frame k-policy verdict",
        "",
        f"Frozen rule SHA-256: `{result['rule']['sha256']}`  ",
        f"Frozen margin threshold: `{result['rule']['margin_threshold']:.9g}`  ",
        f"Bootstrap: `{BOOTSTRAP_REPLICATES}` nonparametric probe resamples, 95% percentile CI.",
        "",
        "## Gate status",
        "",
        "| Gate | Status |",
        "|---|---|",
    ]
    for gate, status in result["gates"].items():
        lines.append(f"| {gate} | **{status}** |")
    lines += [
        "",
        "## Dual-frame class × arm table (verbatim)",
        "",
        "Recall means the required answer-bearing graft is present/derivable. Wrong-read is a registered competing-value answer while the required graft was mounted. F-PROD is production-relevant; F-ISO is the ladder-off mechanism-isolation annex.",
        "",
        "| Frame | Class | Arm | n | Recall (95% CI) | Wrong-read (95% CI) | Miss (95% CI) | Mounts/turn (95% CI) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for frame in ("F-PROD", "F-ISO"):
        table = result["metrics"][frame]
        for cls in CLASSES:
            for arm in ARMS:
                metrics = table[cls][arm]
                lines.append(
                    f"| {frame} | {cls} | {arm} | {metrics['recall']['n']} | "
                    f"{_metric_cell(metrics['recall'])} | "
                    f"{_metric_cell(metrics['wrong_read'])} | "
                    f"{_metric_cell(metrics['miss'])} | "
                    f"{_metric_cell(metrics['mount_count'], percent=False)} |"
                )
    lines += [
        "",
        "## Fresh frame-matched A-k3 anchors",
        "",
        f"**ADM1.3-G0: {result['fresh_g0']['status']}.** Every expectation below was generated under explicit ladder-ON/L2-ON F-PROD and closed before the corresponding comparison arms.",
        "",
        "| Fixture | Fresh A-k3 expected correct | Arm A-k3 observed correct | Match |",
        "|---|---:|---:|---|",
    ]
    for fixture, value in result["fresh_g0"]["fixtures"].items():
        expected = value["fresh_expected_a_k3_answer_correct"]
        observed = value["observed_arm_a_k3_answer_correct"]
        lines.append(
            f"| {fixture} | {expected[0]}/{expected[1]} | "
            f"{observed[0]}/{observed[1]} | {'PASS' if value['pass'] else 'RED'} |"
        )
    lines += [
        "",
        "## Production-relevant verdict",
        "",
        f"**ADM-VERDICT (F-PROD): {result['prod_predictions']['ADM-VERDICT']} — {result['prod_predictions']['verdict_reason']}**",
        "",
        "F-ISO remains the mechanism-isolation annex; its prior ladder-off verdict is not substituted for the production verdict.",
        "",
        "## Frame disagreement",
        "",
        result["frame_disagreement_paragraph"],
        "",
        "## Registered adoption rule",
        "",
    ]
    adoption = result["adoption"]
    if adoption["recommendation"] == "ADOPT-RECOMMENDED":
        lines.append(
            "**ADOPT-RECOMMENDED: in F-PROD, A-DEC recall is at least A-k3 on every measured class, its wrong-read rate is no greater than A-k3, and the declared-synthesis cross-fact receipts are preserved. Adoption remains an operator decision; no wiring was performed.**"
        )
    else:
        lines.append(
            "**NOT-RECOMMENDED: " + "; ".join(adoption["failures"])
            + ". Adoption remains an operator decision; no wiring was performed.**"
        )
    lines += [
        "",
        "## Flag-gated wiring plan (not wired)",
        "",
        "1. Add an experiment-only admission selector behind a new default-off flag (for example, `GRM_ADMISSION_POLICY=decisive`); absent/false retains byte-identical fixed top-3 behavior.",
        "2. Bind the flag to the frozen rule hash and emit the identifier hits, score margin, selected branch, initial plan, ladder trip, and final mounts on every probe.",
        "3. Shadow-run A-DEC beside fixed A-k3, require same-frame G0 plus the registered recall/wrong-read/synthesis gates, then expose a one-switch rollback to fixed A-k3.",
        "4. Do not enable the flag by default until the operator makes the adoption decision.",
        "",
        "## Reused F-ISO receipts",
        "",
        f"`{result['iso_reuse']['run_dir']}` was reused without recomputation. Hash/count validation: **{'PASS' if result['iso_reuse']['all_hashes_and_counts_match'] else 'RED'}**; ADM1.2 zero-change L2 candidate-set audit present: **{result['iso_reuse']['adm1_2_zero_changed_candidate_sets_audit_present']}**.",
        "",
        "## Files and provenance",
        "",
    ]
    for receipt in result["receipt_files"]:
        lines.append(f"- `{receipt['path']}` sha256 `{receipt['sha256']}`")
    lines += ["", "## Anything not done", ""]
    residuals = list(result.get("not_done") or ())
    lines.extend(f"- {item}" for item in residuals) if residuals else lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def summarize(run_dir: Path, iso_run: Path, rule_path: Path) -> tuple[dict[str, Any], Path, Path]:
    run_dir = run_dir.resolve()
    existing_marker = run_dir / "summary_complete.json"
    if existing_marker.is_file():
        marker_value = read_json(existing_marker)
        adjudication_path = ROOT / marker_value["adjudication"]["path"]
        report_path = ROOT / marker_value["report"]["path"]
        if (
            file_record(adjudication_path) != marker_value["adjudication"]
            or file_record(report_path) != marker_value["report"]
        ):
            raise DualFrameError("existing ADM1.3 summary receipt failed hash validation")
        return read_json(adjudication_path), adjudication_path, report_path
    assert_run_inventory(run_dir)
    rule = validate_frozen_rule(rule_path)
    prod_rows = _load_prod_rows(run_dir)
    if len(prod_rows) != 53:
        raise DualFrameError(f"F-PROD expected 53 rows, observed {len(prod_rows)}")
    iso_rows, iso_reuse = _load_iso_rows(iso_run)
    prod_metrics = metric_table(prod_rows)
    iso_metrics = metric_table(iso_rows)
    fresh_g0 = _fresh_g0(run_dir, prod_rows)
    adoption = adoption_rule(prod_metrics, prod_rows)
    prod_predictions = adjudicate_predictions(prod_metrics)
    inventory_before = read_json(run_dir / "run_manifest.json")["production_inventory_before"]
    inventory_after = production_frame_inventory()
    production_unchanged = inventory_before == inventory_after
    frame_config_pass = all(
        row.get("production_config", {}).get("probe_ladder") is True
        and row.get("production_config", {}).get("revision_resolution") is True
        and row.get("frame") == "F-PROD"
        for row in prod_rows
    )
    rule_pass = all(row.get("rule_sha256") == FROZEN_RULE_SHA256 for row in prod_rows)
    iso_pass = bool(
        iso_reuse["all_hashes_and_counts_match"]
        and iso_reuse["adm1_2_zero_changed_candidate_sets_audit_present"]
    )
    differences = _frame_differences(prod_metrics, iso_metrics)
    prod_corpus = {
        arm: _answer_counts(prod_rows, "CORPUS-100", arm) for arm in ARMS
    }
    iso_corpus = {
        arm: _answer_counts(iso_rows, "CORPUS-100", arm) for arm in ARMS
    }
    direction = corpus_frame_direction(prod_corpus, iso_corpus)
    disagreement = frame_disagreement_text(
        direction, prod_corpus, iso_corpus, len(differences)
    )

    receipt_paths = [
        ORDER_PATH,
        rule_path,
        run_dir / "run_manifest.json",
    ]
    for stage in PROD_STAGES:
        receipt_paths += [
            run_dir / stage / "adm_rows.jsonl",
            run_dir / stage / "fresh_anchors.jsonl",
            run_dir / stage / "stage_complete.json",
        ]
        receipt_paths += sorted((run_dir / stage).glob("fresh_anchor_receipt_*.json"))
    receipt_paths += [path for path in ISO_ROW_PATHS if path.is_file()]
    gates = {
        "ADM1.3-G0-FRESH-ANCHORS": fresh_g0["status"],
        "F-PROD-LADDER-ON-L2-ON": "PASS" if frame_config_pass else "RED",
        "F-ISO-REUSE": "PASS" if iso_pass else "RED",
        "FROZEN-RULE-UNCHANGED": "PASS" if rule_pass else "RED",
        "PRODUCTION-FILES-UNCHANGED-DURING-RUN": "PASS" if production_unchanged else "RED",
        "NO-PRODUCTION-WIRING": "PASS",
        "DUAL-FRAME-REPORT": "PASS",
    }
    adoption = gate_adoption(adoption, gates)
    result = {
        "schema": "grm.adm1_3.dual_frame_adjudication.v1",
        "order": file_record(ORDER_PATH),
        "run_dir": str(run_dir),
        "created_unix_ns": time.time_ns(),
        "rule": {**file_record(rule_path), "margin_threshold": float(rule["margin_threshold"])},
        "frames": {
            "F-PROD": {
                "role": "production-relevant",
                "probe_ladder": True,
                "revision_resolution": True,
                "row_count": len(prod_rows),
            },
            "F-ISO": {
                "role": "mechanism-isolation annex",
                "probe_ladder": False,
                "row_count": len(iso_rows),
                "recomputed": False,
            },
        },
        "metrics": {"F-PROD": prod_metrics, "F-ISO": iso_metrics},
        "prod_predictions": prod_predictions,
        "fresh_g0": fresh_g0,
        "iso_reuse": iso_reuse,
        "adoption": adoption,
        "frame_differences": differences,
        "corpus_answer_correct": {"F-PROD": prod_corpus, "F-ISO": iso_corpus},
        "corpus_frame_direction": direction,
        "frame_disagreement_paragraph": disagreement,
        "production_inventory_before": inventory_before,
        "production_inventory_after": inventory_after,
        "gates": gates,
        "receipt_files": [file_record(path) for path in receipt_paths if path.is_file()],
        "not_done": [
            "Adoption was not performed; it remains an operator decision.",
            "The flag-gated wiring plan was not implemented.",
            "P-AMB remains unmeasured: the certified fixtures contain zero qualifying ambiguous probes, and none were manufactured.",
        ],
    }
    adjudication_path = write_content_addressed(
        run_dir, "dual_frame_adjudication", result
    )
    report_result = dict(result)
    report_result["receipt_files"] = [
        *result["receipt_files"], file_record(adjudication_path)
    ]
    report = render_dual_report(report_result)
    report_path = write_content_addressed(
        run_dir, "GRM_ADM1_3_DUAL_FRAME", report, suffix=".md"
    )
    marker = run_dir / "summary_complete.json"
    write_exclusive_json(marker, {
        "schema": "grm.adm1_3.summary_complete.v1",
        "completed_unix_ns": time.time_ns(),
        "adjudication": file_record(adjudication_path),
        "report": file_record(report_path),
        "recommendation": adoption["recommendation"],
        "fresh_g0": fresh_g0["status"],
    })
    return result, adjudication_path, report_path


def result_exit_code(result: Mapping[str, Any]) -> int:
    return 0 if all(status == "PASS" for status in result["gates"].values()) else 2


def cpu_selftest() -> dict[str, Any]:
    attempts = build_dec_driver_attempts(
        ranking=[5, 6, 7, 4],
        topk=3,
        precise=[5],
        point_lookup=True,
        max_trips=1,
        policy_plan=[5, 7],
    )
    if attempts != [([5, 7], True), ([5, 6, 7], True)]:
        raise DualFrameError(f"unexpected A-DEC ladder: {attempts}")
    synthetic = {
        cls: {
            arm: {
                metric: {
                    "n": 1 if cls != "AMBIGUOUS" else 0,
                    "estimate": (0.0 if metric == "wrong_read" else 1.0)
                    if cls != "AMBIGUOUS" else None,
                    "ci95": [0.0, 1.0] if cls != "AMBIGUOUS" else None,
                }
                for metric in ("recall", "wrong_read", "miss", "mount_count")
            }
            for arm in ARMS
        }
        for cls in CLASSES
    }
    synthetic_rows = [{
        "probe_id": "x",
        "fixture": "E2E-34-CROSS-FACT",
        "class_label": "CROSS-FACT / SYNTHESIS",
        "split": "eval",
        "identified_candidates": [1, 2],
        "required_node_ids": [2],
        "arms": {
            "A-DEC": {
                "policy_branch": "declared_synthesis_identified_set",
                "recall": 1,
                "miss": 0,
                "ladder_initial_policy_plan": [1, 2],
                "production_ladder": {"probe_ladder": True},
                "mounted_nodes": [1, 2],
            },
            "A-k3": {"recall": 1, "miss": 0, "mounted_nodes": [1, 2]},
        },
    }, {
        "probe_id": "y",
        "fixture": "P4-REPLICATION-CROSS-FACT",
        "class_label": "CROSS-FACT / SYNTHESIS",
        "split": "eval",
        "identified_candidates": [3, 4],
        "required_node_ids": [4],
        "arms": {
            "A-DEC": {
                "policy_branch": "declared_synthesis_identified_set",
                "recall": 1,
                "miss": 0,
                "ladder_initial_policy_plan": [3, 4],
                "production_ladder": {"probe_ladder": True},
                "mounted_nodes": [3, 4],
            },
            "A-k3": {"recall": 1, "miss": 0, "mounted_nodes": [3, 4]},
        },
    }]
    adoption = adoption_rule(synthetic, synthetic_rows)
    if adoption["recommendation"] != "ADOPT-RECOMMENDED":
        raise DualFrameError("synthetic adoption rule should recommend")
    return {
        "schema": "grm.adm1_3.cpu_selftest.v1",
        "status": "PASS",
        "checks": {
            "a_dec_trip0_then_production_retry": "PASS",
            "registered_adoption_rule": "PASS",
            "frozen_rule_sha256": FROZEN_RULE_SHA256,
        },
    }


def _new_run_dir(base: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = base / f"adm1_3_prod_run_{stamp}_{os.getpid():x}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_run_manifest(run_dir: Path, args: argparse.Namespace) -> None:
    path = run_dir / "run_manifest.json"
    if path.is_file():
        return
    sources = [
        ORDER_PATH,
        Path(__file__).resolve(),
        E2E_WRAPPER,
        ROOT / "scripts" / "grm_adm1_gpu.py",
        ROOT / "scripts" / "grm_adm1_analysis.py",
        ROOT / "scripts" / "grm_e2e_session.py",
        ROOT / "scripts" / "grm_probe_ladder.py",
        ROOT / "core" / "graft_arena.py",
        ROOT / "core" / "grm_supersession.py",
    ]
    write_exclusive_json(path, {
        "schema": "grm.adm1_3.prod_run.v1",
        "created_unix_ns": time.time_ns(),
        "order": file_record(ORDER_PATH),
        "stage_order": list(PROD_STAGES),
        "rule": file_record(args.rule.resolve()),
        "iso_run": str(args.iso_run.resolve()),
        "gpu": str(args.gpu),
        "lease_seconds": int(args.lease_seconds),
        "lock_wait_seconds": int(args.lock_wait_seconds),
        "bare_invocation": (
            "PYTHONPATH=/mnt/ForgeRealm/GraftRepository python3 "
            "scripts/grm_adm1_3_dual_frame.py --stage all"
        ),
        "production_pins": {"probe_ladder": True, "revision_resolution": True},
        "frozen_rule_reused_without_refit": True,
        "production_inventory_before": production_frame_inventory(),
        "source_provenance": {
            str(path.relative_to(ROOT)): file_record(path) for path in sources
        },
    })


def orchestrate(args: argparse.Namespace) -> int:
    validate_frozen_rule(args.rule)
    run_dir = args.run_dir.resolve() if args.run_dir else _new_run_dir(args.artifact_dir.resolve())
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_manifest(run_dir, args)
    assert_run_inventory(run_dir)
    print(f"run_dir={run_dir}", flush=True)
    for index, stage in enumerate(PROD_STAGES):
        marker = run_dir / stage / "stage_complete.json"
        if marker.is_file():
            print(f"stage={stage} status=already_complete", flush=True)
            continue
        if index:
            print(f"inter_stage_gap_s={GAP_SECONDS}", flush=True)
            time.sleep(GAP_SECONDS)
        assert_run_inventory(run_dir)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--stage", stage,
            "--run-dir", str(run_dir),
            "--rule", str(args.rule.resolve()),
            "--iso-run", str(args.iso_run.resolve()),
            "--gpu", str(args.gpu),
            "--lease-seconds", str(args.lease_seconds),
            "--lock-wait-seconds", str(args.lock_wait_seconds),
        ]
        env = os.environ.copy()
        prior_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + prior_path if prior_path else "")
        completed = subprocess.run(command, cwd=ROOT, env=env)
        if completed.returncode != 0:
            print(f"stage={stage} status=FAILED code={completed.returncode}", flush=True)
            return int(completed.returncode)
    result, adjudication, report = summarize(run_dir, args.iso_run, args.rule)
    print(f"adjudication={adjudication}", flush=True)
    print(f"report={report}", flush=True)
    print(f"recommendation={result['adoption']['recommendation']}", flush=True)
    return result_exit_code(result)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "selftest", "summarize", *PROD_STAGES), default="all")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--iso-run", type=Path, default=ISO_RUN_DEFAULT)
    parser.add_argument("--rule", type=Path, default=RULE_DEFAULT)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    parser.add_argument("--lock-wait-seconds", type=int, default=DEFAULT_LOCK_WAIT_SECONDS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "selftest":
        value = cpu_selftest()
        path = write_content_addressed(args.artifact_dir.resolve(), "adm1_3_cpu_selftest", value)
        print(json.dumps(value, sort_keys=True))
        print(f"receipt={path}")
        return 0
    if args.run_dir is None and args.stage != "all":
        raise DualFrameError(f"--stage {args.stage} requires --run-dir")
    if int(args.lease_seconds) > MAX_LEASE_SECONDS:
        raise DualFrameError(f"lease cap exceeds {MAX_LEASE_SECONDS}s")
    if args.stage == "all":
        return orchestrate(args)
    run_dir = args.run_dir.resolve()
    validate_frozen_rule(args.rule)
    if args.stage == "summarize":
        result, adjudication, report = summarize(run_dir, args.iso_run, args.rule)
        print(f"adjudication={adjudication}")
        print(f"report={report}")
        print(f"recommendation={result['adoption']['recommendation']}")
        return result_exit_code(result)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    assert_run_inventory(run_dir)
    with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
        if args.stage == "lookup":
            run_lookup_stage(run_dir, args.rule.resolve())
        else:
            run_e2e_stage(
                run_dir,
                args.stage,
                args.rule.resolve(),
                lease_seconds=int(args.lease_seconds),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
