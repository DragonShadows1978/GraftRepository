#!/usr/bin/env python3
"""Guarded live runner for ORDER GRM-ADM1.

Stages share ``/tmp/forge-gpu.lock`` with CMC1.1.  The ``all`` orchestrator
creates one append-only run directory, runs the held-out MiniCPM fit first,
freezes the decisiveness rule to a content-addressed file, and only then
launches the DIAG/E2E/P4 evaluation replays.  Each live stage uses one visible
GPU, a sub-590-second lease, and a 30-second inter-stage gap.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from grm_adm1_analysis import (  # noqa: E402
    ADMError,
    ARMS,
    ARTIFACT_DIR,
    canonical_json_bytes,
    corpus100_fixture_rows,
    file_record,
    freeze_rule,
    is_identifier_binding,
    normalized_words,
    policy_plan,
    production_inventory,
    read_json,
    read_jsonl,
    sha256_file,
    supersession_fixture_rows,
    write_content_addressed,
    write_exclusive_json,
    _extract_assignment,
)
from grm_adm1_probe_adjudication import (  # noqa: E402
    completed_capture_available,
    validate_frame_receipts,
)


LOCK_PATH = Path("/tmp/forge-gpu.lock")
MAX_LEASE_SECONDS = 590
DEFAULT_LEASE_SECONDS = 580
DEFAULT_LOCK_WAIT_SECONDS = 86_400
GAP_SECONDS = 30
LIVE_STAGES = ("minicpm", "diag", "e2e", "p4")
FRAME_CONFIG = {
    "diag": {
        "anchor_fixture": "DIAG-CURRENT-FULL",
        "metric_fixture": "DIAG-CURRENT-FULL",
        "metric_class": "POINT-LOOKUP",
        "metric_turns": tuple(range(34)),
    },
    "e2e": {
        "anchor_fixture": "E2E-34-FULL-A-k3-ANCHOR",
        "metric_fixture": "E2E-34-CROSS-FACT",
        "metric_class": "CROSS-FACT / SYNTHESIS",
        "metric_turns": (5, 13),
    },
    "p4": {
        "anchor_fixture": "P4-REPLICATION-FULL-A-k3-ANCHOR",
        "metric_fixture": "P4-REPLICATION-CROSS-FACT",
        "metric_class": "CROSS-FACT / SYNTHESIS",
        "metric_turns": (5, 13),
    },
}


class LiveError(ADMError):
    pass


def _alarm(_signum, _frame):
    raise TimeoutError("GPU lease wall-time cap exceeded")


@contextmanager
def gpu_lease(seconds: int, wait_seconds: int) -> Iterator[None]:
    seconds = int(seconds)
    wait_seconds = int(wait_seconds)
    if not (1 <= seconds <= MAX_LEASE_SECONDS):
        raise LiveError(f"lease must be in 1..{MAX_LEASE_SECONDS}s, got {seconds}")
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = LOCK_PATH.open("a+")
    started_wait = time.monotonic()
    next_notice = 0.0
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            waited = time.monotonic() - started_wait
            if waited >= wait_seconds:
                handle.close()
                raise TimeoutError(
                    f"GPU lock unavailable after {wait_seconds}s: {LOCK_PATH}")
            if waited >= next_notice:
                print(f"gpu_lock_waiting_s={waited:.1f} path={LOCK_PATH}", flush=True)
                next_notice += 30.0
            time.sleep(5)
    prior = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    lease_started = time.monotonic()
    try:
        print(
            f"gpu_lease_acquired={LOCK_PATH} cap_s={seconds} "
            f"waited_s={lease_started-started_wait:.3f}",
            flush=True,
        )
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prior)
        elapsed = time.monotonic() - lease_started
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        print(f"gpu_lease_released_s={elapsed:.3f}", flush=True)


def _one_manifest(path: Path | None = None) -> Path:
    if path is not None:
        path = path.resolve()
        if not path.is_file():
            raise LiveError(f"fixture manifest does not exist: {path}")
        return path
    paths = sorted(ARTIFACT_DIR.glob("fixture_manifest_*.json"))
    if len(paths) != 1:
        raise LiveError(f"expected exactly one fixture manifest, found {len(paths)}")
    return paths[0]


def _one_rule(run_dir: Path) -> Path:
    paths = sorted(run_dir.glob("decisiveness_rule_*.json"))
    if len(paths) != 1:
        raise LiveError(f"expected exactly one frozen rule, found {len(paths)}")
    return paths[0]


def assert_production_frozen(manifest_path: Path) -> None:
    registered = read_json(manifest_path).get("production_inventory_before")
    current = production_inventory()
    if registered != current:
        raise LiveError(
            "production inventory drifted after registration; refuse live eval "
            "until an explicit pre-eval registration supersession is frozen")


def append_jsonl_once(path: Path, row: Mapping[str, Any], *, key: str = "probe_id") -> None:
    payload = canonical_json_bytes(dict(row)).decode("utf-8").strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            existing[str(value[key])] = canonical_json_bytes(value).decode("utf-8").strip()
    row_key = str(row[key])
    if row_key in existing:
        if existing[row_key] != payload:
            raise LiveError(f"append-only row collision for {row_key} in {path}")
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload.replace("\n", "") + "\n")
        handle.flush()


def _ordered_identifier_tokens(arena, question: str) -> tuple[list[str], set[str]]:
    rare = {str(value).casefold() for value in arena._rare_tokens(question)}
    selected = rare or {
        str(value).casefold() for value in arena._query_lex_tokens(question)
    }
    ordered = []
    seen = set()
    for word in normalized_words(question):
        if word in selected and word not in seen:
            ordered.append(word)
            seen.add(word)
    # Hyphenated rare tokens normally survive normalized_words whole.  Keep a
    # deterministic tail for any tokenizer-shaped identifier not seen there.
    ordered.extend(sorted(selected - seen))
    return ordered, rare


def route_profile(arena, question: str, *, exclude: set[int]) -> dict[str, Any]:
    eligible = [
        int(value) for value in arena._route_cand_base()
        if int(value) not in exclude
    ]
    if not eligible:
        return {
            "ranking": [],
            "score_rows": [],
            "route_margin_1_2": 0.0,
            "identifier_tokens": [],
            "rare_identifier_tokens": [],
            "identified_candidates": [],
            "identifier_hit_count": 0,
            "route_backend": None,
        }
    probe_key = arena._probe_key(question)
    ranking = list(arena.route(
        question,
        exclude=exclude,
        limit=len(eligible),
        probe_key=probe_key,
    ) or [])
    route_backend = str(getattr(arena, "last_route_backend", "unknown"))
    base = arena._vector_route_scores(probe_key, eligible)
    if base is None:
        base = {}
        for index in eligible:
            score = arena._cent_score(probe_key, arena.grafts[index])
            if math.isfinite(float(score)):
                base[index] = float(score)
    base = arena._length_debias_scores(base, eligible)
    base = arena._normalize_scores(base) or {}
    qlex = arena._query_lex_tokens(question)
    scores = {
        int(index): float(base[index]) + float(arena._lex_bonus(qlex, arena.grafts[index]))
        for index in eligible
        if index in base and math.isfinite(float(base[index]))
    }
    reference = sorted(scores, key=lambda index: (-scores[index], index))
    if reference != ranking:
        raise LiveError(
            "production route ranking differs from exact score reconstruction: "
            f"backend={route_backend} production={ranking} reference={reference}")
    ordered_tokens, rare_tokens = _ordered_identifier_tokens(arena, question)
    identified = [
        index for index in ranking
        if is_identifier_binding(
            candidate_text=str(arena.grafts[index].get("text", "") or ""),
            ordered_identifier_tokens=ordered_tokens,
            rare_identifier_tokens=rare_tokens,
        )
    ]
    margin = 0.0
    if len(ranking) >= 2:
        margin = float(scores[ranking[0]] - scores[ranking[1]])
    return {
        "ranking": [int(value) for value in ranking],
        "score_rows": [
            {
                "rank": rank,
                "node_id": int(index),
                "score": float(scores[index]),
                "identifier_binding": bool(index in set(identified)),
                "text_prefix": str(arena.grafts[index].get("text", "") or "")[:160],
            }
            for rank, index in enumerate(ranking, 1)
        ],
        "route_margin_1_2": margin,
        "identifier_tokens": ordered_tokens,
        "rare_identifier_tokens": sorted(rare_tokens),
        "identified_candidates": [int(value) for value in identified],
        "identifier_hit_count": len(identified),
        "route_backend": route_backend,
    }


def arm_plans(profile: Mapping[str, Any], rule: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    ranking = list(profile["ranking"])
    decisive, branch = policy_plan(
        ranking=ranking,
        identified_candidates=profile["identified_candidates"],
        route_margin_1_2=float(profile["route_margin_1_2"]),
        margin_threshold=float(rule["margin_threshold"]),
    )
    return {
        "A-k1": {"rank_plan": ranking[:1], "policy_branch": "fixed_k1"},
        "A-k2": {"rank_plan": ranking[:2], "policy_branch": "fixed_k2"},
        "A-k3": {"rank_plan": ranking[:3], "policy_branch": "fixed_k3"},
        "A-DEC": {"rank_plan": decisive, "policy_branch": branch},
    }


def fit_plan(arena, rank_plan: Sequence[int]) -> list[int]:
    used = 0
    out = []
    for value in rank_plan:
        index = int(value)
        tokens = int(arena.grafts[index]["ntok"])
        if used + tokens <= int(arena.width):
            out.append(index)
            used += tokens
    # This is the existing Arena.step physical order: selected-by-rank, then
    # stable graft-index order before assembly.
    return sorted(out)


def snapshot_arena(arena) -> dict[str, Any]:
    return {
        # GPT-OSS consumes the mutable outer cache list in-place while it
        # builds the successor caches.  Preserve that container for the next
        # counterfactual arm; the tensor tuples themselves are immutable.
        "caches": None if arena.caches is None else list(arena.caches),
        "pos": int(arena.pos),
        "live_segs": list(arena.live_segs),
        "cur_mounts": list(arena.cur_mounts),
        "cur_mount_n": int(arena.cur_mount_n),
        "graft_count": len(arena.grafts),
        "mount_clock": getattr(arena, "mount_clock", None),
        "page_ins": getattr(arena, "page_ins", None),
        "last_used": [graft.get("last_used") for graft in arena.grafts],
    }


def restore_arena(arena, snapshot: Mapping[str, Any]) -> None:
    from core import kv_graft

    arena.caches = snapshot["caches"]
    arena.pos = int(snapshot["pos"])
    arena.live_segs = list(snapshot["live_segs"])
    arena.cur_mounts = list(snapshot["cur_mounts"])
    arena.cur_mount_n = int(snapshot["cur_mount_n"])
    if len(arena.grafts) != int(snapshot["graft_count"]):
        del arena.grafts[int(snapshot["graft_count"]):]
        arena._bump_cuda_gqa_epoch()
    if snapshot["mount_clock"] is None:
        if hasattr(arena, "mount_clock"):
            delattr(arena, "mount_clock")
    else:
        arena.mount_clock = snapshot["mount_clock"]
    if snapshot["page_ins"] is None:
        if hasattr(arena, "page_ins"):
            delattr(arena, "page_ins")
    else:
        arena.page_ins = snapshot["page_ins"]
    for graft, last_used in zip(arena.grafts, snapshot["last_used"]):
        if last_used is None:
            graft.pop("last_used", None)
        else:
            graft["last_used"] = last_used
    kv_graft.clear_injection(arena.m)


def _contains(answer: str, values: Sequence[str]) -> bool:
    folded = str(answer).casefold()
    return any(str(value).casefold() in folded for value in values)


def evaluate_attempt(
    arena,
    *,
    question: str,
    picks: Sequence[int],
    ngen: int,
    accepted_values: Sequence[str],
    competing_values: Sequence[str],
    required_nodes: Sequence[int],
) -> dict[str, Any]:
    before = snapshot_arena(arena)
    try:
        for layer in arena.m.layers:
            layer.self_attn.live_shift = arena.live_shift
        answer, info = arena._attempt(
            question,
            list(picks),
            int(ngen),
            False,
            arena.stop_sequences,
        )
        actual = [int(value) for value in arena.cur_mounts]
        recall = bool(set(actual) & {int(value) for value in required_nodes})
        correct = _contains(answer, accepted_values)
        competing = _contains(answer, competing_values)
        return {
            "rank_plan_fitted": [int(value) for value in picks],
            "mounted_nodes": actual,
            "answer": str(answer),
            "answer_correct": bool(correct),
            "recall": int(recall),
            "wrong_read": int(recall and not correct and competing),
            "miss": int(not recall),
            "mount_count": len(actual),
            "arena_info": dict(info or {}),
        }
    finally:
        restore_arena(arena, before)


def build_probe_row(
    *,
    probe_id: str,
    fixture: str,
    class_label: str,
    split: str,
    question: str,
    profile: Mapping[str, Any],
    plans: Mapping[str, Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
    required_nodes: Sequence[int],
    accepted_values: Sequence[str],
    rule_path: Path,
    started_ns: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ranking = [int(value) for value in profile["ranking"]]
    required = {int(value) for value in required_nodes}
    row = {
        "schema": "grm.adm1.probe.v1",
        "probe_id": probe_id,
        "fixture": fixture,
        "class_label": class_label,
        "split": split,
        "question": question,
        "evaluation_started_unix_ns": int(started_ns),
        "rule_sha256": sha256_file(rule_path),
        "identifier_hit_count": int(profile["identifier_hit_count"]),
        "identifier_tokens": list(profile["identifier_tokens"]),
        "identified_candidates": list(profile["identified_candidates"]),
        "router_rank_order": ranking,
        "router_score_rows": list(profile["score_rows"]),
        "route_margin_1_2": float(profile["route_margin_1_2"]),
        "route_backend": profile["route_backend"],
        "required_node_ids": sorted(required),
        "accepted_values": list(accepted_values),
        "required_in_top1": bool(required & set(ranking[:1])),
        "required_in_top3": bool(required & set(ranking[:3])),
        "arms": {},
    }
    for arm in ARMS:
        row["arms"][arm] = {
            **dict(results[arm]),
            "rank_plan": [int(value) for value in plans[arm]["rank_plan"]],
            "policy_branch": str(plans[arm]["policy_branch"]),
        }
    if extra:
        row.update(dict(extra))
    return row


def _load_minicpm():
    tensor_path = "/mnt/ForgeRealm/Project-Tensor/tensor_cuda"
    if tensor_path not in sys.path:
        sys.path.insert(0, tensor_path)
    import tensor_cuda as tc
    from core.minicpm3_tc import MiniCPM3_TC, _snap
    from core.mistral7b_tc import QuantLinearTC, RMSNormTC
    from tokenizers import Tokenizer as HFTok

    QuantLinearTC.FUSED_DECODE = True
    RMSNormTC.USE_FUSED = True
    tokenizer = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
    model, info = MiniCPM3_TC.from_pretrained()
    tc.set_alloc_pooling(True)
    for layer in model.layers:
        layer.self_attn.absorbed_decode = True
    return model, tokenizer, info, tc


def _corpus_bank(model, tokenizer):
    from core.graft_arena import ArenaCache

    source = ROOT / "tests" / "test_graft_corpus100.py"
    families = _extract_assignment(source, "FAMILIES")
    arena = ArenaCache(
        model,
        encode=lambda text: tokenizer.encode(text).ids,
        decode=lambda ids: tokenizer.decode(ids),
        sink_text="<conversation>\n",
        arena_width=384,
        route_layer=44,
        topk=3,
        live_turns=2,
        cache_deposits=False,
        length_debias=False,
        # ADM1.2: L2 is production-default ON.  This corpus has no revision
        # edges, so the migration is an identity while keeping the frame pinned.
        revision_resolution=True,
    )
    meta = []
    for family, mk_text, _mk_probe, mk_values in families:
        for instance in range(10):
            code, value = mk_values(instance)
            index = int(arena.deposit(mk_text(code, value)))
            meta.append({
                "family": family,
                "code": str(code),
                "value": str(value),
                "node_id": index,
            })
    return arena, meta


def _fit_rows(arena, meta: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    registered = [row for row in corpus100_fixture_rows() if row["split"] == "fit"]
    rows = []
    for probe in registered:
        profile = route_profile(arena, probe["question"], exclude=set())
        target = next(
            int(row["node_id"]) for row in meta
            if row["family"] == probe["family"]
            and row["node_id"] == int(probe["target_ordinal"])
        )
        rows.append({
            "probe_id": probe["probe_id"],
            "identifier_hit_count": int(profile["identifier_hit_count"]),
            "identified_candidates": list(profile["identified_candidates"]),
            "route_margin_1_2": float(profile["route_margin_1_2"]),
            "rank1_is_required": bool(profile["ranking"] and profile["ranking"][0] == target),
            "target_node": target,
            "ranking": list(profile["ranking"]),
            "score_rows": list(profile["score_rows"]),
        })
    return rows


def _run_corpus_eval(
    arena,
    meta: Sequence[Mapping[str, Any]],
    *,
    rule: Mapping[str, Any],
    rule_path: Path,
    sink: Path,
) -> None:
    lookup = {row["probe_id"]: row for row in corpus100_fixture_rows()}
    ordered = list(lookup.values())
    for probe in ordered:
        started_ns = time.time_ns()
        target = int(probe["target_ordinal"])
        profile = route_profile(arena, probe["question"], exclude=set())
        plans = arm_plans(profile, rule)
        known_values = [
            str(row["value"]) for row in meta
            if row["family"] == probe["family"] and int(row["node_id"]) != target
        ]
        results = {}
        k3_fitted = None
        for arm in ARMS:
            fitted = fit_plan(arena, plans[arm]["rank_plan"])
            result = evaluate_attempt(
                arena,
                question=probe["question"],
                picks=fitted,
                ngen=40,
                accepted_values=probe["expected_values"],
                competing_values=known_values,
                required_nodes=[target],
            )
            results[arm] = result
            if arm == "A-k3":
                k3_fitted = list(fitted)
        if k3_fitted is None:
            raise LiveError("missing canonical A-k3 corpus replay")
        # Advance the registered sequential live-window fixture only after all
        # comparison arms have run and restored the identical pre-probe state.
        answer, info = arena._attempt(
            probe["question"], k3_fitted, 40, False, arena.stop_sequences)
        post_k3 = (str(answer), list(arena.cur_mounts), dict(info or {}))
        replay_equal = (
            post_k3[0] == results["A-k3"]["answer"]
            and post_k3[1] == results["A-k3"]["mounted_nodes"]
        )
        if not replay_equal:
            raise LiveError(f"corpus canonical A-k3 replay diverged: {probe['probe_id']}")
        row = build_probe_row(
            probe_id=probe["probe_id"],
            fixture="CORPUS-100",
            class_label="POINT-LOOKUP",
            split=probe["split"],
            question=probe["question"],
            profile=profile,
            plans=plans,
            results=results,
            required_nodes=[target],
            accepted_values=probe["expected_values"],
            rule_path=rule_path,
            started_ns=started_ns,
            extra={"canonical_a_k3_replay_equal": True},
        )
        append_jsonl_once(sink, row)
        print(
            f"corpus probe={probe['probe_id']} rank1={row['required_in_top1']} "
            f"k1={results['A-k1']['answer_correct']} k3={results['A-k3']['answer_correct']} "
            f"dec={results['A-DEC']['answer_correct']}",
            flush=True,
        )


def _run_supersession_eval(
    model,
    tokenizer,
    *,
    rule: Mapping[str, Any],
    rule_path: Path,
    sink: Path,
) -> None:
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
        # The registered fresh controls contain no lineage edges and are byte-
        # identical across the L2 flip; pin the production setting explicitly.
        revision_resolution=True,
    )
    node_to_idx = {}
    values = {}
    for node in fixture["nodes"]:
        index = int(arena.deposit(node["text"]))
        node_to_idx[node["node_id"]] = index
        values[index] = str(node["value"]).casefold()
        arena.grafts[index]["adm_fixture_role"] = node["role"]
    registered = {row["probe_id"].split(":", 1)[1]: row for row in supersession_fixture_rows()}
    for probe in fixture["probes"]:
        started_ns = time.time_ns()
        reg = registered[probe["probe_id"]]
        target = node_to_idx[probe["target_node"]]
        profile = route_profile(arena, probe["question"], exclude=set())
        plans = arm_plans(profile, rule)
        competing = [value for index, value in values.items() if index != target]
        results = {}
        k3_fitted = None
        for arm in ARMS:
            fitted = fit_plan(arena, plans[arm]["rank_plan"])
            result = evaluate_attempt(
                arena,
                question=probe["question"],
                picks=fitted,
                ngen=32,
                accepted_values=probe["expected_values"],
                competing_values=competing,
                required_nodes=[target],
            )
            results[arm] = result
            if arm == "A-k3":
                k3_fitted = list(fitted)
        if k3_fitted is None:
            raise LiveError("missing canonical A-k3 supersession replay")
        answer, info = arena._attempt(
            probe["question"], k3_fitted, 32, False, arena.stop_sequences)
        canonical = (str(answer), list(arena.cur_mounts), dict(info or {}))
        replay_equal = bool(
            canonical
            and canonical[0] == results["A-k3"]["answer"]
            and canonical[1] == results["A-k3"]["mounted_nodes"]
        )
        if not replay_equal:
            raise LiveError(f"supersession canonical replay diverged: {probe['probe_id']}")
        row = build_probe_row(
            probe_id=reg["probe_id"],
            fixture="SUPERSESSION-FRESH-CONTROLS",
            class_label="POINT-LOOKUP",
            split="eval",
            question=probe["question"],
            profile=profile,
            plans=plans,
            results=results,
            required_nodes=[target],
            accepted_values=probe["expected_values"],
            rule_path=rule_path,
            started_ns=started_ns,
            extra={"canonical_a_k3_replay_equal": True},
        )
        append_jsonl_once(sink, row)
        print(
            f"sup probe={probe['probe_id']} hits={row['identifier_hit_count']} "
            f"k1={results['A-k1']['answer']} k3={results['A-k3']['answer']}",
            flush=True,
        )


def run_minicpm(run_dir: Path, manifest_path: Path) -> None:
    sink = run_dir / "minicpm_rows.jsonl"
    if sink.exists() or list(run_dir.glob("decisiveness_rule_*.json")):
        raise LiveError("append-only MiniCPM stage already started")
    model, tokenizer, model_info, tc = _load_minicpm()
    print(f"loaded={model_info}", flush=True)
    try:
        arena, meta = _corpus_bank(model, tokenizer)
        fit_rows = _fit_rows(arena, meta)
        fit_receipt = {
            "schema": "grm.adm1.fit_receipt.v1",
            "phase": "heldout_before_eval",
            "created_unix_ns": time.time_ns(),
            "manifest": file_record(manifest_path),
            "source_provenance": {
                str(path.relative_to(ROOT)): file_record(path)
                for path in (
                    ROOT / "scripts" / "grm_adm1_analysis.py",
                    ROOT / "scripts" / "grm_adm1_gpu.py",
                    ROOT / "core" / "graft_arena.py",
                    ROOT / "tests" / "test_graft_corpus100.py",
                )
            },
            "rows": fit_rows,
        }
        fit_path = run_dir / "fit_receipt.json"
        write_exclusive_json(fit_path, fit_receipt)
        rule = freeze_rule(
            manifest_path=manifest_path,
            fit_receipt_path=fit_path,
            fit_rows=fit_rows,
        )
        rule_path = write_content_addressed(run_dir, "decisiveness_rule", rule)
        # fsync-by-close above is the registration boundary.  No readout or
        # eval-split route has run before this point.
        print(
            f"rule_frozen={rule_path} threshold={rule['margin_threshold']:.9g} "
            f"frozen_ns={rule['frozen_unix_ns']}",
            flush=True,
        )
        _run_corpus_eval(
            arena,
            meta,
            rule=rule,
            rule_path=rule_path,
            sink=sink,
        )
        restore_arena(arena, snapshot_arena(arena))
        del arena
        gc.collect()
        if hasattr(tc, "empty_cache"):
            tc.empty_cache()
        _run_supersession_eval(
            model,
            tokenizer,
            rule=rule,
            rule_path=rule_path,
            sink=sink,
        )
        marker = {
            "schema": "grm.adm1.stage_complete.v1",
            "stage": "minicpm",
            "completed_unix_ns": time.time_ns(),
            "row_count": len(sink.read_text(encoding="utf-8").splitlines()),
            "rule": file_record(rule_path),
        }
        write_exclusive_json(run_dir / "minicpm_complete.json", marker)
    finally:
        del model
        gc.collect()
        if hasattr(tc, "empty_cache"):
            tc.empty_cache()


def run_e2e_frame(
    run_dir: Path,
    frame: str,
    *,
    rule_path: Path,
    lease_seconds: int,
) -> None:
    frame_dir = run_dir / frame
    marker = frame_dir / "stage_complete.json"
    if marker.is_file():
        print(f"stage={frame} status=already_complete", flush=True)
        return
    session_dir = frame_dir / "session"
    # ADM1.2: a complete capture must be adjudicated from disk, not replayed.
    # The validator proves exact fixed-A-k3 anchor identity and exhaustively
    # projects every recorded candidate set through L2 before authorizing reuse.
    if completed_capture_available(run_dir, frame):
        alignment = validate_frame_receipts(run_dir, frame)
        alignment_path = write_content_addressed(
            frame_dir, "baseline_alignment", alignment)
        rows_path = frame_dir / "adm_rows.jsonl"
        write_exclusive_json(marker, {
            "schema": "grm.adm1.stage_complete.v1",
            "stage": frame,
            "completed_unix_ns": time.time_ns(),
            "row_count": len(read_jsonl(rows_path)),
            "rows_sha256": sha256_file(rows_path),
            "scorecard": file_record(session_dir / "probe_scorecard.json"),
            "baseline_alignment": file_record(alignment_path),
            "capture_reused_without_model_replay": True,
        })
        print(
            f"stage={frame} status=reused_complete_capture "
            f"alignment={alignment_path}",
            flush=True,
        )
        return
    command = [
        sys.executable,
        str(ROOT / "scripts" / "grm_adm1_e2e.py"),
        "--mode", "full",
        "--session-dir", str(session_dir),
        # The P4 fixture is the certified three-pass F-FULL replication;
        # DIAG and the original 34-turn E2E fixture retain their single-pass
        # frame. Admission is still the only within-frame arm difference.
        "--turn-pipeline", "three_pass" if frame == "p4" else "single",
        "--topk", "3",
        "--ngen", "32",
        "--max-trips", "1",
        "--live-turns", "2",
        "--restart-after", "17",
        "--no-probe-ladder",
        # ADM1.2 migration: L2 is production now.  Pin it explicitly so an
        # ambient escape env cannot silently return this frame to legacy mode.
        "--sup-resolve",
        "--skip-gpu-idle-check",
    ]
    if session_dir.exists():
        command.insert(2, "--resume")
    env = os.environ.copy()
    env.update({
        "GRM_ADM1_RUN_DIR": str(run_dir),
        "GRM_ADM1_FRAME": frame,
        "GRM_ADM1_RULE": str(rule_path),
        "GRM_ADM1_ENABLED": "1",
        "PYTHONUNBUFFERED": "1",
    })
    timeout = max(1, int(lease_seconds) - 8)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        start_new_session=True,
    )
    try:
        returncode = process.wait(timeout=timeout)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise
    # The generic E2E driver returns 2 for any semantic miss.  ADM frames have
    # a registered 7/9 fixed-A-k3 control, so status 2 is adjudicated below;
    # any other nonzero status remains an execution failure.
    if returncode not in (0, 2):
        raise LiveError(f"{frame} replay failed with status {returncode}")
    rows_path = frame_dir / "adm_rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    expected = 9 if frame == "diag" else 11
    if len(rows) != expected:
        raise LiveError(f"{frame} expected {expected} ADM rows, found {len(rows)}")
    alignment = validate_frame_receipts(run_dir, frame)
    alignment_path = write_content_addressed(
        frame_dir, "baseline_alignment", alignment)
    write_exclusive_json(marker, {
        "schema": "grm.adm1.stage_complete.v1",
        "stage": frame,
        "completed_unix_ns": time.time_ns(),
        "row_count": len(rows),
        "rows_sha256": sha256_file(rows_path),
        "scorecard": file_record(session_dir / "probe_scorecard.json"),
        "baseline_alignment": file_record(alignment_path),
        "replay_process_returncode": int(returncode),
        "capture_reused_without_model_replay": False,
    })


def cpu_selftest() -> dict[str, Any]:
    class Graft:
        def __init__(self, tokens: int):
            self.ntok = tokens

    class Arena:
        width = 10
        grafts = [{"ntok": 3}, {"ntok": 8}, {"ntok": 4}]

    assert fit_plan(Arena(), [0, 1, 2]) == [0, 2]
    rule = {"margin_threshold": 0.2}
    profile = {
        "ranking": [2, 0, 1],
        "identified_candidates": [2],
        "route_margin_1_2": 0.1,
    }
    plans = arm_plans(profile, rule)
    assert plans["A-DEC"]["rank_plan"] == [2]
    lease_guard = False
    try:
        with gpu_lease(MAX_LEASE_SECONDS + 1, 0):
            pass
    except LiveError:
        lease_guard = True
    if not lease_guard:
        raise LiveError("lease cap guard failed")
    return {
        "schema": "grm.adm1.gpu_runner_cpu_selftest.v1",
        "status": "PASS",
        "checks": {
            "budget_fit_rank_then_physical_order": "PASS",
            "policy_plan_bridge": "PASS",
            "lease_cap_guard": "PASS",
            "live_stage_order": list(LIVE_STAGES),
        },
    }


def _new_run_dir(base: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"{os.getpid():x}"
    path = base / f"run_{stamp}_{suffix}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def orchestrate(args: argparse.Namespace) -> int:
    manifest = _one_manifest(args.manifest)
    assert_production_frozen(manifest)
    run_dir = args.run_dir.resolve() if args.run_dir else _new_run_dir(args.artifact_dir.resolve())
    run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = run_dir / "run_manifest.json"
    if not run_manifest.exists():
        write_exclusive_json(run_manifest, {
            "schema": "grm.adm1.live_run.v1",
            "created_unix_ns": time.time_ns(),
            "manifest": file_record(manifest),
            "stage_order": list(LIVE_STAGES),
            "gpu": str(args.gpu),
            "lease_seconds": int(args.lease_seconds),
            "lock_wait_seconds": int(args.lock_wait_seconds),
            "lock_path": str(LOCK_PATH),
            "supersession_frame": "production_l2_on_explicit",
            "source_provenance": {
                str(path.relative_to(ROOT)): file_record(path)
                for path in (
                    ROOT / "orders" / "GRM_ADM1_K_POLICY.md",
                    ROOT / "scripts" / "grm_adm1_analysis.py",
                    ROOT / "scripts" / "grm_adm1_gpu.py",
                    ROOT / "scripts" / "grm_adm1_e2e.py",
                    ROOT / "scripts" / "grm_e2e_session.py",
                    ROOT / "core" / "graft_arena.py",
                )
            },
        })
    print(f"run_dir={run_dir}", flush=True)
    for index, stage in enumerate(LIVE_STAGES):
        complete = (
            run_dir / "minicpm_complete.json"
            if stage == "minicpm" else run_dir / stage / "stage_complete.json"
        )
        if complete.is_file():
            print(f"stage={stage} status=already_complete", flush=True)
            continue
        if index:
            print(f"inter_stage_gap_s={GAP_SECONDS}", flush=True)
            time.sleep(GAP_SECONDS)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--stage", stage,
            "--run-dir", str(run_dir),
            "--manifest", str(manifest),
            "--gpu", str(args.gpu),
            "--lease-seconds", str(args.lease_seconds),
            "--lock-wait-seconds", str(args.lock_wait_seconds),
        ]
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode != 0:
            print(f"stage={stage} status=FAILED code={completed.returncode}", flush=True)
            return completed.returncode
    command = [
        sys.executable,
        str(ROOT / "scripts" / "grm_adm1_analysis.py"),
        "summarize",
        "--run-dir", str(run_dir),
        "--manifest", str(manifest),
        "--append-report",
    ]
    return subprocess.run(command, cwd=ROOT).returncode


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "selftest", *LIVE_STAGES), default="all")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    parser.add_argument("--lock-wait-seconds", type=int, default=DEFAULT_LOCK_WAIT_SECONDS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "selftest":
        value = cpu_selftest()
        path = write_content_addressed(args.artifact_dir.resolve(), "gpu_runner_cpu_selftest", value)
        print(f"receipt={path}")
        print("status=PASS")
        return 0
    if int(args.lease_seconds) > MAX_LEASE_SECONDS:
        raise LiveError(f"lease cap exceeds {MAX_LEASE_SECONDS}s")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.stage == "all":
        return orchestrate(args)
    if args.run_dir is None:
        raise LiveError("live child stage requires --run-dir")
    run_dir = args.run_dir.resolve()
    manifest = _one_manifest(args.manifest)
    assert_production_frozen(manifest)
    if args.stage != "minicpm" and completed_capture_available(run_dir, args.stage):
        # Receipt-only reconciliation is CPU-safe and should not queue behind a
        # GPU job or reload a model merely to rediscover a complete session.
        run_e2e_frame(
            run_dir,
            args.stage,
            rule_path=_one_rule(run_dir),
            lease_seconds=int(args.lease_seconds),
        )
        return 0
    with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
        if args.stage == "minicpm":
            run_minicpm(run_dir, manifest)
        else:
            run_e2e_frame(
                run_dir,
                args.stage,
                rule_path=_one_rule(run_dir),
                lease_seconds=int(args.lease_seconds),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
