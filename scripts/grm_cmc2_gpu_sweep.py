#!/usr/bin/env python3
"""Guarded live GRM-CMC2 depth sweep (GPU handoff; CPU self-test included).

The ``all`` orchestrator launches one fresh child per stage.  Each child
holds the shared flock for at most 580 seconds, one GPU is visible, and a
30-second idle gap separates stages.  The runner reuses the exact CMC1.2
engine-softmax-operand tap and the CMC1 supersession fresh-control fixture.

Production frame is pinned in every receipt: probe ladder ON and L2 ON.
The CMC direct-read arm intentionally retains CMC1's registered explicit
mount order; the probe-ladder setting therefore has no branch to alter in
that arm.  The supersession leg uses the existing battery harness with L2
enabled and records whether v2 wins while v1 is excluded.

Bare lead invocation (no tee/pipeline):
``PYTHONPATH=/mnt/ForgeRealm/GraftRepository python3 scripts/grm_cmc2_gpu_sweep.py``
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from grm_cmc1_gpu_arms import (  # noqa: E402
    G2_TOLERANCE,
    LiveFixture,
    SDPAInterceptor,
    gpu_lease,
)
from grm_cmc1_mechanism import (  # noqa: E402
    canonical_json_bytes,
    sha256_file,
    summarize_attention_capture,
    write_content_addressed,
)
from grm_cmc2_depth_sweep import (  # noqa: E402
    ADDENDUM,
    DEPTHS,
    DEPTH_LABELS,
    ORDER,
    PRODUCTION_FRAME,
    CMC2Error,
    read_outcome,
    roundtrip_arrays,
    summarize_mass_split,
)


DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "grm_cmc2"
CMC_FIXTURE = (
    ROOT / "tests" / "fixtures" / "supersession_battery"
    / "fresh_fact_controls.json"
)
SUP_FIXTURE = (
    ROOT / "tests" / "fixtures" / "supersession_battery"
    / "short_correction_long_competitor.json"
)
SUP_HARNESS = ROOT / "tests" / "test_grm_supersession_battery.py"
LOCK_PATH = Path("/tmp/forge-gpu.lock")
MAX_LEASE_SECONDS = 590
DEFAULT_LEASE_SECONDS = 580
GAP_SECONDS = 30
BASE_STAGES = ("g0", "g1")
DEPTH_STAGES = tuple(f"depth_{label}" for label in DEPTH_LABELS)
SUP_STAGES = tuple(f"sup_{label}" for label in DEPTH_LABELS)
STAGES = BASE_STAGES + tuple(
    stage for label in DEPTH_LABELS
    for stage in (f"depth_{label}", f"sup_{label}")
)


class LiveError(CMC2Error):
    pass


def _load_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise LiveError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _assert_production_frame() -> None:
    from core.grm_supersession import sup_resolve_enabled
    from grm_probe_ladder import probe_ladder_enabled

    explicit = argparse.Namespace(probe_ladder=True)
    if probe_ladder_enabled(explicit) is not True:
        raise LiveError("explicit production probe-ladder pin did not resolve ON")
    if sup_resolve_enabled(True) is not True:
        raise LiveError("explicit production L2 pin did not resolve ON")


def _source_provenance() -> dict[str, Any]:
    paths = (
        ORDER,
        ADDENDUM,
        Path(__file__).resolve(),
        ROOT / "scripts" / "grm_cmc2_depth_sweep.py",
        ROOT / "scripts" / "grm_cmc1_gpu_arms.py",
        ROOT / "scripts" / "grm_cmc1_mechanism.py",
        ROOT / "core" / "graft_quant.py",
        ROOT / "core" / "graft_arena.py",
        ROOT / "core" / "minicpm3_tc.py",
        CMC_FIXTURE,
        SUP_FIXTURE,
        SUP_HARNESS,
    )
    return {
        str(path.relative_to(ROOT)): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
    }


def _stage_base(stage: str, model_info: Any) -> dict[str, Any]:
    return {
        "schema": f"grm.cmc2.{stage}.v1",
        "order": "GRM-CMC2",
        "stage": stage,
        "frame": dict(PRODUCTION_FRAME),
        "frame_application": {
            "probe_ladder": (
                "explicitly pinned ON; CMC direct-order arms have no driver "
                "ladder branch, supersession uses the registered battery task path"
            ),
            "supersession_l2": "explicitly pinned ON in every ArenaCache",
        },
        "model": "MiniCPM3",
        "model_info": str(model_info),
        "dialect": "mla",
        "runtime": {
            "arena_width": 256,
            "route_layer": 44,
            "topk": 3,
            "live_turns": 0,
            "ngen": 32,
            "attention_mode": "standard",
        },
        "provenance": _source_provenance(),
    }


def _normalize_answer(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "probe_id": row["probe_id"],
        "answer_text": row["answer_text"],
        "classification": row["classification"],
        "classification_match": row.get("classification_match"),
        "ranking_nodes": row.get("ranking_nodes"),
        "mounted_nodes": row.get("mounted_nodes"),
        "mounted_indices": row.get("mounted_indices"),
    }


def _unique_stage(run_dir: Path, stem: str) -> tuple[Path, dict[str, Any]]:
    matches = sorted(run_dir.glob(f"{stem}_*.json"))
    if len(matches) != 1:
        raise LiveError(
            f"expected one {stem}_*.json under {run_dir}, found {len(matches)}")
    value = json.loads(matches[0].read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LiveError(f"stage receipt must be an object: {matches[0]}")
    return matches[0], value


def _require_gates(run_dir: Path, *, g1: bool = True) -> tuple[dict[str, Any], dict[str, Any] | None]:
    _g0_path, g0 = _unique_stage(run_dir, "g0")
    if not (
        g0.get("fixture_status") == "REPRODUCES"
        and g0.get("fresh_control_correct") == 1
        and g0.get("fresh_control_total") == 2
    ):
        raise LiveError("CMC2-G0 does not authorize the sweep")
    if not g1:
        return g0, None
    _g1_path, g1_receipt = _unique_stage(run_dir, "g1")
    if g1_receipt.get("byte_identity") is not True:
        raise LiveError("CMC2-G1 does not authorize the sweep")
    return g0, g1_receipt


def _depth_bits(label: str) -> int:
    try:
        return dict(DEPTHS)[label]
    except KeyError as exc:
        raise LiveError(f"unknown depth {label!r}") from exc


def _install_depth(live: Any, storage_bits: int) -> tuple[list[Any], dict[str, Any]]:
    reconstructed_h = []
    graft_receipts = {}
    all_pass = True
    for idx, original_h in enumerate(live.original_h):
        arrays = live.arena.pack_node(original_h)
        rebuilt, receipt = roundtrip_arrays(arrays, int(storage_bits))
        new_h = live.arena.unpack_node(rebuilt)
        reconstructed_h.append(new_h)
        node_id = live.idx_to_node[int(idx)]
        graft_receipts[node_id] = receipt
        all_pass = all_pass and bool(receipt["pass"])
    result = {
        "schema": "grm.cmc2.roundtrip_set.v1",
        "frame": dict(PRODUCTION_FRAME),
        "storage_bits": int(storage_bits),
        "group_size": 32,
        "grafts": graft_receipts,
        "pass": bool(all_pass),
    }
    return reconstructed_h, result


def _activate_depth(live: Any, reconstructed_h: list[Any]) -> list[Any]:
    original_h = live.original_h
    live.original_h = reconstructed_h
    for idx, h in enumerate(reconstructed_h):
        live.arena.grafts[idx]["h"] = h
    return original_h


def _restore_original(live: Any, original_h: list[Any]) -> None:
    live.original_h = original_h
    for idx, h in enumerate(original_h):
        live.arena.grafts[idx]["h"] = h


def run_g0(live: LiveFixture) -> dict[str, Any]:
    _assert_production_frame()
    live.arena.revision_resolution = True
    rows = live.run_default_fixture()
    normalized = [_normalize_answer(row) for row in rows]
    correct = sum(row["classification"] == "correct" for row in rows)
    praxis = next(row for row in rows if row["probe_id"] == "praxis_fresh")
    reproduces = bool(
        correct == 1
        and praxis["classification"] == "wrong-fact"
        and "praxis_fact" in praxis["mounted_nodes"]
        and "solace_fact" in praxis["mounted_nodes"]
    )
    result = _stage_base("g0", live.model_info)
    result.update({
        "gate": "CMC2-G0",
        "fixture": "supersession_fresh_control",
        "fixture_status": "REPRODUCES" if reproduces else "HEALED_EXCLUDED",
        "fresh_control_correct": int(correct),
        "fresh_control_total": len(rows),
        "arms_authorized": reproduces,
        "registered_baseline_payload_sha256": hashlib.sha256(
            canonical_json_bytes(normalized)).hexdigest(),
        "transcript": normalized,
    })
    return result


def run_g1(live: LiveFixture, g0: dict[str, Any]) -> dict[str, Any]:
    _assert_production_frame()
    live.arena.revision_resolution = True
    rows = [_normalize_answer(row) for row in live.run_default_fixture()]
    replay_hash = hashlib.sha256(canonical_json_bytes(rows)).hexdigest()
    baseline_hash = str(g0["registered_baseline_payload_sha256"])
    result = _stage_base("g1", live.model_info)
    result.update({
        "gate": "CMC2-G1",
        "flags": {
            "storage_depth_intervention": False,
            "packed_payload_rehydrate": False,
            "engine_operand_witness": False,
            "mount_order_override": False,
        },
        "registered_baseline_payload_sha256": baseline_hash,
        "flags_off_replay_payload_sha256": replay_hash,
        "byte_identity": replay_hash == baseline_hash,
        "transcript": rows,
    })
    return result


def run_cmc_depth(live: LiveFixture, label: str) -> dict[str, Any]:
    _assert_production_frame()
    live.arena.revision_resolution = True
    bits = _depth_bits(label)
    reconstructed_h, roundtrip = _install_depth(live, bits)
    original_h = _activate_depth(live, reconstructed_h)
    try:
        interceptor = SDPAInterceptor("capture")
        praxis = live.run_attempt(
            "praxis_fresh", list(live.baseline_order), interceptor=interceptor)
        queries = {
            layer: np.stack(values, axis=0)
            for layer, values in interceptor.queries.items()
        }
        operands = {
            layer: np.stack(values, axis=0)
            for layer, values in interceptor.softmax_operands.items()
        }
        attention = summarize_attention_capture(
            queries,
            interceptor.keys,
            live.mount_ranges(live.baseline_order, absolute=False),
            softmax_operands_by_layer=operands,
            target="praxis_fact",
            sibling="solace_fact",
        )
        mass_split = summarize_mass_split(
            attention, target="praxis_fact", sibling="solace_fact")
        if interceptor.g2 is None:
            raise LiveError(f"{label}: engine-operand witness produced no G2 receipt")
        solace = live.run_attempt("solace_fresh", list(live.baseline_order))
    finally:
        _restore_original(live, original_h)
        del reconstructed_h

    result = _stage_base(f"depth_{label}", live.model_info)
    result.update({
        "depth": label,
        "storage_bits": bits,
        "storage_label": "bf16_plain_fp16_at_rest" if bits == 16 else f"INT{bits}",
        "fixture": "supersession_fresh_control",
        "mount_time_transform": "pack_then_unpack_all_mounted_grafts",
        "read": {
            "outcome": read_outcome(str(praxis["classification"])),
            "praxis": praxis,
            "solace": solace,
        },
        "mass_split": mass_split,
        "key_norms": attention["key_norm_distributions"],
        "attention": attention,
        "roundtrip": roundtrip,
        "engine_operand_witness": interceptor.g2,
    })
    return result


class RevisionFixture:
    """Existing v1/v2 battery fixture under explicit production L2."""

    def __init__(self):
        sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
        from tokenizers import Tokenizer as HFTok
        from core.graft_arena import ArenaCache
        from core.minicpm3_tc import MiniCPM3_TC, _snap

        self.harness = _load_file_module(
            "grm_cmc2_supersession_harness", SUP_HARNESS)
        self.fixture = json.loads(SUP_FIXTURE.read_text(encoding="utf-8"))
        self.harness.validate_fixture(self.fixture, str(SUP_FIXTURE))
        self.model, self.model_info = MiniCPM3_TC.from_pretrained()
        self.tokenizer = HFTok.from_file(str(Path(_snap()) / "tokenizer.json"))
        self.arena = ArenaCache(
            self.model,
            encode=lambda text: self.tokenizer.encode(text).ids,
            decode=lambda ids: self.tokenizer.decode(ids),
            sink_text="<conversation>\n",
            arena_width=256,
            route_layer=44,
            topk=3,
            live_turns=0,
            cache_deposits=False,
            length_debias=False,
            revision_resolution=True,
            # Preserve the registered CMC co-mount depth-sweep frame.
            decisive_admission=False,
        )
        self.node_to_idx = self.harness._install_fixture_nodes(
            self.arena, self.fixture)
        self.idx_to_node = {
            int(idx): node_id for node_id, idx in self.node_to_idx.items()
        }
        self.original_h = [graft["h"] for graft in self.arena.grafts]
        self.probe = self.fixture["probes"][0]

    def reset(self) -> None:
        from core import kv_graft

        kv_graft.clear_injection(self.model)
        self.arena.caches = None
        self.arena.pos = 0
        self.arena.live_segs = []
        self.arena.cur_mounts = []
        self.arena.cur_mount_n = 0
        for idx, h in enumerate(self.original_h):
            self.arena.grafts[idx]["h"] = h
        for layer in self.model.layers:
            attn = layer.self_attn
            for name in ("_captured_q", "_captured"):
                if hasattr(attn, name):
                    delattr(attn, name)

    def run(self) -> dict[str, Any]:
        self.reset()
        question = self.probe["question"]
        ranking = list(self.arena.route(
            question, exclude=set(), limit=len(self.arena.grafts)))
        route_backend = self.arena.last_route_backend
        answer, info = self.arena.step(
            question, ngen=32, deposit=False, max_trips=0)
        classification, matched = self.harness._classify_answer(answer, self.probe)
        mounted_indices = [int(value) - 1 for value in info.get("mounts", ())]
        mounted_nodes = [
            self.idx_to_node.get(idx, f"graft:{idx}") for idx in mounted_indices
        ]
        target = self.probe["target_node"]
        stale_nodes = list(self.probe["stale_nodes"])
        target_idx = self.node_to_idx[target]
        stale_indices = [self.node_to_idx[node] for node in stale_nodes]

        def rank(idx: int) -> int | None:
            try:
                return ranking.index(int(idx)) + 1
            except ValueError:
                return None

        stale_ranks = {
            node: rank(idx) for node, idx in zip(stale_nodes, stale_indices)
        }
        l2_excluded = all(idx not in mounted_indices for idx in stale_indices)
        v2_wins = bool(
            classification == "correct"
            and target_idx in mounted_indices
            and l2_excluded
        )
        return {
            "fixture": SUP_FIXTURE.name,
            "probe_id": self.probe["probe_id"],
            "question": question,
            "classification": classification,
            "classification_match": matched,
            "answer_text": answer,
            "route_backend": route_backend,
            "ranking_nodes": [
                self.idx_to_node.get(idx, f"graft:{idx}") for idx in ranking
            ],
            "v2_node": target,
            "v2_rank": rank(target_idx),
            "v1_nodes": stale_nodes,
            "v1_ranks": stale_ranks,
            "mounted_nodes": mounted_nodes,
            "mounted_indices": mounted_indices,
            "l2_excluded_all_v1": l2_excluded,
            "v2_outranks_v1": v2_wins,
            "v2_outranks_v1_definition": (
                "production-L2 task read returns current v2 with v2 mounted "
                "and every declared v1 ancestor excluded"
            ),
            "route_index_quantized": False,
        }

    def close(self) -> None:
        self.reset()
        del self.original_h
        del self.arena
        del self.model
        gc.collect()
        try:
            from core.mistral7b_tc import tc
            tc.empty_cache()
        except Exception:
            pass


def run_sup_depth(live: RevisionFixture, label: str) -> dict[str, Any]:
    _assert_production_frame()
    if live.arena.revision_resolution is not True:
        raise LiveError("supersession fixture is not pinned L2 ON")
    bits = _depth_bits(label)
    reconstructed_h, roundtrip = _install_depth(live, bits)
    original_h = _activate_depth(live, reconstructed_h)
    try:
        supersession = live.run()
    finally:
        _restore_original(live, original_h)
        del reconstructed_h
    result = _stage_base(f"sup_{label}", live.model_info)
    result.update({
        "depth": label,
        "storage_bits": bits,
        "fixture": "short_correction_long_competitor",
        "supersession": supersession,
        "roundtrip": roundtrip,
    })
    return result


def _run_stage(stage: str, run_dir: Path) -> tuple[Path, dict[str, Any]]:
    if stage == "g0":
        live = LiveFixture()
        try:
            result = run_g0(live)
        finally:
            live.close()
    elif stage == "g1":
        g0, _unused = _require_gates(run_dir, g1=False)
        live = LiveFixture()
        try:
            result = run_g1(live, g0)
        finally:
            live.close()
    elif stage.startswith("depth_"):
        _require_gates(run_dir)
        label = stage.removeprefix("depth_")
        live = LiveFixture()
        try:
            result = run_cmc_depth(live, label)
        finally:
            live.close()
    elif stage.startswith("sup_"):
        _require_gates(run_dir)
        label = stage.removeprefix("sup_")
        live = RevisionFixture()
        try:
            result = run_sup_depth(live, label)
        finally:
            live.close()
    else:
        raise LiveError(f"unknown stage {stage!r}")
    path = write_content_addressed(run_dir, stage, result)
    return path, result


def _new_run_dir(base: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = base.resolve() / f"live_sweep_{stamp}_{os.getpid()}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _validate_run_dir(path: Path, artifact_dir: Path) -> Path:
    root = artifact_dir.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise LiveError(f"run dir must remain under {root}: {resolved}")
    if not resolved.is_dir():
        raise LiveError(f"run dir does not exist: {resolved}")
    return resolved


def cpu_self_test() -> dict[str, Any]:
    _assert_production_frame()
    cmc = json.loads(CMC_FIXTURE.read_text(encoding="utf-8"))
    sup = json.loads(SUP_FIXTURE.read_text(encoding="utf-8"))
    if [probe["probe_id"] for probe in cmc["probes"]] != [
        "praxis_fresh", "solace_fresh"
    ]:
        raise LiveError("CMC fixture registration drifted")
    if sup["probes"][0]["probe_id"] != "orion_current":
        raise LiveError("supersession v1/v2 fixture registration drifted")
    if tuple(label for label, _bits in DEPTHS) != DEPTH_LABELS:
        raise LiveError("depth registration drifted")
    expected_stages = BASE_STAGES + tuple(
        stage for label in DEPTH_LABELS
        for stage in (f"depth_{label}", f"sup_{label}")
    )
    if STAGES != expected_stages:
        raise LiveError("stage order drifted")
    try:
        with gpu_lease(MAX_LEASE_SECONDS + 1, 0):
            pass
    except Exception:
        lease_guard = True
    else:
        lease_guard = False
    if not lease_guard:
        raise LiveError("GPU lease cap guard failed")
    result = {
        "schema": "grm.cmc2.gpu_runner.cpu_selftest.v1",
        "frame": dict(PRODUCTION_FRAME),
        "status": "PASS",
        "cmc_fixture": CMC_FIXTURE.name,
        "supersession_fixture": SUP_FIXTURE.name,
        "depth_registration": list(DEPTH_LABELS),
        "stage_registration": list(STAGES),
        "g2_tolerance": G2_TOLERANCE,
        "lease_cap_guard": "PASS",
        "gpu_imported_or_used": False,
    }
    return result


def _orchestrate(args: argparse.Namespace) -> int:
    run_dir = _new_run_dir(args.artifact_dir)
    print(f"run_dir={run_dir}", flush=True)
    for index, stage in enumerate(STAGES):
        if index:
            print(f"gpu_gap_s={GAP_SECONDS}", flush=True)
            time.sleep(GAP_SECONDS)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--stage", stage,
            "--run-dir", str(run_dir),
            "--artifact-dir", str(args.artifact_dir),
            "--gpu", str(args.gpu),
            "--lease-seconds", str(args.lease_seconds),
            "--lock-wait-seconds", str(args.lock_wait_seconds),
        ]
        completed = subprocess.run(cmd, check=False)
        if completed.returncode:
            print(
                f"stage_failed={stage} returncode={completed.returncode}",
                flush=True,
            )
            return completed.returncode
    adjudicate = [
        sys.executable,
        str(ROOT / "scripts" / "grm_cmc2_depth_sweep.py"),
        "adjudicate",
        "--run-dir", str(run_dir),
        "--append-report",
    ]
    return subprocess.run(adjudicate, check=False).returncode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all",) + STAGES, default="all")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    parser.add_argument("--lock-wait-seconds", type=int, default=7200)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        result = cpu_self_test()
        path = write_content_addressed(
            args.artifact_dir.resolve(), "gpu_runner_cpu_selftest", result)
        print(canonical_json_bytes(result).decode("utf-8"), end="")
        print(f"receipt={path}")
        return 0
    if args.stage == "all":
        if args.run_dir is not None:
            raise LiveError("--run-dir is child-stage-only; all creates a new run")
        return _orchestrate(args)
    if args.run_dir is None:
        raise LiveError("a child stage requires --run-dir")
    if not (0 < int(args.lease_seconds) <= MAX_LEASE_SECONDS):
        raise LiveError(f"lease must be in 1..{MAX_LEASE_SECONDS}s")
    gpu = str(args.gpu).strip()
    if not gpu or "," in gpu:
        raise LiveError("--gpu must name exactly one GPU index")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    run_dir = _validate_run_dir(args.run_dir, args.artifact_dir)
    if list(run_dir.glob(f"{args.stage}_*.json")):
        raise LiveError(f"append-only stage already exists: {args.stage}")
    with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
        path, result = _run_stage(args.stage, run_dir)
    print(canonical_json_bytes({
        "stage": args.stage,
        "receipt": str(path),
        "schema": result["schema"],
        "frame": result["frame"],
    }).decode("utf-8"), end="")
    if args.stage == "g0" and not result.get("arms_authorized"):
        return 3
    if args.stage == "g1" and not result.get("byte_identity"):
        return 4
    if args.stage.startswith("depth_"):
        if not result.get("roundtrip", {}).get("pass"):
            return 5
        if not result.get("engine_operand_witness", {}).get("pass"):
            return 6
    if args.stage.startswith("sup_") and not result.get("roundtrip", {}).get("pass"):
        return 7
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
