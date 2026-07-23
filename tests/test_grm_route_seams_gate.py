#!/usr/bin/env python3
"""GPU-only E3 gate for the GQA ragged-bank route seams.

This module deliberately keeps model/CUDA imports inside ``run_gpu_gate`` so
the ordinary CPU pytest collection remains safe.  The lead runs the gate as:

    python3 tests/test_grm_route_seams_gate.py --run-gpu

It creates a Qwen3-4B GQA arena with at least 37 resident document grafts,
warms both control and flagged route paths, then records an exact Python
control ranking and an adjacent flagged ``step()`` route for every capture.
The emitted JSON is an E3 receipt, not a claim that the GPU gate was run by
the CPU suite.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "grm_route_seams_e3_v1"
TOPKS = (1, 3, 5, 16)


def _route_wall_ms(receipt):
    terms = dict(receipt.get("terms_ms") or {})
    return float(terms["route_wall_ms"])


def _require_route_receipt(receipt):
    if not isinstance(receipt, dict):
        raise AssertionError("route profile did not emit a receipt")
    if receipt.get("schema") != "grm_route_seams_route_v1":
        raise AssertionError(f"unexpected route receipt schema: {receipt!r}")
    if "route_wall_ms" not in dict(receipt.get("terms_ms") or {}):
        raise AssertionError("route receipt is missing route_wall_ms")
    return receipt


def _assert_topk_parity(python_order, cuda_order):
    for topk in TOPKS:
        if list(python_order[:topk]) != list(cuda_order[:topk]):
            raise AssertionError(
                f"route rank parity failed at k={topk}: "
                f"python={python_order[:topk]} cuda={cuda_order[:topk]}")


def _summary(receipt):
    """Keep full timing terms while making the capture rows easy to scan."""
    receipt = _require_route_receipt(receipt)
    return {
        "route_backend": receipt.get("route_backend"),
        "requested_backend": receipt.get("requested_backend"),
        "route_wall_ms": _route_wall_ms(receipt),
        "ranking_count": int(receipt.get("ranking_count", 0)),
        "ranking_prefix": list(receipt.get("ranking_prefix") or ()),
        "parity": dict(receipt.get("parity") or {}),
        "fallbacks": list(receipt.get("fallbacks") or ()),
        "terms_ms": dict(receipt.get("terms_ms") or {}),
        "rebuilds": dict(receipt.get("rebuilds") or {}),
    }


def _node_text(index):
    """Ragged, lexical-disjoint source text for the raw-key route gate."""
    lengths = ("amber", "velvet", "cinder", "saffron", "meridian",
               "lattice", "orchard")
    words = " ".join(lengths[: 2 + (index % len(lengths))])
    return (
        f"Archive parcel {index} records {words}. "
        "Its catalog entry is retained for arena route-bank residency.")


def _probe_text(_turn):
    # None of these lowercase content words occur in _node_text().  That
    # leaves the query-side lexical extension enabled while proving it is
    # silent, rather than silently changing the production lexical policy.
    return "Could the quasar hyacinth protocol be summarized?"


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-gpu", action="store_true",
                        help="acknowledge the CUDA/model-loading gate")
    parser.add_argument("--node-count", type=int, default=37)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--warmup-turns", type=int, default=1)
    parser.add_argument("--arena-width", type=int, default=256)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--ngen", type=int, default=1)
    parser.add_argument("--max-route-ms", type=float, default=5.0)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("artifacts/grm_route_seams/e3_qwen3_4b.json"),
    )
    args = parser.parse_args(argv)
    if args.node_count < 37:
        parser.error("--node-count must be at least 37 for the registered E3 gate")
    if args.turns < 1:
        parser.error("--turns must be positive")
    if args.warmup_turns < 1:
        parser.error("--warmup-turns must be at least one (first-run law)")
    if args.topk < max(TOPKS):
        parser.error(f"--topk must be at least {max(TOPKS)} for k={TOPKS}")
    return args


def _load_qwen3_4b():
    """Mirror tests/test_graft_gqa_arena.py without import-time CUDA work."""
    tensor_cuda_root = "/mnt/ForgeRealm/Project-Tensor/tensor_cuda"
    if tensor_cuda_root not in sys.path:
        sys.path.insert(0, tensor_cuda_root)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from core.qwen3_tc import Qwen3_TC, _snap
    from tokenizers import Tokenizer as HFTok

    tokenizer = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
    model, model_info = Qwen3_TC.from_pretrained()
    for layer in model.layers:
        layer.self_attn.quant_kv_cache = False
    encode = lambda text: tokenizer.encode(text).ids
    decode = lambda ids: tokenizer.decode(
        ids, clean_up_tokenization_spaces=False)
    return model, model_info, encode, decode


def _route_control(arena, question, exclude, *, backend, parity_check):
    arena.set_route_backend(backend)
    arena.route_profile = True
    arena.route_parity_check = bool(parity_check)
    ranking = arena.route(question, exclude=exclude, limit=16)
    receipt = _require_route_receipt(arena.last_route_receipt)
    return list(ranking), receipt


def _new_receipt(args):
    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": "Qwen3-4B",
            "node_count": int(args.node_count),
            "turns": int(args.turns),
            "warmup_turns": int(args.warmup_turns),
            "arena_width": int(args.arena_width),
            "topk": int(args.topk),
            "ngen": int(args.ngen),
            "topks": list(TOPKS),
            "registered_max_route_ms": float(args.max_route_ms),
        },
        "warmup": [],
        "captures": [],
        "gate": {"passed": False, "violations": []},
    }


def run_gpu_gate(args):
    """Run the E3 session and return a receipt even for a RED outcome."""
    receipt = _new_receipt(args)
    repo = None
    try:
        model, model_info, encode, decode = _load_qwen3_4b()
        receipt["model_info"] = str(model_info)

        from core.graft_arena import GQAArenaCache
        from core.graft_repository import GraftRepository
        from scripts import grm_router_baseline as baseline

        with tempfile.TemporaryDirectory(prefix="grm-route-seams-e3-") as td:
            temp_root = Path(td)
            native_dir = temp_root / "native"
            native_dir.mkdir()
            native_lib = baseline.build_native_lib(
                native_dir, cxx=args.cxx, openmp=True)
            repo = GraftRepository(
                model, encode, decode, str(temp_root / "arena_repo"),
                autosave=False,
                librarian_mode="deferred",
                arena_cls=GQAArenaCache,
                native_lib_path=str(native_lib),
                native_auto=False,
                sink_text="<conversation>\n",
                arena_width=int(args.arena_width),
                route_layer=0,
                topk=int(args.topk),
                live_turns=2,
                cache_deposits=False,
                route_backend="python",
                route_profile=True,
            )
            arena = repo.arena
            for node_index in range(int(args.node_count)):
                repo.add_document(_node_text(node_index), tags=("route-seams",))

            resident = [g for g in arena.grafts if not g.get("retired")]
            if len(resident) < int(args.node_count):
                raise AssertionError(
                    f"only {len(resident)} resident nodes after setup; "
                    f"need {args.node_count}")
            receipt["setup"] = {
                "resident_nodes": len(resident),
                "total_grafts": len(arena.grafts),
                "route_bank_epoch": int(getattr(arena, "_cuda_gqa_epoch", 0)),
            }

            # First-run law: populate Python caches and attach/upload the
            # immutable ragged CUDA bank before any timing capture.
            for warmup_turn in range(int(args.warmup_turns)):
                question = _probe_text(warmup_turn)
                exclude = {idx for idx, _ in arena.live_segs if idx is not None}
                python_order, python_receipt = _route_control(
                    arena, question, exclude, backend="python", parity_check=False)
                cuda_order, cuda_receipt = _route_control(
                    arena, question, exclude, backend="cuda_ragged",
                    parity_check=True)
                _assert_topk_parity(python_order, cuda_order)
                if arena.last_route_backend != "cuda":
                    raise AssertionError(
                        f"warmup flagged route used {arena.last_route_backend!r}")
                receipt["warmup"].append({
                    "turn": int(warmup_turn),
                    "python": _summary(python_receipt),
                    "cuda": _summary(cuda_receipt),
                })

            for turn in range(int(args.turns)):
                question = _probe_text(turn)
                exclude = {idx for idx, _ in arena.live_segs if idx is not None}
                # Python's forced control and CUDA's parity-gated preflight
                # operate on the exact same arena state and bare question.
                python_order, python_receipt = _route_control(
                    arena, question, exclude, backend="python", parity_check=False)
                cuda_order, cuda_preflight = _route_control(
                    arena, question, exclude, backend="cuda_ragged",
                    parity_check=True)
                _assert_topk_parity(python_order, cuda_order)
                if arena.last_route_backend != "cuda":
                    raise AssertionError(
                        f"flagged preflight used {arena.last_route_backend!r}")

                # The preflight performed the expensive Python comparison.
                # Disable it for the timed real step so its route receipt
                # measures the resident flagged path, not a double-scoring
                # debug mode. It is still profiled term-by-term.
                arena.route_parity_check = False
                arena.route_profile = True
                arena.set_route_backend("cuda_ragged")
                step_started = time.perf_counter_ns()
                _answer, info = arena.step(
                    question, ngen=int(args.ngen), deposit=False, max_trips=0)
                step_wall_ms = (time.perf_counter_ns() - step_started) / 1.0e6
                step_receipt = _require_route_receipt(info.get("route_receipt"))
                if info.get("route_backend") != "cuda":
                    raise AssertionError(
                        f"flagged step used {info.get('route_backend')!r}")
                step_order = list(step_receipt.get("ranking_prefix") or ())
                _assert_topk_parity(python_order, step_order)

                receipt["captures"].append({
                    "turn": int(turn),
                    "candidate_count": int(step_receipt.get("candidate_count", 0)),
                    "python": _summary(python_receipt),
                    "cuda_preflight": _summary(cuda_preflight),
                    "cuda_step": _summary(step_receipt),
                    "step_wall_ms_including_decode": step_wall_ms,
                })

            flagged_walls = [
                row["cuda_step"]["route_wall_ms"]
                for row in receipt["captures"]
            ]
            python_walls = [
                row["python"]["route_wall_ms"]
                for row in receipt["captures"]
            ]
            violations = []
            if any(wall > float(args.max_route_ms) for wall in flagged_walls):
                violations.append(
                    "flagged route wall exceeds "
                    f"{float(args.max_route_ms):.3f} ms: {flagged_walls}")
            receipt["summary"] = {
                "python_route_wall_ms": python_walls,
                "cuda_route_wall_ms": flagged_walls,
                "python_route_wall_ms_mean": sum(python_walls) / len(python_walls),
                "cuda_route_wall_ms_mean": sum(flagged_walls) / len(flagged_walls),
                "cuda_route_wall_ms_max": max(flagged_walls),
            }
            receipt["gate"] = {
                "passed": not violations,
                "violations": violations,
                "parity_topks": list(TOPKS),
                "flagged_backend": "cuda_ragged",
            }
    except Exception as exc:  # Receipt first; CUDA/device failures are evidence.
        receipt["gate"] = {
            "passed": False,
            "violations": ["execution_error"],
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        if repo is not None:
            repo.close()
    return receipt


def _write_receipt(path, receipt):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def test_e3_receipt_schema_and_route_terms_are_json_serializable():
    # Pure CPU contract test: no CUDA constructor or model import belongs in
    # pytest collection. The real gate returns this outer schema plus the
    # route-level schema emitted by GQAArenaCache.route().
    example = {
        "schema": SCHEMA,
        "config": {"node_count": 37, "topks": list(TOPKS)},
        "warmup": [],
        "captures": [{
            "python": {
                "route_wall_ms": 1.0,
                "terms_ms": {"python_centroid_score_ms": 0.5},
            },
            "cuda_step": {
                "route_wall_ms": 0.5,
                "terms_ms": {"cuda_route_call_ms": 0.2},
                "rebuilds": {"cuda_bank": {"status": "reused"}},
            },
        }],
        "gate": {"passed": True, "violations": []},
    }
    assert json.loads(json.dumps(example))["schema"] == SCHEMA


def test_e3_requires_all_registered_topks():
    with pytest.raises(AssertionError, match="k=16"):
        _assert_topk_parity(list(range(16)), list(range(15)) + [99])


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if not args.run_gpu:
        raise SystemExit(
            "refusing model/CUDA work without --run-gpu; use pytest for CPU "
            "schema checks")
    receipt = run_gpu_gate(args)
    _write_receipt(args.out, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt.get("gate", {}).get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
