#!/usr/bin/env python3
"""MOE-E1.2 teacher-arm mechanism diagnostics.

This diagnosis-only runner preserves the registered E1/E1.1 receipts.  It
writes new artifacts under artifacts/moe_e1_diag and provides:

* D1/D2: exact E1 multi-arm scheduling with a per-target NLL profile and
  eight deterministic top-5 prediction probes.
* D3: WikiText-2048 and tail-truncated guide-256 prefix controls.
* D4: an isolated, one-call-per-block cross-check matching the established
  context-ladder/real-text streamed forward machinery.
* D5: a static source audit with file and line anchors.

The frozen GPT-OSS-20B is used only for inference.  CPU self-tests are
synthetic machinery checks and are never presented as mechanism evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gpt_oss20b_expert_e1 as e1


ORDER_PATH = REPO_ROOT / "orders" / "MOE_E1_2_DIAG_TEACHER_ARM.md"
E1_OUTPUT = REPO_ROOT / "artifacts" / "moe_e1"
OUTPUT = REPO_ROOT / "artifacts" / "moe_e1_diag"
RT1_MANIFEST = REPO_ROOT / "artifacts" / "moe_rt1" / "corpora_manifest.json"
RT1_WINDOWS = REPO_ROOT / "artifacts" / "moe_rt1" / "corpus_windows.npz"
CONTEXT_LADDER_PATH = SCRIPT_DIR / "gpt_oss20b_context_ladder.py"
REALTEXT_PATH = SCRIPT_DIR / "gpt_oss20b_realtext_ppl_gate.py"
STREAM_PATH = SCRIPT_DIR / "gpt_oss20b_stream_forward_smoke.py"
CORE_PATH = REPO_ROOT / "core" / "gpt_oss20b_tc.py"

NAMED_WINDOWS = (11, 13, 15)
PREFIX_KINDS = ("guide2048", "wikitext2048", "guide256")
PATH_KINDS = ("e1", "established")
SAMPLE_POSITIONS = (1, 74, 146, 219, 291, 364, 436, 509)
N_TARGETS = e1.WINDOW_TOKENS - 1
MEAN_NLL_AGREEMENT_ATOL = 0.02
PROFILE_NLL_P99_ATOL = 0.10

GPU_WRAPPER = (
    "flock -w 7200 /tmp/forge-gpu.lock "
    "timeout --signal=TERM --kill-after=5s 590s "
    "env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 "
    "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
)

D5_AUDIT_RESULT = (
    "D5 STATIC AUDIT — No position-id, attention-mask, sink, or teacher-tail "
    "gather divergence was found between capture-pairs and eval-gates: both "
    "concatenate the prefix before the shared window, use position_offset=0, "
    "construct full-length GPT-OSS YARN tables, set standard sink-aware "
    "attention, and call the same GptOssDiagnosticBlockTC. capture-pairs "
    "forwards 2,560 tokens through L*=22 and captures the final 512 block "
    "outputs; eval-gates forwards 2,559 inputs through all 24 layers, slices "
    "the final 511 hidden rows, and scores them against window[1:]. The only "
    "execution-path divergence from the established stream runner is that "
    "eval-gates reuses each loaded block for teacher, base, and expert calls "
    "before synchronizing, whereas the established path calls each block once "
    "and synchronizes; D4 is required to determine whether that multi-arm "
    "scheduling/reuse is causal. The opt-in --eval-fix e12 path isolates arm "
    "forwards, writes append-only eval_e11_e12 receipts, and is guarded by a "
    "MECHANISM=BUG diagnostic; the original path remains the default. G1 "
    "proves token-ID equality only, not teacher-logit correctness."
)

_WIKITEXT_PREFIX_CACHE: tuple[np.ndarray, dict[str, Any]] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run registered MOE-E1.2 teacher-arm diagnostics."
    )
    parser.add_argument(
        "mode",
        choices=("self-test", "audit", "score", "analyze", "analyze-e12"),
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--model-dir", type=Path, default=e1.SNAPSHOT)
    parser.add_argument("--window-index", type=int, choices=NAMED_WINDOWS)
    parser.add_argument("--prefix-kind", choices=PREFIX_KINDS)
    parser.add_argument("--path-kind", choices=PATH_KINDS)
    parser.add_argument("--attempt", type=int, default=0)
    return parser.parse_args()


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    work = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(work).cast("B")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def ensure_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != OUTPUT.resolve():
        raise ValueError(f"diagnostic artifacts are restricted to {OUTPUT.resolve()}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_model(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != e1.SNAPSHOT.resolve():
        raise ValueError(f"diagnosis is sealed to {e1.SNAPSHOT.resolve()}")
    return resolved


def cuda_probe() -> dict[str, Any]:
    entries = sorted(Path("/dev").glob("nvidia*"))
    nodes: list[str] = []
    for path in entries:
        try:
            if stat.S_ISCHR(path.stat().st_mode):
                nodes.append(str(path))
        except OSError:
            pass
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        smi: dict[str, Any] = {
            "returncode": int(proc.returncode),
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        smi = {"error_type": type(exc).__name__, "error": str(exc)}
    return {
        "character_device_nodes": nodes,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi": smi,
    }


def require_cuda() -> dict[str, Any]:
    probe = cuda_probe()
    required = {"/dev/nvidia0", "/dev/nvidiactl"}
    if not required.issubset(set(probe["character_device_nodes"])):
        raise RuntimeError(
            "CUDA device nodes unavailable; run score modes on the lead GPU "
            "under the registered flock/590s wrapper"
        )
    return probe


def nvidia_smi() -> str | None:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,"
                "power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=15,
        ).strip()
    except Exception:
        return None


def receipt_path(
    output: Path,
    window_index: int,
    path_kind: str,
    prefix_kind: str,
    attempt: int = 0,
) -> Path:
    suffix = "" if attempt == 0 else f"_attempt{attempt:02d}"
    return (
        output
        / "runs"
        / f"window_{window_index:03d}"
        / f"{path_kind}_{prefix_kind}{suffix}.json"
    )


def load_wikitext_prefix() -> tuple[np.ndarray, dict[str, Any]]:
    global _WIKITEXT_PREFIX_CACHE
    if _WIKITEXT_PREFIX_CACHE is not None:
        prefix, source = _WIKITEXT_PREFIX_CACHE
        return prefix.copy(), dict(source)
    if not RT1_MANIFEST.is_file() or not RT1_WINDOWS.is_file():
        raise FileNotFoundError("frozen RT1 WikiText windows are missing")
    manifest = read_json(RT1_MANIFEST)
    if manifest.get("windows_file_sha256") != sha256_file(RT1_WINDOWS):
        raise RuntimeError("frozen RT1 window payload hash mismatch")
    sources = manifest["corpora"]["generic"]["sources"]
    allowed = Path("/home/vader/.cache/huggingface/hub/datasets--wikitext").resolve()
    for source in sources:
        if not Path(source["path"]).resolve().is_relative_to(allowed):
            raise RuntimeError("WikiText prefix source escapes the registered cache")
    with np.load(RT1_WINDOWS, allow_pickle=False) as stored:
        generic = np.ascontiguousarray(stored["generic"], dtype=np.int64)
    if generic.shape != (16, e1.WINDOW_TOKENS):
        raise RuntimeError(f"RT1 generic windows have unexpected shape {generic.shape}")

    import gpt_oss20b_router_telemetry as rt1
    from transformers import AutoTokenizer

    text, reconstructed_sources = rt1.build_generic_corpus()
    canonical = text.encode("utf-8")
    declared = manifest["corpora"]["generic"]
    if hashlib.sha256(canonical).hexdigest() != declared["corpus_sha256"]:
        raise RuntimeError("reconstructed WikiText corpus hash differs from RT1")
    tokenizer = AutoTokenizer.from_pretrained(
        str(e1.SNAPSHOT),
        local_files_only=True,
    )
    all_ids = rt1.tokenizer_ids(tokenizer, text)
    if int(all_ids.size) != int(declared["corpus_token_count"]):
        raise RuntimeError("reconstructed WikiText token count differs from RT1")
    if not np.array_equal(all_ids[: e1.WINDOW_TOKENS], generic[0]):
        raise RuntimeError("reconstructed WikiText IDs differ from frozen RT1 window 0")
    prefix = np.ascontiguousarray(all_ids[: e1.PREFIX_TOKENS], dtype=np.int64)
    if prefix.shape != (e1.PREFIX_TOKENS,):
        raise RuntimeError(f"WikiText prefix has unexpected shape {prefix.shape}")
    source = {
        "kind": "wikitext2048",
        "construction": (
            "first contiguous 2048 tokens of the canonical frozen RT1 WikiText "
            "corpus; first 512 IDs cross-checked against frozen RT1 window 0"
        ),
        "start_token": 0,
        "stop_token_exclusive": e1.PREFIX_TOKENS,
        "rt1_crosscheck_window_index": 0,
        "source_manifest": str(RT1_MANIFEST),
        "source_manifest_sha256": sha256_file(RT1_MANIFEST),
        "source_windows": str(RT1_WINDOWS),
        "source_windows_sha256": sha256_file(RT1_WINDOWS),
        "sources": sources,
        "reconstructed_sources": reconstructed_sources,
        "canonical_corpus_sha256": hashlib.sha256(canonical).hexdigest(),
        "token_ids_sha256": sha256_array(prefix),
    }
    _WIKITEXT_PREFIX_CACHE = (prefix.copy(), dict(source))
    return prefix, source


def diagnostic_inputs(
    window_index: int,
    prefix_kind: str,
) -> dict[str, Any]:
    manifest, prepared = e1.load_prepared(E1_OUTPUT)
    window = np.ascontiguousarray(
        prepared["behavioral_ids"][window_index],
        dtype=np.int64,
    )
    registered_prefix = np.ascontiguousarray(
        prepared["behavioral_prefix_ids"][window_index],
        dtype=np.int64,
    )
    if prefix_kind == "guide2048":
        prefix = registered_prefix
        prefix_source = {
            **manifest["windows"]["behavioral_prefixes"][window_index],
            "kind": "guide2048",
            "construction": "registered E1 behavioral TRAIN-guide prefix",
        }
    elif prefix_kind == "guide256":
        prefix = np.ascontiguousarray(registered_prefix[-256:], dtype=np.int64)
        prefix_source = {
            **manifest["windows"]["behavioral_prefixes"][window_index],
            "kind": "guide256",
            "construction": (
                "final 256 tokens of the registered E1 behavioral prefix; "
                "preserves the immediate prefix/window boundary"
            ),
            "parent_prefix_ids_sha256": sha256_array(registered_prefix),
        }
    elif prefix_kind == "wikitext2048":
        prefix, prefix_source = load_wikitext_prefix()
    else:
        raise AssertionError(prefix_kind)
    inputs = np.concatenate([prefix, window[:-1]])[None, :]
    full_sequence = np.concatenate([prefix, window])
    targets = np.ascontiguousarray(window[1:], dtype=np.int64)
    predictor_ids = np.ascontiguousarray(window[:-1], dtype=np.int64)
    if targets.shape != (N_TARGETS,) or predictor_ids.shape != (N_TARGETS,):
        raise RuntimeError("target/predictor shape contract failed")
    return {
        "window": window,
        "prefix": prefix,
        "inputs": np.ascontiguousarray(inputs, dtype=np.int64),
        "full_sequence": np.ascontiguousarray(full_sequence, dtype=np.int64),
        "targets": targets,
        "predictor_ids": predictor_ids,
        "source_window": manifest["windows"]["behavioral_heldout"][window_index],
        "prefix_source": prefix_source,
        "prepared_windows": manifest["prepared_windows"],
    }


def stable_nlls(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    work = np.asarray(logits, dtype=np.float32)
    target_flat = np.asarray(targets, dtype=np.int64).reshape(-1)
    if work.ndim != 2 or work.shape[0] != target_flat.size:
        raise ValueError(
            f"logit/target shape mismatch {work.shape} vs {target_flat.shape}"
        )
    nlls = np.empty(target_flat.size, dtype=np.float64)
    for index, (row, target) in enumerate(zip(work, target_flat)):
        maximum = float(row.max())
        logsumexp = maximum + float(
            np.log(np.exp(row - maximum).sum())
        )
        nlls[index] = logsumexp - float(row[int(target)])
    return nlls


def token_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def token_probe(
    row: np.ndarray,
    token_id: int | None,
    tokenizer,
) -> dict[str, Any] | None:
    if token_id is None:
        return None
    maximum = float(row.max())
    logsumexp = maximum + float(
        np.log(np.exp(row.astype(np.float64) - maximum).sum())
    )
    logit = float(row[int(token_id)])
    return {
        "token_id": int(token_id),
        "text": token_text(tokenizer, int(token_id)),
        "logit": logit,
        "logprob": logit - logsumexp,
        "nll": logsumexp - logit,
        "rank_1_based": int(1 + np.count_nonzero(row > logit)),
    }


def summarize_position_profile(nlls: np.ndarray) -> dict[str, Any]:
    values = np.asarray(nlls, dtype=np.float64).reshape(-1)
    positions = np.arange(values.size, dtype=np.float64)
    edges = np.linspace(0, values.size, 9, dtype=np.int64)
    bins = []
    for index, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        part = values[start:stop]
        bins.append(
            {
                "bin": index,
                "start_target_position": int(start),
                "stop_target_position_exclusive": int(stop),
                "token_count": int(part.size),
                "mean_nll": float(part.mean()),
                "median_nll": float(np.median(part)),
            }
        )
    correlation = (
        float(np.corrcoef(positions, values)[0, 1])
        if values.size > 1 and float(values.std()) > 0.0
        else 0.0
    )
    slope = float(np.polyfit(positions, values, 1)[0]) if values.size > 1 else 0.0
    boundary = values[:16]
    remainder = values[16:]
    return {
        "token_count": int(values.size),
        "mean_nll": float(values.mean()),
        "ppl": float(math.exp(values.mean())),
        "min_nll": float(values.min()),
        "max_nll": float(values.max()),
        "median_nll": float(np.median(values)),
        "p90_nll": float(np.quantile(values, 0.90)),
        "p99_nll": float(np.quantile(values, 0.99)),
        "position_nll_correlation": correlation,
        "linear_nll_per_position": slope,
        "first_16_mean_nll": float(boundary.mean()),
        "remaining_mean_nll": float(remainder.mean()),
        "boundary_minus_remaining_mean_nll": float(
            boundary.mean() - remainder.mean()
        ),
        "bins": bins,
    }


def profile_from_logits(
    logits: np.ndarray,
    *,
    targets: np.ndarray,
    predictor_ids: np.ndarray,
    prefix_tokens: int,
    tokenizer,
) -> dict[str, Any]:
    work = np.asarray(logits, dtype=np.float32)
    nlls = stable_nlls(work, targets)
    if nlls.shape != (N_TARGETS,):
        raise RuntimeError(f"per-position NLL shape {nlls.shape}")
    per_position = [
        {
            "target_position": int(index),
            "sequence_logit_position": int(prefix_tokens + index),
            "predictor_token_id": int(predictor_ids[index]),
            "target_token_id": int(targets[index]),
            "nll": float(nlls[index]),
        }
        for index in range(N_TARGETS)
    ]
    samples = []
    for position in SAMPLE_POSITIONS:
        row = work[position]
        candidate = np.argpartition(-row, 4)[:5]
        top_ids = candidate[np.argsort(-row[candidate], kind="stable")]
        actual_id = int(targets[position])
        previous_id = int(targets[position - 1]) if position > 0 else None
        next_id = int(targets[position + 1]) if position + 1 < targets.size else None
        top = [
            {
                "rank": rank,
                "token_id": int(token_id),
                "text": token_text(tokenizer, int(token_id)),
                "logit": float(row[int(token_id)]),
                "matches_actual_target": int(token_id) == actual_id,
                "matches_previous_target": (
                    previous_id is not None and int(token_id) == previous_id
                ),
                "matches_next_target": (
                    next_id is not None and int(token_id) == next_id
                ),
            }
            for rank, token_id in enumerate(top_ids.tolist(), start=1)
        ]
        samples.append(
            {
                "target_position": int(position),
                "sequence_logit_position": int(prefix_tokens + position),
                "predictor": {
                    "token_id": int(predictor_ids[position]),
                    "text": token_text(tokenizer, int(predictor_ids[position])),
                },
                "actual_target": token_probe(row, actual_id, tokenizer),
                "previous_target": token_probe(row, previous_id, tokenizer),
                "next_target": token_probe(row, next_id, tokenizer),
                "top5": top,
                "top5_contains_actual": any(
                    item["matches_actual_target"] for item in top
                ),
                "top5_contains_previous": any(
                    item["matches_previous_target"] for item in top
                ),
                "top5_contains_next": any(
                    item["matches_next_target"] for item in top
                ),
            }
        )
    return {
        "summary": summarize_position_profile(nlls),
        "per_position": per_position,
        "sample_positions": list(SAMPLE_POSITIONS),
        "prediction_sanity": samples,
        "target_ids_sha256": sha256_array(targets),
        "nll_fp64_sha256": sha256_array(nlls.astype(np.float64)),
    }


def aggregate_from_logits(logits: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    nlls = stable_nlls(logits, targets)
    return {
        "token_count": int(nlls.size),
        "nll_sum": float(nlls.sum()),
        "mean_nll": float(nlls.mean()),
        "ppl": float(math.exp(nlls.mean())),
        "target_ids_sha256": sha256_array(targets),
    }


def logits_from_hidden(lm_head, hidden) -> np.ndarray:
    logits = lm_head(hidden)
    host = logits.float().numpy().astype(np.float32, copy=True).reshape(
        -1, logits.shape[-1]
    )
    del logits
    return host


def run_e1_multi_arm(
    *,
    args: argparse.Namespace,
    data: dict[str, Any],
    receipt: dict[str, Any],
    path: Path,
    tokenizer,
) -> dict[str, Any]:
    runtime = e1.load_runtime()
    tc = runtime["tc"]
    cfg = runtime["GptOss20BConfig"].from_model_dir(args.model_dir.resolve())
    e1.validate_model_contract(cfg)
    where = runtime["build_safetensors_map"](args.model_dir.resolve())
    pack, key, A, B = e1.load_expert_arrays(E1_OUTPUT, e1.ADDRESS_RULE_E11)
    install_layer = int(pack["layer"])
    compute_dtype = runtime["BlockTC"].COMPUTE_DTYPE
    base_ids = data["window"][:-1][None, :]
    targets = data["targets"]
    fire_info: dict[str, Any] | None = None
    with tc.no_grad():
        A_device = tc.tensor(A.astype(np.float32), dtype=compute_dtype)
        B_device = tc.tensor(B.astype(np.float32), dtype=compute_dtype)
        embed = runtime["GptOssRowEmbedding"](where)
        h_teacher = embed(data["inputs"])
        h_base = embed(base_ids)
        h_expert = embed(base_ids)
        cos, sin = runtime["gpt_oss_yarn_rope_tables"](
            cfg, int(data["inputs"].shape[1])
        )
        receipt["layers"] = []
        for layer in range(e1.N_LAYERS):
            layer_started = time.perf_counter()
            block = runtime["GptOssDiagnosticBlockTC"].from_safetensors(
                cfg,
                where,
                layer,
                expert_mode="resident_packed_mxfp4",
            )
            e1.configure_block(block)
            h_teacher, kv_t, route_t = block(h_teacher, cos, sin)
            h_base, kv_b, route_b = block(h_base, cos, sin)
            if layer == install_layer:
                h_expert, kv_e, route_e, fire_info = e1.block_forward_with_expert(
                    block,
                    h_expert,
                    cos,
                    sin,
                    tc=tc,
                    compute_dtype=compute_dtype,
                    key=key,
                    tau=float(pack["tau"]),
                    A_device=A_device,
                    B_device=B_device,
                )
            else:
                h_expert, kv_e, route_e = block(h_expert, cos, sin)
            tc.synchronize()
            receipt["layers"].append(
                {
                    "layer": int(layer),
                    "attention_backend": block.self_attn.last_attention_backend,
                    "wall_seconds": float(time.perf_counter() - layer_started),
                }
            )
            receipt["completed_layers"] = layer + 1
            receipt["status"] = "running_layers"
            write_json(path, receipt)
            del block, kv_t, kv_b, kv_e, route_t, route_b, route_e
            gc.collect()
            if hasattr(tc, "empty_cache"):
                tc.empty_cache()
        if fire_info is None:
            raise RuntimeError("install layer did not execute expert arm")
        teacher_tail = h_teacher.slice(
            1,
            int(h_teacher.shape[1]) - N_TARGETS,
            N_TARGETS,
        )
        teacher_norm = e1.final_norm(runtime, cfg, where, teacher_tail)
        base_norm = e1.final_norm(runtime, cfg, where, h_base)
        expert_norm = e1.final_norm(runtime, cfg, where, h_expert)
        del h_teacher, h_base, h_expert, teacher_tail
        if hasattr(tc, "empty_cache"):
            tc.empty_cache()
        lm_head = e1.load_lm_head(runtime, cfg, where)
        base_logits = logits_from_hidden(lm_head, base_norm)
        base = aggregate_from_logits(base_logits, targets)
        del base_logits, base_norm
        teacher_logits = logits_from_hidden(lm_head, teacher_norm)
        teacher = aggregate_from_logits(teacher_logits, targets)
        profile = profile_from_logits(
            teacher_logits,
            targets=targets,
            predictor_ids=data["predictor_ids"],
            prefix_tokens=int(data["prefix"].size),
            tokenizer=tokenizer,
        )
        del teacher_logits, teacher_norm
        expert_logits = logits_from_hidden(lm_head, expert_norm)
        expert = aggregate_from_logits(expert_logits, targets)
        del expert_logits, expert_norm
        tc.synchronize()
    return {
        "arms": {"base": base, "teacher": teacher, "expert": expert},
        "profile": profile,
        "fire": {
            key_name: value
            for key_name, value in fire_info.items()
            if key_name not in {"scores", "fired"}
        },
        "install_layer": install_layer,
        "rank": int(pack["rank"]),
        "tau": float(pack["tau"]),
        "expertpack_manifest": str(
            E1_OUTPUT / e1.expertpack_dirname(e1.ADDRESS_RULE_E11) / "manifest.json"
        ),
    }


def run_established_isolated(
    *,
    args: argparse.Namespace,
    data: dict[str, Any],
    receipt: dict[str, Any],
    path: Path,
    tokenizer,
) -> dict[str, Any]:
    runtime = e1.load_runtime()
    tc = runtime["tc"]
    cfg = runtime["GptOss20BConfig"].from_model_dir(args.model_dir.resolve())
    e1.validate_model_contract(cfg)
    where = runtime["build_safetensors_map"](args.model_dir.resolve())
    targets = data["targets"]
    with tc.no_grad():
        embed = runtime["GptOssRowEmbedding"](where)
        hidden = embed(data["inputs"])
        cos, sin = runtime["gpt_oss_yarn_rope_tables"](
            cfg, int(data["inputs"].shape[1])
        )
        receipt["layers"] = []
        for layer in range(e1.N_LAYERS):
            layer_started = time.perf_counter()
            block = runtime["GptOssDiagnosticBlockTC"].from_safetensors(
                cfg,
                where,
                layer,
                expert_mode="resident_packed_mxfp4",
            )
            e1.configure_block(block)
            hidden, kv, route = block(hidden, cos, sin)
            tc.synchronize()
            receipt["layers"].append(
                {
                    "layer": int(layer),
                    "attention_backend": block.self_attn.last_attention_backend,
                    "wall_seconds": float(time.perf_counter() - layer_started),
                }
            )
            receipt["completed_layers"] = layer + 1
            receipt["status"] = "running_layers"
            write_json(path, receipt)
            del block, kv, route
            gc.collect()
            if hasattr(tc, "empty_cache"):
                tc.empty_cache()
        tail = hidden.slice(1, int(hidden.shape[1]) - N_TARGETS, N_TARGETS)
        normalized = e1.final_norm(runtime, cfg, where, tail)
        del hidden, tail
        if hasattr(tc, "empty_cache"):
            tc.empty_cache()
        lm_head = e1.load_lm_head(runtime, cfg, where)
        logits = logits_from_hidden(lm_head, normalized)
        arm = aggregate_from_logits(logits, targets)
        profile = profile_from_logits(
            logits,
            targets=targets,
            predictor_ids=data["predictor_ids"],
            prefix_tokens=int(data["prefix"].size),
            tokenizer=tokenizer,
        )
        del logits, normalized
        tc.synchronize()
    return {
        "arms": {"teacher": arm},
        "profile": profile,
        "established_path_contract": {
            "one_sequence_call_per_loaded_block": True,
            "synchronize_after_each_block": True,
            "attention_mode": "standard",
            "route_detail": "summary",
            "expert_empty_cache_interval": 0,
            "stream_runner": str(STREAM_PATH),
            "context_ladder": str(CONTEXT_LADDER_PATH),
            "realtext_ppl_gate": str(REALTEXT_PATH),
        },
    }


def score(args: argparse.Namespace) -> int:
    if args.window_index is None or args.prefix_kind is None or args.path_kind is None:
        raise ValueError(
            "score requires --window-index, --prefix-kind, and --path-kind"
        )
    if args.attempt < 0:
        raise ValueError("--attempt must be nonnegative")
    if args.path_kind == "established" and args.prefix_kind != "guide2048":
        raise ValueError("D4 established cross-check is registered for guide2048")
    output = ensure_output(args.output_dir)
    ensure_model(args.model_dir)
    probe = require_cuda()
    index = int(args.window_index)
    path = receipt_path(
        output,
        index,
        args.path_kind,
        args.prefix_kind,
        args.attempt,
    )
    if path.exists():
        prior = read_json(path)
        if prior.get("status") == "complete":
            print(json.dumps({"status": "already_complete", "receipt": str(path)}))
            return 0
        raise FileExistsError(
            f"attempt receipt already exists at {path}; use the next --attempt"
        )
    data = diagnostic_inputs(index, args.prefix_kind)
    reference_path = (
        E1_OUTPUT / "eval_e11" / f"narrative_{index:03d}.json"
    )
    reference = read_json(reference_path)
    if reference.get("status") != "complete":
        raise RuntimeError(f"E1.1 reference receipt is incomplete: {reference_path}")
    target_hash = sha256_array(data["targets"])
    if target_hash != reference["target_ids_sha256"]:
        raise RuntimeError("diagnostic target hash differs from E1.1 receipt")
    started = time.perf_counter()
    receipt: dict[str, Any] = {
        "schema": "moe_e1_diag_score_v1",
        "created_at": now_iso(),
        "status": "starting",
        "argv": sys.argv,
        "required_shell_wrapper": GPU_WRAPPER,
        "order": str(ORDER_PATH),
        "window_index": index,
        "path_kind": args.path_kind,
        "prefix_kind": args.prefix_kind,
        "attempt": int(args.attempt),
        "model_dir": str(args.model_dir.resolve()),
        "compute_dtype": "bfloat16",
        "attention_mode": "standard",
        "expert_mode": "resident_packed_mxfp4",
        "teacher_input_tokens": int(data["inputs"].shape[1]),
        "full_sequence_tokens": int(data["full_sequence"].size),
        "prefix_tokens": int(data["prefix"].size),
        "scored_target_tokens": int(data["targets"].size),
        "input_ids_sha256": sha256_array(data["inputs"]),
        "full_sequence_ids_sha256": sha256_array(data["full_sequence"]),
        "prefix_ids_sha256": sha256_array(data["prefix"]),
        "target_ids_sha256": target_hash,
        "source_window": data["source_window"],
        "prefix_source": data["prefix_source"],
        "reference_e11_receipt": str(reference_path),
        "reference_e11_receipt_sha256": sha256_file(reference_path),
        "reference_e11_arms": reference["arms"],
        "teacher_scores_prefix_tokens": False,
        "cuda_environment": probe,
        "gpu_before": nvidia_smi(),
        "script_sha256_at_run": sha256_file(SCRIPT_PATH),
        "e1_script_sha256_at_run": sha256_file(e1.SCRIPT_PATH),
        "core_script_sha256_at_run": sha256_file(CORE_PATH),
        "completed_layers": 0,
    }
    write_json(path, receipt)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir.resolve()))
        if args.path_kind == "e1":
            result = run_e1_multi_arm(
                args=args,
                data=data,
                receipt=receipt,
                path=path,
                tokenizer=tokenizer,
            )
        else:
            result = run_established_isolated(
                args=args,
                data=data,
                receipt=receipt,
                path=path,
                tokenizer=tokenizer,
            )
        receipt.update(
            {
                **result,
                "status": "complete",
                "wall_seconds": float(time.perf_counter() - started),
                "gpu_after": nvidia_smi(),
            }
        )
        write_json(path, receipt)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "receipt": str(path),
                    "window_index": index,
                    "path_kind": args.path_kind,
                    "prefix_kind": args.prefix_kind,
                    "teacher": receipt["arms"]["teacher"],
                }
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        receipt.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "wall_seconds": float(time.perf_counter() - started),
                "gpu_after": nvidia_smi(),
            }
        )
        write_json(path, receipt)
        raise


def source_anchor(path: Path, needle: str) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    matches = [index + 1 for index, line in enumerate(lines) if needle in line]
    if not matches:
        raise RuntimeError(f"static-audit anchor missing in {path}: {needle}")
    return {
        "path": str(path),
        "line": int(matches[0]),
        "needle": needle,
        "file_sha256": sha256_file(path),
    }


def audit_payload() -> dict[str, Any]:
    anchors = {
        "capture_concat": source_anchor(
            e1.SCRIPT_PATH,
            "teacher_ids = np.concatenate([prefix_ids, pair_ids])[None, :]",
        ),
        "capture_rope": source_anchor(
            e1.SCRIPT_PATH,
            "cfg, PREFIX_TOKENS + WINDOW_TOKENS",
        ),
        "capture_teacher_forward": source_anchor(
            e1.SCRIPT_PATH,
            "h_teacher, kv_teacher, _route_teacher = block(h_teacher, cos, sin)",
        ),
        "capture_tail": source_anchor(
            e1.SCRIPT_PATH,
            "teacher_out = h_teacher.float().numpy().astype(np.float32, copy=True)[",
        ),
        "eval_concat": source_anchor(
            e1.SCRIPT_PATH,
            "teacher_ids = np.concatenate([prefix, window[:-1]])[None, :]",
        ),
        "eval_rope": source_anchor(
            e1.SCRIPT_PATH,
            "cfg, teacher_ids.shape[1]",
        ),
        "eval_teacher_forward": source_anchor(
            e1.SCRIPT_PATH,
            "h_teacher, kv_t, _route_t = block(h_teacher, cos, sin)",
        ),
        "eval_base_forward": source_anchor(
            e1.SCRIPT_PATH,
            "h_base, kv_b, _route_b = block(h_base, cos, sin)",
        ),
        "eval_tail": source_anchor(
            e1.SCRIPT_PATH,
            "teacher_tail = h_teacher.slice(",
        ),
        "eval_target_score": source_anchor(
            e1.SCRIPT_PATH,
            '"teacher": nll_from_hidden(lm_head, h_teacher_norm, targets)',
        ),
        "established_forward": source_anchor(
            STREAM_PATH,
            "h, kv, route_info = block(h, cos, sin)",
        ),
        "established_sync": source_anchor(
            STREAM_PATH,
            "tc.synchronize()",
        ),
        "core_position_default": source_anchor(
            CORE_PATH,
            "def __call__(self, x, cos, sin, position_offset: int = 0, kv_cache=None):",
        ),
        "core_rope_slice": source_anchor(
            CORE_PATH,
            "cseg = cos.slice(0, position_offset + int(shift), L)",
        ),
        "core_sink_sliding": source_anchor(
            CORE_PATH,
            "attn = sliding_sink_attention_tc(",
        ),
        "core_full_mask": source_anchor(
            CORE_PATH,
            "mask = _gpt_oss_attention_mask(",
        ),
    }
    return {
        "schema": "moe_e1_diag_d5_static_audit_v1",
        "created_at": now_iso(),
        "status": "complete",
        "result_verbatim": D5_AUDIT_RESULT,
        "anchors": anchors,
        "expected_differences": [
            {
                "topic": "input length",
                "capture_pairs": "prefix 2048 + full shared window 512 = 2560",
                "eval_gates": "prefix 2048 + window predictors 511 = 2559",
                "reason": "eval shifts window[0:511] against targets window[1:512]",
            },
            {
                "topic": "depth and output",
                "capture_pairs": "through install layer L*=22; final 512 block outputs",
                "eval_gates": "all 24 layers; final 511 rows, final norm, lm_head",
                "reason": "pair supervision versus next-token behavioral scoring",
            },
        ],
        "candidate_divergence_for_d4": {
            "e1_eval": (
                "one block instance called for teacher, base, expert, then synchronized"
            ),
            "established": (
                "one block instance called once for the long sequence, then synchronized"
            ),
            "candidate_defect_anchor": "eval_teacher_forward through eval_base_forward",
            "verdict": "NOT_MEASURED until D4",
        },
        "g1_scope": (
            "token-ID equality and nonzero hidden difference only; no teacher NLL or "
            "long-context correctness assertion"
        ),
    }


def audit(args: argparse.Namespace) -> int:
    output = ensure_output(args.output_dir)
    payload = audit_payload()
    path = output / "d5_static_audit.json"
    write_json(path, payload)
    print(
        json.dumps(
            {
                "status": "complete",
                "audit": str(path),
                "result_verbatim": payload["result_verbatim"],
            }
        ),
        flush=True,
    )
    return 0


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    expected: Any,
    observed: Any,
) -> None:
    checks.append(
        {
            "name": name,
            "expected": expected,
            "observed": observed,
            "passed": expected == observed,
        }
    )


class TinyTokenizer:
    def decode(self, ids, **_kwargs):
        return f"<{int(ids[0])}>"


def classify_mechanism_rows(rows: Sequence[dict[str, Any]]) -> str:
    if len(rows) != len(NAMED_WINDOWS):
        return "NOT_MEASURED"
    if not all(row["hashes_match"] for row in rows):
        return "INDETERMINATE"
    e1_reproduced = all(
        abs(float(row["e1_diag_minus_original_mean_nll"]))
        <= MEAN_NLL_AGREEMENT_ATOL
        for row in rows
    )
    d4_disagrees = any(
        abs(float(row["established_minus_e1_diag_mean_nll"]))
        > MEAN_NLL_AGREEMENT_ATOL
        or float(row["profile_abs_nll_delta_p99"]) > PROFILE_NLL_P99_ATOL
        for row in rows
    )
    if e1_reproduced and d4_disagrees:
        return "BUG"
    d4_matches = all(
        abs(float(row["established_minus_e1_diag_mean_nll"]))
        <= MEAN_NLL_AGREEMENT_ATOL
        and float(row["profile_abs_nll_delta_p99"]) <= PROFILE_NLL_P99_ATOL
        for row in rows
    )
    degrades = all(
        float(row["established_mean_nll"]) > float(row["original_base_mean_nll"])
        for row in rows
    )
    if d4_matches and degrades:
        return "REAL"
    return "INDETERMINATE"


def self_test(args: argparse.Namespace) -> int:
    output = ensure_output(args.output_dir)
    ensure_model(args.model_dir)
    checks: list[dict[str, Any]] = []
    saved_argv = sys.argv[:]
    try:
        sys.argv = [str(e1.SCRIPT_PATH), "self-test"]
        default_e1_args = e1.parse_args()
    finally:
        sys.argv = saved_argv
    add_check(
        checks,
        "e1_original_eval_path_remains_default",
        e1.EVAL_FIX_ORIGINAL,
        default_e1_args.eval_fix,
    )
    add_check(
        checks,
        "registered_sample_positions",
        [1, 74, 146, 219, 291, 364, 436, 509],
        list(SAMPLE_POSITIONS),
    )

    vocab = 32
    targets = (np.arange(N_TARGETS, dtype=np.int64) + 3) % vocab
    predictors = (targets - 1) % vocab
    logits = np.full((N_TARGETS, vocab), -8.0, dtype=np.float32)
    logits[np.arange(N_TARGETS), targets] = 8.0
    profile = profile_from_logits(
        logits,
        targets=targets,
        predictor_ids=predictors,
        prefix_tokens=e1.PREFIX_TOKENS,
        tokenizer=TinyTokenizer(),
    )
    add_check(
        checks,
        "profile_target_count",
        N_TARGETS,
        profile["summary"]["token_count"],
    )
    add_check(
        checks,
        "prediction_samples_count",
        8,
        len(profile["prediction_sanity"]),
    )
    add_check(
        checks,
        "perfect_actual_in_all_top5",
        True,
        all(item["top5_contains_actual"] for item in profile["prediction_sanity"]),
    )

    shifted_logits = np.full((N_TARGETS, vocab), -8.0, dtype=np.float32)
    previous = np.roll(targets, 1)
    shifted_logits[np.arange(N_TARGETS), previous] = 8.0
    shifted = profile_from_logits(
        shifted_logits,
        targets=targets,
        predictor_ids=predictors,
        prefix_tokens=e1.PREFIX_TOKENS,
        tokenizer=TinyTokenizer(),
    )
    add_check(
        checks,
        "previous_target_shift_detected",
        True,
        all(
            item["top5_contains_previous"]
            for item in shifted["prediction_sanity"]
        ),
    )

    for index in NAMED_WINDOWS:
        guide = diagnostic_inputs(index, "guide2048")
        guide_short = diagnostic_inputs(index, "guide256")
        wiki = diagnostic_inputs(index, "wikitext2048")
        reference = read_json(
            E1_OUTPUT / "eval_e11" / f"narrative_{index:03d}.json"
        )
        add_check(
            checks,
            f"window_{index}_target_hash_matches_e11",
            reference["target_ids_sha256"],
            sha256_array(guide["targets"]),
        )
        add_check(
            checks,
            f"window_{index}_guide2048_full_sequence",
            2560,
            int(guide["full_sequence"].size),
        )
        add_check(
            checks,
            f"window_{index}_guide256_full_sequence",
            768,
            int(guide_short["full_sequence"].size),
        )
        add_check(
            checks,
            f"window_{index}_wikitext2048_full_sequence",
            2560,
            int(wiki["full_sequence"].size),
        )
        add_check(
            checks,
            f"window_{index}_guide256_is_tail",
            True,
            bool(np.array_equal(guide["prefix"][-256:], guide_short["prefix"])),
        )

    audit_info = audit_payload()
    add_check(
        checks,
        "d5_anchor_count",
        16,
        len(audit_info["anchors"]),
    )

    bug_rows = [
        {
            "hashes_match": True,
            "e1_diag_minus_original_mean_nll": 0.0,
            "established_minus_e1_diag_mean_nll": -4.0,
            "profile_abs_nll_delta_p99": 5.0,
            "established_mean_nll": 4.0,
            "original_base_mean_nll": 3.0,
        }
        for _ in NAMED_WINDOWS
    ]
    real_rows = [
        {
            "hashes_match": True,
            "e1_diag_minus_original_mean_nll": 0.0,
            "established_minus_e1_diag_mean_nll": 0.0,
            "profile_abs_nll_delta_p99": 0.0,
            "established_mean_nll": 5.0,
            "original_base_mean_nll": 3.0,
        }
        for _ in NAMED_WINDOWS
    ]
    add_check(
        checks,
        "mechanism_classifier_bug",
        "BUG",
        classify_mechanism_rows(bug_rows),
    )
    add_check(
        checks,
        "mechanism_classifier_real",
        "REAL",
        classify_mechanism_rows(real_rows),
    )
    add_check(
        checks,
        "mechanism_classifier_incomplete",
        "NOT_MEASURED",
        classify_mechanism_rows(real_rows[:2]),
    )

    resume_path = output / "GPU_RESUME_COMMANDS.sh"
    resume_text = resume_path.read_text(encoding="utf-8") if resume_path.is_file() else ""
    add_check(
        checks,
        "resume_flock",
        True,
        "flock -w 7200 /tmp/forge-gpu.lock" in resume_text,
    )
    add_check(
        checks,
        "resume_590s",
        True,
        "590s" in resume_text,
    )
    add_check(
        checks,
        "resume_single_gpu",
        True,
        "CUDA_VISIBLE_DEVICES=0" in resume_text,
    )
    add_check(
        checks,
        "resume_sleep_gap",
        True,
        "sleep 30" in resume_text,
    )
    e12_resume_path = output / "GPU_E12_RESUME_COMMANDS.sh"
    e12_resume_text = (
        e12_resume_path.read_text(encoding="utf-8")
        if e12_resume_path.is_file()
        else ""
    )
    add_check(
        checks,
        "e12_resume_bug_guard",
        True,
        'x.get("mechanism")=="BUG"' in e12_resume_text,
    )
    add_check(
        checks,
        "e12_resume_opt_in_flag",
        True,
        "--eval-fix e12" in e12_resume_text,
    )
    add_check(
        checks,
        "e12_receipt_root_is_append_only_sibling",
        str(E1_OUTPUT / "eval_e11_e12"),
        str(
            e1.eval_root_for_fix(
                E1_OUTPUT,
                e1.ADDRESS_RULE_E11,
                e1.EVAL_FIX_E12,
            )
        ),
    )

    failures = [item["name"] for item in checks if not item["passed"]]
    payload = {
        "schema": "moe_e1_diag_cpu_self_test_v1",
        "created_at": now_iso(),
        "status": "passed" if not failures else "failed",
        "synthetic_only_not_mechanism_evidence": True,
        "checks_total": len(checks),
        "checks_passed": len(checks) - len(failures),
        "failed_check_names": failures,
        "checks": checks,
        "d5_result_verbatim": D5_AUDIT_RESULT,
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
    }
    path = output / "cpu_self_test.json"
    write_json(path, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "checks_passed": payload["checks_passed"],
                "checks_total": payload["checks_total"],
                "artifact": str(path),
            }
        ),
        flush=True,
    )
    return 0 if not failures else 2


def complete_receipt(
    output: Path,
    index: int,
    path_kind: str,
    prefix_kind: str,
) -> tuple[dict[str, Any] | None, str]:
    path = receipt_path(output, index, path_kind, prefix_kind)
    if not path.is_file():
        return None, "missing"
    payload = read_json(path)
    if payload.get("status") != "complete":
        return None, str(payload.get("status", "unknown"))
    payload["_path"] = str(path)
    payload["_sha256"] = sha256_file(path)
    return payload, "complete"


def d1_shape_label(summary: dict[str, Any]) -> str:
    boundary_delta = float(summary["boundary_minus_remaining_mean_nll"])
    correlation = float(summary["position_nll_correlation"])
    if boundary_delta >= 2.0:
        return "boundary_spike"
    if correlation >= 0.35:
        return "degrading_with_position"
    if (
        float(summary["median_nll"]) >= 8.0
        and float(summary["p90_nll"]) - float(summary["median_nll"]) <= 3.0
    ):
        return "uniform_high"
    return "mixed_or_content_dependent"


def mechanism_note(
    mechanism: str,
    rows: Sequence[dict[str, Any]],
    controls: Sequence[dict[str, Any]],
) -> str:
    if mechanism == "BUG":
        anchors = audit_payload()["anchors"]
        teacher_line = anchors["eval_teacher_forward"]["line"]
        base_line = anchors["eval_base_forward"]["line"]
        deltas = ", ".join(
            f"w{row['window_index']} {row['established_minus_e1_diag_mean_nll']:+.4f}"
            for row in rows
        )
        return (
            "MECHANISM=BUG: the E1 eval path's per-layer reuse of one loaded block "
            "for teacher, base, and expert before synchronization diverges from the "
            "established isolated stream "
            f"(scripts/gpt_oss20b_expert_e1.py:{teacher_line}-{base_line}). D4 "
            f"established-minus-E1 mean-NLL deltas were {deltas}; target hashes "
            "matched and the exact E1 diagnostic reproduced the historical teacher "
            "receipt. The opt-in e12 correction must isolate arm forwards; original "
            "receipts and default behavior remain preserved."
        )
    if mechanism == "REAL":
        deltas = ", ".join(
            f"w{row['window_index']} {row['established_minus_e1_diag_mean_nll']:+.4f}"
            for row in rows
        )
        return (
            "MECHANISM=REAL under the registered diagnostic: the established "
            "one-call-per-block long-context path reproduced the E1 teacher "
            f"degradation on all three named windows (mean-NLL deltas {deltas}) "
            "with matching target hashes. No E1-only forward defect was localized; "
            "the E1.1 premise finding therefore stands for this model and sealed "
            "corpus."
        )
    if mechanism == "INDETERMINATE":
        return (
            "MECHANISM=INDETERMINATE: D1-D4 receipts exist but do not satisfy the "
            "registered BUG localization pattern or the all-window REAL "
            "confirmation pattern. Do not upgrade this mixed result to either "
            "registered outcome; inspect the per-window profiles and controls."
        )
    missing = 12 - len(
        [
            item
            for item in controls
            if item.get("status") == "complete"
        ]
    )
    return (
        "MECHANISM=NOT_MEASURED: GPU D1-D4 receipts are incomplete in this CPU-only "
        f"sandbox ({max(0, missing)} of 12 registered score runs remain). The static "
        "D5 audit found no target/position/mask/sink/gather mismatch and isolated "
        "multi-arm block scheduling as the runtime divergence D4 must adjudicate."
    )


def render_report(analysis: dict[str, Any]) -> str:
    lines = [
        "# MOE-E1.2 Teacher-Arm Mechanism Diagnosis",
        "",
        f"Generated: {analysis['created_at']}",
        "",
        "Evidence class: mechanism diagnosis on one model.",
        "",
        "## DG verdicts",
        "",
        analysis["dg"]["DG1"]["report_line"],
        analysis["dg"]["DG2"]["report_line"],
        analysis["dg"]["DG3"]["report_line"],
        analysis["dg"]["DG4"]["report_line"],
        "",
        "## D1-D2 per-window summary",
        "",
    ]
    if analysis["dg"]["DG1"]["rows"]:
        for row in analysis["dg"]["DG1"]["rows"]:
            lines.append(
                f"- window {row['window_index']}: mean_nll={row['mean_nll']:.6f}, "
                f"ppl={row['ppl']:.6f}, profile={row['shape_label']}, "
                f"position_corr={row['position_nll_correlation']:.6f}, "
                f"top5 actual/previous/next="
                f"{row['sample_actual_top5_count']}/"
                f"{row['sample_previous_top5_count']}/"
                f"{row['sample_next_top5_count']} of 8"
            )
    else:
        lines.append("- NOT_MEASURED")
    lines.extend(
        [
            "",
            "## D3 prefix controls",
            "",
        ]
    )
    complete_controls = [
        row
        for row in analysis["dg"]["DG2"]["controls"]
        if row["status"] == "complete"
    ]
    if complete_controls:
        for row in complete_controls:
            lines.append(
                f"- window {row['window_index']} {row['prefix_kind']}: "
                f"mean_nll={row['mean_nll']:.6f}, ppl={row['ppl']:.6f}, "
                f"minus_guide2048_mean_nll="
                f"{row.get('minus_guide2048_mean_nll', float('nan')):.6f}"
            )
    else:
        lines.append("- NOT_MEASURED")
    lines.extend(
        [
            "",
            "## D4 established-path deltas",
            "",
        ]
    )
    if analysis["dg"]["DG3"]["rows"]:
        for row in analysis["dg"]["DG3"]["rows"]:
            lines.append(
                f"- window {row['window_index']}: original_teacher="
                f"{row['original_teacher_mean_nll']:.6f}, e1_diag="
                f"{row['e1_diag_mean_nll']:.6f}, established="
                f"{row['established_mean_nll']:.6f}, "
                f"established_minus_e1="
                f"{row['established_minus_e1_diag_mean_nll']:+.6f}, "
                f"profile_abs_delta_p99={row['profile_abs_nll_delta_p99']:.6f}"
            )
    else:
        lines.append("- NOT_MEASURED")
    lines.extend(
        [
        "",
        "## Mechanism note",
        "",
        analysis["mechanism_note"],
        "",
        "## D5 audit result",
        "",
        analysis["d5_result_verbatim"],
        "",
        "## Resume",
        "",
        f"GPU resume script: {analysis['gpu_resume_script']}",
        "",
        f"BUG-only fixed G4/G5' resume script: {analysis['e12_resume_script']}",
        "",
        "The original E1/E1.1 receipts remain untouched.",
        "",
        ]
    )
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> int:
    output = ensure_output(args.output_dir)
    audit_info = audit_payload()
    write_json(output / "d5_static_audit.json", audit_info)
    records: list[dict[str, Any]] = []
    by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    for index in NAMED_WINDOWS:
        for path_kind, prefix_kind in (
            ("e1", "guide2048"),
            ("e1", "wikitext2048"),
            ("e1", "guide256"),
            ("established", "guide2048"),
        ):
            payload, status = complete_receipt(
                output, index, path_kind, prefix_kind
            )
            row = {
                "window_index": index,
                "path_kind": path_kind,
                "prefix_kind": prefix_kind,
                "status": status,
                "receipt": None if payload is None else payload["_path"],
            }
            records.append(row)
            if payload is not None:
                by_key[(index, path_kind, prefix_kind)] = payload

    d1_receipts = [
        by_key.get((index, "e1", "guide2048")) for index in NAMED_WINDOWS
    ]
    d1_complete = all(
        item is not None
        and len(item.get("profile", {}).get("per_position", [])) == N_TARGETS
        and len(item.get("profile", {}).get("prediction_sanity", [])) == 8
        for item in d1_receipts
    )
    d1_rows = []
    for index, item in zip(NAMED_WINDOWS, d1_receipts):
        if item is None:
            continue
        summary = item["profile"]["summary"]
        sanity = item["profile"]["prediction_sanity"]
        d1_rows.append(
            {
                "window_index": index,
                "mean_nll": float(summary["mean_nll"]),
                "ppl": float(summary["ppl"]),
                "shape_label": d1_shape_label(summary),
                "position_nll_correlation": float(
                    summary["position_nll_correlation"]
                ),
                "boundary_minus_remaining_mean_nll": float(
                    summary["boundary_minus_remaining_mean_nll"]
                ),
                "sample_actual_top5_count": sum(
                    bool(row["top5_contains_actual"]) for row in sanity
                ),
                "sample_previous_top5_count": sum(
                    bool(row["top5_contains_previous"]) for row in sanity
                ),
                "sample_next_top5_count": sum(
                    bool(row["top5_contains_next"]) for row in sanity
                ),
                "receipt": item["_path"],
                "receipt_sha256": item["_sha256"],
            }
        )

    controls = []
    for index in NAMED_WINDOWS:
        base_item = by_key.get((index, "e1", "guide2048"))
        for prefix_kind in ("wikitext2048", "guide256"):
            item = by_key.get((index, "e1", prefix_kind))
            row: dict[str, Any] = {
                "window_index": index,
                "prefix_kind": prefix_kind,
                "status": "complete" if item is not None else "missing",
            }
            if item is not None:
                row.update(
                    {
                        "mean_nll": float(item["arms"]["teacher"]["mean_nll"]),
                        "ppl": float(item["arms"]["teacher"]["ppl"]),
                        "target_ids_sha256": item["target_ids_sha256"],
                        "receipt": item["_path"],
                        "receipt_sha256": item["_sha256"],
                    }
                )
                if base_item is not None:
                    row["minus_guide2048_mean_nll"] = float(
                        item["arms"]["teacher"]["mean_nll"]
                        - base_item["arms"]["teacher"]["mean_nll"]
                    )
            controls.append(row)
    d3_complete = all(row["status"] == "complete" for row in controls)

    d4_rows = []
    for index in NAMED_WINDOWS:
        multi = by_key.get((index, "e1", "guide2048"))
        established = by_key.get((index, "established", "guide2048"))
        if multi is None or established is None:
            continue
        original_path = E1_OUTPUT / "eval_e11" / f"narrative_{index:03d}.json"
        original = read_json(original_path)
        left = np.asarray(
            [row["nll"] for row in multi["profile"]["per_position"]],
            dtype=np.float64,
        )
        right = np.asarray(
            [row["nll"] for row in established["profile"]["per_position"]],
            dtype=np.float64,
        )
        abs_delta = np.abs(right - left)
        hashes = {
            original["target_ids_sha256"],
            multi["target_ids_sha256"],
            established["target_ids_sha256"],
        }
        d4_rows.append(
            {
                "window_index": index,
                "hashes_match": len(hashes) == 1,
                "target_ids_sha256": multi["target_ids_sha256"],
                "original_base_mean_nll": float(
                    original["arms"]["base"]["mean_nll"]
                ),
                "original_teacher_mean_nll": float(
                    original["arms"]["teacher"]["mean_nll"]
                ),
                "e1_diag_mean_nll": float(
                    multi["arms"]["teacher"]["mean_nll"]
                ),
                "established_mean_nll": float(
                    established["arms"]["teacher"]["mean_nll"]
                ),
                "e1_diag_minus_original_mean_nll": float(
                    multi["arms"]["teacher"]["mean_nll"]
                    - original["arms"]["teacher"]["mean_nll"]
                ),
                "established_minus_e1_diag_mean_nll": float(
                    established["arms"]["teacher"]["mean_nll"]
                    - multi["arms"]["teacher"]["mean_nll"]
                ),
                "profile_abs_nll_delta_mean": float(abs_delta.mean()),
                "profile_abs_nll_delta_p99": float(
                    np.quantile(abs_delta, 0.99)
                ),
                "profile_abs_nll_delta_max": float(abs_delta.max()),
                "e1_receipt": multi["_path"],
                "e1_receipt_sha256": multi["_sha256"],
                "established_receipt": established["_path"],
                "established_receipt_sha256": established["_sha256"],
                "original_receipt": str(original_path),
                "original_receipt_sha256": sha256_file(original_path),
            }
        )
    d4_complete = len(d4_rows) == len(NAMED_WINDOWS)
    mechanism = classify_mechanism_rows(d4_rows)

    dg1_line = (
        "DG1 GREEN — D1 per-position NLL (511 targets) and D2 eight-position "
        "top-5 probes are complete for windows 11, 13, and 15."
        if d1_complete
        else "DG1 NOT_MEASURED — required D1/D2 GPU receipts for windows 11, 13, and 15 are incomplete."
    )
    dg2_line = (
        "DG2 GREEN — WikiText-2048 and guide-256 prefix controls are complete "
        "for windows 11, 13, and 15."
        if d3_complete
        else "DG2 NOT_MEASURED — D3 WikiText-2048 and/or guide-256 GPU control receipts are incomplete."
    )
    dg3_line = (
        "DG3 GREEN — established isolated long-context cross-checks and "
        "per-window deltas are complete for windows 11, 13, and 15."
        if d4_complete
        else "DG3 NOT_MEASURED — D4 established-path GPU cross-check receipts are incomplete."
    )
    if mechanism == "BUG":
        dg4_line = (
            "DG4 BUG — established isolation disagrees with the reproduced E1 "
            "multi-arm schedule; defect localized to eval_narrative block reuse/"
            "pre-synchronization scheduling (see D5 anchors)."
        )
    elif mechanism == "REAL":
        dg4_line = (
            "DG4 REAL — the established long-context path reproduces degradation "
            "on all three named windows; the E1.1 premise finding stands."
        )
    elif mechanism == "INDETERMINATE":
        dg4_line = (
            "DG4 NOT_MEASURED — runtime receipts are mixed and do not support a "
            "registered BUG or REAL verdict."
        )
    else:
        dg4_line = (
            "DG4 NOT_MEASURED — GPU D1-D4 evidence is incomplete; no mechanism "
            "verdict is issued from CPU/static evidence."
        )

    note = mechanism_note(mechanism, d4_rows, records)
    analysis = {
        "schema": "moe_e1_diag_analysis_v1",
        "created_at": now_iso(),
        "status": "complete" if mechanism in {"BUG", "REAL"} else "incomplete",
        "mechanism": mechanism,
        "evidence_class": "mechanism diagnosis on one model",
        "registered_windows": list(NAMED_WINDOWS),
        "agreement_thresholds": {
            "mean_nll_abs_atol": MEAN_NLL_AGREEMENT_ATOL,
            "profile_abs_nll_delta_p99_atol": PROFILE_NLL_P99_ATOL,
            "diagnostic_only_not_gate_semantics": True,
        },
        "dg": {
            "DG1": {
                "verdict": "GREEN" if d1_complete else "NOT_MEASURED",
                "report_line": dg1_line,
                "rows": d1_rows,
            },
            "DG2": {
                "verdict": "GREEN" if d3_complete else "NOT_MEASURED",
                "report_line": dg2_line,
                "controls": controls,
            },
            "DG3": {
                "verdict": "GREEN" if d4_complete else "NOT_MEASURED",
                "report_line": dg3_line,
                "rows": d4_rows,
            },
            "DG4": {
                "verdict": mechanism,
                "report_line": dg4_line,
            },
        },
        "mechanism_note": note,
        "d5_result_verbatim": D5_AUDIT_RESULT,
        "d5_audit": str(output / "d5_static_audit.json"),
        "d5_audit_sha256": sha256_file(output / "d5_static_audit.json"),
        "run_inventory": records,
        "gpu_resume_script": str(output / "GPU_RESUME_COMMANDS.sh"),
        "e12_resume_script": str(output / "GPU_E12_RESUME_COMMANDS.sh"),
        "original_e1_receipts_preserved": True,
        "limitations": [
            "CPU self-tests are synthetic and are not DG1-DG4 evidence.",
            "Mechanism evidence is limited to GPT-OSS-20B and three registered windows.",
        ],
    }
    analysis_path = output / "analysis.json"
    report_path = output / "MOE_E1_2_DIAG_REPORT.md"
    note_path = output / "MECHANISM_NOTE.md"
    write_json(analysis_path, analysis)
    write_text(report_path, render_report(analysis))
    write_text(note_path, note + "\n")
    if mechanism not in {"BUG", "REAL"}:
        blocked = {
            "schema": "moe_e1_diag_blocked_v1",
            "created_at": now_iso(),
            "status": "not_measured",
            "reason": "GPU D1-D4 receipts incomplete or diagnostically mixed",
            "mechanism": mechanism,
            "cuda_environment": cuda_probe(),
            "gpu_resume_script": str(output / "GPU_RESUME_COMMANDS.sh"),
            "gpu_resume_script_sha256": (
                sha256_file(output / "GPU_RESUME_COMMANDS.sh")
                if (output / "GPU_RESUME_COMMANDS.sh").is_file()
                else None
            ),
        }
        write_json(output / "blocked.json", blocked)
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "mechanism": mechanism,
                "dg": {
                    name: value["verdict"]
                    for name, value in analysis["dg"].items()
                },
                "analysis": str(analysis_path),
                "report": str(report_path),
                "resume": analysis["gpu_resume_script"],
            }
        ),
        flush=True,
    )
    return 0


def load_e12_receipts(
    kind: str,
    indices: Sequence[int],
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    complete: list[dict[str, Any]] = []
    missing: list[int] = []
    errors: list[int] = []
    for index in indices:
        path = e1.eval_receipt_path(
            E1_OUTPUT,
            kind,
            index,
            e1.ADDRESS_RULE_E11,
            e1.EVAL_FIX_E12,
        )
        if not path.is_file():
            missing.append(int(index))
            continue
        payload = read_json(path)
        if (
            payload.get("status") == "complete"
            and payload.get("eval_fix") == e1.EVAL_FIX_E12
        ):
            payload["_path"] = str(path)
            payload["_sha256"] = sha256_file(path)
            complete.append(payload)
        else:
            errors.append(int(index))
    return complete, missing, errors


def analyze_e12(args: argparse.Namespace) -> int:
    output = ensure_output(args.output_dir)
    diagnostic_path = output / "analysis.json"
    if not diagnostic_path.is_file():
        raise FileNotFoundError("run the D1-D4 analyze mode before analyze-e12")
    diagnostic = read_json(diagnostic_path)
    if diagnostic.get("mechanism") != "BUG":
        raise RuntimeError(
            "fixed-path aggregation is registered only after MECHANISM=BUG; "
            f"diagnostic currently says {diagnostic.get('mechanism')!r}"
        )

    narrative, narrative_missing, narrative_errors = load_e12_receipts(
        "narrative", range(e1.N_BEHAVIORAL_WINDOWS)
    )
    generic, generic_missing, generic_errors = load_e12_receipts(
        "generic", e1.EVAL_INDICES
    )
    code, code_missing, code_errors = load_e12_receipts(
        "code", e1.EVAL_INDICES
    )

    g4: dict[str, Any] = {
        "complete_windows": len(narrative),
        "missing_windows": narrative_missing,
        "error_windows": narrative_errors,
        "verdict": "NOT_MEASURED",
    }
    g4_row = (
        "G4 row (e12): ppl_base=n/a, ppl_teacher=n/a, ppl_expert=n/a, "
        "recovery=n/a, bootstrap 95% CI=n/a"
    )
    if len(narrative) == e1.N_BEHAVIORAL_WINDOWS and not narrative_errors:
        base_nll = np.asarray(
            [item["arms"]["base"]["mean_nll"] for item in narrative],
            dtype=np.float64,
        )
        teacher_nll = np.asarray(
            [item["arms"]["teacher"]["mean_nll"] for item in narrative],
            dtype=np.float64,
        )
        expert_nll = np.asarray(
            [item["arms"]["expert"]["mean_nll"] for item in narrative],
            dtype=np.float64,
        )
        recovery = e1.bootstrap_recovery(
            base_nll,
            teacher_nll,
            expert_nll,
            resamples=e1.DEFAULT_BOOTSTRAP_RESAMPLES,
            seed=e1.DEFAULT_SEED + 404,
        )
        premise_failed = bool(recovery["teacher_gap"] <= 0.0)
        if premise_failed:
            verdict = "RED"
            verdict_reason = (
                "fixed isolated teacher prefix still does not improve heldout "
                "perplexity; E1.1 premise finding stands"
            )
        else:
            verdict = (
                "GREEN"
                if recovery["recovery_fraction"] >= e1.G4_RECOVERY_FLOOR
                else "RED"
            )
            verdict_reason = (
                "expert recovery meets the registered 25 percent floor"
                if verdict == "GREEN"
                else "expert recovery is below the registered 25 percent floor"
            )
        g4.update(
            {
                **recovery,
                "verdict": verdict,
                "premise_failed": premise_failed,
                "verdict_reason": verdict_reason,
            }
        )
        g4_row = (
            f"G4 row (e12): ppl_base={recovery['ppl_base']:.6f}, "
            f"ppl_teacher={recovery['ppl_teacher']:.6f}, "
            f"ppl_expert={recovery['ppl_expert']:.6f}, "
            f"recovery={recovery['recovery_percent']:.3f}%, bootstrap 95% CI="
            f"[{recovery['ci95_low_percent']:.3f}%, "
            f"{recovery['ci95_high_percent']:.3f}%]"
        )

    g5: dict[str, Any] = {
        "generic_complete_windows": len(generic),
        "code_complete_windows": len(code),
        "generic_missing": generic_missing,
        "code_missing": code_missing,
        "generic_errors": generic_errors,
        "code_errors": code_errors,
        "verdict": "NOT_MEASURED",
    }
    g5_line = (
        "G5' NOT_MEASURED (e12) — fixed WikiText/code receipts are incomplete."
    )
    if (
        len(generic) == len(e1.EVAL_INDICES)
        and len(code) == len(e1.EVAL_INDICES)
        and not generic_errors
        and not code_errors
    ):
        generic_base = e1.aggregate_nll(generic, "base")
        generic_expert = e1.aggregate_nll(generic, "expert")
        ppl_delta_pct = (
            (generic_expert["ppl"] - generic_base["ppl"])
            / generic_base["ppl"]
            * 100.0
        )
        generic_fires = sum(int(item["fire"]["fire_count"]) for item in generic)
        generic_tokens = sum(int(item["fire"]["token_count"]) for item in generic)
        code_fires = sum(int(item["fire"]["fire_count"]) for item in code)
        code_tokens = sum(int(item["fire"]["token_count"]) for item in code)
        generic_rate = generic_fires / generic_tokens
        code_rate = code_fires / code_tokens
        qualified = e1.g5_qualifies(
            address_rule=e1.ADDRESS_RULE_E11,
            wikitext_ppl_delta_percent=ppl_delta_pct,
            generic_fire_rate=generic_rate,
            code_fire_rate=code_rate,
        )
        verdict = "GREEN" if qualified else "RED"
        per_window = []
        for item in generic:
            base_ppl = float(item["arms"]["base"]["ppl"])
            expert_ppl = float(item["arms"]["expert"]["ppl"])
            per_window.append(
                {
                    "window_index": int(item["window_index"]),
                    "registered_high_firing_narrativeish": (
                        int(item["window_index"])
                        in e1.E11_HIGH_FIRING_WIKITEXT_WINDOWS
                    ),
                    "fire_rate": float(item["fire"]["fire_rate"]),
                    "ppl_base": base_ppl,
                    "ppl_expert": expert_ppl,
                    "ppl_delta_percent": (
                        (expert_ppl - base_ppl) / base_ppl * 100.0
                    ),
                    "receipt": item["_path"],
                    "receipt_sha256": item["_sha256"],
                }
            )
        g5.update(
            {
                "verdict": verdict,
                "wikitext_base": generic_base,
                "wikitext_expert": generic_expert,
                "wikitext_ppl_delta_percent": ppl_delta_pct,
                "generic_fire_rate_descriptive": generic_rate,
                "code_fire_rate": code_rate,
                "per_window_wikitext_by_descending_fire_rate": sorted(
                    per_window,
                    key=lambda row: (-row["fire_rate"], row["window_index"]),
                ),
                "registered_rule_unchanged": {
                    "wikitext_overall_ppl_delta_percent_lte": (
                        e1.G5_WIKITEXT_PPL_DELTA_CAP_PCT
                    ),
                    "code_fire_rate_lte": e1.G5_FIRE_CAPS["code"],
                    "generic_fire_rate": "descriptive_no_pass_fail_bound",
                },
            }
        )
        g5_line = (
            f"G5' {verdict} (e12) — WikiText ppl delta={ppl_delta_pct:.4f}% "
            f"(cap 0.5%); code fire={code_rate:.6f} (cap 0.05); "
            f"descriptive generic fire={generic_rate:.6f}."
        )

    status = (
        "complete"
        if g4["verdict"] != "NOT_MEASURED"
        and g5["verdict"] != "NOT_MEASURED"
        else "incomplete"
    )
    payload = {
        "schema": "moe_e1_diag_e12_fixed_rerun_analysis_v1",
        "created_at": now_iso(),
        "status": status,
        "eval_fix": e1.EVAL_FIX_E12,
        "mechanism_prerequisite": "BUG",
        "diagnostic_analysis": str(diagnostic_path),
        "diagnostic_analysis_sha256": sha256_file(diagnostic_path),
        "registered_gate_semantics_changed": False,
        "original_eval_root": str(E1_OUTPUT / "eval_e11"),
        "fixed_eval_root": str(E1_OUTPUT / "eval_e11_e12"),
        "g4": g4,
        "g4_row": g4_row,
        "g5": g5,
        "g5_report_line": g5_line,
        "evidence_class": "mechanism diagnosis on one model",
    }
    path = output / "e12_fixed_rerun_analysis.json"
    report = output / "MOE_E1_2_E12_FIXED_RERUN_REPORT.md"
    write_json(path, payload)
    write_text(
        report,
        "\n".join(
            [
                "# MOE-E1.2 e12 Fixed-Path Rerun",
                "",
                f"Generated: {payload['created_at']}",
                "",
                "Registered G4/G5' definitions are unchanged.",
                "",
                g4_row,
                g5_line,
                "",
                f"Fixed receipt root: {payload['fixed_eval_root']}",
                "",
            ]
        ),
    )
    print(
        json.dumps(
            {
                "status": status,
                "g4": g4["verdict"],
                "g5_prime": g5["verdict"],
                "analysis": str(path),
                "report": str(report),
            }
        ),
        flush=True,
    )
    return 0


def main() -> int:
    args = parse_args()
    try:
        if args.mode != "score" and any(
            value is not None
            for value in (args.window_index, args.prefix_kind, args.path_kind)
        ):
            raise ValueError("window/prefix/path selectors are valid only for score")
        if args.mode == "self-test":
            return self_test(args)
        if args.mode == "audit":
            return audit(args)
        if args.mode == "score":
            return score(args)
        if args.mode == "analyze":
            return analyze(args)
        if args.mode == "analyze-e12":
            return analyze_e12(args)
        raise AssertionError(args.mode)
    except SystemExit:
        raise
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "mode": args.mode,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
