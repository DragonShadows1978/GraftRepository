#!/usr/bin/env python3
"""GLC P0.b: aligned HF-vs-port first-divergence capture and analysis.

Captures seven corresponding decoder boundaries for every layer while retaining
only registered probe positions.  Port capture is GPU-only and layer-streamed;
HF capture uses the explicit offline CPU/dequantized reference route from P0.a.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import signal
import stat
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import glc_p0_common as common


DIV_OUTPUT = common.OUTPUT_ROOT / "divergence"
N_LAYERS = 24
HIDDEN_DIM = 2880
OP_NAMES = (
    "layer_input",
    "attention_norm",
    "attention_output",
    "post_attention_residual",
    "mlp_norm",
    "mlp_output",
    "layer_output",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("preflight", "self-test", "capture-port", "capture-hf", "analyze"),
    )
    parser.add_argument("--sequence", choices=("short", "long", "both"), default="both")
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--model-dir", type=Path, default=common.MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=common.OUTPUT_ROOT)
    parser.add_argument("--offload-dir", type=Path, default=Path("/tmp/glc_p0_hf_div_offload"))
    parser.add_argument("--cpu-memory", default="42GiB")
    parser.add_argument("--threads", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def capture_paths(backend: str, sequence: str, attempt: int) -> tuple[Path, Path]:
    stem = f"{backend}_{sequence}_attempt{int(attempt):02d}"
    return DIV_OUTPUT / f"{stem}.npz", DIV_OUTPUT / f"{stem}.json"


def max_rss_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _timeout_handler(signum, frame):  # type: ignore[no-untyped-def]
    del signum, frame
    raise TimeoutError("received SIGTERM from the bounded run wrapper")


def cuda_probe() -> dict[str, Any]:
    nodes: list[str] = []
    for path in sorted(Path("/dev").glob("nvidia*")):
        try:
            if stat.S_ISCHR(path.stat().st_mode):
                nodes.append(str(path))
        except OSError:
            pass
    return {
        "character_device_nodes": nodes,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def require_cuda() -> dict[str, Any]:
    probe = cuda_probe()
    required = {"/dev/nvidia0", "/dev/nvidiactl"}
    if not required.issubset(set(probe["character_device_nodes"])):
        raise RuntimeError(
            "TensorCUDA port capture requires GPU 0 under the registered flock/590s wrapper"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("port capture requires CUDA_VISIBLE_DEVICES=0")
    return probe


def _sequences(selection: str) -> tuple[str, ...]:
    return ("short", "long") if selection == "both" else (selection,)


def preflight(args: argparse.Namespace) -> int:
    common.ensure_output_root(args.output_dir)
    common.ensure_model_dir(args.model_dir)
    _manifest, arrays = common.validate_sealed_controls()
    wiki = arrays["wikitext_0_2560"]
    short = common.divergence_sequence(wiki, "short")
    long = common.divergence_sequence(wiki, "long")
    if not np.array_equal(short["input_ids"][0], long["input_ids"][0, :512]):
        raise RuntimeError("short sequence is not the exact long-sequence prefix")
    print(
        json.dumps(
            {
                "status": "ready",
                "model_revision": common.MODEL_REVISION,
                "control_arrays_sha256": common.sha256_file(common.CONTROL_ARRAYS),
                "short": {
                    "length": short["length"],
                    "positions": short["positions"].tolist(),
                    "input_ids_sha256": short["input_ids_sha256"],
                },
                "long": {
                    "length": long["length"],
                    "positions": long["positions"].tolist(),
                    "input_ids_sha256": long["input_ids_sha256"],
                },
                "op_names": list(OP_NAMES),
                "runtime": common.runtime_facts(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _tc_sample(tc: Any, tensor: Any, positions: np.ndarray) -> np.ndarray:
    pieces = [tensor.slice(1, int(position), 1) for position in positions.tolist()]
    sampled = pieces[0] if len(pieces) == 1 else tc.cat(pieces, dim=1)
    return np.ascontiguousarray(sampled.float().numpy()[0], dtype=np.float32)


def _cast_port(tensor: Any, compute_dtype: str) -> Any:
    return tensor.half() if compute_dtype == "float16" else tensor.astype(compute_dtype)


def _capture_payload_base(
    *, backend: str, data: dict[str, Any], attempt: int, model_dir: Path
) -> dict[str, Any]:
    return {
        "schema": "glc_p0b_aligned_capture_v1",
        "created_at": common.now_iso(),
        "status": "starting",
        "backend": backend,
        "sequence": data["sequence"],
        "sequence_length": int(data["length"]),
        "positions": data["positions"].tolist(),
        "input_ids_sha256": data["input_ids_sha256"],
        "attempt": int(attempt),
        "model_dir": str(model_dir),
        "model_revision": common.MODEL_REVISION,
        "control_arrays": str(common.CONTROL_ARRAYS),
        "control_arrays_sha256": common.sha256_file(common.CONTROL_ARRAYS),
        "op_names": list(OP_NAMES),
        "layers": N_LAYERS,
        "hidden_dim": HIDDEN_DIM,
        "argv": sys.argv,
    }


def capture_port(args: argparse.Namespace) -> int:
    common.ensure_output_root(args.output_dir)
    model_dir = common.ensure_model_dir(args.model_dir)
    if args.sequence == "both":
        raise ValueError("capture-port is one GPU invocation; choose --sequence short or long")
    if args.attempt < 0:
        raise ValueError("--attempt must be nonnegative")
    _manifest, arrays = common.validate_sealed_controls()
    data = common.divergence_sequence(arrays["wikitext_0_2560"], args.sequence)
    npz_path, meta_path = capture_paths("port", args.sequence, args.attempt)
    if npz_path.exists() or meta_path.exists():
        if meta_path.exists() and common.read_json(meta_path).get("status") == "complete":
            print(json.dumps({"status": "already_complete", "receipt": str(meta_path)}))
            return 0
        raise FileExistsError("append-only capture attempt exists; choose --attempt")

    receipt = _capture_payload_base(
        backend="port_tensorcuda_standard", data=data, attempt=args.attempt, model_dir=model_dir
    )
    receipt["required_gpu_wrapper"] = (
        "flock -w 7200 /tmp/forge-gpu.lock timeout --signal=TERM --kill-after=5s 590s "
        "env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
    )
    started = time.perf_counter()
    signal.signal(signal.SIGTERM, _timeout_handler)
    completed_layers = 0
    try:
        receipt["cuda_environment"] = require_cuda()
        import gpt_oss20b_expert_e1 as e1

        runtime = e1.load_runtime()
        tc = runtime["tc"]
        cfg = runtime["GptOss20BConfig"].from_model_dir(model_dir)
        e1.validate_model_contract(cfg)
        where = runtime["build_safetensors_map"](model_dir)
        compute_dtype = str(runtime["BlockTC"].COMPUTE_DTYPE)
        states = np.empty(
            (N_LAYERS, len(OP_NAMES), data["positions"].size, HIDDEN_DIM),
            dtype=np.float32,
        )
        layer_rows: list[dict[str, Any]] = []
        with tc.no_grad():
            embed = runtime["GptOssRowEmbedding"](where)
            hidden = embed(data["input_ids"])
            cos, sin = runtime["gpt_oss_yarn_rope_tables"](cfg, data["length"])
            for layer_index in range(N_LAYERS):
                layer_started = time.perf_counter()
                block = runtime["GptOssDiagnosticBlockTC"].from_safetensors(
                    cfg, where, layer_index, expert_mode="resident_packed_mxfp4"
                )
                e1.configure_block(block)
                if block.self_attn.attention_mode != "standard":
                    raise RuntimeError("P0.b port capture left registered standard attention")

                states[layer_index, 0] = _tc_sample(tc, hidden, data["positions"])
                normed = _cast_port(block.input_layernorm(hidden), compute_dtype)
                states[layer_index, 1] = _tc_sample(tc, normed, data["positions"])
                attention_output, kv = block.self_attn(normed, cos, sin)
                states[layer_index, 2] = _tc_sample(
                    tc, attention_output, data["positions"]
                )
                post_attention = hidden + attention_output
                states[layer_index, 3] = _tc_sample(
                    tc, post_attention, data["positions"]
                )
                mlp_norm = _cast_port(
                    block.post_attention_layernorm(post_attention), compute_dtype
                )
                states[layer_index, 4] = _tc_sample(tc, mlp_norm, data["positions"])
                mlp_output, route = block.mlp(mlp_norm)
                states[layer_index, 5] = _tc_sample(tc, mlp_output, data["positions"])
                hidden = post_attention + mlp_output
                states[layer_index, 6] = _tc_sample(tc, hidden, data["positions"])
                tc.synchronize()
                layer_rows.append(
                    {
                        "layer": layer_index,
                        "layer_type": cfg.layer_types[layer_index],
                        "attention_backend": block.self_attn.last_attention_backend,
                        "wall_seconds": float(time.perf_counter() - layer_started),
                    }
                )
                completed_layers = layer_index + 1
                del block, normed, attention_output, post_attention, mlp_norm, mlp_output, kv, route
                gc.collect()
                if hasattr(tc, "empty_cache"):
                    tc.empty_cache()
        common.write_new_npz(
            npz_path,
            states=states,
            positions=np.ascontiguousarray(data["positions"], dtype=np.int64),
            op_names=np.asarray(OP_NAMES),
        )
        receipt.update(
            {
                "status": "complete",
                "arrays": str(npz_path),
                "arrays_sha256": common.sha256_file(npz_path),
                "states_shape": list(states.shape),
                "states_dtype": str(states.dtype),
                "compute_dtype": compute_dtype,
                "expert_mode": "resident_packed_mxfp4",
                "attention_mode": "standard",
                "layer_rows": layer_rows,
                "completed_layers": completed_layers,
                "wall_seconds": float(time.perf_counter() - started),
                "max_rss_mib": max_rss_mib(),
            }
        )
        common.write_new_json(meta_path, receipt)
        print(json.dumps({"status": "complete", "receipt": str(meta_path)}))
        return 0
    except BaseException as exc:
        receipt.update(
            {
                "status": "error",
                "completed_layers": completed_layers,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "wall_seconds": float(time.perf_counter() - started),
            }
        )
        if not meta_path.exists():
            common.write_new_json(meta_path, receipt)
        raise


def _first_tensor(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value
    except Exception:
        pass
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _hf_capture_one(model: Any, data: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    positions_t = torch.as_tensor(data["positions"], dtype=torch.long)
    captured: dict[tuple[int, str], np.ndarray] = {}
    handles: list[Any] = []

    def store(layer: int, op: str, value: Any) -> None:
        tensor = _first_tensor(value)
        if tensor is None:
            raise RuntimeError(f"HF hook {layer}/{op} returned no tensor")
        local_positions = positions_t.to(tensor.device)
        rows = tensor.index_select(1, local_positions)[0]
        captured[(layer, op)] = np.ascontiguousarray(
            rows.detach().to(torch.float32).cpu().numpy(), dtype=np.float32
        )

    def pre_hook(layer: int, op: str) -> Callable[..., None]:
        def hook(module, inputs):  # type: ignore[no-untyped-def]
            del module
            store(layer, op, inputs[0])

        return hook

    def post_hook(layer: int, op: str) -> Callable[..., None]:
        def hook(module, inputs, output):  # type: ignore[no-untyped-def]
            del module, inputs
            store(layer, op, output)

        return hook

    for layer_index, layer in enumerate(model.model.layers):
        handles.extend(
            [
                layer.register_forward_pre_hook(pre_hook(layer_index, "layer_input")),
                layer.input_layernorm.register_forward_hook(
                    post_hook(layer_index, "attention_norm")
                ),
                layer.self_attn.register_forward_hook(
                    post_hook(layer_index, "attention_output")
                ),
                layer.post_attention_layernorm.register_forward_pre_hook(
                    pre_hook(layer_index, "post_attention_residual")
                ),
                layer.post_attention_layernorm.register_forward_hook(
                    post_hook(layer_index, "mlp_norm")
                ),
                layer.mlp.register_forward_hook(post_hook(layer_index, "mlp_output")),
                layer.register_forward_hook(post_hook(layer_index, "layer_output")),
            ]
        )

    started = time.perf_counter()
    try:
        ids = torch.from_numpy(data["input_ids"])
        with torch.inference_mode():
            output = model.model(input_ids=ids, use_cache=False, return_dict=True)
        last_shape = [int(value) for value in output.last_hidden_state.shape]
        last_dtype = str(output.last_hidden_state.dtype)
    finally:
        for handle in handles:
            handle.remove()
    expected = {(layer, op) for layer in range(N_LAYERS) for op in OP_NAMES}
    missing = sorted(expected - set(captured))
    if missing:
        raise RuntimeError(f"HF hooks missed {len(missing)} states: {missing[:5]}")
    states = np.stack(
        [
            np.stack([captured[(layer, op)] for op in OP_NAMES], axis=0)
            for layer in range(N_LAYERS)
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    if states.shape != (N_LAYERS, len(OP_NAMES), data["positions"].size, HIDDEN_DIM):
        raise RuntimeError(f"unexpected HF capture shape {states.shape}")
    return states, {
        "forward_wall_seconds": float(time.perf_counter() - started),
        "forward_last_hidden_shape": last_shape,
        "forward_last_hidden_dtype": last_dtype,
    }


def capture_hf(args: argparse.Namespace) -> int:
    common.ensure_output_root(args.output_dir)
    model_dir = common.ensure_model_dir(args.model_dir)
    if args.attempt < 0:
        raise ValueError("--attempt must be nonnegative")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    _manifest, arrays = common.validate_sealed_controls()
    pending: list[tuple[str, dict[str, Any], Path, Path]] = []
    for sequence in _sequences(args.sequence):
        data = common.divergence_sequence(arrays["wikitext_0_2560"], sequence)
        npz_path, meta_path = capture_paths("hf", sequence, args.attempt)
        if npz_path.exists() or meta_path.exists():
            if meta_path.exists() and common.read_json(meta_path).get("status") == "complete":
                continue
            raise FileExistsError(
                f"append-only HF capture exists for {sequence}; choose --attempt"
            )
        pending.append((sequence, data, npz_path, meta_path))
    if not pending:
        print(json.dumps({"status": "already_complete"}))
        return 0

    signal.signal(signal.SIGTERM, _timeout_handler)
    overall_started = time.perf_counter()
    model = None
    current: tuple[str, dict[str, Any], Path, Path] | None = None
    try:
        import torch

        torch.set_num_threads(args.threads)
        load_started = time.perf_counter()
        model, load_info = common.load_hf_reference_model(
            model_dir=model_dir,
            cpu_memory=args.cpu_memory,
            offload_dir=args.offload_dir,
            for_causal_lm=True,
        )
        load_wall = float(time.perf_counter() - load_started)
        for current in pending:
            sequence, data, npz_path, meta_path = current
            receipt = _capture_payload_base(
                backend="hf_transformers_eager",
                data=data,
                attempt=args.attempt,
                model_dir=model_dir,
            )
            sequence_started = time.perf_counter()
            try:
                states, forward_info = _hf_capture_one(model, data)
                common.write_new_npz(
                    npz_path,
                    states=states,
                    positions=np.ascontiguousarray(data["positions"], dtype=np.int64),
                    op_names=np.asarray(OP_NAMES),
                )
                receipt.update(
                    {
                        "status": "complete",
                        "load": {**load_info, "wall_seconds": load_wall},
                        "arrays": str(npz_path),
                        "arrays_sha256": common.sha256_file(npz_path),
                        "states_shape": list(states.shape),
                        "states_dtype": str(states.dtype),
                        **forward_info,
                        "wall_seconds_since_process_start": float(
                            time.perf_counter() - overall_started
                        ),
                        "sequence_wall_seconds": float(
                            time.perf_counter() - sequence_started
                        ),
                        "max_rss_mib": max_rss_mib(),
                    }
                )
                common.write_new_json(meta_path, receipt)
            except BaseException as exc:
                receipt.update(
                    {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "sequence_wall_seconds": float(
                            time.perf_counter() - sequence_started
                        ),
                    }
                )
                if not meta_path.exists():
                    common.write_new_json(meta_path, receipt)
                raise
        print(
            json.dumps(
                {
                    "status": "complete",
                    "receipts": [str(item[3]) for item in pending],
                    "load_wall_seconds": load_wall,
                }
            )
        )
        return 0
    except BaseException as exc:
        # A load failure occurs before there is a current per-sequence receipt.
        if current is None:
            for sequence, data, _npz_path, meta_path in pending:
                if meta_path.exists():
                    continue
                receipt = _capture_payload_base(
                    backend="hf_transformers_eager",
                    data=data,
                    attempt=args.attempt,
                    model_dir=model_dir,
                )
                receipt.update(
                    {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "wall_seconds": float(time.perf_counter() - overall_started),
                    }
                )
                common.write_new_json(meta_path, receipt)
        raise
    finally:
        del model
        gc.collect()


def relative_frobenius(port: np.ndarray, hf: np.ndarray) -> float:
    left = port.astype(np.float64, copy=False)
    right = hf.astype(np.float64, copy=False)
    denominator = max(float(np.linalg.norm(right.ravel())), 1.0e-30)
    return float(np.linalg.norm((left - right).ravel()) / denominator)


def ratio_with_floor(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1.0e-30))


def _analyze_arrays(
    *,
    port_short: np.ndarray,
    hf_short: np.ndarray,
    port_long: np.ndarray,
    hf_long: np.ndarray,
) -> dict[str, Any]:
    expected_short = (N_LAYERS, len(OP_NAMES), len(common.SHORT_PROBE_POSITIONS), HIDDEN_DIM)
    expected_long = (N_LAYERS, len(OP_NAMES), len(common.LONG_PROBE_POSITIONS), HIDDEN_DIM)
    for name, array, expected in (
        ("port_short", port_short, expected_short),
        ("hf_short", hf_short, expected_short),
        ("port_long", port_long, expected_long),
        ("hf_long", hf_long, expected_long),
    ):
        if array.shape != expected:
            raise RuntimeError(f"{name} shape {array.shape} != {expected}")

    long_positions = np.asarray(common.LONG_PROBE_POSITIONS)
    prefix_indices = np.flatnonzero(long_positions <= 500)
    tail_indices = np.flatnonzero(long_positions > 500)
    rows: list[dict[str, Any]] = []
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for layer in range(N_LAYERS):
        for op_index, op in enumerate(OP_NAMES):
            short_dev = relative_frobenius(
                port_short[layer, op_index], hf_short[layer, op_index]
            )
            long_prefix_dev = relative_frobenius(
                port_long[layer, op_index, prefix_indices],
                hf_long[layer, op_index, prefix_indices],
            )
            long_tail_dev = relative_frobenius(
                port_long[layer, op_index, tail_indices],
                hf_long[layer, op_index, tail_indices],
            )
            ratio = ratio_with_floor(long_tail_dev, short_dev)
            per_position = []
            crossing_positions: list[int] = []
            for pos_index, position in enumerate(common.LONG_PROBE_POSITIONS):
                deviation = relative_frobenius(
                    port_long[layer, op_index, pos_index],
                    hf_long[layer, op_index, pos_index],
                )
                position_ratio = ratio_with_floor(deviation, short_dev)
                if position > 500 and position_ratio > common.LONG_TO_SHORT_DEVIATION_RATIO:
                    crossing_positions.append(int(position))
                per_position.append(
                    {
                        "position": int(position),
                        "relative_frobenius_deviation": deviation,
                        "ratio_to_short_position_deviation": position_ratio,
                    }
                )
            row = {
                "layer": layer,
                "operation": op,
                "short_position_relative_frobenius_deviation": short_dev,
                "long_prefix_relative_frobenius_deviation": long_prefix_dev,
                "long_position_relative_frobenius_deviation": long_tail_dev,
                "long_to_short_deviation_ratio": ratio,
                "exceeds_registered_10x": bool(
                    ratio > common.LONG_TO_SHORT_DEVIATION_RATIO
                ),
                "crossing_long_positions": crossing_positions,
                "per_long_capture_position": per_position,
            }
            rows.append(row)
            by_key[(layer, op)] = row

    layer_output_rows = [by_key[(layer, "layer_output")] for layer in range(N_LAYERS)]
    first_layer = next(
        (row for row in layer_output_rows if row["exceeds_registered_10x"]), None
    )
    execution_states: list[dict[str, Any]] = [by_key[(0, "layer_input")]]
    for layer in range(N_LAYERS):
        execution_states.extend(by_key[(layer, op)] for op in OP_NAMES[1:])
    first_op = next(
        (row for row in execution_states if row["exceeds_registered_10x"]), None
    )

    prefix_consistency = {
        "hf_short_vs_long_prefix_relative_frobenius": relative_frobenius(
            hf_short[:, -1], hf_long[:, -1, prefix_indices]
        ),
        "port_short_vs_long_prefix_relative_frobenius": relative_frobenius(
            port_short[:, -1], port_long[:, -1, prefix_indices]
        ),
    }
    if first_layer is None:
        hg2_status = "UNRESOLVED_NO_10X_LONG_SPECIFIC_DIVERGENCE"
        localization = None
    else:
        hg2_status = "LOCALIZED"
        localization = {
            "first_layer": int(first_layer["layer"]),
            "layer_output_long_to_short_ratio": float(
                first_layer["long_to_short_deviation_ratio"]
            ),
            "first_operation_boundary": (
                None if first_op is None else str(first_op["operation"])
            ),
            "first_operation_layer": (
                None if first_op is None else int(first_op["layer"])
            ),
            "first_operation_long_to_short_ratio": (
                None
                if first_op is None
                else float(first_op["long_to_short_deviation_ratio"])
            ),
            "position_band_probes": (
                [] if first_op is None else first_op["crossing_long_positions"]
            ),
        }
    return {
        "hg2_status": hg2_status,
        "registered_ratio_threshold": common.LONG_TO_SHORT_DEVIATION_RATIO,
        "short_position_definition": list(common.SHORT_PROBE_POSITIONS),
        "long_position_definition": [1100, 2100, 2500],
        "relative_frobenius_denominator_floor": 1.0e-30,
        "localization": localization,
        "prefix_consistency": prefix_consistency,
        "layer_outputs": layer_output_rows,
        "operation_rows": rows,
    }


def _load_capture(backend: str, sequence: str) -> tuple[Path, dict[str, Any], np.ndarray]:
    found = common.latest_complete_receipt(DIV_OUTPUT, f"{backend}_{sequence}")
    if found is None:
        raise FileNotFoundError(f"no complete {backend}/{sequence} capture")
    meta_path, receipt = found
    arrays_path = Path(receipt["arrays"])
    if common.sha256_file(arrays_path) != receipt["arrays_sha256"]:
        raise RuntimeError(f"capture payload hash mismatch: {arrays_path}")
    with np.load(arrays_path, allow_pickle=False) as stored:
        states = np.ascontiguousarray(stored["states"], dtype=np.float32)
        positions = tuple(int(value) for value in stored["positions"].tolist())
        op_names = tuple(str(value) for value in stored["op_names"].tolist())
    expected_positions = (
        common.SHORT_PROBE_POSITIONS if sequence == "short" else common.LONG_PROBE_POSITIONS
    )
    if positions != expected_positions or op_names != OP_NAMES:
        raise RuntimeError(f"capture alignment metadata mismatch: {arrays_path}")
    return meta_path, receipt, states


def _analysis_markdown(core: dict[str, Any]) -> str:
    analysis = core["analysis"]
    lines = [
        "# GLC P0.b First-Divergence Analysis",
        "",
        "Evidence class: aligned hidden-state measurement on the pinned model and sealed WikiText prefix.",
        "",
        f"HG2 status: **{analysis['hg2_status']}**.",
        "",
    ]
    localization = analysis["localization"]
    if localization is None:
        lines.append(
            "No decoder layer crossed the registered long/short relative-deviation ratio of 10; this does not assert exact parity."
        )
    else:
        lines.extend(
            [
                f"First layer-output crossing: layer {localization['first_layer']} "
                f"({localization['layer_output_long_to_short_ratio']:.9g}x).",
                "",
                f"First operation-boundary crossing: layer {localization['first_operation_layer']} "
                f"`{localization['first_operation_boundary']}` "
                f"({localization['first_operation_long_to_short_ratio']:.9g}x), "
                f"at long probes {localization['position_band_probes']}.",
            ]
        )
    lines.extend(
        [
            "",
            "| Layer | short-position rel-F | long-position rel-F | long/short | >10x | crossing long probes |",
            "|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in analysis["layer_outputs"]:
        lines.append(
            f"| {row['layer']} | {row['short_position_relative_frobenius_deviation']:.9g} | "
            f"{row['long_position_relative_frobenius_deviation']:.9g} | "
            f"{row['long_to_short_deviation_ratio']:.9g} | "
            f"{'yes' if row['exceeds_registered_10x'] else 'no'} | "
            f"{row['crossing_long_positions']} |"
        )
    if localization is not None:
        focus = int(localization["first_layer"])
        lines.extend(
            [
                "",
                f"## Operation boundaries at first layer {focus}",
                "",
                "| Operation | short rel-F | long rel-F | ratio | >10x |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for row in analysis["operation_rows"]:
            if row["layer"] != focus:
                continue
            lines.append(
                f"| {row['operation']} | {row['short_position_relative_frobenius_deviation']:.9g} | "
                f"{row['long_position_relative_frobenius_deviation']:.9g} | "
                f"{row['long_to_short_deviation_ratio']:.9g} | "
                f"{'yes' if row['exceeds_registered_10x'] else 'no'} |"
            )
    lines.append("")
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> int:
    common.ensure_output_root(args.output_dir)
    _manifest, _arrays = common.validate_sealed_controls()
    try:
        loaded = {
            (backend, sequence): _load_capture(backend, sequence)
            for backend in ("port", "hf")
            for sequence in ("short", "long")
        }
    except FileNotFoundError as exc:
        print(json.dumps({"status": "incomplete", "hg2_status": "NOT_MEASURED", "error": str(exc)}))
        return 2 if args.require_complete else 0

    input_hashes = {
        sequence: {
            loaded[(backend, sequence)][1]["input_ids_sha256"]
            for backend in ("port", "hf")
        }
        for sequence in ("short", "long")
    }
    for sequence, hashes in input_hashes.items():
        if len(hashes) != 1:
            raise RuntimeError(f"port/HF {sequence} input hashes differ: {hashes}")
    analysis = _analyze_arrays(
        port_short=loaded[("port", "short")][2],
        hf_short=loaded[("hf", "short")][2],
        port_long=loaded[("port", "long")][2],
        hf_long=loaded[("hf", "long")][2],
    )
    receipts = []
    for backend in ("port", "hf"):
        for sequence in ("short", "long"):
            path, payload, _states = loaded[(backend, sequence)]
            receipts.append(
                {
                    "backend": backend,
                    "sequence": sequence,
                    "path": str(path),
                    "sha256": common.sha256_file(path),
                    "arrays": payload["arrays"],
                    "arrays_sha256": payload["arrays_sha256"],
                    "wall_seconds": float(
                        payload.get("wall_seconds", payload.get("sequence_wall_seconds", 0.0))
                    ),
                }
            )
    core = {
        "schema": "glc_p0b_first_divergence_analysis_v1",
        "status": "complete",
        "model_revision": common.MODEL_REVISION,
        "control_arrays_sha256": common.sha256_file(common.CONTROL_ARRAYS),
        "input_hashes": {key: sorted(value) for key, value in input_hashes.items()},
        "receipts": receipts,
        "analysis": analysis,
        "script_sha256": common.sha256_file(SCRIPT_PATH),
    }
    digest = common.canonical_json_sha256(core)[:16]
    json_path = common.OUTPUT_ROOT / "analysis" / f"p0b_{digest}.json"
    report_path = common.OUTPUT_ROOT / f"GLC_P0B_DIVERGENCE_REPORT_{digest}.md"
    if not json_path.exists():
        common.write_new_json(json_path, {**core, "created_at": common.now_iso()})
    common.write_once_text(report_path, _analysis_markdown(core))
    print(
        json.dumps(
            {
                "status": "complete",
                "hg2_status": analysis["hg2_status"],
                "localization": analysis["localization"],
                "analysis": str(json_path),
                "report": str(report_path),
            }
        )
    )
    return 0


def self_test(args: argparse.Namespace) -> int:
    common.ensure_output_root(args.output_dir)
    _manifest, _arrays = common.validate_sealed_controls()
    short_shape = (N_LAYERS, len(OP_NAMES), len(common.SHORT_PROBE_POSITIONS), HIDDEN_DIM)
    long_shape = (N_LAYERS, len(OP_NAMES), len(common.LONG_PROBE_POSITIONS), HIDDEN_DIM)
    # Use four columns for the arithmetic test, then pad by broadcasting so the
    # production shape contracts are exercised without model weights.
    hf_short_small = np.ones(short_shape[:-1] + (4,), dtype=np.float32)
    hf_long_small = np.ones(long_shape[:-1] + (4,), dtype=np.float32)
    port_short_small = hf_short_small + np.float32(0.01)
    port_long_small = hf_long_small + np.float32(0.01)
    # First actual operation crossing is layer 3 attention_output; all later
    # boundaries in that layer retain it, so layer 3 output is the first layer gate.
    tail = slice(2, None)
    port_long_small[3, 2:, tail] = np.float32(1.20)

    # The analyzer's dimensional contract is production-sized.  Exercise the
    # metric and localization logic with a local equivalent using tiled columns.
    factor = HIDDEN_DIM // 4
    result = _analyze_arrays(
        port_short=np.tile(port_short_small, (1, 1, 1, factor)),
        hf_short=np.tile(hf_short_small, (1, 1, 1, factor)),
        port_long=np.tile(port_long_small, (1, 1, 1, factor)),
        hf_long=np.tile(hf_long_small, (1, 1, 1, factor)),
    )
    localization = result["localization"]
    checks = [
        {
            "name": "first_layer",
            "expected": 3,
            "observed": None if localization is None else localization["first_layer"],
        },
        {
            "name": "first_operation",
            "expected": "attention_output",
            "observed": None
            if localization is None
            else localization["first_operation_boundary"],
        },
        {
            "name": "first_operation_layer",
            "expected": 3,
            "observed": None
            if localization is None
            else localization["first_operation_layer"],
        },
        {
            "name": "status",
            "expected": "LOCALIZED",
            "observed": result["hg2_status"],
        },
    ]
    for check in checks:
        check["passed"] = check["expected"] == check["observed"]
    core = {
        "schema": "glc_p0b_self_test_v1",
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "checks": checks,
        "script_sha256": common.sha256_file(SCRIPT_PATH),
        "control_arrays_sha256": common.sha256_file(common.CONTROL_ARRAYS),
    }
    digest = common.canonical_json_sha256(core)[:16]
    path = common.OUTPUT_ROOT / "self_test" / f"p0b_{digest}.json"
    if not path.exists():
        common.write_new_json(path, {**core, "created_at": common.now_iso()})
    print(json.dumps({"status": core["status"], "receipt": str(path)}))
    return 0 if core["status"] == "pass" else 1


def main() -> int:
    args = parse_args()
    if args.mode == "preflight":
        return preflight(args)
    if args.mode == "self-test":
        return self_test(args)
    if args.mode == "capture-port":
        return capture_port(args)
    if args.mode == "capture-hf":
        return capture_hf(args)
    if args.mode == "analyze":
        return analyze(args)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())

