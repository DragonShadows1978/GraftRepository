"""Sharded safetensors loader with bf16->fp16 conversion for Mistral 7B.

Loads weights from multiple shard files, converting bf16 to fp16 on the fly.
Supports both single-file and sharded safetensors formats.
"""

import json
import os
import struct
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable

import numpy as np

_DEFAULT_MODEL_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1",
    "snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da",
)

_DTYPE_MAP = {
    "BF16": ("uint16", 2),
    "F16": ("float16", 2),
    "F32": ("float32", 4),
    "I32": ("int32", 4),
    "I64": ("int64", 8),
}


def _validate_safe_path(path: str, allowed_dir: str) -> Path:
    if ".." in Path(path).parts:
        raise ValueError(f"Path contains '..': {path!r}")
    resolved = Path(path).resolve()
    allowed_resolved = Path(allowed_dir).resolve()
    if not resolved.is_relative_to(allowed_resolved):
        raise ValueError(
            f"Path escapes allowed directory: {path!r} resolves to {resolved}, "
            f"which is outside {allowed_resolved}"
        )
    return resolved


def _bf16_to_fp16(raw: np.ndarray) -> np.ndarray:
    u16 = raw.view(np.uint16)
    u32 = u16.astype(np.uint32) << 16
    f32 = u32.view(np.float32)
    return f32.astype(np.float16)


def parse_header(path: str, allowed_dir: str) -> Tuple[Dict, int]:
    safe_path = _validate_safe_path(path, allowed_dir)
    with open(safe_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_size)
    header = json.loads(header_bytes)
    header.pop("__metadata__", None)
    data_offset = 8 + header_size
    return header, data_offset


def load_safetensors_streaming(
    path: str,
    target_dtype: str = "float16",
    allowed_dir: str = "",
    on_tensor: Optional[Callable] = None,
) -> Dict[str, np.ndarray]:
    """Load tensors from a safetensors file one at a time to limit peak RAM.

    If on_tensor is provided, calls on_tensor(name, array) for each tensor
    and does NOT accumulate into the returned dict (returns empty dict).
    This allows the caller to quantize and move to GPU immediately per-tensor.
    """
    if not allowed_dir:
        allowed_dir = str(Path(path).resolve().parent)
    safe_path = _validate_safe_path(path, allowed_dir)
    header, data_offset = parse_header(str(safe_path), allowed_dir=allowed_dir)

    tensors = {}
    with open(safe_path, "rb") as f:
        for name, meta in header.items():
            dtype_str = meta["dtype"]
            shape = tuple(meta["shape"])
            start, end = meta["data_offsets"]

            if dtype_str not in _DTYPE_MAP:
                raise ValueError(f"Unsupported dtype {dtype_str} for tensor {name}")
            np_dtype, _ = _DTYPE_MAP[dtype_str]

            f.seek(data_offset + start)
            raw = np.frombuffer(f.read(end - start), dtype=np_dtype)

            if dtype_str == "BF16":
                arr = _bf16_to_fp16(raw).reshape(shape)
            elif dtype_str == "F32" and target_dtype == "float16":
                arr = raw.reshape(shape).astype(np.float16)
            else:
                arr = raw.reshape(shape)

            if target_dtype == "float16" and arr.dtype != np.float16:
                arr = arr.astype(np.float16)

            if on_tensor is not None:
                on_tensor(name, arr)
            else:
                tensors[name] = arr

    return tensors


def load_mistral_weights(
    model_dir: str = None,
    on_tensor: Optional[Callable] = None,
) -> Dict[str, np.ndarray]:
    """Load Mistral 7B weights from sharded safetensors.

    If on_tensor is provided, streams each tensor through the callback
    instead of accumulating in memory (critical for 14.5GB model on 64GB RAM).
    """
    if model_dir is None:
        model_dir = _DEFAULT_MODEL_DIR

    resolved_dir = str(Path(model_dir).resolve())
    hub_root = resolved_dir
    if "/snapshots/" in resolved_dir:
        hub_root = resolved_dir.split("/snapshots/")[0]

    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        return load_safetensors_streaming(
            single, target_dtype="float16", allowed_dir=hub_root, on_tensor=on_tensor,
        )

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        all_tensors = {}
        for shard in shard_files:
            shard_path = os.path.join(model_dir, shard)
            result = load_safetensors_streaming(
                shard_path, target_dtype="float16", allowed_dir=hub_root, on_tensor=on_tensor,
            )
            all_tensors.update(result)
        return all_tensors

    raise FileNotFoundError(f"No safetensors files found in {model_dir}")
