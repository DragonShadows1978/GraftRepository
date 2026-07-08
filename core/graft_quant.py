"""Packed on-disk quantization core for GRM graft K/V storage (P3).

This is the shared, reusable implementation of the storage-quantization
transform proven out in P0-P2 of `docs/GRM_GRAFT_QUANT_PLAN.md` (see
`docs/GRM_GRAFT_QUANT_LEDGER.md` and
`artifacts/grm_graft_quant/SWEEP_RESULTS.md` for the recall-gate curve this
math was validated against). `scripts/grm_graft_quant_transform.py` is the
REFERENCE implementation this module refactors: the quantize/dequantize math
here is bit-for-bit the same uniform-symmetric-per-group transform (group-32
along the last axis, `head_dim`), not a reimplementation.

Scope law (unchanged from P0/P1/P2): this is a storage transform of
witnessed K/V banks — never synthesis, never APA. Two independent things
live in this module:

1. The MATH — `quantize_dequantize_symmetric_group32` (round-trip through a
   fp16 numpy array, used by the existing P1/P2 harness) plus the split
   quantize/dequantize halves (`quantize_symmetric_group32` /
   `dequantize_symmetric_group32`) that actually emit/consume packed uint8
   codes + per-group scales instead of only round-tripping to fp16.
2. The PACKED FORMAT — `pack_kv_arrays` / `unpack_kv_arrays` and
   `save_packed_npz` / `load_packed_npz`: an explicit on-disk schema with a
   `format_version` and `storage_bits` field, fail-closed on anything it
   does not recognize (never a silent misread of a foreign/newer format).

SCALE DTYPE NOTE (measured, not assumed): the plan text says "per-group fp16
scales" as a size estimate. Measured against the P1 reference transform,
storing the scale AS fp16 changes the dequantized bytes at 8/6/4/3 bits
(confirmed empirically: ~10-22% of elements differ by one fp16 ULP,
0.0078125 abs at the tested magnitude) — because `scale = max|group| / qmax`
is a float32-precision value not generally exactly representable in fp16,
and the P1 reference's `codes = round(grouped / scale)` step is itself
computed against that unrounded scale. Since the registered P3 gate is
BIT-IDENTICAL pack-then-unpack vs the P1 reference's dequantized output
(hard equality, not "close"), and the P1 reference is frozen (its output is
what P0-P2's already-published recall-gate curve was measured against),
this module stores the per-group scale as **float32**, not fp16. This is a
deliberate deviation from the plan's casual size estimate, made to satisfy
the stricter, later-registered, and more specific gate (spec-is-law: the
explicit round-trip equality gate overrides the plan's aside about scale
dtype). Disk-multiplier receipts in P3 report the true float32-scale
overhead (0.125 B/32 vals = ~0.0625 B/val more than the plan's fp16-scale
estimate assumed), not the plan's original headline number.

Both graft storage formats in this repo (format 2 = GPT-OSS capture-harness
`layer_NNN.npz` shards; format 1 = production `nodes/NNNN.npz` GQA payloads)
share the exact K/V shape convention this module targets: `(..., S, D)`
arrays with `D == head_dim` as the last axis, D divisible by `group_size`
(house convention: group-32, matching `tensor_cuda.quantization
.quantize_affine_per_group`'s reduction axis and `core.mistral7b_tc
._kv_quant`'s per-vector-over-D reduction). Format 2 shards are rank-4
`(1, H, S, D)`; format 1 GQA node payloads (`pack_node` in `graft_arena.py`)
are rank-4 `(L, H, S, D)` (layers stacked on axis 0 instead of batch=1) —
this module is shape-agnostic on all but the last axis, so both call the
same functions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

GROUP_SIZE = 32  # plan: "group-32, the house packing convention"
FORMAT_VERSION = 1  # this module's packed-npz schema version
SUPPORTED_BITS = (16, 8, 6, 4, 3, 2)
# int8 is the on-disk code container for every sub-8-bit depth too (packing
# multiple sub-8-bit codes into fewer bytes is future work — P3 scope is the
# format + hooks, not bit-packing below one byte/code; codes for bits<8 are
# still one byte each, just using a narrower signed range within that byte).
CODE_DTYPE = np.int8


class GraftQuantFormatError(ValueError):
    """Raised on any unknown/unsupported packed-format version or bit depth.
    Fail-closed: callers must never guess at a silent default."""


def qmax_signed(bits: int) -> int:
    """Symmetric signed grid max code magnitude: 2^(bits-1) - 1 (e.g.
    bits=8 -> 127, matching `_kv_quant`'s scale=max|x|/127). bits=16 has no
    integer grid (fp16 passthrough, see `quantize_dequantize_symmetric_group32`)."""
    return (1 << (bits - 1)) - 1


def _validate_bits(bits: int) -> None:
    if bits not in SUPPORTED_BITS:
        raise GraftQuantFormatError(
            f"unsupported storage_bits {bits!r}; supported: {SUPPORTED_BITS}"
        )


def quantize_symmetric_group32(
    x: np.ndarray, bits: int, group_size: int = GROUP_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform symmetric per-group quantize. x: (..., D) float16/float32,
    D % group_size == 0. Groups D into (D // group_size) groups along the
    LAST axis, one scale per (..., group). Returns (codes, scales), shapes:
      codes: (..., groups, group_size) — same total element count as x,
             reshaped; dtype int8 (signed grid fits bits<=8 always).
      scales: (..., groups, 1) float32 — NOT fp16 (see module docstring's
             "SCALE DTYPE NOTE": storing scale as fp16 changes the
             dequantized bytes vs the P1 reference at 8/6/4/3 bits, which
             would fail the registered bit-identical round-trip gate; scale
             is kept at the same float32 precision the P1 reference computes
             and applies it at).

    bits must not be 16 here (16 has no quantization grid — callers should
    special-case identity themselves, exactly as
    `quantize_dequantize_symmetric_group32` does for the round-trip path).
    """
    _validate_bits(bits)
    if bits == 16:
        raise GraftQuantFormatError(
            "quantize_symmetric_group32: bits=16 has no packed grid; "
            "16 is the identity/passthrough path, not a packable depth"
        )
    if x.shape[-1] % group_size != 0:
        raise ValueError(
            f"last axis {x.shape[-1]} not divisible by group_size {group_size}"
        )
    orig_shape = x.shape
    groups = orig_shape[-1] // group_size
    xf = x.astype(np.float32)
    grouped = xf.reshape(*orig_shape[:-1], groups, group_size)

    qmax = qmax_signed(bits)
    scale = np.abs(grouped).max(axis=-1, keepdims=True) / float(qmax)
    scale = np.where(scale == 0, 1.0, scale)  # avoid div-by-zero on all-zero groups

    codes = np.clip(np.round(grouped / scale), -qmax - 1, qmax).astype(CODE_DTYPE)
    scale = scale.astype(np.float32)
    return codes, scale


def dequantize_symmetric_group32(
    codes: np.ndarray, scale: np.ndarray, orig_shape: tuple[int, ...]
) -> np.ndarray:
    """Inverse of `quantize_symmetric_group32`. codes: (..., groups,
    group_size) int8; scale: (..., groups, 1) float32; orig_shape:
    the pre-grouping shape to reshape back to (e.g. (1, H, S, D)). Returns
    float16, matching the at-rest dtype of both graft formats."""
    dequant = (codes.astype(np.float32) * scale.astype(np.float32)).reshape(orig_shape)
    return dequant.astype(np.float16)


def quantize_dequantize_symmetric_group32(
    x: np.ndarray, bits: int, group_size: int = GROUP_SIZE
) -> np.ndarray:
    """Quantize-then-immediately-dequantize round trip (the P1/P2 harness's
    original transform: `scripts/grm_graft_quant_transform.py
    .quantize_dequantize_symmetric_group32`, reproduced here bit-for-bit so
    that module can delegate to this one instead of duplicating the math).

    bits=16 is an exact identity path: fp16 has no lossy quantization step to
    apply at 16 bits under this scheme, so we return x unchanged bit-for-bit
    — this IS the harness's own parity gate (bits=16 == identity, verified by
    callers via bit-compare)."""
    _validate_bits(bits)
    if bits == 16:
        return x.copy()
    codes, scale = quantize_symmetric_group32(x, bits, group_size)
    return dequantize_symmetric_group32(codes, scale, x.shape)


def rmse_maxabs(orig: np.ndarray, recon: np.ndarray) -> tuple[float, float]:
    diff = orig.astype(np.float32) - recon.astype(np.float32)
    rmse = float(np.sqrt(np.mean(diff * diff)))
    maxabs = float(np.max(np.abs(diff))) if diff.size else 0.0
    return rmse, maxabs


# --------------------------------------------------------------------------
# Packed payload: {name}_codes / {name}_scales / {name}_shape per tensor.
# --------------------------------------------------------------------------

def pack_kv_arrays(
    arrays: dict[str, np.ndarray], bits: int, group_size: int = GROUP_SIZE
) -> dict[str, Any]:
    """Pack a dict of named fp16 arrays (e.g. {"k": ..., "v": ...} for format
    2, or the same for a format-1 node payload) into a flat dict of
    numpy-savez-able fields: per array `name` ->
      f"{name}_codes"  int8  (..., groups, group_size)
      f"{name}_scales" float32 (..., groups, 1) — see module docstring's
                       "SCALE DTYPE NOTE" for why this is float32, not the
                       plan's fp16 estimate.
      f"{name}_shape"  int64 1-D — the ORIGINAL (pre-group) shape, needed to
                       reshape on unpack (np.savez cannot store a bare shape
                       tuple as metadata without a carrier array).
    Plus top-level `format_version` and `storage_bits` (scalar int arrays —
    np.savez only stores arrays, so these are 0-d int64 arrays).

    bits=16 is REJECTED here (raise GraftQuantFormatError): packing at 16
    bits would double-store scale overhead for zero compression benefit, and
    the identity path has no meaningful "codes" — callers wanting a 16-bit
    at-rest artifact should simply save the fp16 arrays directly (the
    existing/default path), not go through the packed format.
    """
    _validate_bits(bits)
    if bits == 16:
        raise GraftQuantFormatError(
            "pack_kv_arrays: storage_bits=16 is not a packable depth "
            "(use the plain fp16 save path instead)"
        )
    out: dict[str, Any] = {
        "format_version": np.asarray(FORMAT_VERSION, dtype=np.int64),
        "storage_bits": np.asarray(bits, dtype=np.int64),
        "group_size": np.asarray(group_size, dtype=np.int64),
    }
    for name, arr in arrays.items():
        if arr.dtype != np.float16:
            raise ValueError(f"pack_kv_arrays: {name!r} expected float16, got {arr.dtype}")
        codes, scale = quantize_symmetric_group32(arr, bits, group_size)
        out[f"{name}_codes"] = codes
        out[f"{name}_scales"] = scale
        out[f"{name}_shape"] = np.asarray(arr.shape, dtype=np.int64)
    return out


def _scalar_int_field(payload: dict[str, Any], name: str) -> int:
    arr = np.asarray(payload[name])
    if arr.shape == ():
        return int(arr)
    if arr.size == 1:
        return int(arr.reshape(-1)[0])
    raise GraftQuantFormatError(
        f"unpack_kv_arrays: field {name!r} must be scalar-like, got shape "
        f"{arr.shape}"
    )


def unpack_kv_arrays(payload: dict[str, Any], names: list[str]) -> dict[str, np.ndarray]:
    """Inverse of `pack_kv_arrays`. `payload` is whatever a packed npz load
    yields (dict-like: NpzFile or a plain dict of arrays). Fail-closed: an
    unrecognized `format_version` or `storage_bits` raises
    GraftQuantFormatError rather than silently misreading bytes as something
    they are not.
    """
    if "format_version" not in payload:
        raise GraftQuantFormatError(
            "unpack_kv_arrays: payload has no format_version field "
            "(not a packed graft payload, or a corrupt/foreign file)"
        )
    version = _scalar_int_field(payload, "format_version")
    if version != FORMAT_VERSION:
        raise GraftQuantFormatError(
            f"unpack_kv_arrays: unknown format_version {version} "
            f"(this build only understands {FORMAT_VERSION})"
        )
    bits = _scalar_int_field(payload, "storage_bits")
    _validate_bits(bits)
    if bits == 16:
        raise GraftQuantFormatError(
            "unpack_kv_arrays: storage_bits=16 encoded in a packed payload "
            "is invalid (16 is never packed, see pack_kv_arrays)"
        )
    out: dict[str, np.ndarray] = {}
    for name in names:
        codes_key, scales_key, shape_key = f"{name}_codes", f"{name}_scales", f"{name}_shape"
        if codes_key not in payload or scales_key not in payload or shape_key not in payload:
            raise GraftQuantFormatError(
                f"unpack_kv_arrays: payload missing {codes_key}/{scales_key}/{shape_key}"
            )
        codes = np.ascontiguousarray(payload[codes_key])
        scales = np.ascontiguousarray(payload[scales_key])
        orig_shape = tuple(int(v) for v in np.asarray(payload[shape_key]))
        out[name] = dequantize_symmetric_group32(codes, scales, orig_shape)
    return out


def is_packed_payload(payload: Any) -> bool:
    """True if `payload` (an NpzFile, dict, or anything with .keys()/'in')
    looks like a packed graft payload (has a format_version field) rather
    than a plain fp16 {"k": ..., "v": ...} / {"c": ..., "kpe": ...} payload."""
    try:
        keys = payload.files if hasattr(payload, "files") else payload.keys()
    except Exception:
        return False
    return "format_version" in keys


def save_packed_npz(path, arrays: dict[str, np.ndarray], bits: int,
                     group_size: int = GROUP_SIZE, compressed: bool = False) -> None:
    """Quantize `arrays` (plain fp16 dict) and write a packed npz to `path`.
    `compressed=True` uses np.savez_compressed (format-1 convention, zlib);
    `compressed=False` uses np.savez (format-2 convention, uncompressed)."""
    payload = pack_kv_arrays(arrays, bits, group_size)
    saver = np.savez_compressed if compressed else np.savez
    saver(path, **payload)


def load_packed_npz(path, names: list[str]) -> dict[str, np.ndarray]:
    """Load a packed npz from `path` and dequantize to a plain fp16 dict
    keyed by `names`. Fail-closed via `unpack_kv_arrays`."""
    with np.load(path) as z:
        return unpack_kv_arrays(z, names)
