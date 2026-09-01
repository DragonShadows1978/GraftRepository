#!/usr/bin/env python3
"""Fail-closed dual snapshots for ORDER GRM-DET1.3.

The capture boundary is immediately before the probe prefill.  Tensor values
are copied to canonical C-contiguous host value bytes and content-addressed;
TensorCUDA currently exposes no raw device-storage exporter, so the manifest
records that limitation instead of calling the host representation literal
BF16 device bytes.  Equal BF16 values still compare exactly after TensorCUDA's
deterministic BF16-to-FP32 export.

The comparator refuses to compare dynamic bytes across different model frames.
In particular, a MiniCPM3 MLA producer snapshot can never be a byte oracle for
a GPT-OSS GQA replay snapshot.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np


SNAPSHOT_SCHEMA = "grm.det1_3.model_visible_snapshot.v2"
COMPARISON_SCHEMA = "grm.det1_3.dual_snapshot_comparison.v2"
CAPTURE_PHASE = "before_probe_prefill"
ARM_PROTOCOLS = {
    "lived": "same_model_chronological_harmony_feed_v1",
    "replay": "det1_independent_deposit_counterfactual_replay_v1",
    "fork": "det1_lived_snapshot_fork_v1",
}
BYTE_SEMANTICS = {
    "comparison_bytes": "C_contiguous_host_value_bytes",
    "literal_device_storage_bytes": False,
    "tensor_cuda_bfloat16_export": "host_float32_value_preserving",
    "qualification": (
        "byte equality means canonical exported value-byte equality until "
        "TensorCUDA provides a raw device-storage byte exporter"
    ),
}
REQUIRED_GROUPS = (
    "provenance",
    "identity",
    "arena_kv",
    "positions_rope",
    "sink_rows",
    "masks",
    "text_tokens",
    "live_tokens",
    "admission_plan",
)
REQUIRED_IDENTITY_FIELDS = (
    "model_revision",
    "model_weights_inventory_sha256",
    "tokenizer_sha256",
    "engine_build_sha256",
    "prompt_template_sha256",
    "dialect",
)
REQUIRED_ADMISSION_FIELDS = (
    "schema",
    "probe_key_sha256",
    "ranking",
    "identifier_tokens",
    "identified_candidates",
    "policy_branch",
    "rank_plan",
    "ladder_attempts",
    "current_attempt_ordinal",
    "current_planned",
    "current_clean_room",
    "current_fitted",
    "current_dropped",
    "selection_state",
    "final_mounts",
)
REQUIRED_PROVENANCE_FIELDS = (
    "schema",
    "arm",
    "protocol",
    "protocol_source_sha256",
    "run_id",
    "order_sha256",
    "source_amendment_sha256",
    "registration_sha256",
    "runtime_frame_sha256",
    "fixture_sha256",
    "fixture_id",
    "probe_id",
    "probe_question_sha256",
    "capture_boundary",
    "probe_driver",
    "topk",
    "max_trips",
    "probe_ladder",
    "defer_memory",
    "attempt_ordinal",
    "selection_boundary",
    "process_instance_sha256",
    "process_pid",
    "process_start_ticks",
    "boot_id",
)
COMMON_PROVENANCE_FIELDS = tuple(
    field for field in REQUIRED_PROVENANCE_FIELDS
    if field not in {
        "arm", "protocol", "protocol_source_sha256",
        "process_instance_sha256", "process_pid", "process_start_ticks",
        "attempt_ordinal",
    }
)


class SnapshotError(RuntimeError):
    """A snapshot is incomplete, corrupt, or incomparable."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def process_instance_sha256(
    process_pid: int,
    process_start_ticks: int,
    boot_id: str,
) -> str:
    return _sha256_bytes(_canonical_json({
        "boot_id": str(boot_id),
        "process_pid": int(process_pid),
        "process_start_ticks": int(process_start_ticks),
    }))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive_or_verify(path: Path, value: Mapping[str, Any]) -> Path:
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise SnapshotError(f"immutable comparison collision: {path}")
        return path
    with path.open("xb") as handle:
        handle.write(payload)
    return path


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _public_config(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return _jsonable(value)
    items = {}
    for key, item in vars(value).items():
        if key.startswith("_") or callable(item):
            continue
        converted = _jsonable(item)
        if not isinstance(converted, str) or not converted.startswith("<"):
            items[str(key)] = converted
    return items


def infer_frame_identity(
    arena: Any,
    explicit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect stable model/dialect/runtime identity, then apply explicit pins."""
    model = arena.m
    layers = list(getattr(model, "layers", ()) or ())
    layer_types = []
    attention = {}
    for layer in layers:
        att = getattr(layer, "self_attn", None)
        layer_types.append(str(getattr(att, "layer_type", "unknown")))
    if layers:
        att = getattr(layers[0], "self_attn", None)
        for key in (
            "num_heads", "num_kv_heads", "head_dim", "num_heads_per_kv",
            "attention_mode", "sliding_window", "scaling", "bulk_bits",
            "attn_block", "refine_percentile",
        ):
            if hasattr(att, key):
                attention[key] = _jsonable(getattr(att, key))
    payload = [
        {"name": str(name), "sequence_dim": int(dim)}
        for name, dim in getattr(arena, "PAYLOAD", ())
    ]
    identity = {
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "arena_class": f"{type(arena).__module__}.{type(arena).__name__}",
        "dialect_payload": payload,
        "layer_count": len(layers),
        "layer_types": layer_types,
        "model_config": _public_config(getattr(model, "config", None)),
        "attention_geometry": attention,
        "compute_dtype": str(getattr(arena, "dt", "unknown")),
        "storage_bits": _jsonable(getattr(arena, "storage_bits", None)),
        "route_layer": _jsonable(getattr(arena, "route_layer", None)),
        "route_backend": _jsonable(getattr(arena, "route_backend", None)),
    }
    for key, value in _jsonable(dict(explicit or {})).items():
        if key in identity and identity[key] != value:
            raise SnapshotError(
                f"explicit frame pin conflicts with inferred identity {key}: "
                f"inferred={identity[key]!r} explicit={value!r}")
        identity[key] = value
    identity["frame_sha256"] = _sha256_bytes(_canonical_json(identity))
    return identity


def _as_host_array(value: Any) -> tuple[np.ndarray, str]:
    source_dtype = str(getattr(value, "dtype", type(value).__name__))
    if isinstance(value, np.ndarray):
        array = value
    elif hasattr(value, "numpy"):
        array = value.numpy()
    else:
        array = np.asarray(value)
    array = np.asarray(array)
    if array.dtype.hasobject:
        raise SnapshotError("object arrays have no stable byte representation")
    return np.ascontiguousarray(array), source_dtype


class SnapshotBuilder:
    def __init__(
        self,
        root: Path,
        *,
        label: str,
        phase: str,
        provenance: Mapping[str, Any],
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.blob_root = self.root / "blobs"
        self.blob_root.mkdir()
        self.label = str(label)
        self.phase = str(phase)
        self.provenance = _jsonable(dict(provenance))
        self.arrays: dict[str, dict[str, Any]] = {}
        self.state: dict[str, Any] = {}
        self.coverage = {group: 0 for group in REQUIRED_GROUPS}
        self.unavailable: list[dict[str, Any]] = []
        self.required_checks: list[dict[str, Any]] = []

    def scalar(self, field: str, value: Any, *, group: str) -> None:
        self.state[str(field)] = _jsonable(value)
        self.coverage[group] = self.coverage.get(group, 0) + 1

    def unavailable_field(
        self, field: str, reason: str, *, group: str, required: bool = True,
    ) -> None:
        self.unavailable.append({
            "field": str(field),
            "group": str(group),
            "required": bool(required),
            "reason": str(reason),
        })

    def array(self, field: str, value: Any, *, group: str) -> None:
        array, source_dtype = _as_host_array(value)
        payload = array.tobytes(order="C")
        digest = _sha256_bytes(payload)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(field)).strip("_")
        path = self.blob_root / f"{len(self.arrays):04d}_{safe}_{digest[:16]}.bin"
        path.write_bytes(payload)
        self.arrays[str(field)] = {
            "group": str(group),
            "source_dtype": source_dtype,
            "host_dtype": array.dtype.str,
            "shape": [int(value) for value in array.shape],
            "order": "C",
            "bytes": len(payload),
            "sha256": digest,
            "blob": str(path.relative_to(self.root)),
            "source_representation": (
                "host_numpy" if isinstance(value, np.ndarray) else "exported_tensor_value"
            ),
            "literal_device_storage_bytes": False,
        }
        self.coverage[group] = self.coverage.get(group, 0) + 1

    def require(self, field: str, *, group: str, reason: str) -> None:
        present = str(field) in self.state or str(field) in self.arrays
        self.required_checks.append({
            "field": str(field), "group": str(group), "present": present,
        })
        if not present:
            self.unavailable_field(field, reason, group=group)

    def finish(self, identity: Mapping[str, Any]) -> Path:
        self.coverage["provenance"] = len(self.provenance)
        self.coverage["identity"] = len(identity)
        missing_groups = [
            group for group in REQUIRED_GROUPS
            if self.coverage.get(group, 0) == 0
        ]
        required_unavailable = [
            value for value in self.unavailable if value["required"]
        ]
        manifest = {
            "schema": SNAPSHOT_SCHEMA,
            "label": self.label,
            "phase": self.phase,
            "provenance": self.provenance,
            "byte_semantics": BYTE_SEMANTICS,
            "identity": _jsonable(identity),
            "state": self.state,
            "arrays": self.arrays,
            "availability": {
                "required_groups": list(REQUIRED_GROUPS),
                "coverage_counts": self.coverage,
                "missing_groups": missing_groups,
                "unavailable_fields": self.unavailable,
                "required_unavailable_fields": required_unavailable,
                "required_field_checks": self.required_checks,
            },
            "complete": not missing_groups and not required_unavailable,
        }
        manifest["manifest_payload_sha256"] = _sha256_bytes(
            _canonical_json(manifest))
        path = self.root / "manifest.json"
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


def _payload_items(arena: Any, payload: Any) -> list[tuple[str, Any]]:
    specs = list(getattr(arena, "PAYLOAD", ()) or ())
    if isinstance(payload, Mapping):
        return [(str(name), payload[name]) for name, _dim in specs if name in payload]
    if isinstance(payload, (list, tuple)):
        names = [str(name) for name, _dim in specs]
        return [
            (names[index] if index < len(names) else f"component_{index}", value)
            for index, value in enumerate(payload)
        ]
    return [("value", payload)]


def _capture_payload_layers(
    builder: SnapshotBuilder,
    arena: Any,
    prefix: str,
    layers: Sequence[Any],
    *,
    group: str,
) -> None:
    for layer_index, payload in enumerate(layers):
        for name, value in _payload_items(arena, payload):
            builder.array(
                f"{prefix}.layer_{layer_index:02d}.{name}", value, group=group)


def _cache_sequence_length(arena: Any, cache: Any) -> int:
    specs = list(getattr(arena, "PAYLOAD", ()) or ())
    if not specs:
        return 0
    first_name, first_dim = specs[0]
    if isinstance(cache, Mapping):
        tensor = cache[first_name]
    else:
        tensor = cache[0]
    return int(tensor.shape[int(first_dim)])


def _injection_sequence_length(arena: Any, injection: Any) -> int:
    if injection is None:
        return 0
    specs = list(getattr(arena, "PAYLOAD", ()) or ())
    if not specs:
        return 0
    first_name, first_dim = specs[0]
    if isinstance(injection, Mapping):
        tensor = injection[first_name]
    else:
        tensor = injection[0]
    return int(tensor.shape[int(first_dim)])


def _causal_allowed_mask(length: int, total: int, window: int | None) -> np.ndarray:
    q_abs = np.arange(total - length, total, dtype=np.int64)[:, None]
    k_abs = np.arange(total, dtype=np.int64)[None, :]
    allowed = k_abs <= q_abs
    if window is not None:
        allowed &= k_abs > (q_abs - int(window))
    return np.ascontiguousarray(allowed.astype(np.uint8).reshape(1, 1, length, total))


def capture_arena_snapshot(
    arena: Any,
    output_dir: Path,
    *,
    label: str,
    provenance: Mapping[str, Any],
    phase: str = CAPTURE_PHASE,
    question: str,
    prompt_ids: Sequence[int],
    admission_plan: Mapping[str, Any] | None,
    live_token_ids: Sequence[int] | None,
    sink_text: str | None = None,
    sink_token_ids: Sequence[int] | None = None,
    explicit_identity: Mapping[str, Any] | None = None,
) -> Path:
    """Capture every dynamic field visible to the next arena model call."""
    builder = SnapshotBuilder(
        Path(output_dir), label=label, phase=phase, provenance=provenance)
    if str(phase) != CAPTURE_PHASE:
        builder.unavailable_field(
            "metadata.phase",
            f"snapshot phase {phase!r} is not the registered prefill boundary",
            group="positions_rope",
        )
    for field in REQUIRED_PROVENANCE_FIELDS:
        value = provenance.get(field)
        if value is None or value == "":
            builder.unavailable_field(
                f"provenance.{field}",
                "required lived/replay capture provenance was not supplied",
                group="provenance",
            )
    arm = str(provenance.get("arm", ""))
    if str(label) != arm:
        builder.unavailable_field(
            "provenance.label_arm_consistency",
            f"snapshot label {label!r} does not equal provenance arm {arm!r}",
            group="provenance",
        )
    expected_protocol = ARM_PROTOCOLS.get(arm)
    if expected_protocol is None or provenance.get("protocol") != expected_protocol:
        builder.unavailable_field(
            "provenance.registered_arm_protocol",
            f"arm {arm!r} does not carry its registered protocol",
            group="provenance",
        )
    if provenance.get("capture_boundary") != CAPTURE_PHASE:
        builder.unavailable_field(
            "provenance.capture_boundary",
            "provenance is not bound to the registered prefill boundary",
            group="provenance",
        )
    try:
        expected_process = process_instance_sha256(
            int(provenance["process_pid"]),
            int(provenance["process_start_ticks"]),
            str(provenance["boot_id"]),
        )
    except (KeyError, TypeError, ValueError):
        expected_process = None
    if expected_process != provenance.get("process_instance_sha256"):
        builder.unavailable_field(
            "provenance.process_instance_sha256",
            "process instance digest does not bind pid/start ticks/boot id",
            group="provenance",
        )
    if provenance.get("selection_boundary") not in (
        None, "before_generation_and_grounding_selection",
    ):
        builder.unavailable_field(
            "provenance.selection_boundary",
            "capture claims a post-generation selection at a prefill boundary",
            group="provenance",
        )
    identity = infer_frame_identity(arena, explicit_identity)
    for field in REQUIRED_IDENTITY_FIELDS:
        if identity.get(field) in (None, "", "unknown"):
            builder.unavailable_field(
                f"identity.{field}",
                "required explicit frame pin was not supplied",
                group="identity",
            )
    model = arena.m
    layers = list(getattr(model, "layers", ()) or ())
    prompt = np.asarray([int(value) for value in prompt_ids], dtype=np.int64)

    builder.array(
        "text.question_utf8", np.frombuffer(question.encode("utf-8"), dtype=np.uint8),
        group="text_tokens",
    )
    builder.array("tokens.prompt_ids", prompt, group="text_tokens")
    builder.scalar("tokens.prompt_count", int(prompt.size), group="text_tokens")
    if not int(prompt.size):
        builder.unavailable_field(
            "tokens.prompt_ids_nonempty",
            "probe prefill has no prompt token IDs",
            group="text_tokens",
        )
    if sink_text is None:
        builder.unavailable_field(
            "text.sink_utf8", "arena does not retain constructor sink_text",
            group="text_tokens",
        )
    else:
        builder.array(
            "text.sink_utf8",
            np.frombuffer(str(sink_text).encode("utf-8"), dtype=np.uint8),
            group="text_tokens",
        )
    if sink_token_ids is None:
        builder.unavailable_field(
            "tokens.sink_ids", "exact sink token IDs were not supplied",
            group="text_tokens",
        )
    else:
        builder.array(
            "tokens.sink_ids", np.asarray(sink_token_ids, dtype=np.int64),
            group="text_tokens",
        )

    live_segs = [
        [None if graft is None else int(graft), int(count)]
        for graft, count in (getattr(arena, "live_segs", ()) or ())
    ]
    live_count = sum(count for _graft, count in live_segs)
    builder.scalar("live.segments", live_segs, group="live_tokens")
    builder.scalar("live.segment_token_count", live_count, group="live_tokens")
    if live_token_ids is None:
        if live_count:
            builder.unavailable_field(
                "tokens.live_ids",
                "live_segs stores lengths, not the exact chronological token ledger",
                group="live_tokens",
            )
        else:
            builder.array(
                "tokens.live_ids", np.asarray([], dtype=np.int64),
                group="live_tokens",
            )
    else:
        live_ids = np.asarray(live_token_ids, dtype=np.int64)
        builder.array("tokens.live_ids", live_ids, group="live_tokens")
        if int(live_ids.size) != live_count:
            builder.unavailable_field(
                "tokens.live_ids_length_check",
                f"ledger has {live_ids.size} IDs but live_segs claims {live_count}",
                group="live_tokens",
            )

    scalar_fields = (
        "pos", "n_sink", "live_shift", "width", "cur_mount_n", "topk",
        "live_turns", "max_live", "cache_deposits", "revision_resolution",
        "decisive_admission", "length_debias", "route_layer", "route_backend",
    )
    for name in scalar_fields:
        builder.scalar(
            f"arena.{name}", getattr(arena, name, None), group="positions_rope")
    builder.scalar(
        "arena.cur_mounts",
        [int(value) for value in (getattr(arena, "cur_mounts", ()) or ())],
        group="positions_rope",
    )
    builder.scalar(
        "positions.query_absolute_rows",
        [
            int(getattr(arena, "live_shift", 0)) + int(getattr(arena, "pos", 0)) + index
            for index in range(int(prompt.size))
        ],
        group="positions_rope",
    )
    builder.scalar(
        "positions.physical_sections",
        {
            "textual_sink": [0, int(getattr(arena, "n_sink", 0))],
            "mounted_payload": [
                int(getattr(arena, "n_sink", 0)),
                int(getattr(arena, "n_sink", 0)) + int(getattr(arena, "cur_mount_n", 0)),
            ],
            "live_position_origin": int(getattr(arena, "live_shift", 0)),
        },
        group="positions_rope",
    )

    layer_shifts = []
    for layer in layers:
        att = getattr(layer, "self_attn", None)
        shift = getattr(att, "live_shift", None)
        if shift is None:
            shift = getattr(att, "graft_seats", 0)
        layer_shifts.append(int(shift or 0))
    required_rope_len = (
        int(getattr(arena, "pos", 0))
        + max(layer_shifts or [int(getattr(arena, "live_shift", 0))])
        + int(prompt.size)
    )
    rope_before = getattr(model, "_rope_len", None)
    if (
        rope_before is not None
        and int(rope_before) < required_rope_len
        and callable(getattr(model, "extend_rope", None))
    ):
        # Mirror the first operation in GPT-OSS's next model call.  Snapshot
        # arms terminate before inference, so this deterministic table growth
        # cannot perturb a measured answer branch.
        model.extend_rope(required_rope_len)
    rope_after = getattr(model, "_rope_len", None)
    if rope_after is not None and int(rope_after) < required_rope_len:
        builder.unavailable_field(
            "model.rope_coverage",
            f"RoPE length {rope_after} does not cover required {required_rope_len}",
            group="positions_rope",
        )
    builder.scalar(
        "model.rope_required_for_next_forward", required_rope_len,
        group="positions_rope",
    )
    builder.scalar(
        "model.rope_len_before_capture_extension", rope_before,
        group="positions_rope",
    )
    rope_cos = getattr(model, "rope_cos", None)
    rope_sin = getattr(model, "rope_sin", None)
    if rope_cos is None or rope_sin is None:
        builder.unavailable_field(
            "model.rope_tables", "model exposes no rope_cos/rope_sin tables",
            group="positions_rope",
        )
    else:
        builder.array("model.rope_cos", rope_cos, group="positions_rope")
        builder.array("model.rope_sin", rope_sin, group="positions_rope")
        builder.scalar(
            "model.rope_len", rope_after,
            group="positions_rope",
        )

    caches = getattr(arena, "caches", None)
    builder.scalar("arena.caches_present", caches is not None, group="arena_kv")
    if caches is not None:
        _capture_payload_layers(builder, arena, "cache", caches, group="arena_kv")
    sink_h = getattr(arena, "sink_h", None)
    if sink_h is None:
        builder.unavailable_field(
            "arena.sink_h", "arena has no harvested textual sink payload",
            group="sink_rows",
        )
    else:
        _capture_payload_layers(builder, arena, "textual_sink", sink_h, group="sink_rows")

    mounts = [int(value) for value in (getattr(arena, "cur_mounts", ()) or ())]
    grafts = list(getattr(arena, "grafts", ()) or ())
    for graft_index in mounts:
        if not (0 <= graft_index < len(grafts)) or grafts[graft_index].get("h") is None:
            builder.unavailable_field(
                f"mounted_graft.{graft_index}.payload",
                "mounted graft payload is absent from the Python arena",
                group="arena_kv",
            )
            continue
        _capture_payload_layers(
            builder, arena, f"mounted_graft.{graft_index}",
            grafts[graft_index]["h"], group="arena_kv",
        )

    cache_lengths = []
    for layer_index, layer in enumerate(layers):
        att = getattr(layer, "self_attn", None)
        injection = getattr(att, "inject_kv", None)
        if injection is not None:
            for name, value in _payload_items(arena, injection):
                # GQA appends a scalar scale as a third tuple item.
                if np.isscalar(value):
                    builder.scalar(
                        f"injection.layer_{layer_index:02d}.{name}", value,
                        group="arena_kv",
                    )
                else:
                    builder.array(
                        f"injection.layer_{layer_index:02d}.{name}", value,
                        group="arena_kv",
                    )
        builder.scalar(
            f"attention.layer_{layer_index:02d}.graft_seats",
            getattr(att, "graft_seats", 0), group="positions_rope",
        )
        builder.scalar(
            f"attention.layer_{layer_index:02d}.live_shift",
            getattr(att, "live_shift", None), group="positions_rope",
        )
        sinks = getattr(att, "sinks", None)
        if sinks is not None:
            builder.array(
                f"learned_sink.layer_{layer_index:02d}", sinks, group="sink_rows")

        cache_len = 0 if caches is None else _cache_sequence_length(arena, caches[layer_index])
        inject_len = _injection_sequence_length(arena, injection) if caches is None else 0
        total = cache_len + inject_len + int(prompt.size)
        window = getattr(att, "sliding_window", None)
        cache_lengths.append(cache_len)
        builder.scalar(
            f"mask.layer_{layer_index:02d}.inputs",
            {
                "query_length": int(prompt.size),
                "cache_length": cache_len,
                "injection_length": inject_len,
                "key_length": total,
                "sliding_window": None if window is None else int(window),
                "layer_type": str(getattr(att, "layer_type", "unknown")),
                "attention_mode": str(getattr(att, "attention_mode", "unknown")),
                "representation": "uint8_canonical_allowed_equivalent",
            },
            group="masks",
        )
        # A hydrated DET1.4 fork carries the lived mask (or its registered
        # seat-deletion projection) as a one-shot override. Recomputing a
        # shorter sliding-window mask here would describe different bytes and
        # make a DET1.6 post-hydration capture incapable of proving the delta.
        installed_mask = getattr(att, "_det1_fork_allowed_mask", None)
        allowed_mask = (
            _causal_allowed_mask(int(prompt.size), total, window)
            if installed_mask is None else _fork_export(installed_mask)
        )
        expected_mask_shape = (1, 1, int(prompt.size), total)
        if tuple(int(value) for value in allowed_mask.shape) != expected_mask_shape:
            raise SnapshotError(
                "installed fork mask geometry differs from the captured arena: "
                f"layer={layer_index} observed={allowed_mask.shape} "
                f"expected={expected_mask_shape}")
        if not np.isin(allowed_mask, (0, 1)).all():
            raise SnapshotError(
                f"installed fork mask is not canonical binary: layer={layer_index}")
        builder.array(
            f"mask.layer_{layer_index:02d}.allowed",
            allowed_mask,
            group="masks",
        )
    builder.scalar("cache.lengths_by_layer", cache_lengths, group="arena_kv")

    if admission_plan is None:
        builder.scalar(
            "admission.authoritative_mounts", mounts, group="admission_plan")
        builder.scalar(
            "admission.last_route_receipt",
            getattr(arena, "last_route_receipt", None), group="admission_plan",
        )
        builder.unavailable_field(
            "admission.full_plan",
            "caller did not supply ranking, branch, fit/drop, and ladder plan",
            group="admission_plan",
        )
    else:
        builder.scalar("admission.full_plan", admission_plan, group="admission_plan")
        for key, value in admission_plan.items():
            builder.scalar(f"admission.{key}", value, group="admission_plan")
        builder.scalar(
            "admission.authoritative_mounts", mounts, group="admission_plan")

    payload_names = [str(name) for name, _dim in getattr(arena, "PAYLOAD", ())]
    if not layers:
        builder.unavailable_field(
            "model.layers", "model has no attention layers", group="arena_kv")
    if caches is not None and len(caches) != len(layers):
        builder.unavailable_field(
            "cache.layer_count",
            f"cache has {len(caches)} layers but model has {len(layers)}",
            group="arena_kv",
        )
    for layer_index in range(len(layers)):
        for name in payload_names:
            field = (
                f"cache.layer_{layer_index:02d}.{name}"
                if caches is not None else
                f"injection.layer_{layer_index:02d}.{name}"
            )
            builder.require(
                field,
                group="arena_kv",
                reason=(
                    "target prefill lacks a complete per-layer cache or "
                    "cold-bootstrap injection payload"
                ),
            )
            builder.require(
                f"textual_sink.layer_{layer_index:02d}.{name}",
                group="sink_rows",
                reason="textual sink payload is incomplete",
            )
        builder.require(
            f"mask.layer_{layer_index:02d}.inputs",
            group="masks",
            reason="mask generator inputs are missing",
        )
        builder.require(
            f"mask.layer_{layer_index:02d}.allowed",
            group="masks",
            reason="canonical attention mask bytes are missing",
        )
        if hasattr(getattr(layers[layer_index], "self_attn", None), "sinks"):
            sinks = getattr(layers[layer_index].self_attn, "sinks", None)
            if sinks is not None:
                builder.require(
                    f"learned_sink.layer_{layer_index:02d}",
                    group="sink_rows",
                    reason="learned attention-sink row is missing",
                )
    for field in (
        "model.rope_cos", "model.rope_sin", "arena.pos", "arena.live_shift",
        "positions.query_absolute_rows", "positions.physical_sections",
    ):
        builder.require(
            field,
            group="positions_rope",
            reason="position/RoPE capture is incomplete",
        )
    for field in (
        "text.question_utf8", "text.sink_utf8", "tokens.prompt_ids",
        "tokens.sink_ids",
    ):
        builder.require(
            field, group="text_tokens", reason="exact text/token capture is incomplete")
    builder.require(
        "tokens.live_ids", group="live_tokens",
        reason="exact chronological live token ledger is missing")
    for field in REQUIRED_ADMISSION_FIELDS:
        builder.require(
            f"admission.{field}", group="admission_plan",
            reason="full admission/routing plan is incomplete")
        if admission_plan is not None and admission_plan.get(field) is None:
            builder.unavailable_field(
                f"admission.{field}",
                "required admission field is null",
                group="admission_plan",
            )
    if admission_plan is not None:
        final_mounts = [int(value) for value in admission_plan.get("final_mounts", ())]
        if final_mounts != mounts:
            builder.unavailable_field(
                "admission.final_mounts_consistency",
                f"plan final_mounts={final_mounts} but arena cur_mounts={mounts}",
                group="admission_plan",
            )
        attempts = list(admission_plan.get("ladder_attempts", ()))
        ordinal = admission_plan.get("current_attempt_ordinal")
        if not isinstance(ordinal, int) or not (0 <= ordinal < len(attempts)):
            builder.unavailable_field(
                "admission.current_attempt_ordinal_consistency",
                f"current ordinal {ordinal!r} is outside {len(attempts)} attempts",
                group="admission_plan",
            )
        else:
            current = attempts[ordinal]
            planned = [int(value) for value in current.get("planned", ())]
            clean = bool(current.get("clean_room"))
            if planned != [
                int(value) for value in admission_plan.get("current_planned", ())
            ] or clean != bool(admission_plan.get("current_clean_room")):
                builder.unavailable_field(
                    "admission.current_attempt_schedule_consistency",
                    "current attempt does not equal its entry in the captured ladder",
                    group="admission_plan",
                )
        if admission_plan.get("selection_state") != "PENDING_AT_PREFILL":
            builder.unavailable_field(
                "admission.selection_state",
                "ultimate ladder selection is unknowable before generation/grounding",
                group="admission_plan",
            )
        if admission_plan.get("current_attempt_ordinal") != provenance.get(
            "attempt_ordinal"
        ):
            builder.unavailable_field(
                "admission.provenance_attempt_consistency",
                "admission ordinal differs from capture provenance",
                group="admission_plan",
            )

    return builder.finish(identity)


@contextmanager
def stop_at_next_forward(
    arena: Any,
    output_dir: Path,
    *,
    label: str,
    provenance: (
        Mapping[str, Any] | Callable[[], Mapping[str, Any]]
    ),
    question: str,
    admission_plan: Mapping[str, Any] | Callable[[], Mapping[str, Any] | None] | None,
    live_token_ids: Sequence[int] | Callable[[], Sequence[int] | None] | None,
    sink_text: str | None,
    sink_token_ids: Sequence[int] | None,
    explicit_identity: Mapping[str, Any],
    fork_callback: Callable[
        [Path, list[int], Callable[[Sequence[int]], dict[str, Any]]], Any
    ] | None = None,
) -> Iterator[dict[str, Any]]:
    """Capture the next prefill, optionally fork inline, then continue lived.

    The callback runs with the production ``_forward`` temporarily restored.
    Its ``restore(withheld_aliases)`` closure always hydrates from the just-made
    lived snapshot in the same process.  After the callback, ZERO is restored
    once more and the original lived prefill runs exactly once.
    """
    original = arena._forward
    result: dict[str, Any] = {
        "manifest_path": None,
        "capture_count": 0,
        "fork_result": None,
        "canonical_zero_restore": None,
    }

    def resolve(value):
        return value() if callable(value) else value

    def wrapped(ids, *args, **kwargs):
        if result["manifest_path"] is None:
            path = capture_arena_snapshot(
                arena,
                output_dir,
                label=label,
                provenance=resolve(provenance),
                question=question,
                prompt_ids=ids,
                admission_plan=resolve(admission_plan),
                live_token_ids=resolve(live_token_ids),
                sink_text=sink_text,
                sink_token_ids=sink_token_ids,
                explicit_identity=explicit_identity,
            )
            result["manifest_path"] = path
            result["capture_count"] = 1
            if fork_callback is not None:
                manifest_path = Path(path)

                def restore(withheld: Sequence[int] = ()) -> dict[str, Any]:
                    return restore_prefill_fork(
                        arena,
                        manifest_path,
                        withheld_mounts=withheld,
                        target_identity=explicit_identity,
                        require_same_process_index=True,
                        require_finalized_source=False,
                    )

                # Branches resume at the captured _forward boundary; routing,
                # L2, fit, mount, and admission must never be rerun here.
                arena._forward = original
                try:
                    result["fork_result"] = fork_callback(
                        manifest_path,
                        [int(value) for value in ids],
                        restore,
                    )
                    result["canonical_zero_restore"] = restore(())
                finally:
                    # Whether successful or fatal, never leave the interception
                    # wrapper installed for decode or a later turn.
                    arena._forward = original
        return original(ids, *args, **kwargs)

    arena._forward = wrapped
    try:
        yield result
    finally:
        arena._forward = original
    if result["manifest_path"] is None or result["capture_count"] != 1:
        raise SnapshotError("registered target prefill was not captured exactly once")


def load_snapshot(path: Path) -> dict[str, Any]:
    path = Path(path)
    manifest_path = path / "manifest.json" if path.is_dir() else path
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("schema") != SNAPSHOT_SCHEMA:
        raise SnapshotError(f"unsupported snapshot schema: {manifest_path}")
    expected = value.get("manifest_payload_sha256")
    unsigned = dict(value)
    unsigned.pop("manifest_payload_sha256", None)
    if expected != _sha256_bytes(_canonical_json(unsigned)):
        raise SnapshotError(f"snapshot manifest digest mismatch: {manifest_path}")
    identity = dict(value.get("identity") or {})
    frame_digest = identity.pop("frame_sha256", None)
    if frame_digest != _sha256_bytes(_canonical_json(identity)):
        raise SnapshotError(f"snapshot frame digest mismatch: {manifest_path}")
    blob_root = (manifest_path.parent / "blobs").resolve()
    for field, record in (value.get("arrays") or {}).items():
        blob = (manifest_path.parent / str(record.get("blob"))).resolve()
        try:
            blob.relative_to(blob_root)
        except ValueError as exc:
            raise SnapshotError(
                f"snapshot blob escapes content-addressed root for {field}: {blob}"
            ) from exc
        if not blob.is_file():
            raise SnapshotError(f"snapshot blob is missing for {field}: {blob}")
        if blob.stat().st_size != int(record.get("bytes", -1)):
            raise SnapshotError(f"snapshot blob size mismatch for {field}")
        if _sha256_file(blob) != record.get("sha256"):
            raise SnapshotError(f"snapshot blob digest mismatch for {field}")
    value["_manifest_path"] = str(manifest_path.resolve())
    return value


def _load_validated_snapshot_array(
    value: Mapping[str, Any], field: str,
) -> np.ndarray:
    """Read one blob from a mapping already returned by ``load_snapshot``."""
    manifest_path = Path(str(value.get("_manifest_path", "")))
    if not manifest_path.is_file():
        raise SnapshotError("snapshot mapping lacks a validated manifest path")
    record = (value.get("arrays") or {}).get(str(field))
    if not isinstance(record, Mapping):
        raise SnapshotError(f"snapshot array is missing: {field}")
    blob = (manifest_path.parent / str(record.get("blob", ""))).resolve()
    blob_root = (manifest_path.parent / "blobs").resolve()
    try:
        blob.relative_to(blob_root)
    except ValueError as exc:
        raise SnapshotError(
            f"snapshot blob escapes the content-addressed blob root for {field}: {blob}"
        ) from exc
    payload = blob.read_bytes()
    if len(payload) != int(record.get("bytes", -1)):
        raise SnapshotError(f"snapshot blob size mismatch for {field}")
    if _sha256_bytes(payload) != record.get("sha256"):
        raise SnapshotError(f"snapshot blob digest mismatch for {field}")
    try:
        dtype = np.dtype(str(record["host_dtype"]))
        shape = tuple(int(value) for value in record["shape"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SnapshotError(f"invalid array metadata for {field}") from exc
    if dtype.hasobject:
        raise SnapshotError(f"object dtype is forbidden for snapshot array {field}")
    if record.get("order") not in (None, "C"):
        raise SnapshotError(f"snapshot array is not C-contiguous for {field}")
    expected = int(dtype.itemsize)
    for dimension in shape:
        if dimension < 0:
            raise SnapshotError(f"negative snapshot dimension for {field}: {shape}")
        expected *= dimension
        if expected > len(payload) and dimension != 0:
            raise SnapshotError(
                f"snapshot array byte geometry exceeds blob for {field}")
    if expected != len(payload):
        raise SnapshotError(
            f"snapshot array byte geometry mismatch for {field}: "
            f"metadata={expected} blob={len(payload)}")
    array = np.frombuffer(payload, dtype=dtype).reshape(shape)
    return np.ascontiguousarray(array.copy())


def load_snapshot_array(
    snapshot: Path | Mapping[str, Any], field: str,
) -> np.ndarray:
    """Load one validated snapshot blob as a private C-contiguous array."""
    if isinstance(snapshot, Mapping):
        shown = dict(snapshot)
        manifest_path = Path(str(shown.get("_manifest_path", "")))
        value = load_snapshot(manifest_path)
        shown_without_path = dict(shown)
        shown_without_path.pop("_manifest_path", None)
        loaded_without_path = dict(value)
        loaded_without_path.pop("_manifest_path", None)
        if _canonical_json(_jsonable(shown_without_path)) != _canonical_json(
            _jsonable(loaded_without_path)
        ):
            raise SnapshotError("snapshot mapping differs from its signed manifest")
    else:
        value = load_snapshot(Path(snapshot))
    return _load_validated_snapshot_array(value, field)


def compare_fork_substrate(
    lived_path: Path,
    fork_path: Path,
) -> dict[str, Any]:
    """Compare the model-visible substrate while excluding arm provenance.

    A fork is expected to run in a different execution branch, so its label,
    process provenance, and linked answer cannot be byte-equal to the lived
    capture.  DET1.4's ZERO gate instead requires exact identity, state, byte
    semantics, and canonical array value bytes.  This comparator deliberately
    has no delta allowlist; any intervention belongs in a separate G0 receipt.
    """
    lived = load_snapshot(Path(lived_path))
    fork = load_snapshot(Path(fork_path))
    lived_manifest = Path(lived.pop("_manifest_path"))
    fork_manifest = Path(fork.pop("_manifest_path"))
    rows: list[dict[str, Any]] = []

    def scalar_row(field: str, left: Any, right: Any, group: str) -> None:
        equal = _canonical_json(_jsonable(left)) == _canonical_json(_jsonable(right))
        rows.append({
            "group": group,
            "field": field,
            "status": "EQUAL" if equal else "DIVERGENT",
            "lived": _jsonable(left),
            "fork": _jsonable(right),
        })

    scalar_row("metadata.phase", lived.get("phase"), fork.get("phase"), "metadata")
    scalar_row("metadata.complete", lived.get("complete"), fork.get("complete"), "metadata")
    scalar_row("identity", lived.get("identity"), fork.get("identity"), "identity")
    scalar_row(
        "byte_semantics", lived.get("byte_semantics"), fork.get("byte_semantics"),
        "byte_semantics",
    )
    lived_state = lived.get("state") or {}
    fork_state = fork.get("state") or {}
    for field in sorted(set(lived_state) | set(fork_state)):
        if field not in lived_state:
            rows.append({
                "group": "state",
                "field": f"state.{field}",
                "status": "MISSING_LIVED",
                "lived": None,
                "fork": _jsonable(fork_state[field]),
            })
        elif field not in fork_state:
            rows.append({
                "group": "state",
                "field": f"state.{field}",
                "status": "MISSING_FORK",
                "lived": _jsonable(lived_state[field]),
                "fork": None,
            })
        else:
            scalar_row(
                f"state.{field}", lived_state[field], fork_state[field], "state")

    lived_arrays = lived.get("arrays") or {}
    fork_arrays = fork.get("arrays") or {}
    for field in sorted(set(lived_arrays) | set(fork_arrays)):
        left = lived_arrays.get(field)
        right = fork_arrays.get(field)
        if left is None or right is None:
            rows.append({
                "group": "arrays",
                "field": f"arrays.{field}",
                "status": "MISSING_LIVED" if left is None else "MISSING_FORK",
                "lived": None if left is None else _jsonable(left),
                "fork": None if right is None else _jsonable(right),
            })
            continue
        projection = ("group", "source_dtype", "host_dtype", "shape", "bytes", "sha256")
        left_projection = {key: left.get(key) for key in projection}
        right_projection = {key: right.get(key) for key in projection}
        equal = left_projection == right_projection
        rows.append({
            "group": "arrays",
            "field": f"arrays.{field}",
            "status": "EQUAL" if equal else "DIVERGENT",
            "lived": left_projection,
            "fork": right_projection,
        })

    counts = dict(Counter(row["status"] for row in rows))
    passed = bool(rows) and counts.get("EQUAL", 0) == len(rows)
    return {
        "schema": "grm.det1_4.zero_fork_substrate_comparison.v1",
        "status": "PASS_EXACT_CANONICAL_VALUE_BYTES" if passed else "DIVERGENT",
        "gate_pass": passed,
        "intervention": "ZERO",
        "lived_manifest": {
            "path": str(lived_manifest),
            "bytes": lived_manifest.stat().st_size,
            "sha256": _sha256_file(lived_manifest),
        },
        "fork_manifest": {
            "path": str(fork_manifest),
            "bytes": fork_manifest.stat().st_size,
            "sha256": _sha256_file(fork_manifest),
        },
        "counts": counts,
        "rows": rows,
        "excluded_by_design": [
            "label",
            "provenance",
            "linked_answer",
            "capture_finalized",
            "manifest_payload_sha256",
            "blob_relative_paths",
        ],
        "byte_claim": (
            "exact C-contiguous canonical exported value-byte equality under "
            "the DET1.3 snapshot qualification; not literal device-storage bytes"
        ),
        "race_resume_authorized": False,
    }


def _fork_upload(array: np.ndarray, source_dtype: str, tensor_factory=None):
    array = np.ascontiguousarray(array)
    if tensor_factory is not None:
        return tensor_factory(array, str(source_dtype))
    try:
        import tensor_cuda as tc
    except Exception as exc:  # pragma: no cover - exercised only on GPU hosts
        raise SnapshotError(
            "TensorCUDA is required to hydrate a device fork; pass a "
            "tensor_factory only for a controlled CPU test") from exc
    try:
        value = tc.tensor(array)
        if str(getattr(value, "dtype", "")) != str(source_dtype):
            value = value.astype(str(source_dtype))
        return value
    except Exception as exc:  # pragma: no cover - device-specific failure
        raise SnapshotError(
            f"failed to upload fork array as {source_dtype}: {array.shape}") from exc


def _fork_export(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value
    elif hasattr(value, "numpy"):
        array = value.numpy()
    else:
        array = np.asarray(value)
    return np.ascontiguousarray(np.asarray(array))


def _remove_ranges(array: np.ndarray, axis: int, ranges: Sequence[tuple[int, int]]) -> np.ndarray:
    if not ranges:
        return np.ascontiguousarray(array.copy())
    length = int(array.shape[int(axis)])
    keep = np.ones(length, dtype=bool)
    for start, end in ranges:
        if not (0 <= int(start) <= int(end) <= length):
            raise SnapshotError(
                f"withholding range {(start, end)} exceeds axis length {length}")
        keep[int(start):int(end)] = False
    return np.ascontiguousarray(np.compress(keep, array, axis=int(axis)))


_PLANTED_DELTA_PATTERNS = tuple(re.compile(value) for value in (
    r"arrays\.(?:cache|injection)\.layer_\d{2}\.(?:k|v)",
    r"arrays\.mask\.layer_\d{2}\.allowed",
    r"state\.arena\.cur_mounts",
    r"state\.arena\.cur_mount_n",
    r"state\.attention\.layer_\d{2}\.graft_seats",
    r"state\.positions\.physical_sections\.mounted_payload",
    r"state\.cache\.lengths_by_layer",
    r"state\.mask\.layer_\d{2}\.inputs\.(?:cache_length|injection_length|key_length)",
    r"state\.admission\.(?:authoritative_mounts|rank_plan|current_planned|current_fitted|current_dropped|final_mounts)",
    r"state\.admission\.ladder_attempts\[\d+\]\.planned",
))


def _assert_registered_planted_deltas(deltas: Sequence[Mapping[str, Any]]) -> None:
    unexpected = sorted({
        str(row.get("field")) for row in deltas
        if not any(pattern.fullmatch(str(row.get("field")))
                   for pattern in _PLANTED_DELTA_PATTERNS)
    })
    if unexpected:
        raise SnapshotError(
            "fork attempted non-registered state deltas: " + ", ".join(unexpected))


def _fork_mount_ranges(
    snapshot: Mapping[str, Any],
    mounts: Sequence[int],
    n_sink: int,
    payload: Sequence[tuple[str, int]],
    layer_count: int,
) -> tuple[dict[int, tuple[int, int]], dict[int, int]]:
    arrays = snapshot.get("arrays") or {}
    ranges: dict[int, tuple[int, int]] = {}
    lengths: dict[int, int] = {}
    cursor = int(n_sink)
    for mount in mounts:
        records = []
        for layer_index in range(int(layer_count)):
            for name, dim in payload:
                field = (
                    f"mounted_graft.{int(mount)}."
                    f"layer_{layer_index:02d}.{name}"
                )
                record = arrays.get(field)
                if not isinstance(record, Mapping):
                    raise SnapshotError(
                        f"snapshot lacks mounted payload for graft {mount}: {field}")
                shape = tuple(int(value) for value in record.get("shape", ()))
                if not (0 <= int(dim) < len(shape)):
                    raise SnapshotError(
                        f"mounted payload has invalid sequence axis for {field}: "
                        f"axis={dim} shape={shape}")
                records.append((record, int(dim)))
        observed = {
            int(record["shape"][dim]) for record, dim in records
        }
        if len(observed) != 1:
            raise SnapshotError(
                f"mounted graft {mount} has inconsistent sequence lengths: {observed}")
        length = observed.pop()
        ranges[int(mount)] = (cursor, cursor + length)
        lengths[int(mount)] = length
        cursor += length
    return ranges, lengths


def compare_fork_hydration_delta(
    lived_path: Path,
    fork_path: Path,
    *,
    withheld_mounts: Sequence[int],
) -> dict[str, Any]:
    """Prove that a hydrated fork differs only by registered mount seats.

    This deliberately re-runs DET1.4's strict ZERO comparator.  Its expected
    result is ``DIVERGENT`` under a planted miss; this wrapper independently
    derives the exact fork projection from the lived manifest and accepts only
    when the ZERO comparator's non-equal rows are exactly that projection.
    The restore receipt is not an allowlist input.
    """
    lived = load_snapshot(Path(lived_path))
    fork = load_snapshot(Path(fork_path))
    lived_manifest = Path(lived["_manifest_path"])
    fork_manifest = Path(fork["_manifest_path"])
    if not (lived.get("complete") and fork.get("complete")):
        raise SnapshotError("fork-hydration delta requires two complete snapshots")

    requested = [int(value) for value in withheld_mounts]
    if not requested:
        raise SnapshotError("fork-hydration delta requires a registered target")
    if len(set(requested)) != len(requested):
        raise SnapshotError("registered planted-miss aliases contain duplicates")
    withheld = set(requested)

    lived_state = copy.deepcopy(lived.get("state") or {})
    fork_state = copy.deepcopy(fork.get("state") or {})
    source_mounts = [int(value) for value in lived_state.get("arena.cur_mounts", ())]
    if len(set(source_mounts)) != len(source_mounts):
        raise SnapshotError("lived delta source contains duplicate mounted grafts")
    source_mounted_aliases = [value for value in source_mounts if value in withheld]
    if not source_mounted_aliases:
        raise SnapshotError(
            "registered planted target has no hydrated seats in the lived snapshot: "
            f"targets={sorted(withheld)} mounts={source_mounts}")
    already_absent_aliases = sorted(withheld - set(source_mounted_aliases))
    lived_aliases = {
        int(value[0])
        for value in lived_state.get("live.segments", ())
        if value and value[0] is not None and int(value[0]) in withheld
    }
    if lived_aliases:
        raise SnapshotError(
            "registered planted target remains model-visible in live seats: "
            f"{sorted(lived_aliases)}")

    identity = lived.get("identity") or {}
    payload = [
        (str(value["name"]), int(value["sequence_dim"]))
        for value in identity.get("dialect_payload", ())
    ]
    layer_count = int(identity.get("layer_count", -1))
    if not payload or layer_count <= 0:
        raise SnapshotError("lived delta source lacks payload/layer geometry")
    n_sink = int(lived_state.get("arena.n_sink", 0))
    mount_ranges, mount_lengths = _fork_mount_ranges(
        lived, source_mounts, n_sink, payload, layer_count)
    removed_ranges = [mount_ranges[value] for value in source_mounted_aliases]
    removed_rows = sum(end - start for start, end in removed_ranges)
    source_mount_n = int(lived_state.get("arena.cur_mount_n", -1))
    if source_mount_n != sum(mount_lengths.values()):
        raise SnapshotError(
            "lived delta source mount count differs from mounted payload bytes")
    kept_mounts = [value for value in source_mounts if value not in withheld]

    # Independently derive every scalar parent row that the strict ZERO
    # comparator must see change.  This mirrors the registered semantics, not
    # a restore receipt's self-reported leaf list.
    expected_state = copy.deepcopy(lived_state)
    expected_state["arena.cur_mounts"] = list(kept_mounts)
    expected_state["arena.cur_mount_n"] = source_mount_n - removed_rows
    physical = copy.deepcopy(lived_state.get("positions.physical_sections") or {})
    physical["mounted_payload"] = [n_sink, n_sink + source_mount_n - removed_rows]
    expected_state["positions.physical_sections"] = physical
    caches_present = lived_state.get("arena.caches_present") is True
    if caches_present:
        expected_state["cache.lengths_by_layer"] = [
            int(value) - removed_rows
            for value in lived_state.get("cache.lengths_by_layer", ())
        ]
    active_key = "cache_length" if caches_present else "injection_length"
    for layer_index in range(layer_count):
        seat_field = f"attention.layer_{layer_index:02d}.graft_seats"
        expected_state[seat_field] = int(lived_state[seat_field]) - removed_rows
        mask_field = f"mask.layer_{layer_index:02d}.inputs"
        mask_inputs = copy.deepcopy(lived_state[mask_field])
        mask_inputs[active_key] = int(mask_inputs[active_key]) - removed_rows
        mask_inputs["key_length"] = int(mask_inputs["key_length"]) - removed_rows
        expected_state[mask_field] = mask_inputs

    admission = copy.deepcopy(lived_state.get("admission.full_plan") or {})
    if not admission:
        raise SnapshotError("lived delta source lacks the full admission plan")
    for key in (
        "rank_plan", "current_planned", "current_fitted",
        "current_dropped", "final_mounts",
    ):
        if isinstance(admission.get(key), list):
            admission[key] = [
                int(value) for value in admission[key]
                if int(value) not in withheld
            ]
    for attempt in admission.get("ladder_attempts", ()):
        if isinstance(attempt, dict) and isinstance(attempt.get("planned"), list):
            attempt["planned"] = [
                int(value) for value in attempt["planned"]
                if int(value) not in withheld
            ]
    admission["final_mounts"] = list(kept_mounts)
    expected_state["admission.full_plan"] = admission
    for key, value in admission.items():
        field = f"admission.{key}"
        if field in expected_state:
            expected_state[field] = copy.deepcopy(value)
    expected_state["admission.authoritative_mounts"] = list(kept_mounts)

    expected_state_delta_fields = {
        f"state.{field}"
        for field in set(lived_state) | set(expected_state)
        if _canonical_json(_jsonable(lived_state.get(field, "<MISSING>")))
        != _canonical_json(_jsonable(expected_state.get(field, "<MISSING>")))
    }
    expected_value_mismatches: list[str] = []
    for field in sorted(set(expected_state) | set(fork_state)):
        if field not in expected_state or field not in fork_state:
            expected_value_mismatches.append(f"state.{field}")
            continue
        if _canonical_json(_jsonable(expected_state[field])) != _canonical_json(
            _jsonable(fork_state[field])
        ):
            expected_value_mismatches.append(f"state.{field}")

    lived_arrays = lived.get("arrays") or {}
    fork_arrays = fork.get("arrays") or {}
    payload_dims = {name: dim for name, dim in payload}
    active_prefix = "cache" if caches_present else "injection"
    target_payload_fields = sorted(
        field for field in lived_arrays
        if any(
            field.startswith(f"mounted_graft.{target}.")
            for target in source_mounted_aliases
        )
    )
    if not target_payload_fields:
        raise SnapshotError("lived delta source has no target mounted-payload fields")
    target_payload_set = set(target_payload_fields)
    expected_array_delta_fields: set[str] = {
        f"arrays.{field}" for field in target_payload_fields
    }
    transformed_arrays: list[dict[str, Any]] = []
    retained_array_bindings: list[dict[str, Any]] = []
    projection_keys = (
        "group", "source_dtype", "host_dtype", "shape", "bytes", "sha256")

    for field in sorted(lived_arrays):
        source_record = lived_arrays[field]
        if field in target_payload_set:
            if field in fork_arrays:
                expected_value_mismatches.append(f"arrays.{field}")
            transformed_arrays.append({
                "field": f"arrays.{field}",
                "transformation": "TARGET_MOUNTED_PAYLOAD_ABSENT",
                "source_sha256": source_record.get("sha256"),
                "observed": "MISSING_FORK" if field not in fork_arrays else "PRESENT",
                "retained_bytes_exact": field not in fork_arrays,
            })
            continue

        axis = None
        transformation = None
        active_match = re.fullmatch(
            rf"{re.escape(active_prefix)}\.layer_\d{{2}}\.([^.]+)", field)
        if active_match and active_match.group(1) in payload_dims:
            axis = int(payload_dims[active_match.group(1)])
            transformation = "DELETE_REGISTERED_MOUNT_SEAT_RANGES"
        elif re.fullmatch(r"mask\.layer_\d{2}\.allowed", field):
            axis = 3
            transformation = "DELETE_REGISTERED_MOUNT_MASK_COLUMNS"

        source_array = _load_validated_snapshot_array(lived, field)
        expected_array = (
            source_array if axis is None
            else _remove_ranges(source_array, axis, removed_ranges)
        )
        expected_bytes = np.ascontiguousarray(expected_array).tobytes(order="C")
        expected_projection = {
            "group": source_record.get("group"),
            "source_dtype": source_record.get("source_dtype"),
            "host_dtype": expected_array.dtype.str,
            "shape": [int(value) for value in expected_array.shape],
            "bytes": len(expected_bytes),
            "sha256": _sha256_bytes(expected_bytes),
        }
        fork_record = fork_arrays.get(field)
        observed_projection = (
            None if fork_record is None else
            {key: fork_record.get(key) for key in projection_keys}
        )
        observed_bytes_equal = False
        if fork_record is not None:
            observed = _load_validated_snapshot_array(fork, field)
            observed_bytes_equal = (
                observed.tobytes(order="C") == expected_bytes
            )
        projection_equal = observed_projection == expected_projection
        if not (projection_equal and observed_bytes_equal):
            expected_value_mismatches.append(f"arrays.{field}")
        if transformation is not None:
            expected_array_delta_fields.add(f"arrays.{field}")
            transformed_arrays.append({
                "field": f"arrays.{field}",
                "transformation": transformation,
                "axis": int(axis),
                "removed_ranges": [
                    [int(start), int(end)] for start, end in removed_ranges],
                "source": {
                    "shape": list(source_record.get("shape") or ()),
                    "bytes": source_record.get("bytes"),
                    "sha256": source_record.get("sha256"),
                },
                "expected_fork": expected_projection,
                "observed_fork": observed_projection,
                "retained_bytes_exact": bool(
                    projection_equal and observed_bytes_equal),
            })
        else:
            retained_array_bindings.append({
                "field": f"arrays.{field}",
                "source_sha256": source_record.get("sha256"),
                "fork_sha256": (
                    None if fork_record is None else fork_record.get("sha256")),
                "bytes_equal": bool(projection_equal and observed_bytes_equal),
            })

    for field in sorted(set(fork_arrays) - set(lived_arrays)):
        expected_value_mismatches.append(f"arrays.{field}")

    expected_divergent = expected_state_delta_fields | expected_array_delta_fields
    strict_zero = compare_fork_substrate(lived_manifest, fork_manifest)
    observed_non_equal = {
        str(row["field"])
        for row in strict_zero.get("rows", ())
        if row.get("status") != "EQUAL"
    }
    unexpected = sorted(observed_non_equal - expected_divergent)
    missing = sorted(expected_divergent - observed_non_equal)
    expected_value_mismatches = sorted(set(expected_value_mismatches))
    exact_divergence_set = not unexpected and not missing
    non_delta_rows = [
        row for row in strict_zero.get("rows", ())
        if str(row.get("field")) not in expected_divergent
    ]
    non_delta_fields_equal = bool(non_delta_rows) and all(
        row.get("status") == "EQUAL" for row in non_delta_rows)

    fork_mounts = [int(value) for value in fork_state.get("arena.cur_mounts", ())]
    fork_live = {
        int(value[0])
        for value in fork_state.get("live.segments", ())
        if value and value[0] is not None
    }
    target_payload_absent = all(field not in fork_arrays for field in target_payload_fields)
    target_absence_checks = {
        "at_least_one_target_had_lived_seats": bool(source_mounted_aliases),
        "fork_mounts_equal_source_minus_target": fork_mounts == kept_mounts,
        "registered_aliases_absent_from_fork_mounts": not bool(
            withheld & set(fork_mounts)),
        "registered_aliases_absent_from_live_seats": not bool(withheld & fork_live),
        "source_target_payload_fields_absent_from_fork": target_payload_absent,
        "one_seat_range_per_source_target": (
            len(removed_ranges) == len(source_mounted_aliases)),
    }
    target_absence_pass = all(target_absence_checks.values())
    transformed_arrays_exact = all(
        row.get("retained_bytes_exact") is True for row in transformed_arrays)
    retained_arrays_exact = all(
        row.get("bytes_equal") is True for row in retained_array_bindings)
    gate_pass = bool(
        target_absence_pass
        and exact_divergence_set
        and non_delta_fields_equal
        and not expected_value_mismatches
        and transformed_arrays_exact
        and retained_arrays_exact
    )
    strict_rows = list(strict_zero.get("rows", ()))
    delta_rows = [
        row for row in strict_rows if str(row.get("field")) in expected_divergent]
    return {
        "schema": "grm.det1_6.fork_hydration_delta.v1",
        "status": (
            "PASS_EXACT_REGISTERED_DELTA_CANONICAL_VALUE_BYTES"
            if gate_pass else "FAIL_FORK_HYDRATION_DELTA"),
        "gate_pass": gate_pass,
        "intervention": "REGISTERED_PLANTED_MISS_WITHHOLDING",
        "lived_manifest": {
            "path": str(lived_manifest),
            "bytes": lived_manifest.stat().st_size,
            "sha256": _sha256_file(lived_manifest),
        },
        "fork_manifest": {
            "path": str(fork_manifest),
            "bytes": fork_manifest.stat().st_size,
            "sha256": _sha256_file(fork_manifest),
        },
        "withheld_logical_aliases": sorted(withheld),
        "source_mounted_aliases": source_mounted_aliases,
        "already_absent_aliases": already_absent_aliases,
        "source_mounts": source_mounts,
        "fork_mounts": fork_mounts,
        "seat_ranges": [
            {
                "graft_id": int(value),
                "range": [
                    int(mount_ranges[value][0]), int(mount_ranges[value][1])],
            }
            for value in source_mounted_aliases
        ],
        "target_absence": {
            "gate_pass": target_absence_pass,
            "checks": target_absence_checks,
            "source_target_payload_fields": target_payload_fields,
        },
        "strict_zero_comparator": {
            "schema": strict_zero.get("schema"),
            "status": strict_zero.get("status"),
            "gate_pass": strict_zero.get("gate_pass"),
            "counts": strict_zero.get("counts"),
            "row_count": len(strict_rows),
            "rows_sha256": _sha256_bytes(_canonical_json(strict_rows)),
        },
        "expected_divergent_fields": sorted(expected_divergent),
        "observed_non_equal_fields": sorted(observed_non_equal),
        "unexpected_divergent_fields": unexpected,
        "missing_expected_divergent_fields": missing,
        "exact_divergence_set": exact_divergence_set,
        "non_delta_fields_equal": non_delta_fields_equal,
        "non_delta_field_count": len(non_delta_rows),
        "non_delta_rows_sha256": _sha256_bytes(_canonical_json(non_delta_rows)),
        "expected_fork_value_mismatches": expected_value_mismatches,
        "transformed_arrays": transformed_arrays,
        "retained_array_count": len(retained_array_bindings),
        "retained_arrays_exact": retained_arrays_exact,
        "retained_array_bindings_sha256": _sha256_bytes(
            _canonical_json(retained_array_bindings)),
        "delta_rows": delta_rows,
        "byte_semantics": copy.deepcopy(lived.get("byte_semantics")),
        "byte_claim": (
            "exact C-contiguous canonical exported value bytes after deleting "
            "only the registered mounted-seat ranges; not literal device bytes"
        ),
    }


def restore_prefill_fork(
    arena: Any,
    snapshot_path: Path,
    *,
    withheld_mounts: Sequence[int] = (),
    target_identity: Mapping[str, Any],
    expected_frame_sha256: str | None = None,
    require_same_process_index: bool = True,
    require_finalized_source: bool = True,
    tensor_factory=None,
) -> dict[str, Any]:
    """Hydrate the next prefill from a lived DET1.3 snapshot.

    ZERO intervention restores canonical exported values verbatim.  The only
    supported mutation is registered planted-miss withholding: exact mounted
    seat ranges are deleted from cache/injection K/V and from the captured
    canonical allowed masks.  Routing/index state is deliberately not rebuilt;
    callers must use the same lived process/repository for D-LQR.
    """
    snapshot = load_snapshot(Path(snapshot_path))
    manifest_path = Path(snapshot["_manifest_path"])
    if not snapshot.get("complete"):
        raise SnapshotError("fork source must be a complete lived snapshot")
    if require_finalized_source and not snapshot.get("capture_finalized"):
        raise SnapshotError("fork source must be finalized before cross-process use")
    if not require_finalized_source and not require_same_process_index:
        raise SnapshotError(
            "an unfinalized lived capture may only fork inline in its same process")
    if snapshot.get("phase") != CAPTURE_PHASE:
        raise SnapshotError(f"fork source is not at {CAPTURE_PHASE}")
    provenance = snapshot.get("provenance") or {}
    if not (
        snapshot.get("label") == "lived"
        and provenance.get("arm") == "lived"
        and provenance.get("protocol") == ARM_PROTOCOLS["lived"]
        and provenance.get("capture_boundary") == CAPTURE_PHASE
    ):
        raise SnapshotError(
            "fork source must be a registered lived capture, never a replay capture")
    identity = snapshot.get("identity") or {}
    observed_identity = infer_frame_identity(arena, target_identity)
    if observed_identity.get("frame_sha256") != identity.get("frame_sha256"):
        raise SnapshotError(
            "fork target arena frame differs from the lived snapshot: "
            f"target={observed_identity.get('frame_sha256')} "
            f"snapshot={identity.get('frame_sha256')}")
    if (
        expected_frame_sha256 is not None
        and identity.get("frame_sha256") != str(expected_frame_sha256)
    ):
        raise SnapshotError(
            "fork target frame differs from lived snapshot: "
            f"expected={expected_frame_sha256} "
            f"snapshot={identity.get('frame_sha256')}")
    if require_same_process_index:
        pid = os.getpid()
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        close = stat_text.rfind(")")
        if close < 0:
            raise SnapshotError("cannot parse current process start time")
        start_ticks = int(stat_text[close + 2:].split()[19])
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8").strip()
        current_process = process_instance_sha256(pid, start_ticks, boot_id)
        if provenance.get("process_instance_sha256") != current_process:
            raise SnapshotError(
                "forked detector execution requires the full lived repository/index "
                "from the same process; the snapshot captures mounted payload only")
    layers = list(getattr(getattr(arena, "m", None), "layers", ()) or ())
    if len(layers) != int(identity.get("layer_count", -1)):
        raise SnapshotError(
            f"fork layer-count mismatch: arena={len(layers)} "
            f"snapshot={identity.get('layer_count')}")
    payload = [(str(name), int(dim)) for name, dim in getattr(arena, "PAYLOAD", ())]
    if not payload:
        raise SnapshotError("fork target arena has no payload descriptor")
    routing_index = None
    if require_same_process_index:
        from scripts.grm_det1_common import routing_index_digest

        routing_index = routing_index_digest(arena)

    state = copy.deepcopy(snapshot.get("state") or {})
    source_mounts = [int(value) for value in state.get("arena.cur_mounts", ())]
    if len(set(source_mounts)) != len(source_mounts):
        raise SnapshotError("fork source contains duplicate mounted graft indices")
    withheld_values = [int(value) for value in withheld_mounts]
    if len(set(withheld_values)) != len(withheld_values):
        raise SnapshotError("registered planted-miss aliases contain duplicates")
    withheld = set(withheld_values)
    live_withheld = sorted({
        int(value[0])
        for value in state.get("live.segments", ())
        if value and value[0] is not None and int(value[0]) in withheld
    })
    if live_withheld:
        raise SnapshotError(
            "registered planted-miss target remains model-visible in live seats: "
            f"{live_withheld}")
    kept_mounts = [value for value in source_mounts if value not in withheld]
    n_sink = int(state.get("arena.n_sink", 0))
    mount_ranges, mount_lengths = _fork_mount_ranges(
        snapshot, source_mounts, n_sink, payload, len(layers))
    removed_ranges = [mount_ranges[value] for value in source_mounts if value in withheld]
    if withheld and not removed_ranges:
        raise SnapshotError(
            "registered planted-miss withholding would be a model-state no-op: "
            f"withheld={sorted(withheld)} mounted={source_mounts}")
    removed_rows = sum(end - start for start, end in removed_ranges)
    source_mount_n = int(state.get("arena.cur_mount_n", -1))
    if source_mount_n != sum(mount_lengths.values()):
        raise SnapshotError(
            "snapshot cur_mount_n differs from mounted payload lengths: "
            f"state={source_mount_n} payload={sum(mount_lengths.values())}")

    live_count = int(state.get("live.segment_token_count", 0))
    caches_present = state.get("arena.caches_present") is True
    active_lengths = []
    for layer_index in range(len(layers)):
        component_lengths = []
        for name, dim in payload:
            field = (
                f"cache.layer_{layer_index:02d}.{name}"
                if caches_present else
                f"injection.layer_{layer_index:02d}.{name}"
            )
            record = (snapshot.get("arrays") or {}).get(field)
            if not isinstance(record, Mapping):
                raise SnapshotError(f"fork source lacks active K/V field {field}")
            shape = tuple(int(value) for value in record.get("shape", ()))
            if not (0 <= dim < len(shape)):
                raise SnapshotError(
                    f"fork active K/V sequence axis is invalid for {field}")
            component_lengths.append(int(shape[dim]))
        if len(set(component_lengths)) != 1:
            raise SnapshotError(
                f"fork active K/V components disagree at layer {layer_index}: "
                f"{component_lengths}")
        active_length = component_lengths[0]
        expected_active = (
            n_sink + source_mount_n + live_count
            if caches_present else n_sink + source_mount_n
        )
        if active_length != expected_active:
            raise SnapshotError(
                f"fork active K/V length mismatch at layer {layer_index}: "
                f"observed={active_length} expected={expected_active}")
        active_lengths.append(active_length)
        mask_inputs = state.get(f"mask.layer_{layer_index:02d}.inputs") or {}
        if (
            mask_inputs.get("attention_mode") != "standard"
            or mask_inputs.get("representation")
            != "uint8_canonical_allowed_equivalent"
        ):
            raise SnapshotError(
                f"fork source mask semantics are unsupported at layer {layer_index}")
        prompt_count = int(state.get("tokens.prompt_count", -1))
        expected_inputs = {
            "query_length": prompt_count,
            "cache_length": active_length if caches_present else 0,
            "injection_length": 0 if caches_present else active_length,
            "key_length": active_length + prompt_count,
        }
        for key, expected_value in expected_inputs.items():
            if int(mask_inputs.get(key, -1)) != expected_value:
                raise SnapshotError(
                    f"fork mask input {key} disagrees at layer {layer_index}: "
                    f"observed={mask_inputs.get(key)} expected={expected_value}")
        mask_record = (snapshot.get("arrays") or {}).get(
            f"mask.layer_{layer_index:02d}.allowed") or {}
        if tuple(mask_record.get("shape") or ()) != (
            1, 1, prompt_count, active_length + prompt_count,
        ):
            raise SnapshotError(
                f"fork mask record geometry disagrees at layer {layer_index}")
        if int(state.get(
            f"attention.layer_{layer_index:02d}.graft_seats", -1
        )) != n_sink + source_mount_n:
            raise SnapshotError(
                f"fork graft-seat geometry disagrees at layer {layer_index}")
    expected_cache_lengths = active_lengths if caches_present else [0] * len(layers)
    if state.get("cache.lengths_by_layer") != expected_cache_lengths:
        raise SnapshotError(
            "fork cache-length ledger differs from active K/V geometry")
    physical = state.get("positions.physical_sections") or {}
    if physical.get("textual_sink") != [0, n_sink] or physical.get(
        "mounted_payload"
    ) != [n_sink, n_sink + source_mount_n]:
        raise SnapshotError("fork physical-section geometry is inconsistent")

    verification: dict[str, dict[str, Any]] = {}
    deltas: list[dict[str, Any]] = []

    def record_delta(field: str, source: Any, fork: Any) -> None:
        if _canonical_json(_jsonable(source)) == _canonical_json(_jsonable(fork)):
            return
        deltas.append({
            "field": str(field),
            "reason": "registered_planted_miss_withholding",
            "source": _jsonable(source),
            "fork": _jsonable(fork),
        })

    def restored_array(field: str, *, axis: int | None = None):
        source = _load_validated_snapshot_array(snapshot, field)
        value = source if axis is None else _remove_ranges(source, axis, removed_ranges)
        if value.tobytes(order="C") != source.tobytes(order="C"):
            deltas.append({
                "field": f"arrays.{field}",
                "reason": "registered_planted_miss_withholding",
                "source_sha256": _sha256_bytes(source.tobytes(order="C")),
                "fork_sha256": _sha256_bytes(value.tobytes(order="C")),
                "removed_ranges": [[int(a), int(b)] for a, b in removed_ranges],
            })
        return value

    def upload_checked(field: str, array: np.ndarray):
        record = snapshot["arrays"][field]
        uploaded = _fork_upload(array, str(record["source_dtype"]), tensor_factory)
        observed = _fork_export(uploaded)
        payload_bytes = observed.tobytes(order="C")
        expected_bytes = np.ascontiguousarray(array).tobytes(order="C")
        if payload_bytes != expected_bytes:
            raise SnapshotError(
                f"fork upload/re-export changed canonical values for {field}")
        verification[field] = {
            "source_dtype": record["source_dtype"],
            "host_dtype": observed.dtype.str,
            "shape": [int(value) for value in observed.shape],
            "sha256": _sha256_bytes(payload_bytes),
            "bytes_equal_expected_fork_value": True,
        }
        return uploaded

    # Model-owned position tables and learned sinks are part of the lived
    # model-visible substrate and are restored before any forked forward.
    model = arena.m
    for name in ("rope_cos", "rope_sin"):
        field = f"model.{name}"
        array = restored_array(field)
        setattr(model, name, upload_checked(field, array))
    model._rope_len = int(state["model.rope_len"])
    for layer_index, layer in enumerate(layers):
        att = layer.self_attn
        field = f"learned_sink.layer_{layer_index:02d}"
        att.sinks = upload_checked(field, restored_array(field))

    # Restore host textual sink payload and mounted device payloads.  The
    # direct injection/cache below remains the authoritative next-forward K/V.
    sink_h = []
    for layer_index in range(len(layers)):
        values = {}
        for name, _dim in payload:
            field = f"textual_sink.layer_{layer_index:02d}.{name}"
            values[name] = restored_array(field)
            verification[field] = {
                "source_dtype": snapshot["arrays"][field]["source_dtype"],
                "host_dtype": values[name].dtype.str,
                "shape": [int(value) for value in values[name].shape],
                "sha256": _sha256_bytes(values[name].tobytes(order="C")),
                "bytes_equal_expected_fork_value": True,
            }
        sink_h.append(values)
    arena.sink_h = sink_h
    grafts = list(getattr(arena, "grafts", ()) or ())
    invalid_withheld = sorted(value for value in withheld if not (0 <= value < len(grafts)))
    if invalid_withheld:
        raise SnapshotError(
            "fork target lacks planted logical aliases: "
            f"{invalid_withheld}")
    for mount in source_mounts:
        if not (0 <= mount < len(grafts)):
            raise SnapshotError(
                "same-process fork target lacks mounted graft index "
                f"{mount}; reconstruction is forbidden")
        if int(grafts[mount].get("ntok", -1)) != int(mount_lengths[mount]):
            raise SnapshotError(
                "same-process fork target mounted-graft length differs from "
                f"the lived snapshot: graft={mount} "
                f"target={grafts[mount].get('ntok')} "
                f"snapshot={mount_lengths[mount]}")
        mounted_layers = []
        for layer_index in range(len(layers)):
            values = {}
            for name, _dim in payload:
                field = f"mounted_graft.{mount}.layer_{layer_index:02d}.{name}"
                values[name] = upload_checked(field, restored_array(field))
            mounted_layers.append(values)
        grafts[mount]["h"] = mounted_layers

    if caches_present:
        caches = []
        for layer_index in range(len(layers)):
            values = []
            for name, dim in payload:
                field = f"cache.layer_{layer_index:02d}.{name}"
                array = restored_array(field, axis=dim)
                values.append(upload_checked(field, array))
            caches.append(tuple(values))
        arena.caches = caches
        for layer in layers:
            layer.self_attn.inject_kv = None
    else:
        arena.caches = None
        for layer_index, layer in enumerate(layers):
            values = []
            for name, dim in payload:
                field = f"injection.layer_{layer_index:02d}.{name}"
                array = restored_array(field, axis=dim)
                values.append(upload_checked(field, array))
            scale = state.get(f"injection.layer_{layer_index:02d}.component_2", 1.0)
            layer.self_attn.inject_kv = (*values, float(scale))

    prompt_ids = _load_validated_snapshot_array(snapshot, "tokens.prompt_ids").astype(
        np.int64, copy=False).reshape(-1)
    sink_ids = _load_validated_snapshot_array(snapshot, "tokens.sink_ids").astype(
        np.int64, copy=False).reshape(-1)
    live_ids = _load_validated_snapshot_array(snapshot, "tokens.live_ids").astype(
        np.int64, copy=False).reshape(-1)
    if len(sink_ids) != n_sink:
        raise SnapshotError("fork sink-token ledger disagrees with n_sink")
    live_segs = [
        (None if value[0] is None else int(value[0]), int(value[1]))
        for value in state.get("live.segments", ())
    ]
    if sum(count for _graft, count in live_segs) != len(live_ids):
        raise SnapshotError("fork live-token ledger disagrees with live segments")

    for name in (
        "pos", "n_sink", "live_shift", "width", "topk", "live_turns",
        "max_live", "cache_deposits", "revision_resolution",
        "decisive_admission", "length_debias", "route_layer", "route_backend",
    ):
        if f"arena.{name}" in state:
            setattr(arena, name, copy.deepcopy(state[f"arena.{name}"]))
    arena.live_segs = live_segs
    arena.cur_mounts = kept_mounts
    arena.cur_mount_n = source_mount_n - removed_rows
    if withheld:
        record_delta("state.arena.cur_mounts", source_mounts, kept_mounts)
        record_delta(
            "state.arena.cur_mount_n", source_mount_n, arena.cur_mount_n)
        source_sections = state.get("positions.physical_sections") or {}
        record_delta(
            "state.positions.physical_sections.mounted_payload",
            source_sections.get("mounted_payload"),
            [n_sink, n_sink + int(arena.cur_mount_n)],
        )
        if caches_present:
            record_delta(
                "state.cache.lengths_by_layer",
                state.get("cache.lengths_by_layer"),
                [int(value) - removed_rows for value in active_lengths],
            )
    for layer_index, layer in enumerate(layers):
        att = layer.self_attn
        if str(getattr(att, "attention_mode", "unknown")) != "standard":
            raise SnapshotError(
                "DET1.4 captured-mask consumption is implemented only for "
                f"the registered standard attention path; layer={layer_index} "
                f"mode={getattr(att, 'attention_mode', None)!r}")
        source_graft_seats = int(
            state[f"attention.layer_{layer_index:02d}.graft_seats"])
        att.graft_seats = source_graft_seats - removed_rows
        if withheld:
            record_delta(
                f"state.attention.layer_{layer_index:02d}.graft_seats",
                source_graft_seats,
                att.graft_seats,
            )
        att.live_shift = copy.deepcopy(
            state[f"attention.layer_{layer_index:02d}.live_shift"])
        mask_field = f"mask.layer_{layer_index:02d}.allowed"
        mask = restored_array(mask_field, axis=3)
        expected_keys = (
            int(arena.caches[layer_index][0].shape[payload[0][1]])
            if caches_present else
            int(att.inject_kv[0].shape[payload[0][1]])
        ) + int(prompt_ids.size)
        if mask.shape != (1, 1, int(prompt_ids.size), expected_keys):
            raise SnapshotError(
                f"fork mask geometry mismatch for layer {layer_index}: "
                f"mask={mask.shape} expected={(1, 1, int(prompt_ids.size), expected_keys)}")
        if withheld:
            source_inputs = state[f"mask.layer_{layer_index:02d}.inputs"]
            fork_inputs = dict(source_inputs)
            active_key = "cache_length" if caches_present else "injection_length"
            fork_inputs[active_key] = int(fork_inputs[active_key]) - removed_rows
            fork_inputs["key_length"] = int(fork_inputs["key_length"]) - removed_rows
            for key in (active_key, "key_length"):
                record_delta(
                    f"state.mask.layer_{layer_index:02d}.inputs.{key}",
                    source_inputs[key],
                    fork_inputs[key],
                )
        att._det1_fork_allowed_mask = mask
        att._det1_fork_mask_consumed = False
        verification[mask_field] = {
            "source_dtype": snapshot["arrays"][mask_field]["source_dtype"],
            "host_dtype": mask.dtype.str,
            "shape": [int(value) for value in mask.shape],
            "sha256": _sha256_bytes(mask.tobytes(order="C")),
            "bytes_equal_expected_fork_value": True,
            "semantics": "canonical_uint8_allowed_equivalent_consumed_once",
        }

    admission = copy.deepcopy(state.get("admission.full_plan") or {})
    if not admission:
        admission = {
            key.removeprefix("admission."): copy.deepcopy(value)
            for key, value in state.items() if key.startswith("admission.")
        }
    if withheld:
        for key in (
            "rank_plan", "current_planned", "current_fitted",
            "current_dropped", "final_mounts",
        ):
            if isinstance(admission.get(key), list):
                before = list(admission[key])
                admission[key] = [
                    int(value) for value in before if int(value) not in withheld]
                if before != admission[key]:
                    record_delta(
                        f"state.admission.{key}", before, admission[key])
        for attempt in admission.get("ladder_attempts", ()):
            if isinstance(attempt, dict) and isinstance(attempt.get("planned"), list):
                before = list(attempt["planned"])
                attempt["planned"] = [
                    int(value) for value in before if int(value) not in withheld]
                if before != attempt["planned"]:
                    record_delta(
                        "state.admission.ladder_attempts"
                        f"[{int(attempt.get('ordinal', -1))}].planned",
                        before,
                        attempt["planned"],
                    )
    admission["final_mounts"] = list(kept_mounts)
    if withheld:
        record_delta(
            "state.admission.authoritative_mounts", source_mounts, kept_mounts)
        _assert_registered_planted_deltas(deltas)
    if callable(getattr(arena, "_commit_native_mount", None)):
        arena._commit_native_mount(arena.cur_mounts, arena.cur_mount_n)
    if require_same_process_index:
        observed_index = routing_index_digest(arena)
        if observed_index != routing_index:
            raise SnapshotError(
                "fork hydration changed the full D-LQR routing-index projection")

    return {
        "schema": "grm.det1_4.prefill_fork_restore.v1",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "frame_sha256": identity.get("frame_sha256"),
        "byte_semantics": copy.deepcopy(snapshot.get("byte_semantics")),
        "intervention": (
            "ZERO" if not withheld else "REGISTERED_PLANTED_MISS_WITHHOLDING"),
        "withheld_logical_aliases": sorted(withheld),
        "source_mounts": source_mounts,
        "fork_mounts": kept_mounts,
        "removed_seat_ranges": [
            [int(start), int(end)] for start, end in removed_ranges],
        "prompt_ids": [int(value) for value in prompt_ids],
        "sink_token_ids": [int(value) for value in sink_ids],
        "live_token_ids": [int(value) for value in live_ids],
        "admission_state": admission,
        "admission_projection": {
            "full_plan": copy.deepcopy(admission),
            "authoritative_mounts": list(kept_mounts),
            "mirrors": {
                key: copy.deepcopy(admission.get(key))
                for key in REQUIRED_ADMISSION_FIELDS
            },
        },
        "array_verification": verification,
        "field_deltas": deltas,
        "zero_intervention_no_deltas": not deltas,
        "mask_semantics": (
            "DET1.3 canonical uint8 allowed-equivalent bytes are installed "
            "as a one-shot prefill override; literal device mask storage was "
            "not available in the source format"
        ),
        "routing_index_contract": (
            "same-process lived repository required for detector execution; "
            "snapshot restoration does not reconstruct the full unmounted index"
        ),
        "same_process_index_verified": bool(require_same_process_index),
        "routing_index": (
            routing_index if require_same_process_index else {
                "status": "NOT_VERIFIED_CROSS_PROCESS_ZERO_GATE_ONLY",
            }
        ),
        "source_finalized_at_restore": bool(snapshot.get("capture_finalized")),
        "failure_contract": (
            "any restore exception is process-fatal; a partially hydrated arena "
            "must never continue to a forward or emit gate evidence"
        ),
    }


def verify_fork_masks_consumed(arena: Any) -> dict[str, Any]:
    """Fail closed unless every installed lived prefill mask fired once."""
    rows = []
    for layer_index, layer in enumerate(
        list(getattr(getattr(arena, "m", None), "layers", ()) or ())
    ):
        attention = layer.self_attn
        installed = getattr(attention, "_det1_fork_allowed_mask", None)
        consumed = getattr(attention, "_det1_fork_mask_consumed", False) is True
        passed = installed is None and consumed
        rows.append({
            "layer": int(layer_index),
            "installed_mask_cleared": installed is None,
            "consumed_exactly_once": consumed,
            "status": "PASS" if passed else "FAIL",
        })
    gate_pass = bool(rows) and all(row["status"] == "PASS" for row in rows)
    result = {
        "schema": "grm.det1_4.fork_mask_consumption.v1",
        "status": "PASS" if gate_pass else "FAIL",
        "gate_pass": gate_pass,
        "layer_count": len(rows),
        "rows": rows,
    }
    if not gate_pass:
        raise SnapshotError(
            "fork prefill did not consume every lived allowed-mask exactly once")
    return result


def finalize_snapshot_with_answer(
    path: Path,
    linked_answer: Mapping[str, Any],
) -> Path:
    """Bind post-capture behavior from the same process to a state snapshot."""
    value = load_snapshot(path)
    manifest_path = Path(value.pop("_manifest_path"))
    answer = _jsonable(dict(linked_answer))
    required = (
        "schema", "process_instance_sha256", "captured_attempt_ordinal",
        "attempt_answer", "attempt_mounts", "attempt_answer_correct",
        "attempt_refusal", "probe_answer", "probe_selected_attempt",
        "probe_mounts", "probe_answer_correct", "probe_refusal",
    )
    missing = [field for field in required if answer.get(field) is None]
    if missing:
        raise SnapshotError(f"linked answer is incomplete: {missing}")
    provenance = value.get("provenance") or {}
    if answer["process_instance_sha256"] != provenance.get(
        "process_instance_sha256"
    ):
        raise SnapshotError("linked answer comes from a different process instance")
    if answer["captured_attempt_ordinal"] != provenance.get("attempt_ordinal"):
        raise SnapshotError("linked answer comes from a different ladder attempt")
    existing = value.get("linked_answer")
    if existing is not None and existing != answer:
        raise SnapshotError("snapshot already has a different linked answer")
    value["linked_answer"] = answer
    value["capture_finalized"] = True
    value.pop("manifest_payload_sha256", None)
    value["manifest_payload_sha256"] = _sha256_bytes(_canonical_json(value))
    manifest_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _flatten(value: Any, prefix: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        out = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(value[key], child))
        return out
    if isinstance(value, list):
        out = {}
        for index, item in enumerate(value):
            out.update(_flatten(item, f"{prefix}[{index}]"))
        if not value:
            out[prefix] = []
        return out
    return {prefix: value}


def _array_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record.get(key)
        for key in (
            "source_dtype", "host_dtype", "shape", "order", "bytes", "sha256",
            "literal_device_storage_bytes",
            "source_representation",
        )
    }


def compare_snapshots(
    left_path: Path,
    right_path: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    left = load_snapshot(left_path)
    right = load_snapshot(right_path)
    frame_equal = left["identity"] == right["identity"]
    rows: list[dict[str, Any]] = []

    left_manifest = str(left["_manifest_path"])
    right_manifest = str(right["_manifest_path"])
    left_provenance = dict(left.get("provenance") or {})
    right_provenance = dict(right.get("provenance") or {})
    provenance_errors: list[str] = []

    def provenance_row(
        field: str, lv: Any, rv: Any, status: str, *, detail: str | None = None,
    ) -> None:
        row = {
            "group": "provenance",
            "field": f"provenance.{field}",
            "status": status,
            "left": lv,
            "right": rv,
        }
        if detail is not None:
            row["detail"] = detail
        rows.append(row)

    if left_manifest == right_manifest:
        provenance_errors.append("SELF_COMPARISON_SAME_MANIFEST")
        provenance_row(
            "manifest_path", left_manifest, right_manifest, "INVALID_PROVENANCE",
            detail="lived and replay cannot be the same manifest",
        )
    else:
        provenance_row("manifest_path", left_manifest, right_manifest, "DISTINCT")

    for side, manifest, provenance, expected_arm in (
        ("left", left, left_provenance, "lived"),
        ("right", right, right_provenance, "replay"),
    ):
        missing = [
            field for field in REQUIRED_PROVENANCE_FIELDS
            if provenance.get(field) is None or provenance.get(field) == ""
        ]
        if missing:
            provenance_errors.append(f"{side.upper()}_MISSING_PROVENANCE:{missing}")
        if provenance.get("arm") != expected_arm:
            provenance_errors.append(f"{side.upper()}_ARM_NOT_{expected_arm.upper()}")
        if manifest.get("label") != expected_arm:
            provenance_errors.append(f"{side.upper()}_LABEL_NOT_{expected_arm.upper()}")
        if provenance.get("protocol") != ARM_PROTOCOLS[expected_arm]:
            provenance_errors.append(f"{side.upper()}_PROTOCOL_NOT_REGISTERED")
        if provenance.get("capture_boundary") != CAPTURE_PHASE:
            provenance_errors.append(f"{side.upper()}_BOUNDARY_NOT_REGISTERED")
        try:
            process_digest = process_instance_sha256(
                int(provenance["process_pid"]),
                int(provenance["process_start_ticks"]),
                str(provenance["boot_id"]),
            )
        except (KeyError, TypeError, ValueError):
            process_digest = None
        if process_digest != provenance.get("process_instance_sha256"):
            provenance_errors.append(f"{side.upper()}_PROCESS_DIGEST_INVALID")
        if manifest.get("phase") != CAPTURE_PHASE:
            provenance_errors.append(f"{side.upper()}_PHASE_NOT_REGISTERED")
        if not manifest.get("capture_finalized") or not manifest.get("linked_answer"):
            provenance_errors.append(f"{side.upper()}_ANSWER_NOT_LINKED")
        elif manifest["linked_answer"].get("process_instance_sha256") != provenance.get(
            "process_instance_sha256"
        ):
            provenance_errors.append(f"{side.upper()}_ANSWER_PROCESS_MISMATCH")

    provenance_row(
        "arm",
        left_provenance.get("arm"),
        right_provenance.get("arm"),
        "EXPECTED_ROLE_DIFFERENCE"
        if (
            left_provenance.get("arm") == "lived"
            and right_provenance.get("arm") == "replay"
        ) else "INVALID_PROVENANCE",
    )
    provenance_row(
        "protocol",
        left_provenance.get("protocol"),
        right_provenance.get("protocol"),
        "EXPECTED_ROLE_DIFFERENCE"
        if (
            left_provenance.get("protocol") == ARM_PROTOCOLS["lived"]
            and right_provenance.get("protocol") == ARM_PROTOCOLS["replay"]
        ) else "INVALID_PROVENANCE",
    )
    for field in COMMON_PROVENANCE_FIELDS:
        lv = left_provenance.get(field, "<MISSING>")
        rv = right_provenance.get(field, "<MISSING>")
        status = "EQUAL" if lv == rv and lv != "<MISSING>" else "INVALID_PROVENANCE"
        if status != "EQUAL":
            provenance_errors.append(f"COMMON_FIELD_MISMATCH:{field}")
        provenance_row(field, lv, rv, status)
    left_process = left_provenance.get("process_instance_sha256")
    right_process = right_provenance.get("process_instance_sha256")
    if not left_process or left_process == right_process:
        provenance_errors.append("CAPTURES_NOT_DISTINCT_FRESH_PROCESSES")
        process_status = "INVALID_PROVENANCE"
    else:
        process_status = "DISTINCT"
    provenance_row(
        "process_instance_sha256", left_process, right_process, process_status)
    left_ordinal = left_provenance.get("attempt_ordinal", "<MISSING>")
    right_ordinal = right_provenance.get("attempt_ordinal", "<MISSING>")
    provenance_row(
        "attempt_ordinal",
        left_ordinal,
        right_ordinal,
        "EQUAL" if left_ordinal == right_ordinal else "DIVERGENT",
        detail=(
            "different first executable ladder attempts are a measured state-plan "
            "divergence, not invalid capture provenance"
        ),
    )

    for field, lv, rv in (
        ("metadata.phase", left.get("phase"), right.get("phase")),
        (
            "metadata.byte_semantics",
            left.get("byte_semantics"),
            right.get("byte_semantics"),
        ),
    ):
        rows.append({
            "group": "metadata",
            "field": field,
            "status": "EQUAL" if lv == rv else "DIVERGENT",
            "left": lv,
            "right": rv,
        })

    left_availability = _flatten(
        left.get("availability") or {}, "availability")
    right_availability = _flatten(
        right.get("availability") or {}, "availability")
    for field in sorted(set(left_availability) | set(right_availability)):
        lv = left_availability.get(field, "<MISSING>")
        rv = right_availability.get(field, "<MISSING>")
        if lv == "<MISSING>":
            status = "MISSING_LEFT"
        elif rv == "<MISSING>":
            status = "MISSING_RIGHT"
        else:
            status = "EQUAL" if lv == rv else "DIVERGENT"
        rows.append({
            "group": "availability",
            "field": field,
            "status": status,
            "left": lv,
            "right": rv,
        })

    left_identity = _flatten(left["identity"], "identity")
    right_identity = _flatten(right["identity"], "identity")
    for field in sorted(set(left_identity) | set(right_identity)):
        lv = left_identity.get(field, "<MISSING>")
        rv = right_identity.get(field, "<MISSING>")
        rows.append({
            "group": "identity",
            "field": field,
            "status": "EQUAL" if lv == rv else "DIVERGENT",
            "left": lv,
            "right": rv,
        })

    left_state = _flatten(left.get("state") or {}, "state")
    right_state = _flatten(right.get("state") or {}, "state")
    for field in sorted(set(left_state) | set(right_state)):
        lv = left_state.get(field, "<MISSING>")
        rv = right_state.get(field, "<MISSING>")
        if not frame_equal:
            status = "NOT_COMPARABLE_FRAME"
        elif lv == "<MISSING>":
            status = "MISSING_LEFT"
        elif rv == "<MISSING>":
            status = "MISSING_RIGHT"
        else:
            status = "EQUAL" if lv == rv else "DIVERGENT"
        rows.append({
            "group": "dynamic_scalar",
            "field": field,
            "status": status,
            "left": lv,
            "right": rv,
        })

    left_arrays = left.get("arrays") or {}
    right_arrays = right.get("arrays") or {}
    for field in sorted(set(left_arrays) | set(right_arrays)):
        lrec = left_arrays.get(field)
        rrec = right_arrays.get(field)
        if not frame_equal:
            status = "NOT_COMPARABLE_FRAME"
        elif lrec is None:
            status = "MISSING_LEFT"
        elif rrec is None:
            status = "MISSING_RIGHT"
        else:
            status = (
                "EQUAL"
                if _array_projection(lrec) == _array_projection(rrec)
                else "DIVERGENT"
            )
        rows.append({
            "group": (lrec or rrec or {}).get("group", "array"),
            "field": f"arrays.{field}",
            "status": status,
            "left": None if lrec is None else _array_projection(lrec),
            "right": None if rrec is None else _array_projection(rrec),
        })

    incomplete = not bool(left.get("complete")) or not bool(right.get("complete"))
    divergences = [row for row in rows if row["status"] == "DIVERGENT"]
    missing = [row for row in rows if row["status"].startswith("MISSING_")]
    left_answer = left.get("linked_answer") or {}
    right_answer = right.get("linked_answer") or {}
    linked_behavior_fields = (
        "schema", "captured_attempt_ordinal", "attempt_answer",
        "attempt_mounts", "attempt_answer_correct", "attempt_refusal",
        "probe_answer", "probe_selected_attempt", "probe_mounts",
        "probe_answer_correct", "probe_refusal",
    )
    linked_behavior_equal = True
    for field in linked_behavior_fields:
        lv = left_answer.get(field, "<MISSING>")
        rv = right_answer.get(field, "<MISSING>")
        field_equal = lv == rv and lv != "<MISSING>"
        linked_behavior_equal &= field_equal
        rows.append({
            "group": "linked_behavior",
            "field": f"linked_answer.{field}",
            "status": "EQUAL" if field_equal else "DIVERGENT",
            "left": lv,
            "right": rv,
        })
    if provenance_errors:
        status = "INVALID_PROVENANCE"
    elif not frame_equal:
        status = "INCOMPATIBLE_FRAME"
    elif incomplete:
        status = "INCOMPLETE_SNAPSHOT"
    elif divergences or missing:
        status = "DIVERGENT"
    elif not linked_behavior_equal:
        status = "STATE_EQUAL_BEHAVIOR_DIVERGENT"
    else:
        status = "PASS_STATE_BYTES_EQUAL_PENDING_OBSERVER_CONTROL"
    state_gate_pass = status == "PASS_STATE_BYTES_EQUAL_PENDING_OBSERVER_CONTROL"
    comparison = {
        "schema": COMPARISON_SCHEMA,
        "status": status,
        "state_gate_pass": state_gate_pass,
        "gate_pass": False,
        "race_resume_authorized": False,
        "frame_compatible": bool(frame_equal),
        "left": {
            "manifest": left["_manifest_path"],
            "manifest_file": {
                "bytes": Path(left["_manifest_path"]).stat().st_size,
                "sha256": _sha256_file(Path(left["_manifest_path"])),
            },
            "label": left.get("label"),
            "complete": bool(left.get("complete")),
            "frame_sha256": left["identity"].get("frame_sha256"),
            "availability": left.get("availability"),
            "provenance": left_provenance,
            "linked_answer": left.get("linked_answer"),
        },
        "right": {
            "manifest": right["_manifest_path"],
            "manifest_file": {
                "bytes": Path(right["_manifest_path"]).stat().st_size,
                "sha256": _sha256_file(Path(right["_manifest_path"])),
            },
            "label": right.get("label"),
            "complete": bool(right.get("complete")),
            "frame_sha256": right["identity"].get("frame_sha256"),
            "availability": right.get("availability"),
            "provenance": right_provenance,
            "linked_answer": right.get("linked_answer"),
        },
        "counts": {
            key: sum(row["status"] == key for row in rows)
            for key in (
                "EQUAL", "DIVERGENT", "MISSING_LEFT", "MISSING_RIGHT",
                "NOT_COMPARABLE_FRAME",
            )
        },
        "rows": rows,
        "provenance_errors": provenance_errors,
        "linked_attempt_behavior_equal": linked_behavior_equal,
        "law": (
            "replay may serve as a counterfactual only when both complete "
            "same-frame snapshots have equal canonical bytes and independent "
            "bare-answer controls prove that capture did not move behavior"
        ),
    }
    if output_path is not None:
        _write_json_exclusive_or_verify(Path(output_path), comparison)
    return comparison


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    compare = sub.add_parser("compare", help="compare two completed snapshots")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "compare":
        value = compare_snapshots(args.left, args.right, output_path=args.output)
        print(json.dumps({
            "status": value["status"],
            "gate_pass": value["gate_pass"],
            "counts": value["counts"],
        }, sort_keys=True))
        return 0 if value["gate_pass"] else 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
