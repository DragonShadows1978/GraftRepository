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
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import hashlib
import json
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
        builder.array(
            f"mask.layer_{layer_index:02d}.allowed",
            _causal_allowed_mask(int(prompt.size), total, window),
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
) -> Iterator[dict[str, Any]]:
    """Capture the next prefill, then continue that exact process to its answer."""
    original = arena._forward
    result: dict[str, Any] = {"manifest_path": None, "capture_count": 0}

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
    for field, record in (value.get("arrays") or {}).items():
        blob = manifest_path.parent / str(record.get("blob"))
        if not blob.is_file():
            raise SnapshotError(f"snapshot blob is missing for {field}: {blob}")
        if blob.stat().st_size != int(record.get("bytes", -1)):
            raise SnapshotError(f"snapshot blob size mismatch for {field}")
        if _sha256_file(blob) != record.get("sha256"):
            raise SnapshotError(f"snapshot blob digest mismatch for {field}")
    value["_manifest_path"] = str(manifest_path.resolve())
    return value


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
