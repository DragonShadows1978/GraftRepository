"""ctypes wrapper for the dependency-free GRM C++ host runtime ABI."""

from dataclasses import dataclass
import ctypes
import json
import os
import numpy as np


class _StatsC(ctypes.Structure):
    _fields_ = [
        ("nodes", ctypes.c_uint64),
        ("dirty_nodes", ctypes.c_uint64),
        ("durable_nodes", ctypes.c_uint64),
        ("host_payload_bytes", ctypes.c_uint64),
        ("host_payload_tensors", ctypes.c_uint64),
        ("route_entries", ctypes.c_uint64),
    ]


class _DirtyNodeC(ctypes.Structure):
    _fields_ = [
        ("node_id", ctypes.c_uint64),
        ("payload_dirty", ctypes.c_int),
        ("metadata_dirty", ctypes.c_int),
        ("payload_bytes", ctypes.c_uint64),
        ("durability_priority", ctypes.c_uint64),
    ]


class _PayloadStatsC(ctypes.Structure):
    _fields_ = [
        ("tensor_count", ctypes.c_uint64),
        ("payload_bytes", ctypes.c_uint64),
    ]


class _TensorInfoC(ctypes.Structure):
    _fields_ = [
        ("rank", ctypes.c_uint64),
        ("payload_bytes", ctypes.c_uint64),
    ]


class _GraphEdgesInfoC(ctypes.Structure):
    _fields_ = [
        ("source_turns", ctypes.c_uint64),
        ("source_grafts", ctypes.c_uint64),
        ("supersedes", ctypes.c_uint64),
        ("superseded_by", ctypes.c_uint64),
    ]


class _ArenaSwapPlanC(ctypes.Structure):
    _fields_ = [
        ("sink_tokens", ctypes.c_uint64),
        ("arena_width", ctypes.c_uint64),
        ("old_mount_tokens", ctypes.c_uint64),
        ("new_mount_tokens", ctypes.c_uint64),
        ("old_mount_end", ctypes.c_uint64),
        ("live_tail_start", ctypes.c_uint64),
        ("live_tail_tokens", ctypes.c_uint64),
        ("input_cache_tokens", ctypes.c_uint64),
        ("output_cache_tokens", ctypes.c_uint64),
        ("overflow", ctypes.c_int),
    ]


class _ArenaEvictPlanC(ctypes.Structure):
    _fields_ = [
        ("sink_tokens", ctypes.c_uint64),
        ("arena_width", ctypes.c_uint64),
        ("mount_tokens", ctypes.c_uint64),
        ("head_tokens", ctypes.c_uint64),
        ("drop_tokens", ctypes.c_uint64),
        ("input_cache_tokens", ctypes.c_uint64),
        ("output_cache_tokens", ctypes.c_uint64),
        ("underflow", ctypes.c_int),
    ]


@dataclass(frozen=True)
class NativeStats:
    nodes: int
    dirty_nodes: int
    durable_nodes: int
    host_payload_bytes: int
    host_payload_tensors: int
    route_entries: int


@dataclass(frozen=True)
class DirtyNode:
    node_id: int
    payload_dirty: bool
    metadata_dirty: bool
    payload_bytes: int
    durability_priority: int


@dataclass(frozen=True)
class PayloadStats:
    tensor_count: int
    payload_bytes: int


@dataclass(frozen=True)
class TensorInfo:
    shape: tuple
    payload_bytes: int
    dtype: str


@dataclass(frozen=True)
class GraphEdges:
    source_turns: tuple
    source_grafts: tuple
    supersedes: tuple
    superseded_by: tuple


@dataclass(frozen=True)
class ArenaSwapPlan:
    sink_tokens: int
    arena_width: int
    old_mount_tokens: int
    new_mount_tokens: int
    old_mount_end: int
    live_tail_start: int
    live_tail_tokens: int
    input_cache_tokens: int
    output_cache_tokens: int
    overflow: bool


@dataclass(frozen=True)
class ArenaEvictPlan:
    sink_tokens: int
    arena_width: int
    mount_tokens: int
    head_tokens: int
    drop_tokens: int
    input_cache_tokens: int
    output_cache_tokens: int
    underflow: bool


def _default_lib_path():
    return os.environ.get("GRM_RUNTIME_LIB", "libgrm_runtime.so")


class NativeGraftStore:
    supports_multi_route_keys = True

    def __init__(self, lib_path=None, *, model_type="model",
                 num_layers=0, hidden_dim=0, vals_per_tok_layer=0,
                 route_layer=0, payload_kind="mla", latent_rank=0,
                 rope_dim=0, num_kv_heads=0, head_dim=0,
                 position_law=None, state_kind=None, graftability=None,
                 remountable=None, composition=None):
        self._lib = ctypes.CDLL(lib_path or _default_lib_path())
        self._bind()
        payload_kind = str(payload_kind).lower()
        defaults = self._profile_defaults(payload_kind)
        position_law = position_law or defaults["position_law"]
        state_kind = state_kind or defaults["state_kind"]
        graftability = graftability or defaults["graftability"]
        remountable = (
            defaults["remountable"] if remountable is None
            else bool(remountable))
        composition = composition or defaults["composition"]
        self._validate_profile(position_law, graftability, remountable)
        profile_args = (
            str(position_law).encode("utf-8"),
            str(state_kind).encode("utf-8"),
            str(graftability).encode("utf-8"),
            int(remountable),
            str(composition).encode("utf-8"),
        )
        if payload_kind == "mla":
            if hasattr(self._lib, "grm_store_create_mla_profile"):
                self._handle = self._lib.grm_store_create_mla_profile(
                    model_type.encode("utf-8"), int(num_layers),
                    int(hidden_dim), int(vals_per_tok_layer), int(route_layer),
                    int(latent_rank), int(rope_dim), *profile_args)
            else:
                self._handle = self._lib.grm_store_create_mla(
                    model_type.encode("utf-8"), int(num_layers),
                    int(hidden_dim), int(vals_per_tok_layer), int(route_layer),
                    int(latent_rank), int(rope_dim))
        elif payload_kind == "gqa":
            if not hasattr(self._lib, "grm_store_create_gqa"):
                raise RuntimeError("native GRM library lacks GQA store ABI")
            if hasattr(self._lib, "grm_store_create_gqa_profile"):
                self._handle = self._lib.grm_store_create_gqa_profile(
                    model_type.encode("utf-8"), int(num_layers),
                    int(hidden_dim), int(vals_per_tok_layer), int(route_layer),
                    int(num_kv_heads), int(head_dim), *profile_args)
            else:
                self._handle = self._lib.grm_store_create_gqa(
                    model_type.encode("utf-8"), int(num_layers),
                    int(hidden_dim), int(vals_per_tok_layer), int(route_layer),
                    int(num_kv_heads), int(head_dim))
        else:
            raise ValueError(f"unsupported native GRM payload kind: {payload_kind}")
        if not self._handle:
            raise RuntimeError("failed to create native GRM store")

    @staticmethod
    def _profile_defaults(payload_kind):
        if payload_kind == "mla":
            return {
                "position_law": "rope_partial_mla",
                "state_kind": "mla_latent_plus_rope",
                "graftability": "seat_remountable",
                "remountable": True,
                "composition": "multi_mount",
            }
        if payload_kind == "gqa":
            return {
                "position_law": "rope_full_kv",
                "state_kind": "kv",
                "graftability": "seat_remountable",
                "remountable": True,
                "composition": "multi_mount",
            }
        raise ValueError(f"unsupported native GRM payload kind: {payload_kind}")

    @staticmethod
    def _validate_profile(position_law, graftability, remountable):
        if not remountable:
            return
        law = str(position_law or "").lower()
        graftability = str(graftability or "").lower()
        fixed = ("absolute" in law or "fixed" in law or
                 "absolute" in graftability or "fixed" in graftability)
        if fixed:
            raise ValueError(
                "fixed-position GRM profiles must set remountable=False")
        if not any(token in law for token in ("rope", "rotary", "relative")):
            raise ValueError(
                "remountable GRM profiles require a RoPE or relative "
                "position law")

    def _bind(self):
        lib = self._lib
        lib.grm_store_create_mla.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.grm_store_create_mla.restype = ctypes.c_void_p
        if hasattr(lib, "grm_store_create_mla_profile"):
            lib.grm_store_create_mla_profile.argtypes = [
                ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                ctypes.c_int, ctypes.c_char_p]
            lib.grm_store_create_mla_profile.restype = ctypes.c_void_p
        if hasattr(lib, "grm_store_create_gqa"):
            lib.grm_store_create_gqa.argtypes = [
                ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int]
            lib.grm_store_create_gqa.restype = ctypes.c_void_p
        if hasattr(lib, "grm_store_create_gqa_profile"):
            lib.grm_store_create_gqa_profile.argtypes = [
                ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                ctypes.c_int, ctypes.c_char_p]
            lib.grm_store_create_gqa_profile.restype = ctypes.c_void_p
        lib.grm_store_destroy.argtypes = [ctypes.c_void_p]
        lib.grm_store_destroy.restype = None
        lib.grm_store_dialect_id.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        lib.grm_store_dialect_id.restype = ctypes.c_int
        self._has_dialect_profile = hasattr(lib, "grm_store_dialect_profile")
        if self._has_dialect_profile:
            lib.grm_store_dialect_profile.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
            lib.grm_store_dialect_profile.restype = ctypes.c_int
        lib.grm_store_add_node.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64)]
        lib.grm_store_add_node.restype = ctypes.c_int
        lib.grm_store_set_tensor.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64]
        lib.grm_store_set_tensor.restype = ctypes.c_int
        lib.grm_store_payload_stats.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(_PayloadStatsC)]
        lib.grm_store_payload_stats.restype = ctypes.c_int
        self._has_clear_payload = hasattr(lib, "grm_store_clear_payload")
        if self._has_clear_payload:
            lib.grm_store_clear_payload.argtypes = [
                ctypes.c_void_p, ctypes.c_uint64]
            lib.grm_store_clear_payload.restype = ctypes.c_int
        lib.grm_store_tensor_info.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64,
            ctypes.c_char_p, ctypes.c_size_t,
            ctypes.POINTER(_TensorInfoC)]
        lib.grm_store_tensor_info.restype = ctypes.c_int
        lib.grm_store_read_tensor.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64)]
        lib.grm_store_read_tensor.restype = ctypes.c_int
        self._has_slice_tensor = hasattr(lib, "grm_store_slice_tensor")
        if self._has_slice_tensor:
            lib.grm_store_slice_tensor.argtypes = [
                ctypes.c_void_p, ctypes.c_uint64, ctypes.c_char_p,
                ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64,
                ctypes.c_char_p, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64,
                ctypes.POINTER(_TensorInfoC)]
            lib.grm_store_slice_tensor.restype = ctypes.c_int
        lib.grm_store_set_metadata_json.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_char_p]
        lib.grm_store_set_metadata_json.restype = ctypes.c_int
        self._has_set_active = hasattr(lib, "grm_store_set_active")
        if self._has_set_active:
            lib.grm_store_set_active.argtypes = [
                ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
            lib.grm_store_set_active.restype = ctypes.c_int
        self._has_route_metadata = hasattr(lib, "grm_store_set_route_metadata")
        if self._has_route_metadata:
            lib.grm_store_set_route_metadata.argtypes = [
                ctypes.c_void_p, ctypes.c_uint64, ctypes.c_char_p,
                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
            lib.grm_store_set_route_metadata.restype = ctypes.c_int
        self._has_fact_identity = (
            hasattr(lib, "grm_store_set_fact_identity") and
            hasattr(lib, "grm_store_fact_matches"))
        if self._has_fact_identity:
            lib.grm_store_set_fact_identity.argtypes = [
                ctypes.c_void_p, ctypes.c_uint64, ctypes.c_char_p,
                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                ctypes.c_char_p, ctypes.c_char_p]
            lib.grm_store_set_fact_identity.restype = ctypes.c_int
            lib.grm_store_fact_matches.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint64,
                ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64,
                ctypes.POINTER(ctypes.c_uint64)]
            lib.grm_store_fact_matches.restype = ctypes.c_int
            self._has_fact_matches_ex = hasattr(
                lib, "grm_store_fact_matches_ex")
            if self._has_fact_matches_ex:
                lib.grm_store_fact_matches_ex.argtypes = [
                    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
                    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                    ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64,
                    ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64,
                    ctypes.POINTER(ctypes.c_uint64)]
                lib.grm_store_fact_matches_ex.restype = ctypes.c_int
        self._has_filter_active_nodes = hasattr(
            lib, "grm_store_filter_active_nodes")
        if self._has_filter_active_nodes:
            u64p = ctypes.POINTER(ctypes.c_uint64)
            lib.grm_store_filter_active_nodes.argtypes = [
                ctypes.c_void_p, u64p, ctypes.c_uint64, u64p,
                ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
            lib.grm_store_filter_active_nodes.restype = ctypes.c_int
        self._has_active_text_matches = hasattr(
            lib, "grm_store_active_text_matches")
        if self._has_active_text_matches:
            lib.grm_store_active_text_matches.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64,
                ctypes.POINTER(ctypes.c_uint64)]
            lib.grm_store_active_text_matches.restype = ctypes.c_int
        self._has_graph_edges = (
            hasattr(lib, "grm_store_set_graph_edges") and
            hasattr(lib, "grm_store_graph_edges_info") and
            hasattr(lib, "grm_store_read_graph_edges"))
        if self._has_graph_edges:
            u64p = ctypes.POINTER(ctypes.c_uint64)
            lib.grm_store_set_graph_edges.argtypes = [
                ctypes.c_void_p, ctypes.c_uint64,
                u64p, ctypes.c_uint64, u64p, ctypes.c_uint64,
                u64p, ctypes.c_uint64, u64p, ctypes.c_uint64]
            lib.grm_store_set_graph_edges.restype = ctypes.c_int
            lib.grm_store_graph_edges_info.argtypes = [
                ctypes.c_void_p, ctypes.c_uint64,
                ctypes.POINTER(_GraphEdgesInfoC)]
            lib.grm_store_graph_edges_info.restype = ctypes.c_int
            lib.grm_store_read_graph_edges.argtypes = [
                ctypes.c_void_p, ctypes.c_uint64,
                u64p, ctypes.c_uint64, u64p, ctypes.c_uint64,
                u64p, ctypes.c_uint64, u64p, ctypes.c_uint64,
                ctypes.POINTER(_GraphEdgesInfoC)]
            lib.grm_store_read_graph_edges.restype = ctypes.c_int
        self._has_source_closure = hasattr(lib, "grm_store_source_closure")
        if self._has_source_closure:
            u64p = ctypes.POINTER(ctypes.c_uint64)
            lib.grm_store_source_closure.argtypes = [
                ctypes.c_void_p, u64p, ctypes.c_uint64, ctypes.c_uint64,
                ctypes.c_int, u64p, ctypes.c_uint64,
                ctypes.POINTER(ctypes.c_uint64)]
            lib.grm_store_source_closure.restype = ctypes.c_int
        self._has_apply_revision = hasattr(lib, "grm_store_apply_revision")
        if self._has_apply_revision:
            lib.grm_store_apply_revision.argtypes = [
                ctypes.c_void_p, ctypes.c_uint64,
                ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64]
            lib.grm_store_apply_revision.restype = ctypes.c_int
        self._has_apply_expire = hasattr(lib, "grm_store_apply_expire")
        if self._has_apply_expire:
            lib.grm_store_apply_expire.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_uint64]
            lib.grm_store_apply_expire.restype = ctypes.c_int
        lib.grm_store_metadata_json.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_char_p,
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)]
        lib.grm_store_metadata_json.restype = ctypes.c_int
        self._has_parse_memory_command = hasattr(
            lib, "grm_store_parse_memory_command")
        if self._has_parse_memory_command:
            lib.grm_store_parse_memory_command.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
                ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)]
            lib.grm_store_parse_memory_command.restype = ctypes.c_int
        self._has_extraction_policy_plan = hasattr(
            lib, "grm_store_plan_extraction_policy")
        if self._has_extraction_policy_plan:
            lib.grm_store_plan_extraction_policy.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
                ctypes.c_double, ctypes.c_double, ctypes.c_uint64,
                ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint64)]
            lib.grm_store_plan_extraction_policy.restype = ctypes.c_int
        self._has_reinforcement_plan = hasattr(
            lib, "grm_store_plan_reinforcement")
        if self._has_reinforcement_plan:
            lib.grm_store_plan_reinforcement.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
                ctypes.c_double, ctypes.c_double, ctypes.c_uint64,
                ctypes.c_char_p, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint64)]
            lib.grm_store_plan_reinforcement.restype = ctypes.c_int
        self._has_review_transition_plan = hasattr(
            lib, "grm_store_plan_review_transition")
        if self._has_review_transition_plan:
            lib.grm_store_plan_review_transition.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
                ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint64)]
            lib.grm_store_plan_review_transition.restype = ctypes.c_int
        self._has_cull_span_plan = hasattr(lib, "grm_store_plan_cull_spans")
        if self._has_cull_span_plan:
            lib.grm_store_plan_cull_spans.argtypes = [
                ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
                ctypes.c_int, ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64,
                ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint64)]
            lib.grm_store_plan_cull_spans.restype = ctypes.c_int
        lib.grm_store_set_route.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint64, ctypes.c_char_p]
        lib.grm_store_set_route.restype = ctypes.c_int
        lib.grm_store_set_route_multi.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_char_p]
        lib.grm_store_set_route_multi.restype = ctypes.c_int
        self._has_route_list = hasattr(lib, "grm_store_set_route_list")
        if self._has_route_list:
            lib.grm_store_set_route_list.argtypes = [
                ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_float),
                ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_uint64, ctypes.c_char_p]
            lib.grm_store_set_route_list.restype = ctypes.c_int
        lib.grm_store_route.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint64,
            ctypes.c_char_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
        lib.grm_store_route.restype = ctypes.c_int
        self._has_route_filtered = hasattr(lib, "grm_store_route_filtered")
        if self._has_route_filtered:
            lib.grm_store_route_filtered.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                ctypes.c_uint64, ctypes.c_char_p, ctypes.c_char_p,
                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
            lib.grm_store_route_filtered.restype = ctypes.c_int
        self._has_route_gqa = hasattr(lib, "grm_store_route_gqa")
        if self._has_route_gqa:
            lib.grm_store_route_gqa.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint64,
                ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64,
                ctypes.POINTER(ctypes.c_uint64)]
            lib.grm_store_route_gqa.restype = ctypes.c_int
        lib.grm_store_configure_arena.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
        lib.grm_store_configure_arena.restype = ctypes.c_int
        lib.grm_store_plan_swap.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.POINTER(_ArenaSwapPlanC)]
        lib.grm_store_plan_swap.restype = ctypes.c_int
        lib.grm_store_plan_evict.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.POINTER(_ArenaEvictPlanC)]
        lib.grm_store_plan_evict.restype = ctypes.c_int
        lib.grm_store_apply_swap_tensor.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64,
            ctypes.c_uint64, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(_ArenaSwapPlanC)]
        lib.grm_store_apply_swap_tensor.restype = ctypes.c_int
        lib.grm_store_apply_evict_tensor.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64,
            ctypes.c_uint64, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(_ArenaEvictPlanC)]
        lib.grm_store_apply_evict_tensor.restype = ctypes.c_int
        lib.grm_store_commit_mount.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint64, ctypes.c_uint64]
        lib.grm_store_commit_mount.restype = ctypes.c_int
        lib.grm_store_save_checkpoint.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.grm_store_save_checkpoint.restype = ctypes.c_int
        lib.grm_store_load_checkpoint.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.grm_store_load_checkpoint.restype = ctypes.c_int
        lib.grm_store_mark_durable.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.grm_store_mark_durable.restype = ctypes.c_int
        lib.grm_store_evict_device_copy.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64]
        lib.grm_store_evict_device_copy.restype = ctypes.c_int
        self._has_dirty_nodes = hasattr(lib, "grm_store_dirty_nodes")
        if self._has_dirty_nodes:
            lib.grm_store_dirty_nodes.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
            lib.grm_store_dirty_nodes.restype = ctypes.c_int
        self._has_dirty_plan = hasattr(lib, "grm_store_dirty_plan")
        if self._has_dirty_plan:
            lib.grm_store_dirty_plan.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(_DirtyNodeC),
                ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
            lib.grm_store_dirty_plan.restype = ctypes.c_int
        lib.grm_store_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(_StatsC)]
        lib.grm_store_stats.restype = ctypes.c_int
        lib.grm_store_last_error.argtypes = [ctypes.c_void_p]
        lib.grm_store_last_error.restype = ctypes.c_char_p

    def close(self):
        if getattr(self, "_handle", None):
            self._lib.grm_store_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()

    def _check(self, code):
        if code != 0:
            err = self._lib.grm_store_last_error(self._handle)
            raise RuntimeError((err or b"native GRM error").decode("utf-8"))

    def dialect_id(self):
        buf = ctypes.create_string_buffer(256)
        self._check(self._lib.grm_store_dialect_id(
            self._handle, buf, ctypes.sizeof(buf)))
        return buf.value.decode("utf-8")

    def dialect_profile(self):
        if not self._has_dialect_profile:
            return None
        buf = ctypes.create_string_buffer(512)
        self._check(self._lib.grm_store_dialect_profile(
            self._handle, buf, ctypes.sizeof(buf)))
        return buf.value.decode("utf-8")

    def add_node(self, text, payload, ntok=0):
        data = bytes(payload)
        arr = None
        if data:
            arr_t = ctypes.c_uint8 * len(data)
            arr = arr_t.from_buffer_copy(data)
        out = ctypes.c_uint64()
        self._check(self._lib.grm_store_add_node(
            self._handle, text.encode("utf-8"), int(ntok), arr, len(data),
            ctypes.byref(out)))
        return int(out.value)

    @staticmethod
    def _byte_array(data):
        b = bytes(data)
        if not b:
            return None, 0
        arr_t = ctypes.c_uint8 * len(b)
        return arr_t.from_buffer_copy(b), len(b)

    @staticmethod
    def _shape_array(shape):
        dims = [int(d) for d in shape]
        if not dims:
            return None, 0
        arr_t = ctypes.c_uint64 * len(dims)
        return arr_t(*dims), len(dims)

    @staticmethod
    def _u64_array(values):
        vals = [int(v) for v in (values or ())]
        if not vals:
            return None, 0
        arr_t = ctypes.c_uint64 * len(vals)
        return arr_t(*vals), len(vals)

    def set_tensor(self, node_id, name, array):
        arr_np = np.ascontiguousarray(array)
        shape, rank = self._shape_array(arr_np.shape)
        data, nbytes = self._byte_array(arr_np.tobytes())
        self._check(self._lib.grm_store_set_tensor(
            self._handle, int(node_id), str(name).encode("utf-8"),
            str(arr_np.dtype).encode("utf-8"), shape, rank, data, nbytes))

    def payload_stats(self, node_id):
        out = _PayloadStatsC()
        self._check(self._lib.grm_store_payload_stats(
            self._handle, int(node_id), ctypes.byref(out)))
        return PayloadStats(int(out.tensor_count), int(out.payload_bytes))

    def tensor_info(self, node_id, name, max_rank=8):
        cap = max(0, int(max_rank))
        shape = None
        if cap:
            shape_t = ctypes.c_uint64 * cap
            shape = shape_t()
        dtype = ctypes.create_string_buffer(64)
        out = _TensorInfoC()
        self._check(self._lib.grm_store_tensor_info(
            self._handle, int(node_id), str(name).encode("utf-8"),
            shape, cap, dtype, ctypes.sizeof(dtype), ctypes.byref(out)))
        n = min(int(out.rank), cap)
        dims = tuple(int(shape[i]) for i in range(n)) if shape is not None else ()
        if int(out.rank) > cap:
            raise RuntimeError("tensor rank exceeded local shape buffer")
        return TensorInfo(dims, int(out.payload_bytes),
                          dtype.value.decode("utf-8"))

    def get_tensor(self, node_id, name):
        info = self.tensor_info(node_id, name)
        out = None
        if info.payload_bytes:
            out_t = ctypes.c_uint8 * info.payload_bytes
            out = out_t()
        count = ctypes.c_uint64()
        self._check(self._lib.grm_store_read_tensor(
            self._handle, int(node_id), str(name).encode("utf-8"),
            out, info.payload_bytes, ctypes.byref(count)))
        data = bytes(out[:int(count.value)]) if out is not None else b""
        arr = np.frombuffer(data, dtype=np.dtype(info.dtype)).copy()
        return arr.reshape(info.shape)

    def slice_tensor(self, node_id, name, axis, start, length, max_rank=8):
        if not getattr(self, "_has_slice_tensor", False):
            raise RuntimeError("native GRM slice_tensor is unavailable")
        info = self.tensor_info(node_id, name, max_rank=max_rank)
        seq = int(axis)
        if seq < 0:
            seq += len(info.shape)
        if seq < 0 or seq >= len(info.shape):
            raise ValueError("slice axis out of range")
        start = int(start)
        length = int(length)
        if start < 0 or length < 0:
            raise ValueError("slice start and length must be nonnegative")
        out_shape_py = list(info.shape)
        out_shape_py[seq] = length
        out_elements = int(np.prod(out_shape_py, dtype=np.uint64))
        out_nbytes = out_elements * np.dtype(info.dtype).itemsize

        shape_t = ctypes.c_uint64 * len(out_shape_py)
        shape = shape_t()
        dtype = ctypes.create_string_buffer(64)
        out = None
        if out_nbytes:
            out_t = ctypes.c_uint8 * out_nbytes
            out = out_t()
        result = _TensorInfoC()
        self._check(self._lib.grm_store_slice_tensor(
            self._handle, int(node_id), str(name).encode("utf-8"),
            seq, start, length, shape, len(out_shape_py),
            dtype, ctypes.sizeof(dtype), out, out_nbytes,
            ctypes.byref(result)))
        if int(result.payload_bytes) != out_nbytes:
            raise RuntimeError("native slice output length mismatch")
        dims = tuple(int(shape[i]) for i in range(int(result.rank)))
        data = bytes(out[:out_nbytes]) if out is not None else b""
        arr = np.frombuffer(
            data, dtype=np.dtype(dtype.value.decode("utf-8"))).copy()
        return arr.reshape(dims)

    def set_metadata(self, node_id, metadata):
        metadata = metadata or {}
        data = json.dumps(metadata, sort_keys=True).encode("utf-8")
        self._check(self._lib.grm_store_set_metadata_json(
            self._handle, int(node_id), data))
        if "active" in metadata:
            self.set_active(node_id, bool(metadata.get("active", True)))
        self.set_route_metadata(
            node_id,
            kind=metadata.get("kind"),
            scope=metadata.get("scope"),
            durability=metadata.get("durability"),
            mutability=metadata.get("mutability"))
        self.set_graph_edges(
            node_id,
            source_turns=metadata.get("source_turns", ()),
            source_grafts=metadata.get("source_grafts", ()),
            supersedes=metadata.get("supersedes", ()),
            superseded_by=metadata.get("superseded_by", ()))
        self.set_fact_identity(
            node_id,
            subject=metadata.get("subject"),
            predicate=metadata.get("predicate"),
            value=metadata.get("value"),
            scope=metadata.get("scope"),
            valid_from=metadata.get("valid_from"),
            expires_at=metadata.get("expires_at"))

    def set_active(self, node_id, active=True):
        if not getattr(self, "_has_set_active", False):
            return
        self._check(self._lib.grm_store_set_active(
            self._handle, int(node_id), 1 if active else 0))

    def set_route_metadata(self, node_id, *, kind=None, scope=None,
                           durability=None, mutability=None):
        if not getattr(self, "_has_route_metadata", False):
            return
        self._check(self._lib.grm_store_set_route_metadata(
            self._handle, int(node_id),
            ("" if kind is None else str(kind)).encode("utf-8"),
            ("" if scope is None else str(scope)).encode("utf-8"),
            ("" if durability is None else str(durability)).encode("utf-8"),
            ("" if mutability is None else str(mutability)).encode("utf-8")))

    def set_fact_identity(self, node_id, *, subject=None, predicate=None,
                          value=None, scope=None, valid_from=None,
                          expires_at=None):
        if not getattr(self, "_has_fact_identity", False):
            return
        self._check(self._lib.grm_store_set_fact_identity(
            self._handle, int(node_id),
            ("" if subject is None else str(subject)).encode("utf-8"),
            ("" if predicate is None else str(predicate)).encode("utf-8"),
            ("" if value is None else str(value)).encode("utf-8"),
            ("project" if scope is None else str(scope)).encode("utf-8"),
            ("" if valid_from is None else str(valid_from)).encode("utf-8"),
            ("" if expires_at is None else str(expires_at)).encode("utf-8")))

    def fact_matches(self, *, subject=None, predicate=None, value=None,
                     scope=None, value_mode=0, valid_from=None,
                     expires_at=None, temporal_mode=0):
        if not getattr(self, "_has_fact_identity", False):
            raise RuntimeError("native GRM fact identity scan is unavailable")
        use_ex = (int(temporal_mode) != 0 or valid_from is not None
                  or expires_at is not None)
        if use_ex and not getattr(self, "_has_fact_matches_ex", False):
            raise RuntimeError("native GRM extended fact scan is unavailable")
        needed = ctypes.c_uint64()
        args = (
            ("" if subject is None else str(subject)).encode("utf-8"),
            ("" if predicate is None else str(predicate)).encode("utf-8"),
            ("" if value is None else str(value)).encode("utf-8"),
            ("project" if scope is None else str(scope)).encode("utf-8"))
        if use_ex:
            ex_args = args + (
                ("" if valid_from is None else str(valid_from)).encode("utf-8"),
                ("" if expires_at is None else str(expires_at)).encode("utf-8"),
                int(value_mode), int(temporal_mode))
            self._check(self._lib.grm_store_fact_matches_ex(
                self._handle, *ex_args, None, 0, ctypes.byref(needed)))
        else:
            self._check(self._lib.grm_store_fact_matches(
                self._handle, *args, int(value_mode), None, 0,
                ctypes.byref(needed)))
        if not int(needed.value):
            return ()
        out_t = ctypes.c_uint64 * int(needed.value)
        out = out_t()
        got = ctypes.c_uint64()
        if use_ex:
            self._check(self._lib.grm_store_fact_matches_ex(
                self._handle, *ex_args, out, int(needed.value),
                ctypes.byref(got)))
        else:
            self._check(self._lib.grm_store_fact_matches(
                self._handle, *args, int(value_mode), out, int(needed.value),
                ctypes.byref(got)))
        return tuple(int(out[i]) for i in range(int(got.value)))

    def filter_active_nodes(self, node_ids):
        if not getattr(self, "_has_filter_active_nodes", False):
            raise RuntimeError("native GRM active-node filter is unavailable")
        requested, requested_n = self._u64_array(node_ids)
        needed = ctypes.c_uint64()
        self._check(self._lib.grm_store_filter_active_nodes(
            self._handle, requested, requested_n, None, 0,
            ctypes.byref(needed)))
        if not int(needed.value):
            return ()
        out_t = ctypes.c_uint64 * int(needed.value)
        out = out_t()
        got = ctypes.c_uint64()
        self._check(self._lib.grm_store_filter_active_nodes(
            self._handle, requested, requested_n, out, int(needed.value),
            ctypes.byref(got)))
        return tuple(int(out[i]) for i in range(int(got.value)))

    def active_text_matches(self, query):
        if not getattr(self, "_has_active_text_matches", False):
            raise RuntimeError("native GRM active text scan is unavailable")
        data = str(query or "").encode("utf-8")
        needed = ctypes.c_uint64()
        self._check(self._lib.grm_store_active_text_matches(
            self._handle, data, None, 0, ctypes.byref(needed)))
        if not int(needed.value):
            return ()
        out_t = ctypes.c_uint64 * int(needed.value)
        out = out_t()
        got = ctypes.c_uint64()
        self._check(self._lib.grm_store_active_text_matches(
            self._handle, data, out, int(needed.value), ctypes.byref(got)))
        return tuple(int(out[i]) for i in range(int(got.value)))

    def set_graph_edges(self, node_id, *, source_turns=(),
                        source_grafts=(), supersedes=(),
                        superseded_by=()):
        if not getattr(self, "_has_graph_edges", False):
            return
        st, st_n = self._u64_array(source_turns)
        sg, sg_n = self._u64_array(source_grafts)
        sp, sp_n = self._u64_array(supersedes)
        sb, sb_n = self._u64_array(superseded_by)
        self._check(self._lib.grm_store_set_graph_edges(
            self._handle, int(node_id), st, st_n, sg, sg_n,
            sp, sp_n, sb, sb_n))

    def graph_edges(self, node_id):
        if not getattr(self, "_has_graph_edges", False):
            raise RuntimeError("native GRM graph edges are unavailable")
        info = _GraphEdgesInfoC()
        self._check(self._lib.grm_store_graph_edges_info(
            self._handle, int(node_id), ctypes.byref(info)))

        def alloc(n):
            if not n:
                return None
            arr_t = ctypes.c_uint64 * int(n)
            return arr_t()

        st = alloc(info.source_turns)
        sg = alloc(info.source_grafts)
        sp = alloc(info.supersedes)
        sb = alloc(info.superseded_by)
        counts = _GraphEdgesInfoC()
        self._check(self._lib.grm_store_read_graph_edges(
            self._handle, int(node_id),
            st, int(info.source_turns),
            sg, int(info.source_grafts),
            sp, int(info.supersedes),
            sb, int(info.superseded_by),
            ctypes.byref(counts)))

        def tup(arr, n):
            return tuple(int(arr[i]) for i in range(int(n))) if arr is not None else ()

        return GraphEdges(
            tup(st, counts.source_turns),
            tup(sg, counts.source_grafts),
            tup(sp, counts.supersedes),
            tup(sb, counts.superseded_by))

    def source_closure(self, node_ids, max_depth=3, include_roots=False):
        if not getattr(self, "_has_source_closure", False):
            raise RuntimeError("native GRM source closure is unavailable")
        seeds, seed_n = self._u64_array(node_ids)
        needed = ctypes.c_uint64()
        self._check(self._lib.grm_store_source_closure(
            self._handle, seeds, seed_n, int(max_depth),
            1 if include_roots else 0, None, 0, ctypes.byref(needed)))
        if not needed.value:
            return ()
        arr_t = ctypes.c_uint64 * int(needed.value)
        out = arr_t()
        got = ctypes.c_uint64()
        self._check(self._lib.grm_store_source_closure(
            self._handle, seeds, seed_n, int(max_depth),
            1 if include_roots else 0, out, int(needed.value),
            ctypes.byref(got)))
        return tuple(int(out[i]) for i in range(int(got.value)))

    def apply_revision(self, replacement_node_id, supersedes):
        if not getattr(self, "_has_apply_revision", False):
            return
        arr, n = self._u64_array(supersedes)
        self._check(self._lib.grm_store_apply_revision(
            self._handle, int(replacement_node_id), arr, n))

    def apply_expire(self, node_ids):
        if not getattr(self, "_has_apply_expire", False):
            return
        arr, n = self._u64_array(node_ids)
        self._check(self._lib.grm_store_apply_expire(self._handle, arr, n))

    def metadata(self, node_id):
        needed = ctypes.c_uint64()
        self._check(self._lib.grm_store_metadata_json(
            self._handle, int(node_id), None, 0, ctypes.byref(needed)))
        cap = int(needed.value) + 1
        buf = ctypes.create_string_buffer(cap)
        self._check(self._lib.grm_store_metadata_json(
            self._handle, int(node_id), buf, cap, ctypes.byref(needed)))
        return json.loads(buf.value.decode("utf-8") or "{}")

    def parse_memory_command(self, text):
        if not getattr(self, "_has_parse_memory_command", False):
            raise RuntimeError("native GRM memory command parser is unavailable")
        needed = ctypes.c_uint64()
        data = str(text).encode("utf-8")
        self._check(self._lib.grm_store_parse_memory_command(
            self._handle, data, None, 0, ctypes.byref(needed)))
        cap = int(needed.value) + 1
        buf = ctypes.create_string_buffer(cap)
        self._check(self._lib.grm_store_parse_memory_command(
            self._handle, data, buf, cap, ctypes.byref(needed)))
        return json.loads(buf.value.decode("utf-8") or "{}")

    def plan_extraction_policy(self, *, action, write_intent, confidence,
                               write_direct_threshold, conflict_count=0,
                               requested_supersede_count=0,
                               requested_id_count=0, equivalent_count=0,
                               expire_target_count=0):
        if not getattr(self, "_has_extraction_policy_plan", False):
            raise RuntimeError(
                "native GRM extraction policy planner is unavailable")
        needed = ctypes.c_uint64()
        self._check(self._lib.grm_store_plan_extraction_policy(
            self._handle, str(action).encode("utf-8"),
            str(write_intent).encode("utf-8"), float(confidence),
            float(write_direct_threshold), int(conflict_count),
            int(requested_supersede_count), int(requested_id_count),
            int(equivalent_count), int(expire_target_count), None, 0,
            ctypes.byref(needed)))
        cap = int(needed.value) + 1
        buf = ctypes.create_string_buffer(cap)
        self._check(self._lib.grm_store_plan_extraction_policy(
            self._handle, str(action).encode("utf-8"),
            str(write_intent).encode("utf-8"), float(confidence),
            float(write_direct_threshold), int(conflict_count),
            int(requested_supersede_count), int(requested_id_count),
                int(equivalent_count), int(expire_target_count), buf, cap,
                ctypes.byref(needed)))
        return json.loads(buf.value.decode("utf-8") or "{}")

    def plan_reinforcement(self, *, old_write_intent, new_write_intent,
                           old_confidence, new_confidence,
                           old_reinforcement_count=0):
        if not getattr(self, "_has_reinforcement_plan", False):
            raise RuntimeError(
                "native GRM reinforcement planner is unavailable")
        needed = ctypes.c_uint64()
        self._check(self._lib.grm_store_plan_reinforcement(
            self._handle, str(old_write_intent).encode("utf-8"),
            str(new_write_intent).encode("utf-8"), float(old_confidence),
            float(new_confidence), int(old_reinforcement_count), None, 0,
            ctypes.byref(needed)))
        cap = int(needed.value) + 1
        buf = ctypes.create_string_buffer(cap)
        self._check(self._lib.grm_store_plan_reinforcement(
            self._handle, str(old_write_intent).encode("utf-8"),
            str(new_write_intent).encode("utf-8"), float(old_confidence),
            float(new_confidence), int(old_reinforcement_count), buf, cap,
            ctypes.byref(needed)))
        return json.loads(buf.value.decode("utf-8") or "{}")

    def plan_review_transition(self, *, command, status,
                               has_approved_node_id=False):
        if not getattr(self, "_has_review_transition_plan", False):
            raise RuntimeError(
                "native GRM review transition planner is unavailable")
        needed = ctypes.c_uint64()
        self._check(self._lib.grm_store_plan_review_transition(
            self._handle, str(command).encode("utf-8"),
            str(status or "pending").encode("utf-8"),
            1 if has_approved_node_id else 0, None, 0,
            ctypes.byref(needed)))
        cap = int(needed.value) + 1
        buf = ctypes.create_string_buffer(cap)
        self._check(self._lib.grm_store_plan_review_transition(
            self._handle, str(command).encode("utf-8"),
            str(status or "pending").encode("utf-8"),
            1 if has_approved_node_id else 0, buf, cap,
            ctypes.byref(needed)))
        return json.loads(buf.value.decode("utf-8") or "{}")

    def plan_cull_spans(self, *, ntok, max_tokens=None, spans=None,
                        retire_parent=True):
        if not getattr(self, "_has_cull_span_plan", False):
            raise RuntimeError("native GRM cull span planner is unavailable")
        spans = list(spans or ())
        if spans:
            arr_t = ctypes.c_uint64 * len(spans)
            starts = arr_t(*[int(s) for s, _ in spans])
            ends = arr_t(*[int(e) for _, e in spans])
            starts_ptr = starts
            ends_ptr = ends
        else:
            starts = ends = None
            starts_ptr = ends_ptr = None
        max_value = 0 if max_tokens is None else int(max_tokens)
        has_max = 0 if max_tokens is None else 1
        needed = ctypes.c_uint64()
        self._check(self._lib.grm_store_plan_cull_spans(
            self._handle, int(ntok), max_value, has_max, starts_ptr, ends_ptr,
            len(spans), 1 if retire_parent else 0, None, 0,
            ctypes.byref(needed)))
        cap = int(needed.value) + 1
        buf = ctypes.create_string_buffer(cap)
        self._check(self._lib.grm_store_plan_cull_spans(
            self._handle, int(ntok), max_value, has_max, starts_ptr, ends_ptr,
            len(spans), 1 if retire_parent else 0, buf, cap,
            ctypes.byref(needed)))
        out = json.loads(buf.value.decode("utf-8") or "{}")
        out["spans"] = [tuple(int(v) for v in span)
                        for span in out.get("spans", [])]
        return out

    def add_structured_node(self, text, payload, ntok=0):
        node_id = self.add_node(text, b"", ntok=ntok)
        for key in sorted(payload):
            self.set_tensor(node_id, key, payload[key])
        return node_id

    @staticmethod
    def _lexical_blob(keys):
        if not keys:
            return b""
        return "\n".join(str(k) for k in keys).encode("utf-8")

    @staticmethod
    def _filter_blob(values):
        if values is None:
            return b""
        if isinstance(values, str):
            values = (values,)
        vals = [str(v) for v in values if v is not None and str(v) != ""]
        if not vals:
            return b""
        return "\n".join(vals).encode("utf-8")

    @staticmethod
    def _float_array(values):
        vals = [float(v) for v in values]
        if not vals:
            return None, 0
        arr_t = ctypes.c_float * len(vals)
        return arr_t(*vals), len(vals)

    def set_route(self, node_id, route_key, lexical_keys=()):
        arr, n = self._float_array(route_key)
        self._check(self._lib.grm_store_set_route(
            self._handle, int(node_id), arr, n,
            self._lexical_blob(lexical_keys)))

    def set_route_keys(self, node_id, route_keys, lexical_keys=()):
        keys = np.asarray(route_keys, dtype=np.float32)
        if keys.ndim == 1:
            keys = keys.reshape(1, -1)
        if keys.ndim != 2:
            raise ValueError("route_keys must be a 1D or 2D float array")
        flat = np.ascontiguousarray(keys.reshape(-1), dtype=np.float32)
        arr = None
        if flat.size:
            arr_t = ctypes.c_float * int(flat.size)
            arr = arr_t(*[float(v) for v in flat])
        self._check(self._lib.grm_store_set_route_multi(
            self._handle, int(node_id), arr, int(keys.shape[0]),
            int(keys.shape[1]), self._lexical_blob(lexical_keys)))

    def set_route_key_list(self, node_id, route_keys, lexical_keys=()):
        if not getattr(self, "_has_route_list", False):
            raise RuntimeError("native GRM route key lists are unavailable")
        keys = [np.ascontiguousarray(k, dtype=np.float32).reshape(-1)
                for k in route_keys]
        offsets = [0]
        for key in keys:
            offsets.append(offsets[-1] + int(key.size))
        flat = (np.concatenate(keys).astype(np.float32, copy=False)
                if keys else np.asarray([], dtype=np.float32))
        values = None
        if flat.size:
            values_t = ctypes.c_float * int(flat.size)
            values = values_t(*[float(v) for v in flat])
        offsets_t = ctypes.c_uint64 * len(offsets)
        offsets_arr = offsets_t(*offsets)
        self._check(self._lib.grm_store_set_route_list(
            self._handle, int(node_id), values, int(flat.size),
            offsets_arr, len(keys), self._lexical_blob(lexical_keys)))

    def route(self, query, lexical_keys=(), topk=3, kinds=(), scopes=(),
              durabilities=(), mutabilities=()):
        q, n = self._float_array(query)
        cap = max(0, int(topk))
        out = None
        if cap:
            out_t = ctypes.c_uint64 * cap
            out = out_t()
        count = ctypes.c_uint64()
        filters = (
            self._filter_blob(kinds), self._filter_blob(scopes),
            self._filter_blob(durabilities), self._filter_blob(mutabilities))
        if any(filters):
            if not getattr(self, "_has_route_filtered", False):
                raise RuntimeError("native GRM route filters are unavailable")
            self._check(self._lib.grm_store_route_filtered(
                self._handle, q, n, self._lexical_blob(lexical_keys),
                filters[0], filters[1], filters[2], filters[3], cap,
                out, cap, ctypes.byref(count)))
        else:
            self._check(self._lib.grm_store_route(
                self._handle, q, n, self._lexical_blob(lexical_keys), cap,
                out, cap, ctypes.byref(count)))
        return [int(out[i]) for i in range(int(count.value))]

    def route_gqa(self, query, lexical_keys=(), topk=3, kinds=(), scopes=(),
                  durabilities=(), mutabilities=()):
        if not getattr(self, "_has_route_gqa", False):
            raise RuntimeError("native GRM GQA route is unavailable")
        q_np = np.ascontiguousarray(query, dtype=np.float32)
        if q_np.ndim != 3:
            raise ValueError("GQA query must have shape (heads, tokens, dim)")
        q_heads, q_tokens, head_dim = (int(q_np.shape[0]), int(q_np.shape[1]),
                                      int(q_np.shape[2]))
        flat = q_np.reshape(-1)
        q, _ = self._float_array(flat)
        cap = max(0, int(topk))
        out = None
        if cap:
            out_t = ctypes.c_uint64 * cap
            out = out_t()
        count = ctypes.c_uint64()
        filters = (
            self._filter_blob(kinds), self._filter_blob(scopes),
            self._filter_blob(durabilities), self._filter_blob(mutabilities))
        self._check(self._lib.grm_store_route_gqa(
            self._handle, q, q_heads, q_tokens, head_dim,
            self._lexical_blob(lexical_keys),
            filters[0], filters[1], filters[2], filters[3], cap,
            out, cap, ctypes.byref(count)))
        return [int(out[i]) for i in range(int(count.value))]

    def configure_arena(self, sink_tokens, arena_width):
        self._check(self._lib.grm_store_configure_arena(
            self._handle, int(sink_tokens), int(arena_width)))

    def plan_swap(self, new_mount_tokens, input_cache_tokens):
        out = _ArenaSwapPlanC()
        self._check(self._lib.grm_store_plan_swap(
            self._handle, int(new_mount_tokens), int(input_cache_tokens),
            ctypes.byref(out)))
        return ArenaSwapPlan(
            int(out.sink_tokens), int(out.arena_width),
            int(out.old_mount_tokens), int(out.new_mount_tokens),
            int(out.old_mount_end), int(out.live_tail_start),
            int(out.live_tail_tokens), int(out.input_cache_tokens),
            int(out.output_cache_tokens), bool(out.overflow))

    def plan_evict(self, drop_tokens, input_cache_tokens):
        out = _ArenaEvictPlanC()
        self._check(self._lib.grm_store_plan_evict(
            self._handle, int(drop_tokens), int(input_cache_tokens),
            ctypes.byref(out)))
        return ArenaEvictPlan(
            int(out.sink_tokens), int(out.arena_width),
            int(out.mount_tokens), int(out.head_tokens),
            int(out.drop_tokens), int(out.input_cache_tokens),
            int(out.output_cache_tokens), bool(out.underflow))

    def apply_swap_tensor(self, old_tensor, mount_tensor, seq_dim,
                          new_mount_tokens=None, input_cache_tokens=None):
        old = np.ascontiguousarray(old_tensor)
        mount = np.ascontiguousarray(mount_tensor)
        if old.ndim == 0:
            raise ValueError("swap tensor rank must be nonzero")
        if mount.ndim != old.ndim:
            raise ValueError("mount tensor rank must match old tensor rank")
        if mount.dtype != old.dtype:
            raise ValueError("mount tensor dtype must match old tensor dtype")
        seq = int(seq_dim)
        if seq < 0:
            seq += old.ndim
        if seq < 0 or seq >= old.ndim:
            raise ValueError("seq_dim out of range")
        if new_mount_tokens is None:
            new_mount_tokens = int(mount.shape[seq])
        if input_cache_tokens is None:
            input_cache_tokens = int(old.shape[seq])

        plan = self.plan_swap(int(new_mount_tokens), int(input_cache_tokens))
        if plan.overflow:
            raise RuntimeError("arena mount exceeds configured width")
        out_shape_py = list(old.shape)
        out_shape_py[seq] = plan.output_cache_tokens
        out_elements = int(np.prod(out_shape_py, dtype=np.uint64))
        out_nbytes = out_elements * old.dtype.itemsize

        old_shape, old_rank = self._shape_array(old.shape)
        mount_shape, mount_rank = self._shape_array(mount.shape)
        old_data, old_nbytes = self._byte_array(old.tobytes())
        mount_data, mount_nbytes = self._byte_array(mount.tobytes())
        out_shape_t = ctypes.c_uint64 * old.ndim
        out_shape = out_shape_t()
        out_payload = None
        if out_nbytes:
            out_payload_t = ctypes.c_uint8 * out_nbytes
            out_payload = out_payload_t()
        out_len = ctypes.c_uint64()
        out_plan = _ArenaSwapPlanC()
        self._check(self._lib.grm_store_apply_swap_tensor(
            self._handle, int(new_mount_tokens), int(input_cache_tokens),
            old_shape, old_rank, seq, old.dtype.itemsize,
            old_data, old_nbytes,
            mount_shape, mount_rank, mount_data, mount_nbytes,
            out_shape, old.ndim, out_payload, out_nbytes,
            ctypes.byref(out_len), ctypes.byref(out_plan)))
        if int(out_len.value) != out_nbytes:
            raise RuntimeError("native swap output length mismatch")
        out_shape_final = tuple(int(out_shape[i]) for i in range(old.ndim))
        data = bytes(out_payload[:out_nbytes]) if out_payload is not None else b""
        return np.frombuffer(data, dtype=old.dtype).copy().reshape(out_shape_final)

    def apply_swap_payload(self, old_payload, mount_payload, payload_dims,
                           new_mount_tokens=None, input_cache_tokens=None):
        out = {}
        for key, seq_dim in payload_dims:
            if key not in old_payload:
                raise KeyError(f"old payload missing tensor {key!r}")
            if key not in mount_payload:
                raise KeyError(f"mount payload missing tensor {key!r}")
            out[key] = self.apply_swap_tensor(
                old_payload[key], mount_payload[key], seq_dim,
                new_mount_tokens=new_mount_tokens,
                input_cache_tokens=input_cache_tokens)
        return out

    def apply_evict_tensor(self, old_tensor, seq_dim, drop_tokens,
                           input_cache_tokens=None):
        old = np.ascontiguousarray(old_tensor)
        if old.ndim == 0:
            raise ValueError("evict tensor rank must be nonzero")
        seq = int(seq_dim)
        if seq < 0:
            seq += old.ndim
        if seq < 0 or seq >= old.ndim:
            raise ValueError("seq_dim out of range")
        if input_cache_tokens is None:
            input_cache_tokens = int(old.shape[seq])

        plan = self.plan_evict(int(drop_tokens), int(input_cache_tokens))
        if plan.underflow:
            raise RuntimeError("evict drop exceeds live tail")
        out_shape_py = list(old.shape)
        out_shape_py[seq] = plan.output_cache_tokens
        out_elements = int(np.prod(out_shape_py, dtype=np.uint64))
        out_nbytes = out_elements * old.dtype.itemsize

        old_shape, old_rank = self._shape_array(old.shape)
        old_data, old_nbytes = self._byte_array(old.tobytes())
        out_shape_t = ctypes.c_uint64 * old.ndim
        out_shape = out_shape_t()
        out_payload = None
        if out_nbytes:
            out_payload_t = ctypes.c_uint8 * out_nbytes
            out_payload = out_payload_t()
        out_len = ctypes.c_uint64()
        out_plan = _ArenaEvictPlanC()
        self._check(self._lib.grm_store_apply_evict_tensor(
            self._handle, int(drop_tokens), int(input_cache_tokens),
            old_shape, old_rank, seq, old.dtype.itemsize, old_data, old_nbytes,
            out_shape, old.ndim, out_payload, out_nbytes,
            ctypes.byref(out_len), ctypes.byref(out_plan)))
        if int(out_len.value) != out_nbytes:
            raise RuntimeError("native evict output length mismatch")
        out_shape_final = tuple(int(out_shape[i]) for i in range(old.ndim))
        data = bytes(out_payload[:out_nbytes]) if out_payload is not None else b""
        return np.frombuffer(data, dtype=old.dtype).copy().reshape(out_shape_final)

    def apply_evict_payload(self, old_payload, payload_dims, drop_tokens,
                            input_cache_tokens=None):
        out = {}
        for key, seq_dim in payload_dims:
            if key not in old_payload:
                raise KeyError(f"old payload missing tensor {key!r}")
            out[key] = self.apply_evict_tensor(
                old_payload[key], seq_dim, drop_tokens,
                input_cache_tokens=input_cache_tokens)
        return out

    def commit_mount(self, node_ids, mount_tokens):
        ids = [int(i) for i in node_ids]
        arr = None
        if ids:
            arr_t = ctypes.c_uint64 * len(ids)
            arr = arr_t(*ids)
        self._check(self._lib.grm_store_commit_mount(
            self._handle, arr, len(ids), int(mount_tokens)))

    def save_checkpoint(self, root):
        self._check(self._lib.grm_store_save_checkpoint(
            self._handle, os.fspath(root).encode("utf-8")))

    def load_checkpoint(self, root):
        self._check(self._lib.grm_store_load_checkpoint(
            self._handle, os.fspath(root).encode("utf-8")))

    def mark_durable(self, node_id):
        self._check(self._lib.grm_store_mark_durable(
            self._handle, int(node_id)))

    def clear_payload(self, node_id):
        if not getattr(self, "_has_clear_payload", False):
            raise RuntimeError("native GRM clear_payload is unavailable")
        self._check(self._lib.grm_store_clear_payload(
            self._handle, int(node_id)))

    def evict_device_copy(self, node_id):
        self._check(self._lib.grm_store_evict_device_copy(
            self._handle, int(node_id)))

    def dirty_node_ids(self):
        if not getattr(self, "_has_dirty_nodes", False):
            raise RuntimeError("native GRM dirty_nodes is unavailable")
        count = ctypes.c_uint64()
        self._check(self._lib.grm_store_dirty_nodes(
            self._handle, None, 0, ctypes.byref(count)))
        if not int(count.value):
            return ()
        out_t = ctypes.c_uint64 * int(count.value)
        out = out_t()
        written = ctypes.c_uint64()
        self._check(self._lib.grm_store_dirty_nodes(
            self._handle, out, int(count.value), ctypes.byref(written)))
        return tuple(int(out[i]) for i in range(int(written.value)))

    def dirty_plan(self):
        if not getattr(self, "_has_dirty_plan", False):
            raise RuntimeError("native GRM dirty_plan is unavailable")
        count = ctypes.c_uint64()
        self._check(self._lib.grm_store_dirty_plan(
            self._handle, None, 0, ctypes.byref(count)))
        if not int(count.value):
            return ()
        out_t = _DirtyNodeC * int(count.value)
        out = out_t()
        written = ctypes.c_uint64()
        self._check(self._lib.grm_store_dirty_plan(
            self._handle, out, int(count.value), ctypes.byref(written)))
        return tuple(
            DirtyNode(
                int(out[i].node_id),
                bool(out[i].payload_dirty),
                bool(out[i].metadata_dirty),
                int(out[i].payload_bytes),
                int(out[i].durability_priority))
            for i in range(int(written.value)))

    def stats(self):
        out = _StatsC()
        self._check(self._lib.grm_store_stats(self._handle, ctypes.byref(out)))
        return NativeStats(out.nodes, out.dirty_nodes, out.durable_nodes,
                           out.host_payload_bytes, out.host_payload_tensors,
                           out.route_entries)
