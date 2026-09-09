"""Three-pass turn helpers with no model or TensorCUDA dependency.

The session driver owns scheduling.  This module owns the two pieces that
must be independently testable: the pass-2 persistent-arena read-only guard
and the frozen per-turn memory-ledger receipt producer.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import time
from typing import Any, Callable


MEMORY_LEDGER_SCHEMA = "grm.memory_ledger.turn.v1"
# LSR-P2B: route receipts get their OWN versioned schema.  The memory-ledger
# mutation schema above is FROZEN; a new fact never re-versions an old record.
ROUTE_RECEIPT_SCHEMA = "grm.route_receipt.v1"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class Pass2ArenaMutationError(AssertionError):
    """Raised when pass 2 changes persistent arena state."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(str(value).encode("utf-8"))


def _array_projection(value: Any) -> dict[str, Any] | None:
    """Return a deterministic digest projection for numpy/tensor values."""
    candidate = value
    if hasattr(candidate, "numpy") and not hasattr(candidate, "tobytes"):
        try:
            candidate = candidate.numpy()
        except Exception:
            return None
    if not (hasattr(candidate, "tobytes") and hasattr(candidate, "shape")):
        return None
    try:
        raw = candidate.tobytes(order="C")
    except TypeError:
        raw = candidate.tobytes()
    return {
        "dtype": str(getattr(candidate, "dtype", type(candidate).__name__)),
        "shape": [int(x) for x in getattr(candidate, "shape", ())],
        "sha256": sha256_bytes(raw),
    }


def canonical_value(value: Any) -> Any:
    """Convert repository values to deterministic JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, bytes):
        return {"bytes_sha256": sha256_bytes(value), "size": len(value)}
    if isinstance(value, Mapping):
        return {
            str(key): canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        projected = [canonical_value(item) for item in value]
        return sorted(
            projected,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    array = _array_projection(value)
    if array is not None:
        return {"array": array}
    if hasattr(value, "item"):
        try:
            return canonical_value(value.item())
        except Exception:
            pass
    return {"type": type(value).__name__, "repr": repr(value)}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


_PERSISTENT_GRAFT_KEYS = (
    "node_id",
    "native_node_id",
    "kind",
    "text",
    "ntok",
    "sources",
    "retired",
    "no_fold",
    "tags",
    "metadata",
    "payload_pending",
    "host_present",
    "device_present",
    "dirty",
    "durable",
    "cold_only",
    "saved",
)


def graft_projection(arena: Any, graft: Mapping[str, Any], *,
                     include_payload: bool = False) -> dict[str, Any]:
    """Canonical projection of persisted memory, excluding read-side caches."""
    projected = {
        key: canonical_value(graft[key])
        for key in _PERSISTENT_GRAFT_KEYS
        if key in graft
    }
    if "provenance" in graft:
        # Existing repositories stamp provenance with wall-clock seconds.
        # The frozen receipt contract deliberately excludes wall clocks so
        # schedule-equivalence compares canonical arena bytes rather than
        # unrelated run start times. All identity/source fields remain.
        projected["provenance"] = canonical_value([
            {key: value for key, value in entry.items() if key != "created_at"}
            if isinstance(entry, Mapping) else entry
            for entry in graft.get("provenance", ())
        ])
    if "cent" in graft:
        projected["cent"] = canonical_value(graft["cent"])
    if include_payload and graft.get("h") is not None:
        try:
            packed = arena.pack_node(graft["h"])
        except Exception as exc:
            projected["payload_error"] = repr(exc)
        else:
            projected["payload"] = canonical_value(packed)
    elif include_payload and graft.get("host_payload") is not None:
        projected["payload"] = canonical_value(graft["host_payload"])
    return projected


def arena_control_projection(arena: Any) -> dict[str, Any]:
    """Persistent mutation counters, not route/mount/live-cache transients."""
    out = {}
    for key in ("_s4_turn", "_cuda_gqa_epoch"):
        if hasattr(arena, key):
            out[key] = canonical_value(getattr(arena, key))
    return out


def arena_state_projection(repository: Any, *,
                           include_payload: bool = False) -> dict[str, Any]:
    arena = repository.arena
    return {
        "control": arena_control_projection(arena),
        "grafts": [
            graft_projection(arena, graft, include_payload=include_payload)
            for graft in arena.grafts
        ],
    }


def arena_state_bytes(repository: Any, *, include_payload: bool = False) -> bytes:
    return canonical_json_bytes(
        arena_state_projection(repository, include_payload=include_payload))


def arena_state_sha256(repository: Any, *,
                       include_payload: bool = False) -> str:
    return sha256_bytes(
        arena_state_bytes(repository, include_payload=include_payload))


def _target_snapshots(repository: Any) -> dict[tuple[str, str], str]:
    arena = repository.arena
    snapshots: dict[tuple[str, str], str] = {
        ("arena_control", "@arena"): sha256_bytes(
            canonical_json_bytes(arena_control_projection(arena)))
    }
    for idx, graft in enumerate(arena.grafts):
        snapshots[("grafts", str(idx))] = sha256_bytes(
            canonical_json_bytes(graft_projection(arena, graft)))
    return snapshots


class Pass2ReadOnlyGuard:
    """Instrument and assert the pass-2 persistent mutation boundary."""

    def __init__(self, repository: Any):
        self.repository = repository
        self.before_sha256: str | None = None
        self.after_sha256: str | None = None
        self.visible_overhead_ms = 0.0
        self.read_only = False

    def __enter__(self) -> "Pass2ReadOnlyGuard":
        started = time.perf_counter()
        self.before_sha256 = arena_state_sha256(self.repository)
        self.visible_overhead_ms += (time.perf_counter() - started) * 1000.0
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        started = time.perf_counter()
        self.after_sha256 = arena_state_sha256(self.repository)
        self.visible_overhead_ms += (time.perf_counter() - started) * 1000.0
        self.read_only = self.before_sha256 == self.after_sha256
        if not self.read_only and exc_type is None:
            raise Pass2ArenaMutationError(
                "pass 2 mutated persistent arena state: "
                f"before={self.before_sha256} after={self.after_sha256}")
        return False


class TurnStepIOTracker:
    """Attribute graft page-ins and device uploads to one turn step.

    The tracker deliberately covers repository payload materialization and
    immutable CUDA route-bank attachment: the two upload classes controlled
    by prep staging.  Ordinary model-token transfers are inference traffic,
    not graft page-ins, and are outside this receipt.
    """

    STEP_NAMES = ("1_prep", "2_inference", "3_cleanup")

    def __init__(self, arena: Any):
        self.arena = arena
        self.current_step: str | None = None
        self.events: dict[str, list[dict[str, Any]]] = {
            step: [] for step in self.STEP_NAMES
        }
        self._node_loader = None
        self._configure_cuda_gqa_route_bank = None
        self._store = None

    def set_step(self, step: str | None) -> None:
        if step is not None and step not in self.STEP_NAMES:
            raise ValueError(f"unknown turn step {step!r}")
        self.current_step = step

    def _append(self, event: dict[str, Any]) -> None:
        if self.current_step is not None:
            self.events[self.current_step].append(event)

    def __enter__(self) -> "TurnStepIOTracker":
        loader = getattr(self.arena, "node_loader", None)
        if callable(loader):
            self._node_loader = loader

            def tracked_loader(node_id):
                graft = self.arena.grafts[int(node_id)]
                source = (
                    "ram_host_payload"
                    if graft.get("host_payload") is not None else "nvme"
                )
                started = time.perf_counter()
                result = loader(node_id)
                wall_ms = (time.perf_counter() - started) * 1000.0
                success = result is not None
                self._append({
                    "kind": "graft_page_in",
                    "node_id": int(node_id),
                    "source": source,
                    "success": success,
                    "wall_ms": wall_ms,
                })
                if success:
                    self._append({
                        "kind": "graft_payload_upload",
                        "node_id": int(node_id),
                        "source": source,
                        "success": True,
                        "wall_ms": wall_ms,
                    })
                return result

            self.arena.node_loader = tracked_loader

        store = getattr(self.arena, "native_store", None)
        configure = getattr(store, "configure_cuda_gqa_route_bank", None)
        if callable(configure):
            self._store = store
            self._configure_cuda_gqa_route_bank = configure

            def tracked_configure(route_bank, node_ids=None, **kwargs):
                started = time.perf_counter()
                bank = configure(route_bank, node_ids, **kwargs)
                self._append({
                    "kind": "cuda_route_bank_upload",
                    "success": True,
                    "node_count": int(
                        len(node_ids) if node_ids is not None
                        else getattr(route_bank, "shape", (0,))[0]),
                    "bytes": int(getattr(route_bank, "nbytes", 0)),
                    "wall_ms": (time.perf_counter() - started) * 1000.0,
                })
                return bank

            store.configure_cuda_gqa_route_bank = tracked_configure
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._node_loader is not None:
            self.arena.node_loader = self._node_loader
        if (self._store is not None
                and self._configure_cuda_gqa_route_bank is not None):
            self._store.configure_cuda_gqa_route_bank = (
                self._configure_cuda_gqa_route_bank)
        self.current_step = None
        return False

    def receipt(self) -> dict[str, Any]:
        steps = {}
        for step in self.STEP_NAMES:
            events = list(self.events[step])
            steps[step] = {
                "page_in_count": sum(
                    event.get("kind") == "graft_page_in"
                    and event.get("success") is True
                    for event in events
                ),
                "upload_count": sum(
                    event.get("kind") in (
                        "graft_payload_upload", "cuda_route_bank_upload")
                    and event.get("success") is True
                    for event in events
                ),
                "events": events,
            }
        return {
            "schema": "grm.three_pass.step_io.v1",
            "steps": steps,
        }


class StagedWorkingSetResolver:
    """Resolve pass-2 routes and mount payloads from a pre-staged set.

    A miss falls back to the repository through the original methods and is
    always counted.  The registered smoke gate expects no such fallback; the
    fallback exists so a production miss is observable rather than fatal or
    silently truncated.
    """

    def __init__(
        self,
        arena: Any,
        *,
        probe_text: str,
        ranking_ids: list[int] | tuple[int, ...],
        staged_ids: list[int] | tuple[int, ...],
        route_backend: str = "cuda",
    ) -> None:
        self.arena = arena
        self.probe_text = str(probe_text)
        self.ranking_ids = [int(i) for i in ranking_ids]
        self.staged_ids = {int(i) for i in staged_ids}
        self.route_backend = str(route_backend)
        self.route_l1_calls = 0
        self.payload_l1_resolutions = 0
        self.l2_misses: list[dict[str, Any]] = []
        self._route = None
        self._ensure_h = None

    def __enter__(self) -> "StagedWorkingSetResolver":
        self._route = self.arena.route
        self._ensure_h = self.arena._ensure_h

        def staged_route(bare_text, exclude, limit=None, **kwargs):
            if str(bare_text) != self.probe_text:
                self.l2_misses.append({
                    "kind": "route_probe_mismatch",
                    "probe": str(bare_text),
                })
                return self._route(bare_text, exclude, limit, **kwargs)
            excluded = {int(i) for i in (exclude or ())}
            available = [
                i for i in self.ranking_ids if i not in excluded
            ]
            want = len(available) if limit is None else min(
                max(0, int(limit)), len(available))
            requested = available[:want]
            missing = [i for i in requested if i not in self.staged_ids]
            if missing:
                self.l2_misses.extend({
                    "kind": "route_id_not_staged",
                    "node_id": int(i),
                } for i in missing)
                return self._route(bare_text, exclude, limit, **kwargs)
            self.route_l1_calls += 1
            self.arena.last_route_backend = self.route_backend
            return requested

        def staged_ensure_h(idxs):
            requested = [int(i) for i in idxs]
            for node_id in requested:
                graft = self.arena.grafts[node_id]
                if node_id in self.staged_ids and graft.get("h") is not None:
                    self.payload_l1_resolutions += 1
                else:
                    self.l2_misses.append({
                        "kind": "payload_not_staged",
                        "node_id": node_id,
                        "device_resident": graft.get("h") is not None,
                    })
            return self._ensure_h(requested)

        self.arena.route = staged_route
        self.arena._ensure_h = staged_ensure_h
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._route is not None:
            self.arena.route = self._route
        if self._ensure_h is not None:
            self.arena._ensure_h = self._ensure_h
        return False

    def receipt(self) -> dict[str, Any]:
        return {
            "route_l1_calls": int(self.route_l1_calls),
            "payload_l1_resolutions": int(self.payload_l1_resolutions),
            "l2_miss_count": len(self.l2_misses),
            "l2_misses": list(self.l2_misses),
        }


# --------------------------------------------------------------- LSR-P2B
# Route-receipt persistence.  LSR Phase 1 earned the principle that "route
# receipts are not persisted at serving time"; every field below is one that
# Phase 1 had to RECONSTRUCT from snapshot manifests after the fact.  The
# record is built from serve-time state only and carries its own version.

# Generic pass-through prefixes/keys.  P2A adds fit-honesty and abstention
# fields to arena/driver ``info``; naming them by PREFIX means P2B persists
# them the moment they exist and needs no re-edit when P2A lands.
ROUTE_RECEIPT_INFO_PREFIXES = ("fit_", "abstain", "demand_", "grounding_",
                               "frame_", "recency_", "live_segments_",
                               "admission_")
# Prior art: GRM P2B receipt pass-through (GRM contributors, 2026).
# FIX-4 live-service evidence must survive the same route receipt boundary.
ROUTE_RECEIPT_INFO_KEYS = ("served_without_plan_head", "served_from",
                           "served_from_node_ids")

# ``info`` keys already projected into named receipt sections, so the generic
# pass-through does not duplicate them.
_ROUTE_RECEIPT_CLAIMED_INFO_KEYS = frozenset({
    "route_receipt", "route_backend", "trip", "clean_room", "mounts",
    "mount_plan", "mount_fitted", "mount_dropped_for_width", "ranking_ids",
    "no_mount_fit", "resident", "evicted", "live_tokens", "extraction",
    "_deferred_memory", "driver_probe_multimount", "driver_probe_ladder",
    "driver_topk", "point_lookup", "precise_first", "ungrounded_kept_first",
    "admission_policy", "admission_policy_branch", "admission_rank_plan",
    "admission_identifier_hit_count", "admission_identified_candidates",
    "admission_route_margin_1_2", "admission_route_margin_evaluated",
    "admission_margin_threshold", "admission_rule_sha256",
})


def _int_list(values: Any) -> list[int]:
    if values is None:
        return []
    out: list[int] = []
    for value in values:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def _mounts_to_graft_indices(values: Any) -> list[int]:
    """``_attempt`` reports ``mounts`` 1-based; the receipt is graft-indexed."""
    return [value - 1 for value in _int_list(values)]


def _route_receipt_generic_info(info: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in info.items():
        name = str(key)
        if name in _ROUTE_RECEIPT_CLAIMED_INFO_KEYS:
            continue
        if (name.startswith(ROUTE_RECEIPT_INFO_PREFIXES)
                or name in ROUTE_RECEIPT_INFO_KEYS):
            out[name] = canonical_value(value)
    return out


def route_receipt_sha256(record: Mapping[str, Any]) -> str:
    """Sha rule: sha256 over canonical JSON of the record MINUS the stamp.

    Canonical JSON is ``canonical_json_bytes`` (sorted keys, no spaces,
    UTF-8, deterministic float/array/bytes projections), so the same serve
    state always yields the same digest and the digest never covers itself.
    """
    payload = {key: value for key, value in record.items()
               if key != "receipt_sha256"}
    return sha256_bytes(canonical_json_bytes(payload))


def build_route_receipt(
    *,
    session_id: str,
    turn_id: str,
    info: Mapping[str, Any] | None = None,
    arena: Any = None,
    repository: Any = None,
    request_text: str | None = None,
    output_text: str | None = None,
    request_sha256: str | None = None,
    output_sha256: str | None = None,
    arena_before_sha256: str | None = None,
    repository_size: int | None = None,
    serving_path: str = "unknown",
    route_limit: int | None = None,
    candidate_count: int | None = None,
    ranking_ids: Any = None,
    ranking_scores: Any = None,
    excluded_live_ids: Any = None,
    excluded_recency_ids: Any = None,
    admission_profile: Mapping[str, Any] | None = None,
    trips: Any = None,
    turn_kind: str | None = None,
) -> dict[str, Any]:
    """Assemble the per-turn ``grm.route_receipt.v1`` record.

    Written on EVERY turn — mutation-free, abstained, and ``no_mount_fit``
    turns included.  Every argument is optional so a caller that genuinely
    cannot observe a field records ``null`` rather than a fabrication.
    """
    info = dict(info or {})
    if arena is None and repository is not None:
        arena = getattr(repository, "arena", None)

    arena_receipt = info.get("route_receipt")
    if not isinstance(arena_receipt, Mapping):
        arena_receipt = getattr(arena, "last_route_receipt", None)
    arena_receipt = (dict(arena_receipt)
                     if isinstance(arena_receipt, Mapping) else None)

    if ranking_ids is None:
        ranking_ids = info.get("ranking_ids")
    if ranking_ids is None and admission_profile is not None:
        ranking_ids = admission_profile.get("ranking")
    ranking = _int_list(ranking_ids)

    scores = None
    if ranking_scores is not None:
        scores = [None if value is None else float(value)
                  for value in ranking_scores]

    if candidate_count is None and arena_receipt is not None:
        candidate_count = arena_receipt.get("candidate_count")

    backend = info.get("route_backend")
    if backend is None and arena_receipt is not None:
        backend = arena_receipt.get("route_backend")
    if backend is None and arena is not None:
        backend = getattr(arena, "last_route_backend", None)

    fallbacks = []
    if arena_receipt is not None:
        for reason in arena_receipt.get("fallbacks") or ():
            fallbacks.append({"reason_code": str(reason)})

    planned = _int_list(info.get("mount_plan"))
    if not planned and admission_profile is not None:
        planned = _int_list(admission_profile.get("rank_plan"))
    seated = _int_list(info.get("mount_fitted"))
    dropped = _int_list(info.get("mount_dropped_for_width"))
    final_mounts = _mounts_to_graft_indices(info.get("mounts"))
    if not final_mounts and arena is not None:
        final_mounts = _int_list(getattr(arena, "cur_mounts", ()))

    if repository_size is None and arena is not None:
        grafts = getattr(arena, "grafts", None)
        if grafts is not None:
            repository_size = len(grafts)

    if arena_before_sha256 is None and repository is not None:
        try:
            arena_before_sha256 = arena_state_sha256(repository)
        except Exception:
            arena_before_sha256 = None

    if request_sha256 is None and request_text is not None:
        request_sha256 = sha256_text(request_text)
    if output_sha256 is None and output_text is not None:
        output_sha256 = sha256_text(output_text)

    trip_rows: list[dict[str, Any]] = []
    for row in trips or ():
        if not isinstance(row, Mapping):
            continue
        trip_rows.append({
            "ordinal": int(row.get("ordinal", len(trip_rows))),
            "clean_room": bool(row.get("clean_room", False)),
            "planned": _int_list(row.get("planned")),
            "mount_set": _int_list(row.get("mount_set")),
            "grounded": (None if row.get("grounded") is None
                         else bool(row.get("grounded"))),
            # LSR-P2A Ruling 1.2 shuttle rungs are additive to max_trips.
            "shuttle": bool(row.get("shuttle", False)),
        })
    # ``trips=None`` means the caller observed nothing and the single served
    # attempt is reconstructed from ``info``.  An EMPTY list is a positive
    # observation of zero rungs -- P2A's abstention short-circuit returns
    # before any mount work, and synthesizing a rung there would invent a
    # trip that never ran.
    if trips is None and not trip_rows and info.get("trip") is not None:
        trip_rows.append({
            "ordinal": int(info["trip"]),
            "clean_room": bool(info.get("clean_room", False)),
            "planned": list(planned),
            "mount_set": list(final_mounts),
            "grounded": None,
            "shuttle": False,
        })

    record: dict[str, Any] = {
        "schema": ROUTE_RECEIPT_SCHEMA,
        "session_id": str(session_id),
        "turn_id": str(turn_id),
        "turn_kind": None if turn_kind is None else str(turn_kind),
        "serving_path": str(serving_path),
        "route": {
            "backend": None if backend is None else str(backend),
            "route_limit": None if route_limit is None else int(route_limit),
            "candidate_count": (None if candidate_count is None
                                else int(candidate_count)),
            "ranking_ids": ranking,
            "ranking_scores": scores,
            "ranking_length": len(ranking),
            "excluded_live_ids": sorted(_int_list(excluded_live_ids)),
            "excluded_recency_ids": sorted(_int_list(excluded_recency_ids)),
            "fallbacks": fallbacks,
            "arena_route_receipt": (canonical_value(arena_receipt)
                                    if arena_receipt is not None else None),
        },
        "admission": {
            "policy": info.get("admission_policy"),
            "identifier_tokens": (
                [str(token) for token in admission_profile["identifier_tokens"]]
                if admission_profile is not None
                and admission_profile.get("identifier_tokens") is not None
                else None),
            "rare_identifier_tokens": (
                [str(token)
                 for token in admission_profile["rare_identifier_tokens"]]
                if admission_profile is not None
                and admission_profile.get("rare_identifier_tokens") is not None
                else None),
            "identified_candidates": _int_list(
                info.get("admission_identified_candidates")
                if info.get("admission_identified_candidates") is not None
                else (admission_profile or {}).get("identified_candidates")),
            "identifier_hit_count": (
                int(info["admission_identifier_hit_count"])
                if info.get("admission_identifier_hit_count") is not None
                else None),
            "policy_branch": (
                str(info["admission_policy_branch"])
                if info.get("admission_policy_branch") is not None
                else None),
            "rank_plan": _int_list(
                info.get("admission_rank_plan")
                if info.get("admission_rank_plan") is not None
                else (admission_profile or {}).get("rank_plan")),
            "route_margin_1_2": (
                float(info["admission_route_margin_1_2"])
                if info.get("admission_route_margin_1_2") is not None
                else None),
            "route_margin_evaluated": (
                bool(info["admission_route_margin_evaluated"])
                if info.get("admission_route_margin_evaluated") is not None
                else None),
            "margin_threshold": (
                float(info["admission_margin_threshold"])
                if info.get("admission_margin_threshold") is not None
                else None),
            "rule_sha256": info.get("admission_rule_sha256"),
            # Prior art: LSR-P2B projection and RT1 receipts (project, 2026).
            # Carry existing RT1 evidence additively; absence is not False.
            **{key.removeprefix("admission_"): info[key]
               for key in (
                   "admission_split_child_demoted",
                   "admission_split_demoted_ids",
                   "admission_split_family_ids",
                   "admission_ranking_before_demotion")
               if key in info},
        },
        "fit": {
            "planned": planned,
            "seated": seated,
            "dropped": dropped,
            "final_mounts": final_mounts,
            "cur_mount_n": (None if arena is None
                            else int(getattr(arena, "cur_mount_n", 0) or 0)),
            "width": (None if arena is None or getattr(arena, "width", None) is None
                      else int(arena.width)),
            "trips": trip_rows,
            "trip_count": len(trip_rows),
            "no_mount_fit": bool(info.get("no_mount_fit", False)),
            "info_pass_through": _route_receipt_generic_info(info),
        },
        "provenance": {
            "repository_size_at_probe": (None if repository_size is None
                                         else int(repository_size)),
            "arena_state_sha256_before_serve": arena_before_sha256,
            "request_sha256": request_sha256,
            "output_sha256": output_sha256,
        },
    }
    record["receipt_sha256"] = route_receipt_sha256(record)
    return record


class MemoryLedgerBuilder:
    """Record every observed pass-3 target mutation in frozen-schema form."""

    def __init__(
        self,
        repository: Any,
        *,
        session_id: str,
        turn_id: str,
        request_text: str,
        output_text: str,
    ) -> None:
        self.repository = repository
        self.session_id = str(session_id)
        self.turn_id = str(turn_id)
        self.request_sha256 = sha256_text(request_text)
        self.output_sha256 = sha256_text(output_text)
        self.arena_before_sha256 = arena_state_sha256(repository)
        self._overall_before = _target_snapshots(repository)
        self.mutations: list[dict[str, Any]] = []
        # LSR-P2B: optional sidecar.  Never attached => finalize() returns the
        # frozen key set byte-for-byte, so every pre-P2B consumer and test is
        # unaffected.
        self.route_receipt: dict[str, Any] | None = None

    def attach_route_receipt(self, receipt: Mapping[str, Any] | None) -> None:
        """Attach a ``grm.route_receipt.v1`` record alongside the mutations."""
        if receipt is None:
            self.route_receipt = None
            return
        if str(receipt.get("schema")) != ROUTE_RECEIPT_SCHEMA:
            raise ValueError(
                "route receipt must carry schema "
                f"{ROUTE_RECEIPT_SCHEMA!r}, got {receipt.get('schema')!r}")
        self.route_receipt = dict(receipt)

    def record_operation(
        self,
        kind: str,
        *,
        reason_code: str,
        reason_detail: str,
        source_text: str,
        operation: Callable[[], Any],
    ) -> Any:
        if kind not in ("deposit", "supersession", "importance_bookkeeping"):
            raise ValueError(f"unsupported memory-ledger mutation kind {kind!r}")
        before = _target_snapshots(self.repository)
        result = operation()
        after = _target_snapshots(self.repository)
        source_sha256 = sha256_text(source_text)
        changed = sorted(
            set(before) | set(after),
            key=lambda target: (target[0], target[1]),
        )
        for arena_name, record_id in changed:
            before_sha256 = before.get((arena_name, record_id), EMPTY_SHA256)
            after_sha256 = after.get((arena_name, record_id), EMPTY_SHA256)
            if before_sha256 == after_sha256:
                continue
            sequence = len(self.mutations)
            decision = {
                "kind": kind,
                "target": {"arena": arena_name, "record_id": record_id},
                "reason": {"code": reason_code, "detail": reason_detail},
                "source_sha256": source_sha256,
                "before_sha256": before_sha256,
                "after_sha256": after_sha256,
            }
            self.mutations.append({
                "sequence": sequence,
                "kind": kind,
                "target": decision["target"],
                "reason": decision["reason"],
                "provenance": {
                    "source_sha256": source_sha256,
                    "decision_sha256": sha256_bytes(
                        canonical_json_bytes(decision)),
                },
                "before_sha256": before_sha256,
                "after_sha256": after_sha256,
            })
        return result

    def finalize(self) -> tuple[dict[str, Any], dict[str, Any]]:
        overall_after = _target_snapshots(self.repository)
        actual_changed = {
            target for target in set(self._overall_before) | set(overall_after)
            if self._overall_before.get(target, EMPTY_SHA256)
            != overall_after.get(target, EMPTY_SHA256)
        }
        receipted = {
            (row["target"]["arena"], row["target"]["record_id"])
            for row in self.mutations
        }
        missing = sorted(actual_changed - receipted)
        receipt = {
            "schema": MEMORY_LEDGER_SCHEMA,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "turn_pipeline": "three_pass",
            "pass": 3,
            "provenance": {
                "request_sha256": self.request_sha256,
                "output_sha256": self.output_sha256,
                "arena_before_sha256": self.arena_before_sha256,
                "arena_after_sha256": arena_state_sha256(self.repository),
            },
            "mutations": self.mutations,
            "mutation_count": len(self.mutations),
        }
        if self.route_receipt is not None:
            # Sidecar only.  Mutation rows and the completeness audit below
            # are computed exactly as before this key existed.
            receipt["route_receipt"] = self.route_receipt
        audit = {
            "schema": "grm.memory_ledger.completeness.v1",
            "turn_id": self.turn_id,
            "observed_changed_targets": [
                {"arena": arena, "record_id": record_id}
                for arena, record_id in sorted(actual_changed)
            ],
            "receipted_targets": [
                {"arena": arena, "record_id": record_id}
                for arena, record_id in sorted(receipted)
            ],
            "missing_targets": [
                {"arena": arena, "record_id": record_id}
                for arena, record_id in missing
            ],
            "mutation_count_matches": (
                receipt["mutation_count"] == len(receipt["mutations"])),
            "sequence_contiguous": all(
                row["sequence"] == idx
                for idx, row in enumerate(receipt["mutations"])),
            "complete": not missing,
        }
        return receipt, audit
