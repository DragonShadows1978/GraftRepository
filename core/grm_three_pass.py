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
