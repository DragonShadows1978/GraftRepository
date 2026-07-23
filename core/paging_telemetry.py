"""Opt-in paging / mount-payload telemetry for GRM3P-DIAG-CONTAM.

Default-off. Enabled only when env paths are set:

  GRM_PAGING_TELEMETRY_PATH  — JSONL path for page-in / eviction / pack events
  GRM_MOUNT_SNAPSHOT_DIR     — directory for probe-turn device payload dumps

When both are unset, every public entry point is a cheap no-op so the
registered default smoke path is byte-identical to an uninstrumented build.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


_ENV_PAGING_PATH = "GRM_PAGING_TELEMETRY_PATH"
_ENV_SNAPSHOT_DIR = "GRM_MOUNT_SNAPSHOT_DIR"


def _env_path(name: str) -> Optional[str]:
    raw = os.environ.get(name, "").strip()
    return raw or None


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def _tensor_to_numpy(t) -> np.ndarray:
    """Device/host tensor -> contiguous numpy float32 (read-only clone)."""
    if isinstance(t, np.ndarray):
        return np.ascontiguousarray(t)
    # tensor_cuda / numpy-bridge tensors
    if hasattr(t, "float") and callable(t.float):
        t = t.float()
    if hasattr(t, "numpy") and callable(t.numpy):
        arr = t.numpy()
    elif hasattr(t, "get"):
        arr = np.array(t.get())
    else:
        arr = np.array(t)
    return np.ascontiguousarray(arr)


def device_payload_arrays(h) -> dict[str, np.ndarray]:
    """Stack per-layer device graft h into (L, H, S, D) float32 arrays.

    GQA node payload is a list[dict] with keys k/v, each shape (1, H, S, D).
    """
    if h is None:
        raise ValueError("device payload h is None")
    k = np.stack([_tensor_to_numpy(d["k"])[0] for d in h]).astype(np.float32)
    v = np.stack([_tensor_to_numpy(d["v"])[0] for d in h]).astype(np.float32)
    return {"k": k, "v": v}


def payload_stats(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Per-layer and global key/value stats for P3 comparisons."""
    out: dict[str, Any] = {}
    for name, arr in arrays.items():
        # arr: (L, H, S, D)
        per_layer = []
        for li in range(arr.shape[0]):
            layer = arr[li]
            norms = np.linalg.norm(layer.reshape(layer.shape[0], -1), axis=1)
            per_layer.append({
                "layer": int(li),
                "mean_abs": float(np.mean(np.abs(layer))),
                "max_abs": float(np.max(np.abs(layer))),
                "mean": float(np.mean(layer)),
                "std": float(np.std(layer)),
                "min": float(np.min(layer)),
                "max": float(np.max(layer)),
                "key_norm_mean": float(np.mean(norms)),
                "key_norm_max": float(np.max(norms)),
                "key_norm_min": float(np.min(norms)),
            })
        flat = arr.reshape(-1)
        out[name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "mean_abs": float(np.mean(np.abs(flat))),
            "max_abs": float(np.max(np.abs(flat))),
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
            "per_layer": per_layer,
        }
    return out


def host_payload_control(host_payload) -> Optional[dict[str, Any]]:
    """Packed-host control stats (no dequant required for equality check)."""
    if host_payload is None:
        return None
    if not isinstance(host_payload, dict):
        return {"type": type(host_payload).__name__}
    meta: dict[str, Any] = {
        "keys": sorted(str(k) for k in host_payload.keys()),
        "format_version": host_payload.get("format_version"),
        "storage_bits": host_payload.get("storage_bits"),
        "group_size": host_payload.get("group_size"),
    }
    # Hash all array-like fields for byte identity.
    digester = hashlib.sha256()
    field_hashes = {}
    for key in sorted(host_payload.keys(), key=str):
        val = host_payload[key]
        if isinstance(val, np.ndarray):
            raw = np.ascontiguousarray(val).tobytes()
            field_hashes[str(key)] = {
                "shape": list(val.shape),
                "dtype": str(val.dtype),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            digester.update(str(key).encode())
            digester.update(raw)
        elif isinstance(val, (bytes, bytearray)):
            field_hashes[str(key)] = {
                "sha256": hashlib.sha256(val).hexdigest(),
                "nbytes": len(val),
            }
            digester.update(str(key).encode())
            digester.update(val)
        else:
            field_hashes[str(key)] = {"value": _jsonable(val)}
            digester.update(str(key).encode())
            digester.update(repr(val).encode())
    meta["field_hashes"] = field_hashes
    meta["aggregate_sha256"] = digester.hexdigest()
    return meta


class PagingTelemetry:
    """Session-scoped JSONL logger for payload lifecycle events."""

    def __init__(
        self,
        path: Optional[str] = None,
        snapshot_dir: Optional[str] = None,
    ):
        self.path = path
        self.snapshot_dir = snapshot_dir
        self.enabled = bool(path)
        self.snapshot_enabled = bool(snapshot_dir)
        self.turn: Optional[int] = None
        self.step: Optional[str] = None
        self._lock = threading.Lock()
        self._fh = None
        if self.enabled:
            parent = os.path.dirname(os.path.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            # Truncate on first open of a fresh session path.
            self._fh = open(path, "a", encoding="utf-8")
        if self.snapshot_enabled:
            os.makedirs(snapshot_dir, exist_ok=True)

    @classmethod
    def from_env(cls) -> "PagingTelemetry":
        return cls(
            path=_env_path(_ENV_PAGING_PATH),
            snapshot_dir=_env_path(_ENV_SNAPSHOT_DIR),
        )

    def set_context(
        self,
        turn: Optional[int] = None,
        step: Optional[str] = None,
    ) -> None:
        if not (self.enabled or self.snapshot_enabled):
            return
        if turn is not None:
            self.turn = int(turn)
        if step is not None:
            self.step = str(step)

    def log(self, kind: str, node_id: int, **fields: Any) -> None:
        if not self.enabled or self._fh is None:
            return
        rec = {
            "schema": "grm.paging_telemetry.v1",
            "ts": time.time(),
            "kind": str(kind),
            "node_id": int(node_id),
            "turn": self.turn,
            "step": self.step,
        }
        for key, value in fields.items():
            rec[key] = _jsonable(value)
        line = json.dumps(rec, sort_keys=True)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                finally:
                    self._fh = None

    def snapshot_nodes(
        self,
        arena: Any,
        node_ids: Iterable[int],
        *,
        turn: int,
        label: str,
        extra: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Dump device (and packed-host control) payload for each node id.

        Read-only: clones device tensors to host numpy; never mutates grafts.
        """
        if not self.snapshot_enabled or not self.snapshot_dir:
            return []
        out_meta: list[dict[str, Any]] = []
        turn_dir = Path(self.snapshot_dir) / f"turn_{int(turn):04d}_{label}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        for raw_id in node_ids:
            node_id = int(raw_id)
            if node_id < 0 or node_id >= len(arena.grafts):
                continue
            g = arena.grafts[node_id]
            rec: dict[str, Any] = {
                "schema": "grm.mount_snapshot.v1",
                "turn": int(turn),
                "label": label,
                "node_id": node_id,
                "device_present": g.get("h") is not None,
                "host_present": g.get("host_payload") is not None,
                "last_used": g.get("last_used"),
                "ntok": g.get("ntok"),
                "kind": g.get("kind"),
                "text_prefix": str(g.get("text", ""))[:160],
            }
            if extra:
                rec["extra"] = _jsonable(extra)
            host_ctrl = host_payload_control(g.get("host_payload"))
            if host_ctrl is not None:
                rec["host_payload_control"] = host_ctrl
            arrays = None
            if g.get("h") is not None:
                try:
                    arrays = device_payload_arrays(g["h"])
                    rec["device_stats"] = payload_stats(arrays)
                except Exception as err:  # keep probe path alive
                    rec["device_error"] = repr(err)
            # Always write meta; write arrays only when device present.
            meta_path = turn_dir / f"node_{node_id:04d}.json"
            meta_path.write_text(
                json.dumps(rec, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if arrays is not None:
                npz_path = turn_dir / f"node_{node_id:04d}_device.npz"
                np.savez_compressed(
                    npz_path,
                    k=arrays["k"].astype(np.float16),
                    v=arrays["v"].astype(np.float16),
                )
                rec["device_npz"] = str(npz_path)
            rec["meta_path"] = str(meta_path)
            out_meta.append(rec)
            if self.enabled:
                self.log(
                    "mount_snapshot",
                    node_id,
                    label=label,
                    device_present=rec["device_present"],
                    host_present=rec["host_present"],
                    device_sha256=(
                        (rec.get("device_stats") or {})
                        .get("k", {})
                        .get("sha256")
                    ),
                    host_sha256=(
                        (rec.get("host_payload_control") or {})
                        .get("aggregate_sha256")
                    ),
                    meta_path=str(meta_path),
                )
        index_path = turn_dir / "index.json"
        index_path.write_text(
            json.dumps(
                {
                    "turn": int(turn),
                    "label": label,
                    "nodes": [m["node_id"] for m in out_meta],
                    "extra": _jsonable(extra or {}),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return out_meta


# Module-level singleton bound by GraftRepository / driver. Default is a
# disabled instance so import-time and default-path call sites stay cheap.
_ACTIVE = PagingTelemetry()


def get_telemetry() -> PagingTelemetry:
    return _ACTIVE


def configure_from_env() -> PagingTelemetry:
    """Replace the process-wide telemetry sink from env (idempotent-ish)."""
    global _ACTIVE
    prev = _ACTIVE
    _ACTIVE = PagingTelemetry.from_env()
    # Close previous only if it was a different enabled sink.
    if prev is not _ACTIVE:
        try:
            prev.close()
        except Exception:
            pass
    return _ACTIVE


def set_context(turn: Optional[int] = None, step: Optional[str] = None) -> None:
    _ACTIVE.set_context(turn=turn, step=step)


def log_event(kind: str, node_id: int, **fields: Any) -> None:
    _ACTIVE.log(kind, node_id, **fields)
