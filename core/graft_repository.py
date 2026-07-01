"""GraftRepository — the full graft-repo memory system (Phases 3+4).

Wraps ArenaCache (persistent arena, 3-channel routing, grounded trips,
consolidate()) with the two layers that make it a complete memory:

  AUTO-LIBRARIAN  threshold-triggered consolidation. Aging turn-grafts
                  fold into digest grafts (E2: one-time ~0.89 coefficient,
                  then digests are fixed points); aging digests fold into
                  ERA grafts (digests-of-digests — E2 chained says safe).
                  Hierarchical descent keys flatten through generations so
                  an era node stays addressable per leaf topic. Retired
                  sources drop their VRAM (disk keeps them — cold storage).

  PERSISTENCE     a repository directory:
                    manifest.json   dialect, config, node metadata, rare
                                    tokens, lineage (sources), tags
                    index.npz       routing centroids (N, 256) fp32
                    nodes/NNNN.npz  per-node latent graft: c (L,S,256),
                                    kpe (L,S,32), fp16
                  Cross-session resume: load() rebuilds the routing index
                  and re-uploads ACTIVE nodes to device; retired nodes stay
                  on disk. The live cache is NOT persisted — by design,
                  history lives in the repository, sessions start with a
                  fresh window. The DIALECT GUARD refuses artifacts from a
                  different model (text survives; K/V never transfers).

API:
  repo = GraftRepository(model, encode, decode, path)   # create or resume
  repo.chat(user_text) -> answer                        # the hot path
  repo.add_turn(user, assistant)                        # scripted/observed
  repo.add_document(text, tags=...)                     # knowledge ingest
  repo.cull_graft_sections(idx, max_tokens=...)         # sectioned slicing
  repo.save() / repo.stats()
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
import threading
import time

import numpy as np

from core.graft_arena import ArenaCache
from core.grm_runtime import GRMRuntime


@dataclass(frozen=True)
class DialectDescriptor:
    """Stable model/dialect shape for the future C++ runtime boundary."""

    model_type: str
    num_layers: int
    hidden_dim: int
    payload_kind: str
    vals_per_tok_layer: int
    route_layer: int
    latent_rank: int = 0
    rope_dim: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    position_law: str = "rope"
    state_kind: str = "kv"
    graftability: str = "seat_remountable"
    remountable: bool = True
    composition: str = "multi_mount"

    @classmethod
    def from_model(cls, model, arena):
        cfg = model.config
        route_layer = int(getattr(arena, "route_layer", 0))
        profile = cls._infer_graftability_profile(cfg, arena)
        if hasattr(cfg, "kv_lora_rank"):
            latent_rank = int(cfg.kv_lora_rank)
            rope_dim = int(getattr(cfg, "qk_rope_head_dim", 0))
            vals = latent_rank + rope_dim
            return cls(type(model).__name__, int(cfg.num_layers),
                       int(cfg.hidden_dim), "mla", vals, route_layer,
                       latent_rank=latent_rank, rope_dim=rope_dim,
                       **profile)
        num_kv_heads = int(getattr(cfg, "num_kv_heads", 0))
        head_dim = int(getattr(cfg, "head_dim", 0))
        vals = int(getattr(arena, "VALS_PER_TOK_LAYER",
                           num_kv_heads * head_dim * 2))
        return cls(type(model).__name__, int(cfg.num_layers),
                   int(cfg.hidden_dim), "gqa", vals, route_layer,
                   num_kv_heads=num_kv_heads, head_dim=head_dim,
                   **profile)

    @classmethod
    def _infer_graftability_profile(cls, cfg, arena):
        explicit = {
            "position_law": getattr(arena, "POSITION_LAW", None),
            "state_kind": getattr(arena, "STATE_KIND", None),
            "graftability": getattr(arena, "GRAFTABILITY", None),
            "remountable": getattr(arena, "REMOUNTABLE", None),
            "composition": getattr(arena, "COMPOSITION", None),
        }
        if any(v is not None for v in explicit.values()):
            base = cls._default_profile(cfg)
            for k, v in explicit.items():
                if v is not None:
                    base[k] = cls._as_bool(v) if k == "remountable" else str(v)
            return base
        return cls._default_profile(cfg)

    @staticmethod
    def _as_bool(value):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    @staticmethod
    def _default_profile(cfg):
        if str(getattr(cfg, "position_embedding_type", "")).lower() in (
                "absolute", "learned_absolute"):
            return {
                "position_law": "learned_absolute",
                "state_kind": "kv",
                "graftability": "same_position_restore",
                "remountable": False,
                "composition": "prefix_restore_only",
            }
        if getattr(cfg, "alibi", False) or getattr(cfg, "use_alibi", False):
            return {
                "position_law": "relative_bias",
                "state_kind": "kv",
                "graftability": "bias_recomputed_remountable",
                "remountable": True,
                "composition": "multi_mount",
            }
        if hasattr(cfg, "full_attention_interval") and hasattr(cfg, "conv_kernel"):
            return {
                "position_law": "rope_attention_plus_recurrent_state",
                "state_kind": "hybrid_kv_recurrent",
                "graftability": "prefix_restore_only",
                "remountable": False,
                "composition": "single_prefix_state",
            }
        if hasattr(cfg, "sliding_window") and hasattr(cfg, "num_global_kv_heads"):
            return {
                "position_law": "rope_mixed_sliding_global",
                "state_kind": "mixed_window_kv",
                "graftability": "window_limited_remountable",
                "remountable": True,
                "composition": "bounded_window_multi_mount",
            }
        if hasattr(cfg, "kv_lora_rank"):
            return {
                "position_law": "rope_partial_mla",
                "state_kind": "mla_latent_plus_rope",
                "graftability": "seat_remountable",
                "remountable": True,
                "composition": "multi_mount",
            }
        return {
            "position_law": "rope_full_kv",
            "state_kind": "kv",
            "graftability": "seat_remountable",
            "remountable": True,
            "composition": "multi_mount",
        }

    @property
    def dialect_id(self):
        if self.payload_kind == "mla":
            tail = f"r{self.latent_rank}"
        else:
            tail = f"g{self.num_kv_heads}x{self.head_dim}"
        return f"{self.model_type}:{self.num_layers}x{self.hidden_dim}:{tail}"

    def to_json(self):
        return asdict(self)


class GraftRepository:
    # librarian thresholds: consolidate when this many ACTIVE nodes of a
    # kind are older than the live window; how many to fold per pass.
    # Era folding is ON and safe BY CONSTRUCTION: eras are INDEX nodes —
    # the trips ladder expands them to their child digests at the primary
    # attempt, so era text is routed into but never read (2026-06-10 it
    # was read, and both list-form and prose-form era texts corrupted
    # relations; descent + relational first-gen digests took the
    # era-folded 42-turn gate from 3/8 to 8/8).
    TURNS_HIGH, TURNS_FOLD = 8, 4
    DIGESTS_HIGH, DIGESTS_FOLD = 6, 3
    WAL_DURABILITY_MODES = {"session_safe", "project_safe", "durable_strict"}
    DURABILITY_MODES = WAL_DURABILITY_MODES | {"volatile", "volatile_fast"}

    def __init__(self, model, encode, decode, path, autosave=True,
                 vram_budget_mb=None, librarian_mode="inline",
                 arena_cls=ArenaCache, durability_mode="session_safe",
                 wal_enabled=None, native_store=None, native_lib_path=None,
                 native_enabled=False, extractor=None,
                 extraction_write_threshold=0.95,
                 extraction_error_policy="record", **arena_kw):
        self.path = path
        self.autosave = autosave
        self.durability_mode = self._normalize_durability_mode(durability_mode)
        self._wal_enabled_override = wal_enabled is not None
        self.wal_enabled = (
            self._wal_enabled_for_mode(self.durability_mode)
            if wal_enabled is None else bool(wal_enabled))
        self._flush_lock = threading.RLock()
        self._flush_thread = None
        self._flush_error = None
        self._wal_lsn = 0
        self.review_buffer = []
        self.fold_history = []
        self.extractor = extractor
        self.extraction_write_threshold = float(extraction_write_threshold)
        self.extraction_error_policy = extraction_error_policy
        self.last_extraction_results = []
        self.last_extraction_error = None
        # device-byte budget for node tensors; least-recently-MOUNTED saved
        # nodes spill to cold storage above it (node_loader reloads on
        # demand — the descent machinery). None = unbounded.
        self.vram_budget = vram_budget_mb * 1024 * 1024 if vram_budget_mb else None
        # "inline": folds run inside chat/add_turn when thresholds trip
        # (simple; a fold stalls that turn ~3s). "deferred": the hot path
        # NEVER folds — due work is computed statelessly and executed by
        # idle() between turns (host loop / background mission drives it);
        # a 2x backpressure threshold folds inline only as a last resort.
        # One GPU = no true concurrency: background means BETWEEN turns.
        self.librarian_mode = librarian_mode
        self.arena = arena_cls(model, encode, decode, **arena_kw)
        self.dialect_desc = DialectDescriptor.from_model(model, self.arena)
        self.dialect = self.dialect_desc.dialect_id
        self.dirty_nodes = {}
        self.native_store = native_store
        self._own_native_store = False
        self._native_node_ids = {}
        self._native_checkpoint_loaded = False
        if self.native_store is None and (native_enabled or native_lib_path):
            self.native_store = self._open_native_store(native_lib_path)
            self._own_native_store = True
        self.arena.native_store = self.native_store
        if self.native_store is not None:
            self._native_configure_arena()
        # descent re-mounts retired children from cold storage on demand
        self.arena.node_loader = self._load_node
        self._ensure_repo_dirs()
        if os.path.exists(os.path.join(path, "manifest.json")):
            self.load()
        else:
            self.recovered_wal = self._read_wal()
            self.replayed_config_lsn = self._apply_config_wal_records(
                self.recovered_wal)
            self.recovered_nodes = self._recover_wal_summary(
                self.recovered_wal)
            self._rehydrate_from_wal(self.recovered_nodes)
            self._sync_lifecycle()
            self.dirty_nodes.clear()
        self.runtime = GRMRuntime(self)

    def close(self):
        if self._own_native_store and self.native_store is not None:
            self.native_store.close()
        self.native_store = None
        self.arena.native_store = None

    @classmethod
    def _normalize_durability_mode(cls, mode):
        mode = str(mode or "session_safe").strip().lower()
        mode = mode.replace("-", "_").replace(" ", "_")
        if mode == "volatile":
            return "volatile"
        if mode == "volatile_fast":
            return "volatile_fast"
        if mode in cls.DURABILITY_MODES:
            return mode
        raise ValueError(f"unknown durability mode {mode!r}")

    @classmethod
    def _wal_enabled_for_mode(cls, mode):
        return cls._normalize_durability_mode(mode) in cls.WAL_DURABILITY_MODES

    def _set_durability_mode_fields(self, mode, wal_enabled=None):
        mode = self._normalize_durability_mode(mode)
        self.durability_mode = mode
        if wal_enabled is None:
            wal_enabled = self._wal_enabled_for_mode(mode)
        if not self._wal_enabled_override:
            self.wal_enabled = bool(wal_enabled)
        return mode

    def set_durability_mode(self, mode):
        mode = self._normalize_durability_mode(mode)
        old_mode = self.durability_mode
        old_wal = bool(self.wal_enabled)
        new_wal = self._wal_enabled_for_mode(mode)
        if old_wal:
            self._append_wal("CONFIG", durability_mode=mode,
                             wal_enabled=bool(new_wal),
                             previous_durability_mode=old_mode)
        self.durability_mode = mode
        if not self._wal_enabled_override:
            self.wal_enabled = bool(new_wal)
        if not old_wal and self.wal_enabled:
            self._append_wal("CONFIG", durability_mode=mode,
                             wal_enabled=bool(self.wal_enabled),
                             previous_durability_mode=old_mode)
        return {"durability_mode": self.durability_mode,
                "previous_durability_mode": old_mode,
                "wal_enabled": bool(self.wal_enabled)}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ----------------------------------------------------------- hot path
    def chat(self, user_text, ngen=64, max_trips=2):
        return self.runtime.chat(user_text, ngen=ngen, max_trips=max_trips)

    def add_turn(self, user, assistant):
        """Deposit an already-complete turn (scripted or externally run)."""
        self.runtime.add_turn(user, assistant)

    def add_document(self, text, tags=()):
        before = self._snapshot_state()
        idx = self.arena.deposit(text)
        g = self.arena.grafts[idx]
        g["kind"] = "doc"
        g["tags"] = list(tags)
        g["provenance"] = [self._provenance("doc_span", idx)]
        self._mark_mutations(before)
        self._page()
        return idx

    @staticmethod
    def _cull_boundary_set(boundary):
        if boundary is None:
            return {"blank"}
        if isinstance(boundary, str):
            name = boundary.lower()
            if name in ("paragraph", "paragraphs"):
                return {"blank"}
            if name in ("turn", "turns"):
                return {"blank", "speaker"}
            if name in ("heading", "headings"):
                return {"blank", "heading"}
            if name in ("section", "sections"):
                return {"blank", "heading", "speaker"}
            raise ValueError(f"unknown cull boundary strategy {boundary!r}")
        out = {str(v).lower() for v in boundary}
        allowed = {"blank", "heading", "speaker"}
        unknown = out - allowed
        if unknown:
            raise ValueError(f"unknown cull boundaries {sorted(unknown)!r}")
        return out or {"blank"}

    @staticmethod
    def _is_speaker_boundary(line):
        if ":" not in line:
            return False
        speaker = line.split(":", 1)[0].strip().lower()
        return speaker in {
            "user", "assistant", "system", "developer", "tool", "human",
            "ai",
        }

    def _section_text_chunks(self, text, boundary="section"):
        boundaries = self._cull_boundary_set(boundary)
        chunks = []
        cur = []

        def flush():
            if cur:
                chunk = "\n".join(cur).strip()
                if chunk:
                    chunks.append(chunk)
                cur.clear()

        for raw in str(text).splitlines():
            line = raw.strip()
            if not line:
                if "blank" in boundaries:
                    flush()
                continue
            is_heading = "heading" in boundaries and line.startswith("#")
            is_speaker = (
                "speaker" in boundaries and self._is_speaker_boundary(line))
            if is_heading or is_speaker:
                flush()
                cur.append(line)
                continue
            cur.append(line)
        flush()
        if chunks:
            return chunks
        text = str(text).strip()
        return [text] if text else []

    @staticmethod
    def _cap_cull_spans(spans, max_tokens=None):
        if max_tokens is None:
            return list(spans)
        max_tokens = int(max_tokens)
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        out = []
        for start, end in spans:
            cursor = int(start)
            end = int(end)
            while end - cursor > max_tokens:
                out.append((cursor, cursor + max_tokens))
                cursor += max_tokens
            if end > cursor:
                out.append((cursor, end))
        return out

    def _section_cull_spans(self, text, ntok, *, max_tokens=None,
                            boundary="section"):
        ntok = int(ntok)
        if ntok <= 0:
            return []
        spans = []
        cursor = 0
        for chunk in self._section_text_chunks(text, boundary=boundary):
            n = len(self.arena.encode(chunk))
            if n <= 0:
                continue
            start = cursor
            end = min(ntok, cursor + n)
            if end > start:
                spans.append((start, end))
            cursor = end
            if cursor >= ntok:
                break
        if cursor < ntok:
            spans.append((cursor, ntok))
        if not spans:
            spans = [(0, ntok)]
        return self._cap_cull_spans(spans, max_tokens=max_tokens)

    def plan_cull_sections(self, idx, *, max_tokens=None,
                           boundary="section"):
        idx = int(idx)
        if idx < 0 or idx >= len(self.arena.grafts):
            raise IndexError("unknown graft id")
        parent = self.arena.grafts[idx]
        ntok = int(parent.get("ntok", 0))
        if ntok <= 0:
            raise ValueError("cannot cull a graft with no token length")
        spans = self._section_cull_spans(
            parent.get("text", ""), ntok, max_tokens=max_tokens,
            boundary=boundary)
        return self._normalize_cull_spans(ntok, spans=spans,
                                          retire_parent=True)

    def cull_graft_sections(self, idx, *, max_tokens=None,
                            boundary="section", retire_parent=True,
                            kind=None, tags=(), recompute_route=True):
        spans = self.plan_cull_sections(
            idx, max_tokens=max_tokens, boundary=boundary)
        return self.cull_graft(
            idx, spans=spans, retire_parent=retire_parent, kind=kind,
            tags=tags, recompute_route=recompute_route)

    def _normalize_cull_spans(self, ntok, max_tokens=None, spans=None,
                              retire_parent=True):
        if spans is None:
            if max_tokens is None:
                raise ValueError("cull_graft requires max_tokens or spans")
            max_tokens = int(max_tokens)
            if max_tokens <= 0:
                raise ValueError("max_tokens must be positive")
            spans = [(i, min(i + max_tokens, ntok))
                     for i in range(0, ntok, max_tokens)]
        out = []
        for start, end in spans:
            start, end = int(start), int(end)
            if start < 0 or end > ntok or end <= start:
                raise ValueError(f"invalid cull span {(start, end)} for "
                                 f"{ntok} tokens")
            out.append((start, end))
        if not out:
            raise ValueError("cull_graft produced no spans")
        if retire_parent:
            cursor = 0
            for start, end in sorted(out):
                if start != cursor:
                    raise ValueError("retiring a parent requires cull spans "
                                     "to cover every token without gaps")
                cursor = end
            if cursor != ntok:
                raise ValueError("retiring a parent requires cull spans to "
                                 "cover the full graft")
        return out

    def _payload_token_axis(self, key, arr, ntok):
        declared = {k: dim for k, dim in getattr(self.arena, "PAYLOAD", ())}
        candidates = [i for i, n in enumerate(arr.shape) if int(n) == ntok]
        dim = declared.get(key)
        if dim in candidates:
            return dim
        if dim is not None and dim - 1 in candidates:
            return dim - 1
        if len(candidates) == 1:
            return candidates[0]
        if dim is not None:
            raise ValueError(f"cannot identify token axis for payload {key!r}")
        return None

    def _slice_host_payload(self, payload, ntok, start, end, native_node_id=None):
        out = {}
        for key, value in payload.items():
            arr = np.ascontiguousarray(value)
            axis = self._payload_token_axis(key, arr, ntok)
            if axis is None:
                out[key] = np.ascontiguousarray(arr.copy())
                continue
            if (native_node_id is not None and self.native_store is not None
                    and hasattr(self.native_store, "slice_tensor")):
                try:
                    out[key] = self.native_store.slice_tensor(
                        native_node_id, key, axis, start, end - start)
                    continue
                except RuntimeError:
                    pass
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(start, end)
            out[key] = np.ascontiguousarray(arr[tuple(sl)])
        return out

    def _decode_token_span(self, text, start, end):
        try:
            ids = list(self.arena.encode(text))
            if len(ids) >= end:
                decoded = self.arena.decode(ids[start:end]).strip()
                if decoded:
                    return decoded
        except Exception:
            pass
        words = str(text).split()
        if len(words) >= end:
            return " ".join(words[start:end])
        return f"{text} [tokens {start}:{end}]"

    def _cull_child_centroid(self, parent, text, recompute_route):
        if recompute_route and hasattr(self.arena, "_node_key"):
            try:
                return np.asarray(self.arena._node_key(text),
                                  dtype=np.float32)
            except Exception:
                pass
        return np.asarray(parent.get("cent"), dtype=np.float32).copy()

    @staticmethod
    def _append_unique(values, extra):
        out = []
        for value in list(values or ()) + list(extra or ()):
            if value not in out:
                out.append(value)
        return out

    def cull_graft(self, idx, *, max_tokens=None, spans=None,
                   retire_parent=True, kind=None, tags=(),
                   recompute_route=True):
        return self.runtime.cull_graft(
            idx, max_tokens=max_tokens, spans=spans,
            retire_parent=retire_parent, kind=kind, tags=tags,
            recompute_route=recompute_route)

    def _cull_graft_direct(self, idx, *, max_tokens=None, spans=None,
                           retire_parent=True, kind=None, tags=(),
                           recompute_route=True):
        """Split one long graft into shorter child grafts.

        Child nodes receive RAM payload slices and lineage back to the parent.
        By default the parent is retired, leaving it as cold evidence while the
        shorter children become the active route/mount surfaces.
        """
        idx = int(idx)
        if idx < 0 or idx >= len(self.arena.grafts):
            raise IndexError("unknown graft id")
        parent = self.arena.grafts[idx]
        ntok = int(parent.get("ntok", 0))
        if ntok <= 0:
            raise ValueError("cannot cull a graft with no token length")
        spans = self._normalize_cull_spans(
            ntok, max_tokens=max_tokens, spans=spans,
            retire_parent=retire_parent)
        before = self._snapshot_state()
        self._ensure_host_payload(idx, parent)
        native_node_id = None
        if self.native_store is not None:
            native_node_id = self._native_sync_node(idx)
        parent_meta = parent.setdefault(
            "metadata", self._default_metadata(parent))
        child_kind = kind or parent.get("kind", "doc")
        child_ids = []
        for order, (start, end) in enumerate(spans):
            text = self._decode_token_span(parent.get("text", ""), start, end)
            payload = self._slice_host_payload(
                parent["host_payload"], ntok, start, end,
                native_node_id=native_node_id)
            meta = {
                "kind": child_kind,
                "durability": parent_meta.get("durability", "project"),
                "mutability": parent_meta.get("mutability", "stable"),
                "scope": parent_meta.get("scope", "project"),
                "write_intent": parent_meta.get("write_intent", "observed"),
                "confidence": parent_meta.get("confidence", 1.0),
                "source_grafts": self._append_unique(
                    parent_meta.get("source_grafts", ()), (idx,)),
                "supersedes": [idx] if retire_parent else [],
                "active": True,
                "culled_from": idx,
                "token_start": start,
                "token_end": end,
                "cull_index": order,
                "cull_total": len(spans),
            }
            child = {
                "kind": child_kind,
                "text": text,
                "ntok": end - start,
                "sources": [idx],
                "retired": False,
                "no_fold": False,
                "tags": self._append_unique(parent.get("tags", ()), tags),
                "rare": self.arena._rare_tokens(text),
                "cent": self._cull_child_centroid(parent, text,
                                                  recompute_route),
                "metadata": meta,
                "provenance": [self._provenance(
                    "cull_span", len(self.arena.grafts),
                    source_graft=idx, token_start=start, token_end=end)],
                "host_payload": payload,
                "host_present": True,
                "device_present": False,
                "dirty": True,
                "durable": False,
                "cold_only": False,
                "payload_pending": False,
                "h": None,
            }
            child_idx = len(self.arena.grafts)
            self._ensure_lifecycle(child_idx, child)
            self.arena.grafts.append(child)
            self._mark_dirty(child_idx, payload=True, metadata=True)
            child_ids.append(child_idx)

        parent_meta["culled_into"] = list(child_ids)
        if retire_parent:
            parent_meta["active"] = False
            parent_meta["superseded_by"] = list(child_ids)
            parent["retired"] = True
        self._mark_mutations(before)
        if not retire_parent:
            self._mark_dirty(idx, payload=False, metadata=True)
            self._append_wal("NODE_META", node_id=idx,
                             metadata=parent.get("metadata", {}),
                             state=list(self._state_tuple(parent)))
        self._rebuild_child_keys()
        self._free_retired()
        self._page()
        return {"action": "cull_graft", "parent": idx,
                "children": list(child_ids),
                "retired_parent": bool(retire_parent)}

    def split_graft(self, *args, **kwargs):
        return self.cull_graft(*args, **kwargs)

    def remember(self, text, durability="project", mutability="stable",
                 scope="project", kind="fact", write_intent="user_asserted",
                 confidence=1.0, tags=(), metadata=None):
        """Create an explicit semantic memory node.

        The current Phase-1 bridge still harvests a normal graft payload via
        the arena. The metadata/lifecycle shape is the contract the C++ host
        store will later preserve while taking over RAM payload ownership.
        """
        before = self._snapshot_state()
        idx = self.arena.deposit(text)
        g = self.arena.grafts[idx]
        g["kind"] = kind
        g["tags"] = list(tags)
        g["provenance"] = [self._provenance("fact_span", idx)]
        self._ensure_lifecycle(idx, g)
        g["metadata"].update({
            "kind": kind,
            "durability": durability,
            "mutability": mutability,
            "scope": scope,
            "write_intent": write_intent,
            "confidence": float(confidence),
            "active": True,
        })
        if metadata:
            g["metadata"].update(metadata)
        self._mark_mutations(before)
        self._page()
        return idx

    @staticmethod
    def _normalize_cull_command_boundary(name):
        name = str(name).strip().lower()
        if name in ("section", "sections"):
            return "section"
        if name in ("paragraph", "paragraphs"):
            return "paragraph"
        if name in ("turn", "turns"):
            return "turn"
        if name in ("heading", "headings"):
            return "heading"
        raise ValueError(f"unknown cull boundary strategy {name!r}")

    @staticmethod
    def _parse_cull_command_python(original, low):
        words = low.replace(",", " ").replace(":", " ").replace(
            "=", " ").split()
        if len(words) < 3 or words[0] not in ("cull", "split"):
            return None
        if words[1] != "graft":
            return None
        try:
            node_id = int(words[2])
        except ValueError as exc:
            raise ValueError("cull graft requires a numeric graft id") from exc
        if node_id < 0:
            raise ValueError("cull graft id must be nonnegative")

        plan = {"action": "cull_graft", "node_id": node_id}
        cursor = 3
        while cursor < len(words):
            word = words[cursor]
            if word in ("into", "by"):
                cursor += 1
                if cursor >= len(words):
                    raise ValueError("cull graft boundary is missing")
                plan["boundary"] = GraftRepository._normalize_cull_command_boundary(
                    words[cursor])
                cursor += 1
                continue
            if word in ("section", "sections", "paragraph", "paragraphs",
                        "turn", "turns", "heading", "headings"):
                plan["boundary"] = GraftRepository._normalize_cull_command_boundary(
                    word)
                cursor += 1
                continue
            if word in ("max", "max_tokens", "max-token", "max-tokens"):
                if word == "max":
                    cursor += 1
                    if cursor < len(words) and words[cursor] in (
                            "token", "tokens"):
                        cursor += 1
                else:
                    cursor += 1
                if cursor >= len(words):
                    raise ValueError("cull graft max tokens is missing")
                try:
                    max_tokens = int(words[cursor])
                except ValueError as exc:
                    raise ValueError(
                        "cull graft max tokens must be numeric") from exc
                if max_tokens <= 0:
                    raise ValueError("cull graft max tokens must be positive")
                plan["max_tokens"] = max_tokens
                cursor += 1
                continue
            raise ValueError(f"unknown cull graft option {word!r}")
        if "boundary" not in plan and "max_tokens" not in plan:
            raise ValueError(
                "cull graft requires max tokens or a boundary strategy")
        return plan

    @staticmethod
    def _command_body(original, prefix):
        return original[len(prefix):].strip()

    @staticmethod
    def _parse_metadata_command_python(original, low):
        table = (
            ("pin memory:", "pin_memory", {"pinned": True}),
            ("pin this:", "pin_memory", {"pinned": True}),
            ("unpin memory:", "unpin_memory", {"pinned": False}),
            ("unpin this:", "unpin_memory", {"pinned": False}),
            ("mark memory mutable:", "mark_mutable",
             {"mutability": "mutable"}),
            ("mark this as mutable:", "mark_mutable",
             {"mutability": "mutable"}),
            ("mark memory stable:", "mark_stable",
             {"mutability": "stable"}),
            ("mark this as stable:", "mark_stable",
             {"mutability": "stable"}),
            ("mark memory immutable:", "mark_immutable",
             {"mutability": "immutable"}),
            ("mark this as immutable:", "mark_immutable",
             {"mutability": "immutable"}),
        )
        for prefix, command, metadata in table:
            if low.startswith(prefix):
                return {"action": "update_memory_metadata",
                        "command": command,
                        "query": GraftRepository._command_body(
                            original, prefix),
                        "metadata": dict(metadata)}
        return None

    @staticmethod
    def _parse_read_memory_command_python(original, low):
        table = (
            ("show memory about:", "show_memory"),
            ("why do you remember that:", "why_memory"),
            ("why do you remember:", "why_memory"),
        )
        for prefix, action in table:
            if low.startswith(prefix):
                return {"action": action,
                        "query": GraftRepository._command_body(
                            original, prefix)}
        return None

    @staticmethod
    def _parse_mode_command_python(low):
        table = (
            ("switch to volatile mode", "volatile"),
            ("switch to volatile-fast mode", "volatile_fast"),
            ("switch to volatile fast mode", "volatile_fast"),
            ("switch to session-safe mode", "session_safe"),
            ("switch to session safe mode", "session_safe"),
            ("switch to project-safe mode", "project_safe"),
            ("switch to project safe mode", "project_safe"),
            ("switch to durable-strict mode", "durable_strict"),
            ("switch to durable strict mode", "durable_strict"),
        )
        for prefix, mode in table:
            if low == prefix:
                return {"action": "set_durability_mode",
                        "durability_mode": mode}
        return None

    @staticmethod
    def _parse_memory_command_python(text):
        original = text.strip()
        low = original.lower()
        cull = GraftRepository._parse_cull_command_python(original, low)
        if cull is not None:
            return cull
        metadata = GraftRepository._parse_metadata_command_python(original, low)
        if metadata is not None:
            return metadata
        read = GraftRepository._parse_read_memory_command_python(original, low)
        if read is not None:
            return read
        mode = GraftRepository._parse_mode_command_python(low)
        if mode is not None:
            return mode
        table = (
            ("remember permanently:", dict(durability="permanent",
                                           scope="user", kind="fact",
                                           flush_immediately=True)),
            ("remember this for the project:", dict(durability="project",
                                                    scope="project",
                                                    kind="task_state")),
            ("remember this for this session:", dict(durability="session",
                                                     scope="session",
                                                     kind="task_state")),
            ("this is temporary:", dict(durability="volatile",
                                        mutability="ephemeral",
                                        scope="session",
                                        kind="task_state")),
        )
        for prefix, opts in table:
            if low.startswith(prefix):
                out = {"action": "remember",
                       "body": original[len(prefix):].strip()}
                out.update(opts)
                return out
        if low.startswith("forget:"):
            return {"action": "forget",
                    "query": original.split(":", 1)[1].strip()}
        if low.startswith(("correct memory:", "update memory:")):
            body = original.split(":", 1)[1].strip()
            if "=>" not in body:
                return {"action": "review", "body": body,
                        "reason": "correction missing => separator"}
            query, replacement = [p.strip() for p in body.split("=>", 1)]
            return {"action": "correct", "query": query,
                    "replacement": replacement}
        if low.startswith("do not remember this"):
            return {"action": "ignore"}
        if low.startswith("flush memory now"):
            return {"action": "flush"}
        raise ValueError(f"unknown memory command: {text!r}")

    def _parse_memory_command(self, text):
        """Return a structured memory-command plan.

        Native-backed repositories use the C++ parser so command grammar is a
        stable runtime boundary. Python remains the operation/policy executor.
        """
        if self.native_store is not None and hasattr(
                self.native_store, "parse_memory_command"):
            try:
                return self.native_store.parse_memory_command(text)
            except RuntimeError as exc:
                if "unavailable" not in str(exc):
                    raise ValueError(f"unknown memory command: {text!r}") from exc
        return self._parse_memory_command_python(text)

    def apply_memory_command(self, text):
        """Apply an explicit chat memory command from the runtime plan."""
        return self.runtime.apply_memory_command(text)

    def forget(self, query):
        q = query.lower()
        count = 0
        before = self._snapshot_state()
        for i, g in enumerate(self.arena.grafts):
            if q and q not in g.get("text", "").lower():
                continue
            meta = g.setdefault("metadata", self._default_metadata(g))
            if not meta.get("active", True):
                continue
            meta["active"] = False
            meta["superseded_by"] = []
            g["retired"] = True
            count += 1
            self._mark_dirty(i, payload=False, metadata=True)
            self._append_wal("NODE_FORGET", node_id=i, query=query)
        if count:
            self._mark_mutations(before)
        return count

    def update_memory_metadata(self, query, updates):
        updates = dict(updates or {})
        if not updates:
            return {"count": 0, "node_ids": []}
        q = str(query).lower()
        changed = []
        before = self._snapshot_state()
        for i, g in enumerate(self.arena.grafts):
            if q and q not in g.get("text", "").lower():
                continue
            meta = g.setdefault("metadata", self._default_metadata(g))
            if not meta.get("active", True) or g.get("retired"):
                continue
            meta.update(updates)
            changed.append(i)
            self._mark_dirty(i, payload=False, metadata=True)
            self._append_wal("NODE_META", node_id=i, metadata=meta,
                             state=list(self._state_tuple(g)))
        if changed:
            self._mark_mutations(before)
        return {"count": len(changed), "node_ids": changed}

    def correct_memory(self, query, replacement, **metadata):
        supersedes = []
        q = query.lower()
        before = self._snapshot_state()
        for i, g in enumerate(self.arena.grafts):
            if q and q in g.get("text", "").lower():
                meta = g.setdefault("metadata", self._default_metadata(g))
                if meta.get("active", True):
                    meta["active"] = False
                    g["retired"] = True
                    supersedes.append(i)
                    self._mark_dirty(i, payload=False, metadata=True)
        meta = dict(metadata)
        meta["supersedes"] = supersedes
        idx = self.remember(replacement, metadata=meta,
                            write_intent="user_asserted")
        for i in supersedes:
            self.arena.grafts[i]["metadata"]["superseded_by"] = [idx]
            self._mark_dirty(i, payload=False, metadata=True)
        self._native_apply_revision(idx, supersedes)
        self._append_wal("MEMORY_CORRECT", query=query, replacement=replacement,
                         supersedes=supersedes, node_id=idx)
        self._mark_mutations(before)
        return idx

    @staticmethod
    def _norm_fact_field(value):
        return str(value).strip().lower() if value is not None else ""

    @classmethod
    def _norm_fact_scope(cls, value):
        return cls._norm_fact_field(value) or "project"

    @classmethod
    def _candidate_scope_conflicts(cls, candidate, metadata):
        candidate_scope = cls._norm_fact_scope(candidate.get("scope", "project"))
        existing_scope = cls._norm_fact_scope(metadata.get("scope", "project"))
        return candidate_scope == existing_scope

    @staticmethod
    def _parse_fact_time(value):
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), timezone.utc)
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return False
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @classmethod
    def _norm_fact_time_value(cls, value):
        parsed = cls._parse_fact_time(value)
        if parsed is None:
            return None
        if parsed is False:
            return ("invalid", str(value).strip())
        return parsed.isoformat()

    @classmethod
    def _fact_effective_now(cls, metadata, now=None):
        now = now or datetime.now(timezone.utc)
        valid_from = cls._parse_fact_time(metadata.get("valid_from"))
        expires_at = cls._parse_fact_time(metadata.get("expires_at"))
        if valid_from is False or expires_at is False:
            return True
        if valid_from is not None and valid_from > now:
            return False
        if expires_at is not None and expires_at <= now:
            return False
        return True

    @classmethod
    def _candidate_time_conflicts(cls, candidate, metadata):
        return (cls._fact_effective_now(candidate)
                and cls._fact_effective_now(metadata))

    @staticmethod
    def _candidate_target_ids(candidate):
        explicit = (candidate.get("target_node_ids")
                    or candidate.get("target_node_id")
                    or candidate.get("targets")
                    or candidate.get("supersedes")
                    or ())
        if isinstance(explicit, (str, int, float)):
            explicit = (explicit,)
        out = []
        for node_id in explicit:
            try:
                out.append(int(node_id))
            except (TypeError, ValueError):
                continue
        return out

    def _candidate_expire_targets(self, candidate):
        explicit = self._candidate_target_ids(candidate)
        if explicit:
            out = []
            for i in explicit:
                if i < 0 or i >= len(self.arena.grafts):
                    continue
                g = self.arena.grafts[i]
                meta = g.get("metadata", self._default_metadata(g))
                if meta.get("active", True) and not g.get("retired"):
                    out.append(i)
            return list(dict.fromkeys(out))

        subject = self._norm_fact_field(candidate.get("subject"))
        predicate = self._norm_fact_field(candidate.get("predicate"))
        value = self._norm_fact_field(candidate.get("value"))
        if not (subject and predicate and value):
            return []

        out = []
        for i, g in enumerate(self.arena.grafts):
            meta = g.get("metadata", self._default_metadata(g))
            if not meta.get("active", True) or g.get("retired"):
                continue
            if self._norm_fact_field(meta.get("subject")) != subject:
                continue
            if self._norm_fact_field(meta.get("predicate")) != predicate:
                continue
            if self._norm_fact_field(meta.get("value")) != value:
                continue
            if not self._candidate_scope_conflicts(candidate, meta):
                continue
            if not self._candidate_time_conflicts(candidate, meta):
                continue
            out.append(i)
        return out

    def _candidate_supersede_targets(self, candidate):
        explicit = self._candidate_target_ids(candidate)
        out = []
        for i in explicit:
            if i < 0 or i >= len(self.arena.grafts):
                continue
            g = self.arena.grafts[i]
            meta = g.get("metadata", self._default_metadata(g))
            if meta.get("active", True) and not g.get("retired"):
                out.append(i)
        return list(dict.fromkeys(out))

    def _expire_extraction_targets(self, targets, text):
        expired_at = datetime.now(timezone.utc).isoformat()
        expired = []
        for i in targets:
            g = self.arena.grafts[int(i)]
            meta = g.setdefault("metadata", self._default_metadata(g))
            meta["active"] = False
            meta["expired_at"] = expired_at
            meta["expired_by"] = text
            g["retired"] = True
            self._mark_dirty(int(i), payload=False, metadata=True)
            expired.append(int(i))
        if expired:
            self._append_wal("MEMORY_EXTRACT_EXPIRE",
                             expired=list(expired), text=text,
                             expired_at=expired_at)
        return expired

    def _candidate_text(self, candidate, source_text=None):
        if candidate.get("text"):
            return str(candidate["text"]).strip()
        if source_text is not None and candidate.get("text_span"):
            start, end = candidate["text_span"]
            return str(source_text)[int(start):int(end)].strip()
        parts = [candidate.get("subject"), candidate.get("predicate"),
                 candidate.get("value")]
        text = " ".join(str(p).strip() for p in parts if p is not None)
        if text:
            return text
        raise ValueError("memory candidate has no text or semantic fields")

    def _candidate_metadata(self, candidate, source_turns=(),
                            source_grafts=()):
        reserved = {"action", "candidate_type", "kind", "text", "scope",
                    "durability", "mutability", "write_intent", "confidence",
                    "active", "supersedes", "target_node_id",
                    "target_node_ids", "targets"}
        meta = {k: v for k, v in dict(candidate.get("metadata", {}) or {}).items()
                if k not in reserved}
        keys = ("subject", "predicate", "value", "valid_from", "expires_at")
        meta.update({k: candidate[k] for k in keys if k in candidate})
        if source_turns or candidate.get("source_turns"):
            meta["source_turns"] = list(candidate.get("source_turns",
                                                     source_turns))
        if source_grafts or candidate.get("source_grafts"):
            meta["source_grafts"] = list(candidate.get("source_grafts",
                                                      source_grafts))
        return meta

    def _candidate_conflicts(self, candidate):
        subject = self._norm_fact_field(candidate.get("subject"))
        predicate = self._norm_fact_field(candidate.get("predicate"))
        value = self._norm_fact_field(candidate.get("value"))
        if not (subject and predicate and value):
            return []
        out = []
        for i, g in enumerate(self.arena.grafts):
            meta = g.get("metadata", self._default_metadata(g))
            if not meta.get("active", True):
                continue
            if self._norm_fact_field(meta.get("subject")) != subject:
                continue
            if self._norm_fact_field(meta.get("predicate")) != predicate:
                continue
            if not self._candidate_scope_conflicts(candidate, meta):
                continue
            if not self._candidate_time_conflicts(candidate, meta):
                continue
            old_value = self._norm_fact_field(meta.get("value"))
            if old_value and old_value != value:
                out.append(i)
        return out

    def _candidate_equivalent_targets(self, candidate):
        subject = self._norm_fact_field(candidate.get("subject"))
        predicate = self._norm_fact_field(candidate.get("predicate"))
        value = self._norm_fact_field(candidate.get("value"))
        if not (subject and predicate and value):
            return []
        valid_from = self._norm_fact_time_value(candidate.get("valid_from"))
        expires_at = self._norm_fact_time_value(candidate.get("expires_at"))
        out = []
        for i, g in enumerate(self.arena.grafts):
            meta = g.get("metadata", self._default_metadata(g))
            if not meta.get("active", True) or g.get("retired"):
                continue
            if self._norm_fact_field(meta.get("subject")) != subject:
                continue
            if self._norm_fact_field(meta.get("predicate")) != predicate:
                continue
            if self._norm_fact_field(meta.get("value")) != value:
                continue
            if not self._candidate_scope_conflicts(candidate, meta):
                continue
            if self._norm_fact_time_value(meta.get("valid_from")) != valid_from:
                continue
            if self._norm_fact_time_value(meta.get("expires_at")) != expires_at:
                continue
            out.append(i)
        return out

    def _reinforce_extraction_target(self, idx, candidate, metadata,
                                     confidence, write_intent):
        idx = int(idx)
        g = self.arena.grafts[idx]
        meta = g.setdefault("metadata", self._default_metadata(g))
        for key in ("source_turns", "source_grafts"):
            if metadata.get(key):
                meta[key] = self._append_unique(meta.get(key, ()),
                                                metadata[key])
        identity_keys = {
            "subject", "predicate", "value", "valid_from", "expires_at",
            "source_turns", "source_grafts", "supersedes", "superseded_by",
            "active",
        }
        for key, value in metadata.items():
            if key not in identity_keys:
                meta[key] = value
        meta["confidence"] = max(float(meta.get("confidence", 0.0)),
                                 float(confidence))
        rank = {
            "imported": 0,
            "inferred": 1,
            "observed": 2,
            "system_asserted": 3,
            "user_asserted": 4,
        }
        old_intent = meta.get("write_intent", "observed")
        if rank.get(write_intent, 0) >= rank.get(old_intent, 0):
            meta["write_intent"] = write_intent
        meta["reinforcement_count"] = int(meta.get("reinforcement_count", 0)) + 1
        meta["reinforced_at"] = datetime.now(timezone.utc).isoformat()
        self._mark_dirty(idx, payload=False, metadata=True)
        self._append_wal("NODE_META", node_id=idx, metadata=meta,
                         state=list(self._state_tuple(g)))
        return idx

    def _candidate_to_review(self, candidate, text, reason, metadata):
        return {"action": "review_candidate",
                "review_id": self.review_candidate(
                    text,
                    proposed_kind=candidate.get(
                        "candidate_type", candidate.get("kind", "fact")),
                    proposed_scope=candidate.get("scope", "project"),
                    proposed_durability=candidate.get("durability", "project"),
                    proposed_mutability=candidate.get("mutability", "stable"),
                    confidence=float(candidate.get("confidence", 0.5)),
                    action="review_candidate", reason=reason,
                    metadata=metadata)}

    def apply_extraction_candidate(self, candidate, source_text=None,
                                   source_turns=(), source_grafts=(),
                                   write_direct_threshold=0.95):
        return self.runtime.apply_extraction_candidate(
            candidate, source_text=source_text, source_turns=source_turns,
            source_grafts=source_grafts,
            write_direct_threshold=write_direct_threshold)

    def _apply_extraction_candidate_direct(self, candidate, source_text=None,
                                           source_turns=(), source_grafts=(),
                                           write_direct_threshold=0.95):
        """Apply one classifier/extractor memory candidate conservatively."""
        text = self._candidate_text(candidate, source_text=source_text)
        metadata = self._candidate_metadata(candidate, source_turns,
                                            source_grafts)
        action = candidate.get("action", "review_candidate")
        confidence = float(candidate.get("confidence", 0.5))
        write_intent = candidate.get("write_intent", "observed")
        if action in ("ignore", "keep_turn_only"):
            return {"action": action}
        if action == "pin":
            metadata["pinned"] = True
            action = "write_direct"
        authoritative = write_intent in ("user_asserted", "system_asserted")
        if action in ("expire",):
            if not authoritative:
                return self._candidate_to_review(
                    candidate, text,
                    "expire action requires authoritative intent",
                    metadata)
            targets = self._candidate_expire_targets(candidate)
            if not targets:
                return self._candidate_to_review(
                    candidate, text, "expire action found no active target",
                    metadata)
            expired = self._expire_extraction_targets(targets, text)
            return {"action": "expire", "expired": expired}

        requested_supersedes = self._candidate_supersede_targets(candidate)
        requested_supersede_ids = self._candidate_target_ids(candidate)
        if requested_supersede_ids and not requested_supersedes:
            return self._candidate_to_review(
                candidate, text, "supersede action found no active target",
                metadata)
        if requested_supersedes and not authoritative:
            return self._candidate_to_review(
                candidate, text,
                "supersede action requires authoritative intent",
                metadata)

        conflicts = self._candidate_conflicts(candidate)
        imported = write_intent == "imported"
        if conflicts and not authoritative:
            reason = ("conflicts with active memory"
                      if not imported else
                      "imported candidate conflicts with active memory")
            return self._candidate_to_review(candidate, text, reason, metadata)
        if (action == "write_direct" and confidence < write_direct_threshold
                and not authoritative):
            return self._candidate_to_review(
                candidate, text, "confidence below direct-write threshold",
                metadata)
        if action in ("review_candidate", "update_existing",
                      "supersede_existing") and not (
                authoritative and conflicts):
            return self._candidate_to_review(
                candidate, text, f"{action} requires review", metadata)
        if action not in ("write_direct", "update_existing",
                          "supersede_existing"):
            return self._candidate_to_review(
                candidate, text, f"unsupported extraction action: {action}",
                metadata)

        supersedes = conflicts if conflicts else list(requested_supersedes)
        if not supersedes:
            equivalent = self._candidate_equivalent_targets(candidate)
            if equivalent:
                idx = self._reinforce_extraction_target(
                    equivalent[0], candidate, metadata, confidence,
                    write_intent)
                return {"action": "reinforce_existing", "node_id": idx}
        metadata["supersedes"] = list(supersedes)
        idx = self.remember(
            text,
            durability=candidate.get("durability", "project"),
            mutability=candidate.get("mutability", "stable"),
            scope=candidate.get("scope", "project"),
            kind=candidate.get("candidate_type", candidate.get("kind", "fact")),
            write_intent=write_intent,
            confidence=confidence,
            metadata=metadata)
        for i in supersedes:
            g = self.arena.grafts[int(i)]
            meta = g.setdefault("metadata", self._default_metadata(g))
            meta["active"] = False
            meta["superseded_by"] = [idx]
            g["retired"] = True
            self._mark_dirty(int(i), payload=False, metadata=True)
        if supersedes:
            self._native_apply_revision(idx, supersedes)
            self._append_wal("MEMORY_EXTRACT_SUPERSEDE",
                             node_id=idx, supersedes=list(supersedes))
            return {"action": "supersede_existing", "node_id": idx,
                    "supersedes": list(supersedes)}
        return {"action": "write_direct", "node_id": idx}

    def apply_extraction_candidates(self, candidates, source_text=None,
                                    source_turns=(), source_grafts=(),
                                    write_direct_threshold=0.95):
        return self.runtime.apply_extraction_candidates(
            candidates, source_text=source_text, source_turns=source_turns,
            source_grafts=source_grafts,
            write_direct_threshold=write_direct_threshold)

    def _apply_extraction_candidates_direct(self, candidates, source_text=None,
                                            source_turns=(), source_grafts=(),
                                            write_direct_threshold=0.95):
        return [self._apply_extraction_candidate_direct(
            c, source_text=source_text, source_turns=source_turns,
            source_grafts=source_grafts,
            write_direct_threshold=write_direct_threshold)
                for c in candidates]

    def _new_turn_grafts(self, before):
        return [i for i in range(len(before), len(self.arena.grafts))
                if self.arena.grafts[i].get("kind", "turn") in
                ("turn", "recall")]

    def _extractor_call(self, text, source_grafts, context):
        fn = getattr(self.extractor, "extract", self.extractor)
        return fn(text, repository=self, source_grafts=list(source_grafts),
                  source_turns=list(source_grafts), context=dict(context or {}))

    def _extract_from_new_turns(self, before, context=None):
        self.last_extraction_results = []
        self.last_extraction_error = None
        if self.extractor is None:
            return []
        source_grafts = self._new_turn_grafts(before)
        if not source_grafts:
            return []
        source_text = "\n".join(self.arena.grafts[i]["text"]
                                for i in source_grafts)
        try:
            candidates = self._extractor_call(source_text, source_grafts,
                                              context or {})
        except Exception as exc:
            self.last_extraction_error = str(exc)
            if self.extraction_error_policy == "raise":
                raise
            result = {"action": "extract_error", "error": str(exc)}
            self.last_extraction_results = [result]
            self._append_wal("EXTRACTION_ERROR", error=str(exc),
                             source_grafts=list(source_grafts),
                             context=dict(context or {}))
            return self.last_extraction_results
        if candidates is None:
            candidates = []
        elif isinstance(candidates, dict):
            candidates = [candidates]
        else:
            candidates = list(candidates)
        self.last_extraction_results = self._apply_extraction_candidates_direct(
            candidates, source_text=source_text,
            source_turns=source_grafts, source_grafts=source_grafts,
            write_direct_threshold=self.extraction_write_threshold)
        return self.last_extraction_results

    @staticmethod
    def _normalize_review_item(item, fallback_id):
        out = dict(item)
        out["id"] = int(out.get("id", fallback_id))
        out.setdefault("status", "pending")
        return out

    @classmethod
    def _apply_review_wal_records(cls, base_reviews, records, since_lsn=0):
        reviews = {}
        for pos, item in enumerate(base_reviews or ()):
            norm = cls._normalize_review_item(item, pos)
            reviews[norm["id"]] = norm
        for rec in records or ():
            if int(rec.get("lsn", 0)) <= int(since_lsn):
                continue
            typ = rec.get("type")
            if typ == "REVIEW_CANDIDATE":
                item = {k: v for k, v in rec.items()
                        if k not in ("lsn", "type", "time")}
                norm = cls._normalize_review_item(item, len(reviews))
                reviews[norm["id"]] = norm
            elif typ == "REVIEW_EDIT":
                rid = int(rec.get("review_id", -1))
                if rid in reviews:
                    updates = dict(rec.get("updates", {}) or {})
                    if updates.get("metadata") is not None:
                        updates["metadata"] = dict(updates["metadata"])
                    reviews[rid].update(updates)
                    reviews[rid]["status"] = "pending"
                    if rec.get("reason"):
                        reviews[rid]["edit_reason"] = rec["reason"]
            elif typ == "REVIEW_APPROVE":
                rid = int(rec.get("review_id", -1))
                if rid in reviews:
                    reviews[rid]["status"] = "approved"
                    if "node_id" in rec:
                        reviews[rid]["approved_node_id"] = int(rec["node_id"])
                    if rec.get("approved_action"):
                        reviews[rid]["approved_action"] = rec["approved_action"]
            elif typ == "REVIEW_REJECT":
                rid = int(rec.get("review_id", -1))
                if rid in reviews:
                    reviews[rid]["status"] = "rejected"
                    if rec.get("reason"):
                        reviews[rid]["rejection_reason"] = rec["reason"]
        return [reviews[k] for k in sorted(reviews)]

    def _review_item(self, review_id):
        rid = int(review_id)
        if rid < 0 or rid >= len(self.review_buffer):
            raise IndexError(f"review item {rid} does not exist")
        item = self.review_buffer[rid]
        item.setdefault("id", rid)
        item.setdefault("status", "pending")
        return item

    def review_candidate(self, text, proposed_kind="fact",
                         proposed_scope="project",
                         proposed_durability="project",
                         proposed_mutability="stable",
                         confidence=0.5, action="review_candidate",
                         reason="", metadata=None):
        item = {"id": len(self.review_buffer), "text": text,
                "proposed_kind": proposed_kind,
                "proposed_scope": proposed_scope,
                "proposed_durability": proposed_durability,
                "proposed_mutability": proposed_mutability,
                "confidence": float(confidence), "action": action,
                "reason": reason, "status": "pending"}
        if metadata:
            item["metadata"] = dict(metadata)
        self.review_buffer.append(item)
        self._append_wal("REVIEW_CANDIDATE", **item)
        return item["id"]

    def edit_review(self, review_id, text=None, proposed_kind=None,
                    proposed_scope=None, proposed_durability=None,
                    proposed_mutability=None, confidence=None,
                    metadata=None, reason=""):
        return self.runtime.edit_review(
            review_id, text=text, proposed_kind=proposed_kind,
            proposed_scope=proposed_scope,
            proposed_durability=proposed_durability,
            proposed_mutability=proposed_mutability,
            confidence=confidence, metadata=metadata, reason=reason)

    def _edit_review_direct(self, review_id, text=None, proposed_kind=None,
                            proposed_scope=None, proposed_durability=None,
                            proposed_mutability=None, confidence=None,
                            metadata=None, reason=""):
        item = self._review_item(review_id)
        if item.get("status") == "approved":
            raise RuntimeError("approved review items cannot be edited")
        if item.get("status") == "rejected":
            raise RuntimeError("rejected review items cannot be edited")
        updates = {}
        if text is not None:
            updates["text"] = str(text)
        for key, val in (
                ("proposed_kind", proposed_kind),
                ("proposed_scope", proposed_scope),
                ("proposed_durability", proposed_durability),
                ("proposed_mutability", proposed_mutability)):
            if val is not None:
                updates[key] = str(val)
        if confidence is not None:
            updates["confidence"] = float(confidence)
        if metadata is not None:
            merged = dict(item.get("metadata", {}))
            merged.update(dict(metadata))
            updates["metadata"] = merged
        item.update(updates)
        item["status"] = "pending"
        if reason:
            item["edit_reason"] = reason
        self._append_wal("REVIEW_EDIT", review_id=int(review_id),
                         updates=updates, reason=reason)
        return dict(item)

    def change_review_scope(self, review_id, scope, durability=None,
                            mutability=None):
        return self.edit_review(
            review_id,
            proposed_scope=scope,
            proposed_durability=durability,
            proposed_mutability=mutability,
            reason="scope changed")

    def reject_review(self, review_id, reason=""):
        return self.runtime.reject_review(review_id, reason=reason)

    def _reject_review_direct(self, review_id, reason=""):
        item = self._review_item(review_id)
        if item.get("status") == "approved":
            raise RuntimeError("approved review items cannot be rejected")
        item["status"] = "rejected"
        if reason:
            item["rejection_reason"] = reason
        self._append_wal("REVIEW_REJECT", review_id=int(review_id),
                         reason=reason)
        return dict(item)

    def approve_review(self, review_id):
        return self.runtime.approve_review(review_id)

    def _review_semantic_candidate(self, item):
        metadata = dict(item.get("metadata", {}) or {})
        required = ("subject", "predicate", "value")
        if not all(metadata.get(k) for k in required):
            return None
        candidate = {k: metadata[k] for k in (
            "subject", "predicate", "value", "valid_from", "expires_at",
            "source_turns", "source_grafts") if k in metadata}
        candidate.update({
            "action": "write_direct",
            "candidate_type": item.get("proposed_kind", "fact"),
            "text": item.get("text", ""),
            "scope": item.get("proposed_scope", "project"),
            "durability": item.get("proposed_durability", "project"),
            "mutability": item.get("proposed_mutability", "stable"),
            "confidence": float(item.get("confidence", 0.5)),
            "write_intent": "user_asserted",
            "metadata": metadata,
        })
        return candidate

    def _approve_review_direct(self, review_id):
        item = self._review_item(review_id)
        if item.get("status") == "rejected":
            raise RuntimeError("rejected review items cannot be approved")
        if item.get("status") == "approved" and "approved_node_id" in item:
            return int(item["approved_node_id"])
        result = None
        candidate = self._review_semantic_candidate(item)
        if candidate is not None:
            result = self._apply_extraction_candidate_direct(
                candidate, write_direct_threshold=1.0)
        if result is not None:
            if result.get("action") not in (
                    "write_direct", "reinforce_existing",
                    "supersede_existing"):
                raise RuntimeError(
                    f"approved review produced unsupported action "
                    f"{result.get('action')!r}")
            idx = int(result["node_id"])
            item["approved_action"] = result["action"]
        else:
            idx = self.remember(
                item["text"], durability=item["proposed_durability"],
                mutability=item["proposed_mutability"],
                scope=item["proposed_scope"],
                kind=item["proposed_kind"],
                confidence=item["confidence"],
                metadata=item.get("metadata"))
            item["approved_action"] = "remember"
        item["status"] = "approved"
        item["approved_node_id"] = idx
        self._append_wal("REVIEW_APPROVE", review_id=review_id, node_id=idx,
                         approved_action=item["approved_action"])
        return idx

    def show_memory_about(self, query):
        q = query.lower()
        out = []
        for i, g in enumerate(self.arena.grafts):
            if q in g.get("text", "").lower():
                meta = g.get("metadata", self._default_metadata(g))
                if meta.get("active", True):
                    out.append({"node_id": i, "text": g["text"],
                                "metadata": meta})
        return out

    def why_remember(self, query):
        rows = self.show_memory_about(query)
        return [{"node_id": r["node_id"],
                 "write_intent": r["metadata"].get("write_intent"),
                 "kind": r["metadata"].get("kind"),
                 "durability": r["metadata"].get("durability"),
                 "mutability": r["metadata"].get("mutability"),
                 "scope": r["metadata"].get("scope"),
                 "confidence": r["metadata"].get("confidence"),
                 "pinned": r["metadata"].get("pinned", False),
                 "source_grafts": r["metadata"].get("source_grafts", []),
                 "provenance": self.arena.grafts[r["node_id"]].get(
                     "provenance", [])}
                for r in rows]

    # ---------------------------------------------------------- librarian
    def _active(self, kinds):
        live = {gi for gi, _ in self.arena.live_segs if gi is not None}
        return [i for i, g in enumerate(self.arena.grafts)
                if not g.get("retired") and i not in live
                and g.get("kind", "turn") in kinds]

    def _due(self):
        """Stateless fold plan. Documents are never folded — they are
        reference material, not history. Fold-exempt nodes (a prior fidelity
        abort) drop out so the window advances instead of looping."""
        ok = lambda i: not self.arena.grafts[i].get("no_fold")
        jobs = []
        turns = [i for i in self._active(("turn",)) if ok(i)]
        if len(turns) >= self.TURNS_HIGH:
            jobs.append(("digest", turns[:self.TURNS_FOLD]))
        if self.DIGESTS_HIGH and getattr(self.arena, "ENABLE_ERA_FOLDING", True):
            digests = [i for i in self._active(("digest",)) if ok(i)]
            if len(digests) >= self.DIGESTS_HIGH:
                jobs.append(("era", digests[:self.DIGESTS_FOLD]))
        return jobs

    def fold_pending(self):
        return len(self._due())

    def _fold_once(self):
        jobs = self._due()
        if not jobs:
            return False
        kind, idxs = jobs[0]
        didx, _ = self.arena.consolidate(idxs)
        fold_event = {
            "kind": kind,
            "sources": list(idxs),
            "accepted": didx is not None,
            "digest_idx": didx,
            "result": dict(getattr(self.arena, "last_consolidation_result",
                                   {})),
            "attempts": [dict(a) for a in getattr(
                self.arena, "last_consolidation_attempts", [])],
        }
        self.fold_history.append(fold_event)
        if didx is None:
            # fidelity abort: keep these sources unfolded (clean readers and
            # routers), exempt them so the planner moves to another window.
            self.folds_aborted = getattr(self, "folds_aborted", 0) + 1
            for i in idxs:
                self.arena.grafts[i]["no_fold"] = True
            return True
        self.arena.grafts[didx]["kind"] = kind
        self._free_retired()
        return True

    def idle(self, max_jobs=1):
        """Run deferred librarian work; call between turns or when the
        conversation is quiet. Returns folds executed."""
        return self.runtime.idle(max_jobs=max_jobs)

    def _librarian(self):
        if self.librarian_mode == "inline":
            while self._fold_once():
                pass
        else:
            # deferred: backpressure only — bound the active pool if the
            # host never grants idle time. Count FOLDABLE turns (exempt
            # ones from a fidelity abort are permanently resident and must
            # NOT re-trigger inline folds that would just abort again — the
            # 9s hot-path spike, measured 2026-06-11).
            foldable = sum(1 for i in self._active(("turn",))
                           if not self.arena.grafts[i].get("no_fold"))
            if foldable >= self.TURNS_HIGH * 2:
                self._fold_once()

    def _node_bytes(self, g):
        # dialect vals/token/layer (MLA: c 256 + kpe 32; GQA: kv heads x
        # head_dim x 2), fp16
        return (g["ntok"] * self.arena.VALS_PER_TOK_LAYER * 2
                * len(self.arena.m.layers))

    def _host_payload_bytes(self, g):
        payload = g.get("host_payload")
        if payload is None:
            return 0
        return int(sum(np.asarray(v).nbytes for v in payload.values()))

    def _open_native_store(self, lib_path):
        from core.grm_native import NativeGraftStore
        return NativeGraftStore(
            lib_path, model_type=self.dialect_desc.model_type,
            num_layers=self.dialect_desc.num_layers,
            hidden_dim=self.dialect_desc.hidden_dim,
            vals_per_tok_layer=self.dialect_desc.vals_per_tok_layer,
            route_layer=self.dialect_desc.route_layer,
            payload_kind=self.dialect_desc.payload_kind,
            latent_rank=self.dialect_desc.latent_rank,
            rope_dim=self.dialect_desc.rope_dim,
            num_kv_heads=self.dialect_desc.num_kv_heads,
            head_dim=self.dialect_desc.head_dim,
            position_law=self.dialect_desc.position_law,
            state_kind=self.dialect_desc.state_kind,
            graftability=self.dialect_desc.graftability,
            remountable=self.dialect_desc.remountable,
            composition=self.dialect_desc.composition)

    def _native_checkpoint_root(self):
        return os.path.join(self.path, "native")

    def _native_checkpoint_file(self):
        return os.path.join(self._native_checkpoint_root(), "grm_store.bin")

    def _native_save_checkpoint(self):
        if self.native_store is None:
            return False
        if not hasattr(self.native_store, "save_checkpoint"):
            return False
        if hasattr(self.native_store, "clear_payload"):
            for idx, node_id in list(self._native_node_ids.items()):
                g = self.arena.grafts[int(idx)]
                if g.get("retired") and g.get("durable"):
                    try:
                        self.native_store.clear_payload(node_id)
                    except RuntimeError as exc:
                        if "unavailable" not in str(exc):
                            raise
        if (hasattr(self.native_store, "dirty_node_ids")
                and os.path.exists(self._native_checkpoint_file())):
            try:
                if not self.native_store.dirty_node_ids():
                    return True
            except RuntimeError as exc:
                if "unavailable" not in str(exc):
                    raise
        self.native_store.save_checkpoint(self._native_checkpoint_root())
        return True

    def _native_load_checkpoint(self):
        self._native_checkpoint_loaded = False
        if self.native_store is None:
            return False
        if not hasattr(self.native_store, "load_checkpoint"):
            return False
        if not os.path.exists(self._native_checkpoint_file()):
            return False
        self.native_store.load_checkpoint(self._native_checkpoint_root())
        self._native_checkpoint_loaded = True
        return True

    def _native_payload_blob(self, payload):
        if payload is None:
            return b""
        chunks = []
        for key in sorted(payload):
            chunks.append(np.ascontiguousarray(payload[key]).tobytes())
        return b"".join(chunks)

    def _native_set_payload(self, node_id, payload):
        if payload is None:
            return
        if not hasattr(self.native_store, "set_tensor"):
            return
        for key in sorted(payload):
            self.native_store.set_tensor(node_id, key, payload[key])

    def _native_configure_arena(self):
        if self.native_store is None:
            return
        if not hasattr(self.native_store, "configure_arena"):
            return
        self.native_store.configure_arena(
            getattr(self.arena, "n_sink", 0), getattr(self.arena, "width", 0))

    def _native_lexical_keys(self, g):
        rare = g.get("rare")
        if rare is None:
            rare = ArenaCache._rare_tokens(g.get("text", ""))
        return sorted(str(k) for k in rare)

    def _native_set_route(self, idx):
        if self.native_store is None:
            return
        g = self.arena.grafts[idx]
        if "cent" not in g:
            return
        node_id = self._native_node_ids.get(int(idx))
        if node_id is None:
            return
        route_keys = [np.asarray(g["cent"], dtype=np.float32).reshape(-1)]
        for child in g.get("child_cents", ()):
            route_keys.append(np.asarray(child, dtype=np.float32).reshape(-1))
        if hasattr(self.native_store, "set_route_key_list"):
            self.native_store.set_route_key_list(
                node_id, route_keys, self._native_lexical_keys(g))
        elif hasattr(self.native_store, "set_route_keys"):
            self.native_store.set_route_keys(
                node_id, np.stack(route_keys), self._native_lexical_keys(g))
        else:
            self.native_store.set_route(
                node_id, route_keys[0].tolist(), self._native_lexical_keys(g))

    def _native_ref_ids(self, refs):
        out = []
        for ref in refs or ():
            try:
                idx = int(ref)
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx >= len(self.arena.grafts):
                continue
            node_id = self._native_node_ids.get(idx)
            if node_id is None:
                node_id = self._native_sync_node(idx, payload_required=False)
            if node_id is not None and int(node_id) not in out:
                out.append(int(node_id))
        return out

    def _native_set_metadata(self, idx):
        if self.native_store is None:
            return
        node_id = self._native_node_ids.get(int(idx))
        if node_id is None:
            return
        g = self.arena.grafts[idx]
        metadata = g.get("metadata", {})
        if hasattr(self.native_store, "set_metadata"):
            self.native_store.set_metadata(node_id, metadata)
        if hasattr(self.native_store, "set_active"):
            active = bool(metadata.get("active", not bool(g.get("retired"))))
            self.native_store.set_active(node_id, active)
        if hasattr(self.native_store, "set_graph_edges"):
            self.native_store.set_graph_edges(
                node_id,
                source_turns=self._native_ref_ids(
                    metadata.get("source_turns", ())),
                source_grafts=self._native_ref_ids(
                    metadata.get("source_grafts", ())),
                supersedes=self._native_ref_ids(
                    metadata.get("supersedes", ())),
                superseded_by=self._native_ref_ids(
                    metadata.get("superseded_by", ())))

    def _native_apply_revision(self, replacement_idx, supersedes):
        if self.native_store is None:
            return
        if not hasattr(self.native_store, "apply_revision"):
            return
        replacement_id = self._native_sync_node(replacement_idx)
        superseded_ids = [self._native_sync_node(i) for i in supersedes]
        self.native_store.apply_revision(replacement_id, superseded_ids)

    def _native_sync_node(self, idx, payload_required=True):
        if self.native_store is None:
            return None
        idx = int(idx)
        g = self.arena.grafts[idx]
        if idx in self._native_node_ids:
            node_id = self._native_node_ids[idx]
            if payload_required and g.get("host_payload") is None:
                self._ensure_host_payload(idx, g)
            self._native_set_payload(node_id, g.get("host_payload"))
            self._native_set_route(idx)
            self._native_set_metadata(idx)
            return node_id
        if payload_required and g.get("host_payload") is None:
            self._ensure_host_payload(idx, g)
        if (g.get("host_payload") is not None
                and hasattr(self.native_store, "add_structured_node")):
            node_id = self.native_store.add_structured_node(
                g.get("text", ""), g.get("host_payload"),
                ntok=g.get("ntok", 0))
        else:
            node_id = self.native_store.add_node(
                g.get("text", ""), self._native_payload_blob(
                    g.get("host_payload")), ntok=g.get("ntok", 0))
        self._native_node_ids[int(idx)] = int(node_id)
        g["native_node_id"] = int(node_id)
        self._native_set_route(idx)
        self._native_set_metadata(idx)
        return int(node_id)

    def _native_mark_durable(self, idx):
        if self.native_store is None:
            return
        node_id = self._native_sync_node(idx)
        self.native_store.mark_durable(node_id)

    def _native_evict_device_copy(self, idx):
        if self.native_store is None:
            return
        node_id = self._native_sync_node(idx)
        if self.arena.grafts[int(idx)].get("durable"):
            self.native_store.mark_durable(node_id)
        self.native_store.evict_device_copy(node_id)

    def _sync_native_full(self):
        if self.native_store is None:
            return
        for i, g in enumerate(self.arena.grafts):
            if (g.get("host_payload") is None and not g.get("durable")
                    and not g.get("payload_pending") and not g.get("retired")):
                continue
            node_id = self._native_sync_node(
                i, payload_required=g.get("host_payload") is not None)
            if g.get("durable"):
                self.native_store.mark_durable(node_id)

    def native_route(self, query_key, lexical_keys=(), topk=3, kinds=(),
                     scopes=(), durabilities=(), mutabilities=()):
        if self.native_store is None:
            raise RuntimeError("native GRM store is not enabled")
        def norm_filter(values):
            if values is None:
                return ()
            if isinstance(values, str):
                return (values,)
            return tuple(values)
        kinds = norm_filter(kinds)
        scopes = norm_filter(scopes)
        durabilities = norm_filter(durabilities)
        mutabilities = norm_filter(mutabilities)
        inverse = {native_id: idx for idx, native_id in self._native_node_ids.items()}
        routed = self.native_store.route(
            query_key, lexical_keys, topk, kinds=kinds, scopes=scopes,
            durabilities=durabilities, mutabilities=mutabilities)
        out = []
        for nid in routed:
            idx = inverse.get(nid)
            if idx is None:
                continue
            g = self.arena.grafts[idx]
            meta = g.get("metadata", self._default_metadata(g))
            if g.get("retired") or not meta.get("active", True):
                continue
            if kinds and meta.get("kind", g.get("kind", "turn")) not in kinds:
                continue
            if scopes and meta.get("scope") not in scopes:
                continue
            if durabilities and meta.get("durability") not in durabilities:
                continue
            if mutabilities and meta.get("mutability") not in mutabilities:
                continue
            out.append(idx)
        return out

    def _page(self):
        """Spill least-recently-mounted device tensors above the VRAM budget.

        RAM is authoritative: a dirty node can leave VRAM as soon as its
        host payload exists. Disk durability is no longer a prerequisite for
        device eviction.
        """
        if self.vram_budget is None:
            return 0
        for i, g in enumerate(self.arena.grafts):
            if g.get("h") is not None and g.get("host_payload") is None:
                self._ensure_host_payload(i, g)
        indexed_live = [(i, g.get("last_used", 0), g)
                        for i, g in enumerate(self.arena.grafts)
                        if g.get("h") is not None
                        and g.get("host_payload") is not None]
        used = sum(self._node_bytes(g) for _, _, g in indexed_live)
        freed = 0
        for i, _, g in sorted(indexed_live, key=lambda x: x[1]):
            if used <= self.vram_budget:
                break
            used -= self._node_bytes(g)
            g["h"] = None
            self._ensure_lifecycle(i, g)
            self._native_evict_device_copy(i)
            freed += 1
        return freed

    def _free_retired(self):
        """Retired nodes leave VRAM; disk is their cold storage."""
        for i, g in enumerate(self.arena.grafts):
            if (g.get("retired") and g.get("h") is not None
                    and g.get("host_payload") is not None):
                g["h"] = None
                self._ensure_lifecycle(i, g)

    # --------------------------------------------------------- persistence
    def _load_node(self, i):
        """RAM-first loader: host payload -> device, NVMe only as fallback."""
        g = self.arena.grafts[i]
        if g.get("host_payload") is None:
            g["host_payload"] = self._read_payload_file(i)
        h = self.arena.unpack_node(g["host_payload"])
        g["h"] = h
        self._ensure_lifecycle(i, g)
        return h

    def _ensure_repo_dirs(self):
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(os.path.join(self.path, "nodes"), exist_ok=True)
        os.makedirs(os.path.join(self.path, "wal"), exist_ok=True)

    @staticmethod
    def _fsync_parent_dir(path):
        parent = os.path.dirname(os.path.abspath(path))
        try:
            fd = os.open(parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _durability_tmp_path(path):
        return f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"

    def _atomic_write_json(self, path, payload, *, indent=None):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = self._durability_tmp_path(path)
        try:
            with open(tmp, "w") as fh:
                json.dump(payload, fh, indent=indent)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
            self._fsync_parent_dir(path)
        except Exception:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise

    def _atomic_savez_compressed(self, path, **payload):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = self._durability_tmp_path(path)
        try:
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, **payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
            self._fsync_parent_dir(path)
        except Exception:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise

    def _wal_path(self):
        return os.path.join(self.path, "wal", "000001.wal")

    def _append_wal(self, rec_type, **fields):
        if not self.wal_enabled:
            return None
        self._ensure_repo_dirs()
        self._wal_lsn += 1
        rec = {"lsn": self._wal_lsn, "type": rec_type,
               "time": time.time(), **fields}
        with open(self._wal_path(), "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._fsync_parent_dir(self._wal_path())
        return self._wal_lsn

    def _read_wal(self):
        p = self._wal_path()
        if not os.path.exists(p):
            return []
        out = []
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        if out:
            self._wal_lsn = max(int(r.get("lsn", 0)) for r in out)
        return out

    def _apply_config_wal_records(self, records, since_lsn=0):
        applied = []
        for rec in records or ():
            if int(rec.get("lsn", 0)) <= int(since_lsn):
                continue
            if rec.get("type") != "CONFIG":
                continue
            if rec.get("durability_mode"):
                self._set_durability_mode_fields(
                    rec["durability_mode"],
                    wal_enabled=rec.get("wal_enabled"))
                applied.append(int(rec.get("lsn", 0)))
        return tuple(applied)

    def _recover_wal_summary(self, records):
        nodes = {}

        def meta_for(node):
            meta = dict(node.get("metadata") or {})
            node["metadata"] = meta
            return meta

        def retire_node(node_id, superseded_by=None):
            node = nodes.get(int(node_id))
            if node is None:
                return
            meta = meta_for(node)
            meta["active"] = False
            if superseded_by is not None:
                meta["superseded_by"] = [int(superseded_by)]
            node["active"] = False
            node["retired"] = True

        def link_revision(replacement_id, supersedes):
            replacement_id = int(replacement_id)
            supersedes = [int(i) for i in (supersedes or [])]
            repl = nodes.get(replacement_id)
            if repl is not None:
                meta = meta_for(repl)
                prior = [int(i) for i in meta.get("supersedes", [])]
                meta["supersedes"] = list(dict.fromkeys(prior + supersedes))
                meta["active"] = True
                repl["active"] = True
                repl["retired"] = False
            for old in supersedes:
                retire_node(old, superseded_by=replacement_id)

        def expire_nodes(node_ids, expired_by="", expired_at=None):
            for node_id in node_ids or ():
                node = nodes.get(int(node_id))
                if node is None:
                    continue
                meta = meta_for(node)
                meta["active"] = False
                if expired_at:
                    meta["expired_at"] = expired_at
                if expired_by:
                    meta["expired_by"] = expired_by
                node["active"] = False
                node["retired"] = True

        for rec in records:
            typ = rec.get("type")
            if typ == "NODE_UPSERT":
                node_id = int(rec["node_id"])
                metadata = dict(rec.get("metadata", {}) or {})
                active = bool(metadata.get("active", True))
                nodes[node_id] = {
                    "node_id": node_id,
                    "text": rec.get("text", ""),
                    "kind": rec.get("kind", "turn"),
                    "metadata": metadata,
                    "payload_pending": bool(rec.get("has_payload", False)),
                    "active": active,
                    "retired": not active,
                    "no_fold": bool(metadata.get("no_fold", False)),
                }
            elif typ == "NODE_META" and int(rec.get("node_id", -1)) in nodes:
                node = nodes[int(rec["node_id"])]
                metadata = dict(rec.get("metadata", {}) or {})
                node["metadata"] = metadata
                state = rec.get("state") or ()
                if len(state) >= 1:
                    node["kind"] = state[0]
                    metadata.setdefault("kind", state[0])
                if len(state) >= 2:
                    node["retired"] = bool(state[1])
                    node["active"] = not bool(state[1])
                    metadata["active"] = not bool(state[1])
                if len(state) >= 3:
                    node["no_fold"] = bool(state[2])
                    metadata["no_fold"] = bool(state[2])
                if len(state) >= 4:
                    node["sources"] = list(state[3])
                    metadata.setdefault("source_grafts", list(state[3]))
                if len(state) >= 5:
                    node["tags"] = list(state[4])
                    metadata.setdefault("tags", list(state[4]))
            elif typ == "NODE_FORGET":
                q = rec.get("query", "").lower()
                for n in nodes.values():
                    if q and q in n.get("text", "").lower():
                        retire_node(n["node_id"])
            elif typ in ("MEMORY_CORRECT", "MEMORY_EXTRACT_SUPERSEDE"):
                if "node_id" in rec:
                    link_revision(rec["node_id"], rec.get("supersedes", ()))
            elif typ == "MEMORY_EXTRACT_EXPIRE":
                expire_nodes(rec.get("expired", ()),
                             expired_by=rec.get("text", ""),
                             expired_at=rec.get("expired_at"))
        self.recovered_reviews = self._apply_review_wal_records([], records)
        return [nodes[k] for k in sorted(nodes)]

    def _wal_placeholder_graft(self, n, width):
        meta = dict(n.get("metadata") or {})
        retired = bool(n.get("retired", not bool(n.get("active", True))))
        meta.setdefault("kind", n.get("kind", "turn"))
        meta["active"] = not retired
        sources = list(n.get("sources", meta.get("source_grafts", [])) or [])
        tags = list(n.get("tags", meta.get("tags", [])) or [])
        no_fold = bool(n.get("no_fold", meta.get("no_fold", False)))
        if no_fold:
            meta["no_fold"] = True
        text = n.get("text", "")
        return {
            "kind": n.get("kind", "turn"),
            "text": text,
            "ntok": int(meta.get("ntok", 0) or 0),
            "sources": sources,
            "retired": retired,
            "no_fold": no_fold,
            "tags": tags,
            "rare": ArenaCache._rare_tokens(text),
            "cent": np.zeros(width, np.float32),
            "metadata": meta,
            "provenance": [self._provenance(
                "wal_recovery", node_id=n.get("node_id"))],
            "host_payload": None,
            "host_present": False,
            "device_present": False,
            "dirty": False,
            "durable": False,
            "cold_only": False,
            "payload_pending": bool(n.get("payload_pending", False)),
            "recovered": True,
            "h": None,
        }

    def _apply_wal_metadata_state(self, g, metadata, state):
        metadata = dict(metadata or {})
        g["metadata"] = metadata
        if len(state) >= 1:
            g["kind"] = state[0]
            metadata.setdefault("kind", state[0])
        if len(state) >= 2:
            g["retired"] = bool(state[1])
            metadata["active"] = not bool(state[1])
        if len(state) >= 3:
            g["no_fold"] = bool(state[2])
            metadata["no_fold"] = bool(state[2])
        if len(state) >= 4:
            g["sources"] = list(state[3])
            metadata.setdefault("source_grafts", list(state[3]))
        if len(state) >= 5:
            g["tags"] = list(state[4])
            metadata.setdefault("tags", list(state[4]))

    def _apply_manifest_wal_records(self, records, since_lsn):
        width = self._wal_cent_width()
        changed = set()

        def meta_for(g):
            meta = dict(g.get("metadata") or {})
            g["metadata"] = meta
            return meta

        def ensure_node(node_id, rec=None):
            node_id = int(node_id)
            if node_id < len(self.arena.grafts):
                return self.arena.grafts[node_id]
            while len(self.arena.grafts) < node_id:
                gap_id = len(self.arena.grafts)
                gap = self._wal_placeholder_graft({
                    "node_id": gap_id,
                    "text": "",
                    "kind": "recovered_gap",
                    "metadata": {"active": False},
                    "active": False,
                    "payload_pending": False,
                }, width)
                gap["retired"] = True
                self._ensure_lifecycle(gap_id, gap)
                self.arena.grafts.append(gap)
            payload_pending = bool((rec or {}).get("has_payload", False))
            g = self._wal_placeholder_graft({
                "node_id": node_id,
                "text": (rec or {}).get("text", ""),
                "kind": (rec or {}).get("kind", "turn"),
                "metadata": dict((rec or {}).get("metadata", {}) or {}),
                "payload_pending": payload_pending,
            }, width)
            self._ensure_lifecycle(node_id, g)
            self.arena.grafts.append(g)
            return g

        def retire_node(node_id, superseded_by=None):
            node_id = int(node_id)
            if node_id < 0 or node_id >= len(self.arena.grafts):
                return
            g = self.arena.grafts[node_id]
            meta = meta_for(g)
            meta["active"] = False
            if superseded_by is not None:
                meta["superseded_by"] = [int(superseded_by)]
            g["retired"] = True
            changed.add(node_id)

        def link_revision(replacement_id, supersedes):
            replacement_id = int(replacement_id)
            if replacement_id < 0 or replacement_id >= len(self.arena.grafts):
                return
            supersedes = [int(i) for i in (supersedes or [])]
            repl = self.arena.grafts[replacement_id]
            meta = meta_for(repl)
            prior = [int(i) for i in meta.get("supersedes", [])]
            meta["supersedes"] = list(dict.fromkeys(prior + supersedes))
            meta["active"] = True
            repl["retired"] = False
            changed.add(replacement_id)
            for old in supersedes:
                retire_node(old, superseded_by=replacement_id)

        def expire_nodes(node_ids, expired_by="", expired_at=None):
            for node_id in node_ids or ():
                node_id = int(node_id)
                if node_id < 0 or node_id >= len(self.arena.grafts):
                    continue
                g = self.arena.grafts[node_id]
                meta = meta_for(g)
                meta["active"] = False
                if expired_at:
                    meta["expired_at"] = expired_at
                if expired_by:
                    meta["expired_by"] = expired_by
                g["retired"] = True
                changed.add(node_id)

        for rec in records:
            if int(rec.get("lsn", 0)) <= int(since_lsn):
                continue
            typ = rec.get("type")
            if typ == "NODE_UPSERT":
                node_id = int(rec["node_id"])
                g = ensure_node(node_id, rec)
                metadata = dict(rec.get("metadata", {}) or {})
                active = bool(metadata.get("active", True))
                g["text"] = rec.get("text", g.get("text", ""))
                g["kind"] = rec.get("kind", g.get("kind", "turn"))
                g["metadata"] = metadata
                g["retired"] = not active
                g["payload_pending"] = bool(
                    rec.get("has_payload", g.get("payload_pending", False)))
                g["recovered"] = True
                g["rare"] = ArenaCache._rare_tokens(g.get("text", ""))
                changed.add(node_id)
            elif typ == "NODE_META":
                node_id = int(rec.get("node_id", -1))
                if 0 <= node_id < len(self.arena.grafts):
                    self._apply_wal_metadata_state(
                        self.arena.grafts[node_id],
                        rec.get("metadata", {}),
                        rec.get("state") or ())
                    changed.add(node_id)
            elif typ == "NODE_FORGET":
                if "node_id" in rec:
                    retire_node(rec["node_id"])
                    continue
                q = rec.get("query", "").lower()
                for i, g in enumerate(self.arena.grafts):
                    if q and q in g.get("text", "").lower():
                        retire_node(i)
            elif typ in ("MEMORY_CORRECT", "MEMORY_EXTRACT_SUPERSEDE"):
                if "node_id" in rec:
                    link_revision(rec["node_id"], rec.get("supersedes", ()))
            elif typ == "MEMORY_EXTRACT_EXPIRE":
                expire_nodes(rec.get("expired", ()),
                             expired_by=rec.get("text", ""),
                             expired_at=rec.get("expired_at"))

        for node_id in sorted(changed):
            if 0 <= node_id < len(self.arena.grafts):
                self._ensure_lifecycle(node_id, self.arena.grafts[node_id])
        if changed:
            self._rebuild_child_keys()
            self._free_retired()
        return tuple(sorted(changed))

    def _wal_cent_width(self):
        """Routing-centroid width for placeholder cents on recovered nodes.

        Probe the arena's own empty index shape so the placeholder matches the
        dialect (pack_index falls back to 256 only when there are no grafts)."""
        try:
            return int(self.arena.pack_index()["cents"].shape[1])
        except Exception:
            return 256

    def _rehydrate_from_wal(self, recovered):
        """Rebuild a usable (text/metadata) repository from WAL after a crash
        that left no manifest.

        The WAL is lightweight: it carries text, kind, metadata, active state,
        and a payload-pending flag, but never the K/V payload or the routing
        centroid (those only reach NVMe through a manifest checkpoint). So a
        recovered node is text-authoritative but NOT routable until it is
        re-harvested: it gets a zero centroid (cosine ~0 against any query, so
        it never wins routing by accident), no host_payload, no device tensor,
        and payload_pending=True. The node IS visible to show_memory_about /
        why_remember and can be re-harvested or re-flushed. Forgotten nodes
        recover as retired (active=False) so superseded memory stays inert.

        Returns the number of nodes rehydrated."""
        if not recovered or self.arena.grafts:
            return 0
        width = self._wal_cent_width()
        for n in recovered:
            g = self._wal_placeholder_graft(n, width)
            self._ensure_lifecycle(len(self.arena.grafts), g)
            self.arena.grafts.append(g)
        self.review_buffer = list(getattr(self, "recovered_reviews", []))
        return len(self.arena.grafts)

    def _provenance(self, segment_type, node_id=None, **fields):
        sid = getattr(self, "_segment_id", 0)
        self._segment_id = sid + 1
        return {"segment_id": sid, "node_id": node_id,
                "segment_type": segment_type, "created_at": time.time(),
                **fields}

    def _set_new_node_provenance(self, before, segment_type):
        for i in range(len(before), len(self.arena.grafts)):
            self.arena.grafts[i]["provenance"] = [
                self._provenance(segment_type, i)]

    def _payload_to_ram(self, payload):
        if hasattr(payload, "files"):
            keys = payload.files
        else:
            keys = payload.keys()
        return {k: np.ascontiguousarray(payload[k]) for k in keys}

    def _read_payload_file(self, i):
        with np.load(os.path.join(self.path, "nodes", f"{i:04d}.npz")) as z:
            return self._payload_to_ram(z)

    def _ensure_host_payload(self, i, g):
        if g.get("host_payload") is not None:
            self._ensure_lifecycle(i, g)
            return False
        if g.get("h") is None:
            if g.get("durable"):
                g["host_payload"] = self._read_payload_file(i)
                self._ensure_lifecycle(i, g)
                return True
            raise RuntimeError(f"graft {i} has no RAM payload and no device "
                               "payload to snapshot")
        g["host_payload"] = self._payload_to_ram(self.arena.pack_node(g["h"]))
        self._ensure_lifecycle(i, g)
        return True

    def _default_metadata(self, g):
        kind = g.get("kind", "turn")
        return {
            "kind": kind,
            "durability": "session" if kind == "turn" else "project",
            "mutability": "ephemeral" if kind == "turn" else "stable",
            "scope": "conversation" if kind == "turn" else "project",
            "write_intent": "observed",
            "confidence": 1.0,
            "source_turns": [],
            "source_grafts": list(g.get("sources", [])),
            "supersedes": [],
            "active": not bool(g.get("retired")),
        }

    def _ensure_lifecycle(self, i, g):
        g["node_id"] = int(g.get("node_id", i))
        meta = dict(self._default_metadata(g))
        meta.update(g.get("metadata", {}))
        meta["kind"] = g.get("kind", meta.get("kind", "turn"))
        meta["active"] = not bool(g.get("retired"))
        g["metadata"] = meta
        if "durable" not in g:
            g["durable"] = bool(g.get("saved", False))
        g["host_present"] = g.get("host_payload") is not None
        g["device_present"] = g.get("h") is not None
        g["cold_only"] = bool(g.get("durable") and not g["host_present"])
        g["dirty"] = bool(g.get("dirty", not g.get("durable", False)))
        g["saved"] = bool(g["durable"])  # legacy compatibility
        return g

    def _sync_lifecycle(self):
        for i, g in enumerate(self.arena.grafts):
            self._ensure_lifecycle(i, g)

    def _snapshot_state(self):
        return [self._state_tuple(g) for g in self.arena.grafts]

    def _state_tuple(self, g):
        return (
            g.get("kind", "turn"),
            bool(g.get("retired")),
            bool(g.get("no_fold")),
            tuple(g.get("sources", [])),
            tuple(g.get("tags", [])),
        )

    def _mark_dirty(self, idx, payload=False, metadata=True):
        if payload:
            self._ensure_host_payload(idx, self.arena.grafts[idx])
            self._native_sync_node(idx)
        entry = self.dirty_nodes.setdefault(int(idx),
                                            {"payload": False,
                                             "metadata": False})
        entry["payload"] = bool(entry["payload"] or payload)
        entry["metadata"] = bool(entry["metadata"] or metadata)
        g = self.arena.grafts[idx]
        g["dirty"] = True
        g["durable"] = False if payload else bool(g.get("durable", False))
        g["saved"] = bool(g.get("durable", False))
        self._ensure_lifecycle(idx, g)
        if metadata:
            self._native_set_metadata(idx)

    def _mark_mutations(self, before):
        self._sync_lifecycle()
        for i, g in enumerate(self.arena.grafts):
            if i >= len(before):
                self._mark_dirty(i, payload=g.get("h") is not None,
                                 metadata=True)
                self._append_wal("NODE_UPSERT", node_id=i,
                                 kind=g.get("kind", "turn"),
                                 text=g.get("text", ""),
                                 metadata=g.get("metadata", {}),
                                 has_payload=g.get("host_payload") is not None)
            elif self._state_tuple(g) != before[i]:
                self._mark_dirty(i, payload=False, metadata=True)
                self._append_wal("NODE_META", node_id=i,
                                 metadata=g.get("metadata", {}),
                                 state=list(self._state_tuple(g)))

    def _node_manifest(self, g):
        return {"kind": g.get("kind", "turn"),
                "text": g["text"], "ntok": g["ntok"],
                "sources": g.get("sources", []),
                "retired": bool(g.get("retired")),
                "no_fold": bool(g.get("no_fold")),
                "tags": g.get("tags", []),
                "rare": sorted(g["rare"]),
                "metadata": g.get("metadata", self._default_metadata(g)),
                "host_present": bool(g.get("host_present", False)),
                "device_present": bool(g.get("device_present", False)),
                "dirty": bool(g.get("dirty", False)),
                "durable": bool(g.get("durable", False)),
                "cold_only": bool(g.get("cold_only", False)),
                "payload_pending": bool(g.get("payload_pending", False)),
                "native_node_id": g.get("native_node_id"),
                "provenance": g.get("provenance", [])}

    def flush_async(self):
        """Start an async RAM-payload durability flush."""
        self._sync_lifecycle()
        queued = {"queued_nodes": len(self.dirty_nodes),
                  "dirty_bytes": sum(self._host_payload_bytes(
                      self.arena.grafts[i]) for i in self.dirty_nodes),
                  "mode": "threaded"}
        if self._flush_thread and self._flush_thread.is_alive():
            queued["running"] = True
            return queued

        def worker():
            try:
                self.flush_now()
            except Exception as exc:  # surfaced through flush_wait()
                self._flush_error = exc

        self._flush_error = None
        self._flush_thread = threading.Thread(target=worker, daemon=True)
        self._flush_thread.start()
        queued["running"] = True
        return queued

    def flush_wait(self, timeout=None):
        t = self._flush_thread
        if t is not None:
            t.join(timeout)
            if t.is_alive():
                return False
        if self._flush_error is not None:
            err = self._flush_error
            self._flush_error = None
            raise err
        return True

    def flush_now(self):
        with self._flush_lock:
            self._sync_lifecycle()
            nodes = []
            self._ensure_repo_dirs()
            for i, g in enumerate(self.arena.grafts):
                f = os.path.join(self.path, "nodes", f"{i:04d}.npz")
                dirty = self.dirty_nodes.get(i, {})
                needs_payload = (dirty.get("payload") or not g.get("durable"))
                # A WAL-recovered node is text/metadata authoritative but has no
                # K/V payload (the cache was never checkpointed). It stays in the
                # manifest as payload_pending and is NOT marked durable; trying to
                # snapshot a payload it never had would abort the whole flush.
                no_payload = (g.get("host_payload") is None
                              and g.get("h") is None and not g.get("durable"))
                if needs_payload and g.get("payload_pending") and no_payload:
                    needs_payload = False
                if needs_payload:
                    if g.get("host_payload") is None:
                        self._ensure_host_payload(i, g)
                    self._native_sync_node(i)
                    self._atomic_savez_compressed(f, **g["host_payload"])
                    g["durable"] = True
                    g["payload_pending"] = False
                    self._native_mark_durable(i)
                if "rare" not in g:
                    g["rare"] = ArenaCache._rare_tokens(g["text"])
                g["dirty"] = False
                self._ensure_lifecycle(i, g)
                nodes.append(self._node_manifest(g))
            self._atomic_savez_compressed(
                os.path.join(self.path, "index.npz"), **self.arena.pack_index())
            native_checkpoint = self._native_save_checkpoint()
            self._atomic_write_json(
                os.path.join(self.path, "manifest.json"),
                {"dialect": self.dialect,
                 "dialect_descriptor": self.dialect_desc.to_json(),
                 "route_layer": self.arena.route_layer,
                 "durability_mode": self.durability_mode,
                 "native_checkpoint": (
                     "native/grm_store.bin" if native_checkpoint else None),
                 "wal_lsn": self._wal_lsn,
                 "review_buffer": self.review_buffer,
                 "nodes": nodes},
                indent=1)
            self.dirty_nodes.clear()
            self._append_wal("CHECKPOINT", nodes=len(nodes),
                             manifest="manifest.json")

    def save(self):
        """Compatibility alias for the old persistence entry point."""
        return self.flush_now()

    def load(self):
        with open(os.path.join(self.path, "manifest.json")) as fh:
            man = json.load(fh)
        if man["dialect"] != self.dialect:
            raise RuntimeError(
                f"dialect wall: repository was harvested on {man['dialect']!r}, "
                f"this model is {self.dialect!r} — K/V artifacts never transfer "
                f"across models (texts survive; re-harvest to migrate)")
        if man["route_layer"] != self.arena.route_layer:
            raise RuntimeError(
                f"routing index was built at layer {man['route_layer']}, "
                f"arena is configured for {self.arena.route_layer}")
        self._set_durability_mode_fields(
            man.get("durability_mode", self.durability_mode))
        idx = np.load(os.path.join(self.path, "index.npz"))
        self.arena.grafts = []
        self._native_node_ids = {}
        for i, n in enumerate(man["nodes"]):
            g = {"kind": n["kind"], "text": n["text"], "ntok": n["ntok"],
                 "sources": n["sources"], "retired": n["retired"],
                 "no_fold": n.get("no_fold", False),
                 "tags": n["tags"], "rare": set(n["rare"]),
                 "cent": self.arena.unpack_index(idx, i),
                 "metadata": n.get("metadata", {}),
                 "provenance": n.get("provenance", []),
                 "host_payload": None,
                 "host_present": n.get("host_present", False),
                 "dirty": False,
                 "durable": n.get("durable", True),
                 "saved": n.get("durable", True),
                 "payload_pending": n.get("payload_pending", False),
                 "recovered": n.get("payload_pending", False),
                 "h": None}
            if n.get("native_node_id") is not None:
                native_id = int(n["native_node_id"])
                g["native_node_id"] = native_id
                self._native_node_ids[int(i)] = native_id
            # A payload-pending node (WAL-recovered, never re-harvested) has no
            # .npz on disk — keep it text-only rather than reading a missing file.
            if not n["retired"] and not g["payload_pending"]:
                g["host_payload"] = self._read_payload_file(i)
                g["h"] = self.arena.unpack_node(g["host_payload"])
            self._ensure_lifecycle(i, g)
            self.arena.grafts.append(g)
        self._rebuild_child_keys()
        manifest_wal_lsn = int(man.get("wal_lsn", 0))
        self._wal_lsn = max(manifest_wal_lsn, self._wal_lsn)
        self.recovered_wal = self._read_wal()
        self.replayed_config_lsn = self._apply_config_wal_records(
            self.recovered_wal, manifest_wal_lsn)
        self.recovered_nodes = self._recover_wal_summary(self.recovered_wal)
        self.replayed_wal_nodes = self._apply_manifest_wal_records(
            self.recovered_wal, manifest_wal_lsn)
        self.review_buffer = self._apply_review_wal_records(
            man.get("review_buffer", []), self.recovered_wal,
            since_lsn=manifest_wal_lsn)
        self.dirty_nodes.clear()
        native_loaded = self._native_load_checkpoint()
        if native_loaded and not self._native_node_ids:
            for i, g in enumerate(self.arena.grafts):
                g["native_node_id"] = int(i)
                self._native_node_ids[int(i)] = int(i)
        if native_loaded:
            for i in getattr(self, "replayed_wal_nodes", ()):
                g = self.arena.grafts[int(i)]
                self._native_sync_node(
                    int(i), payload_required=g.get("host_payload") is not None)
        if not native_loaded:
            self._sync_native_full()
        self._page()

    def _rebuild_child_keys(self):
        """Descent keys rebuild from lineage (recursive: eras reach leaves)."""
        def cents_of(i, depth=0):
            g = self.arena.grafts[i]
            out = [g["cent"]]
            if depth < 3:
                for s in g.get("sources", ()):
                    out += cents_of(s, depth + 1)
            return out
        for g in self.arena.grafts:
            if g.get("sources"):
                g["child_cents"] = [c for s in g["sources"]
                                    for c in cents_of(s)]

    def migrate(self, src_path):
        """Rebuild THIS (empty) repository from another repository's TEXTS.

        K/V artifacts never cross the dialect wall; text does. Every node —
        turns, documents, digests, eras, retired or not — is re-harvested
        by the resident model under its own weights. Digest/era texts are
        carried VERBATIM (they are text, readable by any dialect); lineage
        (sources), kinds, tags, retirement and fold-exemption flags are
        preserved, so descent keys rebuild exactly. The source's tokenizer
        does not matter: ntok is recounted in the destination's tokens.
        Returns the number of migrated nodes."""
        if self.arena.grafts:
            raise RuntimeError("migrate into an EMPTY repository (got "
                               f"{len(self.arena.grafts)} nodes)")
        with open(os.path.join(src_path, "manifest.json")) as fh:
            man = json.load(fh)
        A = self.arena
        before = self._snapshot_state()
        for n in man["nodes"]:
            gi = A.deposit(n["text"])
            g = A.grafts[gi]
            g["kind"] = n["kind"]
            g["tags"] = n["tags"]
            g["sources"] = n["sources"]
            g["retired"] = n["retired"]
            g["no_fold"] = n.get("no_fold", False)
            # lexical keys are text-derived — recompute, don't copy (the
            # source may predate a tokenizer-rule fix)
            g["rare"] = A._rare_tokens(n["text"])
        self._rebuild_child_keys()
        self._mark_mutations(before)
        self.flush_now()
        self._free_retired()
        self._page()
        return len(A.grafts)

    def stats(self):
        kinds = {}
        for g in self.arena.grafts:
            k = ("retired " if g.get("retired") else "") + g.get("kind", "turn")
            kinds[k] = kinds.get(k, 0) + 1
        dev = [g for g in self.arena.grafts if g.get("h") is not None]
        self._sync_lifecycle()
        out = {"nodes": len(self.arena.grafts), "kinds": kinds,
               "durability_mode": self.durability_mode,
               "wal_enabled": bool(self.wal_enabled),
               "active_device": len(dev),
               "device_mb": round(sum(self._node_bytes(g) for g in dev) / 1e6),
               "ram_payload_mb": round(sum(self._host_payload_bytes(g)
                                           for g in self.arena.grafts) / 1e6),
               "dirty_nodes": len(self.dirty_nodes),
               "durable_nodes": sum(1 for g in self.arena.grafts
                                    if g.get("durable")),
               "cold_nodes": sum(1 for g in self.arena.grafts
                                 if g.get("cold_only")),
               "recovered_wal_records": len(getattr(self, "recovered_wal", [])),
               "recovered_nodes": len(getattr(self, "recovered_nodes", [])),
               "replayed_wal_nodes": len(getattr(
                   self, "replayed_wal_nodes", ())),
               "replayed_config_records": len(getattr(
                   self, "replayed_config_lsn", ())),
               "page_ins": getattr(self.arena, "page_ins", 0),
               "folds_aborted": getattr(self, "folds_aborted", 0),
               "no_fold": sum(1 for g in self.arena.grafts
                              if g.get("no_fold"))}
        if self.native_store is not None:
            ns = self.native_store.stats()
            out["native"] = {
                "nodes": ns.nodes,
                "dirty_nodes": ns.dirty_nodes,
                "durable_nodes": ns.durable_nodes,
                "host_payload_bytes": ns.host_payload_bytes,
                "host_payload_tensors": getattr(ns, "host_payload_tensors", 0),
                "route_entries": ns.route_entries,
                "checkpoint_loaded": bool(self._native_checkpoint_loaded),
            }
        return out
