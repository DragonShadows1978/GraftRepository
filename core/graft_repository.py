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
  repo.save() / repo.stats()
"""
from dataclasses import asdict, dataclass
import json
import os
import threading
import time

import numpy as np

from core.graft_arena import ArenaCache


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

    @classmethod
    def from_model(cls, model, arena):
        cfg = model.config
        route_layer = int(getattr(arena, "route_layer", 0))
        if hasattr(cfg, "kv_lora_rank"):
            latent_rank = int(cfg.kv_lora_rank)
            rope_dim = int(getattr(cfg, "qk_rope_head_dim", 0))
            vals = latent_rank + rope_dim
            return cls(type(model).__name__, int(cfg.num_layers),
                       int(cfg.hidden_dim), "mla", vals, route_layer,
                       latent_rank=latent_rank, rope_dim=rope_dim)
        num_kv_heads = int(getattr(cfg, "num_kv_heads", 0))
        head_dim = int(getattr(cfg, "head_dim", 0))
        vals = int(getattr(arena, "VALS_PER_TOK_LAYER",
                           num_kv_heads * head_dim * 2))
        return cls(type(model).__name__, int(cfg.num_layers),
                   int(cfg.hidden_dim), "gqa", vals, route_layer,
                   num_kv_heads=num_kv_heads, head_dim=head_dim)

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

    def __init__(self, model, encode, decode, path, autosave=True,
                 vram_budget_mb=None, librarian_mode="inline",
                 arena_cls=ArenaCache, durability_mode="session_safe",
                 wal_enabled=None, native_store=None, native_lib_path=None,
                 native_enabled=False, **arena_kw):
        self.path = path
        self.autosave = autosave
        self.durability_mode = durability_mode
        self.wal_enabled = (durability_mode in ("session_safe", "project_safe",
                                                "durable_strict")
                            if wal_enabled is None else bool(wal_enabled))
        self._flush_lock = threading.RLock()
        self._flush_thread = None
        self._flush_error = None
        self._wal_lsn = 0
        self.review_buffer = []
        self.fold_history = []
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
            self.recovered_nodes = self._recover_wal_summary(
                self.recovered_wal)
            self._rehydrate_from_wal(self.recovered_nodes)
            self._sync_lifecycle()
            self.dirty_nodes.clear()

    def close(self):
        if self._own_native_store and self.native_store is not None:
            self.native_store.close()
        self.native_store = None
        self.arena.native_store = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ----------------------------------------------------------- hot path
    def chat(self, user_text, ngen=64, max_trips=2):
        before = self._snapshot_state()
        ans, info = self.arena.step(user_text, ngen=ngen, max_trips=max_trips)
        self._librarian()
        self._mark_mutations(before)
        if self.autosave:
            self.flush_now()
        self._page()
        return ans, info

    def add_turn(self, user, assistant):
        """Deposit an already-complete turn (scripted or externally run)."""
        before = self._snapshot_state()
        self.arena.feed(f"User: {user}\nAssistant: {assistant}\n")
        self._set_new_node_provenance(before, "exchange_span")
        self._librarian()
        self._mark_mutations(before)
        self._page()

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
    def _parse_memory_command_python(text):
        original = text.strip()
        low = original.lower()
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
        plan = self._parse_memory_command(text)
        action = plan.get("action")
        if action == "remember":
            opts = {k: plan[k] for k in ("durability", "mutability",
                                         "scope", "kind")
                    if plan.get(k)}
            idx = self.remember(plan.get("body", ""), **opts)
            if plan.get("flush_immediately"):
                self.flush_now()
            return {"action": "remember", "node_id": idx}
        if action == "forget":
            return {"action": "forget",
                    "count": self.forget(plan.get("query", ""))}
        if action == "correct":
            return {"action": "correct",
                    "node_id": self.correct_memory(
                        plan.get("query", ""), plan.get("replacement", ""))}
        if action == "review":
            self.review_candidate(plan.get("body", ""),
                                  action="review_candidate",
                                  reason=plan.get(
                                      "reason",
                                      "correction missing => separator"))
            return {"action": "review", "count": len(self.review_buffer)}
        if action == "ignore":
            self._append_wal("DO_NOT_REMEMBER", text=text)
            return {"action": "ignore"}
        if action == "flush":
            self.flush_now()
            return {"action": "flush"}
        raise ValueError(f"unknown memory command: {text!r}")

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
        keys = ("subject", "predicate", "value", "valid_from", "expires_at")
        meta = {k: candidate[k] for k in keys if k in candidate}
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
            old_value = self._norm_fact_field(meta.get("value"))
            if old_value and old_value != value:
                out.append(i)
        return out

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
        if action in ("expire",):
            return self._candidate_to_review(
                candidate, text, "expire action requires explicit policy",
                metadata)

        conflicts = self._candidate_conflicts(candidate)
        authoritative = write_intent in ("user_asserted", "system_asserted")
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

        supersedes = conflicts if conflicts else list(
            candidate.get("supersedes", []))
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
        return [self.apply_extraction_candidate(
            c, source_text=source_text, source_turns=source_turns,
            source_grafts=source_grafts,
            write_direct_threshold=write_direct_threshold)
                for c in candidates]

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
                "reason": reason}
        if metadata:
            item["metadata"] = dict(metadata)
        self.review_buffer.append(item)
        self._append_wal("REVIEW_CANDIDATE", **item)
        return item["id"]

    def approve_review(self, review_id):
        item = self.review_buffer[review_id]
        idx = self.remember(item["text"], durability=item["proposed_durability"],
                            mutability=item["proposed_mutability"],
                            scope=item["proposed_scope"],
                            kind=item["proposed_kind"],
                            confidence=item["confidence"],
                            metadata=item.get("metadata"))
        item["approved_node_id"] = idx
        self._append_wal("REVIEW_APPROVE", review_id=review_id, node_id=idx)
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
        before = self._snapshot_state()
        done = 0
        while done < max_jobs and self._fold_once():
            done += 1
        if done:
            self._mark_mutations(before)
            if self.autosave:
                self.flush_now()
            self._page()
        return done

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
        if self.dialect_desc.payload_kind != "mla":
            raise RuntimeError("native GRM host store currently exposes only "
                               "the MLA C ABI")
        from core.grm_native import NativeGraftStore
        return NativeGraftStore(
            lib_path, model_type=self.dialect_desc.model_type,
            num_layers=self.dialect_desc.num_layers,
            hidden_dim=self.dialect_desc.hidden_dim,
            vals_per_tok_layer=self.dialect_desc.vals_per_tok_layer,
            route_layer=self.dialect_desc.route_layer,
            latent_rank=self.dialect_desc.latent_rank,
            rope_dim=self.dialect_desc.rope_dim)

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
        if hasattr(self.native_store, "set_route_keys"):
            self.native_store.set_route_keys(
                node_id, np.stack(route_keys), self._native_lexical_keys(g))
        else:
            self.native_store.set_route(
                node_id, route_keys[0].tolist(), self._native_lexical_keys(g))

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

    def _recover_wal_summary(self, records):
        nodes = {}
        reviews = []

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
            elif typ == "REVIEW_CANDIDATE":
                reviews.append({k: v for k, v in rec.items()
                                if k not in ("lsn", "type", "time")})
        self.recovered_reviews = reviews
        return [nodes[k] for k in sorted(nodes)]

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
            meta = dict(n.get("metadata") or {})
            retired = bool(n.get("retired", not bool(n.get("active", True))))
            meta.setdefault("kind", n.get("kind", "turn"))
            meta["active"] = not retired
            sources = list(n.get("sources", meta.get("source_grafts", [])) or [])
            tags = list(n.get("tags", meta.get("tags", [])) or [])
            no_fold = bool(n.get("no_fold", meta.get("no_fold", False)))
            if no_fold:
                meta["no_fold"] = True
            g = {
                "kind": n.get("kind", "turn"),
                "text": n.get("text", ""),
                "ntok": int(meta.get("ntok", 0) or 0),
                "sources": sources,
                "retired": retired,
                "no_fold": no_fold,
                "tags": tags,
                "rare": set(),
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
                    np.savez_compressed(f, **g["host_payload"])
                    g["durable"] = True
                    g["payload_pending"] = False
                    self._native_mark_durable(i)
                if "rare" not in g:
                    g["rare"] = ArenaCache._rare_tokens(g["text"])
                g["dirty"] = False
                self._ensure_lifecycle(i, g)
                nodes.append(self._node_manifest(g))
            np.savez_compressed(os.path.join(self.path, "index.npz"),
                                **self.arena.pack_index())
            with open(os.path.join(self.path, "manifest.json"), "w") as fh:
                json.dump({"dialect": self.dialect,
                           "dialect_descriptor": self.dialect_desc.to_json(),
                           "route_layer": self.arena.route_layer,
                           "durability_mode": self.durability_mode,
                           "wal_lsn": self._wal_lsn,
                           "review_buffer": self.review_buffer,
                           "nodes": nodes}, fh, indent=1)
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
        idx = np.load(os.path.join(self.path, "index.npz"))
        self.arena.grafts = []
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
            # A payload-pending node (WAL-recovered, never re-harvested) has no
            # .npz on disk — keep it text-only rather than reading a missing file.
            if not n["retired"] and not g["payload_pending"]:
                g["host_payload"] = self._read_payload_file(i)
                g["h"] = self.arena.unpack_node(g["host_payload"])
            self._ensure_lifecycle(i, g)
            self.arena.grafts.append(g)
        self._rebuild_child_keys()
        self.review_buffer = man.get("review_buffer", [])
        self._wal_lsn = max(int(man.get("wal_lsn", 0)), self._wal_lsn)
        self.recovered_wal = self._read_wal()
        self.recovered_nodes = self._recover_wal_summary(self.recovered_wal)
        self.dirty_nodes.clear()
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
            }
        return out
