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
import json
import os

import numpy as np

from core.graft_arena import ArenaCache


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
                 arena_cls=ArenaCache, **arena_kw):
        self.path = path
        self.autosave = autosave
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
        cfg = model.config
        # the dialect string names the K/V geometry: MLA models by latent
        # rank, GQA models by kv-head shape
        r = getattr(cfg, "kv_lora_rank", None)
        tail = (f"r{r}" if r is not None
                else f"g{cfg.num_kv_heads}x{cfg.head_dim}")
        self.dialect = (f"{type(model).__name__}:{cfg.num_layers}"
                        f"x{cfg.hidden_dim}:{tail}")
        # descent re-mounts retired children from cold storage on demand
        self.arena.node_loader = self._load_node
        if os.path.exists(os.path.join(path, "manifest.json")):
            self.load()
        else:
            os.makedirs(os.path.join(path, "nodes"), exist_ok=True)

    # ----------------------------------------------------------- hot path
    def chat(self, user_text, ngen=64, max_trips=2):
        ans, info = self.arena.step(user_text, ngen=ngen, max_trips=max_trips)
        self._librarian()
        if self.autosave:
            self.save()
        self._page()
        return ans, info

    def add_turn(self, user, assistant):
        """Deposit an already-complete turn (scripted or externally run)."""
        self.arena.feed(f"User: {user}\nAssistant: {assistant}\n")
        self._librarian()
        self._page()

    def add_document(self, text, tags=()):
        idx = self.arena.deposit(text)
        g = self.arena.grafts[idx]
        g["kind"] = "doc"
        g["tags"] = list(tags)
        self._page()
        return idx

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
        if self.DIGESTS_HIGH:
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
        done = 0
        while done < max_jobs and self._fold_once():
            done += 1
        if done:
            if self.autosave:
                self.save()
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

    def _page(self):
        """Spill least-recently-mounted node tensors above the VRAM budget.
        Only saved nodes spill (disk must hold them); unsaved nodes are
        saved first when autosave is off would be unsafe — they just stay
        resident until the next save()."""
        if self.vram_budget is None:
            return 0
        resident = [g for g in self.arena.grafts if g.get("h") is not None]
        if (sum(self._node_bytes(g) for g in resident) > self.vram_budget
                and any(not g.get("saved") for g in resident)):
            self.save()        # spill is a write-back; disk must hold it
        live = [(g.get("last_used", 0), g) for g in self.arena.grafts
                if g.get("h") is not None and g.get("saved")]
        used = sum(self._node_bytes(g) for _, g in live)
        freed = 0
        for _, g in sorted(live, key=lambda x: x[0]):
            if used <= self.vram_budget:
                break
            used -= self._node_bytes(g)
            g["h"] = None
            freed += 1
        return freed

    def _free_retired(self):
        """Retired nodes leave VRAM; disk is their cold storage."""
        for g in self.arena.grafts:
            if g.get("retired") and g.get("h") is not None and g.get("saved"):
                g["h"] = None

    # --------------------------------------------------------- persistence
    def _load_node(self, i):
        """Cold-storage loader: node tensors disk -> device (descent hook).
        The on-disk format is the arena dialect's (pack/unpack_node)."""
        z = np.load(os.path.join(self.path, "nodes", f"{i:04d}.npz"))
        return self.arena.unpack_node(z)

    def save(self):
        nodes = []
        for i, g in enumerate(self.arena.grafts):
            f = os.path.join(self.path, "nodes", f"{i:04d}.npz")
            if not g.get("saved") and g.get("h") is not None:
                np.savez_compressed(f, **self.arena.pack_node(g["h"]))
                g["saved"] = True
            if "rare" not in g:    # never routed yet — lexical keys must persist
                g["rare"] = ArenaCache._rare_tokens(g["text"])
            nodes.append({"kind": g.get("kind", "turn"),
                          "text": g["text"], "ntok": g["ntok"],
                          "sources": g.get("sources", []),
                          "retired": bool(g.get("retired")),
                          "no_fold": bool(g.get("no_fold")),
                          "tags": g.get("tags", []),
                          "rare": sorted(g["rare"])})
        np.savez_compressed(os.path.join(self.path, "index.npz"),
                            **self.arena.pack_index())
        with open(os.path.join(self.path, "manifest.json"), "w") as fh:
            json.dump({"dialect": self.dialect,
                       "route_layer": self.arena.route_layer,
                       "nodes": nodes}, fh, indent=1)

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
                 "saved": True, "h": None}
            if not n["retired"]:
                g["h"] = self._load_node(i)
            self.arena.grafts.append(g)
        self._rebuild_child_keys()

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
        self.save()
        self._free_retired()
        self._page()
        return len(A.grafts)

    def stats(self):
        kinds = {}
        for g in self.arena.grafts:
            k = ("retired " if g.get("retired") else "") + g.get("kind", "turn")
            kinds[k] = kinds.get(k, 0) + 1
        dev = [g for g in self.arena.grafts if g.get("h") is not None]
        return {"nodes": len(self.arena.grafts), "kinds": kinds,
                "active_device": len(dev),
                "device_mb": round(sum(self._node_bytes(g) for g in dev) / 1e6),
                "page_ins": getattr(self.arena, "page_ins", 0),
                "folds_aborted": getattr(self, "folds_aborted", 0),
                "no_fold": sum(1 for g in self.arena.grafts
                               if g.get("no_fold"))}
