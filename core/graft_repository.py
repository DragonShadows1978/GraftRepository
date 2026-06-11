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
from core.mistral7b_tc import BlockTC, tc


class GraftRepository:
    # librarian thresholds: consolidate when this many ACTIVE nodes of a
    # kind are older than the live window; how many to fold per pass.
    # ERA FOLDING DEFAULT-OFF (DIGESTS_HIGH=None): measured 2026-06-10,
    # neither digest-of-digest form works at 4B — list-style eras STRIP
    # relations (probes bleed across facts), chronicle-prose eras INVENT
    # them ("NIGHTJAR was conducted by Priya" — fact fusion). The right
    # mechanism is DESCENT: route into the era, re-mount its child digests
    # on grounding failure — not yet implemented. Digests are ~100 tokens;
    # letting them accumulate is cheap and validated (E4-C 6/6).
    TURNS_HIGH, TURNS_FOLD = 8, 4
    DIGESTS_HIGH, DIGESTS_FOLD = None, 3

    def __init__(self, model, encode, decode, path, autosave=True, **arena_kw):
        self.path = path
        self.autosave = autosave
        self.arena = ArenaCache(model, encode, decode, **arena_kw)
        cfg = model.config
        self.dialect = (f"{type(model).__name__}:{cfg.num_layers}"
                        f"x{cfg.hidden_dim}:r{cfg.kv_lora_rank}")
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
        return ans, info

    def add_turn(self, user, assistant):
        """Deposit an already-complete turn (scripted or externally run)."""
        self.arena.feed(f"User: {user}\nAssistant: {assistant}\n")
        self._librarian()

    def add_document(self, text, tags=()):
        idx = self.arena.deposit(text)
        g = self.arena.grafts[idx]
        g["kind"] = "doc"
        g["tags"] = list(tags)
        return idx

    # ---------------------------------------------------------- librarian
    def _active(self, kinds):
        live = {gi for gi, _ in self.arena.live_segs if gi is not None}
        return [i for i, g in enumerate(self.arena.grafts)
                if not g.get("retired") and i not in live
                and g.get("kind", "turn") in kinds]

    def _librarian(self):
        """Threshold-triggered consolidation. Documents are never folded —
        they are reference material, not history."""
        turns = self._active(("turn",))
        if len(turns) >= self.TURNS_HIGH:
            didx, _ = self.arena.consolidate(turns[:self.TURNS_FOLD])
            self.arena.grafts[didx]["kind"] = "digest"
            self._free_retired()
        if self.DIGESTS_HIGH:
            digests = self._active(("digest",))
            if len(digests) >= self.DIGESTS_HIGH:
                eidx, _ = self.arena.consolidate(digests[:self.DIGESTS_FOLD])
                self.arena.grafts[eidx]["kind"] = "era"
                self._free_retired()

    def _free_retired(self):
        """Retired nodes leave VRAM; disk is their cold storage."""
        for g in self.arena.grafts:
            if g.get("retired") and g.get("h") is not None and g.get("saved"):
                g["h"] = None

    # --------------------------------------------------------- persistence
    def _host(self, h):
        """Per-layer device dicts -> stacked host arrays (L,S,256)/(L,S,32)."""
        c = np.concatenate([d["c"].float().numpy().astype(np.float16)
                            for d in h], axis=0)
        kpe = np.concatenate([d["kpe"].float().numpy().astype(np.float16)[:, 0]
                              for d in h], axis=0)
        return c, kpe

    def save(self):
        nodes = []
        for i, g in enumerate(self.arena.grafts):
            f = os.path.join(self.path, "nodes", f"{i:04d}.npz")
            if not g.get("saved") and g.get("h") is not None:
                c, kpe = self._host(g["h"])
                np.savez_compressed(f, c=c, kpe=kpe)
                g["saved"] = True
            if "rare" not in g:    # never routed yet — lexical keys must persist
                g["rare"] = ArenaCache._rare_tokens(g["text"])
            nodes.append({"kind": g.get("kind", "turn"),
                          "text": g["text"], "ntok": g["ntok"],
                          "sources": g.get("sources", []),
                          "retired": bool(g.get("retired")),
                          "tags": g.get("tags", []),
                          "rare": sorted(g["rare"])})
        cents = np.stack([g["cent"] for g in self.arena.grafts]) \
            if self.arena.grafts else np.zeros((0, 256), np.float32)
        np.savez_compressed(os.path.join(self.path, "index.npz"), cents=cents)
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
        cents = np.load(os.path.join(self.path, "index.npz"))["cents"]
        dt = BlockTC.COMPUTE_DTYPE
        nl = len(self.arena.m.layers)
        self.arena.grafts = []
        for i, n in enumerate(man["nodes"]):
            g = {"kind": n["kind"], "text": n["text"], "ntok": n["ntok"],
                 "sources": n["sources"], "retired": n["retired"],
                 "tags": n["tags"], "rare": set(n["rare"]),
                 "cent": cents[i].astype(np.float32), "saved": True, "h": None}
            if not n["retired"]:
                z = np.load(os.path.join(self.path, "nodes", f"{i:04d}.npz"))
                c, kpe = z["c"], z["kpe"]
                g["h"] = [{"c": tc.tensor(np.ascontiguousarray(
                               c[li:li + 1])).astype(dt),
                           "kpe": tc.tensor(np.ascontiguousarray(
                               kpe[li:li + 1][None])).astype(dt)}
                          for li in range(nl)]
            self.arena.grafts.append(g)
        # descent keys rebuild from lineage (recursive: eras reach leaves)
        def cents_of(i, depth=0):
            g = self.arena.grafts[i]
            out = [g["cent"]]
            if depth < 3:
                for s in g["sources"]:
                    out += cents_of(s, depth + 1)
            return out
        for g in self.arena.grafts:
            if g["sources"]:
                g["child_cents"] = [c for s in g["sources"]
                                    for c in cents_of(s)]

    def stats(self):
        kinds = {}
        for g in self.arena.grafts:
            k = ("retired " if g.get("retired") else "") + g.get("kind", "turn")
            kinds[k] = kinds.get(k, 0) + 1
        return {"nodes": len(self.arena.grafts), "kinds": kinds,
                "active_device": sum(1 for g in self.arena.grafts
                                     if g.get("h") is not None)}
