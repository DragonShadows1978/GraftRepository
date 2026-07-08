"""Graft Arena: persistent routed conversation memory on one live KV cache.

The Phase-1 seating plan, realized for MiniCPM3's MLA latent cache:

    seats [0 .. n_sink)              SINK   permanent graft, never touched
    seats [n_sink .. n_sink+width)   ARENA  mounts occupy a PREFIX; the
                                            unused remainder is a positional
                                            hole (free_seats finding)
    seats [n_sink+width .. )         LIVE   conversation tokens, recency-
                                            windowed by EVICTION

Three operations, none of which ever re-prefills the conversation:
  swap(picks)   — cache SURGERY: replace the arena slice of every layer's
                  (c_n, k_pe) with the new grafts. The latent is position-
                  free; only the 32-dim shared k_pe re-RoPEs (at its arena
                  seats). Live tokens keep their baked positions.
  step(...)     — route (bare user text, latent-centroid cosine), swap,
                  prefill the turn, greedy-decode, evict, deposit.
  evict()       — drop live segments older than the recency window from the
                  cache. Remaining tokens keep their positions (holes are
                  fine); evicted content survives only as haunting + its
                  deposited graft (selective-amnesia semantics, by design).

Position law: live token positions = live_shift + running counter, where
live_shift = n_sink + arena_width is FIXED for the cache's lifetime (the
`live_shift` attribute on MLAAttentionTC — decoupled from mount size).
"""
import gc
import os
import re

import numpy as np

from core.mistral7b_tc import BlockTC, F, tc
from core import kv_graft
from core.graft_quant import (
    SUPPORTED_BITS, is_packed_payload, pack_kv_arrays, unpack_kv_arrays,
)


class ArenaCache:
    def __init__(self, model, encode, decode, sink_text="<conversation>\n",
                 arena_width=256, route_layer=44, topk=3, live_turns=2,
                 max_live=4096, cache_deposits=True,
                 ephemeral=False, recency_mounts=2, prompt_template=None,
                 stop_sequences=None):
        # EPHEMERAL MODE ("clear the boat"): the live cache is reset at the
        # START of every turn — each turn runs on [sink | mounts | turn]
        # alone, so resident seats are CONSTANT for a conversation of ANY
        # length (the context window IS the repository). Recency becomes a
        # MOUNT: the last `recency_mounts` turn-grafts are always co-seated
        # for discourse cohesion (anaphora), ~40 seats instead of a growing
        # live region. Side effect: the live-window echo failure class
        # (corpus-100) cannot occur — there is no window to echo from.
        self.ephemeral = ephemeral
        self.recency_mounts = recency_mounts
        self.m = model
        self.encode = encode            # text -> list of token ids
        self.decode = decode            # list of token ids -> text
        self.width = arena_width
        self.route_layer = route_layer
        self.topk = topk
        self.live_turns = live_turns
        self.cache_deposits = cache_deposits
        self.prompt_template = prompt_template
        self.stop_sequences = tuple(
            stop_sequences or
            ("\nUser:", "User:", "\nAssistant:", "Assistant:", "\n\n"))
        self.dt = BlockTC.COMPUTE_DTYPE
        # the model auto-extends RoPE only to position_offset+L; arena
        # positions run live_shift further. Extend once, up front.
        model.extend_rope(len(encode(sink_text)) + arena_width + max_live)

        sink_ids = encode(sink_text)
        self.sink_h = self._harvest(sink_ids)
        self.n_sink = len(sink_ids)
        self.live_shift = self.n_sink + arena_width

        self.node_loader = None         # callable(idx) -> device h; lets
                                        # DESCENT re-mount retired children
                                        # from cold storage (repository disk)
        self.native_store = None        # optional C++ host runtime mirror
        self.caches = None              # per layer (c_n, k_pe), built on turn 1
        self.pos = 0                    # live tokens processed (position counter)
        self.cur_mounts = []            # graft idxs currently seated
        self.cur_mount_n = 0            # arena seats currently occupied
        self.live_segs = []             # [(graft_idx or None, ntok), ...]
        self.grafts = []                # {h, cent, ntok, text}
        self.last_route_backend = "python"

    def _format_step_prompt(self, user_text):
        if self.prompt_template is None:
            return f"User: {user_text}\nAssistant:"
        if callable(self.prompt_template):
            return self.prompt_template(user_text, None)
        return self.prompt_template.format(user=user_text, assistant="")

    def _format_step_turn(self, user_text, assistant_text):
        if self.prompt_template is None:
            return f"User: {user_text}\nAssistant: {assistant_text}\n"
        if callable(self.prompt_template):
            return self.prompt_template(user_text, assistant_text)
        return self.prompt_template.format(user=user_text,
                                           assistant=assistant_text)

    def _clear_transients(self):
        gc.collect()
        if hasattr(tc, "empty_cache"):
            tc.empty_cache()

    def _bump_cuda_gqa_epoch(self):
        """Mutation-epoch bump, shared by both dialects.

        Historically a no-op on the MLA (base-class) dialect — GQAArenaCache
        overrode it with the real counter driving its opt-in CUDA route bank
        cache. MLA CUDA route P1 (docs/GRM_MLA_CUDA_ROUTE_PLAN.md) makes it
        real here too: `_native_to_idx_cache` (the epoch-cached full
        native_node_id -> graft-index map used by `_native_route_order`)
        keys on this same counter. Defined at every self.grafts mutation
        site shared by both dialects (deposit, deposit_from_cache, step's
        rollback/restore) plus every graft_repository.py mutation site
        (forget/correct_memory/migrate/cull_graft/...), so those call sites
        never needed dialect-specific branching and still don't."""
        self._cuda_gqa_epoch = getattr(self, "_cuda_gqa_epoch", 0) + 1

    def reset_live_cache(self):
        """Drop the live device cache without touching persisted graft nodes."""
        self.caches = None
        self.pos = 0
        self.cur_mounts = []
        self.cur_mount_n = 0
        self.live_segs = []
        kv_graft.clear_injection(self.m)
        self._clear_transients()

    # ------------------------------------------------------ dialect surface
    # Everything model-specific lives behind these members. The base class
    # IS the MLA dialect (MiniCPM3: latent payload, latent-cosine router);
    # GQAArenaCache at the bottom of this file overrides them for Qwen3.
    PAYLOAD = (("c", 1), ("kpe", 2))    # payload tensors: (key, seq dim)
    ROPE_KEYS = ("kpe",)
    ROPE_PAIR_SWAP = False
    VALS_PER_TOK_LAYER = 288            # c 256 + kpe 32 (node VRAM math)

    def _harvest(self, ids, layer_filter=None, max_layers=None):
        return kv_graft.harvest_kv_mla(self.m, ids, layer_filter=layer_filter,
                                       max_layers=max_layers)

    def _probe_key(self, text):
        """Routing key for a PROBE (bare user text)."""
        pl = self._harvest(self.encode(text), layer_filter={self.route_layer},
                           max_layers=self.route_layer + 1)
        p = pl[self.route_layer]["c"][0].astype(np.float32).mean(0)
        del pl
        self._clear_transients()
        return p / (np.linalg.norm(p) + 1e-8)

    def _node_key(self, text, h_host=None):
        """Routing key for a NODE (pass h_host to reuse a full harvest)."""
        if h_host is None:
            h_host = self._harvest(self.encode(text),
                                   layer_filter={self.route_layer},
                                   max_layers=self.route_layer + 1)
        return kv_graft.latent_centroid(h_host, self.route_layer)

    def _key_score(self, pkey, nkey):
        """Latent cos lives in ~[0.4, 0.9] — the lexical channel's +1 per
        full identifier match dominates BY CALIBRATION. Any dialect's score
        must stay in O(1) cosine range or routing law (2) breaks."""
        return float(np.dot(pkey, nkey))

    def _pair_swap_last(self, x):
        d = x.shape[-1]
        return x.reshape(list(x.shape[:-1]) + [d // 2, 2]).transpose(
            -1, -2).reshape(list(x.shape))

    def _rope_tensor(self, x, pos0, inverse=False, pair_swap=False):
        if hasattr(tc, "rope_apply"):
            with tc.no_grad():
                return tc.rope_apply(x, self.m.rope_cos, self.m.rope_sin,
                                     int(pos0), bool(inverse), bool(pair_swap))
        if pair_swap:
            x = self._pair_swap_last(x)
        L = x.shape[-2]
        cs = self.m.rope_cos.slice(0, pos0, L)
        sn = self.m.rope_sin.slice(0, pos0, L)
        if inverse:
            sn = sn * -1.0
        return F.apply_rotary(x, cs, sn)

    def _rope_block_at(self, blk, pos0, inverse=False):
        """Apply rotation slices to the POSITIONAL key component of a
        payload block (MLA: the 32-d shared k_pe; the latent is
        position-free). Direction is the caller's: re-RoPE at mount seats,
        or un-RoPE a cache slice with -sin (rotation composition)."""
        for key in self.ROPE_KEYS:
            if key in blk:
                blk[key] = self._rope_tensor(
                    blk[key], pos0, inverse, self.ROPE_PAIR_SWAP)
        return blk

    def _export_cache_tensor(self, key, tensor, dim, start, n, pos0):
        if key in self.ROPE_KEYS:
            if hasattr(tc, "export_rope_rows"):
                with tc.no_grad():
                    return tc.export_rope_rows(
                        tensor, self.m.rope_cos, self.m.rope_sin,
                        dim, start, n, pos0, True, self.ROPE_PAIR_SWAP)
            seg = tensor.slice(dim, start, n)
            return self._rope_tensor(
                seg, pos0, inverse=True, pair_swap=self.ROPE_PAIR_SWAP)
        if hasattr(tc, "export_rows"):
            with tc.no_grad():
                return tc.export_rows(tensor, dim, start, n)
        return tensor.slice(dim, start, n)

    def _export_cache_payload(self, cache, n, pos0):
        if len(self.PAYLOAD) == 2 and hasattr(tc, "export_row_pair"):
            raw = [(i, key, dim) for i, (key, dim) in enumerate(self.PAYLOAD)
                   if key not in self.ROPE_KEYS]
            rope = [(i, key, dim) for i, (key, dim) in enumerate(self.PAYLOAD)
                    if key in self.ROPE_KEYS]
            if len(raw) == 1 and len(rope) == 1:
                ri, rkey, rdim = raw[0]
                pi, pkey, pdim = rope[0]
                rt, pt = cache[ri], cache[pi]
                rstart = rt.shape[rdim] - n
                pstart = pt.shape[pdim] - n
                with tc.no_grad():
                    rseg, pseg = tc.export_row_pair(
                        rt, pt, self.m.rope_cos, self.m.rope_sin,
                        rdim, pdim, rstart, pstart, n, pos0, True,
                        self.ROPE_PAIR_SWAP)
                return {rkey: rseg, pkey: pseg}
        seg = {}
        for ei, (key, dim) in enumerate(self.PAYLOAD):
            t = cache[ei]
            S = t.shape[dim]
            seg[key] = self._export_cache_tensor(
                key, t, dim, S - n, n, pos0)
        return seg

    def _paired_export_spec(self):
        if len(self.PAYLOAD) != 2:
            return None
        raw = [(i, key, dim) for i, (key, dim) in enumerate(self.PAYLOAD)
               if key not in self.ROPE_KEYS]
        rope = [(i, key, dim) for i, (key, dim) in enumerate(self.PAYLOAD)
                if key in self.ROPE_KEYS]
        if len(raw) == 1 and len(rope) == 1:
            return raw[0], rope[0]
        return None

    def _export_cache_payloads(self, n, pos0):
        if not hasattr(tc, "export_row_pairs"):
            return None
        spec = self._paired_export_spec()
        if spec is None:
            return None
        (ri, rkey, rdim), (pi, pkey, pdim) = spec
        raw_ts, rope_ts, raw_starts, rope_starts = [], [], [], []
        for cache in self.caches:
            rt, pt = cache[ri], cache[pi]
            raw_ts.append(rt)
            rope_ts.append(pt)
            raw_starts.append(rt.shape[rdim] - n)
            rope_starts.append(pt.shape[pdim] - n)
        with tc.no_grad():
            raw_out, rope_out = tc.export_row_pairs(
                raw_ts, rope_ts, self.m.rope_cos, self.m.rope_sin,
                rdim, pdim, raw_starts, rope_starts, n, pos0, True,
                self.ROPE_PAIR_SWAP)
        return [{rkey: raw_out[i], pkey: rope_out[i]}
                for i in range(len(raw_out))]

    def _cache_key_of(self, seg):
        """key_from_cache=True exploratory mode (measured 5/6 on MLA:
        contextualized keys are polluted). None = standalone _node_key."""
        if not getattr(self, "key_from_cache", False):
            return None
        v = seg["c"].numpy()[0].astype(np.float32).mean(0)
        return v / (np.linalg.norm(v) + 1e-8)

    def _set_inject(self, att, blk):
        att.inject_kv = (blk["c"], blk["kpe"])
        att.graft_seats = int(blk["c"].shape[1])

    def _set_injection_host(self, inj):
        kv_graft.set_injection_mla(self.m, inj)

    def _cache_len(self):
        return self.caches[0][0].shape[self.PAYLOAD[0][1]]

    # persistence pack/unpack: the disk format is part of the dialect
    # (GraftRepository delegates here). The MLA format predates this
    # surface and stays byte-compatible with existing repositories.
    def pack_node(self, h):
        c = np.concatenate([d["c"].float().numpy().astype(np.float16)
                            for d in h], axis=0)
        kpe = np.concatenate([d["kpe"].float().numpy().astype(np.float16)[:, 0]
                              for d in h], axis=0)
        return {"c": c, "kpe": kpe}

    def unpack_node(self, z):
        dt = BlockTC.COMPUTE_DTYPE
        c, kpe = z["c"], z["kpe"]
        return [{"c": tc.tensor(np.ascontiguousarray(c[li:li + 1])).astype(dt),
                 "kpe": tc.tensor(np.ascontiguousarray(
                     kpe[li:li + 1][None])).astype(dt)}
                for li in range(len(self.m.layers))]

    def pack_index(self):
        cents = np.stack([g["cent"] for g in self.grafts]) \
            if self.grafts else np.zeros((0, 256), np.float32)
        return {"cents": cents}

    def unpack_index(self, z, i):
        return z["cents"][i].astype(np.float32)

    # ------------------------------------------------------------ repository
    def deposit(self, text):
        """Standalone harvest deposit (document-in-isolation semantics, one
        dedicated forward). Stored DEVICE-resident: mounts never re-upload."""
        ids = self.encode(text)
        h = self._harvest(ids)
        dev = [{key: tc.tensor(np.ascontiguousarray(h[li][key])).astype(self.dt)
                for key, _ in self.PAYLOAD}
               for li in range(len(self.m.layers))]
        self.grafts.append({"h": dev, "cent": self._node_key(text, h),
                            "ntok": len(ids), "text": text})
        self._bump_cuda_gqa_epoch()
        return len(self.grafts) - 1

    def deposit_from_cache(self, text, seg_ntok):
        """Harvest-on-generate: the live cache ALREADY holds the turn's
        (c_n, k_pe) — slice the span instead of re-forwarding. c_n is
        position-free as-is; k_pe un-RoPEs by rotation composition
        (apply_rotary with -sin at the span's absolute positions).

        MEASURED SPLIT (E4-arena): the K/V PAYLOAD re-mounts fine
        contextualized (verbatim recall wherever routing was right), but a
        centroid from contextualized latents is polluted by the running
        conversation — early turns become routing attractors (5/6, mounts
        collapsed onto turn 1). So the ROUTING KEY comes from a standalone
        partial forward (layers 0..route_layer, no head) unless
        key_from_cache=True (the measured-5/6 exploratory mode)."""
        p0 = self.live_shift + self.pos - seg_ntok      # span's first seat
        dev = self._export_cache_payloads(seg_ntok, p0)
        if dev is None:
            dev = [self._export_cache_payload(cache, seg_ntok, p0)
                   for cache in self.caches]
        cent = None
        for li, seg in enumerate(dev):
            if li == self.route_layer and cent is None:
                cent = self._cache_key_of(seg)
        if cent is None:
            cent = self._node_key(text)
        self.grafts.append({"h": dev, "cent": cent,
                            "ntok": seg_ntok, "text": text})
        self._bump_cuda_gqa_epoch()
        return len(self.grafts) - 1

    def route(self, bare_text, exclude, limit=None):
        if not self.grafts:
            return []
        route_limit = None if limit is None else max(0, int(limit))
        if route_limit == 0:
            return []
        p = self._probe_key(bare_text)

        # Lexical channel: identifier tokens in the probe (codes, numbers,
        # ALL-CAPS) are exact-match keys. Mean centroids CANNOT separate
        # sibling chunks that differ only in a code token (corpus-100:
        # @1 4/20 latent-only — family right, instance random); an exact
        # identifier hit must dominate. _key_score lives in O(1) cosine
        # range (every dialect must keep it there), so +1 per full match
        # wins outright, partial matches rank between.
        #
        # OPTION 1 (query-side lex extension, 2026-07-08): when enabled
        # (default ON; set GRM_ROUTE_QUERY_LEX=0 to disable), content-word
        # tokens from the query also participate — lowercase labels like
        # "orion"/"pin" that never enter _rare_tokens. Match is against
        # candidate NODE TEXT (query-side only; node rare keys / indexes
        # unchanged). Native/CUDA stores only rare keys, so content hits
        # force a Python rescore; when content words hit no candidate the
        # native/CUDA order is kept (lex silent → latent ordering intact).
        qrare = self._rare_tokens(bare_text)
        qlex = (self._query_lex_tokens(bare_text)
                if self._route_query_lex_enabled() else qrare)
        content_extra = qlex - qrare

        # P1 follow-on (profile-named): the eligible BASE (retired/kind
        # filter) depends only on graft state, so it is epoch-cached in
        # _route_cand_base(); only the per-call `exclude` filter runs here.
        # Byte-identical to the old single comprehension: same ascending
        # order, same three membership conditions. When `exclude` is empty
        # the cached base list itself is used — every downstream consumer
        # (_native_route_order, _vector_route_scores, the Python-fallback
        # scoring loops) reads cand without mutating it.
        cand_base = self._route_cand_base()
        if exclude:
            cand = [i for i in cand_base if i not in exclude]
        else:
            cand = cand_base
        # Native lexical channel is rare-key only (node indexes unchanged).
        native_order = self._native_route_order(
            p, qrare, cand, limit=route_limit, exclude=exclude)
        if native_order is not None and not self._query_lex_needs_rescore(
                content_extra, cand):
            return native_order
        self.last_route_backend = "python"
        base = self._vector_route_scores(p, cand)
        if base is None:
            base = {}
            for i in cand:
                score = self._cent_score(p, self.grafts[i])
                if np.isfinite(score):
                    base[i] = score
        # dialect hook: a raw-score channel (GQA layer-0 |q.k|) rescales
        # per-route into the O(1) band the lexical bonus was calibrated
        # against. MLA cosine is already there — identity.
        base = self._normalize_scores(base)
        scored = []
        for i in cand:
            if i not in base:
                continue
            score = base[i] + self._lex_bonus(qlex, self.grafts[i])
            if np.isfinite(score):
                scored.append((score, i))
        scored.sort(key=lambda item: -item[0])
        ranking = [i for _, i in scored]          # best first
        if route_limit is not None:
            return ranking[:route_limit]
        return ranking

    def _vector_route_scores(self, p, cand):
        if (type(self)._key_score is not ArenaCache._key_score
                or type(self)._normalize_scores is not ArenaCache._normalize_scores):
            return None
        rows = []
        row_ids = []
        q = np.asarray(p, dtype=np.float32).reshape(-1)
        for i in cand:
            g = self.grafts[i]
            if g.get("child_cents"):
                return None
            cent = np.asarray(g["cent"], dtype=np.float32).reshape(-1)
            if cent.shape != q.shape:
                return None
            rows.append(cent)
            row_ids.append(i)
        if not rows:
            return {}
        scores = np.stack(rows).astype(np.float32, copy=False) @ q
        return {
            i: float(score)
            for i, score in zip(row_ids, scores)
            if np.isfinite(score)
        }

    def _cent_score(self, p, g):
        # hierarchical descent: a digest node answers for its retired
        # children — score by the best of its own centroid and theirs
        # (a multi-topic digest's own centroid is diluted; the child
        # keys keep it addressable per topic)
        s = self._key_score(p, g["cent"])
        for ch in g.get("child_cents", ()):
            s = max(s, self._key_score(p, ch))
        return s

    def _normalize_scores(self, base):
        return base

    def _lex_bonus(self, qlex, g):
        """Fractional lexical bonus in [0, 1] — same scale as before
        (hits / |query_lex|). Full identifier match still +1. Content-word
        query tokens (when present) match against node text, not only the
        stored rare-key set, so latent ordering still dominates when the
        lex channel is silent (no hits → +0)."""
        if not qlex:
            return 0.0
        if "rare" not in g:
            g["rare"] = self._rare_tokens(g["text"])
        have = set(g["rare"])
        # Query-side residual: tokens that cannot hit stored rare keys
        # (lowercase content words) are matched against node text.
        if not (qlex <= have):
            have |= self._node_text_tokens(g.get("text", ""))
        return len(qlex & have) / len(qlex)

    # Pure stopwords / query scaffolding for the content-word channel.
    # Dialect-generic; deliberately excludes label nouns (orion, pin,
    # cypher, bridge, …). Fact-template glue ("current", "value") is
    # dropped so sibling fact nodes don't share spurious partial credit.
    _QUERY_LEX_STOP = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "am", "do", "does", "did", "will", "would", "could", "should", "may",
        "might", "must", "shall", "can", "to", "of", "in", "on", "at", "for",
        "from", "by", "with", "as", "into", "about", "than", "that", "this",
        "these", "those", "it", "its", "i", "me", "my", "we", "our", "you",
        "your", "he", "she", "they", "them", "their", "what", "which", "who",
        "whom", "whose", "where", "when", "why", "how", "and", "or", "but",
        "not", "no", "nor", "if", "then", "so", "too", "very", "just", "only",
        "also", "any", "all", "each", "few", "more", "most", "other", "some",
        "such", "own", "same", "now", "here", "there", "up", "down", "out",
        "off", "over", "under", "again", "further", "once", "please", "reply",
        "answer", "tell", "recall", "probe", "question", "ask", "asking",
        "user", "assistant", "current", "value",
    })

    @staticmethod
    def _route_query_lex_enabled():
        """Query-side content-word lex extension. Default ON; set
        GRM_ROUTE_QUERY_LEX=0/false/off to restore rare-only behavior."""
        v = os.environ.get("GRM_ROUTE_QUERY_LEX", "1").strip().lower()
        return v not in ("0", "false", "no", "off", "")

    @classmethod
    def _query_content_tokens(cls, text):
        """Lowercase content words from a route query (stopword-filtered).
        Keeps label nouns that _rare_tokens drops (no digit / not ALL-CAPS)."""
        out = set()
        for w in re.findall(r"[A-Za-z0-9][\w:.,\-]*", text):
            tok = w.rstrip(".,:;").lower()
            if len(tok) < 2 or tok in cls._QUERY_LEX_STOP:
                continue
            out.add(tok)
        return out

    @classmethod
    def _query_lex_tokens(cls, text):
        """Full query-side lexical channel: rare identifiers ∪ content words."""
        return cls._rare_tokens(text) | cls._query_content_tokens(text)

    @staticmethod
    def _node_text_tokens(text):
        """Lowercase word tokens from node text for query-side content match.
        Not stored; computed on demand. Node rare keys / indexes untouched."""
        out = set()
        for w in re.findall(r"[A-Za-z0-9][\w:.,\-]*", text or ""):
            tok = w.rstrip(".,:;").lower()
            if tok:
                out.add(tok)
        return out

    def _query_lex_needs_rescore(self, content_extra, cand):
        """True when content-word query tokens hit at least one candidate's
        text — native rare-key scoring cannot see those hits, so Python
        must rescore. Silent content (no hits) keeps native/CUDA order."""
        if not content_extra:
            return False
        for i in cand:
            text = self.grafts[i].get("text", "")
            if content_extra & self._node_text_tokens(text):
                return True
        return False

    def _route_cand_base(self):
        """Epoch-cached eligible-candidate base for route(): every graft
        index that is not retired and not recall-kind, ascending. The two
        fields this reads (`retired`, `kind`) are exactly the eligibility
        fields the GQA route-bank signature walk reads, so every mutation
        site that can change them already bumps `_cuda_gqa_epoch` (W1
        choke points: graft_arena _deposit_consolidation / step's
        kind="recall" flip / rollback truncations; graft_repository
        _mark_mutations / load / _rehydrate_from_wal / _rebuild_child_keys
        / _fold_once / extraction sites). Per-call `exclude` filtering
        happens in route(), against this base. The returned list is SHARED
        (same object until the epoch moves) — callers must treat it as
        read-only, which every current consumer does."""
        epoch = getattr(self, "_cuda_gqa_epoch", 0)
        cached = getattr(self, "_route_cand_base_cache", None)
        cache_epoch = getattr(self, "_route_cand_base_epoch", None)
        if cached is not None and cache_epoch == epoch:
            return cached
        base = [i for i, g in enumerate(self.grafts)
                if not g.get("retired")
                and g.get("kind", "turn") != "recall"]
        self._route_cand_base_cache = base
        self._route_cand_base_epoch = epoch
        return base

    def _native_to_idx_map(self):
        """Epoch-cached full `graft-index -> native_node_id` map plus the
        set of graft indices ineligible for native routing (P1,
        docs/GRM_MLA_CUDA_ROUTE_PLAN.md). Rebuilt only when `_cuda_gqa_epoch`
        (real on this dialect since P1 — see _bump_cuda_gqa_epoch) has moved
        since the cache was built; a fresh mutation-epoch always triggers a
        rebuild, so this is fail-closed the same way GQA's bank cache is
        (over-invalidation is free; under-invalidation is the bug class both
        caches exist to prevent). Nodes with `child_cents` and no native
        multi-key support are marked ineligible on purpose: they must keep
        falling through to the Python route exactly as before (the per-call
        eligibility check in `_native_route_order` used to also police this
        by walking every candidate's raw dict fields; it still does, now
        against this cached map instead of re-deriving from scratch)."""
        store = getattr(self, "native_store", None)
        epoch = getattr(self, "_cuda_gqa_epoch", 0)
        cached = getattr(self, "_native_to_idx_cache", None)
        cache_epoch = getattr(self, "_native_to_idx_cache_epoch", None)
        if cached is not None and cache_epoch == epoch:
            return cached
        multi_ok = bool(store is not None and getattr(
            store, "supports_multi_route_keys", False))
        idx_to_native = {}
        ineligible = set()
        for i, g in enumerate(self.grafts):
            if g.get("child_cents") and not multi_ok:
                ineligible.add(i)
                continue
            node_id = g.get("native_node_id")
            if node_id is None:
                ineligible.add(i)
                continue
            idx_to_native[i] = int(node_id)
        result = (idx_to_native, ineligible)
        self._native_to_idx_cache = result
        self._native_to_idx_cache_epoch = epoch
        return result

    # ---------------------------------------------------- CUDA MLA route
    # P2, docs/GRM_MLA_CUDA_ROUTE_PLAN.md: device-resident centroid arena
    # mirroring GQAArenaCache's opt-in CUDA bridge (_cuda_route_bank_inputs
    # / _ensure_cuda_route_bank / _cuda_route_order below the GQA subclass)
    # but for the MLA dialect's single-centroid-per-node route key. Lives
    # on the base class since MLA IS ArenaCache (GQAArenaCache overrides
    # its own copies of every one of these methods with the GQA-shaped
    # equivalents, so there is no name collision -- Python MRO picks the
    # subclass's version there, this version here).
    def _cuda_route_enabled(self):
        return os.environ.get("GRM_MLA_CUDA_ROUTE", "").lower() in (
            "1", "true", "yes", "on")

    def _cuda_route_bank_signature(self):
        """Cheap O(N) walk building the dense eligible bank: same
        eligibility law as `_route_cand_base` (not retired, not
        kind="recall") intersected with native-routability (`cent` is a
        flat 1-D vector of uniform dim, `native_node_id` present, no
        `child_cents` -- multi-row entries are OUT OF SCOPE for the CUDA
        path per the plan, they fall through to the CPU path exactly as
        `_native_to_idx_map`'s `ineligible` set already does for the
        native ctypes route). Returns (node_ids, signature, rows) or None
        if no eligible bank exists (mirrors GQA's
        `_cuda_route_bank_signature` contract exactly)."""
        rows = []
        node_ids = []
        sig_rows = []
        dim = None
        for g in self.grafts:
            if g.get("retired") or g.get("kind", "turn") == "recall":
                continue
            if g.get("child_cents"):
                return None
            node_id = g.get("native_node_id")
            if node_id is None:
                return None
            if "cent" not in g:
                return None
            cent = g.get("cent")
            key = np.asarray(cent, dtype=np.float32).reshape(-1)
            if dim is None:
                dim = key.shape[0]
            elif key.shape[0] != dim:
                return None
            rows.append(key)
            node_ids.append(int(node_id))
            sig_rows.append((
                int(node_id), int(key.shape[0]), key.dtype.str, id(cent)))
        if not rows:
            return None
        node_ids_np = np.asarray(node_ids, dtype=np.uint64)
        return node_ids_np, tuple(sig_rows), rows

    def _cuda_route_bank_inputs(self):
        """Full bank inputs (route_bank, node_ids, signature), epoch-gated
        exactly like GQA's `_cuda_route_bank_inputs`: the epoch
        (`_cuda_gqa_epoch`, real on both dialects since P1) is the sole
        hot-path staleness gate; the signature walk above only re-runs
        when the epoch has moved, and a content-identical signature after
        an epoch bump (e.g. a metadata-only mutation) reuses the stacked
        bank without re-stacking."""
        epoch = getattr(self, "_cuda_gqa_epoch", 0)
        cached = getattr(self, "_cuda_mla_bank_cache", None)
        cache_epoch = getattr(self, "_cuda_mla_cache_epoch", None)
        if cached is not None and cache_epoch == epoch:
            return cached
        sig = self._cuda_route_bank_signature()
        if sig is None:
            self._cuda_mla_bank_cache = None
            self._cuda_mla_native_to_idx_cache = None
            return None
        node_ids_np, signature, rows = sig
        if cached is not None and cached[2] == signature:
            self._cuda_mla_bank_cache = cached
            self._cuda_mla_cache_epoch = epoch
            return cached
        route_bank = np.ascontiguousarray(np.stack(rows), dtype=np.float32)
        bank_inputs = (route_bank, node_ids_np, signature)
        self._cuda_mla_bank_cache = bank_inputs
        self._cuda_mla_cache_epoch = epoch
        # Invalidate the reverse (native_id -> graft_idx) map derived from
        # this bank -- _cuda_route_native_to_idx() below rebuilds it lazily
        # off the SAME bank_inputs object identity, so a genuine content
        # change (this branch) must drop it; the signature-unchanged reuse
        # branch just above returns before reaching here, so it keeps its
        # existing reverse-map cache untouched.
        self._cuda_mla_native_to_idx_cache = None
        return bank_inputs

    def _cuda_route_native_to_idx(self, bank_inputs):
        """Epoch-cached (reverse map, graft-idx set) pair for the CUDA MLA
        bank's own row set, built ONCE per bank attach rather than
        rebuilt from `cand`/`exclude` on every route call:
          - reverse map: native_node_id -> graft_idx (for the O(k) result
            remap after a route call)
          - graft-idx set: which graft indices the bank actually covers
            (for O(len(exclude)) membership tests instead of an
            O(bank_size) or O(len(cand)) walk)

        This is the fix for the residual P1b named at close ("O(cand)
        subset pass... absorbed into P2's design"): the naive absorption
        still rebuilt an O(len(cand)) dict every call (a 1M-node full
        walk, ~460ms/call measured, against a CUDA kernel that itself
        runs in under 3ms). Caching both here, keyed off `bank_inputs`
        identity (itself epoch-gated by `_cuda_route_bank_inputs`), makes
        the hot path O(len(exclude)) -- typically the live/mounted node
        count, not the eligible-base size.

        Cache-MISS cost (once per bank attach, not per route call): O(N)
        -- re-walks `self.grafts` with the SAME eligibility predicate
        `_cuda_route_bank_signature` used to build the bank, in the same
        ascending order, so row index i of the bank corresponds to the
        i-th eligible graft. This walk only runs when the epoch moves
        (bank rebuild), never on the steady-state hot path."""
        cached = getattr(self, "_cuda_mla_native_to_idx_cache", None)
        if cached is not None and cached[0] is bank_inputs:
            return cached[1], cached[2]
        node_ids_np = bank_inputs[1]
        row_to_graft_idx = []
        for i, g in enumerate(self.grafts):
            if g.get("retired") or g.get("kind", "turn") == "recall":
                continue
            if g.get("child_cents"):
                continue
            if g.get("native_node_id") is None or "cent" not in g:
                continue
            row_to_graft_idx.append(i)
        n = min(len(row_to_graft_idx), node_ids_np.shape[0])
        native_to_idx = {
            int(node_ids_np[row]): row_to_graft_idx[row] for row in range(n)
        }
        graft_idx_set = set(row_to_graft_idx[:n])
        self._cuda_mla_native_to_idx_cache = (
            bank_inputs, native_to_idx, graft_idx_set)
        return native_to_idx, graft_idx_set

    def _ensure_cuda_route_bank(self, store, bank_inputs):
        if getattr(self, "_cuda_mla_route_unavailable", False):
            return False
        if not hasattr(store, "configure_cuda_mla_route_bank"):
            return False
        if bank_inputs is None:
            return False
        route_bank, node_ids, signature = bank_inputs
        # Identity fast path before the tuple `==`: `signature` is a tuple
        # of one (node_id, dim, dtype, id(cent)) entry PER ELIGIBLE GRAFT
        # (`_cuda_route_bank_signature`), so at 1M nodes a value-equality
        # comparison walks 1M elements even when both sides are the exact
        # same cached tuple object (measured ~2.3ms/call — CPython's tuple
        # `==` does not skip element-wise comparison on `a is b`, only
        # `PyObject_RichCompare`'s outer object-identity shortcut applies,
        # which the `==` operator does take for singletons but not
        # arbitrary same-identity containers). `bank_inputs` is only ever
        # a fresh object on a genuine content change
        # (`_cuda_route_bank_inputs`), so comparing `signature is
        # getattr(store, "_cuda_mla_bank_signature", None)` first turns
        # the common (nothing changed) case into an O(1) pointer compare;
        # the O(N) `==` only runs on an actual signature-object swap.
        stored_signature = getattr(store, "_cuda_mla_bank_signature", None)
        if (getattr(store, "_cuda_mla_bank", None) is not None
                and (stored_signature is signature
                     or stored_signature == signature)):
            return True
        try:
            store.configure_cuda_mla_route_bank(route_bank, node_ids)
            store._cuda_mla_bank_signature = signature
        except Exception:
            self._cuda_mla_route_unavailable = True
            return False
        return True

    def _cuda_route_order(self, pkey, cand, limit, exclude=()):
        """CUDA MLA route order (P2). Absorbs the O(cand) residual named
        at P1b close: attach-time defines the dense eligible bank (every
        row the epoch-cached `_cuda_route_bank_signature` walk accepts);
        per-call work is bounded by `len(exclude)` (typically small --
        the live/mounted node set -- not `len(cand)`, which is close to
        the full eligible base N). Concretely: request
        `topk = want + |excludes that are actually in the bank|` from the
        CUDA arena, then drop excluded ids while mapping the O(k)
        response back through the epoch-cached reverse map -- no O(N) or
        O(len(cand)) host pass on the route path, matching the plan's
        contract exactly.

        First cut of this method rebuilt a fresh `{native_id: idx}` dict
        from `cand` on every call (an O(len(cand)) walk -- 1M nodes,
        ~460ms/call measured, entirely Python dict-building overhead
        against a CUDA kernel that itself runs in under 3ms). Fixed by
        caching the reverse map per bank attach (`_cuda_route_native_to_idx`)
        and keying the per-call cost on `exclude` instead of `cand`."""
        if limit is None:
            return None
        store = getattr(self, "native_store", None)
        if store is None or not hasattr(store, "route_mla_cuda"):
            return None
        if not self._cuda_route_enabled():
            return None
        bank_inputs = self._cuda_route_bank_inputs()
        if bank_inputs is None:
            return None
        if not self._ensure_cuda_route_bank(store, bank_inputs):
            return None
        bank_size = int(bank_inputs[1].shape[0])
        # `cand` must equal (bank-eligible rows) minus `exclude`. Rather
        # than proving that by walking `cand` (the expensive direction —
        # O(N)), check the cheap arithmetic invariant instead: `cand`'s
        # eligibility predicate (_route_cand_base: not retired, not
        # kind="recall") is a SUPERSET of the bank's predicate (adds:
        # native_node_id present, cent present as a uniform flat vector,
        # no child_cents). If the bank covers the FULL base
        # (bank_size == len(_route_cand_base())), the two predicates agree
        # on this graft set and `len(cand) == bank_size - |exclude|`
        # follows arithmetically from route()'s own construction
        # (`cand = [i for i in cand_base if i not in exclude]`). Any
        # mismatch (a CUDA-only exclusion actually removed something, or
        # the caller's `cand` was built some other way) fails this check
        # and falls through to the CPU path -- fail-closed, not silently
        # wrong.
        cand_base = self._route_cand_base()
        if bank_size != len(cand_base):
            return None
        native_to_idx, bank_idx_set = self._cuda_route_native_to_idx(bank_inputs)
        excludes_in_bank = 0
        exclude_set = None
        if exclude:
            exclude_set = set(exclude)
            excludes_in_bank = sum(1 for i in exclude_set if i in bank_idx_set)
        if len(cand) != bank_size - excludes_in_bank:
            return None
        want = min(max(0, int(limit)), len(cand))
        if want <= 0:
            return []
        topk = min(16, bank_size, want + excludes_in_bank)
        if topk < want:
            return None
        try:
            routed_native = store.route_mla_cuda(
                np.asarray(pkey, dtype=np.float32), topk=topk)
        except Exception:
            return None
        routed = []
        for node_id in routed_native:
            idx = native_to_idx.get(int(node_id))
            if idx is None:
                continue
            if exclude_set is not None and idx in exclude_set:
                continue
            routed.append(idx)
            if len(routed) >= want:
                break
        if len(routed) < want:
            return None
        self.last_route_backend = "cuda"
        return routed

    def _native_route_order(self, pkey, qrare, cand, limit=None, exclude=()):
        store = getattr(self, "native_store", None)
        if store is None or not hasattr(store, "route"):
            return None
        if (type(self)._key_score is not ArenaCache._key_score
                or type(self)._normalize_scores is not ArenaCache._normalize_scores):
            return None
        if not cand:
            return []
        if not qrare and limit is not None:
            # CUDA MLA route (P2): only for the limited-window path and
            # only when there is no lexical channel to honor (the dense
            # CUDA bank carries no lexical bonus, same restriction GQA's
            # bridge applies at the identical call-site shape). `exclude`
            # is threaded through so the per-call cost stays bounded by
            # len(exclude) rather than len(cand) -- see _cuda_route_order.
            cuda_order = self._cuda_route_order(pkey, cand, limit, exclude)
            if cuda_order is not None:
                return cuda_order
        idx_to_native, ineligible = self._native_to_idx_map()
        # empty ineligible set (the common case) skips the O(len(cand))
        # membership pass entirely — any() over an empty set's genexpr is
        # False for every element, so the guard is byte-identical.
        if ineligible and any(i in ineligible for i in cand):
            return None
        # Subset the epoch-cached map by `cand` (O(len(cand)) dict lookups)
        # instead of rebuilding a fresh {native_node_id: idx} dict from raw
        # graft attribute access every call (the P0-receipted cost:
        # native_to_idx build was ~O(N) per call). The cache itself
        # (idx_to_native) is only rebuilt when the mutation epoch moves.
        native_to_idx = {}
        for i in cand:
            node_id = idx_to_native.get(i)
            if node_id is None:
                return None
            native_to_idx[node_id] = i
        if len(native_to_idx) != len(cand):
            return None

        n_total = len(self.grafts)
        if limit is None:
            # Full-rank repository ordering: byte-identical to the
            # pre-P1 contract. topk = full N; completeness law unchanged.
            topk = n_total
        else:
            want = min(max(0, int(limit)), len(cand))
            if want <= 0:
                return []
            # `store.route` ranks over the ENTIRE native store, not just
            # `cand` — nodes outside `cand` (excluded by the caller, or
            # retired/recall-kind, which native has no concept of) can
            # still occupy native's top ranks and would silently displace
            # eligible ids if topk were just `want`. Slack = however many
            # native-known ids exist outside `cand` (upper bound: native
            # store size never exceeds len(self.grafts) -- verified
            # invariant, GraftRepository._native_sync_node only ever grows
            # _native_node_ids lazily up to len(arena.grafts)). Request
            # enough that even if ALL of that slack out-ranks every
            # eligible id, `want` eligible survivors are still inside the
            # requested window.
            excluded = max(0, n_total - len(cand))
            topk = min(n_total, want + excluded)

        try:
            routed_native = store.route(
                np.asarray(pkey, dtype=np.float32).reshape(-1).tolist(),
                sorted(qrare), topk=topk)
        except Exception:
            return None

        routed = []
        for node_id in routed_native:
            idx = native_to_idx.get(int(node_id))
            if idx is not None:
                routed.append(idx)
                if limit is not None and len(routed) >= min(
                        max(0, int(limit)), len(cand)):
                    break

        if limit is None:
            # REPLACEMENT completeness law, full-rank path: unchanged from
            # pre-P1 — native must account for every eligible candidate.
            if len(routed) != len(cand):
                return None
            self.last_route_backend = "native"
            return routed

        want = min(max(0, int(limit)), len(cand))
        # REPLACEMENT completeness law, limited path: native must return
        # exactly min(topk, n_eligible) ids that map through native_to_idx.
        # We can't observe n_eligible directly (it's native-internal state
        # — could be less than topk if non-finite scores were dropped, M6
        # law), so the fail-closed check is: either we filled the window
        # we asked for (`want` mapped ids found), or native handed back
        # fewer RAW ids than `topk` (meaning it ran out of eligible nodes
        # store-wide, not that our slack guess was wrong) and every one of
        # those raw ids mapped cleanly. Any other shortfall (native filled
        # its full topk quota but we still didn't reach `want` mapped ids)
        # means the slack guess under-covered — same distrust as today,
        # Python fallback, bounded cost.
        if len(routed) >= want:
            self.last_route_backend = "native"
            return routed[:want]
        if len(routed_native) < topk:
            # Native legitimately exhausted its eligible pool (NaN drops or
            # a smaller store than n_total) before filling topk. Every
            # returned id must still map, or something else is wrong.
            if len(routed) != len(routed_native):
                return None
            self.last_route_backend = "native"
            return routed
        return None

    # ------------------------------------------------------------ librarian
    # Mounted DIALOGUE turns pull generation into conversation mode — the
    # model acknowledges the request ("I'll create an archive note...")
    # instead of executing it (E4-C round 1: 0/6, both digests fact-free
    # while routing worked). The primed prefix forces content mode.
    # First-gen folds (turn sources). RELATIONAL SENTENCES required: a
    # bare-bullet "- Priya Raghunathan" keeps the token but loses the
    # relation ("backend hire"), and probes traverse relations (E2:
    # narrative 7/7 vs list 1/3; both era-mode misses traced to bare
    # bullets). Primers force content mode past the acknowledgment trap
    # AND start mid-sentence so the continuation is prose, not a list.
    DIGEST_PROMPTS = (
        "User: For the archive, restate every fact from the conversation "
        "above as a complete sentence that says what each name, code, "
        "number, and time refers to.\n"
        "Assistant: For the archive: the",
        "User: Write a brief archive note covering everything above in "
        "complete sentences, preserving every name, code, number, and "
        "time verbatim and stating what each one refers to.\n"
        "Assistant: ARCHIVE NOTE — The conversation established that the",
        "User: List every fact from the conversation above: every name, "
        "code, number, and time, and what each one refers to.\n"
        "Assistant: The facts to archive are:",
    )
    # Depth>=1 folds (digests/eras as sources) MUST produce SENTENCES: an
    # era built list-style strips relations (E2: lists retrieve 1/3 vs
    # narrative 7/7 — measured again at era depth: "4. Conference room"
    # bled the demo's room into the offsite answer). Chronicle prompts
    # force prose; the list-form QC below rejects relapses.
    ERA_PROMPTS = (
        "User: Rewrite the archive notes above as one brief chronicle in "
        "complete sentences, keeping every name, code, number, and time "
        "verbatim and stating what each one refers to.\n"
        "Assistant: CHRONICLE — In this period,",
        "User: Combine the notes above into flowing sentences that state "
        "each fact together with what it means, quoting every name, code, "
        "number, and time exactly.\nAssistant: Combined record: during "
        "these conversations,",
        "User: List every fact from the notes above: every name, code, "
        "number, and time, and what each one refers to.\n"
        "Assistant: The facts to archive are:",
    )
    TEXT_SCAFFOLD_CONSOLIDATION = False
    TEXT_SCAFFOLD_MAX_CHARS = 6000
    CONSOLIDATE_NGEN = 120
    ALLOW_HIGH_COVERAGE_LIST_DIGESTS = False
    ENABLE_ERA_FOLDING = True
    EXTRACTIVE_ERA_CONSOLIDATION = False
    EXTRACTIVE_ERA_MAX_CHARS = 9000

    def _source_scaffold(self, source_texts):
        remaining = int(self.TEXT_SCAFFOLD_MAX_CHARS)
        parts = []
        for j, text in enumerate(source_texts, 1):
            if remaining <= 0:
                break
            clean = str(text).strip()
            clean = re.sub(r"(?m)^(?:User|Assistant):\s*", "", clean)
            clean = re.sub(r"\s+", " ", clean).strip()
            if len(clean) > remaining:
                clean = clean[:remaining].rstrip()
            parts.append(f"[source {j}]\n{clean}")
            remaining -= len(clean)
        return "\n\n".join(parts)

    def _consolidation_prompts(self, deep, source_texts):
        prompts = self.ERA_PROMPTS if deep else self.DIGEST_PROMPTS
        if not self.TEXT_SCAFFOLD_CONSOLIDATION:
            return prompts
        source_block = self._source_scaffold(source_texts)
        if not source_block:
            return prompts
        out = []
        for prompt in prompts:
            head, tail = prompt.rsplit("\nAssistant:", 1)
            out.append(f"{head}\n\nSource excerpts: use every source below; "
                       f"write at least one complete archive sentence for "
                       f"each source, and do not copy role labels.\n"
                       f"{source_block}\n\n"
                       f"Assistant:{tail}")
        return tuple(out)

    def _extractive_era_text(self, source_texts):
        """Build an index-era from child digest text without model synthesis.

        Era nodes are expanded to children before reading, so their text is a
        routing/index surface. For dialects where digest-of-digest generation is
        too memory-heavy, preserve the child digest facts verbatim and harvest
        this index text under the serving model.
        """
        remaining = int(self.EXTRACTIVE_ERA_MAX_CHARS)
        parts = []
        for j, text in enumerate(source_texts):
            if remaining <= 0:
                break
            label = chr(ord("A") + (j % 26))
            clean = str(text).strip()
            clean = re.sub(r"(?i)\b(?:ARCHIVE NOTE|ERA INDEX)\.\s*", "", clean)
            clean = re.sub(r"\s+", " ", clean).strip()
            if len(clean) > remaining:
                clean = clean[:remaining].rstrip()
            parts.append(f"[digest {label}] {clean}")
            remaining -= len(clean)
        return " ".join(parts)

    @staticmethod
    def _rare_tokens(text):
        """Code/number-shaped tokens — the verbatim payload a digest must
        preserve. Mechanically checkable: the librarian holds the sources."""
        out = set()
        # ',' inside the token class keeps "7,400" whole — fragmenting it
        # made a CORRECT answer fail grounding (descent diag, 2026-06-10)
        for w in re.findall(r"[A-Za-z0-9][\w:.,\-]*", text):
            w = w.rstrip(".,:;")
            if any(ch.isdigit() for ch in w) or (w.isupper() and len(w) >= 3):
                out.add(w.lower())
        return out

    @staticmethod
    def _digest_qc(text, source_texts=None, min_keep=0.5, forbid_lists=False):
        """Reject degenerate digests (E2: comma-list repetition loops lose
        relations — retrieve 1/3 vs narrative 7/7) and CONTENT-FREE digests
        (E4-C: instruction acknowledgments pass fluency checks). Content
        rule: keep >= min_keep of the sources' code/number-shaped tokens.
        forbid_lists (depth>=1 folds): bullet/numbered enumerations strip
        the relations probes traverse — require prose."""
        toks = text.split()
        if len(toks) < 6:
            return False
        if forbid_lists:
            items = len(re.findall(r"(?:^|\n)\s*(?:\d+\.|[-*•])\s", text))
            items += max(0, len(re.findall(r"\d+\.\s+[A-Z]", text)) - 1)
            if items >= 3:
                return False
        if len(set(toks)) / len(toks) < 0.45:
            return False
        seen = {}
        for j in range(len(toks) - 5):
            k = tuple(toks[j:j + 6])
            seen[k] = seen.get(k, 0) + 1
            if seen[k] >= 3:
                return False
        if source_texts is not None:
            need = set()
            for s in source_texts:
                need |= ArenaCache._rare_tokens(s)
            if need:
                have = ArenaCache._rare_tokens(text)
                if len(need & have) / len(need) < min_keep:
                    return False
        return True

    # consolidation fidelity bar: a fold whose best candidate covers fewer
    # than this fraction of the sources' FACTS is ABORTED — a lossy digest
    # is worse than no digest (the unfolded turns are clean readers AND
    # clean topical routers; a drifted digest poisons both — measured
    # 2026-06-11: a {5-8} digest dropped $7,400 + Lake Arrowhead, then
    # folded into an era, and the facts existed in no node's text or
    # centroid -> unroutable, unrecoverable).
    MIN_FOLD_KEEP = 0.70
    _FACT_STOP = {"user", "assistant", "noted", "heads", "the", "i", "archive",
                  "note", "chronicle", "logged", "okay", "for", "in", "this",
                  "project", "small", "update", "still", "true", "that"}

    @classmethod
    def _fact_set(cls, texts):
        """The verbatim payload a fold MUST keep: IDENTIFIERS (digit/ALLCAPS
        — 7,400, BX-44, NIGHTJAR) plus MULTI-WORD named entities (>=2
        consecutive capitalized words — Lake Arrowhead, Priya Raghunathan).
        Single incidental caps (Thursday, Tuesday) are NOT facts — counting
        them made chatty turns look fact-dense and over-aborted folds
        (2026-06-11: 28/34 turns exempted, compression dead)."""
        out = set()
        for t in texts:
            out |= cls._rare_tokens(t)
            for m in re.finditer(r"(?:[A-Z][\w\-]+\s+){1,}[A-Z][\w\-]+", t):
                for w in m.group(0).split():
                    out.add(w.lower().strip(".,;:"))
        return out - cls._FACT_STOP

    @classmethod
    def _coverage(cls, text, need):
        if not need:
            return 1.0
        have = cls._rare_tokens(text) | cls._caps_tokens(text, False)
        return len(need & have) / len(need)

    def _deposit_consolidation(self, idxs, text, prefix="ARCHIVE NOTE."):
        note = f"{prefix} {text}\n"
        didx = self.deposit(note)
        self.grafts[didx]["kind"] = "digest"
        self.grafts[didx]["sources"] = list(idxs)
        # descent keys flatten through generations: an era node (digest of
        # digests) stays addressable per LEAF topic, and inherits the
        # sources' lexical keys so identifier queries still find it
        child = []
        rare = set(self.grafts[didx].get("rare", set()))
        for i in idxs:
            g = self.grafts[i]
            child.append(g["cent"])
            child.extend(g.get("child_cents", ()))
            if "rare" not in g:
                g["rare"] = self._rare_tokens(g["text"])
            rare |= g["rare"]
        self.grafts[didx]["child_cents"] = child
        self.grafts[didx]["rare"] = rare | self._rare_tokens(note)
        for i in idxs:
            self.grafts[i]["retired"] = True
        # kind/child_cents/retired all changed after deposit()'s own bump —
        # each is read by the CUDA route bank eligibility/signature walk.
        self._bump_cuda_gqa_epoch()
        return didx, text

    def consolidate(self, idxs, ngen=None):
        """Phase-2 seat compression: mount the given grafts, generate ONE
        QC'd digest (E2-validated verbatim-preservation prompt), deposit it
        standalone (clean key + payload), RETIRE the sources from routing.
        The digest node carries its children's centroids for hierarchical
        descent. Returns (digest_idx, digest_text)."""
        if ngen is None:
            ngen = int(self.CONSOLIDATE_NGEN)
        # standalone generation: arm device mounts directly, no live_shift
        for L in self.m.layers:
            L.self_attn.live_shift = None
        self._clear_transients()
        srcs = [self.grafts[i]["text"] for i in idxs]
        deep = any(self.grafts[i].get("kind", "turn") != "turn" for i in idxs)
        need = self._fact_set(srcs)
        self.last_consolidation_attempts = []
        self.last_consolidation_result = {"accepted": False,
                                          "best_cov": -1.0,
                                          "need_count": len(need),
                                          "hit_count": 0}
        if deep and self.EXTRACTIVE_ERA_CONSOLIDATION:
            text = self._extractive_era_text(srcs)
            cov = self._coverage(text, need)
            # Extractive eras are index/routing nodes whose children are
            # expanded before reading. Repetition/list QC is for generated
            # reader digests; applying it here rejects high-coverage indexes
            # over naturally repetitive child digest structure.
            qc = len(text.split()) >= 6
            self.last_consolidation_attempts.append({
                "prompt_index": -1,
                "qc": bool(qc),
                "relaxed_list": True,
                "coverage": float(cov),
                "need_count": len(need),
                "hit_count": int(round(cov * len(need))),
                "text": text[:1200],
            })
            self.last_consolidation_result.update({
                "best_cov": float(cov),
                "hit_count": int(round(cov * len(need))),
            })
            if not qc or cov < self.MIN_FOLD_KEEP:
                return None, None
            self.last_consolidation_result["accepted"] = True
            return self._deposit_consolidation(idxs, text,
                                               prefix="ERA INDEX.")

        self._ensure_h(idxs)        # sources may be paged out (cold storage)
        for li, layer in enumerate(self.m.layers):
            att = layer.self_attn
            hs = [self.grafts[i]["h"][li] for i in idxs]
            blk = {key: (hs[0][key] if len(hs) == 1
                         else tc.cat([h[key] for h in hs], dim=dim))
                   for key, dim in self.PAYLOAD}
            self._set_inject(att, blk)
        prompts = self._consolidation_prompts(deep, srcs)
        try:
            text, best, best_cov = None, None, -1.0
            for prompt_idx, prompt in enumerate(prompts):
                ids = out = lg = caches = None
                primer = prompt.rsplit("Assistant:", 1)[1]
                try:
                    ids = self.encode(prompt)
                    with tc.no_grad():
                        lg, caches = self.m(np.array([ids], dtype=np.int64),
                                            last_token_only=True)
                    pos = len(ids)
                    out = [int(lg.numpy()[0, -1].argmax())]
                    for _ in range(ngen - 1):
                        with tc.no_grad():
                            lg, caches = self.m(
                                np.array([[out[-1]]], dtype=np.int64),
                                kv_caches=caches, position_offset=pos,
                                last_token_only=True)
                        pos += 1
                        out.append(int(lg.numpy()[0, -1].argmax()))
                    t = (primer + " " + self.decode(out)).strip()
                    for stop in ("\nUser:", "User:"):
                        if stop in t:
                            t = t.split(stop)[0]
                    t = t.strip()
                    cov = self._coverage(t, need)
                    qc = self._digest_qc(t, None, forbid_lists=True)
                    relaxed_list = False
                    if (not qc and self.ALLOW_HIGH_COVERAGE_LIST_DIGESTS
                            and cov >= self.MIN_FOLD_KEEP):
                        relaxed_list = self._digest_qc(
                            t, None, forbid_lists=False)
                        qc = relaxed_list
                    self.last_consolidation_attempts.append({
                        "prompt_index": prompt_idx,
                        "qc": bool(qc),
                        "relaxed_list": bool(relaxed_list),
                        "coverage": float(cov),
                        "need_count": len(need),
                        "hit_count": int(round(cov * len(need))),
                        "text": t[:1200],
                    })
                    if not qc:
                        continue
                    if cov > best_cov:
                        best, best_cov = t, cov
                        self.last_consolidation_result.update({
                            "best_cov": float(best_cov),
                            "hit_count": int(round(best_cov * len(need))),
                        })
                    if cov >= self.MIN_FOLD_KEEP:
                        text = t
                        break
                finally:
                    del ids, out, lg, caches
                    self._clear_transients()
            if text is None:
                text = best
        finally:
            kv_graft.clear_injection(self.m)
            self._clear_transients()
        # FIDELITY GATE: abort the fold if no candidate kept the facts —
        # the caller keeps the sources unfolded (recall > compression).
        # This applies to ERA folds too — tested and REFUTED 2026-06-11:
        # exempting eras ("index nodes, never read; lexical keys are
        # inherited") dropped the 42-turn gates 8/8 -> 5/8. Folding RETIRES
        # the children's individual routing surfaces, and era expansion is
        # budget-bound — fit() truncated the 300-token child set and the
        # one fact-bearing digest was the one dropped. An era over
        # fact-dense digests can make its own subtree unreachable; the
        # coverage bar keeps such digests directly routable instead.
        if text is None or best_cov < self.MIN_FOLD_KEEP:
            return None, None
        self.last_consolidation_result["accepted"] = True
        return self._deposit_consolidation(idxs, text)

    # ------------------------------------------------------------ cache ops
    def _ensure_h(self, idxs):
        """Re-load freed/retired nodes from cold storage before mounting
        (descent re-mounts children whose VRAM was reclaimed; the pager
        frees least-recently-mounted nodes). Touches the LRU clock."""
        self.mount_clock = getattr(self, "mount_clock", 0) + 1
        for i in idxs:
            g = self.grafts[i]
            g["last_used"] = self.mount_clock
            if g.get("h") is None:
                if self.node_loader is None:
                    raise RuntimeError(f"graft {i} has no tensors and no "
                                       f"node_loader is set")
                g["h"] = self.node_loader(i)
                self.page_ins = getattr(self, "page_ins", 0) + 1

    def _graft_block(self, picks, li):
        """Arena-slice tensors for layer li: the positional key component
        re-RoPEs at the mount's arena seats n_sink..n_sink+n (MLA: only the
        32-d k_pe; GQA: the full key). Grafts are device-resident tc
        tensors — no host->device upload per swap."""
        hs = [self.grafts[i]["h"][li] for i in picks]
        blk = {key: (hs[0][key] if len(hs) == 1
                     else tc.cat([h[key] for h in hs], dim=dim))
               for key, dim in self.PAYLOAD}
        n = blk[self.PAYLOAD[0][0]].shape[self.PAYLOAD[0][1]]
        blk = self._rope_block_at(blk, self.n_sink, inverse=False)
        return blk, n

    def _native_mount_ids(self, picks):
        ids = []
        for i in picks:
            node_id = self.grafts[i].get("native_node_id")
            if node_id is None:
                raise RuntimeError(f"graft {i} has no native_node_id")
            ids.append(int(node_id))
        return ids

    def _commit_native_mount(self, picks, mount_tokens):
        store = getattr(self, "native_store", None)
        if store is None or not hasattr(store, "commit_mount"):
            return
        store.commit_mount(self._native_mount_ids(picks), int(mount_tokens))

    def _splice_cache_tensor(self, tensor, insert, dim, head_tokens,
                             tail_start):
        if hasattr(tc, "splice_rows"):
            with tc.no_grad():
                return tc.splice_rows(tensor, insert, dim, head_tokens,
                                      tail_start)
        parts = [tensor.slice(dim, 0, head_tokens), insert]
        S = tensor.shape[dim]
        if S > tail_start:
            parts.append(tensor.slice(dim, tail_start, S - tail_start))
        return tc.cat(parts, dim=dim)

    def _evict_cache_tensor(self, tensor, dim, head_tokens, drop_tokens):
        if hasattr(tc, "evict_rows"):
            with tc.no_grad():
                try:
                    return tc.evict_rows(tensor, dim, head_tokens, drop_tokens)
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"evict_rows failed shape={tensor.shape} dim={dim} "
                        f"head={head_tokens} drop={drop_tokens}") from exc
        S = tensor.shape[dim]
        return tc.cat([tensor.slice(dim, 0, head_tokens),
                       tensor.slice(dim, head_tokens + drop_tokens,
                                    S - head_tokens - drop_tokens)], dim=dim)

    def _pair_payload_tuple(self, raw_tensor, rope_tensor, spec):
        (ri, _, _), (pi, _, _) = spec
        out = [None] * len(self.PAYLOAD)
        out[ri] = raw_tensor
        out[pi] = rope_tensor
        return tuple(out)

    def _graft_pair_blocks(self, picks, spec):
        (ri, rkey, rdim), (pi, pkey, pdim) = spec
        raw_blocks, rope_blocks = [], []
        n_new = None
        for li in range(len(self.caches)):
            hs = [self.grafts[i]["h"][li] for i in picks]
            raw = (hs[0][rkey] if len(hs) == 1
                   else tc.cat([h[rkey] for h in hs], dim=rdim))
            rope = (hs[0][pkey] if len(hs) == 1
                    else tc.cat([h[pkey] for h in hs], dim=pdim))
            raw_n = raw.shape[rdim]
            rope_n = rope.shape[pdim]
            if raw_n != rope_n:
                raise RuntimeError(
                    f"graft payload token mismatch: {rkey}={raw_n} "
                    f"{pkey}={rope_n}")
            if n_new is None:
                n_new = raw_n
            elif raw_n != n_new:
                raise RuntimeError("graft layer token count mismatch")
            raw_blocks.append(raw)
            rope_blocks.append(rope)
        return raw_blocks, rope_blocks, int(n_new or 0)

    def _swap_cache_payloads(self, picks, head):
        if not picks:
            return None
        if hasattr(tc, "arena_row_pair_transaction"):
            tx = self._arena_cache_transaction(picks)
            if tx is not None:
                return tx
        if not hasattr(tc, "swap_row_pairs_with_rope"):
            return None
        spec = self._paired_export_spec()
        if spec is None:
            return None
        (ri, _, rdim), (pi, _, pdim) = spec
        raw_blocks, rope_blocks, n_new = self._graft_pair_blocks(picks, spec)
        raw_caches = [cache[ri] for cache in self.caches]
        rope_caches = [cache[pi] for cache in self.caches]
        with tc.no_grad():
            raw_out, rope_out = tc.swap_row_pairs_with_rope(
                raw_caches, rope_caches, raw_blocks, rope_blocks,
                self.m.rope_cos, self.m.rope_sin, rdim, pdim,
                self.n_sink, head, self.n_sink, self.ROPE_PAIR_SWAP)
        return ([self._pair_payload_tuple(raw_out[i], rope_out[i], spec)
                 for i in range(len(raw_out))], n_new)

    def _evict_cache_payloads(self, head, drop_tokens):
        if (head == self.n_sink and drop_tokens == self.cur_mount_n
                and hasattr(tc, "arena_row_pair_transaction")):
            tx = self._arena_cache_transaction([])
            if tx is not None:
                return tx[0]
        if not hasattr(tc, "evict_row_pairs"):
            return None
        spec = self._paired_export_spec()
        if spec is None:
            return None
        (ri, _, rdim), (pi, _, pdim) = spec
        raw_caches = [cache[ri] for cache in self.caches]
        rope_caches = [cache[pi] for cache in self.caches]
        with tc.no_grad():
            raw_out, rope_out = tc.evict_row_pairs(
                raw_caches, rope_caches, rdim, pdim, head, drop_tokens)
        return [self._pair_payload_tuple(raw_out[i], rope_out[i], spec)
                for i in range(len(raw_out))]

    def _arena_cache_transaction(self, picks):
        if not hasattr(tc, "arena_row_pair_transaction"):
            return None
        spec = self._paired_export_spec()
        if spec is None:
            return None
        (ri, _, rdim), (pi, _, pdim) = spec
        raw_caches = [cache[ri] for cache in self.caches]
        rope_caches = [cache[pi] for cache in self.caches]
        raw_blocks, rope_blocks = [], []
        if picks:
            raw_blocks, rope_blocks, _ = self._graft_pair_blocks(picks, spec)
        with tc.no_grad():
            raw_out, rope_out, n_new = tc.arena_row_pair_transaction(
                raw_caches, rope_caches, raw_blocks, rope_blocks,
                self.m.rope_cos, self.m.rope_sin, rdim, pdim, self.n_sink,
                self.cur_mount_n, self.width, self.ROPE_PAIR_SWAP)
        return ([self._pair_payload_tuple(raw_out[i], rope_out[i], spec)
                 for i in range(len(raw_out))], int(n_new))

    def swap(self, picks):
        """Replace the arena occupants. Pure cache surgery — live untouched."""
        if picks == self.cur_mounts or self.caches is None:
            self.cur_mounts = picks
            return
        self._ensure_h(picks)
        n_new = 0
        new_caches = []
        head = self.n_sink + self.cur_mount_n
        paired = (self._swap_cache_payloads(picks, head) if picks
                  else (self._evict_cache_payloads(
                      self.n_sink, self.cur_mount_n), 0))
        if paired is not None and paired[0] is not None:
            new_caches, n_new = paired
        else:
            for li, cache in enumerate(self.caches):
                blk = None
                if picks:
                    blk, n_new = self._graft_block(picks, li)
                new = []
                for ei, (key, dim) in enumerate(self.PAYLOAD):
                    t = cache[ei]
                    if blk is not None:
                        new.append(self._splice_cache_tensor(
                            t, blk[key], dim, self.n_sink, head))
                    else:
                        new.append(self._evict_cache_tensor(
                            t, dim, self.n_sink, self.cur_mount_n))
                new_caches.append(tuple(new))
        self.caches = new_caches
        self._clear_transients()
        self.cur_mounts = picks
        self.cur_mount_n = n_new
        if n_new > self.width:
            raise ValueError(f"mounts ({n_new}) exceed arena width ({self.width})")
        self._commit_native_mount(picks, n_new)

    def evict(self):
        """Drop live segments beyond the recency window from the cache."""
        if len(self.live_segs) <= self.live_turns or self.caches is None:
            return 0
        if self.live_turns <= 0:
            drop = self.live_segs
            self.live_segs = []
        else:
            drop = self.live_segs[:-self.live_turns]
            self.live_segs = self.live_segs[-self.live_turns:]
        drop_n = sum(n for _, n in drop)
        head = self.n_sink + self.cur_mount_n
        out = []
        for cache in self.caches:
            new = []
            for ei, (key, dim) in enumerate(self.PAYLOAD):
                t = cache[ei]
                new.append(self._evict_cache_tensor(t, dim, head, drop_n))
            out.append(tuple(new))
        self.caches = out
        self._clear_transients()
        return drop_n

    # ------------------------------------------------------------ forward
    def _forward(self, ids, last_only=True):
        with tc.no_grad():     # inference-only; also unlocks fused-norm paths
            lg, self.caches = self.m(np.array([ids], dtype=np.int64),
                                     kv_caches=self.caches,
                                     position_offset=self.pos,
                                     last_token_only=last_only)
        self.pos += len(ids)
        return lg.numpy()[0, -1].astype(np.float32)

    def feed(self, turn_text, deposit=True):
        """Push an already-complete turn through the live cache (no
        generation, no routing — existing mounts stay seated)."""
        if self.ephemeral:
            # no persistent live cache to push through — a fed turn IS its
            # deposit (recency-as-mount picks it up on the next step)
            return self.deposit(turn_text) if deposit else None
        for L in self.m.layers:
            L.self_attn.live_shift = self.live_shift
        ids = self.encode(turn_text)
        if self.caches is None:    # bootstrap: sink enters via injection
            self._set_injection_host(self.sink_h)
        self._forward(ids)
        kv_graft.clear_injection(self.m)
        gidx = None
        if deposit:
            gidx = (self.deposit_from_cache(turn_text, len(ids))
                    if self.cache_deposits else self.deposit(turn_text))
        self.live_segs.append((gidx, len(ids)))
        self.evict()

    # confabulation / hedge detection between trips: an answer asserting
    # code/number-shaped tokens absent from every mounted source (and the
    # question) is ungrounded; a content-free hedge is a miss. v1 is blind
    # to name-only confabulations (no digit/uppercase signal) — recorded.
    HEDGES = ("don't know", "do not know", "not sure", "no information",
              "doesn't mention", "does not mention", "cannot", "can't find",
              "have access")
    # dialogue scaffolding and meta-commentary words: present in every
    # mounted "User:/Assistant:" turn, so they must never count as content
    # overlap — "Okay, the user is asking about X" grounded via "user"
    # (GQA trips gate, measured)
    SCAFFOLD = {"user", "assistant", "okay", "asking", "about", "question"}

    @staticmethod
    def _caps_tokens(text, skip_sentence_initial=True):
        """Proper-noun-ish tokens: capitalized words, optionally excluding
        sentence starters (for answers; sources keep everything)."""
        out = set()
        for sent in re.split(r"[.!?\n]+", text):
            ws = sent.split()
            for j, w in enumerate(ws):
                if skip_sentence_initial and j == 0:
                    continue
                if re.match(r"^[A-Z][\w\-]+$", w.rstrip(".,;:")):
                    out.add(w.lower().rstrip(".,;:"))
        return out

    def _grounded(self, ans, mount_idxs, question):
        a = ans.lower()
        if any(h in a for h in self.HEDGES):
            return False
        # identifier-aware: if the question names codes and NO mounted
        # source contains any of them, the right document is not mounted —
        # whatever the answer says, it cannot be about the asked entity
        # ("right family, wrong sibling" is grounded-but-wrong otherwise)
        qrare = self._rare_tokens(question)
        if qrare:
            mounted = set()
            for i in mount_idxs:
                mounted |= self._rare_tokens(self.grafts[i]["text"])
            if not (qrare & mounted):
                return False
        content = ((self._rare_tokens(ans) | self._caps_tokens(ans))
                   - self._rare_tokens(question)
                   - self._caps_tokens(question, skip_sentence_initial=False))
        have = set()
        words = set()
        for i in mount_idxs:
            t = self.grafts[i]["text"]
            have |= self._rare_tokens(t) | self._caps_tokens(t, False)
            words |= {w.lower().rstrip(".,:;") for w in t.split()}
        if not content:
            # no identifier-shaped tokens — fall back to substantive words:
            # a correct prose answer ("The header parser is crashing.") has
            # its payload words IN the mounted sources; a deflection ("same
            # place as last time") and an echo of an unmounted turn do not
            qw = {w.lower().rstrip(".,:;?") for w in question.split()}
            subst = ({w.lower().rstrip(".,:;") for w in ans.split()
                      if len(w.rstrip(".,:;")) >= 4} - qw - self.SCAFFOLD)
            return bool(subst) and bool(subst & words)
        return content <= have

    def _native_source_closure_indices(self, picks, max_depth=1,
                                       include_roots=False):
        store = getattr(self, "native_store", None)
        if store is None or not hasattr(store, "source_closure"):
            return None
        native_to_idx = {}
        for idx, g in enumerate(self.grafts):
            node_id = g.get("native_node_id")
            if node_id is not None:
                native_to_idx[int(node_id)] = int(idx)
        native_ids = []
        for idx in picks:
            node_id = self.grafts[int(idx)].get("native_node_id")
            if node_id is None:
                return None
            native_ids.append(int(node_id))
        try:
            native_out = store.source_closure(
                native_ids, max_depth=int(max_depth),
                include_roots=bool(include_roots))
        except Exception:
            return None
        out, seen = [], set()
        for node_id in native_out:
            idx = native_to_idx.get(int(node_id))
            if idx is None:
                return None
            if idx not in seen:
                out.append(idx)
                seen.add(idx)
        return out

    def _descent_source_children(self, idx, qrare=None):
        qrare = set(qrare or ())
        srcs = list(self.grafts[int(idx)].get("sources") or [])
        if not qrare:
            native = self._native_source_closure_indices(
                [idx], max_depth=1, include_roots=False)
            if native is not None and (native or not srcs):
                return native
        if qrare:
            hit = []
            for src in srcs:
                g = self.grafts[src]
                if "rare" not in g:
                    g["rare"] = self._rare_tokens(g["text"])
                if qrare & g["rare"]:
                    hit.append(src)
            return hit or srcs
        return srcs

    def _descent_expand(self, picks, kinds, qrare=None):
        kinds = set(kinds or ())
        out, seen = [], set()
        for idx in picks:
            idx = int(idx)
            if self.grafts[idx].get("kind") in kinds:
                children = self._descent_source_children(idx, qrare=qrare)
                if children:
                    for child in children:
                        child = int(child)
                        if child not in seen:
                            out.append(child)
                            seen.add(child)
                    continue
            if idx not in seen:
                out.append(idx)
                seen.add(idx)
        return out

    def step(self, user_text, ngen=48, deposit=True,
             stops=None, max_trips=0):
        """One conversation turn through the arena. max_trips > 0 enables
        SHUTTLING: if the answer fails the grounding check, restore the
        pre-attempt cache (snapshot = the old tensor list + position —
        cache tensors are immutable), swap in the NEXT ranking slice, and
        retry. Failed attempts never enter the live cache. Returns
        (answer, info)."""
        if stops is None:
            stops = self.stop_sequences
        for L in self.m.layers:
            L.self_attn.live_shift = self.live_shift
        rec = []
        if self.ephemeral:
            # clear the boat: fresh cache every turn, recency as mounts
            self.caches, self.pos, self.live_segs = None, 0, []
            self.cur_mounts, self.cur_mount_n = [], 0
            turns = [i for i, g in enumerate(self.grafts)
                     if not g.get("retired")
                     and g.get("kind", "turn") in ("turn", "recall")]
            rec = turns[-self.recency_mounts:] if self.recency_mounts else []
        # exclude turns already present (live window / recency mounts)
        live_idx = {g for g, _ in self.live_segs if g is not None} | set(rec)
        route_limit = max(1, (int(max_trips) + 1) * int(self.topk))
        ranking = self.route(user_text, exclude=live_idx, limit=route_limit)
        snap = (self.caches, self.pos, list(self.live_segs),
                self.cur_mounts, self.cur_mount_n, len(self.grafts))
        # PRECISE-MOUNT policy (corpus-100 lesson): an identifier query is a
        # point lookup. With the right doc at rank 1 but its near-identical
        # SIBLINGS at ranks 2-3, co-mounting collapses reads (end recall
        # 4/20 despite 18/20 routing — the model answers with a sibling's
        # value). When rank-1 covers ALL the probe's identifier tokens,
        # trip 0 mounts it ALONE; wider slices become later trips.
        attempts = []                # (picks, clean_room)
        qrare = self._rare_tokens(user_text)
        precise = None
        if ranking and qrare:
            g0 = self.grafts[ranking[0]]
            if "rare" not in g0:
                g0["rare"] = self._rare_tokens(g0["text"])
            if qrare <= g0["rare"]:
                precise = [ranking[0]]
                attempts.append((precise, False))
        for t in range(max_trips + 1):
            sl = sorted(ranking[t * self.topk:(t + 1) * self.topk])
            if sl and (sl, False) not in attempts:
                attempts.append((sl, False))
        # DESCENT (measured law, 2026-06-10): era texts are INDEX nodes,
        # never readers — list-form eras strip relations and prose-form
        # eras invent them, and a model reading a corrupt era faithfully
        # reproduces the corruption ("the backend hire was Project
        # NIGHTJAR", grounded). So eras expand to their children at the
        # PRIMARY attempt; digests (E4-C-grade readers) expand only on a
        # descent retry. Children are identifier-filtered when the probe
        # names codes, and every mount set is BUDGET-FITTED to the arena
        # width (an unbounded descent over-filled the arena and collided
        # live positions with mount seats — descent diag). Cold-storage
        # children reload via node_loader.
        def fit(picks):
            # truncation is EXPANSION-ORDERED, deliberately. Score-ordered
            # truncation was tried and REFUTED (2026-06-11, 6/8): max-over-
            # child-cents inflates digest scores over verbatim turns, so
            # "relevance" order kept prose digests and dropped the raw fact
            # turns inside budget-bound expansions. A workable version
            # needs a leaf bias — board item, not a one-liner.
            rec_budget = 0 if qrare else sum(self.grafts[i]["ntok"]
                                             for i in rec)
            budget = self.width - rec_budget
            out, used = [], 0
            for i in picks:
                n = self.grafts[i]["ntok"]
                if used + n <= budget:
                    out.append(i)
                    used += n
            return sorted(out)

        # budget: max_trips+1 attempts total. Ladder: primary (eras
        # pre-expanded) -> descent (digests expanded too) -> clean room on
        # the deepest mount set. Identifier queries keep precise-first.
        if max_trips >= 1 and attempts:
            head = precise or attempts[0][0]
            primary = fit(self._descent_expand(head, ("era",), qrare=qrare))
            deep = fit(self._descent_expand(
                primary, ("era", "digest"), qrare=qrare))
            ladder = [(primary, False)]
            if deep != primary:
                ladder.append((deep, False))
            ladder.append((deep, True))
            if not precise:
                ladder += [(fit(self._descent_expand(
                    a[0], ("era",), qrare=qrare)), False)
                           for a in attempts[1:]]
            attempts = ladder[:max_trips + 1]
        elif attempts:
            attempts = [(fit(self._descent_expand(
                attempts[0][0], ("era",), qrare=qrare)), False)]
        else:
            attempts = [([], False)]
        best = None
        for trip, (picks, clean) in enumerate(attempts):
            if not picks:
                break
            if trip:        # roll back the failed attempt entirely
                (self.caches, self.pos, self.live_segs, self.cur_mounts,
                 self.cur_mount_n) = (snap[0], snap[1], list(snap[2]),
                                      snap[3], snap[4])
                del self.grafts[snap[5]:]
                self._bump_cuda_gqa_epoch()
            if clean:
                # fresh mini-cache: _attempt's bootstrap path rebuilds
                # [sink | mounts | question] via injection — no surgery on
                # a zero-length live tail (engine slice/cat edge case)
                self.caches, self.pos, self.live_segs = None, 0, []
                self.cur_mounts, self.cur_mount_n = [], 0
            # recency joins topical/anaphora attempts only. Identifier
            # lookups are point reads even when rank-1 is a folded parent:
            # previous turns are echo sources that can swamp the mounted fact.
            use_rec = rec and not qrare and not clean and picks != precise
            mset = sorted(set(rec) | set(picks)) if use_rec else sorted(set(picks))
            txt, info = self._attempt(user_text, mset, ngen, deposit, stops)
            info["trip"] = trip
            if clean:
                info["clean_room"] = True
            if self._grounded(txt, mset, user_text):
                return txt, info
            if best is None:
                best = (txt, info, (self.caches, self.pos, list(self.live_segs),
                                    self.cur_mounts, self.cur_mount_n,
                                    list(self.grafts)))
        # nothing grounded — keep the FIRST attempt's answer and state
        if best is None:
            txt, info = self._attempt(user_text, [], ngen, deposit, stops)
            info["trip"] = 0
            info["no_mount_fit"] = True
            return txt, info
        txt, info, st = best
        (self.caches, self.pos, self.live_segs,
         self.cur_mounts, self.cur_mount_n) = st[0], st[1], st[2], st[3], st[4]
        self.grafts[:] = st[5]
        self._bump_cuda_gqa_epoch()
        return txt, info

    def _attempt(self, user_text, picks, ngen, deposit, stops):
        self._ensure_h(picks)
        if self.caches is None:
            # bootstrap: sink (+ first mounts) enter via the injection path
            mounts = [{"h": self.sink_h}] + [self.grafts[i] for i in picks]
            inj = []
            # sink is host numpy; deposited grafts are device tensors
            _np = lambda t: t if isinstance(t, np.ndarray) else t.numpy()
            for li in range(len(self.m.layers)):
                inj.append({key: np.concatenate([_np(g["h"][li][key])
                                                 for g in mounts], axis=dim)
                            for key, dim in self.PAYLOAD})
            self._set_injection_host(inj)
            self.cur_mounts = picks
            self.cur_mount_n = sum(self.grafts[i]["ntok"] for i in picks)
            self._commit_native_mount(picks, self.cur_mount_n)
        else:
            self.swap(picks)
        prompt_ids = self.encode(self._format_step_prompt(user_text))
        seg_start_ntok = len(prompt_ids)
        row = self._forward(prompt_ids)
        kv_graft.clear_injection(self.m)     # bootstrap injection fired once
        out = [int(row.argmax())]
        cached_out = 0
        stopped = False
        for _ in range(ngen - 1):
            # EARLY STOP: break at the first stop sequence so post-answer
            # tokens never enter the cache. Qwen3 leaks reasoning text
            # after its answer; cached leak in the live window became a
            # style attractor that flipped LATER probes into meta-answers
            # (GQA trips gate: 6/6 with trips=0, 0/6 with trips=2 — the
            # extra junk from retried attempts cascaded). MiniCPM3 simply
            # never emitted post-answer junk, which is why decoding the
            # full ngen was harmless on MLA.
            if any(s in self.decode(out) for s in stops):
                stopped = True
                break
            row = self._forward([out[-1]])
            cached_out += 1
            out.append(int(row.argmax()))
        if not stopped and not any(s in self.decode(out) for s in stops):
            # The last predicted token is not in the KV cache until it is fed
            # once. Commit it so live/deposit segment lengths match reality.
            self._forward([out[-1]])
            cached_out += 1
        # the answer tokens are in the cache; record the live segment
        txt = self.decode(out)
        for stop in stops:
            if stop in txt:
                txt = txt.split(stop)[0]
        txt = txt.strip()
        turn_text = self._format_step_turn(user_text, txt)
        seg_cache_ntok = seg_start_ntok + cached_out
        gidx = None
        if deposit:
            # cache-deposit span covers prompt + ALL generated tokens (incl.
            # any post-stop tail) — the cache is the source of truth
            gidx = (self.deposit_from_cache(turn_text, seg_cache_ntok)
                    if self.cache_deposits else self.deposit(turn_text))
            if picks:
                # retrieval hygiene: a turn that adds NO identifier tokens
                # beyond its mounts and question is DERIVATIVE — keep it
                # for recency/anaphora, exclude from routing and folding
                # (deposited Q&A turns are style attractors and fold into
                # answer-mixing digests — measured twice)
                new_rare = (self._rare_tokens(turn_text)
                            - self._rare_tokens(user_text))
                for i in picks:
                    g = self.grafts[i]
                    if "rare" not in g:
                        g["rare"] = self._rare_tokens(g["text"])
                    new_rare -= g["rare"]
                if not new_rare:
                    self.grafts[gidx]["kind"] = "recall"
                    # kind="recall" flips CUDA route-bank eligibility
                    # (excluded from the signature walk) — bump again, the
                    # deposit above already bumped for the append itself.
                    self._bump_cuda_gqa_epoch()
        self.live_segs.append((gidx, seg_cache_ntok))
        evicted = self.evict()
        S = self._cache_len()
        return txt, {"mounts": [i + 1 for i in picks], "resident": S,
                     "evicted": evicted, "live_tokens": sum(n for _, n in self.live_segs)}


class GQAArenaCache(ArenaCache):
    """The GQA dialect (Qwen3-family). Forks from MLA, each one measured:

      - payload = per-layer pre-RoPE (k, v) FULL tensors, both seq dim=2;
        mount surgery re-RoPEs the whole key (MLA re-RoPEs only the 32-d
        shared k_pe — the latent is position-free)
      - router = layer-0 |q.k| in the per-head qk-normed space (E1 router
        law forks by model: MiniCPM3 has NO qk-norm -> outlier keys make
        key-space scores probe-independent -> latent routing; Qwen3's
        per-head norm makes layer-0 keys a normalized routing space).
        Keys are unit-normalized per head-vector so scores stay in O(1)
        cosine range and the lexical channel keeps its +1 dominance
        calibration. route_layer must be 0 — part of the dialect.
      - persistence: nodes carry (k, v) as (L, H, S, D) fp16; routing keys
        are variable-length per node, so the index stores one array per
        node instead of a stacked matrix.
    """
    PAYLOAD = (("k", 2), ("v", 2))
    ROPE_KEYS = ("k",)

    def __init__(self, model, *a, storage_bits=None, **kw):
        cfg = model.config
        # k + v vals/token/layer for node VRAM accounting (Qwen3-4B: 2048)
        self.VALS_PER_TOK_LAYER = cfg.num_kv_heads * cfg.head_dim * 2
        # arena surgery slices (k, v) tuples — the INT8-quantized cache
        # form (k_u8, k_scale, v_u8, v_scale) is not surgeable
        for L in model.layers:
            L.self_attn.quant_kv_cache = False
        # P3 format-1 storage quantization (opt-in, default OFF — unchanged
        # fp16 node payloads unless explicitly requested). Constructor kwarg
        # takes priority; else env var GRM_GRAFT_STORAGE_BITS (same
        # convention as GRM_GQA_CUDA_ROUTE below). Validated against
        # core.graft_quant.SUPPORTED_BITS's packable depths (16 means "off").
        if storage_bits is None:
            env_bits = os.environ.get("GRM_GRAFT_STORAGE_BITS", "").strip()
            storage_bits = int(env_bits) if env_bits else None
        if storage_bits is not None:
            storage_bits = int(storage_bits)
            if storage_bits not in SUPPORTED_BITS or storage_bits == 16:
                raise ValueError(
                    f"GQAArenaCache: storage_bits={storage_bits!r} invalid — "
                    f"must be one of {[b for b in SUPPORTED_BITS if b != 16]} "
                    "(or None/unset/16 for the default fp16 node payload)"
                )
        self.storage_bits = storage_bits
        super().__init__(model, *a, **kw)

    # --------------------------------------------------- dialect overrides
    def _harvest(self, ids, layer_filter=None, max_layers=None):
        # full-depth forward (no early-exit on the GQA path); layer_filter
        # still bounds what is STORED
        return kv_graft.harvest_kv(self.m, ids, layer_filter=layer_filter)

    def _probe_key(self, text):
        qc = kv_graft.capture_queries(self.m, self.encode(text),
                                      layer_filter={self.route_layer})
        return qc[self.route_layer][0].astype(np.float32)

    def _node_key(self, text, h_host=None):
        if h_host is None:
            h_host = self._harvest(self.encode(text),
                                   layer_filter={self.route_layer})
        return h_host[self.route_layer]["k"][0].astype(np.float32)

    def _key_score(self, pkey, nkey):
        # E1 protocol EXACTLY: mean over q heads of max over (probe pos,
        # node pos) of |q.k|/sqrt(Dh), RAW vectors. Unit-normalizing q and
        # k first was tested and REFUTED (2026-06-11, unified gate 2/6):
        # norm information is load-bearing — max-over-pairs keys on
        # high-salience tokens, and under cosine every pair weighs the
        # same, so rankings collapsed probe-independent (the MiniCPM3
        # key-space failure signature, reproduced on the model whose
        # qk-norm was supposed to prevent it).
        H, _, Dh = pkey.shape
        kk = np.repeat(nkey, H // nkey.shape[0], axis=0)
        sc = np.einsum("hqd,hkd->hqk", pkey, kk) / np.sqrt(Dh)
        return float(np.abs(sc).max(axis=(1, 2)).mean())

    def _normalize_scores(self, base):
        # raw |q.k| has no calibrated scale; rescale per route so the best
        # centroid-channel score sits at 1.0 — a monotone transform
        # (ranking-preserving) that restores the lexical channel's +1
        # dominance calibration
        if not base:
            return base
        mx = max(abs(v) for v in base.values()) + 1e-8
        return {i: v / mx for i, v in base.items()}

    def _cuda_route_enabled(self):
        return os.environ.get("GRM_GQA_CUDA_ROUTE", "").lower() in (
            "1", "true", "yes", "on")

    def _bump_cuda_gqa_epoch(self):
        """Mutation-epoch bump. Called at every GQAArenaCache-side site that
        can change what `_cuda_route_bank_signature` would produce (graft
        add/retire/replace, route-key/kind/child_cents changes, truncation or
        restore of self.grafts). O(1) at mutation time.

        This is a FAST-PATH HINT layered over the cheap per-node signature
        walk in `_cuda_route_bank_signature`, which remains the correctness
        authority on every route call (it re-derives from `self.grafts`
        fresh, so it is safe even against mutations this epoch cannot see —
        e.g. a caller poking graft dict fields directly from outside this
        class, such as GraftRepository's lifecycle/WAL/migrate paths). The
        epoch only lets a route call skip the walk itself when nothing this
        class controls has changed; the walk is ~1% of the old per-call cost
        (P0 receipt), so re-running it on an epoch mismatch is cheap
        insurance, not a regression.
        """
        self._cuda_gqa_epoch = getattr(self, "_cuda_gqa_epoch", 0) + 1

    def _cuda_route_bank_signature(self):
        """Cheap O(N) walk: eligibility + signature only, NO stacking. This
        is the part of the old `_cuda_route_bank_inputs` that was NOT the
        P0-receipted defect (~1% of the reused-call cost) — it stays on
        every call as the fail-closed correctness check. Returns
        (node_ids, signature, rows) where `rows` are the raw per-node key
        arrays (not yet stacked) so a cache miss can stack them without a
        second walk."""
        rows = []
        node_ids = []
        sig_rows = []
        shape = None
        for g in self.grafts:
            if g.get("retired") or g.get("kind", "turn") == "recall":
                continue
            node_id = g.get("native_node_id")
            if node_id is None or g.get("child_cents"):
                return None
            if "cent" not in g:
                return None
            cent = g.get("cent")
            key = np.asarray(cent, dtype=np.float32)
            if key.ndim != 3:
                return None
            if shape is None:
                shape = key.shape
            elif key.shape != shape:
                return None
            rows.append(key)
            node_ids.append(int(node_id))
            sig_rows.append((
                int(node_id),
                tuple(int(dim) for dim in key.shape),
                key.dtype.str,
                id(cent),
            ))
        if not rows:
            return None
        node_ids_np = np.asarray(node_ids, dtype=np.uint64)
        return node_ids_np, tuple(sig_rows), rows

    @staticmethod
    def _cuda_gqa_bridge_paranoid():
        # Debug/test-only defense in depth (W2, GRM_CUDA_BRIDGE_OVERHEAD_PLAN
        # P2): forces the O(N) signature walk to run on every call even when
        # the epoch says nothing changed, and asserts the two decisions
        # agree. Off in production (adds back the ~1% walk cost this phase
        # made conditional). Read fresh every call, not cached, so tests can
        # monkeypatch/setenv per-test without reload games.
        return os.environ.get("GRM_GQA_BRIDGE_PARANOID", "").lower() in (
            "1", "true", "yes", "on")

    def _cuda_route_bank_inputs(self):
        """Full bank inputs (route_bank, node_ids, signature).

        Hot path (W1 complete: every graft_repository.py mutation site that
        can change what the signature walk would produce now bumps
        `_cuda_gqa_epoch` — see _bump_cuda_gqa_epoch docstring and the
        GRM_CUDA_BRIDGE_OVERHEAD_LEDGER 2026-07-08 01:30 entry for the full
        site list). The epoch is now the SOLE staleness gate on the common
        path: if `_cuda_gqa_epoch` has not moved since the cached bank was
        built, return the cache with NO signature walk at all (O(1), no
        O(N) component). Only a changed epoch triggers the walk (which
        stays exactly as before: eligibility + signature, stacking only on
        an actual content-signature miss — the P0-receipted defect this
        whole work order exists to close).

        GRM_GQA_BRIDGE_PARANOID=1 disables the fast path: the walk always
        runs, and its answer is asserted to agree with what the epoch-only
        decision would have been — a self-check for this instrumentation
        battery, not a production code path.
        """
        epoch = getattr(self, "_cuda_gqa_epoch", 0)
        cached = getattr(self, "_cuda_gqa_bank_cache", None)
        cache_epoch = getattr(self, "_cuda_gqa_cache_epoch", None)
        paranoid = self._cuda_gqa_bridge_paranoid()
        epoch_says_fresh = cached is not None and cache_epoch == epoch

        if epoch_says_fresh and not paranoid:
            return cached

        sig = self._cuda_route_bank_signature()
        if sig is None:
            if paranoid and epoch_says_fresh:
                assert cached is None, (
                    "GRM_GQA_BRIDGE_PARANOID: epoch says the cached bank is "
                    "fresh but the signature walk finds no eligible bank — "
                    "epoch under-invalidation")
            self._cuda_gqa_bank_cache = None
            return None
        node_ids_np, signature, rows = sig

        if paranoid and epoch_says_fresh:
            assert cached[2] == signature, (
                "GRM_GQA_BRIDGE_PARANOID: epoch says the cached bank is "
                "fresh but the signature walk disagrees — epoch under-"
                "invalidation (a mutation changed the signature without "
                "bumping _cuda_gqa_epoch)")

        if cached is not None and cached[2] == signature:
            # Signature-identical even though the epoch moved (e.g. a
            # metadata-only mutation bumped epoch but didn't touch anything
            # the bank cares about) — over-invalidation is fine, just skip
            # the re-stack and refresh the cache_epoch so the NEXT call can
            # take the fast path again.
            self._cuda_gqa_bank_cache = cached
            self._cuda_gqa_cache_epoch = epoch
            return cached

        route_bank = np.ascontiguousarray(np.stack(rows), dtype=np.float32)
        bank_inputs = (route_bank, node_ids_np, signature)
        self._cuda_gqa_bank_cache = bank_inputs
        self._cuda_gqa_cache_epoch = epoch
        return bank_inputs

    def _ensure_cuda_route_bank(self, store, bank_inputs):
        if getattr(self, "_cuda_gqa_route_unavailable", False):
            return False
        if not hasattr(store, "configure_cuda_gqa_route_bank"):
            return False
        if bank_inputs is None:
            return False
        route_bank, node_ids, signature = bank_inputs
        if (getattr(store, "_cuda_gqa_bank", None) is not None
                and getattr(store, "_cuda_gqa_bank_signature", None) == signature):
            return True
        try:
            store.configure_cuda_gqa_route_bank(route_bank, node_ids)
            store._cuda_gqa_bank_signature = signature
        except Exception:
            self._cuda_gqa_route_unavailable = True
            return False
        return True

    def _cuda_route_order(self, pkey, cand, limit):
        if limit is None:
            return None
        store = getattr(self, "native_store", None)
        if store is None or not hasattr(store, "route_gqa_cuda"):
            return None
        if not self._cuda_route_enabled():
            return None
        bank_inputs = self._cuda_route_bank_inputs()
        if bank_inputs is None:
            return None
        if not self._ensure_cuda_route_bank(store, bank_inputs):
            return None
        native_to_idx = {
            int(self.grafts[i]["native_node_id"]): int(i)
            for i in cand
            if self.grafts[i].get("native_node_id") is not None
        }
        if len(native_to_idx) != len(cand):
            return None
        bank_size = int(bank_inputs[1].shape[0])
        want = min(max(0, int(limit)), len(cand))
        if want <= 0:
            return []
        excluded = max(0, bank_size - len(cand))
        topk = min(16, bank_size, want + excluded)
        if topk < want:
            return None
        try:
            routed_native = store.route_gqa_cuda(
                np.asarray(pkey, dtype=np.float32), topk=topk)
        except Exception:
            return None
        routed = []
        for node_id in routed_native:
            idx = native_to_idx.get(int(node_id))
            if idx is not None:
                routed.append(idx)
            if len(routed) >= want:
                break
        if len(routed) < want:
            return None
        self.last_route_backend = "cuda"
        return routed

    def _native_route_order(self, pkey, qrare, cand, limit=None, exclude=()):
        # `exclude` accepted for base-class call-site compatibility
        # (route() passes it positionally-by-keyword to every dialect's
        # _native_route_order since the P2 MLA CUDA route fix); GQA's own
        # _cuda_route_order below does not yet consume it -- out of scope
        # for this work order, GQA's bridge was not re-profiled at 1M-node
        # scale here.
        store = getattr(self, "native_store", None)
        if store is None or not hasattr(store, "route_gqa"):
            return None
        if not cand:
            return []
        if not qrare:
            cuda_order = self._cuda_route_order(pkey, cand, limit)
            if cuda_order is not None:
                return cuda_order
        native_to_idx = {}
        for i in cand:
            node_id = self.grafts[i].get("native_node_id")
            if node_id is None:
                return None
            native_to_idx[int(node_id)] = i
        try:
            routed_native = store.route_gqa(
                np.asarray(pkey, dtype=np.float32), sorted(qrare),
                topk=len(self.grafts))
        except Exception:
            return None
        routed = []
        for node_id in routed_native:
            idx = native_to_idx.get(int(node_id))
            if idx is not None:
                routed.append(idx)
        if len(routed) != len(cand):
            return None
        self.last_route_backend = "native"
        if limit is not None:
            return routed[:max(0, int(limit))]
        return routed

    def _rope_block_at(self, blk, pos0, inverse=False):
        blk["k"] = self._rope_tensor(blk["k"], pos0, inverse)
        return blk

    def _cache_key_of(self, seg):
        return None     # GQA contextualized keys unimplemented: standalone

    def _set_inject(self, att, blk):
        att.inject_kv = (blk["k"], blk["v"], 1.0)
        att.graft_seats = int(blk["k"].shape[2])

    def _set_injection_host(self, inj):
        kv_graft.set_injection(self.m, inj)

    # ------------------------------------------------- persistence format
    def pack_node(self, h):
        """Node payload for disk (GraftRepository._atomic_savez_compressed
        writes this dict straight to nodes/NNNN.npz). Default: plain fp16
        {"k","v"} stacked (L,H,S,D) — UNCHANGED from before P3.

        P3 format-1 hook (opt-in via self.storage_bits, see __init__): when
        set, returns the PACKED payload instead (core.graft_quant
        .pack_kv_arrays — same group-32 symmetric math as format 2,
        explicit format_version/storage_bits fields, fail-closed on
        unpack). Mutation/WAL semantics untouched — the WAL never carries
        K/V bytes regardless of which payload shape this returns (P0: WAL
        NODE_UPSERT records only a has_payload flag)."""
        _np = lambda t: (t if isinstance(t, np.ndarray)
                         else t.float().numpy()).astype(np.float16)
        k = np.stack([_np(d["k"])[0] for d in h])
        v = np.stack([_np(d["v"])[0] for d in h])
        if self.storage_bits is None:
            return {"k": k, "v": v}
        return pack_kv_arrays({"k": k, "v": v}, self.storage_bits)

    def unpack_node(self, z):
        """Inverse of pack_node. Transparently detects a packed payload
        (format_version field, core.graft_quant.is_packed_payload) and
        dequantizes before device upload; a plain fp16 payload (the
        default, and every node written before P3) is unaffected."""
        dt = BlockTC.COMPUTE_DTYPE
        if is_packed_payload(z):
            arrays = unpack_kv_arrays(z, ["k", "v"])
            k, v = arrays["k"], arrays["v"]
        else:
            k, v = z["k"], z["v"]
        return [{"k": tc.tensor(np.ascontiguousarray(k[li][None])).astype(dt),
                 "v": tc.tensor(np.ascontiguousarray(v[li][None])).astype(dt)}
                for li in range(len(self.m.layers))]

    def pack_index(self):
        return {f"rkey_{i:04d}": g["cent"]
                for i, g in enumerate(self.grafts)}

    def unpack_index(self, z, i):
        return z[f"rkey_{i:04d}"].astype(np.float32)
