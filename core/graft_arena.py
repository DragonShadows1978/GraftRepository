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
import re

import numpy as np

from core.mistral7b_tc import BlockTC, F, tc
from core import kv_graft


class ArenaCache:
    def __init__(self, model, encode, decode, sink_text="<conversation>\n",
                 arena_width=256, route_layer=44, topk=3, live_turns=2,
                 max_live=4096, cache_deposits=True,
                 ephemeral=False, recency_mounts=2):
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
        self.dt = BlockTC.COMPUTE_DTYPE
        # the model auto-extends RoPE only to position_offset+L; arena
        # positions run live_shift further. Extend once, up front.
        model.extend_rope(len(encode(sink_text)) + arena_width + max_live)

        sink_ids = encode(sink_text)
        self.sink_h = kv_graft.harvest_kv_mla(model, sink_ids)
        self.n_sink = len(sink_ids)
        self.live_shift = self.n_sink + arena_width

        self.caches = None              # per layer (c_n, k_pe), built on turn 1
        self.pos = 0                    # live tokens processed (position counter)
        self.cur_mounts = []            # graft idxs currently seated
        self.cur_mount_n = 0            # arena seats currently occupied
        self.live_segs = []             # [(graft_idx or None, ntok), ...]
        self.grafts = []                # {h, cent, ntok, text}

    # ------------------------------------------------------------ repository
    def deposit(self, text):
        """Standalone harvest deposit (document-in-isolation semantics, one
        dedicated forward). Stored DEVICE-resident: mounts never re-upload."""
        ids = self.encode(text)
        h = kv_graft.harvest_kv_mla(self.m, ids)
        dev = [{"c": tc.tensor(np.ascontiguousarray(h[li]["c"])).astype(self.dt),
                "kpe": tc.tensor(np.ascontiguousarray(h[li]["kpe"])).astype(self.dt)}
               for li in range(len(self.m.layers))]
        self.grafts.append({"h": dev,
                            "cent": kv_graft.latent_centroid(h, self.route_layer),
                            "ntok": len(ids), "text": text})
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
        cseg = self.m.rope_cos.slice(0, p0, seg_ntok)
        sneg = self.m.rope_sin.slice(0, p0, seg_ntok) * -1.0
        dev = []
        cent = None
        for li, (c, kpe) in enumerate(self.caches):
            S = c.shape[1]
            cg = c.slice(1, S - seg_ntok, seg_ntok)
            kg = F.apply_rotary(kpe.slice(2, S - seg_ntok, seg_ntok), cseg, sneg)
            dev.append({"c": cg, "kpe": kg})
            if li == self.route_layer and getattr(self, "key_from_cache", False):
                v = cg.numpy()[0].astype(np.float32).mean(0)
                cent = v / (np.linalg.norm(v) + 1e-8)
        if cent is None:
            pl = kv_graft.harvest_kv_mla(self.m, self.encode(text),
                                         layer_filter={self.route_layer},
                                         max_layers=self.route_layer + 1)
            cent = kv_graft.latent_centroid(pl, self.route_layer)
        self.grafts.append({"h": dev, "cent": cent,
                            "ntok": seg_ntok, "text": text})
        return len(self.grafts) - 1

    def route(self, bare_text, exclude):
        if not self.grafts:
            return []
        pl = kv_graft.harvest_kv_mla(self.m, self.encode(bare_text),
                                     layer_filter={self.route_layer},
                                     max_layers=self.route_layer + 1)
        p = pl[self.route_layer]["c"][0].astype(np.float32).mean(0)
        p = p / (np.linalg.norm(p) + 1e-8)

        # Lexical channel: identifier tokens in the probe (codes, numbers,
        # ALL-CAPS) are exact-match keys. Mean centroids CANNOT separate
        # sibling chunks that differ only in a code token (corpus-100:
        # @1 4/20 latent-only — family right, instance random); an exact
        # identifier hit must dominate. Latent cos lives in ~[0.4, 0.9],
        # so +1 per full match wins outright, partial matches rank between.
        qrare = self._rare_tokens(bare_text)

        def score(g):
            # hierarchical descent: a digest node answers for its retired
            # children — score by the best of its own centroid and theirs
            # (a multi-topic digest's own centroid is diluted; the child
            # keys keep it addressable per topic)
            s = float(np.dot(p, g["cent"]))
            for ch in g.get("child_cents", ()):
                s = max(s, float(np.dot(p, ch)))
            if qrare:
                if "rare" not in g:
                    g["rare"] = self._rare_tokens(g["text"])
                s += len(qrare & g["rare"]) / len(qrare)
            return s

        cand = [i for i in range(len(self.grafts))
                if i not in exclude and not self.grafts[i].get("retired")]
        cand.sort(key=lambda i: -score(self.grafts[i]))
        return cand                               # full ranking, best first

    # ------------------------------------------------------------ librarian
    # Mounted DIALOGUE turns pull generation into conversation mode — the
    # model acknowledges the request ("I'll create an archive note...")
    # instead of executing it (E4-C round 1: 0/6, both digests fact-free
    # while routing worked). The primed prefix forces content mode.
    DIGEST_PROMPTS = (
        "User: List every fact from the conversation above: every name, "
        "code, number, and time, and what each one refers to.\n"
        "Assistant: The facts to archive are:",
        "User: Write a brief archive note covering everything above, "
        "preserving every name, code, number, and time verbatim.\n"
        "Assistant: ARCHIVE NOTE —",
        "User: Repeat every name, code, number, and time mentioned above, "
        "each with what it refers to.\nAssistant: 1.",
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

    @staticmethod
    def _rare_tokens(text):
        """Code/number-shaped tokens — the verbatim payload a digest must
        preserve. Mechanically checkable: the librarian holds the sources."""
        out = set()
        for w in re.findall(r"[A-Za-z0-9][\w:.\-]*", text):
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

    def consolidate(self, idxs, ngen=120):
        """Phase-2 seat compression: mount the given grafts, generate ONE
        QC'd digest (E2-validated verbatim-preservation prompt), deposit it
        standalone (clean key + payload), RETIRE the sources from routing.
        The digest node carries its children's centroids for hierarchical
        descent. Returns (digest_idx, digest_text)."""
        # standalone generation: arm device mounts directly, no live_shift
        for L in self.m.layers:
            L.self_attn.live_shift = None
        for li, layer in enumerate(self.m.layers):
            att = layer.self_attn
            hs = [self.grafts[i]["h"][li] for i in idxs]
            c = hs[0]["c"] if len(hs) == 1 else tc.cat([h["c"] for h in hs], dim=1)
            kpe = hs[0]["kpe"] if len(hs) == 1 else tc.cat([h["kpe"] for h in hs], dim=2)
            att.inject_kv = (c, kpe)
            att.graft_seats = int(c.shape[1])
        srcs = [self.grafts[i]["text"] for i in idxs]
        deep = any(self.grafts[i].get("kind", "turn") != "turn" for i in idxs)
        prompts = self.ERA_PROMPTS if deep else self.DIGEST_PROMPTS
        try:
            text, best, best_keep = None, None, -1.0
            for prompt in prompts:
                primer = prompt.rsplit("Assistant:", 1)[1]
                ids = self.encode(prompt)
                with tc.no_grad():
                    lg, caches = self.m(np.array([ids], dtype=np.int64),
                                        last_token_only=True)
                pos = len(ids)
                out = [int(lg.numpy()[0, -1].argmax())]
                for _ in range(ngen - 1):
                    with tc.no_grad():
                        lg, caches = self.m(np.array([[out[-1]]], dtype=np.int64),
                                            kv_caches=caches, position_offset=pos,
                                            last_token_only=True)
                    pos += 1
                    out.append(int(lg.numpy()[0, -1].argmax()))
                t = (primer + " " + self.decode(out)).strip()
                for stop in ("\nUser:", "User:"):
                    if stop in t:
                        t = t.split(stop)[0]
                t = t.strip()
                if self._digest_qc(t, srcs, forbid_lists=deep):
                    text = t
                    break
                need = set()
                for s in srcs:
                    need |= self._rare_tokens(s)
                keep = (len(need & self._rare_tokens(t)) / len(need)) if need else 0.0
                if keep > best_keep:
                    best, best_keep = t, keep
            if text is None:
                text = best   # no attempt passed QC — keep the best keeper
        finally:
            kv_graft.clear_injection(self.m)
        note = f"ARCHIVE NOTE. {text}\n"
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
        return didx, text

    # ------------------------------------------------------------ cache ops
    def _graft_block(self, picks, li):
        """Arena-slice tensors for layer li: latent as-is (position-free),
        k_pe re-RoPE'd at the mount's arena seats n_sink..n_sink+n. Grafts
        are device-resident tc tensors — no host->device upload per swap."""
        hs = [self.grafts[i]["h"][li] for i in picks]
        c = hs[0]["c"] if len(hs) == 1 else tc.cat([h["c"] for h in hs], dim=1)
        kpe = hs[0]["kpe"] if len(hs) == 1 else tc.cat([h["kpe"] for h in hs], dim=2)
        n = c.shape[1]
        kpe = F.apply_rotary(kpe, self.m.rope_cos.slice(0, self.n_sink, n),
                             self.m.rope_sin.slice(0, self.n_sink, n))
        return c, kpe, n

    def swap(self, picks):
        """Replace the arena occupants. Pure cache surgery — live untouched."""
        if picks == self.cur_mounts or self.caches is None:
            self.cur_mounts = picks
            return
        n_new = 0
        new_caches = []
        head = self.n_sink + self.cur_mount_n
        for li, (c, kpe) in enumerate(self.caches):
            S = c.shape[1]
            parts_c = [c.slice(1, 0, self.n_sink)]
            parts_k = [kpe.slice(2, 0, self.n_sink)]
            if picks:
                cg, kg, n_new = self._graft_block(picks, li)
                parts_c.append(cg)
                parts_k.append(kg)
            if S > head:        # zero-length live tail: nothing to carry
                parts_c.append(c.slice(1, head, S - head))
                parts_k.append(kpe.slice(2, head, S - head))
            new_caches.append((tc.cat(parts_c, dim=1) if len(parts_c) > 1 else parts_c[0],
                               tc.cat(parts_k, dim=2) if len(parts_k) > 1 else parts_k[0]))
        self.caches = new_caches
        self.cur_mounts = picks
        self.cur_mount_n = n_new
        if n_new > self.width:
            raise ValueError(f"mounts ({n_new}) exceed arena width ({self.width})")

    def evict(self):
        """Drop live segments beyond the recency window from the cache."""
        if len(self.live_segs) <= self.live_turns or self.caches is None:
            return 0
        drop = self.live_segs[:-self.live_turns]
        self.live_segs = self.live_segs[-self.live_turns:]
        drop_n = sum(n for _, n in drop)
        head = self.n_sink + self.cur_mount_n
        out = []
        for c, kpe in self.caches:
            S = c.shape[1]
            out.append((tc.cat([c.slice(1, 0, head),
                                c.slice(1, head + drop_n, S - head - drop_n)], dim=1),
                        tc.cat([kpe.slice(2, 0, head),
                                kpe.slice(2, head + drop_n, S - head - drop_n)], dim=2)))
        self.caches = out
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
            kv_graft.set_injection_mla(self.m, [self.sink_h[li]
                                                for li in range(len(self.m.layers))])
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
              "doesn't mention", "does not mention", "cannot", "can't find")

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
            subst = {w.lower().rstrip(".,:;") for w in ans.split()
                     if len(w.rstrip(".,:;")) >= 4} - qw
            return bool(subst) and bool(subst & words)
        return content <= have

    def step(self, user_text, ngen=48, deposit=True,
             stops=("\nUser:", "User:", "\n\n"), max_trips=0):
        """One conversation turn through the arena. max_trips > 0 enables
        SHUTTLING: if the answer fails the grounding check, restore the
        pre-attempt cache (snapshot = the old tensor list + position —
        cache tensors are immutable), swap in the NEXT ranking slice, and
        retry. Failed attempts never enter the live cache. Returns
        (answer, info)."""
        for L in self.m.layers:
            L.self_attn.live_shift = self.live_shift
        rec = []
        if self.ephemeral:
            # clear the boat: fresh cache every turn, recency as mounts
            self.caches, self.pos, self.live_segs = None, 0, []
            self.cur_mounts, self.cur_mount_n = [], 0
            turns = [i for i, g in enumerate(self.grafts)
                     if not g.get("retired")
                     and g.get("kind", "turn") == "turn"]
            rec = turns[-self.recency_mounts:] if self.recency_mounts else []
        # exclude turns already present (live window / recency mounts)
        live_idx = {g for g, _ in self.live_segs if g is not None} | set(rec)
        ranking = self.route(user_text, exclude=live_idx)
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
        # budget: max_trips+1 attempts total, with a CLEAN-ROOM retry —
        # repeated grounding failures implicate the live window
        # (corpus-100: the previous same-family Q&A echoes over the
        # mounted doc, and the echo REPEATS across same-window retries).
        # Ladder ORDER follows the information need: for identifier
        # queries (precise mount exists) the point lookup retries in
        # ISOLATION before touching sibling slices — siblings are the
        # known confusion trap (both corpus-100 residual misses were
        # trip-1 sibling slices accepting a grounded-but-wrong sibling
        # value before the clean room could run). For topical queries the
        # clean room stays last.
        if max_trips >= 1 and attempts:
            if precise:
                rest = [a for a in attempts if a[0] != precise]
                attempts = ([(precise, False), (precise, True)] + rest)[:max_trips + 1]
            else:
                attempts = attempts[:max_trips]
                attempts.append((attempts[0][0], True))
        else:
            attempts = attempts[:1]
        best = None
        for trip, (picks, clean) in enumerate(attempts):
            if not picks:
                break
            if trip:        # roll back the failed attempt entirely
                (self.caches, self.pos, self.live_segs, self.cur_mounts,
                 self.cur_mount_n) = (snap[0], snap[1], list(snap[2]),
                                      snap[3], snap[4])
                del self.grafts[snap[5]:]
            if clean:
                # fresh mini-cache: _attempt's bootstrap path rebuilds
                # [sink | mounts | question] via injection — no surgery on
                # a zero-length live tail (engine slice/cat edge case)
                self.caches, self.pos, self.live_segs = None, 0, []
                self.cur_mounts, self.cur_mount_n = [], 0
            mset = sorted(set(rec) | set(picks))
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
        txt, info, st = best
        (self.caches, self.pos, self.live_segs,
         self.cur_mounts, self.cur_mount_n) = st[0], st[1], st[2], st[3], st[4]
        self.grafts[:] = st[5]
        return txt, info

    def _attempt(self, user_text, picks, ngen, deposit, stops):
        if self.caches is None:
            # bootstrap: sink (+ first mounts) enter via the injection path
            mounts = [{"h": self.sink_h}] + [self.grafts[i] for i in picks]
            inj = []
            # sink is host numpy; deposited grafts are device tensors
            _np = lambda t: t if isinstance(t, np.ndarray) else t.numpy()
            for li in range(len(self.m.layers)):
                inj.append({"c": np.concatenate([_np(g["h"][li]["c"]) for g in mounts], axis=1),
                            "kpe": np.concatenate([_np(g["h"][li]["kpe"]) for g in mounts], axis=2)})
            kv_graft.set_injection_mla(self.m, inj)
            self.cur_mounts = picks
            self.cur_mount_n = sum(self.grafts[i]["ntok"] for i in picks)
        else:
            self.swap(picks)
        prompt_ids = self.encode(f"User: {user_text}\nAssistant:")
        seg_start_ntok = len(prompt_ids)
        row = self._forward(prompt_ids)
        kv_graft.clear_injection(self.m)     # bootstrap injection fired once
        out = [int(row.argmax())]
        for _ in range(ngen - 1):
            row = self._forward([out[-1]])
            out.append(int(row.argmax()))
        # the answer tokens are in the cache; record the live segment
        txt = self.decode(out)
        for stop in stops:
            if stop in txt:
                txt = txt.split(stop)[0]
        txt = txt.strip()
        turn_text = f"User: {user_text}\nAssistant: {txt}\n"
        gidx = None
        if deposit:
            # cache-deposit span covers prompt + ALL generated tokens (incl.
            # any post-stop tail) — the cache is the source of truth
            gidx = (self.deposit_from_cache(turn_text, seg_start_ntok + len(out))
                    if self.cache_deposits else self.deposit(turn_text))
        self.live_segs.append((gidx, seg_start_ntok + len(out)))
        evicted = self.evict()
        S = self.caches[0][0].shape[1]
        return txt, {"mounts": [i + 1 for i in picks], "resident": S,
                     "evicted": evicted, "live_tokens": sum(n for _, n in self.live_segs)}
