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
import contextlib
import gc
import os
import re
import time

import numpy as np

from core.mistral7b_tc import BlockTC, F, tc
from core import kv_graft
from core.graft_quant import (
    SUPPORTED_BITS, is_packed_payload, pack_kv_arrays, unpack_kv_arrays,
)
from core.grm_frame import (
    CAPTURE_PIN_LIVE, CAPTURE_PIN_MOUNT, CAPTURE_PIN_OFF, CAPTURE_PINS,
    capture_pin_mode, ephemeral_frame_enabled, frame_receipt, rs3_receipt,
    seat_near_live_enabled,
)
from core.grm_supersession import sup_resolve_enabled
from core import grm_demand
from core.grm_text_norm import normalize_glyphs
from core.grm_admission import (
    admission_info_fields,
    adm_decisive_enabled,
    chunk_trip_cap,
    decisive_admission_profile,
    fit_info_fields,
    identifier_serving_decision,
    is_identifier_binding,
    mountable_budget,
    ordered_identifier_tokens,
    plan_priority_fit,
    shuttle_trip_cap,
    split_info_fields,
)


class GraftPayloadMissingError(RuntimeError):
    """Requested graft payloads could not be restored for mounting."""

    def __init__(self, node_ids):
        self.node_ids = tuple(dict.fromkeys(int(i) for i in node_ids))
        self.missing_node_ids = self.node_ids
        ids = ", ".join(str(i) for i in self.node_ids)
        super().__init__(f"graft payload unavailable for node ids: {ids}")


class ArenaCache:
    def __init__(self, model, encode, decode, sink_text="<conversation>\n",
                 arena_width=256, route_layer=44, topk=3, live_turns=2,
                 max_live=4096, cache_deposits=True,
                 ephemeral=None, recency_mounts=2, prompt_template=None,
                 stop_sequences=None, length_debias=False,
                 revision_resolution=None, decisive_admission=None,
                 route_backend="auto",
                 route_profile=False, route_parity_check=False):
        # EPHEMERAL MODE ("clear the boat"): the live cache is reset at the
        # START of every turn — each turn runs on [sink | mounts | turn]
        # alone, so resident seats are CONSTANT for a conversation of ANY
        # length (the context window IS the repository). Recency becomes a
        # MOUNT: the last `recency_mounts` turn-grafts are always co-seated
        # for discourse cohesion (anaphora), ~40 seats instead of a growing
        # live region. Side effect: the live-window echo failure class
        # (corpus-100) cannot occur — there is no window to echo from.
        #
        # GRM-EB1 (David's spec, 2026-09-02: "the chat log is not kept in
        # memory context; any chat recall on facts is pulled via GRM") makes
        # this the PRODUCTION DEFAULT on every serving path. The registered
        # escape GRM_PERSISTENT_BOAT=1 restores the old persistent live
        # window for reproduction of FROZEN receipts only; it fails CLOSED to
        # ephemeral on an unknown token. An explicit bool still wins, so a
        # harness can pin either frame regardless of the ambient setting.
        self.ephemeral = ephemeral_frame_enabled(ephemeral)
        self.recency_mounts = recency_mounts
        self.m = model
        self.encode = encode            # text -> list of token ids
        self.decode = decode            # list of token ids -> text
        self.width = arena_width
        self.route_layer = route_layer
        self.topk = topk
        self.live_turns = live_turns
        self.cache_deposits = cache_deposits
        # SUP-WO1 L1 remains default OFF: its MLA receipt was undecidable.
        # GRM-SUP-L2-ON (operator decision 2026-08-30) makes M5-edge mount
        # resolution default ON. GRM_SUP_RESOLVE=0 restores the exact legacy
        # pass-through; an explicit bool pins experimental harness frames.
        self.length_debias = bool(length_debias)
        self.revision_resolution = sup_resolve_enabled(revision_resolution)
        # GRM-ADM2 (operator decision 2026-08-31) makes the frozen ADM1
        # decisive-admission rule default ON. GRM_ADM_DECISIVE=0 restores the
        # old fixed-k=3 admission path byte-for-byte; an explicit bool pins an
        # experiment frame independently of the ambient operator setting.
        self.decisive_admission = adm_decisive_enabled(decisive_admission)
        # Route backend is intentionally an operator opt-in.  ``auto`` is
        # the historical behavior (the existing dialect-specific environment
        # toggles still decide whether an optional CUDA bridge may engage),
        # ``python`` is a comparison/control arm, and ``cuda_ragged`` asks
        # the GQA dialect to try the exact padded ragged bank.  The latter
        # always retains the existing fail-closed fallback when a lexical,
        # hierarchy, or geometry rule cannot be represented by the bank.
        self.route_backend = self._normalize_route_backend(route_backend)
        # Profiling and the expensive Python-vs-CUDA rank assertion are
        # independently opt-in.  The E3 harness enables them for its parity
        # sample, then leaves both off while timing the resident hot path.
        self.route_profile = bool(route_profile)
        self.route_parity_check = bool(route_parity_check)
        self.last_route_receipt = None
        self.route_receipt_history = []
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
        # GRM S4 grounding ledger.  The repository installs the callback
        # after load/recovery so accepted counter changes ride its existing
        # metadata dirty/WAL path.  A bare ArenaCache remains fully usable:
        # counters still update in-memory and the callback is simply absent.
        self._s4_turn = 0
        self.s4_metadata_callback = None

    _ROUTE_BACKENDS = frozenset(("auto", "python", "cuda_ragged"))

    @classmethod
    def _normalize_route_backend(cls, value):
        if value is None:
            value = "auto"
        value = str(value).strip().lower()
        if value not in cls._ROUTE_BACKENDS:
            raise ValueError(
                "route_backend must be one of "
                f"{sorted(cls._ROUTE_BACKENDS)}, got {value!r}")
        return value

    def set_route_backend(self, value):
        """Select a route backend without changing any epoch-owned data.

        Backend selection changes execution policy only; it must not mutate
        a route key, candidate set, or the immutable CUDA bank.  Therefore no
        epoch bump belongs here.  This small mutator keeps the GPU E3 harness
        from reaching into a private attribute between its Python and flagged
        comparison arms.
        """
        self.route_backend = self._normalize_route_backend(value)

    def _route_backend_request(self):
        # ``__new__`` arena fixtures predate this field.  Treating their
        # missing field as ``auto`` preserves every old CPU test by
        # construction.
        return self._normalize_route_backend(
            getattr(self, "route_backend", "auto"))

    def _route_profile_enabled(self):
        if bool(getattr(self, "route_profile", False)):
            return True
        return os.environ.get("GRM_ROUTE_SEAMS_PROFILE", "").strip().lower() in (
            "1", "true", "yes", "on")

    def _begin_route_receipt(self):
        if not self._route_profile_enabled():
            # Do not let a previous profiled turn be misreported as the
            # receipt for a later unprofiled step.
            self.last_route_receipt = None
            return None
        receipt = {
            "schema": "grm_route_seams_route_v1",
            "requested_backend": self._route_backend_request(),
            "epoch": int(getattr(self, "_cuda_gqa_epoch", 0)),
            "terms_ms": {},
            "rebuilds": {},
            "parity": {"checked": False, "ok": None},
            "fallbacks": [],
            "_started_ns": time.perf_counter_ns(),
        }
        self._active_route_receipt = receipt
        return receipt

    def _route_receipt_term(self, name, elapsed_ns):
        receipt = getattr(self, "_active_route_receipt", None)
        if receipt is None:
            return
        terms = receipt["terms_ms"]
        terms[name] = terms.get(name, 0.0) + (float(elapsed_ns) / 1.0e6)

    def _route_profile_start(self):
        """Return a clock sample only while a receipt is actively requested."""
        if getattr(self, "_active_route_receipt", None) is None:
            return None
        return time.perf_counter_ns()

    def _route_profile_end(self, name, started_ns):
        if started_ns is not None:
            self._route_receipt_term(
                name, time.perf_counter_ns() - started_ns)

    def _route_receipt_rebuild(self, name, value):
        receipt = getattr(self, "_active_route_receipt", None)
        if receipt is None:
            return
        # Every event producer emits plain Python dict/list/scalar values so
        # this remains directly JSON-serializable in the E3 receipt.
        receipt["rebuilds"][name] = value

    def _route_receipt_fallback(self, reason):
        receipt = getattr(self, "_active_route_receipt", None)
        if receipt is None:
            return
        if reason not in receipt["fallbacks"]:
            receipt["fallbacks"].append(str(reason))

    def _route_receipt_parity(self, *, checked, ok=None, reference=None):
        receipt = getattr(self, "_active_route_receipt", None)
        if receipt is None:
            return
        receipt["parity"] = {
            "checked": bool(checked),
            "ok": None if ok is None else bool(ok),
            "reference": reference,
        }

    def _route_receipt_result(self, ranking):
        receipt = getattr(self, "_active_route_receipt", None)
        if receipt is not None:
            values = list(ranking or ())
            receipt["ranking_count"] = len(values)
            # E3 compares at k <= 16. Keeping only that prefix makes an
            # optional profile safe even for a deliberately unlimited route.
            receipt["ranking_prefix"] = [int(idx) for idx in values[:16]]
        return ranking

    def _finish_route_receipt(self, receipt, *, candidate_count=None):
        if receipt is None:
            return
        receipt["route_backend"] = getattr(self, "last_route_backend", "python")
        if candidate_count is not None:
            receipt["candidate_count"] = int(candidate_count)
        started_ns = receipt.pop("_started_ns", None)
        if started_ns is not None:
            receipt["terms_ms"]["route_wall_ms"] = (
                time.perf_counter_ns() - started_ns) / 1.0e6
        self.last_route_receipt = receipt
        history = getattr(self, "route_receipt_history", None)
        if history is None:
            history = []
            self.route_receipt_history = history
        history.append(receipt)
        if getattr(self, "_active_route_receipt", None) is receipt:
            self._active_route_receipt = None

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

    # ---------------------------------------------- GRM-RS3 capture geometry
    def capture_shift_for(self, pin=None):
        """The absolute query position a harvest forward runs at, per pin.

        DERIVED FROM THE ARENA, never typed.  ``mount`` is ``n_sink`` — the
        first arena seat, which is where ``_attempt``'s bootstrap branch always
        lands a mount (the sink occupies [0, n_sink) and the mount block
        follows it immediately, RoPE'd at ``cos.slice(0, 0, graft_seats)``).
        ``live`` is ``live_shift`` — ``n_sink + arena_width``, the first LIVE
        seat, which is where text fed live immediately before a question sits.

        ``off`` returns ``None``, which is not a position: it means "do not
        touch ``live_shift``", i.e. leave whatever the caller's state left, so
        the harvest is byte-identical to the legacy one.
        """
        if pin is None:
            pin = getattr(self, "_rs3_capture_pin_explicit", None)
        mode = capture_pin_mode(pin)
        if mode == CAPTURE_PIN_MOUNT:
            return int(self.n_sink)
        if mode == CAPTURE_PIN_LIVE:
            return int(self.live_shift)
        return None

    @contextlib.contextmanager
    def _capture_geometry(self, pin=None):
        """Pin every layer's capture-time QUERY position for one harvest.

        GRM-RS3 Part 1.  ``GptOssAttentionTC.__call__`` rotates queries at
        ``cos.slice(0, position_offset + shift, L)`` with ``shift =
        self_attn.live_shift`` (falling back to ``graft_seats`` when it is
        ``None``).  The harvested KEYS are pre-RoPE and therefore position-free
        — the relocatable-keys invariant is untouched by this — but the
        QUERIES decide each layer's attention output, which is the next
        layer's K/V input.  So the capture-time query position propagates into
        every layer above 0, and pinning it is what makes ``deposit`` produce
        the same graft whether or not a turn has been served yet.

        OFF is not "pin to the current value": it does not touch ``live_shift``
        at all, so the OFF path is the legacy path operand for operand.

        Yields the receipt dict the caller stamps onto the graft.
        """
        if pin is None:
            # Arena-level explicit channel; ``None`` still defers to the env,
            # which is DEFAULT OFF. Resolver order: argument > attribute >
            # env > OFF.
            pin = getattr(self, "_rs3_capture_pin_explicit", None)
        mode = capture_pin_mode(pin)
        shift = self.capture_shift_for(mode)
        receipt = {
            "capture_pin": mode,
            "capture_shift": shift,
            "capture_shift_derived_from": (
                "arena.n_sink" if mode == CAPTURE_PIN_MOUNT else
                "arena.live_shift (= n_sink + arena_width)"
                if mode == CAPTURE_PIN_LIVE else
                "not pinned; live_shift left exactly as the caller's state "
                "had it (legacy, byte-identical)"),
            "n_sink": int(self.n_sink),
            "arena_width": int(self.width),
            "live_shift": int(self.live_shift),
        }
        if mode == CAPTURE_PIN_OFF:
            # LEGACY: touch nothing. Record what the unpinned harvest actually
            # ran at, so an OFF receipt still says which geometry it got.
            observed = [getattr(L.self_attn, "live_shift", None)
                        for L in self.m.layers]
            first = observed[0] if observed else None
            # READ-ONLY and never coercive: this branch must not raise on a
            # value it merely reports, or OFF would stop being a no-op for a
            # caller whose live_shift is not an int.
            try:
                first = None if first is None else int(first)
            except (TypeError, ValueError):
                first = repr(first)
            receipt["capture_shift_observed"] = first
            yield receipt
            return
        before = [getattr(L.self_attn, "live_shift", None)
                  for L in self.m.layers]
        for L in self.m.layers:
            L.self_attn.live_shift = shift
        try:
            receipt["capture_shift_observed"] = int(shift)
            yield receipt
        finally:
            # Restore EXACTLY what was there, per layer — the pin is scoped to
            # this one harvest and must not leak into the serving path.
            for L, value in zip(self.m.layers, before):
                L.self_attn.live_shift = value

    # ------------------------------------------- GRM-RS3 band seating (Part 2)
    #: Receipt for the most recent bootstrap seating. ``None`` until a mount
    #: has been seated through ``_attempt``'s bootstrap branch.
    _rs3_last_seating = None

    #: The arena-level EXPLICIT channel for the seating lever, for callers that
    #: cannot reach ``_rs3_seat_plan``'s argument (``_attempt``'s bootstrap is
    #: reached through ``step`` / ``_probe_ladder_chat``, neither of which
    #: takes a seating argument). ``None`` = defer to the env, which is itself
    #: DEFAULT OFF, so the class default changes nothing.
    _rs3_seat_explicit = None

    #: The same channel for the capture pin, used by the deposit paths that
    #: run inside a serving turn. ``None`` = defer to the env (DEFAULT OFF).
    _rs3_capture_pin_explicit = None

    def _rs3_seat_plan(self, picks, seat_near_live=None):
        """Decide the mount block's ORDER and its band offset.

        GRM-RS3 Part 2.  Production seats the block at ``[n_sink, n_sink +
        mount_ntok)``: ``_attempt``'s bootstrap branch concatenates the sink
        payload and the mount payloads into ONE block and
        ``GptOssAttentionTC.__call__`` RoPEs that whole block at
        ``cos.slice(0, 0, graft_seats)``.  So the plan head — ``picks[0]``, the
        first seat ``grm_admission.plan_priority_fit`` fills — lands at the
        band's SINK end, as far from the question as the band allows, with the
        filler between it and the live tokens.

        ON inverts that: the block is filled from the TOP DOWN, so the PLAN
        HEAD is last in the block and its LAST TOKEN is adjacent to
        ``live_shift``, with filler and the other mounts below it.  The sink is
        untouched — moving it would be a second variable.

        Returns ``(seat_order, info)``.  OFF returns ``picks`` unchanged and an
        info dict whose ``seat_near_live`` is ``False``, and the caller then
        takes the legacy path operand for operand.
        """
        picks = [int(v) for v in picks]
        if seat_near_live is None:
            # The ARENA-LEVEL explicit channel, for a caller that cannot reach
            # this call's argument (``_attempt``'s bootstrap branch is reached
            # through ``step``/``_probe_ladder_chat``, neither of which takes a
            # seating argument). ``None`` here still means "ask the env", so
            # the default remains OFF and the resolver order is unchanged:
            # explicit argument > arena attribute > env > OFF.
            seat_near_live = getattr(self, "_rs3_seat_explicit", None)
        enabled = seat_near_live_enabled(seat_near_live)
        mount_ntok = sum(int(self.grafts[i]["ntok"]) for i in picks)
        n_sink = int(self.n_sink)
        info = {
            "seat_near_live": bool(enabled),
            "seat_order": list(picks),
            "seat_order_legacy": list(picks),
            "plan_head": (int(picks[0]) if picks else None),
            "mount_ntok": int(mount_ntok),
            "n_sink": n_sink,
            "arena_width": int(self.width),
            "live_shift": int(self.live_shift),
            "mount_pos0_legacy": n_sink,
            "mount_pos0": n_sink,
            "delta_positions": 0,
            "seat_offset_plan_head": (n_sink if picks else None),
            "seat_rule": (
                "LEGACY: the block is seated at [n_sink, n_sink + mount_ntok) "
                "and the plan head (picks[0]) takes the band's SINK end"),
        }
        if not picks:
            # Nothing to seat: the lever cannot have moved anything, and a
            # receipt claiming otherwise would be false.
            info["seat_near_live"] = False
            info["declined_reason"] = "no_mounts_to_seat"
            info["seat_near_live_requested"] = bool(enabled)
            return picks, info
        if not enabled:
            return picks, info
        # TOP-DOWN FILL. The plan head goes LAST in the block so its final row
        # is the block's final row; the rest keep their relative order below
        # it. The block as a whole then moves up so its last row sits at
        # live_shift - 1, i.e. immediately below the first live token.
        seat_order = picks[1:] + picks[:1]
        mount_pos0 = int(self.live_shift) - int(mount_ntok)
        head_ntok = int(self.grafts[picks[0]]["ntok"])
        info.update({
            "seat_order": [int(v) for v in seat_order],
            "mount_pos0": int(mount_pos0),
            "delta_positions": int(mount_pos0 - n_sink),
            # The plan head is the LAST member of the block, so it starts
            # head_ntok rows before the block's end.
            "seat_offset_plan_head": int(self.live_shift) - head_ntok,
            "plan_head_ntok": head_ntok,
            "plan_head_last_position": int(self.live_shift) - 1,
            "plan_head_adjacent_to_live_shift": True,
            "seat_rule": (
                "GRM_SEAT_NEAR_LIVE: the block is filled from the TOP DOWN — "
                "the plan head (picks[0]) is seated LAST so its final token "
                "sits at live_shift - 1, immediately below the first live "
                "token; filler and the other mounts sit below it; the sink "
                "stays at [0, n_sink)"),
        })
        if mount_pos0 < n_sink:
            # A block wider than the band cannot be seated top-down without
            # overrunning the sink. Production's own width law forbids that
            # block anyway (swap raises above self.width); fail closed to the
            # legacy seating rather than corrupt the sink.
            info.update({
                "seat_near_live": False,
                "seat_order": list(picks),
                "mount_pos0": n_sink,
                "delta_positions": 0,
                "seat_offset_plan_head": n_sink,
                "seat_rule": (
                    "GRM_SEAT_NEAR_LIVE requested but DECLINED: mount_ntok "
                    f"({mount_ntok}) would place the block at {mount_pos0}, "
                    f"below n_sink ({n_sink}), overrunning the sink. Failed "
                    "closed to the legacy seating."),
                "declined_reason": "block_wider_than_band",
                "seat_near_live_requested": True,
            })
            return picks, info
        return [int(v) for v in seat_order], info

    def _rs3_rotate_injection(self, inj, seat_info):
        """Pre-rotate the MOUNT rows of a bootstrap injection block.

        THE MECHANISM, and why it is a POSITION change and nothing else.  The
        attention layer will RoPE the whole injected block at absolute
        positions ``[0, graft_seats)`` no matter what — that call site is not
        changed by this order.  The mount occupies block rows ``[n_sink,
        n_sink + mount_ntok)``, so the layer will rotate row ``n_sink + j`` by
        ``n_sink + j``.  To land it at ``mount_pos0 + j`` the payload is
        pre-rotated HERE by ``delta = mount_pos0 - n_sink``; the layer's own
        rotation then composes with it to the requested net position.  That
        composition is the same one ``_rope_block_at`` already relies on to
        re-seat a graft, which is the pre-RoPE relocatable-keys invariant
        (``docs/GRM_Methodology.md`` §4) doing exactly what it promises.

        Only the ROPE-carrying payload key is touched (``ROPE_KEYS``); the
        value payload is passed through untouched, and the PHYSICAL cache rows
        are unchanged — the mount stays a packed prefix immediately after the
        sink.  The sink's own rows are never rotated.
        """
        delta = int(seat_info["delta_positions"])
        n_sink = int(seat_info["n_sink"])
        if not delta:
            return inj
        rope_keys = set(self.ROPE_KEYS)
        out = []
        for block in inj:
            new_block = {}
            for key, dim in self.PAYLOAD:
                array = block[key]
                array = (array if isinstance(array, np.ndarray)
                         else array.numpy())
                if key not in rope_keys:
                    new_block[key] = array
                    continue
                index = [slice(None)] * array.ndim
                index[dim] = slice(0, n_sink)
                sink_rows = array[tuple(index)]
                index[dim] = slice(n_sink, array.shape[dim])
                mount_rows = array[tuple(index)]
                if mount_rows.shape[dim] == 0:
                    new_block[key] = array
                    continue
                rotated = self._rs3_rotate_rows(mount_rows, dim, delta)
                new_block[key] = np.ascontiguousarray(
                    np.concatenate([sink_rows, rotated], axis=dim))
            out.append(new_block)
        return out

    def _rs3_rotate_rows(self, array, dim, delta):
        """Rotate EVERY row of a host payload by the SAME ``delta`` positions.

        THE CONSTANT-DELTA LAW, and why it is not ``_rope_tensor(x, delta)``.
        RoPE tables are indexed by ABSOLUTE position: ``_rope_tensor(x, pos0)``
        rotates row ``j`` by ``pos0 + j``, because that is what rotating a
        SPAN at a seat range means.  A band RELOCATION is a different
        operation: every row must move by the same amount, so that composing
        with the layer's own ``n_sink + j`` rotation yields ``mount_pos0 + j``
        — the block translated, its internal geometry intact.

        Rotating the block at ``pos0 = delta`` instead composes to
        ``n_sink + delta + 2j``: the extra rotation GROWS along the block, so
        the mount's rows are progressively de-phased relative to each other.
        That is not a relocation, it is a shear — the keys stop being the same
        keys.  MEASURED here in pure numpy before any GPU run: constant-delta
        composition matches the direct rotation to 4.4e-16, the span-style
        pre-rotation is off by 4.56 on the same input.

        So this rotates against ONE table row — position ``|delta|``,
        broadcast over the sequence axis — using the arena's own rope tables,
        which is the operation ``_rope_block_at`` composes with when it
        re-seats a graft.  A negative delta uses the inverse rotation, exactly
        as ``_rope_block_at(..., inverse=True)`` does.
        """
        delta = int(delta)
        if not delta:
            return array
        seq_axis = array.ndim - 2
        moved = array if dim == seq_axis else np.moveaxis(array, dim, seq_axis)
        moved = np.ascontiguousarray(moved).astype(np.float32)
        # ONE table row, broadcast: the constant-delta rotation.
        cos_row = np.asarray(
            self.m.rope_cos.float().numpy()[abs(delta)], dtype=np.float32)
        sin_row = np.asarray(
            self.m.rope_sin.float().numpy()[abs(delta)], dtype=np.float32)
        if delta < 0:
            sin_row = -sin_row
        x = moved
        if self.ROPE_PAIR_SWAP:
            x = self._pair_swap_last(x)
        half = x.shape[-1] // 2
        rotated_half = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
        out = x * cos_row + rotated_half * sin_row
        out = out.astype(array.dtype)
        if dim != seq_axis:
            out = np.moveaxis(out, seq_axis, dim)
        return np.ascontiguousarray(out)

    # ------------------------------------------------------------ repository
    def deposit(self, text, capture_pin=None):
        """Standalone harvest deposit (document-in-isolation semantics, one
        dedicated forward). Stored DEVICE-resident: mounts never re-upload.

        GRM-RS3: ``capture_pin`` pins the harvest forward's QUERY position to a
        registered geometry (``mount`` / ``live``) instead of inheriting
        whatever ``live_shift`` the previous turn left behind.  Default
        ``None`` resolves through ``GRM_CAPTURE_PIN``, itself DEFAULT OFF, and
        OFF does not touch ``live_shift`` at all — the legacy harvest, byte
        for byte.  Every harvest path in the stack (``feed``'s ephemeral
        branch, the P2C split children, the consolidation note, the abstention
        deposit, the repository installers) reaches the model through THIS
        call, so pinning it here pins all of them.
        """
        ids = self.encode(text)
        with self._capture_geometry(capture_pin) as capture:
            h = self._harvest(ids)
            dev = [{key: tc.tensor(
                        np.ascontiguousarray(h[li][key])).astype(self.dt)
                    for key, _ in self.PAYLOAD}
                   for li in range(len(self.m.layers))]
            # The ROUTING KEY comes from the same pinned geometry: _node_key
            # runs its own partial harvest forward when it is not handed one,
            # and a key captured at a different position than the payload
            # would be a second, unregistered variable.
            cent = self._node_key(text, h)
        graft = {"h": dev, "cent": cent, "ntok": len(ids), "text": text}
        graft.update(capture)
        self.grafts.append(graft)
        self._bump_cuda_gqa_epoch()
        return len(self.grafts) - 1

    def deposit_from_cache(self, text, seg_ntok, route_key=None,
                           capture_pin=None):
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
        key_from_cache=True (the measured-5/6 exploratory mode).

        GRM-RS3 ``capture_pin``.  This path runs NO harvest forward for the
        PAYLOAD — it slices the live cache, whose geometry is already fixed by
        the turn that built it — so the pin cannot and does not move the
        payload here, and the receipt says so.  It DOES cover the one forward
        this path can still run: the standalone ``_node_key(text)`` fallback
        when no ``route_key`` was carried.  Pinning that keeps the routing key
        from depending on whether a turn has been served, for the same reason
        the payload pin exists.  OFF touches nothing.
        """
        p0 = self.live_shift + self.pos - seg_ntok      # span's first seat
        dev = self._export_cache_payloads(seg_ntok, p0)
        if dev is None:
            dev = [self._export_cache_payload(cache, seg_ntok, p0)
                   for cache in self.caches]
        cent = None
        for li, seg in enumerate(dev):
            if li == self.route_layer and cent is None:
                cent = self._cache_key_of(seg)
        with self._capture_geometry(capture_pin) as capture:
            if cent is None:
                cent = (
                    np.asarray(route_key, dtype=np.float32)
                    if route_key is not None else self._node_key(text)
                )
        capture = dict(capture)
        capture["capture_payload_source"] = "live_cache_slice"
        capture["capture_span_pos0"] = int(p0)
        capture["capture_pin_moves_payload"] = False
        graft = {"h": dev, "cent": cent, "ntok": seg_ntok, "text": text}
        graft.update(capture)
        self.grafts.append(graft)
        self._bump_cuda_gqa_epoch()
        return len(self.grafts) - 1

    def route(
        self,
        bare_text,
        exclude,
        limit=None,
        *,
        route_backend=None,
        query_lex=None,
        probe_key=None,
        semantic_only=False,
    ):
        """Rank repository grafts for one query.

        The epoch-cached route-seams substrate (per-turn rebuild inventory,
        profiling receipt, epoch-cached candidate base, native/CUDA-ragged
        fallback via the instance ``route_backend`` policy) is the shared
        machinery.  The three-pass keyword-only seams layer on top of it:

        ``route_backend`` and ``query_lex`` are explicit orchestration seams
        for staged/preflight routing.  Their default ``None`` values preserve
        the production selector byte-for-byte:

        - ``route_backend=None`` keeps the native/CUDA-then-Python fallback
          driven by the instance policy (``set_route_backend`` / the
          ``GRM_GQA_CUDA_ROUTE`` gate).
        - ``route_backend="python"`` bypasses native/CUDA scoring so a gate
          can compare the exact reference ranking against an accelerated run.
        - ``route_backend="cuda"`` requires the opt-in CUDA scorer to engage
          and fails closed instead of silently relabelling a fallback. If
          query-side lexical policy then requires an exact Python rescore,
          ``last_route_backend`` remains ``"cuda"`` (the search engine) and
          ``last_route_policy_backend`` names that rescore explicitly.
        - ``query_lex=None`` keeps ``GRM_ROUTE_QUERY_LEX`` authoritative;
          an explicit bool changes only the query-side content-word extension
          for this call.  The established rare-token channel is unchanged.
        - ``semantic_only=True`` disables both lexical channels for the call.
          This is the exact-ragged CUDA contract: model-native fp32 Q/K
          evidence only, with lexical policy reported separately.

        ``probe_key`` lets prep capture the model query once and feed those
        identical fp32 values to both backends.  It is intentionally a
        keyword-only flag-plumbing hook; router kernels and score laws are
        untouched.
        """
        if route_backend not in (None, "python", "cuda"):
            raise ValueError(
                "route_backend must be None, 'python', or 'cuda'")
        self.last_route_policy_backend = None
        receipt = self._begin_route_receipt()
        profiling = receipt is not None
        candidate_count = 0
        try:
            if not self.grafts:
                return self._route_receipt_result([])
            route_limit = None if limit is None else max(0, int(limit))
            if route_limit == 0:
                return self._route_receipt_result([])

            started = self._route_profile_start()
            p = self._probe_key(bare_text) if probe_key is None else probe_key
            self._route_profile_end("probe_key_ms", started)
            # A probe is necessarily turn-specific.  Node keys are a
            # different class: on the flagged GQA path they live in the
            # immutable epoch bank and are not re-marshaled here.
            if profiling:
                self._route_receipt_rebuild(
                    "probe_key", {"status": "per_turn", "shape": tuple(
                        int(dim) for dim in np.asarray(p).shape)})

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
            #
            # THREE-PASS seams: ``semantic_only`` drops both lexical channels
            # for the call; an explicit ``query_lex`` bool overrides only the
            # query-side content-word extension.  Defaults reproduce the
            # route-seams behavior byte-for-byte.
            started = self._route_profile_start()
            qrare = set() if semantic_only else self._rare_tokens(bare_text)
            query_lex_enabled = (
                self._route_query_lex_enabled()
                if query_lex is None else bool(query_lex)
            )
            qlex = (
                set()
                if semantic_only
                else (self._query_lex_tokens(bare_text)
                      if query_lex_enabled else qrare)
            )
            content_extra = qlex - qrare
            self._route_profile_end("lexical_keys_ms", started)
            if profiling:
                self._route_receipt_rebuild(
                    "lexical_keys", {
                        "status": "per_turn",
                        "rare_count": len(qrare),
                        "query_lex_count": len(qlex),
                        "content_extra_count": len(content_extra),
                    })

            # P1 follow-on (profile-named): the eligible BASE (retired/kind
            # filter) depends only on graft state, so it is epoch-cached in
            # _route_cand_base(); only the per-call `exclude` filter runs here.
            # Byte-identical to the old single comprehension: same ascending
            # order, same three membership conditions. When `exclude` is empty
            # the cached base list itself is used — every downstream consumer
            # (_native_route_order, _vector_route_scores, the Python-fallback
            # scoring loops) reads cand without mutating it.
            started = self._route_profile_start()
            cand_base = self._route_cand_base()
            self._route_profile_end("candidate_base_ms", started)
            if profiling:
                self._route_receipt_rebuild(
                    "candidate_base", getattr(
                        self, "_last_route_cand_base_event", {
                            "status": "unknown", "epoch": int(getattr(
                                self, "_cuda_gqa_epoch", 0))}))
            started = self._route_profile_start()
            if exclude:
                cand = [i for i in cand_base if i not in exclude]
            else:
                cand = cand_base
            candidate_count = len(cand)
            self._route_profile_end("candidate_exclude_ms", started)

            # Native lexical channel is rare-key only (node indexes unchanged).
            # The native/CUDA route APIs expose only final ids, not the raw
            # per-node score needed by L1. Debias deliberately takes the
            # Python scoring path.
            #
            # Two orthogonal backend axes coexist here:
            #   * the INSTANCE policy (``self._route_backend_request()`` ->
            #     auto/python/cuda_ragged), the route-seams / E3 selector; and
            #   * the per-call THREE-PASS ``route_backend`` kwarg
            #     (None/python/cuda) for staged preflight comparison.
            # ``route_backend="python"`` (kwarg) or the instance "python"
            # policy both skip the native/CUDA scorer; a fail-closed CUDA
            # request is enforced after the attempt.
            backend_request = self._route_backend_request()
            native_order = None
            if getattr(self, "length_debias", False):
                self._route_receipt_fallback("length_debias")
            elif route_backend == "python" or backend_request == "python":
                self._route_receipt_fallback("route_backend_python")
            else:
                started = self._route_profile_start()
                native_order = self._native_route_order(
                    p, qrare, cand, limit=route_limit, exclude=exclude)
                # CUDA writes its own non-overlapping bank/marshal/kernel/
                # remap terms. Keep this enclosing timer only for the CPU
                # native attempt or a fail-closed fallback, so receipt users
                # never add a parent CUDA total to its child decomposition.
                if (native_order is None
                        or getattr(self, "last_route_backend", None) != "cuda"):
                    self._route_profile_end("native_backend_attempt_ms", started)
            # THREE-PASS fail-closed: an explicit ``route_backend="cuda"`` must
            # see the CUDA scorer actually engage; otherwise raise instead of
            # silently relabelling a Python fallback.
            cuda_engaged = (
                native_order is not None
                and getattr(self, "last_route_backend", None) == "cuda")
            if route_backend == "cuda" and not cuda_engaged:
                self.last_route_policy_backend = "fallback_rejected"
                raise RuntimeError(
                    "requested CUDA route did not engage; the query may "
                    "require lexical/native policy or the CUDA bank may be "
                    "unavailable")
            if native_order is not None:
                started = self._route_profile_start()
                needs_rescore = self._query_lex_needs_rescore(
                    content_extra, cand)
                self._route_profile_end("lex_rescore_check_ms", started)
                if not needs_rescore:
                    return self._route_receipt_result(native_order)
                self._route_receipt_fallback("query_lex_rescore")
                if cuda_engaged:
                    # The CUDA engine ranked; only the query-side lexical
                    # policy finishes the ordering.  Name that explicitly so
                    # the receipt keeps route_backend=cuda (search engine) and
                    # attributes the rescore to python_query_lex_rescore.
                    self.last_route_policy_backend = "python_query_lex_rescore"
            elif backend_request == "cuda_ragged":
                # Exact CUDA eligibility is intentionally narrow.  A missing
                # bank or sidecar falls through to the authoritative Python
                # law; the GQA helper records the concrete reason when it can.
                self._route_receipt_fallback("cuda_ragged_unavailable")

            self.last_route_backend = "python"
            started = self._route_profile_start()
            base = self._vector_route_scores(p, cand)
            if base is None:
                base = {}
                for i in cand:
                    score = self._cent_score(p, self.grafts[i])
                    if np.isfinite(score):
                        base[i] = score
            self._route_profile_end("python_centroid_score_ms", started)
            if profiling:
                self._route_receipt_rebuild(
                    "node_centroids", {
                        "status": "python_per_candidate",
                        "candidate_count": int(candidate_count),
                    })
            started = self._route_profile_start()
            base = self._length_debias_scores(base, cand)
            # dialect hook: a raw-score channel (GQA layer-0 |q.k|) rescales
            # per-route into the O(1) band the lexical bonus was calibrated
            # against. MLA cosine is already there — identity.
            base = self._normalize_scores(base)
            self._route_profile_end("score_normalize_ms", started)
            started = self._route_profile_start()
            scored = []
            for i in cand:
                if i not in base:
                    continue
                score = base[i] + self._lex_bonus(qlex, self.grafts[i])
                if np.isfinite(score):
                    scored.append((score, i))
            scored.sort(key=lambda item: -item[0])
            self._route_profile_end("lex_rescore_apply_ms", started)
            ranking = [i for _, i in scored]          # best first
            if route_limit is not None:
                ranking = ranking[:route_limit]
            # THREE-PASS: when the exact CUDA engine engaged and only the
            # query-lex policy re-ranked, the search backend remains "cuda".
            if route_backend == "cuda" and cuda_engaged:
                self.last_route_backend = "cuda"
            return self._route_receipt_result(ranking)
        finally:
            self._finish_route_receipt(
                receipt, candidate_count=candidate_count)

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

    @staticmethod
    def _route_key_length(g):
        """Number of node-key opportunities represented by a graft.

        GQA keeps its route key as (heads, key_tokens, dim), so the middle
        dimension is authoritative. MLA stores one collapsed centroid; its
        source token count is the only retained length witness and is used as
        the dialect-transfer definition. Missing/malformed lengths fail to 1.
        """
        cent = g.get("cent")
        if cent is not None:
            shape = np.asarray(cent).shape
            if len(shape) >= 2:
                try:
                    return max(1, int(shape[-2]))
                except (TypeError, ValueError, OverflowError):
                    pass
        try:
            return max(1, int(g.get("ntok", 1)))
        except (TypeError, ValueError, OverflowError):
            return 1

    @staticmethod
    def _length_debias_normalizer(key_length):
        """Frozen L1 form: sqrt(log2(K + 1)), anchored to 1 at K=1."""
        try:
            key_length = max(1, int(key_length))
        except (TypeError, ValueError, OverflowError):
            key_length = 1
        return float(np.sqrt(np.log2(key_length + 1.0)))

    def _length_debias_scores(self, base, cand):
        """Apply the SUP-WO1 L1 log-length-normalized latent score.

        The 2026-07-08 E2E decomposition found that GQA ranks the maximum
        absolute q·k pair: a longer node gets more chances to produce an
        extreme even when it has zero lexical overlap. Extreme-value growth
        is logarithmic in the number of opportunities, so the frozen form is
        ``raw_score / sqrt(log2(K + 1))`` where K is node-key length. It keeps
        the load-bearing max/salience statistic while removing its unbounded
        reward for extra keys; it is intentionally milder than 1/sqrt(K).
        MLA has already collapsed its key to a centroid, so source ``ntok``
        supplies K for the plan's required dialect-transfer measurement.
        This is latent-only: no kind, importance, or other prior enters it.
        """
        if not getattr(self, "length_debias", False) or not base:
            return base
        out = {}
        for i in cand:
            if i not in base:
                continue
            score = float(base[i])
            norm = self._length_debias_normalizer(
                self._route_key_length(self.grafts[i]))
            # GQA's max-|q.k| score is non-negative. MLA cosine can be
            # negative; multiplying a negative score prevents length from
            # turning the penalty into an accidental boost toward zero.
            out[i] = score / norm if score >= 0.0 else score * norm
        return out

    @staticmethod
    def _metadata_node_ids(value, node_count):
        """Parse an M5 metadata edge fail-closed at the graph boundary."""
        if value is None:
            return ()
        if isinstance(value, (str, int, float)):
            value = (value,)
        out = []
        try:
            values = iter(value)
        except TypeError:
            return ()
        for raw in values:
            try:
                idx = int(raw)
            except (TypeError, ValueError, OverflowError):
                continue
            if 0 <= idx < node_count and idx not in out:
                out.append(idx)
        return tuple(out)

    def _revision_mount_heads(self, picks):
        """Return the revision heads present in one proposed mount set.

        M5's existing authoritative edge is replacement metadata
        ``supersedes=[older ids]``. Walk that edge transitively: an older
        candidate is removed only when one of its explicit successors is
        also present. Nodes remain in route results, and a descent/request
        that presents an older node without its successor still mounts it.

        Any cycle is corrupt revision metadata. Resolution fails open to the
        original set rather than deleting every member of a cycle or hanging.
        """
        # The OFF arm is the byte-for-byte control path: preserve caller
        # ordering and duplicates exactly, doing no graph parsing at all.
        if not getattr(self, "revision_resolution", False):
            return list(picks or ())

        unique = []
        for raw in picks or ():
            try:
                idx = int(raw)
            except (TypeError, ValueError, OverflowError):
                continue
            if 0 <= idx < len(self.grafts) and idx not in unique:
                unique.append(idx)
        if len(unique) < 2:
            return unique

        present = set(unique)
        superseded_present = set()
        memo = {}

        def ancestors(idx, visiting):
            if idx in visiting:
                raise ValueError("cycle in explicit supersedes metadata")
            if idx in memo:
                return memo[idx]
            visiting.add(idx)
            meta = self.grafts[idx].get("metadata") or {}
            direct = self._metadata_node_ids(
                meta.get("supersedes"), len(self.grafts))
            found = set(direct)
            for old in direct:
                found.update(ancestors(old, visiting))
            visiting.remove(idx)
            memo[idx] = found
            return found

        try:
            for idx in unique:
                superseded_present.update(ancestors(idx, set()) & present)
        except ValueError:
            return unique
        return [idx for idx in unique if idx not in superseded_present]

    def _resolve_revision_mounts(self, picks):
        """Feature-flagged L2 mount-set resolution entry point."""
        return self._revision_mount_heads(picks)

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
            self._last_route_cand_base_event = {
                "status": "reused", "epoch": int(epoch),
                "candidate_count": len(cached),
            }
            return cached
        base = [i for i, g in enumerate(self.grafts)
                if not g.get("retired")
                and g.get("kind", "turn") != "recall"]
        self._route_cand_base_cache = base
        self._route_cand_base_epoch = epoch
        self._last_route_cand_base_event = {
            "status": "rebuilt", "epoch": int(epoch),
            "candidate_count": len(base),
        }
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

    #: SC1.1 -- a ONE-CALL override of the glyph switch, set only by
    #: ``_grounding_verdict``'s explicit-mode contextmanager.  ``None`` means
    #: "read the switch", which is what every other caller gets.  It exists
    #: because the grounding receipt must compute the LEGACY counterfactual
    #: while the switch is ON, and because ``_rare_tokens``/``_caps_tokens``
    #: are single-argument stub seams across the suite (seven fixtures
    #: replace them with one-argument lambdas) -- widening their signatures
    #: to carry the flag would break every one of them.
    _glyph_norm_override = None

    @staticmethod
    @contextlib.contextmanager
    def _glyph_norm(normalized):
        """Force the glyph projection on or off for the enclosed block.

        Restores the previous value on exit, including on exception, so a
        raised grounding call can never leave the override latched.
        """
        previous = ArenaCache._glyph_norm_override
        ArenaCache._glyph_norm_override = (
            None if normalized is None else bool(normalized))
        try:
            yield
        finally:
            ArenaCache._glyph_norm_override = previous

    @staticmethod
    def _norm_text(text):
        """SC1.1: project the two REGISTERED glyph classes, switch-gated.

        ON (``GRM_LSR_FIXES`` default) applies ``normalize_glyphs`` — U+2010/
        U+2011 collapse to "-" and paired Markdown emphasis is stripped —
        so the lexical channels obey the same principle the DET1 value
        comparator does: *value comparison is semantics, not glyphs*.
        Measured cause: the model emits "Quartz‑8‑Jade" (U+2011) and the
        token class ``[A-Za-z0-9][\\w:.,\\-]*`` shatters it into {"8"},
        while the mounted node's ASCII "Quartz-8-Jade" tokenizes whole, so a
        CORRECT answer failed ``content <= have`` (SC1 G3: recovery 0/2 with
        ``recovery_blocked_by_grounding_only = 2``).

        OFF is byte-identical legacy behavior: ``normalize_glyphs`` is not
        called at all, so the P2C Arm-0 reproduction arm is untouched.

        ``_glyph_norm_override`` (set only by ``_glyph_norm``) wins over the
        switch for the enclosed block.  Nothing else may set it: the switch
        is the law everywhere outside grounding's own counterfactual.
        """
        normalized = ArenaCache._glyph_norm_override
        if normalized is None:
            normalized = ArenaCache._lsr_fixes_enabled()
        if not normalized:
            return text
        return normalize_glyphs(text)

    @staticmethod
    def _rare_tokens(text):
        """Code/number-shaped tokens — the verbatim payload a digest must
        preserve. Mechanically checkable: the librarian holds the sources."""
        out = set()
        text = ArenaCache._norm_text(text)
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
        # Prior art: local GRM digest repetition QC (GRM contributors, 2026).
        # Extend word-level QC to character runs seen in C7. No prior art
        # known to me for this exact rule: >=3 ellipses or >=6 punctuation
        # characters, allowing whitespace. Ordinary "..." remains valid.
        if re.search(r"(?:…\s*){3,}|(?:[^\w\s]\s*){6,}", text):
            return False
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
        # Even extractive era folds retire their sources. Validate that every
        # source has a recoverable payload before either folding path can
        # alter cache state or make that lifecycle change.
        self._ensure_h(idxs)        # sources may be paged out (cold storage)
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
                # Prior art: EB1 harmony_turn + Arena._attempt stop contract
                # (GRM contributors, 2026), reused via configured template.
                # New: apply that same contract to standalone folds. No
                # duplicate Harmony formatter or model-name detection.
                # Dispatch on the configured wrapper's final-channel suffix;
                # other model templates keep legacy prompt/decode bytes.
                formatted = None
                if self.prompt_template is not None:
                    user_text = prompt.removeprefix("User: ").rsplit("\nAssistant:", 1)[0]
                    formatted = self._format_step_prompt(user_text)
                harmony = formatted is not None and formatted.endswith(
                    "<|start|>assistant<|channel|>final<|message|>")
                stops = self.stop_sequences if harmony else ()
                if harmony:
                    prompt = formatted + primer
                try:
                    ids = self.encode(prompt)
                    with tc.no_grad():
                        lg, caches = self.m(np.array([ids], dtype=np.int64),
                                            last_token_only=True)
                    pos = len(ids)
                    out = [int(lg.numpy()[0, -1].argmax())]
                    for _ in range(ngen - 1):
                        if any(s in self.decode(out) for s in stops):
                            break
                        with tc.no_grad():
                            lg, caches = self.m(
                                np.array([[out[-1]]], dtype=np.int64),
                                kv_caches=caches, position_offset=pos,
                                last_token_only=True)
                        pos += 1
                        out.append(int(lg.numpy()[0, -1].argmax()))
                    decoded = self.decode(out)
                    for stop in stops:
                        decoded = decoded.split(stop, 1)[0]
                    t = (primer + " " + decoded).strip()
                    for stop in ("\nUser:", "User:"):
                        if stop in t:
                            t = t.split(stop)[0]
                    t = t.strip()
                    qc = self._digest_qc(t, None, forbid_lists=True)
                    relaxed_list = False
                    if not qc and self.ALLOW_HIGH_COVERAGE_LIST_DIGESTS:
                        qc = self._digest_qc(
                            t, None, forbid_lists=False)
                        relaxed_list = qc
                    # Shape QC runs before coverage; a punctuation collapse
                    # cannot be rescued by fact tokens or list relaxation.
                    cov = self._coverage(t, need) if qc else 0.0
                    if relaxed_list and cov < self.MIN_FOLD_KEEP:
                        relaxed_list = qc = False
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
        frees least-recently-mounted nodes). Touches the LRU clock and
        raises GraftPayloadMissingError when a requested payload cannot be
        restored."""
        self.mount_clock = getattr(self, "mount_clock", 0) + 1
        missing = []
        for i in idxs:
            g = self.grafts[i]
            g["last_used"] = self.mount_clock
            if g.get("h") is None:
                if self.node_loader is not None:
                    g["h"] = self.node_loader(i)
                    self.page_ins = getattr(self, "page_ins", 0) + 1
                if g.get("h") is None:
                    missing.append(i)
        if missing:
            raise GraftPayloadMissingError(missing)

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
        """Drop live segments beyond the recency window from the cache.

        GRM-EB1.  Under the SPEC (ephemeral) frame ``live_turns`` governs only
        the CURRENT turn's segments.  ``step()`` clears the boat at the START
        of every turn, so the window this method trims never spans a turn
        boundary: whatever it keeps is discarded wholesale by the next turn's
        clear.  The invariant the frame owes the spec — "after N turns the
        live cache holds only turn N's segments" — is therefore a property of
        ``step()``'s clear, and ``tests/test_grm_eb1_ephemeral_frame.py`` pins
        it directly rather than inferring it from this window arithmetic.
        """
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

    # ------------------------------------------------------ S1 telemetry
    # GRM_IMPORTANCE_PLAN.md WO-1: opt-in, read-only attention-mass tap.
    # The tap itself lives on MLAAttentionTC (core/minicpm3_tc.py, absorbed
    # decode's softmax site); this section owns seat-range -> mount-id
    # aggregation, MEANED over heads (done in the tap) and ALL layers (done
    # here), reported as the registered headline metric: share of total
    # non-live mass. Per-layer numbers are exposed as diagnostics only and
    # never feed the headline scalar (plan: "no post-hoc layer picking").
    def set_telemetry(self, enabled):
        """Enable/disable the S1 tap on every layer. Mirrors the existing
        absorbed_decode/live_shift broadcast pattern (set once, external to
        the per-attempt hot path). Resets the accumulator on EVERY call,
        both directions: _telemetry_mass lives on the LAYER (MLAAttentionTC
        instance), not on this ArenaCache — the model's layers are shared
        across ArenaCache instances (a fresh ArenaCache does NOT imply a
        fresh accumulator), so a stale accumulator from a previous ON
        stretch on a DIFFERENT arena over the SAME model must never leak
        into this one's first aggregation call. _attempt()'s own
        reset-per-attempt narrows the window further once turns are
        running; this call is the outer bound for anything before the
        first attempt (bootstrap forward calls via feed(), which never go
        through _attempt())."""
        for L in self.m.layers:
            L.self_attn.telemetry = bool(enabled)
            L.self_attn.reset_telemetry()

    def _telemetry_enabled(self):
        return any(getattr(L.self_attn, "telemetry", False)
                   for L in self.m.layers)

    def _mount_seat_ranges(self):
        """Arena seats occupied by each currently-mounted node, in mount
        order: mounts are a packed PREFIX of the arena starting at n_sink
        (no hole padding in the physical cache tensor — the hole is a
        RoPE-position-only concept, see module docstring). Returns
        {graft_idx: (start, end)} with end exclusive."""
        ranges = {}
        seat = self.n_sink
        for i in self.cur_mounts:
            n = self.grafts[i]["ntok"]
            ranges[i] = (seat, seat + n)
            seat += n
        return ranges

    def s1_mass(self, per_layer=False):
        """Per-mount S1 mass for the decode steps accumulated since the last
        reset_telemetry() (i.e. the current/most-recent attempt — see
        _attempt's reset-on-entry). Registered metric: sum over reply decode
        steps of softmax mass on the node's seats, mean over heads and ALL
        layers, reported as a share of total non-live mass (SINK + ARENA
        seats, i.e. everything before the live region — the physical seat
        range [0, n_sink + cur_mount_n)).

        Denominator note: "total non-live mass" is ALL mass on seats
        [0, n_sink+cur_mount_n) — sink included — not just the sum over
        mounted nodes' own seats. The two coincide only when sink mass is
        zero; sink is a permanent resident and typically draws some mass,
        so it is counted in the denominator but (having no graft_idx) never
        appears as a numerator term. Returns {graft_idx: share}.
        per_layer=True additionally returns a {layer_idx: {graft_idx:
        raw_mass}} diagnostics dict as a second return value — never
        consumed by the headline scalar above."""
        ranges = self._mount_seat_ranges()
        non_live_end = self.n_sink + self.cur_mount_n
        layers = self.m.layers
        n_layers = len(layers)
        per_layer_diag = {}
        raw = {i: 0.0 for i in ranges}
        total_raw = 0.0
        for li, L in enumerate(layers):
            acc = getattr(L.self_attn, "_telemetry_mass", None)
            layer_raw = {}
            if acc is not None:
                e_all = min(non_live_end, acc.shape[0])
                total_raw += float(acc[:e_all].sum()) if e_all > 0 else 0.0
                for i, (s, e) in ranges.items():
                    e = min(e, acc.shape[0])
                    m = float(acc[s:e].sum()) if e > s else 0.0
                    layer_raw[i] = m
                    raw[i] += m
            per_layer_diag[li] = layer_raw
        # mean over ALL layers (registered metric — no per-layer gating)
        if n_layers:
            for i in raw:
                raw[i] /= n_layers
            total_raw /= n_layers
        if total_raw > 0:
            shares = {i: v / total_raw for i, v in raw.items()}
        else:
            shares = {i: 0.0 for i in raw}
        if per_layer:
            return shares, per_layer_diag
        return shares

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
            result = self.deposit(turn_text) if deposit else None
            self._s4_turn = int(getattr(self, "_s4_turn", 0)) + 1
            return result
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
        # Scripted/observed turns do not route and therefore touch no S4
        # node counter, but they still advance the conversation ordinal used
        # by last_grounded_turn on the next accepted retrieval attempt.
        self._s4_turn = int(getattr(self, "_s4_turn", 0)) + 1

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
        sentence starters (for answers; sources keep everything).

        SC1.1: the same switch-gated glyph projection ``_rare_tokens`` takes.
        ``normalize_glyphs`` deliberately does NOT casefold — this channel is
        case-bearing and casefolding its input would empty it."""
        out = set()
        text = ArenaCache._norm_text(text)
        for sent in re.split(r"[.!?\n]+", text):
            ws = sent.split()
            for j, w in enumerate(ws):
                if skip_sentence_initial and j == 0:
                    continue
                if re.match(r"^[A-Z][\w\-]+$", w.rstrip(".,;:")):
                    out.add(w.lower().rstrip(".,;:"))
        return out

    def _grounding_verdict(self, ans, mount_idxs, question, *, normalized):
        """Grounding v3's pooled coverage, with the glyph projection EXPLICIT.

        ``normalized=False`` is the legacy path byte-for-byte: no
        ``normalize_glyphs`` call anywhere on it.  ``normalized=True`` runs
        the identical branches over glyph-projected text (SC1.1).  The two
        modes are the SAME code, so they cannot drift, and
        ``_grounding_receipt`` evaluates both to fill
        ``grounding_glyph_rescued``.

        The flag is carried by ``_glyph_norm`` rather than by a widened
        tokenizer signature: ``_rare_tokens``/``_caps_tokens`` are
        single-argument stub seams that fixtures across the suite replace
        with one-argument lambdas, and this method must not break them.
        """
        with self._glyph_norm(normalized):
            return self._grounding_verdict_inner(ans, mount_idxs, question)

    def _grounding_verdict_inner(self, ans, mount_idxs, question):
        """The v3 branches verbatim, run under whatever projection is in
        force.  Split out only so ``_glyph_norm`` wraps every token call --
        including the ones inside a stubbed ``_rare_tokens``."""
        norm = self._norm_text
        ans = norm(ans)
        question = norm(question)
        a = ans.lower()
        if any(h in a for h in self.HEDGES):
            return False, set()
        # identifier-aware: if the question names codes and NO mounted
        # source contains any of them, the right document is not mounted —
        # whatever the answer says, it cannot be about the asked entity
        # ("right family, wrong sibling" is grounded-but-wrong otherwise)
        qrare = self._rare_tokens(question)
        per_mount_content = {}
        if qrare:
            mounted = set()
            for i in mount_idxs:
                rare = self._rare_tokens(self.grafts[i]["text"])
                mounted |= rare
            if not (qrare & mounted):
                return False, set()
        content = ((self._rare_tokens(ans) | self._caps_tokens(ans))
                   - self._rare_tokens(question)
                   - self._caps_tokens(question, skip_sentence_initial=False))
        have = set()
        words = set()
        per_mount_words = {}
        for i in mount_idxs:
            t = norm(self.grafts[i]["text"])
            mount_content = self._rare_tokens(t) | self._caps_tokens(t, False)
            mount_words = {w.lower().rstrip(".,:;") for w in t.split()}
            per_mount_content[i] = mount_content
            per_mount_words[i] = mount_words
            have |= mount_content
            words |= mount_words
        if not content:
            # no identifier-shaped tokens — fall back to substantive words:
            # a correct prose answer ("The header parser is crashing.") has
            # its payload words IN the mounted sources; a deflection ("same
            # place as last time") and an echo of an unmounted turn do not
            qw = {w.lower().rstrip(".,:;?") for w in question.split()}
            subst = ({w.lower().rstrip(".,:;") for w in ans.split()
                      if len(w.rstrip(".,:;")) >= 4} - qw - self.SCAFFOLD)
            grounded = bool(subst) and bool(subst & words)
            if not grounded:
                return False, set()
            contributors = {i for i in mount_idxs
                            if subst & per_mount_words[i]}
            return True, contributors
        grounded = content <= have
        if not grounded:
            return False, set()
        contributors = {i for i in mount_idxs
                        if content & per_mount_content[i]}
        return True, contributors

    def _grounding_receipt(self, ans, mount_idxs, question, info):
        """Stamp SC1.1's two grounding-receipt fields onto ``info``.

        Kept SEPARATE from ``_grounding_attribution`` on purpose: that method
        is a documented seam that fixtures across the suite replace with a
        three-argument stub, and widening its signature would break every one
        of them for a receipt they do not exercise.  This runs alongside the
        verdict instead, over the same already-generated text -- pure set
        arithmetic, no forward, no routing, no state mutation.

        ``grounding_normalized`` says whether the glyph projection was in
        force.  ``grounding_glyph_rescued`` is True only when the normalized
        verdict is True AND the legacy verdict would have been False: the
        turn served BECAUSE the projection landed.  Both verdicts are
        computed with an explicit flag, never by reading the env twice, so
        the counterfactual is real while the switch is ON.
        """
        if info is None:
            return
        normalized = bool(self._lsr_fixes_enabled())
        rescued = False
        if normalized:
            new_verdict, _nc = self._grounding_verdict(
                ans, mount_idxs, question, normalized=True)
            if new_verdict:
                legacy, _lc = self._grounding_verdict(
                    ans, mount_idxs, question, normalized=False)
                rescued = not legacy
        info["grounding_normalized"] = bool(normalized)
        info["grounding_glyph_rescued"] = bool(rescued)

    def _grounding_attribution(self, ans, mount_idxs, question):
        """Return ``(pooled_grounded, contributing_mounts)``.

        This is a pure split of grounding v3's existing pooled coverage
        calculation.  ``pooled_grounded`` deliberately follows the former
        ``_grounded`` branches and set operations exactly; the additional
        per-mount sets only identify which mounted sources supplied at least
        one token to the coverage set after that pooled verdict succeeds.
        No routing, model forward, or token machinery is introduced here.

        SC1.1: the verdict is the NORMALIZED one when the LSR fixes switch is
        ON and the legacy one when it is OFF — the switch is the only thing
        that decides.  The receipt that names WHICH ran, and whether the
        projection is what let the turn serve, is stamped by the separate
        ``_grounding_receipt`` (this signature stays three-argument so the
        suite's grounding stubs keep working).
        """
        return self._grounding_verdict(
            ans, mount_idxs, question,
            normalized=bool(self._lsr_fixes_enabled()))

    def _grounded(self, ans, mount_idxs, question):
        """Compatibility wrapper: pooled verdict is unchanged."""
        return self._grounding_attribution(ans, mount_idxs, question)[0]

    @staticmethod
    def _s4_counter_defaults():
        return {"n_routed": 0, "n_mounted": 0, "n_grounded": 0,
                "last_grounded_turn": None}

    def _s4_ledger(self, idx):
        """Return one node's S4 sub-ledger without touching sibling arms."""
        g = self.grafts[int(idx)]
        metadata = g.setdefault("metadata", {})
        importance = metadata.setdefault("importance", {})
        s4 = importance.get("s4")
        if not isinstance(s4, dict):
            s4 = {}
            importance["s4"] = s4
        for key, value in self._s4_counter_defaults().items():
            s4.setdefault(key, value)
        return s4

    def _next_s4_turn(self):
        """One-based conversation-turn ordinal for the next ``step``.

        ``feed`` advances scripted/observed turns; accepted ``step`` commits
        advance retrieval turns.  Repository load establishes the restart-
        stable floor from persisted turn/recall nodes, keeping this hot-path
        lookup O(1).
        """
        return int(getattr(self, "_s4_turn", 0)) + 1

    def _commit_s4_attempt(self, routed, mounted, grounded_mounts,
                           turn=None):
        """Commit the one attempt whose state/answer survives ``step``.

        Routing is a turn-level funnel, so every unique node returned in the
        routed candidate ranking counts once.  Mount and grounding events are
        restricted to the accepted mount set; failed shuttling attempts call
        neither this method nor the repository persistence callback.
        """
        routed = {int(i) for i in routed}
        mounted = {int(i) for i in mounted}
        grounded_mounts = mounted & {int(i) for i in grounded_mounts}
        if turn is None:
            turn = int(getattr(self, "_s4_turn", 0)) + 1
        turn = int(turn)
        self._s4_turn = max(int(getattr(self, "_s4_turn", 0)), turn)
        changed = routed | mounted | grounded_mounts
        for i in sorted(changed):
            s4 = self._s4_ledger(i)
            if i in routed:
                s4["n_routed"] = int(s4.get("n_routed", 0)) + 1
            if i in mounted:
                s4["n_mounted"] = int(s4.get("n_mounted", 0)) + 1
            if i in grounded_mounts:
                s4["n_grounded"] = int(s4.get("n_grounded", 0)) + 1
                s4["last_grounded_turn"] = turn
        callback = getattr(self, "s4_metadata_callback", None)
        if changed and callback is not None:
            callback(tuple(sorted(changed)))

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

    # ------------------------------------------------------------------
    # GRM-LSR-P2C Part 2 — fit-time descent for legacy oversized nodes
    #
    # P2A marks a plan member ``fit_unseatable`` when its own ``ntok`` exceeds
    # the budget: no fit and no plan-shuttle can seat it.  Rather than degrade
    # first, the turn SPLITS the node and DESCENDS: the unseatable node is
    # chunked on the fly (the librarian's chunker when a repository is
    # attached, else the arena's own line/sentence fallback), and the plan
    # head becomes the identifier-bearing CHILD SET.  If the children co-fit,
    # seat them; if not, SHUTTLE across the chunks in document order.
    #
    # PERSISTED vs EPHEMERAL is stated in the receipt, never guessed:
    # ``fit_split_ephemeral`` is False when the split was written through the
    # librarian's ``cull_graft`` (deposit/mutation allowed on this path) and
    # True when the split lived only for this turn.
    # ------------------------------------------------------------------

    #: Set by GraftRepository so the arena can reach the librarian's split.
    #: ``None`` (the default for a bare ArenaCache) forces the ephemeral path.
    graft_splitter = None

    def eb1_begin_turn(self):
        """GRM-EB1: open a turn under the arena's frame. Returns the recency
        nomination (a possibly-empty list of graft indices).

        THE ONE IMPLEMENTATION of the frame's turn-open, shared by
        ``ArenaCache.step()`` and the e2e driver's ``_probe_ladder_chat``.
        Both are PRODUCTION SERVING PATHS, and before EB1 only ``step()``
        carried the ephemeral logic — which meant the driver's probe path,
        run against an ephemeral arena, served a THIRD frame: no live window
        (because nothing feeds one) and no recency mounts either (because it
        never nominated any).  Measured on the G2 sup battery, that third
        frame regressed three probes that the persistent frame served
        correctly.  Sharing the code is what makes "the spec frame is the
        default on every serving path" true rather than merely intended.

        Under the spec frame this clears the live cache and nominates the last
        ``recency_mounts`` turn grafts.  Under the escape it does nothing and
        returns an empty nomination, so the persistent path is byte-identical
        to its pre-EB1 self.
        """
        live_inherited = [
            {"graft_id": (None if g is None else int(g)), "ntok": int(n)}
            for g, n in self.live_segs]
        rec = []
        # Objects created through the real constructor always own this field.
        # ``False`` for legacy ``__new__``-only fixtures preserves their
        # pre-EB1 contract without weakening the production default — the same
        # rule ``step()`` applies to ``decisive_admission``.
        if getattr(self, "ephemeral", False):
            # clear the boat: fresh cache every turn, recency as mounts
            self.caches, self.pos, self.live_segs = None, 0, []
            self.cur_mounts, self.cur_mount_n = [], 0
            turns = [i for i, g in enumerate(self.grafts)
                     if not g.get("retired")
                     and g.get("kind", "turn") in ("turn", "recall")]
            recency_mounts = int(getattr(self, "recency_mounts", 0) or 0)
            rec = turns[-recency_mounts:] if recency_mounts else []
        self._eb1_recency_mounted_ids = [int(i) for i in rec]
        self._eb1_recency_seats_nominated = sum(
            int(self.grafts[i]["ntok"]) for i in rec)
        self._eb1_recency_seats_charged = None
        self._eb1_recency_charge_waived_reason = None
        self._eb1_live_segments_inherited = live_inherited
        # Captured AFTER the clear: the live window this turn actually starts
        # from.  THE SPEC IS THIS FIELD BEING EMPTY — no prior turn's tokens
        # in the model's context.  It is measured, not asserted.
        self._eb1_live_segments_carried_into_turn = [
            {"graft_id": (None if g is None else int(g)), "ntok": int(n)}
            for g, n in self.live_segs]
        return rec

    def eb1_charge_recency(self, rec, qrare):
        """Record what recency actually cost this turn, and return the budget.

        ``qrare`` non-empty means an IDENTIFIER query, and the point-lookup
        rule already excludes recency from the mount set for those; charging
        their seats too would be a double penalty, so the budget is not
        reduced.  This records WHICH of the two happened, per turn, so the
        recency cost table is read off receipts instead of re-derived.
        """
        rec_budget = 0 if qrare else sum(
            int(self.grafts[i]["ntok"]) for i in rec)
        self._eb1_recency_seats_charged = int(rec_budget)
        self._eb1_recency_charge_waived_reason = (
            "identifier_query_point_lookup" if (qrare and rec) else None)
        return int(self.width) - int(rec_budget)

    def _eb1_frame_info(self):
        """GRM-EB1: the per-turn frame receipt, on EVERY served turn.

        Four registered fields plus the two the escape needs to be auditable:

        ``frame_ephemeral``          the frame this turn actually ran under
        ``frame_escape_active``      whether GRM_PERSISTENT_BOAT selected it
        ``recency_mounted_ids``      which grafts recency nominated
        ``recency_seats``            what those grafts COST in arena seats
        ``live_segments_after_turn`` the live cache the turn leaves behind

        Plus the two that make the frame auditable turn by turn:

        ``live_segments_inherited``          what the PREVIOUS turn left
        ``live_segments_carried_into_turn``  what this turn actually STARTED
                                             from, captured after the clear

        The spec — "the chat log is not kept in memory context" — IS
        ``live_segments_carried_into_turn`` being empty.  Under the spec frame
        ``live_segments_inherited`` may be non-empty and every one of those
        segments was DISCARDED by the clear; keeping both fields is what lets
        a reader see the discard happen instead of taking it on trust.

        ``recency_seats`` reports both the nominated cost and the cost the fit
        stage actually charged, because the point-lookup rule waives the
        charge for identifier queries.  A reader can therefore see the recency
        cost WITHOUT re-deriving the branch.
        """
        info = frame_receipt(bool(getattr(self, "ephemeral", False)))
        info["recency_mounted_ids"] = list(
            getattr(self, "_eb1_recency_mounted_ids", ()) or ())
        charged = getattr(self, "_eb1_recency_seats_charged", None)
        info["recency_seats"] = {
            "arena_width": int(self.width),
            "nominated_ids": list(
                getattr(self, "_eb1_recency_mounted_ids", ()) or ()),
            "nominated_ntok": int(
                getattr(self, "_eb1_recency_seats_nominated", 0) or 0),
            "charged_ntok": (None if charged is None else int(charged)),
            "budget_after_charge": (
                None if charged is None else int(self.width) - int(charged)),
            "charge_waived_reason": getattr(
                self, "_eb1_recency_charge_waived_reason", None),
        }
        info["live_segments_inherited"] = list(
            getattr(self, "_eb1_live_segments_inherited", ()) or ())
        info["live_segments_carried_into_turn"] = list(
            getattr(self, "_eb1_live_segments_carried_into_turn", ()) or ())
        info["live_segments_after_turn"] = [
            {"graft_id": (None if g is None else int(g)), "ntok": int(n)}
            for g, n in self.live_segs]
        return info

    @staticmethod
    def _lsr_fixes_enabled():
        """LSR-P2C: the P2A+P2C fix switch, ``GRM_LSR_FIXES``, default ON.

        The G2 Arm-0 reproduction arm turns the fixes OFF so the replay can
        be shown lived-equivalent BEFORE Arm 1 counts as evidence.  It fails
        CLOSED to ON for an unknown token, the same direction the A-DEC
        switch fails: an operator who mistypes the value keeps the fixes.
        """
        value = os.environ.get("GRM_LSR_FIXES", "").strip().casefold()
        return value not in ("0", "false", "no", "off")

    def _split_source_text_chunks(self, text, budget):
        """Arena-side fallback chunker: lines, then sentences, then words.

        Used only when no repository/librarian chunker is reachable.  It obeys
        the same postcondition the librarian's does: NO chunk exceeds
        ``budget`` encoded tokens.
        """
        budget = int(budget)
        if budget <= 0 or not callable(getattr(self, "encode", None)):
            # FAIL CLOSED: no tokenizer on this arena means no honest way to
            # measure a chunk against the budget. Returning the text whole
            # signals "not splittable" and the caller falls back to P2A's
            # explicit degrade, which is still honest.
            return [str(text)]
        units = [u.strip() for u in re.split(r"\n+", str(text)) if u.strip()]
        if not units:
            units = [str(text)]
        atoms = []
        for unit in units:
            if len(self.encode(unit)) <= budget:
                atoms.append(unit)
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", unit):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(self.encode(sentence)) <= budget:
                    atoms.append(sentence)
                    continue
                cur = []
                for word in sentence.split():
                    trial = " ".join(cur + [word])
                    if cur and len(self.encode(trial)) > budget:
                        atoms.append(" ".join(cur))
                        cur = [word]
                    else:
                        cur.append(word)
                if cur:
                    atoms.append(" ".join(cur))
        # Greedy re-pack: as coarse as the budget allows (fewer chunk trips).
        out, cur = [], ""
        for atom in atoms:
            trial = f"{cur} {atom}".strip() if cur else atom
            if cur and len(self.encode(trial)) > budget:
                out.append(cur)
                cur = atom
            else:
                cur = trial
        if cur:
            out.append(cur)
        return out or [str(text)]

    def _split_unseatable(self, idx, budget):
        """Split one unseatable node into width-fitting children.

        Returns ``(children, ephemeral)``.  ``children`` is empty when the
        node could not be split at all (a single token wider than the whole
        arena is not a thing this can fix, and saying so is the honest
        answer — the caller then falls back to P2A's explicit degrade).
        """
        idx = int(idx)
        budget = int(budget)
        # IDEMPOTENT: an index node this machinery already split keeps its full
        # ``ntok`` (its text is still the whole original), so a naive size test
        # would split it again every turn and leave rival child families in the
        # routing surface.  The split is a ONE-TIME repair.
        node = self.grafts[idx]
        if (node.get("metadata") or {}).get("width_guard_parent") or (
                node.get("kind") == "era" and node.get("sources")):
            existing = [int(v) for v in (node.get("sources") or ())]
            return existing, bool(node.get("ephemeral_split", False))
        splitter = getattr(self, "graft_splitter", None)
        if callable(splitter):
            # PERSISTED: the librarian owns the split, so children inherit
            # provenance, tags, lineage and supersession membership through
            # the code path that already owns those transfers, and the node
            # is repaired once rather than at every fit.
            try:
                out = splitter(idx, budget)
            except Exception:  # noqa: BLE001 - a failed persist must not
                out = None    # abort the turn; fall through to ephemeral.
            if out:
                children = [int(v) for v in out]
                if children:
                    return children, False
        # EPHEMERAL: no repository on this path (a bare arena, a fork, a
        # read-only replay). The children live for this turn only; the
        # receipt says so via fit_split_ephemeral=True.
        parent = self.grafts[idx]
        text = str(parent.get("text", "") or "")
        chunks = self._split_source_text_chunks(text, budget)
        if len(chunks) <= 1:
            return [], True
        children = []
        for chunk in chunks:
            child_idx = self.deposit(chunk)
            child = self.grafts[child_idx]
            child["kind"] = parent.get("kind", "turn")
            child["sources"] = [idx]
            child["rare"] = self._rare_tokens(chunk)
            child["tags"] = list(parent.get("tags", ()) or ())
            child["ephemeral_split_of"] = idx
            meta = dict(parent.get("metadata") or {})
            meta.update({
                "kind": child["kind"],
                "active": True,
                "culled_from": idx,
                "width_guard": True,
                "width_guard_budget": int(budget),
                "width_guard_child": True,
                "ephemeral": True,
            })
            child["metadata"] = meta
            children.append(int(child_idx))
        if children:
            # The parent becomes the INDEX node for this turn: routable,
            # never a reader, and _descent_expand("era") descends to the
            # children the identifier picks out.
            parent["kind"] = "era"
            parent["sources"] = list(children)
            parent["ephemeral_split"] = True
            rare = set(parent.get("rare") or self._rare_tokens(text))
            for child_idx in children:
                rare |= set(self.grafts[child_idx]["rare"])
            parent["rare"] = rare
            self._bump_cuda_gqa_epoch()
        return children, True

    # ------------------------------------------------------------------
    # GRM-RT1 — a split child must not outrank the fact node it competes with
    #
    # The width guard leaves THREE first-class candidates in the routing
    # surface where there was one node: the index parent (which
    # ``_guard_deposit_width`` hands the UNION rare surface AND
    # ``child_cents``, so ``_cent_score`` scores it as the max over the
    # family) and every child on its own.  MEASURED (RS1 A0
    # fresh_fact_controls, artifacts/grm_rt1/grm_rt1_diagnosis.json): on
    # ``sup_solace_fresh`` the lexical channel is a FOUR-WAY TIE at
    # ``lex_bonus = 1.000`` across grafts 2, 4, 3 and 1 — the competitor's own
    # text says "the Praxis dock and Solace key references are index context",
    # so both query content words hit every family member as well as the fact
    # node.  Ordering among the tied four is then decided entirely by the
    # latent channel, and one 159-token topically-hot competitor takes ranks
    # 1, 2 and 3 while the 31-token identifier-bound fact node is pushed to
    # rank 4 (``admission_route_margin_1_2 = 0.0`` is the fingerprint: rank 1
    # and rank 2 are exactly tied because the parent's score IS its child's).
    #
    # This method reports MEMBERSHIP only.  The rule itself lives in
    # ``grm_admission.demote_non_binding_split_members`` — structural, no
    # threshold, and applied at admission rather than at readout.
    # ------------------------------------------------------------------

    def _split_family_members(self, candidates):
        """Which of ``candidates`` are width-guard split parents or children.

        Reads ONLY flags the existing split writers already set — RT1
        introduces no new metadata:

        * ``metadata['width_guard_child']`` — a persisted split child
          (``graft_repository._guard_deposit_width``) or an ephemeral one
          (``_split_unseatable``).
        * ``metadata['width_guard_parent']`` — the persisted index parent.
        * ``ephemeral_split_of`` / ``ephemeral_split`` — the ephemeral pair,
          for a bare arena with no repository attached.

        A node the guard never touched is not a member and the rule cannot
        move it.
        """
        out = set()
        for index in candidates:
            index = int(index)
            try:
                node = self.grafts[index]
            except (IndexError, TypeError, KeyError):
                continue
            meta = node.get("metadata") or {}
            if (meta.get("width_guard_child")
                    or meta.get("width_guard_parent")
                    or node.get("ephemeral_split_of") is not None
                    or node.get("ephemeral_split")):
                out.add(index)
        return out

    def _identifier_bearing_children(self, idx, user_text, qrare, fallback,
                                     *, with_binding_flag=False):
        """Which chunks of a split node the probe's identifier actually binds.

        LSR-P2C.  ``_descent_source_children`` filters by
        ``qrare & child["rare"]`` — the CODE/NUMBER channel — and that channel
        is EMPTY for the whole ADMISSION-PRUNE probe class: "What is the
        current Meridian docket value?" contains no code-shaped token, so
        ``_rare_tokens`` returns ``set()``, the filter cannot discriminate,
        and the fallback returns every chunk in DOCUMENT ORDER — filler first.
        MEASURED (LSR-P2C G2 Arm 1): the 86-token filler prefix was seated,
        it grounded, and the turn ended before the chunk holding
        ``Delta-4-Drift`` was ever read.

        The LEXICAL identifier channel is the one that binds these probes, and
        it is not a new rule: ``grm_admission.is_identifier_binding`` is the
        FROZEN ADM1 predicate A-DEC already uses to decide which repository
        nodes the identifier binds.  Applying the same predicate one level
        down — to a split node's children — introduces no threshold and no new
        law; it asks the existing question of the new candidates.

        Order is preserved (document order among the binding chunks).  When
        the predicate binds nothing, the rare-channel descent stands, then the
        full chunk set.

        ``with_binding_flag`` also returns whether any chunk actually BOUND
        the identifier.  The caller needs that to decide rank: chunks that
        bind may stand where their parent stood, but chunks that bind NOTHING
        must never outrank a plan member that does.
        """
        children = [int(v) for v in fallback]
        if not children:
            return ([], False) if with_binding_flag else []
        ordered, rare = ordered_identifier_tokens(self, user_text)
        if ordered:
            binding = [
                index for index in children
                if is_identifier_binding(
                    candidate_text=str(
                        self.grafts[index].get("text", "") or ""),
                    ordered_identifier_tokens=ordered,
                    rare_identifier_tokens=rare,
                )
            ]
            if binding:
                return (binding, True) if with_binding_flag else binding
        descended = [
            int(v) for v in self._descent_source_children(idx, qrare=qrare)]
        out = descended or children
        return (out, False) if with_binding_flag else out

    def _serve_live_binding(self, user_text, decision, rec, *, ngen,
                            deposit, defer_memory, stops):
        """Read the binding live source without an admission mount.

        Prior art: GRM contributors (2026), EB1 nomination/assembly and
        ArenaCache._attempt cache reuse. Materialize only binding recency
        nominees, once; persistent live segments stay in their existing cache.
        No prior art known to me for this exact FIX-4 composition.
        """
        ids = decision["served_from_node_ids"]
        rec_ids = sorted(set(ids) & set(rec))
        picks = sorted(set(self.cur_mounts) | set(rec_ids))
        self._eb1_recency_seats_charged = sum(
            int(self.grafts[i]["ntok"]) for i in rec_ids)
        self._eb1_recency_charge_waived_reason = None
        for layer in self.m.layers:
            layer.self_attn.live_shift = self.live_shift
        txt, info = self._attempt(user_text, picks, ngen, deposit, stops,
                                  defer_memory=defer_memory)
        info.update(decision)
        info["trip"] = 0
        self._grounding_receipt(txt, picks, user_text, info)
        return txt, info

    def step(self, user_text, ngen=48, deposit=True,
             stops=None, max_trips=0, defer_memory=False, demand_ngh=None,
             demand_early_abort=None):
        """One conversation turn through the arena. max_trips > 0 enables
        SHUTTLING: if the answer fails the grounding check, restore the
        pre-attempt cache (snapshot = a private outer-list copy + position —
        cache tensors are immutable), swap in the NEXT ranking slice, and
        retry. Failed attempts never enter the live cache. Returns
        (answer, info).

        ``demand_ngh`` (GRM-SC1, default: the ``GRM_DEMAND_NGH`` env switch,
        itself DEFAULT OFF) enables the Stage C demand loop: the answer is
        generated under the race-winning D-NGH observer, and a turn whose
        mounted mass falls below the CARRIED race threshold rolls back and
        takes ONE demand trip that re-routes on question + the model's own
        partial output.  With it off, this method's served outputs are
        byte-identical to P2C."""
        if defer_memory and deposit:
            raise ValueError("defer_memory requires deposit=False")
        if stops is None:
            stops = self.stop_sequences
        for L in self.m.layers:
            L.self_attn.live_shift = self.live_shift
        # GRM-EB1: open the turn under the frame (clear the boat and nominate
        # recency under the spec frame; a no-op under the escape).  Shared
        # with the driver's probe path so both serving paths run ONE frame.
        rec = self.eb1_begin_turn()
        # exclude turns already present (live window / recency mounts)
        live_idx = {g for g, _ in self.live_segs if g is not None} | set(rec)
        route_limit = max(1, (int(max_trips) + 1) * int(self.topk))
        admission_profile = None
        # Objects created through the real constructor always own this field.
        # ``False`` for legacy ``__new__``-only fixtures preserves their
        # pre-ADM2 test contract without weakening production default-on.
        if getattr(self, "decisive_admission", False):
            admission_profile = decisive_admission_profile(
                self, user_text, exclude=live_idx, route_limit=route_limit)
            ranking = list(admission_profile["ranking"])
        else:
            # Registered GRM_ADM_DECISIVE=0 escape: retain the legacy route
            # call, including its bounded ranking window, exactly.
            ranking = self.route(
                user_text, exclude=live_idx, limit=route_limit)
        # `route()` finalizes this before returning.  Attach the immutable
        # per-turn profile to whichever attempt becomes authoritative; this
        # makes step() receipts describe the route that actually chose the
        # mount ladder, not later decode/deposit work.
        route_receipt = getattr(self, "last_route_receipt", None)

        def attach_route_receipt(info):
            if route_receipt is not None:
                info["route_receipt"] = route_receipt
                info["route_backend"] = getattr(
                    self, "last_route_backend", "python")
            if admission_profile is not None:
                info.update(admission_info_fields(admission_profile))
            info.update(self._eb1_frame_info())
            return info

        s4_turn = self._next_s4_turn()

        # LSR-P2A Ruling 2: not-in-memory abstention.  A point lookup whose
        # identifier tokens bind NO node over the FULL eligible repository
        # must not be answered from the topical nearest neighbour; that is
        # the confabulation-under-retrieval-failure path Phase 0 measured
        # (H-LSR-3, ABSENT x4).  Structural trigger, no threshold.
        abstain = identifier_serving_decision(
            self, user_text, admission_profile, exclude=live_idx)
        if abstain is not None and abstain.get("served_from"):
            txt, info = self._serve_live_binding(
                user_text, abstain, rec, ngen=ngen, deposit=deposit,
                defer_memory=defer_memory, stops=stops)
            _, contributors = self._grounding_attribution(
                txt, self.cur_mounts, user_text)
            if defer_memory:
                info["_deferred_memory"]["importance_bookkeeping"] = {
                    "routed": [int(i) for i in ranking],
                    "mounted": list(self.cur_mounts),
                    "grounded_mounts": list(contributors), "turn": int(s4_turn)}
            else:
                self._commit_s4_attempt(
                    ranking, self.cur_mounts, contributors, turn=s4_turn)
            return txt, attach_route_receipt(info)
        if abstain is not None:
            txt, info = self._serve_abstention(
                user_text, abstain, deposit=deposit,
                defer_memory=defer_memory, stops=stops)
            info["trip"] = 0
            if defer_memory:
                info["_deferred_memory"]["importance_bookkeeping"] = {
                    "routed": [int(i) for i in ranking],
                    "mounted": [],
                    "grounded_mounts": [],
                    "turn": int(s4_turn),
                }
            else:
                self._commit_s4_attempt(ranking, (), (), turn=s4_turn)
            return txt, attach_route_receipt(info)

        # Some backends consume the mutable outer cache list in-place while
        # building its successor.  Snapshot that container; tensor entries
        # remain shared because they are immutable.
        snap = (list(self.caches) if isinstance(self.caches, list)
                else self.caches,
                self.pos, list(self.live_segs),
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
        if admission_profile is not None:
            branch = str(admission_profile["policy_branch"])
            selected = sorted(int(value) for value in admission_profile["rank_plan"])
            if selected:
                attempts.append((selected, False))
            if branch in (
                "exactly_one_identifier_decisive_rank1",
                "fit_margin_decisive_rank1",
                "declared_synthesis_identified_set",
            ):
                precise = selected
        elif ranking and qrare:
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
        # LSR-P2A Ruling 1: the plan is the priority, filler is the leftover.
        # `rank_plan` members are fitted FIRST, in PLAN ORDER, before any
        # expansion/topk/recency filler; filler keeps its EXPANSION-ORDERED
        # truncation (score-ordered truncation was tried and REFUTED
        # 2026-06-11, 6/8: max-over-child-cents inflates digest scores over
        # verbatim turns, so "relevance" order kept prose digests and dropped
        # the raw fact turns inside budget-bound expansions. A workable
        # version needs a leaf bias — board item, not a one-liner).
        # SUP-WO1 L2 sits after routing/descent expansion and before budget
        # fitting/injection, and it must NOT re-sort the plan: resolve first,
        # then re-derive plan membership from the resolved set.
        rank_plan = (
            [int(value) for value in admission_profile["rank_plan"]]
            if admission_profile is not None else []
        )
        fit_receipts = {}

        def fit_detail(picks, plan=None):
            picks = self._resolve_revision_mounts(picks)
            # GRM-EB1: one implementation of the recency charge and its
            # receipt, shared with the driver's probe path.
            budget = self.eb1_charge_recency(rec, qrare)
            receipt = plan_priority_fit(
                plan=(rank_plan if plan is None else plan),
                candidates=picks,
                ntok={int(i): int(self.grafts[i]["ntok"]) for i in picks},
                budget=budget,
            )
            key = tuple(sorted(int(v) for v in receipt["fit_seated"]))
            fit_receipts.setdefault(key, receipt)
            return receipt

        def fit(picks, plan=None):
            return list(fit_detail(picks, plan=plan)["fit_seated"])

        # LSR-P2C Part 2: FIT-TIME DESCENT for legacy oversized nodes.
        # A plan member whose own ntok exceeds the budget is UNSEATABLE: no
        # fit and no plan-shuttle can seat it, and P2A's honest degrade still
        # serves without the answer. Before degrading, SPLIT AND DESCEND —
        # chunk the node and make its identifier-bearing children the plan
        # head. This runs BEFORE the ladder is built so the descended head IS
        # the head every rung is derived from.
        split_parent = None
        split_children = []
        split_ephemeral = None
        descended_head = []
        chunk_trips = []
        chunk_owed = []
        split_head_binds = False
        if rank_plan and self._lsr_fixes_enabled():
            turn_budget = mountable_budget(
                self,
                recency_reserve=(
                    0 if qrare else sum(self.grafts[i]["ntok"] for i in rec)),
            )
            unseatable_now = [
                int(v) for v in rank_plan
                if int(self.grafts[int(v)]["ntok"]) > turn_budget
            ]
            # One split per turn: the plan head is the member the turn is
            # about, and splitting every unseatable member would multiply
            # chunk trips past the registered cap.
            for member in unseatable_now[:1]:
                children, ephemeral = self._split_unseatable(
                    member, turn_budget)
                if not children:
                    # Honest: nothing could be split. P2A's explicit degrade
                    # remains the outcome and the receipt still says so.
                    continue
                split_parent = int(member)
                split_children = [int(v) for v in children]
                split_ephemeral = bool(ephemeral)
                # The plan head becomes the IDENTIFIER-BEARING child set.
                descended_head, head_binds = (
                    self._identifier_bearing_children(
                        member, user_text, qrare, split_children,
                        with_binding_flag=True))
                # SUBSTITUTE IN PLACE, never re-rank. The children stand
                # exactly where their parent stood in the plan; promoting them
                # to the head would demote a SEATABLE, higher-ranked plan
                # member behind the chunks of an unseatable lower-ranked one.
                # (Measured, LSR-P2C G2 Arm 1 first run on
                # fresh_fact_controls: promoting the split competitor's
                # children ahead of the answer node turned the passing
                # sup_solace_fresh control into a refusal.)
                split_head_binds = bool(head_binds)
                was_head = (
                    int(rank_plan[0]) == int(member) and bool(head_binds))
                # A split whose chunks bind NOTHING does not enter the plan at
                # all when the identifier binds some OTHER plan member. Those
                # chunks are neither the answer nor useful filler, and a plan
                # member is owed a SHUTTLE TRIP by Ruling 1.2 — so keeping
                # them in the plan hands the turn to a competitor's filler
                # when the real answer node fails grounding.
                #   MEASURED (LSR-P2C G2 Arm 1, runs 3-5): the solace probe
                #   planned [2, 1, 0]; neither chunk of the competitor binds
                #   "solace key"; demoting them to [1, 0, 3, 4] still produced
                #   fit_shuttle_trips [[0], [3], [4]] and trip 4 served the
                #   competitor's Sable-0-Copper.
                other_binds = bool(
                    {int(v) for v in admission_profile[
                        "identified_candidates"]}
                    - {int(member)}
                ) if admission_profile is not None else False
                drop_chunks = bool(not head_binds and other_binds)
                substituted = []
                seen_plan = set()
                trailing = []
                for value in rank_plan:
                    if int(value) == int(member):
                        if drop_chunks:
                            continue
                        incoming = descended_head
                        sink = substituted if head_binds else trailing
                    else:
                        incoming, sink = [int(value)], substituted
                    for entry in incoming:
                        # DE-DUPLICATE: a previous turn's persisted child can
                        # already be a plan member in its own right, and
                        # substituting its parent would list it twice
                        # (measured: fit_planned = [4, 4] on
                        # sup_reserve_tundra_ledger, rank_plan [2, 4]).
                        if int(entry) not in seen_plan:
                            sink.append(int(entry))
                            seen_plan.add(int(entry))
                rank_plan = [*substituted, *trailing]
                head_picks = fit(rank_plan, plan=rank_plan)
                if head_picks and was_head:
                    # Only when the SPLIT MEMBER WAS THE PLAN HEAD does the
                    # descended set take over the attempt head; otherwise the
                    # existing ladder order stands and the children simply
                    # join the plan at their parent's rank.
                    attempts = [(sorted(head_picks), False),
                                *[a for a in attempts
                                  if sorted(a[0]) != sorted(head_picks)]]
                    if precise is not None:
                        precise = sorted(head_picks)
                elif not head_picks:
                    # Nothing co-fits at all: every chunk is served on its own
                    # trip below.
                    attempts = [a for a in attempts if a[0]]
                # Chunks the head rung could not co-seat are owed their own
                # trip, in DOCUMENT ORDER. They are APPENDED AFTER the ladder
                # build below (which truncates to max_trips + 1), exactly the
                # way P2A's plan-shuttle rungs are additive.
                # Chunk trips exist to SERIALIZE the answer across the chunks
                # of the node the probe is about.  A split whose chunks bind
                # NOTHING is not that node: giving its chunks their own trips
                # lets a competitor's filler ground the turn while the planned
                # answer node goes unread (measured: sup_solace_fresh served
                # the competitor's Sable-0-Copper from chunk trip 3).  Those
                # chunks stay in the plan, at the back, and take a seat only
                # if one is left over.
                chunk_owed = (
                    [int(v) for v in descended_head
                     if int(v) not in set(head_picks)
                     ][:chunk_trip_cap(descended_head)]
                    if head_binds else []
                )
                # The split APPENDED grafts (persisted children, or ephemeral
                # ones). `snap[5]` is the rollback watermark the shuttle uses
                # to `del self.grafts[snap[5]:]` between trips — re-baseline
                # it, or trip 1 would delete the very children this turn is
                # about to mount.
                snap = (*snap[:5], len(self.grafts))

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

        # LSR-P2C Part 2: CHUNK SHUTTLE. Chunks of the split node that the
        # head rung could not co-seat get their own trip, in DOCUMENT ORDER,
        # additive to max_trips and capped at chunk_trip_cap(chunks) — the
        # cap registered before the gates. The answer is composed from the
        # grounded trips; served_without_plan_head becomes true ONLY when
        # every chunk trip fails grounding (attach_fit below reads
        # seated_anywhere_now, which includes every chunk trip).
        for chunk in chunk_owed:
            trip_picks = fit([chunk], plan=[chunk])
            if not trip_picks:
                continue
            rung = (sorted(trip_picks), False)
            if rung not in attempts:
                attempts.append(rung)
                chunk_trips.append(list(rung[0]))

        # LSR-P2A Ruling 1.2: plan members that do not co-fit become SHUTTLE
        # trips — additive to max_trips, capped at len(rank_plan). The turn
        # shuttles because the FIT dropped a planned member, not because a
        # trip failed grounding: a wrong-but-grounded first trip must not end
        # the turn while a planned node is still owed a seat.
        shuttle_trips = []
        plan_head_receipt = {}
        if rank_plan:
            plan_head_receipt = fit_detail(
                self._descent_expand(
                    [int(v) for v in rank_plan], ("era",), qrare=qrare))
            seated_anywhere = {
                int(v) for picks, _clean in attempts for v in picks}
            owed = [
                int(v) for v in plan_head_receipt["fit_shuttle_pending"]
                if int(v) not in seated_anywhere
            ]
            for member in owed[:shuttle_trip_cap(rank_plan)]:
                trip_picks = fit(
                    self._descent_expand([member], ("era",), qrare=qrare),
                    plan=[member])
                if trip_picks and (trip_picks, False) not in attempts:
                    attempts.append((trip_picks, False))
                    shuttle_trips.append(list(trip_picks))

        def attach_fit(info, seated):
            """NEVER SILENT (Ruling 1.3): fit fields on every served turn.

            The receipt describes the PLAN's fate across the whole turn, not
            merely the served rung's own packing: the ladder can serve a rung
            that contains no plan member at all, and a receipt scoped to that
            rung would be silent about a planned node no rung ever seated —
            the exact silence this ruling forbids.
            """
            if admission_profile is None:
                # Legacy path (GRM_ADM_DECISIVE=0): byte-for-byte unchanged,
                # no fit receipt fields are added at all.
                return info
            seated_set = {int(v) for v in seated}
            rung = fit_receipts.get(tuple(sorted(seated_set))) or {}
            # UNSEATABLE is a property of the plan member and the arena
            # width, not of whichever rung happened to serve.
            unseatable = [
                int(v) for v in plan_head_receipt.get("fit_unseatable", ())]
            seated_anywhere_now = seated_set | {
                int(v) for trip in shuttle_trips for v in trip} | {
                int(v) for trip in chunk_trips for v in trip}
            receipt = {
                "fit_planned": list(rank_plan),
                "fit_seated": sorted(seated_set),
                "fit_dropped_planned": [
                    int(v) for v in rank_plan
                    if int(v) in set(unseatable)
                    and int(v) not in seated_anywhere_now
                ],
                "fit_dropped_filler": [
                    int(v) for v in rung.get("fit_dropped_filler", ())],
                "fit_unseatable": unseatable,
            }
            # LSR-P2C: after a split-and-descend the turn served WITH the plan
            # head whenever a grounded trip seated any chunk of it — the
            # answer is composed from the grounded trips, so
            # served_without_plan_head is true ONLY when every chunk trip
            # failed grounding.
            #
            # The chunk set counts as "the plan head" ONLY when the split
            # member WAS the head and its chunks BOUND the identifier.  A
            # demoted, non-binding chunk set is filler that happened to
            # ground; calling that "served with the plan head" would be
            # exactly the unlabeled substitution Ruling 1.4 forbids
            # (measured: sup_solace_fresh served the competitor's
            # Sable-0-Copper from chunk trip 3 while the planned answer node
            # went unseated, and the receipt claimed the head was served).
            head_served = bool(rank_plan) and int(rank_plan[0]) in seated_set
            if (split_parent is not None and not head_served
                    and split_head_binds):
                head_served = bool(seated_set & set(descended_head))
            info.update(fit_info_fields(
                receipt,
                shuttle=bool(shuttle_trips),
                shuttle_trips=shuttle_trips,
                # Ruling 1.4: UNSEATABLE is an explicit degrade, never an
                # unlabeled substitution.
                served_without_plan_head=bool(rank_plan and not head_served),
            ))
            # LSR-P2C receipt fields (all fit_-prefixed, so P2B persists them
            # via ROUTE_RECEIPT_INFO_PREFIXES with no edit to grm_three_pass).
            info.update(split_info_fields(
                split_parent=split_parent,
                split_children=split_children,
                split_ephemeral=split_ephemeral,
                descended_head=descended_head,
                chunk_trips=chunk_trips,
            ))
            return info

        # ---------------------------------------------------------------
        # GRM-SC1 Stage C: the DEMAND LOOP (GRM_DEMAND_NGH, DEFAULT OFF).
        #
        # A turn that is missing a needed memory NOTICES mid-generation and
        # fetches.  The detector is the race winner D-NGH, carried whole:
        # per-token full-layer mean mounted_mass, fire on the FIRST token
        # STRICTLY BELOW the registered threshold (fit on TWO calibration
        # turns — a thin envelope, restated in every receipt).
        #
        # WITH THE FLAG OFF this block is inert: `_serve` is `self._attempt`
        # by identity and nothing else below runs, so served outputs are
        # byte-identical to P2C.  A test pins that.
        demand_on = grm_demand.demand_enabled(demand_ngh)
        demand_state = {
            "rows": [],          # observer rows for the attempt just served
            "info": {},          # demand_* receipt fields
            "unsupported": None,
        }
        if demand_on:
            support = grm_demand.arena_support(self)
            if not support.get("demand_supported"):
                # LOUD, not silent: a turn that cannot carry the detector
                # must not look like a turn that found its memory.
                demand_on = False
                demand_state["unsupported"] = str(
                    support.get("demand_unsupported_reason"))
                demand_state["info"] = grm_demand.unsupported_info(
                    demand_state["unsupported"])
        demand_threshold = (
            grm_demand.registered_threshold() if demand_on else None)
        # GRM-SC2: early abort is consulted ONLY when demand is on. With demand
        # off there is no observer, nothing can fire, and this block is inert
        # exactly as SC1 left it.
        early_abort_on = (
            grm_demand.early_abort_enabled(demand_early_abort)
            if demand_on else False)
        demand_state["early_abort"] = bool(early_abort_on)

        def _serve(*args, **kwargs):
            """One generation attempt, observed when the demand flag is on.

            GRM-SC2: when early abort is on, an attempt whose D-NGH mass falls
            below the line stops generating at that token and comes back as a
            SUSPENSION handle instead of a ``(txt, info)`` pair.  The handle is
            stashed on ``demand_state`` and the caller is told, via the handle
            itself, that there is no answer to ground yet.
            """
            started = time.perf_counter()
            if not demand_on:
                return self._attempt(*args, **kwargs)
            abort_now = bool(early_abort_on) and not demand_state.get(
                "suppress_early_abort")
            with grm_demand.DemandObserver(
                self, int(ngen), float(demand_threshold),
                early_abort=abort_now,
            ) as observer:
                out = self._attempt(*args, **kwargs)
            demand_state["rows"] = observer.finish()
            demand_state["wall_ms"] = (time.perf_counter() - started) * 1000.0
            if isinstance(out, dict) and out.get("grm_sc2_suspended_attempt"):
                # Recover the tokens the aborted attempt emitted. The observer
                # captured a greedy prediction id for EVERY position it closed,
                # including the fire position, so the token the abort unwound
                # before appending is row[abort_index]'s prediction. Rebuilding
                # `out` from the rows rather than from a decoded string is the
                # same reason `demand_prefix_text` does: decode->encode is not
                # a round trip on BPE, so the ids are the only exact carrier.
                rows = list(demand_state["rows"])
                index = int(out["abort_token_index"])
                out["out"] = [
                    int(row["prediction_token_id"]) for row in rows[:index + 1]]
                out["tokens_generated_before_abort"] = len(out["out"])
                demand_state["suspended"] = out
                # The outer trip loop wants a (txt, info) pair. There is no
                # text yet, and inventing one would be a lie the grounding
                # check would then act on. Hand back an EMPTY text carrying the
                # handle: `_demand_trip` consumes the handle, and every caller
                # between here and there is guarded on `_grm_sc2_suspended`.
                return "", {"_grm_sc2_suspended": out}
            return out

        def _resume_if_suspended(suspended, txt, info):
            """Finish a suspended attempt so the turn has its original answer.

            Returns ``(txt, info, resume_wall_ms)``.  With nothing suspended it
            is a pass-through and costs nothing, which is what keeps the
            early-abort-OFF path byte-identical to SC1.

            The arena state has ALREADY been restored to the aborted attempt's
            state by the caller (the same verbatim restore SC1 does when a trip
            fails to ground), so the cache the resume continues from is the one
            the abort left behind -- that is what makes the finished text the
            original text rather than a re-generation of it.
            """
            if suspended is None:
                return txt, info, None
            started = time.perf_counter()
            r_txt, r_info = self._attempt(
                user_text, list(suspended["picks"]), ngen, deposit, stops,
                defer_memory=defer_memory, suspended=suspended)
            resume_ms = (time.perf_counter() - started) * 1000.0
            merged = dict(r_info or {})
            # Keep the pre-resume receipt keys the turn already accumulated
            # (trip index, clean-room marker, grounding fields) -- the resumed
            # attempt only re-establishes the mount/resident/live accounting.
            for key, value in (info or {}).items():
                if key.startswith("_grm_sc2_"):
                    continue
                merged.setdefault(key, value)
            return r_txt, merged, resume_ms

        def _demand_trip(txt, info, mset, picks, contributors, trip,
                         rows=None):
            """The ONE registered demand trip, at the point of serving.

            Sequence (all four steps are the order's, in the order's order):

            1. D-NGH decides on the attempt about to be served.  Not fired ->
               record the decision and serve the original untouched.
            2. Fired -> roll back to the SAME pre-attempt snapshot the
               shuttle already uses (`snap`), then re-route on
               question + the model's own partial output up to the fire
               token, EXCLUDING everything already mounted or live.  Admit
               plan-first through the existing A-DEC path; P2C fit rules
               apply because this goes through `fit()`, the same fitter every
               rung uses.
            3. Serve the demand trip's answer IFF it grounds; otherwise
               restore the original attempt and serve that.  `demand_served`
               always names which one went out.
            4. NEVER LOOP.  A second fire on the demand trip is RECORDED
               (`demand_refired`) and NOT acted on — cap 1, registered.
            """
            if not demand_on:
                if demand_state["info"]:
                    info.update(demand_state["info"])
                return txt, info, mset, picks, contributors
            observed = list(
                demand_state["rows"] if rows is None else rows)
            # GRM-SC2: the suspended attempt this trip is standing in for, if
            # any. `info` carries it because `_serve` had nowhere else to put
            # it; it is popped here so it never reaches a receipt.
            suspended = (info or {}).pop("_grm_sc2_suspended", None)
            attempt_wall_ms = demand_state.get("wall_ms")
            decision = grm_demand.decide(observed, float(demand_threshold))
            if suspended is not None and not decision["demand_fired"]:
                raise grm_demand.DemandError(
                    "GRM-SC2: an attempt was suspended by the early abort but "
                    "the decision over its rows says it never fired; the abort "
                    "and the decision rule have diverged")
            if not decision["demand_fired"]:
                info.update(grm_demand.demand_info_fields(
                    supported=True, decision=decision, served="original",
                    threshold=float(demand_threshold)))
                return txt, info, mset, picks, contributors

            # --- 2. roll back and re-route -----------------------------
            original = (txt, info, mset, picks, contributors,
                        (self.caches, self.pos, list(self.live_segs),
                         self.cur_mounts, self.cur_mount_n,
                         list(self.grafts)))
            prefix = grm_demand.demand_prefix_text(
                self, observed, int(decision["demand_token_index"]))
            query = grm_demand.demand_query_text(user_text, prefix)
            (self.caches, self.pos, self.live_segs, self.cur_mounts,
             self.cur_mount_n) = (
                list(snap[0]) if isinstance(snap[0], list) else snap[0],
                snap[1], list(snap[2]), snap[3], snap[4])
            del self.grafts[snap[5]:]
            self._bump_cuda_gqa_epoch()
            # Exclude what the turn already had: the failed attempt's own
            # mounts and the live window. A demand trip that re-fetches the
            # nodes already read is not a fetch.
            demand_exclude = set(live_idx) | {int(i) for i in mset}
            demand_profile = None
            if getattr(self, "decisive_admission", False):
                demand_profile = decisive_admission_profile(
                    self, query, exclude=demand_exclude,
                    route_limit=route_limit)
                demand_ranking = [int(v) for v in demand_profile["ranking"]]
                demand_plan = [int(v) for v in demand_profile["rank_plan"]]
            else:
                demand_ranking = [
                    int(v) for v in (self.route(
                        query, exclude=demand_exclude,
                        limit=route_limit) or [])]
                demand_plan = demand_ranking[:self.topk]
            demand_picks = fit(
                sorted({int(v) for v in demand_plan}), plan=demand_plan)
            fields = grm_demand.demand_info_fields(
                supported=True, decision=decision, served="original",
                threshold=float(demand_threshold),
                query_text_sha256=grm_demand.demand_query_sha256(query),
                prefix_token_count=int(decision["demand_token_index"]),
                ranking=demand_ranking,
                fetched=[],
                trip_taken=False,
            )
            if not demand_picks:
                # Nothing new to fetch: restore the original attempt's state
                # verbatim and serve it. Honest, and the receipt says the
                # trip found nothing rather than pretending it never fired.
                (self.caches, self.pos, self.live_segs, self.cur_mounts,
                 self.cur_mount_n) = original[5][:5]
                self.grafts[:] = original[5][5]
                self._bump_cuda_gqa_epoch()
                txt, info, resume_ms = _resume_if_suspended(
                    suspended, txt, info)
                info.update(fields)
                info.update(grm_demand.demand_info_fields(
                    supported=True, decision=decision, served="original",
                    threshold=float(demand_threshold),
                    early_abort=bool(early_abort_on),
                    abort_token_index=(
                        None if suspended is None
                        else int(suspended["abort_token_index"])),
                    tokens_generated_before_abort=(
                        None if suspended is None
                        else int(suspended["tokens_generated_before_abort"])),
                    tokens_saved=(None if suspended is None else 0),
                    resumed_original=(None if suspended is None else True),
                    wall_ms_attempt=attempt_wall_ms,
                    wall_ms_resume=resume_ms,
                ))
                return txt, info, mset, picks, contributors

            demand_mset = sorted(self._resolve_revision_mounts(
                sorted(set(demand_picks))))
            trip_started = time.perf_counter()
            # The trip itself is NEVER early-aborted. Cap 1 is registered: a
            # second fire is RECORDED and not acted on, so aborting the trip
            # would throw away the only answer this turn has left to serve and
            # would leave the turn with nothing to put out. `demand_state`
            # carries the suppression because `_serve` reads it there; a plain
            # rebinding would need `nonlocal` and would be easy to leave set.
            demand_state["suppress_early_abort"] = True
            try:
                if defer_memory:
                    d_txt, d_info = _serve(
                        user_text, demand_mset, ngen, deposit, stops,
                        defer_memory=True)
                else:
                    d_txt, d_info = _serve(
                        user_text, demand_mset, ngen, deposit, stops)
            finally:
                demand_state["suppress_early_abort"] = False
            trip_wall_ms = (time.perf_counter() - trip_started) * 1000.0
            # 4. A refire on the demand trip is RECORDED, never acted on.
            refire = grm_demand.decide(
                list(demand_state["rows"]), float(demand_threshold))
            # SC1.1: the grounding receipt for the DEMAND TRIP's verdict —
            # this is the verdict SC1 measured as blocking recovery 2/2.
            grounding_fields = {}
            d_grounded, d_contributors = self._grounding_attribution(
                d_txt, demand_mset, user_text)
            self._grounding_receipt(
                d_txt, demand_mset, user_text, grounding_fields)
            fields = grm_demand.demand_info_fields(
                supported=True, decision=decision,
                served="demand_trip" if d_grounded else "original",
                threshold=float(demand_threshold),
                query_text_sha256=grm_demand.demand_query_sha256(query),
                prefix_token_count=int(decision["demand_token_index"]),
                ranking=demand_ranking,
                fetched=[int(v) for v in demand_mset],
                refired=bool(refire["demand_fired"]),
                refire_token_index=refire["demand_token_index"],
                trip_taken=True,
                trip_grounded=bool(d_grounded),
                early_abort=bool(early_abort_on),
                abort_token_index=(
                    None if suspended is None
                    else int(suspended["abort_token_index"])),
                tokens_generated_before_abort=(
                    None if suspended is None
                    else int(suspended["tokens_generated_before_abort"])),
                wall_ms_attempt=attempt_wall_ms,
                wall_ms_trip=trip_wall_ms,
            )
            # Both branches below apply ``fields``, so folding the grounding
            # receipt in here carries it whether the trip serves or is
            # rejected — the rejection is exactly the case SC1 needed named.
            fields.update(grounding_fields)
            if d_grounded:
                # The trip serves. The suspended attempt is ABANDONED, never
                # resumed: every token after the fire index is one this turn
                # never had to generate, and that is the whole saving.
                d_info = dict(d_info or {})
                d_info["trip"] = int(trip)
                d_info["demand_source_trip"] = int(trip)
                d_info.update(fields)
                if suspended is not None:
                    d_info.update(grm_demand.demand_info_fields(
                        supported=True, decision=decision,
                        served="demand_trip",
                        threshold=float(demand_threshold),
                        trip_taken=True, trip_grounded=True,
                        early_abort=True,
                        abort_token_index=int(suspended["abort_token_index"]),
                        tokens_generated_before_abort=int(
                            suspended["tokens_generated_before_abort"]),
                        resumed_original=False,
                        wall_ms_attempt=attempt_wall_ms,
                        wall_ms_trip=trip_wall_ms,
                    ))
                    d_info.update(grounding_fields)
                return (d_txt, d_info, demand_mset, list(demand_picks),
                        d_contributors)
            # 3. Ungrounded demand trip -> the ORIGINAL answer goes out, and
            # the arena state that goes with it is restored verbatim.
            (self.caches, self.pos, self.live_segs, self.cur_mounts,
             self.cur_mount_n) = original[5][:5]
            self.grafts[:] = original[5][5]
            self._bump_cuda_gqa_epoch()
            # GRM-SC2: under early abort the "original answer" does not exist
            # yet -- the attempt stopped at the fire token. Resume it here and
            # finish it, so what goes out is exactly what SC1 put out on this
            # pair. This is the fallback that makes the abort safe to default
            # on: a failed trip costs the same tokens it always did.
            txt, info, resume_ms = _resume_if_suspended(suspended, txt, info)
            info.update(fields)
            if suspended is not None:
                info.update(grm_demand.demand_info_fields(
                    supported=True, decision=decision, served="original",
                    threshold=float(demand_threshold),
                    trip_taken=True, trip_grounded=False,
                    early_abort=True,
                    abort_token_index=int(suspended["abort_token_index"]),
                    tokens_generated_before_abort=int(
                        suspended["tokens_generated_before_abort"]),
                    tokens_saved=0,
                    resumed_original=True,
                    wall_ms_attempt=attempt_wall_ms,
                    wall_ms_trip=trip_wall_ms,
                    wall_ms_resume=resume_ms,
                ))
                info.update(grounding_fields)
            return txt, info, mset, picks, contributors

        best = None
        for trip, (picks, clean) in enumerate(attempts):
            if not picks:
                # Ruling 1.2: an empty rung no longer ends the turn when a
                # plan is in play — the shuttle rungs behind it are exactly
                # the trips that carry the unseated plan members. The legacy
                # path keeps its `break` byte-for-byte.
                if admission_profile is None:
                    break
                continue
            if trip:        # roll back the failed attempt entirely
                (self.caches, self.pos, self.live_segs, self.cur_mounts,
                 self.cur_mount_n) = (
                    list(snap[0]) if isinstance(snap[0], list) else snap[0],
                    snap[1], list(snap[2]), snap[3], snap[4])
                had_appended_grafts = len(self.grafts) > snap[5]
                del self.grafts[snap[5]:]
                # The three-pass output path never deposits during an
                # attempt, so an empty rollback must not mutate the arena
                # epoch in pass 2.  The default path preserves the legacy
                # unconditional invalidation byte-for-byte.
                if not defer_memory or had_appended_grafts:
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
            # Recency nodes join only at final mount-set assembly, so resolve
            # once more here in case a recency graft and a routed graft are
            # two explicit revisions of the same M5 lineage.
            mset = sorted(self._resolve_revision_mounts(mset))
            if defer_memory:
                txt, info = _serve(
                    user_text, mset, ngen, deposit, stops,
                    defer_memory=True)
            else:
                txt, info = _serve(
                    user_text, mset, ngen, deposit, stops)
            info["trip"] = trip
            if clean:
                info["clean_room"] = True
            # GRM-SC2: a SUSPENDED attempt has no text to ground. Grounding it
            # would attribute the empty string, which cannot ground, and the
            # turn would fall through to the ungrounded-keep-first path with an
            # empty answer. Route it straight into the demand trip instead --
            # which is exactly what an un-aborted fired attempt would have
            # reached anyway, just without generating the rest of the wrong
            # answer first.
            if info.get("_grm_sc2_suspended") is not None:
                txt, info, mset, picks, contributors = _demand_trip(
                    txt, info, mset, picks, (), trip)
                if defer_memory:
                    info["_deferred_memory"]["importance_bookkeeping"] = {
                        "routed": [int(i) for i in ranking],
                        "mounted": [int(i) for i in mset],
                        "grounded_mounts": [int(i) for i in contributors],
                        "turn": int(s4_turn),
                    }
                else:
                    self._commit_s4_attempt(
                        ranking, mset, contributors, turn=s4_turn)
                return txt, attach_route_receipt(attach_fit(info, picks))
            grounded, contributors = self._grounding_attribution(
                txt, mset, user_text)
            self._grounding_receipt(txt, mset, user_text, info)
            if grounded:
                # SC1: the turn is about to serve. If D-NGH fired on THIS
                # attempt, roll back and take the one registered demand trip
                # before committing anything.
                txt, info, mset, picks, contributors = _demand_trip(
                    txt, info, mset, picks, contributors, trip)
                if defer_memory:
                    info["_deferred_memory"]["importance_bookkeeping"] = {
                        "routed": [int(i) for i in ranking],
                        "mounted": [int(i) for i in mset],
                        "grounded_mounts": [int(i) for i in contributors],
                        "turn": int(s4_turn),
                    }
                else:
                    self._commit_s4_attempt(
                        ranking, mset, contributors, turn=s4_turn)
                return txt, attach_route_receipt(attach_fit(info, picks))
            if best is None:
                best = (txt, info, (self.caches, self.pos, list(self.live_segs),
                                    self.cur_mounts, self.cur_mount_n,
                                    list(self.grafts)), tuple(mset),
                        list(picks), list(demand_state["rows"]))
        # nothing grounded — keep the FIRST attempt's answer and state
        if best is None:
            if defer_memory:
                txt, info = _serve(
                    user_text, [], ngen, deposit, stops,
                    defer_memory=True)
            else:
                txt, info = _serve(
                    user_text, [], ngen, deposit, stops)
            info["trip"] = 0
            info["no_mount_fit"] = True
            # SC1: a no-mount turn is the purest demand case there is — the
            # detector sees zero mounted mass by construction. It still gets
            # exactly one trip, and the receipt still records the decision.
            # Whatever the trip ends up serving, the bookkeeping below names
            # the mounts THAT answer was actually read from, never the empty
            # set the pre-trip attempt had.
            txt, info, d_mset, d_picks, d_contrib = _demand_trip(
                txt, info, [], [], (), 0)
            if defer_memory:
                info["_deferred_memory"]["importance_bookkeeping"] = {
                    "routed": [int(i) for i in ranking],
                    "mounted": [int(i) for i in d_mset],
                    "grounded_mounts": [int(i) for i in d_contrib],
                    "turn": int(s4_turn),
                }
            else:
                self._commit_s4_attempt(
                    ranking, d_mset, d_contrib, turn=s4_turn)
            return txt, attach_route_receipt(attach_fit(info, d_picks))
        txt, info, st, accepted_mounts, accepted_picks, best_rows = best
        (self.caches, self.pos, self.live_segs,
         self.cur_mounts, self.cur_mount_n) = st[0], st[1], st[2], st[3], st[4]
        self.grafts[:] = st[5]
        # SC1: the ungrounded-keep-first answer is still an answer about to be
        # SERVED, so the demand loop applies to it too — and its detector rows
        # are the ones captured on THAT attempt (`best_rows`), not whichever
        # later rung happened to run last.
        txt, info, accepted_mounts, accepted_picks, demand_contributors = (
            _demand_trip(
                txt, info, list(accepted_mounts), list(accepted_picks), (), 0,
                rows=best_rows))
        if not defer_memory:
            self._bump_cuda_gqa_epoch()
            self._commit_s4_attempt(
                ranking, accepted_mounts, demand_contributors, turn=s4_turn)
        else:
            info["_deferred_memory"]["importance_bookkeeping"] = {
                "routed": [int(i) for i in ranking],
                "mounted": [int(i) for i in accepted_mounts],
                "grounded_mounts": [int(i) for i in demand_contributors],
                "turn": int(s4_turn),
            }
        return txt, attach_route_receipt(attach_fit(info, accepted_picks))

    def _serve_abstention(self, user_text, abstain, *, deposit,
                          defer_memory, stops):
        """Serve the fixed not-in-memory string without a model forward.

        LSR-P2A Ruling 2.  The abstention is a CONSTANT (core.grm_admission
        .ABSTENTION_TEMPLATE), not a generation, so nothing can be
        confabulated into it. The user turn is still a lived turn and still
        deposits; the abstention output is pinned ``kind="recall"`` so it is
        excluded from the routing candidate base and can never later be
        routed as if it were a stored fact (Ruling 2.2).
        """
        txt = str(abstain["abstain_text"])
        info = {
            "mounts": [],
            # No forward runs, so the cache may still be unbuilt (None).
            "resident": (0 if self.caches is None else self._cache_len()),
            "evicted": 0,
            "live_tokens": sum(n for _, n in self.live_segs),
            "abstained": True,
            "abstain_reason": str(abstain["abstain_reason"]),
            "abstain_identifier_tokens": [
                str(t) for t in abstain["abstain_identifier_tokens"]],
            "no_mount_fit": False,
        }
        if defer_memory:
            info["_deferred_memory"] = {
                "turn_text": self._format_step_turn(user_text, txt),
                "user_text": user_text,
                "seg_cache_ntok": 0,
                "picks": [],
                "deposited": False,
                "importance_committed": False,
                "route_key_prepared": False,
                "route_key_token": 0,
                "abstained": True,
            }
        elif deposit:
            gidx = self.deposit(self._format_step_turn(user_text, txt))
            self.grafts[gidx]["kind"] = "recall"
            self._bump_cuda_gqa_epoch()
            info["abstain_deposited_graft"] = int(gidx)
        return txt, info

    def _attempt(self, user_text, picks, ngen, deposit, stops,
                 defer_memory=False, suspended=None):
        """One generation attempt.

        GRM-SC2 adds two seams and changes nothing else:

        ``suspended``
            A handle previously returned by an aborted attempt (via
            ``grm_demand.DemandAbort``).  When given, mounting, prompt
            encoding and the prompt forward are all SKIPPED -- the arena is
            already carrying that attempt's cache -- and the decode loop picks
            up exactly where it stopped.  The completed text is therefore the
            text the un-aborted attempt would have produced, not a re-run.

        the abort
            When the caller has installed a ``DemandObserver`` with
            ``early_abort`` on, the observer raises ``DemandAbort`` from inside
            ``self._forward``.  It is caught here, and instead of a
            ``(txt, info)`` pair the attempt returns a SUSPENSION handle.  None
            of the post-generation bookkeeping -- decode, deposit, evict,
            ``live_segs`` append -- runs on that path, because none of it is
            valid for an answer that does not exist yet.
        """
        from core import grm_demand as _demand

        if suspended is not None:
            return self._resume_attempt(suspended, deposit, stops, defer_memory)

        # Final assembly choke point: callers such as diagnostic/probe
        # drivers may invoke _attempt directly instead of step(). Keep L2
        # immediately before any payload is loaded, swapped, or injected.
        picks = self._resolve_revision_mounts(picks)
        self._ensure_h(picks)
        # S1 telemetry (WO-1): reset the per-layer accumulator at the start
        # of EVERY attempt (including shuttling retries) so mass only ever
        # reflects this attempt's reply decode steps — a retry's cache
        # rollback already discards the failed attempt's tokens; the mass
        # accumulator must discard its mass the same way. No-op when
        # telemetry is off (reset_telemetry() just clears an already-None
        # accumulator).
        if self._telemetry_enabled():
            for L in self.m.layers:
                L.self_attn.reset_telemetry()
        if self.caches is None:
            # bootstrap: sink (+ first mounts) enter via the injection path
            seat_order, seat_info = self._rs3_seat_plan(picks)
            mounts = [{"h": self.sink_h}] + [
                self.grafts[i] for i in seat_order]
            inj = []
            # sink is host numpy; deposited grafts are device tensors
            _np = lambda t: t if isinstance(t, np.ndarray) else t.numpy()
            for li in range(len(self.m.layers)):
                inj.append({key: np.concatenate([_np(g["h"][li][key])
                                                 for g in mounts], axis=dim)
                            for key, dim in self.PAYLOAD})
            # GRM-RS3 Part 2: when the lever is ON, pre-rotate the MOUNT rows
            # (not the sink) by the band delta so the layer's own RoPE lands
            # them adjacent to live_shift. OFF leaves `inj` exactly as the
            # legacy branch built it — the same object, unrotated.
            if seat_info["seat_near_live"]:
                inj = self._rs3_rotate_injection(inj, seat_info)
            self._set_injection_host(inj)
            self.cur_mounts = picks
            self.cur_mount_n = sum(self.grafts[i]["ntok"] for i in picks)
            self._commit_native_mount(picks, self.cur_mount_n)
            self._rs3_last_seating = seat_info
        else:
            self.swap(picks)
        prompt_ids = self.encode(self._format_step_prompt(user_text))
        seg_start_ntok = len(prompt_ids)
        try:
            row = self._forward(prompt_ids)
        except _demand.DemandAbort as abort:
            # Fired on the very first answer position. The prompt forward has
            # run, so the cache holds [sink | mounts | question] and nothing
            # else; the injection has NOT been cleared yet, so clear it here
            # exactly as the non-aborted path does one line below -- it fired
            # once and must not fire again on resume.
            kv_graft.clear_injection(self.m)
            return self._suspend_attempt(
                abort, user_text=user_text, picks=picks,
                seg_start_ntok=seg_start_ntok, out=[], cached_out=0,
                ngen=ngen, deposit=deposit, stops=stops,
                defer_memory=defer_memory)
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
            try:
                row = self._forward([out[-1]])
            except _demand.DemandAbort as abort:
                # The forward COMPLETED (the abort is raised after the real
                # logits were computed), so `out[-1]` is now committed to the
                # cache and the resume point is one token further along than
                # the pre-forward state. `cached_out` is incremented to match,
                # exactly as the non-aborted path does on the next line.
                return self._suspend_attempt(
                    abort, user_text=user_text, picks=picks,
                    seg_start_ntok=seg_start_ntok, out=list(out),
                    cached_out=cached_out + 1,
                    ngen=ngen, deposit=deposit, stops=stops,
                    defer_memory=defer_memory)
            cached_out += 1
            out.append(int(row.argmax()))
        if not stopped and not any(s in self.decode(out) for s in stops):
            # The last predicted token is not in the KV cache until it is fed
            # once. Commit it so live/deposit segment lengths match reality.
            try:
                self._forward([out[-1]])
            except _demand.DemandAbort:
                # This forward's logits are UNUSED -- it exists only to commit
                # the final token to KV, and the race's flush-drop convention
                # says it is not an answer position at all. So a fire here is
                # not a fire on the answer: the answer is already complete and
                # every one of its positions read its mounts. Swallow the
                # control-flow signal, keep the completed attempt, and let the
                # normal decision path (which drops this row) rule on it.
                # The cache is committed either way -- the abort is raised
                # after the forward did its work.
                pass
            cached_out += 1
        return self._finish_attempt(
            user_text=user_text, picks=picks, out=out,
            seg_start_ntok=seg_start_ntok, cached_out=cached_out,
            deposit=deposit, stops=stops, defer_memory=defer_memory)

    def _finish_attempt(self, *, user_text, picks, out, seg_start_ntok,
                        cached_out, deposit, stops, defer_memory):
        """The post-generation tail of ``_attempt``, shared with resume.

        GRM-SC2 lifted this out of ``_attempt`` VERBATIM so that a resumed
        attempt runs the identical decode / deposit / evict / live-segment
        bookkeeping rather than a second copy of it that could drift.  Nothing
        in the body changed in the move; the only new thing is that it now has
        two callers.
        """
        # the answer tokens are in the cache; record the live segment
        txt = self.decode(out)
        for stop in stops:
            if stop in txt:
                txt = txt.split(stop)[0]
        txt = txt.strip()
        turn_text = self._format_step_turn(user_text, txt)
        seg_cache_ntok = seg_start_ntok + cached_out
        if defer_memory:
            # Canonical step 2 is the only KV-building step. GQA cannot
            # derive its standalone route key from the contextualized cache,
            # so capture that key now and keep it as a one-turn transient.
            # Step 3 consumes it during deposit_from_cache without another
            # model forward.
            route_key_token = int(
                getattr(self, "_deferred_route_key_token", 0)) + 1
            self._deferred_route_key_token = route_key_token
            route_keys = getattr(self, "_deferred_route_keys", None)
            if not isinstance(route_keys, dict):
                route_keys = {}
                self._deferred_route_keys = route_keys
            route_keys[route_key_token] = self._node_key(turn_text)
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
        info = {"mounts": [i + 1 for i in picks], "resident": S,
                "evicted": evicted,
                "live_tokens": sum(n for _, n in self.live_segs)}
        if defer_memory:
            info["_deferred_memory"] = {
                "turn_text": turn_text,
                "user_text": user_text,
                "seg_cache_ntok": int(seg_cache_ntok),
                "picks": [int(i) for i in picks],
                "deposited": False,
                "importance_committed": False,
                "route_key_prepared": True,
                "route_key_token": int(route_key_token),
            }
        return txt, info

    # -- GRM-SC2: suspend / resume ------------------------------------------

    def _suspend_attempt(self, abort, *, user_text, picks, seg_start_ntok,
                         out, cached_out, ngen, deposit, stops, defer_memory):
        """Freeze an attempt aborted at the D-NGH fire token.

        WHAT IS AND IS NOT CAPTURED.  The KV cache, position, live segments and
        mount bookkeeping are NOT copied into the handle: they are the arena's
        own live state, and the demand block's rollback/restore already owns
        snapshotting them (it snapshots the SAME tuple the shuttle uses, and it
        restores it verbatim when a trip fails to ground).  Copying them here
        would give the resume a second, competing idea of the truth.  What the
        handle carries is only what the arena does NOT keep: the tokens emitted
        so far, how many of them are committed to KV, where the prompt ended,
        and the loop parameters needed to continue.

        WHY NO BOOKKEEPING RUNS HERE.  ``deposit``, ``evict`` and the
        ``live_segs`` append all describe a COMPLETED turn.  An aborted attempt
        has no answer yet, so running any of them would deposit a fragment and
        record a live segment for text that may never be served.  They run on
        resume, through ``_finish_attempt``, exactly once.
        """
        return {
            "grm_sc2_suspended_attempt": True,
            "abort_token_index": int(abort.token_index),
            "abort_mounted_mass": float(abort.mounted_mass),
            "tokens_generated_before_abort": len(out),
            "user_text": user_text,
            "picks": list(picks),
            "seg_start_ntok": int(seg_start_ntok),
            "out": list(out),
            "cached_out": int(cached_out),
            "ngen": int(ngen),
            "deposit": bool(deposit),
            "stops": list(stops),
            "defer_memory": bool(defer_memory),
            "resumed": False,
        }

    def _resume_attempt(self, suspended, deposit, stops, defer_memory):
        """Finish a suspended attempt from the token it aborted at.

        The contract this has to honour is exact: the completed text must be
        the text the un-aborted attempt WOULD have produced.  It is, and for a
        structural reason rather than a hopeful one -- generation here is
        greedy ``argmax`` over a KV cache, the abort was raised only AFTER the
        aborting forward had computed its logits and committed its token, and
        the arena still holds that cache.  So the resumed loop reads the same
        state and takes the same argmax the original loop would have taken at
        every remaining position.  There is no sampling, no temperature and no
        re-prefill to make it merely approximate.

        The caller is responsible for having restored the arena to the abort
        point before calling this.  In the demand block that restoration is the
        ordinary "ungrounded trip -> restore the original attempt's state"
        path, unchanged from SC1.
        """
        from core import grm_demand as _demand

        if not (suspended or {}).get("grm_sc2_suspended_attempt"):
            raise ValueError("not a GRM-SC2 suspended attempt handle")
        if suspended.get("resumed"):
            raise RuntimeError("suspended attempt already resumed")

        user_text = suspended["user_text"]
        picks = list(suspended["picks"])
        seg_start_ntok = int(suspended["seg_start_ntok"])
        out = list(suspended["out"])
        cached_out = int(suspended["cached_out"])
        ngen = int(suspended["ngen"])
        stops = list(suspended["stops"] if stops is None else stops)

        # `out` must already carry every token the aborted attempt predicted,
        # INCLUDING the one at the fire position. The abort unwound before that
        # token was appended, so the demand block refills it from the observer
        # rows (the only exact carrier: decode->encode is not a BPE round
        # trip). A handle that arrives without them cannot be resumed into the
        # original text, and saying so is the honest failure.
        if not out:
            raise _demand.DemandError(
                "GRM-SC2: cannot resume a suspended attempt whose predicted "
                "tokens were not recovered from the observer rows; resuming "
                "from an empty prefix would generate a DIFFERENT answer than "
                "the one this attempt was producing")

        stopped = False
        remaining = ngen - 1 - (len(out) - 1)
        for _ in range(max(0, remaining)):
            if any(s in self.decode(out) for s in stops):
                stopped = True
                break
            row = self._forward([out[-1]])
            cached_out += 1
            out.append(int(row.argmax()))
        if not stopped and not any(s in self.decode(out) for s in stops):
            self._forward([out[-1]])
            cached_out += 1
        suspended["resumed"] = True
        return self._finish_attempt(
            user_text=user_text, picks=picks, out=out,
            seg_start_ntok=seg_start_ntok, cached_out=cached_out,
            deposit=bool(suspended["deposit"] if deposit is None else deposit),
            stops=stops,
            defer_memory=bool(suspended["defer_memory"]
                              if defer_memory is None else defer_memory))

    def deposit_deferred_turn(self, info):
        """Pass 3: deposit the accepted pass-2 output from its live cache.

        This is the deposit/classification block from ``_attempt`` executed
        after output.  It requires the accepted cache to remain current and
        replaces the pass-2 ``(None, ntok)`` live-segment marker when that
        segment still resides in the live window.
        """
        pending = (info or {}).get("_deferred_memory")
        if not isinstance(pending, dict):
            raise ValueError("missing deferred-memory payload")
        if pending.get("deposited"):
            raise RuntimeError("deferred turn already deposited")
        turn_text = str(pending["turn_text"])
        user_text = str(pending["user_text"])
        seg_cache_ntok = int(pending["seg_cache_ntok"])
        picks = [int(i) for i in pending.get("picks", ())]
        route_key = None
        if pending.get("route_key_prepared"):
            route_key_token = int(pending.get("route_key_token", -1))
            route_keys = getattr(self, "_deferred_route_keys", {})
            route_key = route_keys.get(route_key_token)
            if route_key is None:
                raise RuntimeError("prepared deferred route key is missing")
        gidx = (
            self.deposit_from_cache(
                turn_text, seg_cache_ntok, route_key=route_key)
            if self.cache_deposits else self.deposit(turn_text)
        )
        if pending.get("route_key_prepared"):
            self._deferred_route_keys = {}
        if picks:
            new_rare = (self._rare_tokens(turn_text)
                        - self._rare_tokens(user_text))
            for i in picks:
                g = self.grafts[i]
                if "rare" not in g:
                    g["rare"] = self._rare_tokens(g["text"])
                new_rare -= g["rare"]
            if not new_rare:
                self.grafts[gidx]["kind"] = "recall"
                self._bump_cuda_gqa_epoch()
        for idx in range(len(self.live_segs) - 1, -1, -1):
            live_gidx, live_ntok = self.live_segs[idx]
            if live_gidx is None and int(live_ntok) == seg_cache_ntok:
                self.live_segs[idx] = (int(gidx), int(live_ntok))
                break
        pending["deposit_node_id"] = int(gidx)
        pending["deposited"] = True
        return int(gidx)

    def commit_deferred_importance(self, info):
        """Pass 3: commit the S4 accounting selected by ``step`` in pass 2."""
        pending = (info or {}).get("_deferred_memory")
        if not isinstance(pending, dict):
            raise ValueError("missing deferred-memory payload")
        importance = pending.get("importance_bookkeeping")
        if importance is None:
            return ()
        if pending.get("importance_committed"):
            raise RuntimeError("deferred importance already committed")
        self._commit_s4_attempt(
            importance.get("routed", ()),
            importance.get("mounted", ()),
            importance.get("grounded_mounts", ()),
            turn=int(importance["turn"]),
        )
        pending["importance_committed"] = True
        return tuple(sorted({
            int(i) for key in ("routed", "mounted", "grounded_mounts")
            for i in importance.get(key, ())
        }))


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
        request = self._route_backend_request()
        if request == "python":
            return False
        if request == "cuda_ragged":
            return True
        # ``auto`` retains the historic, default-off environment toggle.
        return os.environ.get("GRM_GQA_CUDA_ROUTE", "").lower() in (
            "1", "true", "yes", "on")

    def _bump_cuda_gqa_epoch(self):
        """Mutation-epoch bump. Called at every GQAArenaCache-side site that
        can change what `_cuda_route_bank_signature` would produce (graft
        add/retire/replace, route-key/kind/child_cents changes, truncation or
        restore of self.grafts). O(1) at mutation time. Repository mutation
        sites share this choke point, so the epoch is the sole production hot-
        path staleness gate; `GRM_GQA_BRIDGE_PARANOID=1` re-runs the signature
        walk as a development assertion against under-invalidation.
        """
        self._cuda_gqa_epoch = getattr(self, "_cuda_gqa_epoch", 0) + 1

    def _cuda_route_bank_signature(self):
        """Cheap O(N) walk: eligibility + signature only, NO stacking. This
        is the part of the old `_cuda_route_bank_inputs` that was NOT the
        P0-receipted defect (~1% of the reused-call cost). It runs when the
        mutation epoch changes (and on every paranoid-mode call), not on the
        production reused-bank hot path. Returns
        (node_ids, signature, rows) where `rows` are the raw per-node key
        arrays (not yet materialized as a dense bank) so a cache miss can
        build them without a second walk. Leaf rows may differ only in their
        token extent; common KV-head and head dimensions are required."""
        rows = []
        node_ids = []
        sig_rows = []
        geometry = None
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
            if key.ndim != 3 or any(int(dim) <= 0 for dim in key.shape):
                return None
            if not np.isfinite(key).all():
                return None
            row_geometry = (int(key.shape[0]), int(key.shape[2]))
            if geometry is None:
                geometry = row_geometry
            elif row_geometry != geometry:
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
    def _cuda_build_dense_route_bank(rows):
        """Return an exact dense fp32 bank plus its auditable layout receipt.

        Every row has shape ``[kv_heads, tokens_i, head_dim]``. Equal-length
        rows retain the historical ``np.stack`` path. Mixed-length rows are
        copied into zero-filled prefixes of one max-token bank. Because the
        route law takes a maximum of absolute dot products, a zero tail cannot
        raise any non-empty row's score.
        """
        if not rows:
            raise ValueError("GQA CUDA route bank requires at least one row")

        arrays = []
        token_counts = []
        geometry = None
        for row in rows:
            key = np.asarray(row, dtype=np.float32)
            if key.ndim != 3 or any(int(dim) <= 0 for dim in key.shape):
                raise ValueError(
                    "GQA CUDA route rows must be non-empty rank-3 arrays")
            if not np.isfinite(key).all():
                raise ValueError("GQA CUDA route rows must be finite")
            row_geometry = (int(key.shape[0]), int(key.shape[2]))
            if geometry is None:
                geometry = row_geometry
            elif row_geometry != geometry:
                raise ValueError(
                    "GQA CUDA route rows must share KV-head/head-dim geometry")
            arrays.append(key)
            token_counts.append(int(key.shape[1]))

        max_tokens = max(token_counts)
        same_shape = len(set(token_counts)) == 1
        if same_shape:
            route_bank = np.ascontiguousarray(
                np.stack(arrays), dtype=np.float32)
        else:
            kv_heads, head_dim = geometry
            route_bank = np.zeros(
                (len(arrays), kv_heads, max_tokens, head_dim),
                dtype=np.float32,
                order="C",
            )
            for idx, key in enumerate(arrays):
                route_bank[idx, :, :key.shape[1], :] = key

        raw_value_count = sum(int(key.size) for key in arrays)
        padded_value_count = int(route_bank.size)
        receipt = {
            "schema": "grm_gqa_cuda_route_bank_layout_v1",
            "layout": "dense" if same_shape else "zero_padded",
            "node_count": len(arrays),
            "kv_heads": int(geometry[0]),
            "head_dim": int(geometry[1]),
            "row_token_counts": tuple(token_counts),
            "max_tokens": int(max_tokens),
            "raw_value_count": raw_value_count,
            "padded_value_count": padded_value_count,
            "raw_bytes": raw_value_count * np.dtype(np.float32).itemsize,
            "padded_bytes": int(route_bank.nbytes),
            "padding_ratio": padded_value_count / raw_value_count,
        }
        return route_bank, receipt

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
        # ``None`` is an epoch-owned result too: a live session containing a
        # hierarchy, a missing native id, or malformed key must not repeat an
        # O(N) eligibility/signature walk on every turn merely because there
        # is no attachable bank.  The epoch is the cache-valid bit for both
        # the positive bank and the fail-closed negative result.
        epoch_says_fresh = cache_epoch == epoch

        if epoch_says_fresh and not paranoid:
            self._last_cuda_gqa_bank_event = {
                "status": "reused" if cached is not None else "ineligible_reused",
                "epoch": int(epoch),
                "signature_walk": False,
                "bank_materialized": False,
            }
            return cached

        sig = self._cuda_route_bank_signature()
        if sig is None:
            if paranoid and epoch_says_fresh:
                assert cached is None, (
                    "GRM_GQA_BRIDGE_PARANOID: epoch says the cached bank is "
                    "fresh but the signature walk finds no eligible bank — "
                    "epoch under-invalidation")
            self._cuda_gqa_bank_cache = None
            self._cuda_gqa_cache_epoch = epoch
            self._cuda_gqa_padding_receipt = None
            self._cuda_gqa_native_to_idx_cache = None
            self._last_cuda_gqa_bank_event = {
                "status": "ineligible",
                "epoch": int(epoch),
                "signature_walk": True,
                "bank_materialized": False,
            }
            return None
        node_ids_np, signature, rows = sig

        if paranoid and epoch_says_fresh:
            assert cached is not None and cached[2] == signature, (
                "GRM_GQA_BRIDGE_PARANOID: epoch says the cached bank is "
                "fresh but the signature walk disagrees — epoch under-"
                "invalidation (a mutation changed the signature without "
                "bumping _cuda_gqa_epoch)")

        if (cached is not None and (cached[2] is signature
                                    or cached[2] == signature)):
            # Signature-identical even though the epoch moved (e.g. a
            # metadata-only mutation bumped epoch but didn't touch anything
            # the bank cares about) — over-invalidation is fine, just skip
            # the re-stack and refresh the cache_epoch so the NEXT call can
            # take the fast path again.
            self._cuda_gqa_bank_cache = cached
            self._cuda_gqa_cache_epoch = epoch
            self._last_cuda_gqa_bank_event = {
                "status": "signature_reused",
                "epoch": int(epoch),
                "signature_walk": True,
                "bank_materialized": False,
                "node_count": int(node_ids_np.shape[0]),
            }
            return cached

        route_bank, layout_receipt = self._cuda_build_dense_route_bank(rows)
        bank_inputs = (route_bank, node_ids_np, signature)
        self._cuda_gqa_bank_cache = bank_inputs
        self._cuda_gqa_cache_epoch = epoch
        self._cuda_gqa_padding_receipt = layout_receipt
        # A different bank object means the cached O(N) native-id reverse map
        # no longer describes its rows.  Metadata-only epoch bumps take the
        # signature-reused branch above and deliberately retain that map.
        self._cuda_gqa_native_to_idx_cache = None
        self._last_cuda_gqa_bank_event = {
            "status": "rebuilt",
            "epoch": int(epoch),
            "signature_walk": True,
            "bank_materialized": True,
            "node_count": int(node_ids_np.shape[0]),
            "layout": layout_receipt["layout"],
            "raw_bytes": int(layout_receipt["raw_bytes"]),
            "padded_bytes": int(layout_receipt["padded_bytes"]),
        }
        return bank_inputs

    def _ensure_cuda_route_bank(self, store, bank_inputs):
        if getattr(self, "_cuda_gqa_route_unavailable", False):
            self._last_cuda_gqa_upload_event = {
                "status": "unavailable_cached"}
            return False
        if not hasattr(store, "configure_cuda_gqa_route_bank"):
            self._last_cuda_gqa_upload_event = {
                "status": "store_unsupported"}
            return False
        if bank_inputs is None:
            self._last_cuda_gqa_upload_event = {"status": "no_bank"}
            return False
        route_bank, node_ids, signature = bank_inputs
        # This method runs for every flagged route. The bank signature has
        # one entry per resident node, so equality alone is an O(N) re-prep
        # tax even when the arena handed us the identical tuple object.
        stored_signature = getattr(store, "_cuda_gqa_bank_signature", None)
        if (getattr(store, "_cuda_gqa_bank", None) is not None
                and (stored_signature is signature
                     or stored_signature == signature)):
            self._last_cuda_gqa_upload_event = {
                "status": "reused", "node_count": int(node_ids.shape[0])}
            return True
        try:
            configure_kwargs = (
                {"signature": signature}
                if getattr(store, "supports_cuda_gqa_epoch_signature", False)
                else {}
            )
            store.configure_cuda_gqa_route_bank(
                route_bank, node_ids, **configure_kwargs)
            store._cuda_gqa_bank_signature = signature
        except Exception as exc:
            self._cuda_gqa_route_unavailable = True
            self._last_cuda_gqa_upload_event = {
                "status": "failed", "error_type": type(exc).__name__}
            return False
        self._last_cuda_gqa_upload_event = {
            "status": "uploaded",
            "node_count": int(node_ids.shape[0]),
            "bytes": int(route_bank.nbytes),
        }
        return True

    def _cuda_gqa_native_to_idx(self, bank_inputs):
        """Return the exact bank-row native-id -> graft-index map.

        The old GQA bridge rebuilt this map from ``cand`` on every route.
        The ragged bank already fixes row order at its epoch-bound attach, so
        the map and its graft-index membership set are invariant until that
        bank object changes. Caching them removes an O(N) Python dictionary
        build from the resident route path without changing mapping or ties.
        """
        cached = getattr(self, "_cuda_gqa_native_to_idx_cache", None)
        if cached is not None and cached[0] is bank_inputs:
            self._last_cuda_gqa_reverse_map_event = {
                "status": "reused", "node_count": len(cached[1])}
            return cached[1], cached[2]

        node_ids_np = bank_inputs[1]
        row_to_graft_idx = []
        for idx, graft in enumerate(self.grafts):
            if (graft.get("retired")
                    or graft.get("kind", "turn") == "recall"):
                continue
            # These are exactly the reasons the signature walk declines a
            # complete bank. Re-checking them on an attach is fail-closed;
            # an unchanged epoch never reaches this branch again.
            if (graft.get("native_node_id") is None
                    or graft.get("child_cents") or "cent" not in graft):
                self._last_cuda_gqa_reverse_map_event = {
                    "status": "ineligible"}
                return None
            row_to_graft_idx.append(int(idx))

        if len(row_to_graft_idx) != int(node_ids_np.shape[0]):
            self._last_cuda_gqa_reverse_map_event = {
                "status": "row_count_mismatch",
                "graft_rows": len(row_to_graft_idx),
                "bank_rows": int(node_ids_np.shape[0]),
            }
            return None
        if any(int(self.grafts[idx]["native_node_id"]) != int(node_ids_np[row])
               for row, idx in enumerate(row_to_graft_idx)):
            self._last_cuda_gqa_reverse_map_event = {
                "status": "node_id_order_mismatch"}
            return None

        native_to_idx = {
            int(node_ids_np[row]): row_to_graft_idx[row]
            for row in range(len(row_to_graft_idx))
        }
        if len(native_to_idx) != len(row_to_graft_idx):
            self._last_cuda_gqa_reverse_map_event = {
                "status": "duplicate_native_id"}
            return None
        graft_idx_set = frozenset(row_to_graft_idx)
        self._cuda_gqa_native_to_idx_cache = (
            bank_inputs, native_to_idx, graft_idx_set)
        self._last_cuda_gqa_reverse_map_event = {
            "status": "rebuilt", "node_count": len(native_to_idx)}
        return native_to_idx, graft_idx_set

    def _python_gqa_exact_order(self, pkey, cand, limit):
        """Stable raw-GQA reference used only by the opt-in parity gate."""
        scored = []
        for idx in cand:
            score = self._key_score(pkey, self.grafts[idx]["cent"])
            if np.isfinite(score):
                scored.append((score, idx))
        # `cand` is ascending. Python's stable sort is therefore the exact
        # production tie policy; CUDA must preserve this same bank-row order.
        scored.sort(key=lambda item: -item[0])
        want = min(max(0, int(limit)), len(cand))
        return [idx for _, idx in scored[:want]]

    def _cuda_route_order(self, pkey, cand, limit, exclude=()):
        if limit is None:
            self._route_receipt_fallback("cuda_requires_limited_route")
            return None
        store = getattr(self, "native_store", None)
        if store is None or not hasattr(store, "route_gqa_cuda"):
            self._route_receipt_fallback("cuda_store_unavailable")
            return None
        if not self._cuda_route_enabled():
            self._route_receipt_fallback("cuda_disabled")
            return None
        started = self._route_profile_start()
        bank_inputs = self._cuda_route_bank_inputs()
        self._route_profile_end("cuda_bank_inputs_ms", started)
        self._route_receipt_rebuild(
            "cuda_bank", dict(getattr(
                self, "_last_cuda_gqa_bank_event", {"status": "unknown"})))
        if bank_inputs is None:
            self._route_receipt_rebuild(
                "node_centroids", {"status": "cuda_bank_ineligible"})
            self._route_receipt_rebuild(
                "candidate_key_marshal", {"status": "cuda_bank_ineligible"})
            self._route_receipt_fallback("cuda_bank_ineligible")
            return None
        self._route_receipt_rebuild(
            "node_centroids", {
                "status": "epoch_bank",
                "node_count": int(bank_inputs[1].shape[0]),
                "bytes": int(bank_inputs[0].nbytes),
            })
        self._route_receipt_rebuild(
            "candidate_key_marshal", {
                "status": getattr(
                    self, "_last_cuda_gqa_bank_event", {}).get("status", "unknown"),
                "bytes": int(bank_inputs[0].nbytes),
                "node_count": int(bank_inputs[1].shape[0]),
            })
        started = self._route_profile_start()
        if not self._ensure_cuda_route_bank(store, bank_inputs):
            self._route_profile_end("cuda_bank_upload_ms", started)
            self._route_receipt_rebuild(
                "cuda_bank_upload", dict(getattr(
                    self, "_last_cuda_gqa_upload_event", {"status": "unknown"})))
            self._route_receipt_fallback("cuda_bank_attach_failed")
            return None
        self._route_profile_end("cuda_bank_upload_ms", started)
        self._route_receipt_rebuild(
            "cuda_bank_upload", dict(getattr(
                self, "_last_cuda_gqa_upload_event", {"status": "unknown"})))

        # `cand` is route()'s epoch-cached eligible base minus its small
        # current exclude set. Prove that invariant with O(len(exclude)) work
        # rather than rebuilding an O(len(cand)) dict every turn.
        bank_size = int(bank_inputs[1].shape[0])
        cand_base = self._route_cand_base()
        if bank_size != len(cand_base):
            self._route_receipt_fallback("cuda_candidate_base_mismatch")
            return None
        started = self._route_profile_start()
        cached_map = self._cuda_gqa_native_to_idx(bank_inputs)
        self._route_profile_end("cuda_reverse_map_ms", started)
        self._route_receipt_rebuild(
            "cuda_reverse_map", dict(getattr(
                self, "_last_cuda_gqa_reverse_map_event", {"status": "unknown"})))
        if cached_map is None:
            self._route_receipt_fallback("cuda_reverse_map_ineligible")
            return None
        native_to_idx, bank_idx_set = cached_map
        exclude_set = set(exclude) if exclude else None
        excludes_in_bank = (
            sum(1 for idx in exclude_set if idx in bank_idx_set)
            if exclude_set is not None else 0)
        if len(cand) != bank_size - excludes_in_bank:
            self._route_receipt_fallback("cuda_candidate_subset_mismatch")
            return None
        want = min(max(0, int(limit)), len(cand))
        if want <= 0:
            return []
        topk = min(16, bank_size, want + excludes_in_bank)
        if topk < want:
            self._route_receipt_fallback("cuda_topk_cap")
            return None
        started = self._route_profile_start()
        query = np.ascontiguousarray(pkey, dtype=np.float32)
        self._route_profile_end("query_key_marshal_ms", started)
        self._route_receipt_rebuild(
            "query_key_marshal", {
                "status": "per_turn", "bytes": int(query.nbytes),
                "shape": tuple(int(dim) for dim in query.shape),
            })
        try:
            started = self._route_profile_start()
            routed_native = store.route_gqa_cuda(query, topk=topk)
        except Exception:
            self._route_receipt_fallback("cuda_route_exception")
            return None
        self._route_profile_end("cuda_route_call_ms", started)
        started = self._route_profile_start()
        routed = []
        for node_id in routed_native:
            idx = native_to_idx.get(int(node_id))
            if idx is None:
                self._route_receipt_fallback("cuda_unknown_native_id")
                return None
            if exclude_set is not None and idx in exclude_set:
                continue
            routed.append(idx)
            if len(routed) >= want:
                break
        self._route_profile_end("cuda_result_remap_ms", started)
        if len(routed) < want:
            self._route_receipt_fallback("cuda_short_result")
            return None
        if bool(getattr(self, "route_parity_check", False)):
            started = self._route_profile_start()
            expected = self._python_gqa_exact_order(pkey, cand, want)
            self._route_profile_end("cuda_python_parity_ms", started)
            parity_ok = routed == expected
            self._route_receipt_parity(
                checked=True, ok=parity_ok,
                reference="python_raw_gqa_key_score_stable_ties")
            if not parity_ok:
                raise AssertionError(
                    "cuda_ragged rank/tie parity mismatch: "
                    f"expected {expected}, got {routed}")
        self.last_route_backend = "cuda"
        return routed

    def _native_route_order(self, pkey, qrare, cand, limit=None, exclude=()):
        # `exclude` accepted for base-class call-site compatibility
        # (route() passes it positionally-by-keyword to every dialect's
        # _native_route_order since the P2 MLA CUDA route fix). The ragged
        # GQA bridge consumes it too: the epoch-cached native-id map lets the
        # resident route path price only current exclusions, not all nodes.
        store = getattr(self, "native_store", None)
        if store is None or not hasattr(store, "route_gqa"):
            return None
        if not cand:
            return []
        if not qrare:
            cuda_order = self._cuda_route_order(pkey, cand, limit, exclude)
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
