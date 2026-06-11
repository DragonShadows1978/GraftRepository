"""GQA ARENA (Qwen3-4B) — round 1 of the GQA arena port.

The E4-arena protocol on the OTHER dialect: persistent cache, seating
[SINK | ARENA | LIVE], scripted turns feed through the live region and
deposit as standalone full-K/V grafts; per probe the arena swaps to the
routed top-3 by cache surgery; live turns evict outside the 2-turn window.

Dialect forks under test (vs the MiniCPM3/MLA arena):
  - payload = per-layer pre-RoPE (k, v) full GQA tensors (both seq dim=2);
    mount surgery re-RoPEs the FULL key at the arena seats (MLA re-RoPEs
    only the 32-d shared k_pe; the latent is position-free)
  - router = layer-0 |q.k| (E1 Qwen3 law: per-head qk-norm makes layer-0
    keys a normalized routing space; MiniCPM3 has no qk-norm and routes in
    the latent instead)
  - live_shift on GQAAttentionTC (new — same fixed-width law as MLA)
Target: recall parity with the MLA arena gate (6/6), bounded residency,
coherent generation across swaps + evictions on ONE cache.
"""
import os, sys, time, numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.qwen3_tc import Qwen3_TC, _snap
from core.mistral7b_tc import BlockTC, F
from core import kv_graft
from tokenizers import Tokenizer as HFTok
import ast, re
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "test_graft_e4_conversation.py")).read()
SCRIPTED = ast.literal_eval(re.search(r"SCRIPTED = (\[.*?\n\])\n", _src, re.S).group(1))
PROBES = ast.literal_eval(re.search(r"PROBES = (\[.*?\n\])\n", _src, re.S).group(1))

tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = Qwen3_TC.from_pretrained()
for L in m.layers:
    L.self_attn.quant_kv_cache = False
print(f"loaded: {info}", flush=True)

STOPS = ("\nUser:", "User:", "\nAssistant:", "Assistant:", "\n\n")


class GQAArena:
    def __init__(self, m, encode, decode, sink_text="<conversation>\n",
                 arena_width=256, topk=3, live_turns=2, max_live=4096):
        self.m, self.encode, self.decode = m, encode, decode
        self.width, self.topk, self.live_turns = arena_width, topk, live_turns
        self.dt = BlockTC.COMPUTE_DTYPE
        sink_ids = encode(sink_text)
        m.extend_rope(len(sink_ids) + arena_width + max_live)
        self.sink_h = kv_graft.harvest_kv(m, sink_ids)
        self.n_sink = len(sink_ids)
        self.live_shift = self.n_sink + arena_width
        self.caches = None
        self.pos = 0
        self.cur_mounts, self.cur_mount_n = [], 0
        self.live_segs = []                  # [(graft_idx or None, ntok)]
        self.grafts = []                     # {h, k0, ntok, text}

    # ------------------------------------------------------- repository
    def deposit(self, text):
        ids = self.encode(text)
        h = kv_graft.harvest_kv(self.m, ids)
        dev = [{"k": tc.tensor(np.ascontiguousarray(h[li]["k"])).astype(self.dt),
                "v": tc.tensor(np.ascontiguousarray(h[li]["v"])).astype(self.dt)}
               for li in range(len(self.m.layers))]
        # router key: layer-0 pre-RoPE post-qknorm keys (E1 Qwen3 law)
        self.grafts.append({"h": dev, "k0": h[0]["k"][0].astype(np.float32),
                            "ntok": len(ids), "text": text})
        return len(self.grafts) - 1

    def route(self, bare_text, exclude=()):
        qcap = kv_graft.capture_queries(self.m, self.encode(bare_text),
                                        layer_filter={0})
        q = qcap[0][0].astype(np.float32)            # (H, Lq, Dh)
        H, _, Dh = q.shape
        scored = []
        for gi, g in enumerate(self.grafts):
            if gi in exclude:
                continue
            k = g["k0"]                              # (Hkv, Lk, Dh)
            kk = np.repeat(k, H // k.shape[0], axis=0)
            sc = np.einsum("hqd,hkd->hqk", q, kk) / np.sqrt(Dh)
            scored.append((float(np.abs(sc).max(axis=(1, 2)).mean()), gi))
        scored.sort(reverse=True)
        return [gi for _, gi in scored]

    # -------------------------------------------------------- cache ops
    def _graft_block(self, picks, li):
        hs = [self.grafts[i]["h"][li] for i in picks]
        k = hs[0]["k"] if len(hs) == 1 else tc.cat([h["k"] for h in hs], dim=2)
        v = hs[0]["v"] if len(hs) == 1 else tc.cat([h["v"] for h in hs], dim=2)
        n = k.shape[2]
        k = F.apply_rotary(k, self.m.rope_cos.slice(0, self.n_sink, n),
                           self.m.rope_sin.slice(0, self.n_sink, n))
        return k, v, n

    def swap(self, picks):
        if picks == self.cur_mounts or self.caches is None:
            self.cur_mounts = picks
            return
        n_new = 0
        head = self.n_sink + self.cur_mount_n
        out = []
        for li, (k, v) in enumerate(self.caches):
            S = k.shape[2]
            pk = [k.slice(2, 0, self.n_sink)]
            pv = [v.slice(2, 0, self.n_sink)]
            if picks:
                kg, vg, n_new = self._graft_block(picks, li)
                pk.append(kg)
                pv.append(vg)
            if S > head:
                pk.append(k.slice(2, head, S - head))
                pv.append(v.slice(2, head, S - head))
            out.append((tc.cat(pk, dim=2) if len(pk) > 1 else pk[0],
                        tc.cat(pv, dim=2) if len(pv) > 1 else pv[0]))
        self.caches = out
        self.cur_mounts = picks
        self.cur_mount_n = n_new
        if n_new > self.width:
            raise ValueError(f"mounts ({n_new}) exceed arena width ({self.width})")

    def evict(self):
        if len(self.live_segs) <= self.live_turns or self.caches is None:
            return 0
        drop = self.live_segs[:-self.live_turns]
        self.live_segs = self.live_segs[-self.live_turns:]
        drop_n = sum(n for _, n in drop)
        head = self.n_sink + self.cur_mount_n
        out = []
        for k, v in self.caches:
            S = k.shape[2]
            out.append((tc.cat([k.slice(2, 0, head),
                                k.slice(2, head + drop_n, S - head - drop_n)], dim=2),
                        tc.cat([v.slice(2, 0, head),
                                v.slice(2, head + drop_n, S - head - drop_n)], dim=2)))
        self.caches = out
        return drop_n

    # --------------------------------------------------------- forward
    def _forward(self, ids):
        with tc.no_grad():
            lg, self.caches = self.m(np.array([ids], dtype=np.int64),
                                     kv_caches=self.caches,
                                     position_offset=self.pos,
                                     last_token_only=True)
        self.pos += len(ids)
        return lg.numpy()[0, -1].astype(np.float32)

    def feed(self, turn_text, deposit=True):
        for L in self.m.layers:
            L.self_attn.live_shift = self.live_shift
        ids = self.encode(turn_text)
        if self.caches is None:        # bootstrap: sink enters via injection
            kv_graft.set_injection(self.m, self.sink_h)
        self._forward(ids)
        kv_graft.clear_injection(self.m)
        gidx = self.deposit(turn_text) if deposit else None
        self.live_segs.append((gidx, len(ids)))
        self.evict()

    def step(self, user_text, ngen=48, deposit=True):
        for L in self.m.layers:
            L.self_attn.live_shift = self.live_shift
        live_idx = {g for g, _ in self.live_segs if g is not None}
        ranking = self.route(user_text, exclude=live_idx)
        picks = sorted(ranking[:self.topk])
        self.swap(picks)
        ids = self.encode(f"User: {user_text}\nAssistant:")
        if self.caches is None:
            kv_graft.set_injection(self.m, self.sink_h)
        row = self._forward(ids)
        kv_graft.clear_injection(self.m)
        out = [int(row.argmax())]
        for _ in range(ngen - 1):
            row = self._forward([out[-1]])
            out.append(int(row.argmax()))
        ans = self.decode(out)
        for stop in STOPS:
            if stop in ans:
                ans = ans.split(stop)[0]
        ans = ans.strip()
        seg_n = len(ids) + len(out)
        gidx = (self.deposit(f"User: {user_text}\nAssistant: {ans}\n")
                if deposit else None)
        self.live_segs.append((gidx, seg_n))
        ev = self.evict()
        info = {"mounts": picks, "resident": self.caches[0][0].shape[2],
                "live_tokens": sum(n for _, n in self.live_segs),
                "evicted": ev}
        return ans, info


arena = GQAArena(m, encode=lambda t: tok.encode(t).ids,
                 decode=lambda ids: tok.decode(ids),
                 arena_width=256, topk=3, live_turns=2)
print(f"arena: sink={arena.n_sink} seats, width={arena.width}, "
      f"live_shift={arena.live_shift}", flush=True)

for i, (u, a) in enumerate(SCRIPTED):
    t0 = time.perf_counter()
    arena.feed(f"User: {u}\nAssistant: {a}\n")
    S = arena.caches[0][0].shape[2]
    print(f"turn {i+1:2d} fed+deposited  resident={S:3d}  "
          f"({time.perf_counter()-t0:.2f}s)", flush=True)

print("\n=== probes (one persistent cache, layer-0 |q.k| router) ===",
      flush=True)
hits = 0
for q, acc in PROBES:
    t0 = time.perf_counter()
    ans, info_ = arena.step(q, ngen=48, deposit=False)
    dt = time.perf_counter() - t0
    ok = any(s in ans.lower() for s in acc)
    hits += ok
    print(f"  [res={info_['resident']:3d} live={info_['live_tokens']:3d} "
          f"evict={info_['evicted']:3d} {dt:4.1f}s] mounts={info_['mounts']} "
          f"{'HIT ' if ok else 'MISS'} | {ans[:58]!r}", flush=True)
print(f"\nGQA-ARENA: {hits}/{len(PROBES)} (MLA arena gate was 6/6; "
      f"E1 Qwen3 routed memory was 10/10 recall@3)", flush=True)
print("DONE", flush=True)
