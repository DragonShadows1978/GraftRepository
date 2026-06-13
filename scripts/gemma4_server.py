"""Ollama-API-compatible server over the APA-GRM Gemma 4 12B port
(QAT q4_0 exact import, symmetric-8 INT4 kernels).

Same contract as qwen35_server (the corpus driver's POST /api/generate
with {model, prompt, stream:false, options:{temperature,num_predict}}).
GRM PREFIX MOUNT: pure-KV caches — sliding rings (<=1023 keys) and
append-only global K=V — are branch-safe by construction (cat/slice
make fresh tensors); the state after the longest previously-seen token
prefix is restored and only the request's variable tail prefills.

APA is ENABLED EXPLICITLY at startup on the 8 global layers and logged
(the qwen35 shim relied on a default that was never flipped — the mode
the server runs MUST be printed, not assumed).

Single-threaded HTTP. Default port 11436 (qwen shim owns 11435).

  python3 scripts/gemma4_server.py [port]
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.gemma4_tc import Gemma4_TC, _MODEL_DIR, _cast    # noqa: E402

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 11436
MAX_CACHED_PREFIXES = 8
EOS = {1, 106, 50}                 # per generation_config.json
SUPPRESS = (258882, 258883)        # eoi/eoa, per generation_config

from transformers import AutoTokenizer                      # noqa: E402
tok = AutoTokenizer.from_pretrained(_MODEL_DIR)

tc.set_alloc_pooling(True)
model, info = Gemma4_TC.from_pretrained()                   # QAT default
n_apa = 0
for L in model.layers:
    if L.mixer.is_global:
        L.mixer.attention_mode = "apa_selective"
        # r0.10 per the Architect's standing practice — ppl-verified on
        # THIS model 2026-06-12 (sweep: r0.05..r0.15 all within noise of
        # standard, post blend-mask fix)
        L.mixer.refine_percentile = 0.10
        n_apa += 1
print(f"loaded: {info} | APA apa_selective ON ({n_apa} global layers, "
      f"bulk{model.layers[5].mixer.bulk_bits}/refine "
      f"{model.layers[5].mixer.refine_percentile})", flush=True)


def _kv(c):
    """Ordered (k, v) view of a tuple cache or a KVRing (pre-wrap)."""
    return c.ordered() if hasattr(c, "ordered") else c


def to_host(caches):
    """Prefix states live in HOST RAM: at 12B a single full-length
    cache set is ~230-335MB — 8 GPU-resident entries would eat half
    the card. fp32 round-trip is exact for bf16, and host export is a
    COPY (the KVRing ownership contract)."""
    out = []
    for c in caches:
        k, v = _kv(c)
        out.append((k.float().numpy(), v.float().numpy()))
    return out


def to_device(host_caches):
    """Upload on mount (~20-40ms at PCIe speed — noise vs the prefill
    it replaces). A fresh upload is inherently branch-safe and
    satisfies the model's consumed-list contract."""
    return [(_cast(tc.tensor(k)), _cast(tc.tensor(v)))
            for k, v in host_caches]


class PrefixCache:
    def __init__(self):
        self.entries = []          # [ids tuple, HOST caches, hits, stamp]

    def lookup(self, ids):
        best = None
        for e in self.entries:
            p = e[0]
            if len(p) < len(ids) and tuple(ids[:len(p)]) == p:
                if best is None or len(p) > len(best[0]):
                    best = e
        if best is not None:
            best[2] += 1
        return best

    def store(self, ids, caches):
        key = tuple(ids)
        for e in self.entries:
            if e[0] == key:
                return
        self.entries.append([key, to_host(caches), 0, time.time()])
        if len(self.entries) > MAX_CACHED_PREFIXES:
            self.entries.sort(key=lambda e: (e[2], e[3]))
            self.entries.pop(0)


PREFIX = PrefixCache()
LAST_IDS = []


def common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def generate(prompt, temperature, num_predict):
    global LAST_IDS
    enc = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True)
    # transformers 5.12 returns a BatchEncoding here — which subclasses
    # UserDict, NOT dict (an isinstance(enc, dict) check silently
    # misses it and np.asarray turns it into its KEY STRINGS)
    ids = np.asarray(enc["input_ids"] if hasattr(enc, "keys") else enc,
                     np.int64)
    t0 = time.perf_counter()

    hit = PREFIX.lookup(ids)
    with tc.no_grad():
        if hit is not None:
            caches = to_device(hit[1])
            off = len(hit[0])
            lg, caches = model(ids[off:][None, :], caches=caches,
                               position_offset=off, last_token_only=True)
            mounted = off
        else:
            lg, caches = model(ids[None, :], last_token_only=True)
            mounted = 0
        if LAST_IDS is not None and len(LAST_IDS):
            # cap at len-1: identical prompts must still mount (the
            # mount needs >=1 tail token to prefill)
            cp = min(common_prefix_len(ids, LAST_IDS), len(ids) - 1)
            if cp >= 64 and cp > mounted:
                if len(ids) <= 1023:
                    # pure-KV: the state-at-cp is EXACTLY a slice of
                    # this request's own post-prefill caches (global KV
                    # is append-only; below the window the sliding ring
                    # holds every key). Minting is free — no re-prefill
                    # (which was doubling request latency). _kv handles
                    # both tuple caches and (pre-wrap) KVRings.
                    pc = []
                    for c in caches:
                        kk, vv = _kv(c)
                        pc.append((kk.slice(2, 0, cp),
                                   vv.slice(2, 0, cp)))
                else:
                    # past the window the ring has trimmed: state-at-cp
                    # is no longer a slice. Re-prefill to mint.
                    _, pc = model(ids[None, :cp], last_token_only=True)
                PREFIX.store(ids[:cp], pc)
                del pc
                tc.empty_cache()
        LAST_IDS = list(ids)

        off = len(ids)
        out = []
        rng = np.random.default_rng()
        nxt_logits = lg.float().numpy()[0, -1]
        t_prefill = time.perf_counter() - t0
        for _ in range(min(num_predict, 8192)):
            for s in SUPPRESS:
                nxt_logits[s] = -1e9
            if temperature and temperature > 0:
                x = nxt_logits / temperature
                x -= x.max()
                p = np.exp(x)
                p /= p.sum()
                nxt = int(rng.choice(len(p), p=p))
            else:
                nxt = int(nxt_logits.argmax())
            if nxt in EOS:
                break
            out.append(nxt)
            lg, caches = model(np.array([[nxt]], np.int64), caches=caches,
                               position_offset=off)
            off += 1
            nxt_logits = lg.float().numpy()[0, -1]
    dt = time.perf_counter() - t0
    txt = tok.decode(out, skip_special_tokens=True)
    print(f"  req: {len(ids)} in (mounted {mounted}), {len(out)} out, "
          f"prefill {t_prefill:.1f}s, total {dt:.1f}s "
          f"({len(out) / max(dt - t_prefill, 1e-9):.1f} tok/s)", flush=True)
    return txt, len(ids), len(out), dt


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/api/tags":
            body = json.dumps({"models": [
                {"name": "gemma4:12b-apa-grm", "model": "gemma4:12b-apa-grm"}
            ]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/api/generate":
            self.send_response(404)
            self.end_headers()
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        opts = req.get("options", {})
        txt, n_in, n_out, dt = generate(
            req.get("prompt", ""), float(opts.get("temperature", 0.0)),
            int(opts.get("num_predict", 1024)))
        body = json.dumps({
            "model": req.get("model", "gemma4:12b-apa-grm"),
            "response": txt, "done": True,
            "prompt_eval_count": n_in, "eval_count": n_out,
            "eval_duration": int(dt * 1e9),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


print(f"Gemma4 APA-GRM shim serving on 127.0.0.1:{PORT}", flush=True)
HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
