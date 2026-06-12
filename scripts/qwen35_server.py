"""Ollama-API-compatible server over the APA-GRM Qwen3.5-9B port.

The corpus driver (GRAPA corpus/templates/local_wave.py) speaks
ollama's POST /api/generate with {model, prompt, stream:false,
think:false, options:{temperature, num_predict}}. This shim serves
that contract from the tensor_cuda stack — APA on the attention
layers, GRM doing real work underneath:

  GRM PREFIX MOUNT: requests share long constant prompt prefixes
  (conventions + hard rules). The hybrid state after the longest
  previously-seen token prefix is cached (the functional kernel makes
  held states branch-safe) and RESTORED instead of re-prefilled;
  only the request's variable tail pays prefill. Cache is in-memory,
  keyed by token-id prefix, LRU-capped.

Single-threaded HTTP (the driver is sequential). Default port 11435.

  python3 scripts/qwen35_server.py [port]
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
from core.qwen35_tc import Qwen35_TC, _snap                 # noqa: E402

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 11435
MAX_CACHED_PREFIXES = 8        # ~50MB states + small KV each

from transformers import AutoTokenizer                      # noqa: E402
tok = AutoTokenizer.from_pretrained(_snap())

tc.set_alloc_pooling(True)
model, info = Qwen35_TC.from_pretrained()
print(f"loaded: {info}", flush=True)


def clone_caches(caches):
    """Branch-safe copy. DeltaNet states are fp32 tensors the fused
    kernel never mutates, and attention K/V tensors are append-only
    (cat makes new tensors) — so sharing TENSORS is safe; only the
    per-layer tuple list must be fresh."""
    return [tuple(c) for c in caches]


class PrefixCache:
    def __init__(self):
        self.entries = []          # (ids tuple, caches, hits, stamp)

    def lookup(self, ids):
        """Longest cached prefix of ids (must end before len(ids) so the
        model still sees at least one new token)."""
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
        self.entries.append([key, clone_caches(caches), 0, time.time()])
        if len(self.entries) > MAX_CACHED_PREFIXES:
            self.entries.sort(key=lambda e: (e[2], e[3]))
            self.entries.pop(0)


PREFIX = PrefixCache()


def common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


LAST_IDS = []                     # previous request's ids (prefix mining)


def generate(prompt, temperature, num_predict):
    global LAST_IDS
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="np").input_ids.astype(np.int64)[0]
    t0 = time.perf_counter()

    hit = PREFIX.lookup(ids)
    with tc.no_grad():
        if hit is not None:
            caches = clone_caches(hit[1])
            off = len(hit[0])
            tail = ids[off:][None, :]
            lg, caches = model(tail, caches=caches, position_offset=off,
                               last_token_only=True)
            mounted = off
        else:
            lg, caches = model(ids[None, :], last_token_only=True)
            mounted = 0
        # mine a prefix for next time: the shared boilerplate between
        # consecutive requests (rounded down to a stable cut)
        if LAST_IDS is not None and len(LAST_IDS):
            cp = common_prefix_len(ids, LAST_IDS)
            if cp >= 64 and cp > mounted:
                with tc.no_grad():
                    _, pc = model(ids[None, :cp])
                PREFIX.store(ids[:cp], pc)
        LAST_IDS = list(ids)

        off = len(ids)
        out = []
        eos = model.config.eos_token_id
        rng = np.random.default_rng()
        nxt_logits = lg.float().numpy()[0, -1]
        t_prefill = time.perf_counter() - t0
        for _ in range(min(num_predict, 8192)):
            if temperature and temperature > 0:
                x = nxt_logits / temperature
                x -= x.max()
                p = np.exp(x)
                p /= p.sum()
                nxt = int(rng.choice(len(p), p=p))
            else:
                nxt = int(nxt_logits.argmax())
            if nxt == eos:
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
    def log_message(self, *a):                 # quiet access log
        pass

    def do_GET(self):
        if self.path == "/api/tags":
            body = json.dumps({"models": [
                {"name": "qwen3.5:9b-apa-grm", "model": "qwen3.5:9b-apa-grm"}
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
            "model": req.get("model", "qwen3.5:9b-apa-grm"),
            "response": txt, "done": True,
            "prompt_eval_count": n_in, "eval_count": n_out,
            "eval_duration": int(dt * 1e9),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


print(f"APA-GRM shim serving on 127.0.0.1:{PORT}", flush=True)
HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
