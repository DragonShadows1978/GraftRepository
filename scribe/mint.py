"""SCRIBE pair minting — the teacher as a FUNCTION (protocol §3).

mint(text) = one prefill of the frozen target through the EXISTING
harvest path -> (token ids, per-layer pre-RoPE latents). Artifacts are
exactly what the mount machinery consumes (c 256 + k_pe 32 per token per
layer, fp16) — a predicted artifact and an exact artifact are
interchangeable downstream, which is the whole design.

Storage: one .npz per document under a mint root, plus manifest.jsonl
rows {id, sha, domain, ntok, split} — domain tags feed the
coverage-weighted/error-directed minting policy and the held-out-DOMAIN
gates (§7 distribution shift). ~36KB/token (62 layers x 288 x fp16).
"""
import hashlib
import json
import os

import numpy as np

from core import kv_graft


class Minter:
    def __init__(self, model, encode, root):
        self.m, self.encode, self.root = model, encode, root
        os.makedirs(os.path.join(root, "pairs"), exist_ok=True)
        self.manifest = os.path.join(root, "manifest.jsonl")

    def mint(self, text, domain="unspecified", split="train"):
        """One prefill -> stored (text, latents) pair. Returns pair id."""
        ids = self.encode(text)
        h = kv_graft.harvest_kv_mla(self.m, ids)
        c = np.concatenate([h[li]["c"][0].astype(np.float16)[None]
                            for li in range(len(self.m.layers))])   # (L,S,256)
        kpe = np.concatenate([h[li]["kpe"][0, 0].astype(np.float16)[None]
                              for li in range(len(self.m.layers))])  # (L,S,32)
        sha = hashlib.sha256(text.encode()).hexdigest()[:16]
        pid = f"{domain[:12]}_{sha}"
        np.savez_compressed(os.path.join(self.root, "pairs", pid + ".npz"),
                            ids=np.asarray(ids, np.int64), c=c, kpe=kpe)
        with open(self.manifest, "a") as fh:
            fh.write(json.dumps({"id": pid, "sha": sha, "domain": domain,
                                 "ntok": len(ids), "split": split,
                                 "text": text}) + "\n")
        return pid

    def load_pair(self, pid):
        z = np.load(os.path.join(self.root, "pairs", pid + ".npz"))
        return z["ids"], z["c"], z["kpe"]

    def rows(self, split=None, domain=None):
        out = []
        if not os.path.exists(self.manifest):
            return out
        for line in open(self.manifest):
            r = json.loads(line)
            if split and r["split"] != split:
                continue
            if domain and r["domain"] != domain:
                continue
            out.append(r)
        return out
