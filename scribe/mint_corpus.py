"""Mint the SCRIBE Phase-1 seed set from the local corpus inventory
(2026-06-11 sweep). Chunks docs to <=1024 target tokens, mints each chunk
through the frozen teacher, tags domain, splits BY DOCUMENT (protocol:
held-out documents), and holds out the ENTIRE math domain (collatz) for
the distribution-shift door.

Logged composition gap: no general-web text exists locally — all domains
are writing-adjacent. Recorded per §3/§7.

Usage: python3 -m scribe.mint_corpus /mnt/ForgeRealm/scribe_mint_v1
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SOURCES = {
    "narrative": ["/mnt/Shared/StoryForge_The_Last_Thing_Wearing_Your_Fa/*"],
    "research":  ["/mnt/ForgeRealm/RCFT - Evolution Research/**/*"],
    "math":      ["/mnt/ForgeRealm/collatz-experimental-data/**/*"],
    "technical": ["/mnt/ForgeRealm/GraftRepository/docs/*.md",
                  "/mnt/ForgeRealm/GRAPA-Native-LLM/docs/*.md",
                  "/mnt/Shared/Hallucination as Epistemic Backfill*.md",
                  "/mnt/Shared/APA-Quant_CrossModel_Results.md",
                  "/mnt/Shared/Cache Economics*.md"],
}
HELDOUT_DOMAIN = "math"          # entire domain held out (§7 dist shift)
HELDOUT_EVERY = 10               # within-domain: every Nth DOC held out
CHUNK_TOKENS = 1024
MIN_CHUNK_TOKENS = 64


def doc_texts():
    for domain, globs in SOURCES.items():
        paths = []
        for g in globs:
            paths += [p for p in glob.glob(g, recursive=True)
                      if os.path.isfile(p)
                      and os.path.splitext(p)[1].lower() in (".md", ".txt")]
        for di, p in enumerate(sorted(paths)):
            try:
                text = open(p, errors="ignore").read()
            except OSError:
                continue
            if len(text) < 400:
                continue
            split = ("heldout_domain" if domain == HELDOUT_DOMAIN else
                     "heldout" if di % HELDOUT_EVERY == HELDOUT_EVERY - 1
                     else "train")
            yield domain, p, text, split


def main(root):
    import tensor_cuda as tc                              # noqa: F401
    from core.minicpm3_tc import MiniCPM3_TC, _snap
    from scribe.mint import Minter
    from tokenizers import Tokenizer as HFTok

    tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
    m, info = MiniCPM3_TC.from_pretrained()
    m.extend_rope(4096)
    print(f"loaded: {info}", flush=True)
    minter = Minter(m, lambda t: tok.encode(t).ids, root)

    done = {r["sha"] for r in minter.rows()}              # resumable
    n_chunks, n_tokens = 0, 0
    for domain, path, text, split in doc_texts():
        ids = tok.encode(text).ids
        for ci in range(0, len(ids), CHUNK_TOKENS):
            piece = ids[ci:ci + CHUNK_TOKENS]
            if len(piece) < MIN_CHUNK_TOKENS:
                continue
            chunk_text = tok.decode(piece)
            import hashlib
            sha = hashlib.sha256(chunk_text.encode()).hexdigest()[:16]
            if sha in done:
                continue
            minter.mint(chunk_text, domain=domain, split=split)
            n_chunks += 1
            n_tokens += len(piece)
            if n_chunks % 50 == 0:
                print(f"  {n_chunks} chunks / {n_tokens} tokens "
                      f"(last: {domain} {os.path.basename(path)[:40]})",
                      flush=True)
    print(f"MINTED: {n_chunks} chunks, {n_tokens} tokens -> {root}",
          flush=True)
    by = {}
    for r in minter.rows():
        k = (r["domain"], r["split"])
        by[k] = by.get(k, 0) + r["ntok"]
    for k in sorted(by):
        print(f"  {k[0]:10s} {k[1]:14s} {by[k]:8d} tokens", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
