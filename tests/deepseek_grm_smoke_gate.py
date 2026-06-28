"""DeepSeek-V2-Lite GRM live smoke gate.

This is intentionally not a pytest file. It loads the real model and can use
~10GB VRAM, so run it explicitly:

  PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
  python3 tests/deepseek_grm_smoke_gate.py

To include the C++ host runtime mirror:

  python3 tests/deepseek_grm_smoke_gate.py \
    --native-lib /tmp/grm_runtime_build/libgrm_runtime.so

Use --repo-only to skip LM-head logit parity and only validate harvest,
DeepSeekMLAArenaCache deposit, RAM payload flush, and reload. The parity gate
compares last-token logits only to avoid a full sequence-wide vocab projection
on 12GB cards.
"""
import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensor_cuda as tc
from core import kv_graft
from core.deepseek_v2_lite_tc import (DeepSeekMLAArenaCache, DeepSeekV2Lite_TC,
                                      _snap)
from core.graft_repository import GraftRepository
from tokenizers import Tokenizer as HFTok


BRIEFING = ("CLASSIFIED BRIEFING.\nProject codename: AZURE HERON.\n"
            "Access code: 73-4412.\nContact: Dr. Selena Vask.\nEND BRIEFING.\n")
PROMPT = "User: What is the project codename?\nAssistant: The codename is"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/tmp/deepseek_grm_smoke")
    ap.add_argument("--repo-only", action="store_true")
    ap.add_argument("--route-layer", type=int, default=3)
    ap.add_argument("--arena-width", type=int, default=64)
    ap.add_argument("--max-live", type=int, default=256)
    ap.add_argument("--recall-tokens", type=int, default=32)
    ap.add_argument("--native-lib", default=None,
                    help="optional libgrm_runtime.so path for native mirror")
    return ap.parse_args()


def logits_for(model, ids, caches=None, pos=0, last_token_only=False):
    with tc.no_grad():
        lg, c = model(np.array([ids], dtype=np.int64), kv_caches=caches,
                      position_offset=pos, last_token_only=last_token_only)
    return lg.numpy()[0].astype(np.float32), c


def last_logits(model, ids):
    rows, _ = logits_for(model, ids, last_token_only=True)
    return rows[-1]


def run_logit_parity(model, tok):
    b_ids = tok.encode(BRIEFING).ids
    p_ids = tok.encode(PROMPT).ids
    ctx = last_logits(model, b_ids + p_ids)

    harv = kv_graft.harvest_kv_mla(model, b_ids)
    kv_graft.clear_injection(model)
    kv_graft.set_injection_mla(model, harv)
    graft = last_logits(model, p_ids)
    kv_graft.clear_injection(model)

    diff = float(np.max(np.abs(graft - ctx)))
    flips = int(graft.argmax() != ctx.argmax())
    print(f"G1 graft-vs-incontext max|logit diff|={diff:.4f} "
          f"top1_flips={flips}/1", flush=True)
    if flips:
        raise SystemExit("DeepSeek GRM logit parity failed")
    return harv


def run_greedy_recall(model, tok, harv, ngen):
    kv_graft.clear_injection(model)
    kv_graft.set_injection_mla(model, harv)
    ids = tok.encode(
        "User: Repeat the exact access code from the briefing.\n"
        "Assistant: The access code is").ids
    row, caches = logits_for(model, ids, last_token_only=True)
    pos = len(ids)
    out = [int(row[-1].argmax())]
    for _ in range(max(0, int(ngen) - 1)):
        row, caches = logits_for(model, [out[-1]], caches=caches, pos=pos,
                                 last_token_only=True)
        pos += 1
        out.append(int(row[-1].argmax()))
    kv_graft.clear_injection(model)
    text = tok.decode(out)
    ok = "73-4412" in text or ("73" in text and "4412" in text)
    print(f"G1b greedy recall: {'HIT' if ok else 'MISS'} | "
          f"{text.strip()[:120]!r}", flush=True)
    if not ok:
        raise SystemExit("DeepSeek GRM greedy recall failed")


def run_repo_gate(model, tok, args):
    if os.path.exists(args.repo):
        shutil.rmtree(args.repo)
    enc = lambda text: tok.encode(text).ids
    dec = lambda ids: tok.decode(ids)
    repo_kwargs = dict(
        autosave=False, arena_cls=DeepSeekMLAArenaCache,
        route_layer=args.route_layer, arena_width=args.arena_width,
        max_live=args.max_live, native_lib_path=args.native_lib)
    repo = GraftRepository(model, enc, dec, args.repo, **repo_kwargs)
    idx = repo.add_document(BRIEFING, tags=("deepseek", "smoke"))
    g = repo.arena.grafts[idx]
    assert g["host_payload"]["c"][0].shape[-1] == model.config.kv_lora_rank
    assert g["host_payload"]["kpe"][0].shape[-1] == model.config.qk_rope_head_dim
    repo.flush_now()
    st = repo.stats()
    print(f"G2 repository stats: {st}", flush=True)
    if args.native_lib:
        assert "native" in st
        assert st["native"]["nodes"] == st["nodes"]
        assert st["native"]["durable_nodes"] == st["durable_nodes"]
        assert st["native"]["host_payload_tensors"] >= 2
        assert st["native"]["route_entries"] >= 1

    reloaded = GraftRepository(model, enc, dec, args.repo, **repo_kwargs)
    rst = reloaded.stats()
    print(f"G3 reload stats: {rst}", flush=True)
    assert rst["nodes"] == st["nodes"]
    assert rst["durable_nodes"] == st["durable_nodes"]
    if args.native_lib:
        assert "native" in rst
        assert rst["native"]["nodes"] == rst["nodes"]
        assert rst["native"]["durable_nodes"] == rst["durable_nodes"]
        assert rst["native"]["host_payload_tensors"] >= 2
        assert rst["native"]["route_entries"] >= 1


def main():
    args = parse_args()
    tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
    model, info = DeepSeekV2Lite_TC.from_pretrained(
        load_lm_head=not args.repo_only, progress=True)
    print(f"loaded: {info}", flush=True)
    model.extend_rope(2048)
    if not args.repo_only:
        harv = run_logit_parity(model, tok)
        run_greedy_recall(model, tok, harv, args.recall_tokens)
    run_repo_gate(model, tok, args)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
