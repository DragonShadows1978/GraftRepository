"""DeepSeek-V2-Lite GRM routed arena gate.

This is intentionally not a pytest file. It loads the real model and exercises
the persistent arena path: document deposits, native RAM mirror, live turn
feed, eviction, routed swap, greedy recall, flush, and reload.

Run:

  PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
  python3 tests/deepseek_grm_arena_gate.py \
    --native-lib /tmp/grm_runtime_build/libgrm_runtime.so
"""
import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensor_cuda as tc
from core.deepseek_v2_lite_tc import DeepSeekMLAArenaCache, DeepSeekV2Lite_TC, _snap
from core.graft_repository import GraftRepository
from tokenizers import Tokenizer as HFTok


DOCS = [
    ("ROUTE-ALPHA dossier. The exact access code for ROUTE-ALPHA is "
     "A17-9402. The responsible contact is Mara Venn.", ["A17-9402"]),
    ("ROUTE-BRAVO dossier. The exact access code for ROUTE-BRAVO is "
     "B88-1120. The responsible contact is Ilya Sato.", ["B88-1120"]),
    ("ROUTE-CHARLIE dossier. The exact access code for ROUTE-CHARLIE is "
     "C54-7731. The responsible contact is Oren Vale.", ["C54-7731"]),
]

LIVE_TURNS = [
    ("Record a transient operator note: lunch moved to 12:30.",
     "Recorded. Lunch moved to 12:30."),
    ("Record a transient operator note: the dashboard color is teal.",
     "Recorded. The dashboard color is teal."),
    ("Record a transient operator note: the staging room is Delta.",
     "Recorded. The staging room is Delta."),
]

PROBES = [
    ("What is the exact access code for ROUTE-ALPHA?", ["a17-9402", "a17", "9402"]),
    ("What is the exact access code for ROUTE-BRAVO?", ["b88-1120", "b88", "1120"]),
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/tmp/deepseek_grm_arena_gate")
    ap.add_argument("--mode", choices=("both", "build", "resume"),
                    default="both")
    ap.add_argument("--native-lib", default=None)
    ap.add_argument("--route-layer", type=int, default=3)
    ap.add_argument("--arena-width", type=int, default=96)
    ap.add_argument("--max-live", type=int, default=256)
    ap.add_argument("--live-turns", type=int, default=0)
    ap.add_argument("--ngen", type=int, default=32)
    ap.add_argument("--keep-probe-cache", action="store_true")
    return ap.parse_args()


def make_repo(model, tok, args):
    enc = lambda text: tok.encode(text).ids
    dec = lambda ids: tok.decode(ids)
    return GraftRepository(
        model, enc, dec, args.repo, autosave=False,
        arena_cls=DeepSeekMLAArenaCache, route_layer=args.route_layer,
        arena_width=args.arena_width, max_live=args.max_live, topk=1,
        live_turns=args.live_turns, native_lib_path=args.native_lib)


def assert_native_stats(stats, args, min_nodes, min_tensors):
    if not args.native_lib:
        return
    assert stats["native"]["nodes"] == stats["nodes"]
    assert stats["native"]["durable_nodes"] == stats["durable_nodes"]
    assert stats["native"]["host_payload_tensors"] >= min_tensors
    assert stats["native"]["route_entries"] >= min_nodes


def populate_repo(repo):
    for text, _ in DOCS:
        idx = repo.add_document(text, tags=("deepseek", "arena"))
        print(f"doc {idx}: ntok={repo.arena.grafts[idx]['ntok']}", flush=True)

    for user, assistant in LIVE_TURNS:
        repo.add_turn(user, assistant)
        print(f"fed turn: resident={repo.arena._cache_len()} "
              f"live={sum(n for _, n in repo.arena.live_segs)}", flush=True)


def run_probes(repo, args):
    hits = 0
    for q, accepts in PROBES:
        t0 = time.perf_counter()
        ans, meta = repo.arena.step(q, ngen=args.ngen, max_trips=1,
                                    deposit=False)
        ans_l = ans.lower()
        ok = any(a in ans_l for a in accepts)
        hits += int(ok)
        backend = getattr(repo.arena, "last_route_backend", "unknown")
        print(f"probe {'HIT' if ok else 'MISS'} mounts={meta['mounts']} "
              f"backend={backend} "
              f"resident={meta['resident']} evicted={meta['evicted']} "
              f"{time.perf_counter() - t0:.2f}s | {ans[:120]!r}", flush=True)
        if args.native_lib and backend != "native":
            raise SystemExit(f"expected native route backend, got {backend}")
        if not args.keep_probe_cache:
            repo.arena.reset_live_cache()
    if hits != len(PROBES):
        raise SystemExit(f"DeepSeek arena recall failed: {hits}/{len(PROBES)}")


def main():
    args = parse_args()
    if args.mode in ("both", "build") and os.path.exists(args.repo):
        shutil.rmtree(args.repo)
    if args.mode == "resume" and not os.path.exists(
            os.path.join(args.repo, "manifest.json")):
        raise SystemExit(f"resume repo missing manifest: {args.repo}")

    tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
    model, info = DeepSeekV2Lite_TC.from_pretrained(load_lm_head=True,
                                                    progress=True)
    print(f"loaded: {info}", flush=True)
    model.extend_rope(2048)
    tc.set_alloc_pooling(True)
    model.warm_absorbed_decode()
    print("absorbed decode weights: warm", flush=True)

    repo = make_repo(model, tok, args)
    if args.mode in ("both", "build"):
        populate_repo(repo)
    if args.mode in ("both", "resume"):
        run_probes(repo, args)

    repo.flush_now()
    st = repo.stats()
    print(f"stats: {st}", flush=True)
    assert_native_stats(st, args, min_nodes=len(DOCS), min_tensors=2)
    if args.mode == "build":
        print("DEEPSEEK ARENA BUILD: PASS", flush=True)
        return

    reloaded = make_repo(model, tok, args)
    rst = reloaded.stats()
    print(f"reload stats: {rst}", flush=True)
    assert rst["nodes"] == st["nodes"]
    assert rst["durable_nodes"] == st["durable_nodes"]
    assert_native_stats(rst, args, min_nodes=len(DOCS), min_tensors=2)
    if args.mode == "resume":
        print("DEEPSEEK ARENA RESUME GATE: PASS", flush=True)
    else:
        print("DEEPSEEK ARENA GATE: PASS", flush=True)


if __name__ == "__main__":
    main()
