"""DeepSeek-V2-Lite GRM full live regression gate.

This gate is intentionally explicit, not pytest-discovered. It loads the real
DeepSeek-V2-Lite INT4 TensorCUDA model and validates the longer GRM runtime
path: multi-document repository, longer live history, forced RAM paging,
native routing, open-ended greedy recalls, flush, and fresh-process resume.

Run:

  PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
  python3 tests/deepseek_grm_full_gate.py \
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
    ("ORION dossier. The launch bay for ORION is Bay-17. The clearance code "
     "for ORION is O17-4821. The responsible officer is Mira Keene.",
     ("o17-4821", "bay-17", "mira keene")),
    ("VEGA dossier. The launch bay for VEGA is Bay-22. The clearance code "
     "for VEGA is V22-9140. The responsible officer is Tomas Ivar.",
     ("v22-9140", "bay-22", "tomas ivar")),
    ("LYRA dossier. The launch bay for LYRA is Bay-05. The clearance code "
     "for LYRA is L05-3378. The responsible officer is Anika Ro.",
     ("l05-3378", "bay-05", "anika ro")),
    ("NOVA dossier. The launch bay for NOVA is Bay-31. The clearance code "
     "for NOVA is N31-6502. The responsible officer is Dax Ren.",
     ("n31-6502", "bay-31", "dax ren")),
    ("HELIX dossier. The launch bay for HELIX is Bay-44. The clearance code "
     "for HELIX is H44-2088. The responsible officer is Sera Vale.",
     ("h44-2088", "bay-44", "sera vale")),
]

LIVE_TURNS = [
    ("Record a transient operator note: cafeteria service moved to 12:20.",
     "Recorded. Cafeteria service moved to 12:20."),
    ("Record a transient operator note: staging display color is amber.",
     "Recorded. The staging display color is amber."),
    ("Record a transient operator note: courier badge color is violet.",
     "Recorded. The courier badge color is violet."),
    ("Record a transient operator note: the elevator bank is west.",
     "Recorded. The elevator bank is west."),
    ("Record a transient operator note: archive shelf marker is Q-19.",
     "Recorded. Archive shelf marker is Q-19."),
    ("Record a transient operator note: the temporary room is Cedar.",
     "Recorded. The temporary room is Cedar."),
    ("Record a transient operator note: tonight's desk lead is Nolan.",
     "Recorded. Tonight's desk lead is Nolan."),
    ("Record a transient operator note: the test printer is offline.",
     "Recorded. The test printer is offline."),
]

PROBES = [
    ("What is the exact ORION clearance code?",
     ("o17-4821",)),
    ("What is the exact VEGA clearance code?",
     ("v22-9140",)),
    ("For LYRA, what launch bay is listed?",
     ("bay-05",)),
    ("What is the exact HELIX clearance code?",
     ("h44-2088",)),
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/tmp/deepseek_grm_full_gate")
    ap.add_argument("--mode", choices=("both", "build", "resume"),
                    default="both")
    ap.add_argument("--native-lib", default=None)
    ap.add_argument("--route-layer", type=int, default=3)
    ap.add_argument("--arena-width", type=int, default=128)
    ap.add_argument("--max-live", type=int, default=512)
    ap.add_argument("--live-turns", type=int, default=0)
    ap.add_argument("--vram-budget-mb", type=float, default=2.0)
    ap.add_argument("--ngen", type=int, default=48)
    ap.add_argument("--keep-probe-cache", action="store_true")
    return ap.parse_args()


def make_repo(model, tok, args):
    enc = lambda text: tok.encode(text).ids
    dec = lambda ids: tok.decode(ids)
    return GraftRepository(
        model, enc, dec, args.repo, autosave=False,
        arena_cls=DeepSeekMLAArenaCache, route_layer=args.route_layer,
        arena_width=args.arena_width, max_live=args.max_live, topk=1,
        live_turns=args.live_turns, native_lib_path=args.native_lib,
        vram_budget_mb=args.vram_budget_mb)


def assert_native_stats(stats, args, min_nodes, min_tensors):
    if not args.native_lib:
        return
    assert stats["native"]["nodes"] == stats["nodes"]
    assert stats["native"]["durable_nodes"] == stats["durable_nodes"]
    assert stats["native"]["host_payload_tensors"] >= min_tensors
    assert stats["native"]["route_entries"] >= min_nodes


def assert_budget_applied(stats, args):
    if args.vram_budget_mb is None:
        return
    if stats["nodes"] <= 1:
        return
    if stats["active_device"] >= stats["nodes"]:
        raise SystemExit(
            "VRAM budget was not applied; every graft is device-resident")


def populate_repo(repo):
    for text, _ in DOCS:
        idx = repo.add_document(text, tags=("deepseek", "full"))
        print(f"doc {idx}: ntok={repo.arena.grafts[idx]['ntok']} "
              f"device={repo.arena.grafts[idx].get('h') is not None}",
              flush=True)
    for user, assistant in LIVE_TURNS:
        repo.add_turn(user, assistant)
        print(f"fed turn: nodes={len(repo.arena.grafts)} "
              f"active_device={repo.stats()['active_device']} "
              f"live={sum(n for _, n in repo.arena.live_segs)}",
              flush=True)


def run_probes(repo, args):
    hits = 0
    start_page_ins = getattr(repo.arena, "page_ins", 0)
    for q, accepts in PROBES:
        t0 = time.perf_counter()
        ans, meta = repo.chat(q, ngen=args.ngen, max_trips=2)
        ans_l = ans.lower()
        ok = all(a in ans_l for a in accepts)
        hits += int(ok)
        backend = getattr(repo.arena, "last_route_backend", "unknown")
        print(f"probe {'HIT' if ok else 'MISS'} backend={backend} "
              f"mounts={meta['mounts']} resident={meta['resident']} "
              f"evicted={meta['evicted']} active_device={repo.stats()['active_device']} "
              f"{time.perf_counter() - t0:.2f}s | {ans[:140]!r}",
              flush=True)
        if args.native_lib and backend != "native":
            raise SystemExit(f"expected native route backend, got {backend}")
        if not args.keep_probe_cache:
            repo.arena.reset_live_cache()
            repo._page()
    if hits != len(PROBES):
        raise SystemExit(f"DeepSeek full recall failed: {hits}/{len(PROBES)}")
    if getattr(repo.arena, "page_ins", 0) <= start_page_ins:
        raise SystemExit("DeepSeek full gate did not exercise page-in path")


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
    model.extend_rope(4096)
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
    assert st["dirty_nodes"] == 0
    assert st["durable_nodes"] == st["nodes"]
    assert st["page_ins"] > 0 or args.mode == "build"
    assert_native_stats(st, args, min_nodes=len(DOCS), min_tensors=2)

    reloaded = make_repo(model, tok, args)
    rst = reloaded.stats()
    print(f"reload stats: {rst}", flush=True)
    assert rst["nodes"] == st["nodes"]
    assert rst["durable_nodes"] == st["durable_nodes"]
    assert_budget_applied(rst, args)
    assert_native_stats(rst, args, min_nodes=len(DOCS), min_tensors=2)

    if args.mode == "build":
        print("DEEPSEEK FULL BUILD: PASS", flush=True)
    elif args.mode == "resume":
        print("DEEPSEEK FULL RESUME GATE: PASS", flush=True)
    else:
        print("DEEPSEEK FULL GATE: PASS", flush=True)


if __name__ == "__main__":
    main()
