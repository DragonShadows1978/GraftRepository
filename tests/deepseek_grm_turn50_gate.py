"""DeepSeek-V2-Lite GRM turn-50 needle regression gate.

This gate validates the multi-turn memory case where the live model context is
the original GRM "ephemeral boat": the arena clears the live cache each turn
and treats recency as mounts. It plants a needle in an early conversation turn,
stores 50 turn grafts with `ephemeral=True`, flushes/reloads the repository,
then asks a fresh-context open-ended greedy probe that can only succeed by
routing and mounting the stored turn graft.

Folding is disabled by default so this isolates raw off-context turn recall.
Use `--enable-folding` to stress DeepSeek consolidation separately.

Run:

  PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
  python3 tests/deepseek_grm_turn50_gate.py \
    --mode build \
    --native-lib /tmp/grm_runtime_build/libgrm_runtime.so

  PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
  python3 tests/deepseek_grm_turn50_gate.py \
    --mode resume \
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


NEEDLE_NAME = "ASTRA-NODE"
NEEDLE_CODE = "T50-7391"
NEEDLE_BAY = "Bay-38"

TOPICS = (
    "calibration ledger", "staging checklist", "courier manifest",
    "archive rotation", "operator roster", "diagnostic panel",
    "coolant survey", "badge audit", "dispatch board", "tool crib",
    "signal repeater", "maintenance shelf", "loading schedule",
    "inventory sweep", "printer queue", "elevator routing",
    "terminal health", "weather relay", "dock assignment",
    "fixture inspection", "crate label", "supply cabinet",
    "scanner timing", "message board", "key locker",
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/tmp/deepseek_grm_turn50_gate")
    ap.add_argument("--mode", choices=("both", "build", "resume"),
                    default="both")
    ap.add_argument("--native-lib", default=None)
    ap.add_argument("--route-layer", type=int, default=3)
    ap.add_argument("--arena-width", type=int, default=384)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--max-live", type=int, default=512)
    ap.add_argument("--live-turns", type=int, default=0)
    ap.add_argument("--recency-mounts", type=int, default=2)
    ap.add_argument("--no-ephemeral", action="store_true")
    ap.add_argument("--vram-budget-mb", type=float, default=2.0)
    ap.add_argument("--turns", type=int, default=50)
    ap.add_argument("--needle-turn", type=int, default=1)
    ap.add_argument("--ngen", type=int, default=48)
    ap.add_argument("--enable-folding", action="store_true")
    return ap.parse_args()


def make_repo(model, tok, args):
    enc = lambda text: tok.encode(text).ids
    dec = lambda ids: tok.decode(ids)
    repo = GraftRepository(
        model, enc, dec, args.repo, autosave=False,
        arena_cls=DeepSeekMLAArenaCache, route_layer=args.route_layer,
        arena_width=args.arena_width, max_live=args.max_live, topk=args.topk,
        live_turns=args.live_turns, ephemeral=not args.no_ephemeral,
        recency_mounts=args.recency_mounts,
        native_lib_path=args.native_lib, vram_budget_mb=args.vram_budget_mb)
    if not args.enable_folding:
        repo.TURNS_HIGH = args.turns + 1
        repo.DIGESTS_HIGH = 0
    return repo


def filler_turn(i):
    topic = TOPICS[(i - 1) % len(TOPICS)]
    marker = f"M{i:02d}-{(i * 137) % 9000:04d}"
    person = f"Operator-{chr(65 + ((i - 1) % 26))}{i:02d}"
    status = ("amber", "green", "blue", "white", "violet")[i % 5]
    user = (f"Record operations turn {i}: the {topic} checkpoint uses "
            f"marker {marker}, status {status}, and owner {person}.")
    assistant = (f"Recorded turn {i}. {topic.title()} marker {marker} is "
                 f"{status} under {person}.")
    return user, assistant


def needle_turn(i):
    user = (f"Record deep needle turn {i}: {NEEDLE_NAME} has clearance code "
            f"{NEEDLE_CODE}, launch bay {NEEDLE_BAY}, and witness Mira Keene.")
    assistant = (f"Recorded. {NEEDLE_NAME} clearance code is {NEEDLE_CODE}; "
                 f"launch bay is {NEEDLE_BAY}; witness is Mira Keene.")
    return user, assistant


def clear_live(repo):
    if hasattr(repo.arena, "reset_live_cache"):
        repo.arena.reset_live_cache()
    repo._page()


def live_tokens(repo):
    return sum(n for _, n in getattr(repo.arena, "live_segs", ()))


def assert_native_stats(stats, args):
    if not args.native_lib:
        return
    assert stats["native"]["nodes"] == stats["nodes"]
    assert stats["native"]["dirty_nodes"] == 0
    assert stats["native"]["durable_nodes"] == stats["durable_nodes"]
    assert stats["native"]["route_entries"] >= stats["nodes"]


def populate(repo, args):
    if args.turns < 2:
        raise SystemExit("--turns must be at least 2")
    if not 1 <= args.needle_turn <= args.turns:
        raise SystemExit("--needle-turn must be inside --turns")
    for i in range(1, args.turns + 1):
        user, assistant = (
            needle_turn(i) if i == args.needle_turn else filler_turn(i))
        repo.add_turn(user, assistant)
        if i == args.needle_turn or i % 10 == 0 or i == args.turns:
            st = repo.stats()
            print(f"turn {i}: nodes={st['nodes']} "
                  f"active_device={st['active_device']} "
                  f"live_tokens={live_tokens(repo)}", flush=True)


def run_probe(repo, args):
    clear_live(repo)
    if live_tokens(repo) != 0:
        raise SystemExit(f"expected empty live context, got {live_tokens(repo)}")
    start_page_ins = getattr(repo.arena, "page_ins", 0)
    q = (f"After fifty stored turns, what is the exact {NEEDLE_NAME} "
         "clearance code?")
    t0 = time.perf_counter()
    ans, meta = repo.chat(q, ngen=args.ngen, max_trips=2)
    ans_l = ans.lower()
    backend = getattr(repo.arena, "last_route_backend", "unknown")
    ok = NEEDLE_CODE.lower() in ans_l
    print(f"turn50 probe {'HIT' if ok else 'MISS'} backend={backend} "
          f"mounts={meta['mounts']} resident={meta['resident']} "
          f"evicted={meta['evicted']} active_device={repo.stats()['active_device']} "
          f"{time.perf_counter() - t0:.2f}s | {ans[:180]!r}", flush=True)
    if args.native_lib and backend != "native":
        raise SystemExit(f"expected native route backend, got {backend}")
    if getattr(repo.arena, "page_ins", 0) <= start_page_ins:
        raise SystemExit("turn-50 gate did not exercise page-in path")
    if not ok:
        raise SystemExit("turn-50 needle recall failed")


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
        populate(repo, args)
    if args.mode in ("both", "resume"):
        run_probe(repo, args)

    repo.flush_now()
    st = repo.stats()
    print(f"stats: {st}", flush=True)
    assert st["dirty_nodes"] == 0
    assert st["durable_nodes"] == st["nodes"]
    assert st["active_device"] < st["nodes"]
    assert_native_stats(st, args)

    reloaded = make_repo(model, tok, args)
    rst = reloaded.stats()
    print(f"reload stats: {rst}", flush=True)
    assert rst["nodes"] == st["nodes"]
    assert rst["durable_nodes"] == st["durable_nodes"]
    assert rst["active_device"] < rst["nodes"]
    assert_native_stats(rst, args)

    if args.mode == "build":
        print("DEEPSEEK TURN50 BUILD: PASS", flush=True)
    elif args.mode == "resume":
        print("DEEPSEEK TURN50 RESUME GATE: PASS", flush=True)
    else:
        print("DEEPSEEK TURN50 GATE: PASS", flush=True)


if __name__ == "__main__":
    main()
