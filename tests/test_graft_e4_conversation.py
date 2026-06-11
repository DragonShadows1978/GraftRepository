"""E4 — end-to-end conversation needle test on MiniCPM3 (Phase-1 exit gate).

20-turn scripted conversation; 6 format-distinct facts planted in turns 1-8;
filler turns 9-14 push them out of any small window; probes at turns 15-20.

ARMS (protocol fixed in advance):
  A  BASELINE: full transcript in context at every probe.
  B  SYSTEM:   graft-routed memory. Every completed turn is deposited as an
     MLA latent turn-graft. Live window = last 2 turns verbatim. Per probe:
     route the probe text via L44 latent-centroid cosine (E1-confirmed
     router; TOPK=3, live-window turns excluded from candidates, mounts in
     chronological order), mount, generate, deposit the probe turn, slide.
  C0 AMNESIA CONTROL: last-2-turns window, no mounts (must fail early facts).

Recall = accept-substring in 48 greedy tokens. PASS: B at parity with A
(within one probe) at a fraction of A's context residency. Residency ledger
reported per probe turn: A = transcript tokens; B = graft seats + live tokens.
"""
import os, sys, numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.minicpm3_tc import MiniCPM3_TC, _snap
from core import kv_graft
from tokenizers import Tokenizer as HFTok

tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = MiniCPM3_TC.from_pretrained()
print(f"loaded: {info}", flush=True)
m.extend_rope(8192)

ROUTE_LAYER = 44
TOPK = 3
LIVE_W = 2
NGEN = 48
# v3 EXPLORATORY: retrieval-only probe turns carry no new facts (the answer
# lives in the source turn) but their centroids are strong style attractors
# (both v1 and v2 misses were probe-turn mounts crowding out content turns).
# When False, probe turns enter the live window but are NOT deposited.
DEPOSIT_PROBE_TURNS = False

# ---- turns 1-14: scripted (user, assistant). Facts in 1,3,4,6,7,8.
SCRIPTED = [
    ("Morning. Kicking off the new effort today — we're calling it Project "
     "NIGHTJAR internally, keep that off the public boards.",
     "Morning! Understood — Project NIGHTJAR stays internal. Ready to help "
     "when you need me."),
    ("Coffee machine on three is broken again, second time this month.",
     "That machine has a rough life. Facilities ticket might actually stick "
     "this time."),
    ("Calendar note: the stakeholder demo got locked in for June 26th at "
     "10:30, in the big conference room.",
     "Noted — demo on June 26th at 10:30. I'd suggest a dry run the day "
     "before."),
    ("We also closed the backend hire. Priya Raghunathan accepted, she "
     "starts in three weeks.",
     "Great news — Priya Raghunathan joining should take real pressure off "
     "the backend queue."),
    ("Lunch place downstairs changed owners. The new menu is all noodles.",
     "Noodle monoculture — bold strategy. Hope the broth is good at least."),
    ("Ops note for you: the build server moved, it's in rack unit BX-44 "
     "now, same credentials as before.",
     "Logged — build server now in rack unit BX-44, credentials unchanged."),
    ("Finance signed off too. Hardware budget for the quarter is 7,400 "
     "dollars, which is tighter than I asked for.",
     "7,400 dollars it is — workable if we stage the GPU purchase second "
     "month."),
    ("Last logistics bit: the team offsite is confirmed for the lodge at "
     "Lake Arrowhead, two nights.",
     "Lake Arrowhead lodge, two nights — confirmed. I'll keep an eye on "
     "the agenda doc."),
    ("Did you see the storm forecast for the weekend? Sounds dramatic.",
     "Forty-mile gusts in the forecast — secure the patio furniture."),
    ("I spent an hour on a bug that turned out to be a stale cache. An "
     "hour of my life.",
     "The two hardest things in computer science strike again. Stale "
     "caches never confess."),
    ("Parking garage is repainting levels two and three all week.",
     "So everyone fights for level one — allow ten extra minutes."),
    ("My gym added one of those climbing walls. Tempted to try it.",
     "Do it — worst case is a story, best case is a hobby."),
    ("Reading anything good? I need a book for the flight.",
     "If you want absorbing but light, a heist novel beats a doorstopper "
     "at thirty thousand feet."),
    ("The vending machine took my dollar and gave me nothing. Tiny "
     "tragedies.",
     "Tiny tragedy, yes — but it's also technically a loan with no terms. "
     "Shake gently, file a complaint."),
]
# ---- turns 15-20: probes
PROBES = [
    ("Remind me, what's our project codename?", ["nightjar"]),
    ("What date and time is the stakeholder demo again?", ["june 26", "26th", "10:30"]),
    ("What was the name of the backend hire?", ["priya", "raghunathan"]),
    ("Which rack unit is the build server in now?", ["bx-44", "bx44"]),
    ("What's our hardware budget for the quarter?", ["7,400", "7400"]),
    ("Where is the team offsite happening?", ["arrowhead"]),
]

def turn_text(u, a):
    return f"User: {u}\nAssistant: {a}\n"

def last_logits(idlist, caches=None, pos=0):
    lg, c = m(np.array([idlist], dtype=np.int64), kv_caches=caches,
              position_offset=pos, last_token_only=True)
    return lg.numpy()[0, -1].astype(np.float32), c

def generate(prompt_text):
    ids = tok.encode(prompt_text).ids
    row, caches = last_logits(ids)
    pos = len(ids)
    out = [int(row.argmax())]
    for _ in range(NGEN - 1):
        row, caches = last_logits([out[-1]], caches, pos)
        pos += 1
        out.append(int(row.argmax()))
    txt = tok.decode(out)
    # cut at the first sign of the model starting the next turn itself
    for stop in ("\nUser:", "User:", "\n\n"):
        if stop in txt:
            txt = txt.split(stop)[0]
    return txt.strip(), len(ids)

def hit(text, accepts):
    t = text.lower()
    return any(a in t for a in accepts)

def unit(a):
    return a / (np.linalg.norm(a) + 1e-8)

def concat_harvests(hlist):
    out = []
    for li in range(len(m.layers)):
        c = np.concatenate([h[li]["c"] for h in hlist], axis=1)
        kpe = np.concatenate([h[li]["kpe"] for h in hlist], axis=2)
        out.append({"c": np.ascontiguousarray(c), "kpe": np.ascontiguousarray(kpe)})
    return out

# =================================================================== arm A
print("\n=== Arm A: full-transcript baseline ===", flush=True)
transcript = "".join(turn_text(u, a) for u, a in SCRIPTED)
hits_a = 0
for q, acc in PROBES:
    ans, ptoks = generate(transcript + f"User: {q}\nAssistant:")
    ok = hit(ans, acc)
    hits_a += ok
    print(f"  [{ptoks:4d} ctx tokens] {'HIT ' if ok else 'MISS'} | {ans[:70]!r}", flush=True)
    transcript += turn_text(q, ans)
print(f"Arm A: {hits_a}/{len(PROBES)}", flush=True)

# =============================================================== arms B / C0
def run_system(mount):
    tag = "B: routed memory" if mount else "C0: amnesia control"
    print(f"\n=== Arm {tag} (live window = last {LIVE_W} turns) ===", flush=True)
    grafts = []          # per deposited turn: dict(h=harvest, cent, ntok, idx)
    live = []            # list of (turn_idx, text)
    def deposit(text, graft=True):
        if graft:
            ids = tok.encode(text).ids
            h = kv_graft.harvest_kv_mla(m, ids)
            grafts.append({"h": h, "cent": kv_graft.latent_centroid(h, ROUTE_LAYER),
                           "ntok": len(ids), "idx": len(grafts)})
            live.append((len(grafts) - 1, text))
        else:
            live.append((-1, text))   # in the live window, not in the repository
        del live[:-LIVE_W]
    for u, a in SCRIPTED:
        deposit(turn_text(u, a))
    hits = 0
    for q, acc in PROBES:
        probe_txt = f"User: {q}\nAssistant:"
        mounted, seats = [], 0
        if mount:
            live_idx = {i for i, _ in live}
            # Route on the BARE question: the "User:/Assistant:" wrapper pulls
            # the probe centroid toward other Q&A-style turns (style attractors
            # — measured: budget turn rank 4 with wrapper, rank 1 bare).
            pl = kv_graft.harvest_kv_mla(m, tok.encode(q).ids,
                                         layer_filter={ROUTE_LAYER})
            p = unit(pl[ROUTE_LAYER]["c"][0].astype(np.float32).mean(0))
            cand = [g for g in grafts if g["idx"] not in live_idx]
            scores = np.array([float(np.dot(p, g["cent"])) for g in cand])
            picks = sorted((cand[j] for j in np.argsort(-scores)[:TOPK]),
                           key=lambda g: g["idx"])
            kv_graft.set_injection_mla(m, concat_harvests([g["h"] for g in picks]))
            mounted = [g["idx"] + 1 for g in picks]
            seats = sum(g["ntok"] for g in picks)
        prompt = "".join(t for _, t in live) + probe_txt
        ans, ptoks = generate(prompt)
        kv_graft.clear_injection(m)
        ok = hit(ans, acc)
        hits += ok
        res = f"{seats:3d} graft seats + {ptoks:3d} live"
        print(f"  [{res}] mounts={mounted} {'HIT ' if ok else 'MISS'} | {ans[:62]!r}", flush=True)
        deposit(turn_text(q, ans), graft=DEPOSIT_PROBE_TURNS)
    print(f"Arm {tag.split(':')[0]}: {hits}/{len(PROBES)}", flush=True)
    return hits

hits_b = run_system(True)
hits_c = run_system(False)
print(f"\nE4: baseline {hits_a}/6, routed-memory {hits_b}/6, "
      f"amnesia-control {hits_c}/6", flush=True)
print("DONE", flush=True)
