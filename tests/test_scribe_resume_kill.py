"""SCRIBE resume-from-kill gate (protocol §3 standing rule, Phase 0 exit).

  setup — mint 6 deterministic train docs (one-time, shared root)
  ref   — 8 straight warm steps (ARM-S), print losses
  part1 — 4 steps, checkpoint, hard-kill
  part2 — fresh process, load, steps 4..7

PASS: part1+part2 == ref bit-tight (loss trajectory + cursor + layer-err
EMA continuity).
"""
import os, sys, shutil
import numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.minicpm3_tc import MiniCPM3_TC, _snap
from scribe.mint import Minter
from scribe.student import ArmS, StudentConfig
from scribe.train import WarmTrainer
from tokenizers import Tokenizer as HFTok

ROOT, CKPT = "/tmp/scribe_resume_kill", "/tmp/scribe_resume_kill.ckpt"
TOTAL, CUT = 8, 4

tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = MiniCPM3_TC.from_pretrained()
m.extend_rope(4096)
print(f"loaded: {info}", flush=True)

DOCS = [f"LOG ENTRY {i}. Station {chr(65+i)}{i*7+3} reported nominal flow "
        f"at valve V-{100+i*13}; pressure held at {40+i*3} bar through the "
        f"shift. Inspector tag IN-{200+i*11} filed without findings."
        for i in range(6)]

minter = Minter(m, lambda t: tok.encode(t).ids, ROOT)
if not minter.rows(split="train"):
    for i, d in enumerate(DOCS):
        minter.mint(d, domain=f"d{i % 2}", split="train")
    print("minted 6 train docs", flush=True)


def build_trainer():
    np.random.seed(0)
    st = ArmS(StudentConfig())
    return WarmTrainer(st, m, minter, lr=3e-4)


mode = sys.argv[1]
if mode == "setup":
    print("DONE", flush=True)
elif mode == "ref":
    tr = build_trainer()
    for i in range(TOTAL):
        print(f"step {i} loss {tr.step():.8f}", flush=True)
    print("DONE", flush=True)
elif mode == "part1":
    tr = build_trainer()
    for i in range(CUT):
        print(f"step {i} loss {tr.step():.8f}", flush=True)
    tr.save(CKPT)
    print("SAVED; hard-killing", flush=True)
    os._exit(0)
elif mode == "part2":
    tr = build_trainer()
    meta = tr.load(CKPT)
    assert tr.cursor == CUT, tr.cursor
    assert tr.layer_err is not None
    for i in range(CUT, TOTAL):
        print(f"step {i} loss {tr.step():.8f}", flush=True)
    print("DONE", flush=True)
