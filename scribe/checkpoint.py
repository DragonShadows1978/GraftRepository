"""Checkpoint/resume (SCRIBE protocol §3 standing rule; ported
from GRAPA grapa/checkpoint.py — same engine, same gaps). A checkpoint is complete ONLY if it
restores ALL of: model weights+buffers (codebook tables and generation
ride along as buffers), Adam moments + t, LR (+ scheduler position), the
numpy global RNG (the engine has NO RNG API — inits/dropout all hit the
numpy global), the DATA-STREAM INDEX (one integer, by corpus design),
and the loss-phase position (warm vs functional).

Resume protocol (ENGINE_LEDGER.md): construct model -> load_state_dict
(replaces parameter objects) -> THEN construct the optimizer from the new
parameters -> restore moments BY PARAMETER ORDER (engine optimizers are
id(p)-keyed with no state_dict) -> restore RNG last. The LM head is tied
by op reuse (emb.weight appears once in parameters()), so order mapping
is unambiguous.

Writes are atomic (tmp + rename): a kill mid-write leaves the previous
checkpoint intact — that is the resume-from-kill gate's contract.
"""
import os
import pickle

import numpy as np

import tensor_cuda as tc


def save(path, model, opt, *, data_index, refine, scheduler=None,
         extra=None):
    blob = {
        "model": model.state_dict(),
        "adam_t": opt.t,
        "adam_m": [opt._m[id(p)].numpy() for p in opt.params],
        "adam_v": [opt._v[id(p)].numpy() for p in opt.params],
        "lr": opt.lr,
        "rng": np.random.get_state(),
        "data_index": int(data_index),
        "refine": float(refine),  # SCRIBE: loss-phase marker rides here
        "sched_t": scheduler.t if scheduler is not None else None,
        "extra": extra or {},
    }
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(blob, fh, protocol=4)
    os.replace(tmp, path)
    return path


def load(path, model, opt_cls, opt_kw=None, scheduler=None):
    """Returns (opt, meta) — meta carries data_index / refine / extra.
    Call with a FRESHLY CONSTRUCTED model; optimizer is built here."""
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    model.load_state_dict(blob["model"])
    opt = opt_cls(model.parameters(), **(opt_kw or {}))
    assert len(opt.params) == len(blob["adam_m"]), \
        "parameter count changed — checkpoint is from a different model"
    for p, m, v in zip(opt.params, blob["adam_m"], blob["adam_v"]):
        opt._m[id(p)] = tc.tensor(m)
        opt._v[id(p)] = tc.tensor(v)
    opt.t = blob["adam_t"]
    opt.lr = blob["lr"]
    if scheduler is not None and blob["sched_t"] is not None:
        scheduler.t = blob["sched_t"]
        scheduler.opt.lr = scheduler.get_lr()
    np.random.set_state(blob["rng"])
    meta = {k: blob[k] for k in ("data_index", "refine", "extra")}
    return opt, meta
