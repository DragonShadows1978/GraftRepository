"""SCRIBE warm-phase trainer (protocol §3 L-warm).

Stream = manifest rows in fixed order (data position = one integer row
cursor; epochs wrap explicitly — repetition is FINE here, unlike GRAPA's
corpus: the student fits a function, it does not store facts). Per-layer
error EMA is kept for the error-directed minting policy (§3 update) and
checkpoints with the run.
"""
import numpy as np

import tensor_cuda as tc
from tensor_cuda.optim import AdamW

from scribe.losses import huber_per_layer
from scribe.student import target_embeddings
from scribe import checkpoint as ckpt


class WarmTrainer:
    def __init__(self, student, target, minter, lr=3e-4, window=1024):
        self.st, self.m, self.minter = student, target, minter
        self.opt = AdamW(student.parameters(), lr=lr)
        self.window = window
        self.rows = minter.rows(split="train")
        self.cursor = 0                       # data position (one integer)
        self.layer_err = None                 # EMA per target layer

    def _batch(self, row):
        ids, c, kpe = self.minter.load_pair(row["id"])
        ids = list(ids)[:self.window]
        S = len(ids)
        truth = np.concatenate([c[:, :S], kpe[:, :S]], axis=-1)  # (L,S,288)
        emb = target_embeddings(self.m, ids)
        return emb, tc.tensor(truth[None].astype(np.float32))

    def step(self):
        row = self.rows[self.cursor % len(self.rows)]
        emb, truth = self._batch(row)
        pred = self.st(emb)
        loss, per_layer = huber_per_layer(pred, truth)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.cursor += 1
        self.layer_err = (per_layer if self.layer_err is None
                          else 0.95 * self.layer_err + 0.05 * per_layer)
        return float(loss.numpy())

    def save(self, path):
        ckpt.save(path, self.st, self.opt, data_index=self.cursor,
                  refine=0.0,            # 0.0 = warm phase
                  extra={"layer_err": None if self.layer_err is None
                         else self.layer_err.tolist()})

    def load(self, path, lr=3e-4):
        self.opt, meta = ckpt.load(path, self.st, AdamW, {"lr": lr})
        self.cursor = meta["data_index"]
        le = meta["extra"].get("layer_err")
        self.layer_err = None if le is None else np.asarray(le, np.float64)
        return meta
