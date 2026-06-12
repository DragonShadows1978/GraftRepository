"""SCRIBE functional loss — L-func, the real loss (protocol §3.2).

Mount the student's PREDICTED cache in the frozen reader and train the
student on KL(exact-graft logits || predicted-graft logits) over probe
continuations. The equivalence harness, repurposed as the training
signal: gradients flow logits -> frozen reader forward -> injected cache
tensors -> student. The reader's weights are frozen (INT4, non-params);
only activations carry grad.

Key mechanics:
  - GRAPH-PRESERVING injection: per-layer SLICES of the student output
    are set as att.inject_kv directly (no numpy round-trip — that's what
    kv_graft.set_injection_mla does and it would cut the graph).
  - Exact teacher logits per (doc, probe) are computed once and cached
    as CONSTANTS; loss = cross-entropy of predicted logits under the
    exact distribution (KL minus a constant).
  - Probes force CONTENT through the reader (the G3 failure mode):
    identifier-readback + summary prompts.
  - Huber anchor steps interleave (the warm loss keeps the latent
    geometry from drifting while KL reshapes what the reader sees).
"""
import numpy as np

import tensor_cuda as tc

from scribe.losses import huber_per_layer
from scribe.student import target_embeddings

PROBES = (
    "User: State every identifier code mentioned above.\nAssistant:",
    "User: Summarize the key facts above in one sentence.\nAssistant:",
)


def inject_student(m, pred, S):
    """pred: student output tensor (1, 62, S, 288), graph attached.
    Slices land as inject_kv per layer; caller must clear_injection."""
    for li, layer in enumerate(m.layers):
        att = layer.self_attn
        sl = pred.slice(1, li, 1).reshape([1, S, 288])
        c = sl.slice(-1, 0, 256)                       # (1, S, 256)
        kpe = sl.slice(-1, 256, 32).reshape([1, 1, S, 32])
        att.inject_kv = (c, kpe)
        att.graft_seats = S


def clear(m):
    for layer in m.layers:
        layer.self_attn.inject_kv = None
        layer.self_attn.graft_seats = 0


class FuncTrainer:
    def __init__(self, student, target, minter, opt, encode,
                 max_doc_tokens=256, anchor_every=4):
        self.st, self.m, self.minter, self.opt = student, target, minter, opt
        self.encode = encode
        self.max_doc = max_doc_tokens
        self.anchor_every = anchor_every
        self.rows = [r for r in minter.rows(split="train")
                     if r["ntok"] <= max_doc_tokens]
        self.cursor = 0
        self._exact = {}                 # (pair id, probe idx) -> probs np
        self._pids = [self.encode(p) for p in PROBES]

    def _exact_probs(self, row, pi, ids):
        key = (row["id"], pi)
        if key not in self._exact:
            from core import kv_graft
            harv = kv_graft.harvest_kv_mla(self.m, list(ids))
            kv_graft.set_injection_mla(self.m, harv)
            with tc.no_grad():
                lg, _ = self.m(np.array([self._pids[pi]], np.int64),
                               last_token_only=False)
            kv_graft.clear_injection(self.m)
            x = lg.numpy()[0].astype(np.float32)
            x = np.exp(x - x.max(-1, keepdims=True))
            self._exact[key] = (x / x.sum(-1, keepdims=True)).astype(
                np.float32)
        return self._exact[key]

    def step(self):
        row = self.rows[self.cursor % len(self.rows)]
        ids, c, kpe = self.minter.load_pair(row["id"])
        ids = list(ids)[:self.max_doc]
        S = len(ids)
        emb = target_embeddings(self.m, ids)

        if self.cursor % self.anchor_every == self.anchor_every - 1:
            # Huber anchor step (warm loss)
            truth = np.concatenate([c[:, :S], kpe[:, :S]], axis=-1)
            pred = self.st(emb)
            loss, _ = huber_per_layer(pred, tc.tensor(
                truth[None].astype(np.float32)))
            kind = "anchor"
        else:
            pi = self.cursor % len(PROBES)
            probs = self._exact_probs(row, pi, ids)        # (P, V) const
            pred = self.st(emb)                            # graph attached
            inject_student(self.m, pred, S)
            try:
                lg, _ = self.m(np.array([self._pids[pi]], np.int64),
                               last_token_only=False)
            finally:
                clear(self.m)
            lp = lg.float().log_softmax(-1)                # (1, P, V)
            tgt = tc.tensor(probs[None])
            loss = (lp * tgt).sum([0, 1, 2], False) * (-1.0 / probs.shape[0])
            kind = "kl"

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.cursor += 1
        return kind, float(loss.numpy())
