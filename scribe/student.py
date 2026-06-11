"""SCRIBE students (protocol §2).

ARM-L  — the floor: per-layer LINEAR maps from the target's (frozen)
         token embeddings to each layer's (c, kpe). If this passes
         gates, the problem is easier than believed — a finding.
ARM-S  — small causal transformer from scratch + per-layer heads.
         Input featurizer = the TARGET's frozen embedding table (dial,
         logged): the student learns the dialect, not a new tokenizer.

Both predict pre-RoPE latents EXACTLY as the harvest path stores them:
c (S, 256) + kpe (S, 32) per layer — artifacts drop into the existing
mount machinery unchanged.
"""
import math

import numpy as np

import tensor_cuda as tc
from tensor_cuda import nn
from tensor_cuda import functional as F


class StudentConfig:
    n_target_layers = 62
    c_dim = 256
    kpe_dim = 32
    emb_dim = 2560            # MiniCPM3 hidden (frozen embedding input)
    # ARM-S trunk dials (final values set by the Phase-0 fit probe)
    d_model = 512
    n_layers = 8
    n_heads = 8
    rope_base = 10000.0
    eps = 1e-6
    window = 1024             # student context for minting chunks

    @property
    def out_dim(self):
        return self.c_dim + self.kpe_dim                       # 288


class ArmL(nn.Module):
    """Linear floor: one (emb_dim -> 288) map per target layer, shared
    input = frozen target embedding of each token (position-free)."""

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or StudentConfig()
        c = self.cfg
        self.heads = nn.ModuleDict(
            {f"h{li}": nn.Linear(c.emb_dim, c.out_dim, bias=True)
             for li in range(c.n_target_layers)})

    def forward(self, emb):
        """emb: (B, S, emb_dim) frozen target embeddings.
        Returns (B, L_target, S, 288)."""
        c = self.cfg
        outs = [self.heads[f"h{li}"](emb).unsqueeze(1)
                for li in range(c.n_target_layers)]
        return tc.cat(outs, dim=1)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, H = cfg.d_model, cfg.n_heads
        self.Dh = d // H
        self.H = H
        self.ln1 = nn.RMSNorm(d, eps=cfg.eps)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.RMSNorm(d, eps=cfg.eps)
        self.gate = nn.Linear(d, 2 * d, bias=False)
        self.up = nn.Linear(d, 2 * d, bias=False)
        self.down = nn.Linear(2 * d, d, bias=False)
        self.scale = 1.0 / math.sqrt(self.Dh)

    def forward(self, x, cos, sin):
        B, S, d = x.shape
        h = self.ln1(x)
        q = self.q_proj(h).reshape([B, S, self.H, self.Dh]).transpose(1, 2)
        k = self.k_proj(h).reshape([B, S, self.H, self.Dh]).transpose(1, 2)
        v = self.v_proj(h).reshape([B, S, self.H, self.Dh]).transpose(1, 2)
        q = F.apply_rotary(q, cos, sin)
        k = F.apply_rotary(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           scale=self.scale)
        x = x + self.o_proj(a.transpose(1, 2).reshape([B, S, d]))
        h = self.ln2(x)
        return x + self.down(self.gate(h).silu() * self.up(h))


class ArmS(nn.Module):
    """Causal transformer student. Input = frozen target embeddings
    projected down; output = per-target-layer (c, kpe) heads."""

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or StudentConfig()
        c = self.cfg
        self.proj_in = nn.Linear(c.emb_dim, c.d_model, bias=False)
        self.blocks = nn.ModuleDict(
            {f"b{i}": Block(c) for i in range(c.n_layers)})
        self.norm_f = nn.RMSNorm(c.d_model, eps=c.eps)
        self.heads = nn.ModuleDict(
            {f"h{li}": nn.Linear(c.d_model, c.out_dim, bias=True)
             for li in range(c.n_target_layers)})

    def forward(self, emb):
        """emb: (B, S, emb_dim) -> (B, L_target, S, 288)."""
        c = self.cfg
        B, S, _ = emb.shape
        cos, sin = F.rope_tables(S, c.d_model // c.n_heads, "cuda",
                                 base=c.rope_base)
        x = self.proj_in(emb)
        for i in range(c.n_layers):
            x = self.blocks[f"b{i}"](x, cos, sin)
        x = self.norm_f(x)
        outs = [self.heads[f"h{li}"](x).unsqueeze(1)
                for li in range(c.n_target_layers)]
        return tc.cat(outs, dim=1)


def target_embeddings(target_model, ids):
    """Frozen featurizer: the target's embedding row lookup, SCALED as the
    target's layers see it (MiniCPM3 muP: x scale_emb = 12 — the raw table
    is unscaled). (1, S, emb_dim) fp32, detached — never trained."""
    idx = tc.tensor(np.asarray([ids], np.int64), dtype="int64")
    emb = tc.embedding(target_model.embed_tokens.weight, idx)
    return (emb * float(target_model.config.scale_emb)).detach().float()
