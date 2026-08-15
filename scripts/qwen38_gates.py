#!/usr/bin/env python3
"""Qwen3.8-27B architecture and tensor_cuda parity gates.

G0 is CPU-only.  It compares independent fp32 NumPy layer math against the
installed Transformers Qwen3.5 modules for random-small layers and checkpoint
layers 0/3 at teacher-forced sequence length eight.  Both references use the
empirically adjudicated sigmoid attention output gate implemented by installed
Transformers.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.qwen38_tc import (DEFAULT_CACHE_DIR, DEFAULT_MODEL_DIR, GROUP_SIZE,
                            INT3PackCache, Qwen38Config, Qwen38_TC,
                            dequantized_cache_linear, gate_int3_fused_real_shapes,
                            gate_int3_pack, tc)


def silu(x):
    return x / (1.0 + np.exp(-x))


def rmsnorm(x, weight, eps, delta_weight):
    w = 1.0 + weight if delta_weight else weight
    return x * (np.mean(x * x, axis=-1, keepdims=True) + eps) ** -0.5 * w


def rope_tables(seq_len, rotary_dim, theta):
    inv = 1.0 / (theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32)
                           / rotary_dim))
    pos = np.arange(seq_len, dtype=np.float32)[:, None] * inv[None, :]
    emb = np.concatenate([pos, pos], axis=-1)
    return np.cos(emb).astype(np.float32)[None], np.sin(emb).astype(np.float32)[None]


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_partial_rope(x, cos, sin):
    r = cos.shape[-1]
    return np.concatenate([x[..., :r] * cos[:, None]
                           + rotate_half(x[..., :r]) * sin[:, None],
                           x[..., r:]], axis=-1)


def linear(x, module):
    return x @ module.weight.detach().cpu().float().numpy().T


def numpy_deltanet(x, module):
    """Independent fp32 recurrence matching this repo's GatedDeltaNetTC."""
    B, L, _ = x.shape
    H = module.num_v_heads
    Hk = module.num_k_heads
    Dk = module.head_k_dim
    Dv = module.head_v_dim
    K = Hk * Dk
    mixed = linear(x, module.in_proj_qkv)
    z = linear(x, module.in_proj_z).reshape(B, L, H, Dv)
    a = linear(x, module.in_proj_a)
    b = linear(x, module.in_proj_b)
    cw = module.conv1d.weight.detach().cpu().float().numpy()[:, 0, :]
    pad = np.zeros((B, module.conv_kernel_size - 1, mixed.shape[-1]), np.float32)
    raw = np.concatenate([pad, mixed], axis=1)
    conv = np.zeros_like(mixed)
    for j in range(module.conv_kernel_size):
        conv += raw[:, j:j + L] * cw[:, j]
    conv = silu(conv)
    q = conv[..., :K].reshape(B, L, Hk, Dk)
    k = conv[..., K:2 * K].reshape(B, L, Hk, Dk)
    v = conv[..., 2 * K:].reshape(B, L, H, Dv)
    rep = H // Hk
    q = np.repeat(q, rep, axis=2)
    k = np.repeat(k, rep, axis=2)
    q = q * (np.sum(q * q, axis=-1, keepdims=True) + 1e-6) ** -0.5
    k = k * (np.sum(k * k, axis=-1, keepdims=True) + 1e-6) ** -0.5
    q *= Dk ** -0.5
    beta = 1.0 / (1.0 + np.exp(-b))
    A_log = module.A_log.detach().cpu().float().numpy()
    dt_bias = module.dt_bias.detach().cpu().float().numpy()
    g = -np.exp(A_log) * np.logaddexp(0.0, a + dt_bias)
    state = np.zeros((B, H, Dk, Dv), np.float32)
    outs = []
    for t in range(L):
        state *= np.exp(g[:, t])[:, :, None, None]
        kt = k[:, t]
        qt = q[:, t]
        memory = np.sum(state * kt[:, :, :, None], axis=2)
        delta = (v[:, t] - memory) * beta[:, t, :, None]
        state += kt[:, :, :, None] * delta[:, :, None, :]
        outs.append(np.sum(state * qt[:, :, :, None], axis=2))
    out = np.stack(outs, axis=1)
    norm_w = module.norm.weight.detach().cpu().float().numpy()
    out = rmsnorm(out, norm_w, module.layer_norm_epsilon, delta_weight=False)
    out *= silu(z)
    return linear(out.reshape(B, L, H * Dv), module.out_proj)


def numpy_attention(x, module, cos, sin):
    B, L, _ = x.shape
    H = module.config.num_attention_heads
    KV = module.config.num_key_value_heads
    D = module.head_dim
    qg = linear(x, module.q_proj).reshape(B, L, H, 2 * D)
    q, gate = np.split(qg, 2, axis=-1)
    gate = gate.reshape(B, L, H * D)
    q = rmsnorm(q, module.q_norm.weight.detach().cpu().float().numpy(),
                module.q_norm.eps, delta_weight=True).transpose(0, 2, 1, 3)
    k = linear(x, module.k_proj).reshape(B, L, KV, D)
    k = rmsnorm(k, module.k_norm.weight.detach().cpu().float().numpy(),
                module.k_norm.eps, delta_weight=True).transpose(0, 2, 1, 3)
    v = linear(x, module.v_proj).reshape(B, L, KV, D).transpose(0, 2, 1, 3)
    q = apply_partial_rope(q, cos, sin)
    k = apply_partial_rope(k, cos, sin)
    rep = H // KV
    k = np.repeat(k, rep, axis=1)
    v = np.repeat(v, rep, axis=1)
    scores = (q @ k.transpose(0, 1, 3, 2)) * D ** -0.5
    mask = np.triu(np.full((L, L), -np.inf, np.float32), 1)
    scores += mask[None, None]
    scores -= np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    out = (probs @ v).transpose(0, 2, 1, 3).reshape(B, L, H * D)
    out *= 1.0 / (1.0 + np.exp(-gate))
    return linear(out, module.o_proj)


def torch_attention_reference(module, x, cos, sin):
    """Run the installed Transformers attention, including its sigmoid gate."""
    import torch

    L = x.shape[1]
    mask = torch.triu(torch.full((L, L), -torch.inf, dtype=x.dtype), diagonal=1)
    out, _ = module(
        hidden_states=x,
        position_embeddings=(cos, sin),
        attention_mask=mask[None, None],
        past_key_values=None,
    )
    return out


def numpy_mlp(x, module):
    return linear(silu(linear(x, module.gate_proj)) * linear(x, module.up_proj),
                  module.down_proj)


def numpy_layer(x, layer, cos, sin):
    h = rmsnorm(x, layer.input_layernorm.weight.detach().cpu().float().numpy(),
                layer.input_layernorm.eps, delta_weight=True)
    if layer.layer_type == "linear_attention":
        mixed = numpy_deltanet(h, layer.linear_attn)
    else:
        mixed = numpy_attention(h, layer.self_attn, cos, sin)
    residual = x + mixed
    h2 = rmsnorm(residual,
                 layer.post_attention_layernorm.weight.detach().cpu().float().numpy(),
                 layer.post_attention_layernorm.eps, delta_weight=True)
    return residual + numpy_mlp(h2, layer.mlp), mixed


def torch_layer_reference(x, layer, cos, sin):
    h = layer.input_layernorm(x)
    if layer.layer_type == "linear_attention":
        mixed = layer.linear_attn(hidden_states=h, cache_params=None,
                                  attention_mask=None)
    else:
        mixed = torch_attention_reference(layer.self_attn, h, cos, sin)
    residual = x + mixed
    return residual + layer.mlp(layer.post_attention_layernorm(residual)), mixed


def torch_recurrent_accept_kwargs(*args, **kwargs):
    """Adapter for the chunk-call-only cu_seqlens keyword in HF forward."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule
    kwargs.pop("cu_seqlens", None)
    return torch_recurrent_gated_delta_rule(*args, **kwargs)


def rel_metrics(got, expected):
    got = np.asarray(got, np.float64)
    expected = np.asarray(expected, np.float64)
    diff = got - expected
    got_flat = got.reshape(-1)
    expected_flat = expected.reshape(-1)
    return {
        "rel_l2": float(np.linalg.norm(diff) / (np.linalg.norm(expected) + 1e-30)),
        "cosine": float(np.dot(got_flat, expected_flat) /
                        ((np.linalg.norm(got_flat) * np.linalg.norm(expected_flat))
                         + 1e-30)),
        "max_abs": float(np.max(np.abs(diff))),
        "finite": bool(np.isfinite(got).all()),
    }


def small_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, linear_num_key_heads=2, linear_num_value_heads=6,
        linear_key_head_dim=4, linear_value_head_dim=4,
        linear_conv_kernel_dim=4, layer_types=["linear_attention"] * 3
        + ["full_attention"], rms_norm_eps=1e-6, hidden_act="silu",
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.5,
                         "mrope_section": [1, 1, 0]},
        dtype="float32")
    return cfg


def load_real_layer(model_dir, layer_idx):
    import torch
    from safetensors import safe_open
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    with open(os.path.join(model_dir, "config.json")) as fh:
        raw = json.load(fh)
    cfg = Qwen3_5TextConfig(**raw["text_config"])
    layer = Qwen3_5DecoderLayer(cfg, layer_idx).eval()
    prefix = f"model.language_model.layers.{layer_idx}."
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as fh:
        where = json.load(fh)["weight_map"]
    by_shard = {}
    for local_name, _ in list(layer.named_parameters()):
        source = prefix + local_name
        by_shard.setdefault(where[source], []).append((local_name, source))
    for shard, names in by_shard.items():
        with safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu") as fh:
            for local_name, source in names:
                parent = layer
                parts = local_name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], torch.nn.Parameter(
                    fh.get_tensor(source).float(), requires_grad=False))
    if layer.layer_type == "linear_attention":
        layer.linear_attn.chunk_gated_delta_rule = torch_recurrent_accept_kwargs
        layer.linear_attn.recurrent_gated_delta_rule = torch_recurrent_accept_kwargs
    return cfg, layer


def run_case(label, cfg, layer, x):
    import torch
    layer.eval()
    if layer.layer_type == "linear_attention":
        layer.linear_attn.chunk_gated_delta_rule = torch_recurrent_accept_kwargs
    rotary_dim = int(cfg.head_dim * cfg.rope_parameters.get("partial_rotary_factor", 1.0))
    cos_np, sin_np = rope_tables(x.shape[1], rotary_dim,
                                 float(cfg.rope_parameters["rope_theta"]))
    cos = torch.from_numpy(cos_np)
    sin = torch.from_numpy(sin_np)
    with torch.no_grad():
        expected_layer, expected_mixer = torch_layer_reference(
            torch.from_numpy(x), layer, cos, sin)
    got_layer, got_mixer = numpy_layer(x, layer, cos_np, sin_np)
    result = {
        "label": label,
        "layer_type": layer.layer_type,
        "shape": list(x.shape),
        "mixer": rel_metrics(got_mixer, expected_mixer.numpy()),
        "block": rel_metrics(got_layer, expected_layer.numpy()),
    }
    if (not result["mixer"]["finite"] or result["mixer"]["rel_l2"] > 1e-4
            or result["block"]["rel_l2"] > 1e-4):
        raise AssertionError("G0 parity FAIL " + json.dumps(result, sort_keys=True))
    print("G0_CASE " + json.dumps(result, sort_keys=True), flush=True)
    return result


def g0(model_dir=DEFAULT_MODEL_DIR):
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
    rng = np.random.default_rng(3803)
    result = {"pack": gate_int3_pack(), "cases": []}
    print("G0_PACK " + json.dumps(result["pack"], sort_keys=True), flush=True)

    cfg = small_config()
    torch.manual_seed(3803)
    for idx in (0, 3):
        layer = Qwen3_5DecoderLayer(cfg, idx).eval().float()
        x = (rng.standard_normal((2, 8, cfg.hidden_size)).astype(np.float32) * 0.1)
        result["cases"].append(run_case(f"random-small-layer-{idx}", cfg, layer, x))
        del layer, x

    for idx in (0, 3):
        cfg_real, layer = load_real_layer(model_dir, idx)
        x = (rng.standard_normal((1, 8, cfg_real.hidden_size)).astype(np.float32)
             * 0.05)
        result["cases"].append(run_case(f"real-weight-layer-{idx}",
                                         cfg_real, layer, x))
        del layer, x
        gc.collect()
    result["status"] = "PASS"
    result["threshold_rel_l2"] = 1e-4
    result["transformers_source"] = (
        "/home/vader/.local/lib/python3.12/site-packages/transformers/"
        "models/qwen3_5/modeling_qwen3_5.py")
    print("G0_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return result


def _bf16_round(x):
    import torch
    return torch.from_numpy(np.asarray(x, np.float32)).to(torch.bfloat16).float().numpy()


def _cache_layer_numpy(x, cfg, cache, manifest, layer_idx, small_weights):
    """G2 fp32 oracle using dequantized cache weights and real small tensors."""
    p = f"model.language_model.layers.{layer_idx}"
    norm1 = small_weights[f"{p}.input_layernorm.weight"]
    norm2 = small_weights[f"{p}.post_attention_layernorm.weight"]
    h = rmsnorm(x, norm1, cfg.rms_norm_eps, delta_weight=True)
    # Reuse the G0 math by a small linear-dispatch implementation here.
    qlin = lambda a, suffix: dequantized_cache_linear(a, cache, f"{p}.{suffix}", manifest)
    B, L, _ = h.shape
    if cfg.is_attention(layer_idx):
        H, KV, D, R = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.partial_rotary_dim
        qg = qlin(h, "self_attn.q_proj.weight").reshape(B, L, H, 2 * D)
        q, gate = np.split(qg, 2, axis=-1)
        q = rmsnorm(q, small_weights[f"{p}.self_attn.q_norm.weight"],
                    cfg.rms_norm_eps, True).transpose(0, 2, 1, 3)
        k = qlin(h, "self_attn.k_proj.weight").reshape(B, L, KV, D)
        k = rmsnorm(k, small_weights[f"{p}.self_attn.k_norm.weight"],
                    cfg.rms_norm_eps, True).transpose(0, 2, 1, 3)
        v = qlin(h, "self_attn.v_proj.weight").reshape(B, L, KV, D).transpose(0, 2, 1, 3)
        cos, sin = rope_tables(L, R, cfg.rope_theta)
        q, k = apply_partial_rope(q, cos, sin), apply_partial_rope(k, cos, sin)
        rep = H // KV
        k, v = np.repeat(k, rep, 1), np.repeat(v, rep, 1)
        score = q @ k.transpose(0, 1, 3, 2) * D ** -0.5
        score += np.triu(np.full((L, L), -np.inf, np.float32), 1)[None, None]
        score -= score.max(-1, keepdims=True)
        prob = np.exp(score); prob /= prob.sum(-1, keepdims=True)
        mixed = (prob @ v).transpose(0, 2, 1, 3).reshape(B, L, H * D)
        mixed *= 1.0 / (1.0 + np.exp(-gate.reshape(B, L, H * D)))
        mixed = qlin(mixed, "self_attn.o_proj.weight")
    else:
        H, Hk, Dk, Dv = cfg.n_v_heads, cfg.n_k_heads, cfg.d_k, cfg.d_v
        K = Hk * Dk
        raw = qlin(h, "linear_attn.in_proj_qkv.weight")
        z = qlin(h, "linear_attn.in_proj_z.weight").reshape(B, L, H, Dv)
        a = h @ small_weights[f"{p}.linear_attn.in_proj_a.weight"].T
        b = h @ small_weights[f"{p}.linear_attn.in_proj_b.weight"].T
        cw = small_weights[f"{p}.linear_attn.conv1d.weight"].reshape(-1, cfg.conv_kernel)
        padded = np.concatenate([np.zeros((B, cfg.conv_kernel - 1, raw.shape[-1]), np.float32), raw], 1)
        conv = sum(padded[:, j:j + L] * cw[:, j] for j in range(cfg.conv_kernel))
        conv = silu(conv)
        q = np.repeat(conv[..., :K].reshape(B, L, Hk, Dk), H // Hk, 2)
        k = np.repeat(conv[..., K:2*K].reshape(B, L, Hk, Dk), H // Hk, 2)
        v = conv[..., 2*K:].reshape(B, L, H, Dv)
        q *= (np.sum(q*q, -1, keepdims=True) + 1e-6) ** -0.5 * Dk ** -0.5
        k *= (np.sum(k*k, -1, keepdims=True) + 1e-6) ** -0.5
        beta = 1/(1+np.exp(-b))
        g = -np.exp(small_weights[f"{p}.linear_attn.A_log"]) * np.logaddexp(
            0, a + small_weights[f"{p}.linear_attn.dt_bias"])
        state = np.zeros((B,H,Dk,Dv),np.float32); outs=[]
        for t in range(L):
            state *= np.exp(g[:,t])[:,:,None,None]
            mem=np.sum(state*k[:,t,:,:,None],2)
            delta=(v[:,t]-mem)*beta[:,t,:,None]
            state += k[:,t,:,:,None]*delta[:,:,None,:]
            outs.append(np.sum(state*q[:,t,:,:,None],2))
        mixed=np.stack(outs,1)
        mixed=rmsnorm(mixed,small_weights[f"{p}.linear_attn.norm.weight"],cfg.rms_norm_eps,False)*silu(z)
        mixed=qlin(mixed.reshape(B,L,H*Dv),"linear_attn.out_proj.weight")
    residual=x+mixed
    h2=rmsnorm(residual,norm2,cfg.rms_norm_eps,True)
    mlp=silu(qlin(h2,"mlp.gate_proj.weight"))*qlin(h2,"mlp.up_proj.weight")
    return residual+qlin(mlp,"mlp.down_proj.weight")


def g2b(model_dir=DEFAULT_MODEL_DIR, cache_dir=DEFAULT_CACHE_DIR):
    from safetensors import safe_open
    rng=np.random.default_rng(3804)
    model, _ = Qwen38_TC.from_pretrained(model_dir, cache_dir)
    cfg=model.config; cache=INT3PackCache(model_dir,cache_dir); manifest=cache.load_manifest()
    with open(os.path.join(model_dir,"model.safetensors.index.json")) as fh:
        where=json.load(fh)["weight_map"]
    results=[]
    for idx in (0,3):
        p=f"model.language_model.layers.{idx}"
        names=[n for n in where if n.startswith(p) and not cache._is_qlinear(n)]
        small={}
        for n in names:
            with safe_open(os.path.join(model_dir,where[n]),framework="pt",device="cpu") as fh:
                small[n]=fh.get_tensor(n).float().numpy()
        x=_bf16_round(rng.standard_normal((1,8,cfg.hidden_dim)).astype(np.float32)*0.05)
        ref=_cache_layer_numpy(x,cfg,cache,manifest,idx,small)
        with tc.no_grad():
            xt=tc.tensor(x).astype("bfloat16")
            got,_=model.layers[idx](xt,model.rope_cos,model.rope_sin,0,None)
            got=got.float().numpy()
        metrics=rel_metrics(got,ref)
        metrics["layer"]=idx
        print("G2B_LAYER "+json.dumps(metrics,sort_keys=True),flush=True)
        if not metrics["finite"] or metrics["cosine"] < 0.999:
            raise AssertionError("G2b parity FAIL "+json.dumps(metrics,sort_keys=True))
        results.append(metrics)
    print("G2B_RESULT "+json.dumps({"status":"PASS","cosine_threshold":0.999,
                                    "layers":results},sort_keys=True))
    return results


# Backward-compatible name for any A1 command lines; A2 registers cosine G2b.
g2 = g2b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--g0", action="store_true")
    ap.add_argument("--g2", action="store_true")
    ap.add_argument("--g2b", action="store_true")
    ap.add_argument("--fused-int3", action="store_true")
    args = ap.parse_args()
    if not (args.g0 or args.g2 or args.g2b or args.fused_int3):
        ap.error("select --g0, --g2b, or --fused-int3")
    if args.g0:
        g0(args.model_dir)
    if args.fused_int3:
        gate_int3_fused_real_shapes()
    if args.g2 or args.g2b:
        g2b(args.model_dir, args.cache_dir)


if __name__ == "__main__":
    main()
