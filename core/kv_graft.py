"""KV-Graft: inject a document into a frozen model as attention K-V, off-context.

Prometheus Mind's mechanism, generalized from single-token to whole documents on
the tensor_cuda engine. Two phases:

  1. HARVEST  -- run a guide/document through the frozen model ONCE and capture,
     per layer, the key/value tensors the model itself produced for that text.
     This is the model's own representation of the document. Save it.

  2. INJECT   -- on a SEPARATE generation where the document is NOT in the prompt,
     splice the harvested per-layer K/V into each attention layer's K/V (the same
     cat() the KV-cache already uses). Attention then sees
        [ grafted_doc_kv | conversation_kv | new_tokens ]
     The grafted K/V costs ZERO context-window tokens -- it lives at the K/V
     level, below the token level. The model attends to the document through its
     native attention, as if it were always there.

This does NOT modify weights (the base model stays frozen, like PM). It mounts
the document into attention. Whether that is enough to make a small model OBEY a
580k-token rule corpus is exactly the experiment.

Engine hooks (no engine changes required for harvest; inject needs the one-line
cat() addition in GQAAttentionTC, added behind an `inject_kv` attribute so the
normal path is untouched when it's None).
"""
import os
import sys
import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc


def _attention_module(layer):
    att = getattr(layer, "self_attn", None)
    if att is not None:
        return att
    if getattr(layer, "is_attn", False):
        return getattr(layer, "mixer", None)
    return None


# --------------------------------------------------------------------- harvest
def harvest_kv(model, token_ids, layer_filter=None):
    """Run `token_ids` (1D list/array) through `model` once and capture each
    attention layer's PRE-RoPE K and V (num_kv_heads, S, head_dim) — position-
    neutral, so the graft can be re-RoPE'd at any seat range at inject time.

    Returns a list (per layer) of dicts {k: ndarray, v: ndarray} on host (numpy),
    so a whole 580k-token corpus can be harvested in chunks and stored to disk
    without pinning VRAM. layer_filter: optional set of layer indices to keep.
    """
    captured = [None] * len(model.layers)

    # Install a capture hook on each layer's attention by wrapping it. We grab the
    # k,v the layer builds (after projection + RoPE) by asking the layer to stash
    # them. To stay non-invasive we set a flag the patched __call__ honors.
    for i, layer in enumerate(model.layers):
        att = _attention_module(layer)
        if att is not None and (layer_filter is None or i in layer_filter):
            att._capture = True
            att._captured = None

    ii = np.array([token_ids], dtype=np.int64)
    with tc.no_grad():
        model(ii, last_token_only=True)

    for i, layer in enumerate(model.layers):
        att = _attention_module(layer)
        if att is not None and getattr(att, "_capture", False):
            cap = att._captured
            if cap is not None:
                captured[i] = {"k": cap[0], "v": cap[1]}
            att._capture = False
            att._captured = None
    return captured


# ----------------------------------------------------------------- MLA latent
def harvest_kv_mla(model, token_ids, layer_filter=None, max_layers=None):
    """MLA (MiniCPM3) harvest: capture each layer's position-free post-norm
    LATENT (B, S, 256) and PRE-RoPE shared rope-key (B, 1, S, 32) — the whole
    graft is 288 vals/token/layer, ~22x smaller than assembled per-head K/V.
    Returns a list (per layer) of dicts {c: ndarray, kpe: ndarray} on host.
    max_layers: early-exit partial forward (router captures only need layers
    0..route_layer — no head, no logits).
    """
    for i, layer in enumerate(model.layers):
        att = _attention_module(layer)
        if att is not None and (layer_filter is None or i in layer_filter):
            att._capture = True
            att._captured = None

    ii = np.array([token_ids], dtype=np.int64)
    with tc.no_grad():
        model(ii, last_token_only=True, max_layers=max_layers)

    captured = [None] * len(model.layers)
    for i, layer in enumerate(model.layers):
        att = _attention_module(layer)
        if att is not None and getattr(att, "_capture", False):
            cap = att._captured
            if cap is not None:
                captured[i] = {"c": cap[0], "kpe": cap[1]}
            att._capture = False
            att._captured = None
    return captured


def set_injection_mla(model, harvested, layers=None):
    """Arm MLA latent injection (prefill-only, scale-free: a key scale is not
    representable at the latent level and scale!=1.0 is a deprecated recipe
    element anyway). graft_seats = latent length, persists for cached decode.
    """
    sel = set(range(len(model.layers))) if layers is None else set(layers)
    for i, layer in enumerate(model.layers):
        att = _attention_module(layer)
        if att is None:
            continue
        if i in sel and harvested[i] is not None:
            dt = BlockDtype(model)
            c = tc.tensor(np.ascontiguousarray(harvested[i]["c"])).astype(dt)
            kpe = tc.tensor(np.ascontiguousarray(harvested[i]["kpe"])).astype(dt)
            att.inject_kv = (c, kpe)
            att.graft_seats = int(c.shape[1])
        else:
            att.inject_kv = None
            att.graft_seats = 0


def latent_centroid(harvested, layer):
    """MLA routing summary key: unit-norm mean of a graft's layer-`layer`
    latent (256 floats = 512 bytes fp16). E1-confirmed router for MiniCPM3:
    rank grafts by cos(latent_centroid(probe), latent_centroid(graft)) —
    L10 is the cost-optimal routing layer, L44 the most accurate. Key-space
    |q.k| routing FAILS on MLA (no qk-norm -> outlier-key norms dominate);
    on QK-normed GQA models (Qwen3) use layer-0 capture_queries scoring
    instead. The routing index format is part of the model's dialect.
    """
    c = harvested[layer]["c"][0].astype(np.float32).mean(0)
    return c / (np.linalg.norm(c) + 1e-8)


# --------------------------------------------------------------------- router
def capture_queries(model, token_ids, layer_filter=None):
    """Run `token_ids` through `model` once and capture each attention layer's
    PRE-RoPE post-norm queries (num_heads, L, head_dim) — the same position-
    neutral space harvest_kv captures keys in, so dot(q, k_harvest) is a
    position-free relevance score. This is the Graft-Repository router input.

    Returns a list (per layer) of ndarrays on host, None for filtered layers.
    """
    for i, layer in enumerate(model.layers):
        att = _attention_module(layer)
        if att is not None and (layer_filter is None or i in layer_filter):
            att._capture_q = True
            att._captured_q = None

    ii = np.array([token_ids], dtype=np.int64)
    with tc.no_grad():
        model(ii, last_token_only=True)

    out = [None] * len(model.layers)
    for i, layer in enumerate(model.layers):
        att = _attention_module(layer)
        if att is not None and getattr(att, "_capture_q", False):
            out[i] = att._captured_q
            att._capture_q = False
            att._captured_q = None
    return out


def harvest_kv_and_queries(model, token_ids, layer_filter=None):
    """Capture pre-RoPE K/V and pre-RoPE queries in one forward pass.

    Target-side graft translation needs both the native 9B K/V payload and the
    live 9B queries used for key-fidelity scoring. Capturing them together keeps
    corpus harvest to one model forward per chunk.
    """
    for i, layer in enumerate(model.layers):
        att = _attention_module(layer)
        if att is not None and (layer_filter is None or i in layer_filter):
            att._capture = True
            att._captured = None
            att._capture_q = True
            att._captured_q = None

    ii = np.array([token_ids], dtype=np.int64)
    with tc.no_grad():
        model(ii, last_token_only=True)

    captured = [None] * len(model.layers)
    queries = [None] * len(model.layers)
    for i, layer in enumerate(model.layers):
        att = _attention_module(layer)
        if att is None:
            continue
        if getattr(att, "_capture", False):
            cap = att._captured
            if cap is not None:
                captured[i] = {"k": cap[0], "v": cap[1]}
            att._capture = False
            att._captured = None
        if getattr(att, "_capture_q", False):
            queries[i] = att._captured_q
            att._capture_q = False
            att._captured_q = None
    return captured, queries


# --------------------------------------------------------------------- inject
def set_injection(model, harvested, layers=None, scale=1.0, scale_per_layer=None):
    """Arm injection: each selected layer's attention will cat the harvested
    (k,v) in front of its K/V on the next forward.
      layers          : iterable of layer indices to inject at (None = all).
      scale           : global K-multiplier (how strongly to attend the graft).
      scale_per_layer : optional dict {layer_idx: scale} overriding `scale` per
                        layer — to damp layers that garble and keep fact-bearing
                        layers strong.
    """
    sel = set(range(len(model.layers))) if layers is None else set(layers)
    for i, layer in enumerate(model.layers):
        att = _attention_module(layer)
        if att is None:
            continue
        if i in sel and harvested[i] is not None:
            sc = scale_per_layer.get(i, scale) if scale_per_layer else scale
            kg = tc.tensor(np.ascontiguousarray(harvested[i]["k"])).astype(BlockDtype(model))
            vg = tc.tensor(np.ascontiguousarray(harvested[i]["v"])).astype(BlockDtype(model))
            att.inject_kv = (kg, vg, float(sc))
            # Seats the graft occupies. Persists through cached decode (where
            # the injection cat is skipped but the +Sg position shift must
            # survive) — see GQAAttentionTC position handling.
            att.graft_seats = int(kg.shape[2])
        else:
            att.inject_kv = None
            att.graft_seats = 0


def clear_injection(model, free_seats=True):
    """Unmount the graft.

    free_seats=True (default): full reset — positions return to 0-based; use
    when starting a fresh conversation/cache.
    free_seats=False: the graft's content stops being injected but its SEATS
    stay reserved (positional hole) — required when a persistent cache still
    holds live tokens that were RoPE'd at graft-shifted positions.
    """
    for layer in model.layers:
        att = _attention_module(layer)
        if att is None:
            continue
        att.inject_kv = None
        if free_seats:
            att.graft_seats = 0


def BlockDtype(model):
    from core.mistral7b_tc import BlockTC
    return BlockTC.COMPUTE_DTYPE
