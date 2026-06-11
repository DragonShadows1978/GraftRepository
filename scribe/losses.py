"""SCRIBE losses (protocol §3).

L-warm: Huber on latents, PER-LAYER NORMALIZED — each target layer's
error is scaled by that layer's latent RMS (computed from the batch's
ground truth, detached) so deep high-magnitude layers don't drown
shallow ones. Depth-weighted variant is a dial (weights=None default).

Composed from differentiable engine ops (abs/where/mean); tc.where
captures its condition as a constant — exactly right for Huber's branch.
"""
import numpy as np

import tensor_cuda as tc


def huber_per_layer(pred, truth, delta=1.0, layer_weights=None):
    """pred, truth: (B, L_target, S, 288). Returns (scalar loss,
    per-layer host errors (L,) for the error-directed minting policy)."""
    B, L, S, D = truth.shape
    # per-layer RMS normalizer from ground truth (detached constant)
    rms = (truth * truth).mean([0, 2, 3], True).pow(0.5).detach() + 1e-6
    d = (pred - truth) * rms.pow(-1.0)
    ad = d.abs()
    dl = tc.tensor(np.full((1, 1, 1, 1), delta, np.float32))
    small = ad.le(dl)                 # engine compares tensor-vs-tensor only
    if d.dtype != "float32":
        small = small.astype(d.dtype)
    hub = tc.where(small, d * d * 0.5, ad * delta - 0.5 * delta * delta)
    per_layer = hub.mean([0, 2, 3], False)            # (L,)
    loss = per_layer.mean([0], False)
    if layer_weights is not None:
        w = tc.tensor(np.asarray(layer_weights, np.float32))
        loss = (per_layer * w).mean([0], False)
    return loss, per_layer.detach().numpy()
