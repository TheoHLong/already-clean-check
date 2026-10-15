"""Weights between consecutive mixed layers of InceptionV1.

For a mixed layer L with previous layer P, `layer_weights(model, L)` returns a matrix
W [C_L, C_P] (concat channel indexing on both sides):
  * 1x1 and pool_reduce branches: the actual 1x1 conv weights (exact; pool_reduce reads a
    3x3 max pooled copy of P, so it is exact up to that pooling).
  * 3x3 and 5x5 branches: "expanded weights" W_branch(summed over space) @ W_bottleneck,
    which ignores the ReLU on the bottleneck (the linear approximation used in Zoom In).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .models import BRANCHES, NEXT, PREV, branch_conv


@torch.no_grad()
def layer_weights(model, layer: str) -> tuple[np.ndarray, np.ndarray]:
    blocks, exact = [], []
    for b in BRANCHES:
        conv = branch_conv(model, layer, b)
        if b in ("1x1", "pool_reduce"):
            W = conv.weight[:, :, 0, 0]
            ex = True
        else:
            bott = getattr(model, f"{layer}_{b}_bottleneck_pre_relu_conv").weight[:, :, 0, 0]
            W = conv.weight.sum((2, 3)) @ bott
            ex = False
        blocks.append(W.float().cpu().numpy())
        exact += [ex] * W.shape[0]
    return np.concatenate(blocks, 0), np.asarray(exact)


def input_weights(model, layer: str, ch: int, top: int = 10) -> pd.DataFrame:
    """Strongest inputs from the previous mixed layer into layer:ch (both signs)."""
    W, exact = layer_weights(model, layer)
    w = W[ch]
    order = np.argsort(-np.abs(w))[:top]
    return pd.DataFrame({"from": [f"{PREV[layer]}:{i}" for i in order], "weight": w[order],
                         "rank_pos": [int((w > w[i]).sum()) + 1 for i in order],
                         "exact": bool(exact[ch])})


def reader_weights(model, layer: str, ch: int, top: int = 10) -> pd.DataFrame:
    """Channels in the next mixed layer that read layer:ch most strongly (both signs)."""
    nxt = NEXT[layer]
    W, exact = layer_weights(model, nxt)
    w = W[:, ch]
    order = np.argsort(-np.abs(w))[:top]
    return pd.DataFrame({"to": [f"{nxt}:{i}" for i in order], "weight": w[order],
                         "exact": exact[order]})


def rank_of(model, layer: str, ch: int, other: str) -> dict:
    """Rank of a specific connection, e.g. rank_of(m, 'mixed4e', 55, 'mixed4d:493')."""
    other_layer, other_ch = other.split(":")
    other_ch = int(other_ch)
    if other_layer == PREV.get(layer):
        W, ex = layer_weights(model, layer)
        w, v, exact = W[ch], W[ch, other_ch], ex[ch]
        # rank only among inputs to this channel
    elif other_layer == NEXT.get(layer):
        W, ex = layer_weights(model, other_layer)
        w = W[:, ch]
        v, exact = W[other_ch, ch], ex[other_ch]
        # rank among readers in the same branch type (exact vs expanded are not comparable)
        w = w[ex == exact]
    else:
        raise ValueError("other must be in the previous or next mixed layer")
    return {"connection": other, "weight": float(v), "rank": int((w > v).sum()) + 1,
            "out_of": int(len(w)), "exact": bool(exact)}
