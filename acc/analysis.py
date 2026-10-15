"""Dataset example analysis on a cached layer: top/bottom images, class composition,
class level tests of preregistered predictions, and plotting helpers."""
from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import CATEGORIES, CROP, cropped_pil


def top_k(cache: dict, ch: int, k: int = 16, pole: str = "pos") -> np.ndarray:
    """Row positions in the cache: strongest spatial max (pos) or most negative min (neg)."""
    v = cache["max"][:, ch] if pole == "pos" else -cache["min"][:, ch]
    return np.argsort(-v)[:k]


def class_composition(cache: dict, ch: int, n: int = 100, pole: str = "pos") -> pd.DataFrame:
    rows = top_k(cache, ch, n, pole)
    labs = cache["label"][rows]
    vc = pd.Series(labs).value_counts()
    df = pd.DataFrame({"class": [CATEGORIES[i] for i in vc.index], "count": vc.values})
    p = vc.values / vc.values.sum()
    df.attrs["entropy_bits"] = float(-(p * np.log2(p)).sum())
    df.attrs["n_classes"] = int(len(vc))
    return df


def class_means(cache: dict, ch: int, stat: str = "max") -> pd.Series:
    v = cache[stat][:, ch]
    s = pd.Series(v).groupby(cache["label"]).mean()
    s.index = [CATEGORIES[i] for i in s.index]
    return s


def _auc(a: np.ndarray, b: np.ndarray) -> float:
    """P(random a > random b), Mann Whitney U / (n_a n_b)."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = (a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
    return float(gt / (len(a) * len(b)))


def prereg_test(cache: dict, ch: int, prereg_path: str) -> tuple[pd.DataFrame, dict]:
    """Class level test of predictions written down BEFORE looking at the data.

    prereg JSON: {"unit": "mixed4e:55", "predicted_positive": [...], "predicted_negative": [...]}
    For every class we use the mean over its validation images of
      max_score = spatial max of pre ReLU (how strongly it can drive the unit)
      min_score = spatial min of pre ReLU (how strongly it pushes it negative)
    """
    with open(prereg_path) as f:
        pr = json.load(f)
    unknown = [c for k, v in pr.items() if isinstance(v, list) for c in v if c not in CATEGORIES]
    if unknown:
        raise ValueError(f"Not ImageNet class names (see acc.data.CATEGORIES): {unknown}")
    mx, mn = class_means(cache, ch, "max"), class_means(cache, ch, "min")
    rank_max = mx.rank(ascending=False)
    rank_min = mn.rank(ascending=True)
    rows = []
    for group in [k for k, v in pr.items() if isinstance(v, list)]:
        for c in pr[group]:
            present = c in mx.index
            rows.append(dict(group=group, cls=c,
                             max_score=mx[c] if present else np.nan,
                             rank_by_max=rank_max[c] if present else np.nan,
                             min_score=mn[c] if present else np.nan,
                             rank_by_min=rank_min[c] if present else np.nan))
    df = pd.DataFrame(rows)
    summary = {"n_classes_in_cache": int(len(mx))}
    for tag in ("predicted", "essay"):
        pos = [c for c in pr.get(f"{tag}_positive", []) if c in mx.index]
        neg = [c for c in pr.get(f"{tag}_negative", []) if c in mx.index]
        rest = [c for c in mx.index if c not in pos and c not in neg]
        summary[f"{tag}: AUC(max) positive vs negative"] = _auc(mx[pos].values, mx[neg].values)
        summary[f"{tag}: AUC(max) positive vs all other classes"] = _auc(mx[pos].values, mx[rest].values)
        summary[f"{tag}: AUC(min) negative more negative than other classes"] = _auc(-mn[neg].values, -mn[rest].values)
    return df, summary


def crop_box(pos_flat: int, hw: tuple[int, int], size: int = 112) -> tuple[int, int, int, int]:
    """Pixel box (in the 224 crop) centred on a spatial position of a layer's feature map."""
    h, w = hw
    r, c = divmod(int(pos_flat), w)
    cy, cx = (r + 0.5) * CROP / h, (c + 0.5) * CROP / w
    x0 = int(min(max(cx - size / 2, 0), CROP - size))
    y0 = int(min(max(cy - size / 2, 0), CROP - size))
    return x0, y0, x0 + size, y0 + size


def show_examples(source, cache: dict, ch: int, k: int = 12, pole: str = "pos",
                  crop: int | None = None, ncols: int = 6, title: str | None = None):
    rows = top_k(cache, ch, k, pole)
    nrows = math.ceil(len(rows) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    stat, argk = ("max", "argmax") if pole == "pos" else ("min", "argmin")
    for ax, r in zip(axes, rows):
        im = cropped_pil(source, int(cache["idx"][r]))
        if crop:
            im = im.crop(crop_box(cache[argk][r, ch], cache["meta"]["hw"], crop))
        ax.imshow(im)
        ax.set_title(f"{CATEGORIES[cache['label'][r]][:18]}\n{cache[stat][r, ch]:.1f}", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title or f"{cache['layer']}:{ch}  {'positive' if pole == 'pos' else 'negative'} pole")
    fig.tight_layout()
    return fig


def percentile_scale(cache: dict, ch: int, q: float = 99.0) -> float:
    """Natural image reference for normalising synthetic stimulus responses."""
    return float(np.percentile(cache["max"][:, ch], q))
