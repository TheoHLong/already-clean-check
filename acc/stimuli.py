"""Synthetic shape stimuli and tuning curves (a neurophysiology style probe).

Orientation convention: the direction the apex points, in degrees, 0 = up, 90 = right,
180 = down, 270 = left. A cat face or a car front seen head on is roughly a V pointing down.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from .models import PreReluRecorder, to_model_input

SIZE = 224
SS = 4  # supersampling for anti aliasing
KINDS = ["V", "triangle", "U", "O", "disk", "line", "square"]
POINTED = {"V", "triangle"}
ROUND = {"U", "O", "disk"}


def _rot(points, deg, c):
    t = math.radians(deg)
    ct, st = math.cos(t), math.sin(t)
    return [(c[0] + (x - c[0]) * ct - (y - c[1]) * st, c[1] + (x - c[0]) * st + (y - c[1]) * ct)
            for x, y in points]


def render(kind: str, orientation: float = 0.0, scale: float = 0.5, opening: float = 60.0,
           thickness: float = 0.06, polarity: str = "dark", center=(0.5, 0.5)) -> Image.Image:
    """One 224x224 RGB stimulus. scale = shape height as a fraction of the image."""
    S = SIZE * SS
    bg, fg = ((150, 150, 150), (20, 20, 20)) if polarity == "dark" else ((90, 90, 90), (235, 235, 235))
    img = Image.new("RGB", (S, S), bg)
    d = ImageDraw.Draw(img)
    cx, cy = center[0] * S, center[1] * S
    h = scale * S
    w = max(2, int(thickness * S))
    half = math.radians(opening / 2)
    # canonical shapes point UP (apex at top); rotated clockwise by `orientation`
    if kind in ("V", "triangle"):
        apex = (cx, cy - h / 2)
        arm = h / math.cos(half)
        left = (apex[0] - arm * math.sin(half), apex[1] + arm * math.cos(half))
        right = (apex[0] + arm * math.sin(half), apex[1] + arm * math.cos(half))
        pts = _rot([left, apex, right], orientation, (cx, cy))
        if kind == "V":
            d.line(pts, fill=fg, width=w, joint="curve")
        else:
            d.polygon(pts, fill=fg)
    elif kind == "U":  # same footprint as V but with a round apex
        r = h / 2
        arc = [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)) - r / 2)
               for a in np.linspace(180, 360, 40)]
        arms = [(arc[0][0], arc[0][1] + h / 2)] + arc + [(arc[-1][0], arc[-1][1] + h / 2)]
        d.line(_rot(arms, orientation, (cx, cy)), fill=fg, width=w, joint="curve")
    elif kind == "O":
        r = h / 2
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=fg, width=w)
    elif kind == "disk":
        r = h / 2
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg)
    elif kind == "line":
        pts = _rot([(cx, cy - h / 2), (cx, cy + h / 2)], orientation, (cx, cy))
        d.line(pts, fill=fg, width=w)
    elif kind == "square":
        s = h / 2
        pts = _rot([(cx - s, cy - s), (cx + s, cy - s), (cx + s, cy + s), (cx - s, cy + s)],
                   orientation, (cx, cy))
        d.polygon(pts, fill=fg)
    else:
        raise ValueError(kind)
    return img.resize((SIZE, SIZE), Image.LANCZOS)


def _to_tensor(imgs):
    arr = np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in imgs])
    return torch.from_numpy(arr).permute(0, 3, 1, 2)


@torch.no_grad()
def responses(model, imgs, layer: str, ch: int, device="cpu", batch_size=64) -> np.ndarray:
    """Spatial max of the pre-ReLU activation of `layer:ch` for each image."""
    out = []
    with PreReluRecorder(model, [layer]) as rec:
        for k in range(0, len(imgs), batch_size):
            x = to_model_input(_to_tensor(imgs[k:k + batch_size]).to(device))
            model(x)
            out.append(rec[layer][:, ch].flatten(1).max(-1).values.float().cpu().numpy())
            rec.clear()
    return np.concatenate(out)


def shape_battery(orientations=range(0, 360, 30), scales=(0.3, 0.5, 0.7),
                  polarities=("dark", "light"), opening=60.0):
    rows = []
    for kind, o, s, p in itertools.product(KINDS, orientations, scales, polarities):
        if kind in ("O", "disk") and o != 0:
            continue  # rotation invariant
        rows.append(dict(kind=kind, orientation=o, scale=s, polarity=p, opening=opening))
    return rows


def run_battery(model, rows, layer, ch, device="cpu") -> pd.DataFrame:
    imgs = [render(r["kind"], r["orientation"], r["scale"], r["opening"], polarity=r["polarity"])
            for r in rows]
    df = pd.DataFrame(rows)
    df["response"] = responses(model, imgs, layer, ch, device)
    df["family"] = df["kind"].map(lambda k: "pointed" if k in POINTED else ("round" if k in ROUND else "other"))
    return df


def orientation_tuning(model, layer, ch, kinds=("V", "triangle", "U"), step=15, device="cpu"):
    rows = [dict(kind=k, orientation=o, scale=s, polarity=p, opening=60.0)
            for k in kinds for o in range(0, 360, step) for s in (0.3, 0.5, 0.7) for p in ("dark", "light")]
    return run_battery(model, rows, layer, ch, device)


def opening_tuning(model, layer, ch, openings=range(20, 171, 10), device="cpu"):
    rows = [dict(kind="V", orientation=o, scale=s, polarity=p, opening=float(a))
            for a in openings for o in (0, 180) for s in (0.3, 0.5, 0.7) for p in ("dark", "light")]
    return run_battery(model, rows, layer, ch, device)
