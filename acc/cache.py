"""Resumable activation cache over the validation set.

For every image and every channel of each requested layer we store the spatial max and
min of the PRE-ReLU activation (min gives the negative pole) and where they occurred.
Work is split into shards; each finished shard is written atomically and marked done,
so a Colab disconnect or a Modal preemption only loses the shard in progress.

Layout:
    out_dir/meta.json
    out_dir/labels/shard_000.npz          (idx, label)
    out_dir/<layer>/shard_000.npz         (max, min, argmax, argmin)
    out_dir/done/shard_000
"""
from __future__ import annotations

import glob
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ValDataset
from .models import PreReluRecorder, to_model_input


def _atomic_savez(path: str, **arrays):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


@torch.no_grad()
def cache_activations(model, source, layers, out_dir, batch_size=256, shard_size=5000,
                      num_workers=8, device="cuda", limit=None, log_every=1, on_shard_done=None):
    """on_shard_done: optional callback, e.g. a Modal Volume commit after every shard."""
    n = len(source) if limit is None else min(int(limit), len(source))
    n_shards = math.ceil(n / shard_size)
    for sub in ["labels", "done", *layers]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    meta = {"n": n, "shard_size": shard_size, "layers": {}, "input": "Resize256-CenterCrop224",
            "stat": "pre-ReLU spatial max/min"}
    use_cuda = str(device).startswith("cuda")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    t0 = time.time()
    for s in range(n_shards):
        done_flag = os.path.join(out_dir, "done", f"shard_{s:03d}")
        if os.path.exists(done_flag):
            continue
        lo, hi = s * shard_size, min(n, (s + 1) * shard_size)
        loader = DataLoader(ValDataset(source, np.arange(lo, hi)), batch_size=batch_size,
                            num_workers=num_workers, pin_memory=use_cuda,
                            persistent_workers=False)
        bufs = {l: {"max": [], "min": [], "argmax": [], "argmin": []} for l in layers}
        idxs, labs = [], []
        with PreReluRecorder(model, layers) as rec:
            for x, y, i in loader:
                x = to_model_input(x.to(device, non_blocking=True))
                model(x)
                for l in layers:
                    a = rec[l].float()
                    if l not in meta["layers"]:
                        meta["layers"][l] = {"channels": int(a.shape[1]), "hw": [int(a.shape[2]), int(a.shape[3])]}
                    flat = a.flatten(2)
                    mx, amx = flat.max(-1)
                    mn, amn = flat.min(-1)
                    b = bufs[l]
                    b["max"].append(mx.cpu().numpy())
                    b["min"].append(mn.cpu().numpy())
                    b["argmax"].append(amx.to(torch.int16).cpu().numpy())
                    b["argmin"].append(amn.to(torch.int16).cpu().numpy())
                rec.clear()
                idxs.append(i.numpy())
                labs.append(y.numpy())
        for l in layers:
            _atomic_savez(os.path.join(out_dir, l, f"shard_{s:03d}.npz"),
                          **{k: np.concatenate(v) for k, v in bufs[l].items()})
        _atomic_savez(os.path.join(out_dir, "labels", f"shard_{s:03d}.npz"),
                      idx=np.concatenate(idxs), label=np.concatenate(labs))
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        open(done_flag, "w").close()
        if on_shard_done is not None:
            on_shard_done()
        if (s + 1) % log_every == 0:
            rate = (hi - lo) / max(1e-9, time.time() - t0)
            print(f"[cache] shard {s + 1}/{n_shards} done ({hi}/{n} images, {rate:.0f} img/s)")
        t0 = time.time()
    return out_dir


def load_cache(out_dir: str, layer: str) -> dict:
    """Concatenate all finished shards of one layer. Returns max/min/argmax/argmin [N, C],
    idx and label [N], plus meta."""
    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    shards = sorted(glob.glob(os.path.join(out_dir, "done", "shard_*")))
    names = [os.path.basename(p) + ".npz" for p in shards]
    parts = [np.load(os.path.join(out_dir, layer, nm)) for nm in names]
    lab = [np.load(os.path.join(out_dir, "labels", nm)) for nm in names]
    out = {k: np.concatenate([p[k] for p in parts]) for k in ["max", "min", "argmax", "argmin"]}
    out["idx"] = np.concatenate([p["idx"] for p in lab])
    out["label"] = np.concatenate([p["label"] for p in lab])
    out["meta"] = meta["layers"][layer]
    out["layer"] = layer
    return out
