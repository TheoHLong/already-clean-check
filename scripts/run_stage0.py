"""Headless Stage 0: cache activations on ImageNet val, then analyse one unit.

    python scripts/run_stage0.py --data /path/to/imagenet_val --out runs/stage0
    python scripts/run_stage0.py --data fake --out /tmp/smoke --limit 48   # pipeline smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from acc.cache import cache_activations  # noqa: E402
from acc.data import make_fake_imagefolder, open_val  # noqa: E402
from acc.models import load_inceptionv1  # noqa: E402
from acc.stage0 import run_all  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="val dir (parquet, extracted tar or ImageFolder) or 'fake'")
    p.add_argument("--out", required=True)
    p.add_argument("--layers", default="mixed4d,mixed4e,mixed5a")
    p.add_argument("--unit", default="mixed4e:55")
    p.add_argument("--prereg", default=os.path.join(os.path.dirname(__file__), "..", "prereg", "4e55.json"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--shard-size", type=int, default=5000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--featvis", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)

    data = a.data
    if data == "fake":
        data = make_fake_imagefolder(os.path.join(a.out, "fake_val"))
    source = open_val(data)
    model = load_inceptionv1(a.device)
    cache_dir = os.path.join(a.out, "cache_inceptionv1")
    layers = a.layers.split(",")
    unit_layer = a.unit.split(":")[0]
    if unit_layer not in layers:
        layers.append(unit_layer)
    cache_activations(model, source, layers, cache_dir, batch_size=a.batch_size,
                      shard_size=a.shard_size, num_workers=a.workers, device=a.device, limit=a.limit)
    summary = run_all(model, source, cache_dir, a.unit, a.prereg, os.path.join(a.out, a.unit.replace(":", "_")),
                      device=a.device, featvis=a.featvis)
    print(json.dumps(summary, indent=2, default=float)[:4000])


if __name__ == "__main__":
    main()
