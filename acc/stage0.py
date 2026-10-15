"""Stage 0 analyses for one unit (default mixed4e:55), shared by the notebook, the CLI and Modal."""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import pandas as pd

from . import analysis as A
from . import circuits as C
from . import stimuli as S
from .cache import load_cache
from .models import channel_branch

# Connections drawn in the essay's 4e:55 figure
ESSAY_CONNECTIONS = {"mixed4e:55": ["mixed4d:493", "mixed4d:477", "mixed4d:201",
                                    "mixed5a:94", "mixed5a:352", "mixed5a:317"]}


def parse_unit(unit: str) -> tuple[str, int]:
    layer, ch = unit.split(":")
    return layer, int(ch)


def circuit_check(model, unit: str) -> dict[str, pd.DataFrame]:
    layer, ch = parse_unit(unit)
    out = {"inputs": C.input_weights(model, layer, ch, top=12),
           "readers": C.reader_weights(model, layer, ch, top=12)}
    if unit in ESSAY_CONNECTIONS:
        out["essay_connections"] = pd.DataFrame([C.rank_of(model, layer, ch, o) for o in ESSAY_CONNECTIONS[unit]])
    return out


def synthetic_tuning(model, unit: str, device="cpu", natural_p99: float | None = None) -> dict[str, pd.DataFrame]:
    layer, ch = parse_unit(unit)
    res = {"battery": S.run_battery(model, S.shape_battery(), layer, ch, device),
           "orientation": S.orientation_tuning(model, layer, ch, device=device),
           "opening": S.opening_tuning(model, layer, ch, device=device)}
    if natural_p99:
        for df in res.values():
            df["response_norm"] = df["response"] / natural_p99
    return res


def plot_tuning(res: dict[str, pd.DataFrame], unit: str):
    col = "response_norm" if "response_norm" in res["battery"] else "response"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    b = res["battery"].groupby("kind")[col].agg(["mean", "sem"]).reindex(S.KINDS)
    axes[0].bar(b.index, b["mean"], yerr=b["sem"], color=["C3" if k in S.POINTED else "C0" if k in S.ROUND else "C7" for k in b.index])
    axes[0].set_title("shape battery (red pointed, blue round)")
    o = res["orientation"].groupby(["kind", "orientation"])[col].mean().unstack(0)
    o.plot(ax=axes[1], marker="o")
    axes[1].set_xlabel("apex direction (deg, 0 = up, 180 = down)")
    axes[1].set_title("orientation tuning")
    a = res["opening"].groupby("opening")[col].mean()
    axes[2].plot(a.index, a.values, marker="o")
    axes[2].set_xlabel("V opening angle (deg)")
    axes[2].set_title("opening angle tuning")
    for ax in axes:
        ax.set_ylabel(col + (" (fraction of natural p99)" if col.endswith("norm") else ""))
        ax.axhline(0, color="k", lw=0.5)
    fig.suptitle(f"{unit} synthetic stimuli")
    fig.tight_layout()
    return fig


def feature_vis(unit: str, negative: bool = False, steps: int = 512, size: int = 224):
    """Lucent feature visualisation of the pre ReLU unit (needs a GPU to be quick)."""
    import torch
    from lucent.optvis import objectives, param, render

    from .models import load_inceptionv1

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    m = load_inceptionv1(dev, redirected_relu=True)
    layer, ch = parse_unit(unit)
    branch, local = channel_branch(m, layer, ch)
    obj = objectives.channel(f"{layer}_{branch}_pre_relu_conv", local)
    if negative:
        obj = -obj
    imgs = render.render_vis(m, obj, lambda: param.image(size), thresholds=(steps,),
                             show_image=False, progress=False)
    return imgs[-1][0]


def run_all(model, source, cache_dir: str, unit: str, prereg: str | None, out_dir: str,
            device="cpu", featvis: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    layer, ch = parse_unit(unit)
    summary: dict = {"unit": unit}

    circ = circuit_check(model, unit)
    for k, df in circ.items():
        df.to_csv(os.path.join(out_dir, f"circuit_{k}.csv"), index=False)
    if "essay_connections" in circ:
        summary["essay_connections"] = circ["essay_connections"].to_dict("records")

    cache = load_cache(cache_dir, layer)
    p99 = A.percentile_scale(cache, ch)
    summary["n_images"] = int(len(cache["label"]))
    summary["natural_p99_max"] = p99
    for pole in ("pos", "neg"):
        comp = A.class_composition(cache, ch, n=100, pole=pole)
        comp.to_csv(os.path.join(out_dir, f"top100_classes_{pole}.csv"), index=False)
        summary[f"top100_{pole}"] = {"n_classes": comp.attrs["n_classes"], "entropy_bits": comp.attrs["entropy_bits"],
                                     "top5": comp.head(5).to_dict("records")}
        for crop in (None, 112):
            fig = A.show_examples(source, cache, ch, k=18, pole=pole, crop=crop)
            fig.savefig(os.path.join(out_dir, f"examples_{pole}{'_crop' if crop else ''}.png"), dpi=110)
            plt.close(fig)

    if prereg and os.path.exists(prereg):
        df, s = A.prereg_test(cache, ch, prereg)
        df.to_csv(os.path.join(out_dir, "prereg_classes.csv"), index=False)
        summary["prereg"] = s

    tun = synthetic_tuning(model, unit, device, natural_p99=p99)
    for k, df in tun.items():
        df.to_csv(os.path.join(out_dir, f"synthetic_{k}.csv"), index=False)
    fig = plot_tuning(tun, unit)
    fig.savefig(os.path.join(out_dir, "synthetic_tuning.png"), dpi=110)
    plt.close(fig)
    summary["synthetic_by_kind_norm"] = tun["battery"].groupby("kind")["response_norm"].mean().round(3).to_dict()

    if featvis:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, neg in zip(axes, (False, True)):
            ax.imshow(feature_vis(unit, negative=neg))
            ax.set_title("negative" if neg else "positive")
            ax.axis("off")
        fig.savefig(os.path.join(out_dir, "feature_vis.png"), dpi=110)
        plt.close(fig)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary
