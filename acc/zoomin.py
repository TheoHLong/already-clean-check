"""Stage 0b: Zoom In's "superposition" car circuit vs the essay's "generalized pose neurons".

Zoom In (Olah et al. 2020) showed a car detector in mixed4c whose feature is spread into
several dog detecting neurons in mixed4d, read as the model storing the car "in superposition"
to be read back out later. The Goodfire essay says those neurons are generalized pose neurons
(one pose shared by cars and dogs) and that nothing downstream reads the combination back out.

Everything is selected by rules written in prereg/zoomin_car.json before looking at results:
  C  = the most car selective channel of mixed4c (share of car classes in its top 100 images)
  T  = the 8 mixed4d channels with the largest positive weight from C
  D  = members of T whose top 100 images are at least half dogs ("the dog neurons")
Tests:
  pose    : horizontal flip preference measured on dogs and on cars separately (label free pose probe)
  readout : how much of the car evidence in car selective mixed4e channels flows through D
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .circuits import layer_weights
from .data import CATEGORIES, ValDataset, class_index
from .models import PreReluRecorder, to_model_input

CAR_CLASSES = ["ambulance", "beach wagon", "cab", "convertible", "jeep", "limousine", "minivan",
               "Model T", "racer", "sports car", "pickup", "police van"]
DOG_LO, DOG_HI = 151, 268  # ImageNet dog breeds (Chihuahua .. Mexican hairless)


def masks(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    is_car = np.isin(labels, [class_index(c) for c in CAR_CLASSES])
    is_dog = (labels >= DOG_LO) & (labels <= DOG_HI)
    return is_car, is_dog


def top_shares(cache: dict, mask: np.ndarray, n: int = 100) -> np.ndarray:
    """For every channel: fraction of its top n images (by spatial max) inside `mask`."""
    X = cache["max"]
    n = min(n, len(X))
    top = np.argpartition(-X, n - 1, axis=0)[:n]  # [n, C]
    return mask[top].mean(0)


def find_car_detector(cache4c: dict) -> pd.DataFrame:
    is_car, is_dog = masks(cache4c["label"])
    df = pd.DataFrame({"channel": np.arange(cache4c["max"].shape[1]),
                       "car_share": top_shares(cache4c, is_car),
                       "dog_share": top_shares(cache4c, is_dog)})
    return df.sort_values(["car_share", "channel"], ascending=[False, True]).reset_index(drop=True)


def spread_targets(model, c4c: int, cache4d: dict, k: int = 8, dog_min: float = 0.5) -> pd.DataFrame:
    W, exact = layer_weights(model, "mixed4d")  # [528, 512]
    w = W[:, c4c]
    order = np.argsort(-w)[:k]
    is_car, is_dog = masks(cache4d["label"])
    car_s, dog_s = top_shares(cache4d, is_car), top_shares(cache4d, is_dog)
    df = pd.DataFrame({"channel": order, "weight_from_C": w[order], "exact_weight": exact[order],
                       "car_share_top100": car_s[order], "dog_share_top100": dog_s[order]})
    df["in_D"] = df["dog_share_top100"] >= dog_min
    return df


# ----------------------------------------------------------------------------- pose (flip) test
@torch.no_grad()
def flip_responses(model, source, indices, layer: str, device="cuda", batch_size=128, num_workers=8):
    """Spatial max (pre ReLU) of every channel of `layer` for each image and its mirror image."""
    loader = DataLoader(ValDataset(source, indices), batch_size=batch_size, num_workers=num_workers,
                        pin_memory=str(device).startswith("cuda"))
    orig, flip = [], []
    with PreReluRecorder(model, [layer]) as rec:
        for x, _, _ in loader:
            x = x.to(device, non_blocking=True)
            model(to_model_input(torch.cat([x, x.flip(-1)])))
            a = rec[layer].float().flatten(2).max(-1).values.cpu().numpy()
            rec.clear()
            b = len(x)
            orig.append(a[:b])
            flip.append(a[b:])
    return np.concatenate(orig), np.concatenate(flip)


def flip_preference(orig: np.ndarray, flip: np.ndarray, m: int = 50) -> np.ndarray:
    """Per channel, over the m images it responds to most (in either orientation, so the
    selection itself is mirror symmetric): mean sign of (original - mirrored). +1 means it always
    prefers the orientation as photographed, -1 the mirrored one, 0 no lateral preference."""
    s = np.maximum(orig, flip)
    m = min(m, len(s))
    top = np.argpartition(-s, m - 1, axis=0)[:m]
    d = np.take_along_axis(orig - flip, top, axis=0)
    return np.sign(d).mean(0)


def pose_test(model, source, labels, chans_D, device="cuda", lateral_min=0.3, m=50, **kw) -> tuple[pd.DataFrame, dict]:
    is_car, is_dog = masks(labels)
    if is_car.sum() == 0 or is_dog.sum() == 0:  # e.g. smoke data without cars or dogs
        nan = float("nan")
        return pd.DataFrame(), {"D_lateral_n": 0, "D_same_sign_rate": nan, "other_4d_lateral_n": 0,
                                "other_4d_same_sign_rate": nan, "D_mean_abs_pref_dogs": nan,
                                "D_mean_abs_pref_cars": nan}
    o_d, f_d = flip_responses(model, source, np.where(is_dog)[0], "mixed4d", device, **kw)
    o_c, f_c = flip_responses(model, source, np.where(is_car)[0], "mixed4d", device, **kw)
    p_dog, p_car = flip_preference(o_d, f_d, m), flip_preference(o_c, f_c, m)
    df = pd.DataFrame({"channel": np.arange(len(p_dog)), "pref_dogs": p_dog, "pref_cars": p_car})
    df["lateral_on_dogs"] = df["pref_dogs"].abs() >= lateral_min
    df["same_sign"] = np.sign(df["pref_dogs"]) == np.sign(df["pref_cars"])
    df["in_D"] = df["channel"].isin(chans_D)

    def rate(sub):
        sub = sub[sub["lateral_on_dogs"]]
        return (float(sub["same_sign"].mean()) if len(sub) else float("nan")), int(len(sub))

    rD, nD = rate(df[df["in_D"]])
    rAll, nAll = rate(df[~df["in_D"]])
    summary = {"D_lateral_n": nD, "D_same_sign_rate": rD,
               "other_4d_lateral_n": nAll, "other_4d_same_sign_rate": rAll,
               "D_mean_abs_pref_dogs": float(df.loc[df.in_D, "pref_dogs"].abs().mean()) if len(chans_D) else float("nan"),
               "D_mean_abs_pref_cars": float(df.loc[df.in_D, "pref_cars"].abs().mean()) if len(chans_D) else float("nan")}
    return df, summary


# ----------------------------------------------------------------------------- readout test
def attribution(model, source, indices, tgt_layer: str, tgt_ch: int, src_layer: str,
                device="cuda", batch_size=64, num_workers=4) -> np.ndarray:
    """Mean over images of grad x activation from every src_layer channel (post ReLU output of
    the mixed block) into the spatial max of tgt_layer:tgt_ch (pre ReLU)."""
    for p in model.parameters():
        p.requires_grad_(False)
    store = {}

    def hook(_m, _i, out):
        out.retain_grad()
        store["a"] = out

    h = getattr(model, src_layer).register_forward_hook(hook)
    loader = DataLoader(ValDataset(source, indices), batch_size=batch_size, num_workers=num_workers)
    total, n = None, 0
    try:
        with PreReluRecorder(model, [tgt_layer]) as rec:
            for x, _, _ in loader:
                x = to_model_input(x.to(device)).requires_grad_(True)
                model(x)
                rec[tgt_layer][:, tgt_ch].flatten(1).max(-1).values.sum().backward()
                a = store["a"]
                contrib = (a * a.grad).sum((0, 2, 3)).detach().float().cpu().numpy()
                total = contrib if total is None else total + contrib
                n += len(x)
                rec.clear()
    finally:
        h.remove()
    return total / max(n, 1)


def readout_test(model, source, cache4d, cache4e, chans_D, device="cuda", car_min=0.5,
                 max_targets=10, m=50, **kw) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    is_car4e, is_dog4e = masks(cache4e["label"])
    car_s4e, dog_s4e = top_shares(cache4e, is_car4e), top_shares(cache4e, is_dog4e)
    car_s4d = top_shares(cache4d, masks(cache4d["label"])[0])
    car4d = np.where(car_s4d >= car_min)[0]
    targets = np.argsort(-car_s4e)[:max_targets]
    targets = [int(t) for t in targets if car_s4e[t] >= car_min]
    car_rows = np.where(is_car4e)[0]
    rows = []
    for t in targets:
        # the target's m strongest car images
        sel = car_rows[np.argsort(-cache4e["max"][car_rows, t])[:m]]
        attr = attribution(model, source, cache4e["idx"][sel], "mixed4e", t, "mixed4d", device, **kw)
        pos = np.clip(attr, 0, None)
        tot = pos.sum() if pos.sum() > 0 else 1.0
        rows.append({"target_4e": t, "car_share_top100": float(car_s4e[t]),
                     "share_through_D": float(pos[list(chans_D)].sum() / tot) if len(chans_D) else 0.0,
                     "share_through_car_4d": float(pos[car4d].sum() / tot),
                     "top_inputs": ", ".join(f"4d:{i}" for i in np.argsort(-attr)[:5])})
    readout = pd.DataFrame(rows)
    # channels that most strongly combine D (summed positive weights from D)
    W, _ = layer_weights(model, "mixed4e")  # [832, 528]
    comb = np.clip(W[:, list(chans_D)], 0, None).sum(1) if len(chans_D) else np.zeros(W.shape[0])
    order = np.argsort(-comb)[:5]
    combiners = pd.DataFrame({"channel_4e": order, "summed_weight_from_D": comb[order],
                              "car_share_top100": car_s4e[order], "dog_share_top100": dog_s4e[order]})
    summary = {"n_car_selective_4e": len(targets),
               "D_channel_share": len(chans_D) / W.shape[1],
               "mean_share_through_D": float(readout["share_through_D"].mean()) if len(readout) else float("nan"),
               "mean_share_through_car_4d": float(readout["share_through_car_4d"].mean()) if len(readout) else float("nan"),
               "max_car_share_among_top5_combiners": float(combiners["car_share_top100"].max())}
    return readout, combiners, summary


def class_list(cache: dict, ch: int, n: int = 100, k: int = 8) -> str:
    rows = np.argsort(-cache["max"][:, ch])[:n]
    vc = pd.Series(cache["label"][rows]).value_counts().head(k)
    return ", ".join(f"{CATEGORIES[i]} {c}" for i, c in vc.items())


def run_all(model, source, cache4c: dict, cache4d: dict, cache4e: dict, out_dir: str,
            device="cuda", workers: int = 8) -> dict:
    """Run every preregistered check, save CSVs + summary.json, return the summary."""
    import json
    import os

    os.makedirs(out_dir, exist_ok=True)
    det = find_car_detector(cache4c)
    C = int(det.loc[0, "channel"])
    tgt = spread_targets(model, C, cache4d)
    D = [int(c) for c in tgt.loc[tgt["in_D"], "channel"]]
    pose_df, pose_sum = pose_test(model, source, cache4d["label"], D, device, num_workers=workers)
    readout, combiners, read_sum = readout_test(model, source, cache4d, cache4e, D, device)

    det.head(20).to_csv(os.path.join(out_dir, "car_detectors_4c.csv"), index=False)
    tgt.to_csv(os.path.join(out_dir, "targets_of_C_4d.csv"), index=False)
    pose_df.to_csv(os.path.join(out_dir, "flip_preference_4d.csv"), index=False)
    readout.to_csv(os.path.join(out_dir, "readout_4e.csv"), index=False)
    combiners.to_csv(os.path.join(out_dir, "combiners_4e.csv"), index=False)

    summary = {
        "C": f"mixed4c:{C}", "C_car_share": float(det.loc[0, "car_share"]),
        "C_runner_up": f"mixed4c:{int(det.loc[1, 'channel'])} ({det.loc[1, 'car_share']:.2f})",
        "T": [f"mixed4d:{c}" for c in tgt["channel"]], "D": [f"mixed4d:{c}" for c in D],
        "P1_circuit_exists": bool(det.loc[0, "car_share"] >= 0.5 and len(D) >= 3),
        "P2_median_car_share_D": float(tgt.loc[tgt["in_D"], "car_share_top100"].median()) if D else float("nan"),
        "P3_pose": pose_sum, "P4_P5_readout": read_sum,
        "D_top_classes": {f"mixed4d:{c}": class_list(cache4d, c) for c in D},
        "C_top_classes": class_list(cache4c, C),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


def subset(cache: dict, mask: np.ndarray) -> dict:
    """Cache restricted to rows where mask is True (e.g. only car images), for show_examples."""
    out = {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in cache.items()}
    return out
