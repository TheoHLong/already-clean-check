"""InceptionV1 with the original Lucid weights (via lucent), plus pre-ReLU recording.

Channel indexing follows Lucid / Distill circuits: `mixed4e:55` is channel 55 of the
concatenation (1x1, 3x3, 5x5, pool_reduce) of layer mixed4e.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

MIXED_LAYERS = [
    "mixed3a", "mixed3b", "mixed4a", "mixed4b", "mixed4c",
    "mixed4d", "mixed4e", "mixed5a", "mixed5b",
]
BRANCHES = ["1x1", "3x3", "5x5", "pool_reduce"]  # concat order used by Lucid/lucent
PREV = {b: a for a, b in zip(MIXED_LAYERS[:-1], MIXED_LAYERS[1:])}
NEXT = {a: b for a, b in zip(MIXED_LAYERS[:-1], MIXED_LAYERS[1:])}

LUCID_MEAN = 117.0  # Lucid InceptionV1 expects pixels in [-117, 138]

WEIGHT_URLS = [
    "https://github.com/ProGamerGov/pytorch-old-tensorflow-models/raw/master/inception5h.pth",
    "https://raw.githubusercontent.com/ProGamerGov/pytorch-old-tensorflow-models/master/inception5h.pth",
]


def _load_state_dict():
    last_err = None
    for url in WEIGHT_URLS:
        try:
            return torch.hub.load_state_dict_from_url(url, progress=False, file_name="inception5h.pth")
        except Exception as e:  # noqa: BLE001  (mirror fallback)
            last_err = e
    raise RuntimeError(f"Could not download InceptionV1 weights: {last_err}")


def load_inceptionv1(device: str | torch.device = "cpu", redirected_relu: bool = False) -> nn.Module:
    """Original 2014 InceptionV1 (inception5h). Use redirected_relu=True only for feature vis."""
    from lucent.modelzoo.inceptionv1.InceptionV1 import InceptionV1

    model = InceptionV1(pretrained=False, redirected_ReLU=redirected_relu)
    model.load_state_dict(_load_state_dict())
    return model.eval().to(device)


def to_model_input(x01: torch.Tensor) -> torch.Tensor:
    """[0,1] RGB tensor -> Lucid InceptionV1 input range."""
    return x01 * 255.0 - LUCID_MEAN


def branch_conv(model: nn.Module, layer: str, branch: str) -> nn.Conv2d:
    return getattr(model, f"{layer}_{branch}_pre_relu_conv")


def branch_sizes(model: nn.Module, layer: str) -> list[int]:
    return [branch_conv(model, layer, b).out_channels for b in BRANCHES]


def n_channels(model: nn.Module, layer: str) -> int:
    return sum(branch_sizes(model, layer))


def channel_branch(model: nn.Module, layer: str, ch: int) -> tuple[str, int]:
    """Concat channel index -> (branch name, index inside that branch)."""
    start = 0
    for b, n in zip(BRANCHES, branch_sizes(model, layer)):
        if ch < start + n:
            return b, ch - start
        start += n
    raise IndexError(f"{layer} has only {start} channels")


class PreReluRecorder(contextlib.AbstractContextManager):
    """Records pre-ReLU activations of whole mixed layers (concat of the four branches).

    with PreReluRecorder(model, ["mixed4e"]) as rec:
        model(x)
        a = rec["mixed4e"]   # [B, 832, 14, 14], pre-ReLU
    """

    def __init__(self, model: nn.Module, layers: list[str]):
        self.model, self.layers = model, list(layers)
        self._store: dict[tuple[str, str], torch.Tensor] = {}
        self._handles = []

    def __enter__(self):
        for layer in self.layers:
            for b in BRANCHES:
                mod = branch_conv(self.model, layer, b)
                self._handles.append(mod.register_forward_hook(self._hook(layer, b)))
        return self

    def _hook(self, layer, b):
        def fn(_mod, _inp, out):
            self._store[(layer, b)] = out
        return fn

    def __getitem__(self, layer: str) -> torch.Tensor:
        return torch.cat([self._store[(layer, b)] for b in BRANCHES], dim=1)

    def clear(self):
        self._store.clear()

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._store.clear()
        return False
