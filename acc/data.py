"""ImageNet validation set from Hugging Face (parquet or tar layout) or an ImageFolder.

Every source exposes the same interface:
    len(src), src.labels (np.ndarray[int]), src.get_pil(i) -> RGB PIL (uncropped)
and `ValDataset(src)` yields (tensor in [0,1], label, index) after Resize(256)+CenterCrop(224).
Labels use the standard ImageNet-1k index (sorted WordNet ids).
"""
from __future__ import annotations

import glob
import os
import re
import tarfile

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.models import ResNet50_Weights

CATEGORIES: list[str] = list(ResNet50_Weights.IMAGENET1K_V1.meta["categories"])
WNID_RE = re.compile(r"(n\d{8})")
IMG_EXT = (".jpeg", ".jpg", ".png")

RESIZE, CROP = 256, 224
crop_transform = T.Compose([T.Resize(RESIZE), T.CenterCrop(CROP)])
tensor_transform = T.Compose([crop_transform, T.ToTensor()])


def class_index(name: str) -> int:
    return CATEGORIES.index(name)


# ----------------------------------------------------------------------------- sources
class ParquetSource:
    """HF parquet layout: columns `image` ({bytes, path}) and `label` (int)."""

    def __init__(self, files: list[str]):
        from datasets import Image as HFImage
        from datasets import load_dataset

        self.ds = load_dataset("parquet", data_files=sorted(files), split="train")
        self.ds = self.ds.cast_column("image", HFImage(decode=True))
        self.labels = np.asarray(self.ds["label"], dtype=np.int64)

    def __len__(self):
        return len(self.ds)

    def get_pil(self, i: int) -> Image.Image:
        return self.ds[int(i)]["image"].convert("RGB")


class FileListSource:
    """Flat directory of JPEGs whose names contain the WordNet id (HF tar layout)."""

    def __init__(self, files: list[str]):
        self.files = sorted(files)
        wnids = []
        for f in self.files:
            m = WNID_RE.search(os.path.basename(f))
            if m is None:
                raise ValueError(f"No WordNet id in file name: {f}")
            wnids.append(m.group(1))
        order = sorted(set(wnids))
        if len(order) != 1000:
            print(f"[data] warning: found {len(order)} classes, expected 1000")
        lut = {w: i for i, w in enumerate(order)}
        self.labels = np.asarray([lut[w] for w in wnids], dtype=np.int64)

    def __len__(self):
        return len(self.files)

    def get_pil(self, i: int) -> Image.Image:
        return Image.open(self.files[int(i)]).convert("RGB")


class FolderSource:
    """torchvision ImageFolder layout: root/<wnid>/<image>."""

    def __init__(self, root: str):
        from torchvision.datasets import ImageFolder

        self.ds = ImageFolder(root)
        if len(self.ds.classes) != 1000:
            print(f"[data] warning: found {len(self.ds.classes)} classes, expected 1000")
        self.labels = np.asarray(self.ds.targets, dtype=np.int64)

    def __len__(self):
        return len(self.ds)

    def get_pil(self, i: int) -> Image.Image:
        path, _ = self.ds.samples[int(i)]
        return self.ds.loader(path).convert("RGB")


def open_val(path: str):
    """Auto-detect the layout under `path`."""
    parquet = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
    parquet = [p for p in parquet if "val" in os.path.basename(p).lower()] or parquet
    if parquet:
        return ParquetSource(parquet)
    imgs = [p for p in glob.glob(os.path.join(path, "**", "*"), recursive=True)
            if p.lower().endswith(IMG_EXT)]
    if imgs and all(os.path.dirname(p) == os.path.dirname(imgs[0]) for p in imgs[:100]):
        return FileListSource(imgs)
    if imgs:
        return FolderSource(path)
    found = sorted(os.listdir(path))[:20] if os.path.isdir(path) else []
    raise FileNotFoundError(
        f"No parquet files or images found under {path} (contents: {found}). If the HF download "
        "fetched nothing, check that your account accepted the ImageNet terms on the dataset page, "
        "or point open_val at an ImageFolder (val/<wnid>/*.JPEG) you prepared yourself.")


def download_hf_val(local_dir: str, repo_id: str = "ILSVRC/imagenet-1k", token: str | None = None) -> str:
    """Download only the validation split (about 6.4 GB). Needs an HF token whose account
    has accepted the ImageNet terms on the dataset page. Extracts tar layouts in place."""
    from huggingface_hub import snapshot_download

    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id, repo_type="dataset", local_dir=local_dir,
        allow_patterns=["*val*", "*validation*"], token=token or os.environ.get("HF_TOKEN"),
    )
    for tar_path in glob.glob(os.path.join(local_dir, "**", "*.tar*"), recursive=True):
        out = os.path.join(local_dir, "val_images")
        marker = os.path.join(out, ".extracted")
        if not os.path.exists(marker):
            os.makedirs(out, exist_ok=True)
            print(f"[data] extracting {tar_path}")
            with tarfile.open(tar_path) as tf:
                try:
                    tf.extractall(out, filter="data")
                except TypeError:  # older Python without extraction filters
                    tf.extractall(out)
            open(marker, "w").close()
    return local_dir


# ----------------------------------------------------------------------------- torch dataset
class ValDataset(Dataset):
    def __init__(self, source, indices=None):
        self.src = source
        self.indices = np.arange(len(source)) if indices is None else np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        i = int(self.indices[j])
        return tensor_transform(self.src.get_pil(i)), int(self.src.labels[i]), i


def cropped_pil(source, i: int) -> Image.Image:
    """The exact 224x224 view the model saw."""
    return crop_transform(source.get_pil(i))


# ----------------------------------------------------------------------------- smoke data
def make_fake_imagefolder(root: str, n_classes: int = 4, per_class: int = 12, seed: int = 0) -> str:
    """Tiny synthetic ImageFolder with real WordNet-style folder names, for pipeline tests."""
    rng = np.random.default_rng(seed)
    for c in range(n_classes):
        d = os.path.join(root, f"n{c:08d}")
        os.makedirs(d, exist_ok=True)
        for k in range(per_class):
            arr = (rng.random((300, 360, 3)) * 255).astype(np.uint8)
            arr[..., c % 3] = np.clip(arr[..., c % 3].astype(int) + 60, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(d, f"img_{c}_{k}.JPEG"))
    return root
