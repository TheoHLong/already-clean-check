"""Stage 0 on Modal H100 (use when Colab compute runs out).

One time setup (on your Mac):
    pip install modal && modal setup
    modal secret create huggingface HF_TOKEN=hf_xxx      # token of an account that accepted the ImageNet terms

Run from the repo root:
    modal run modal_app.py --step download     # ImageNet val -> Volume (CPU only, no GPU billed)
    modal run modal_app.py --step stage0       # cache activations + analyses on one H100
    modal volume get already-clean-check runs/stage0 ./runs/stage0   # pull results to your Mac
"""
import modal

app = modal.App("already-clean-check")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "torch-lucent==0.1.8", "datasets", "huggingface_hub",
                 "pandas", "matplotlib", "pillow", "numpy")
    .env({"MPLBACKEND": "Agg"})
    .add_local_dir("prereg", remote_path="/root/prereg")
    .add_local_python_source("acc")
)
vol = modal.Volume.from_name("already-clean-check", create_if_missing=True)
VAL_DIR = "/vol/imagenet_val"
RUN_DIR = "/vol/runs/stage0"


@app.function(image=image, volumes={"/vol": vol}, secrets=[modal.Secret.from_name("huggingface")],
              cpu=8, memory=16384, timeout=2 * 60 * 60)
def download_val():
    from acc.data import download_hf_val

    download_hf_val(VAL_DIR)
    vol.commit()


@app.function(image=image, gpu="H100", volumes={"/vol": vol}, cpu=16, memory=65536, timeout=3 * 60 * 60)
def stage0(unit: str = "mixed4e:55", layers: str = "mixed4d,mixed4e,mixed5a", batch_size: int = 512,
           featvis: bool = True):
    import os

    from acc.cache import cache_activations
    from acc.data import open_val
    from acc.models import load_inceptionv1
    from acc.stage0 import run_all

    source = open_val(VAL_DIR)
    model = load_inceptionv1("cuda")
    cache_dir = os.path.join(RUN_DIR, "cache_inceptionv1")
    cache_activations(model, source, layers.split(","), cache_dir, batch_size=batch_size,
                      num_workers=14, device="cuda", on_shard_done=vol.commit)  # resumable across runs
    summary = run_all(model, source, cache_dir, unit, "/root/prereg/4e55.json",
                      os.path.join(RUN_DIR, unit.replace(":", "_")), device="cuda", featvis=featvis)
    vol.commit()
    return summary


@app.local_entrypoint()
def main(step: str = "stage0", unit: str = "mixed4e:55"):
    import json

    if step in ("download", "all"):
        download_val.remote()
    if step in ("stage0", "all"):
        print(json.dumps(stage0.remote(unit=unit), indent=2, default=float))
