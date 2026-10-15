# already-clean-check

Independent, preregistered tests of claims in Goodfire's essay
[*Neural Networks Are Already Clean*](https://already-clean.goodfire.com/) (2026).
The essay ships no code, weights or training recipes, and its companion paper is not out yet,
so the goal is not a figure by figure replication but sharp tests of its most falsifiable claims.

| Stage | Claim | Test | Compute |
|-------|-------|------|---------|
| **0** (this repo now) | The canonical "cat + car" neuron InceptionV1 `mixed4e:55` is one V-shape detector with round shapes on its negative pole | circuit weights, preregistered class level test on ImageNet val, synthetic shape tuning curves, feature vis | < 1 GPU hour |
| 1 | Under capacity pressure neurons get coarser instead of superposed (3b squeezed 480 → 10) | retrain InceptionV1 baseline + squeezed; SAE **and** linear population decoding of curve orientation from the 10-d layer | tens of H100 hours |
| 2 | Polysemanticity is mostly "recursive generalization" (local growth over training) | track last-layer neurons across checkpoints; is each newly recruited class semantically near the old ones? | reuses Stage 1 checkpoints |

## Layout

```
acc/models.py     InceptionV1 (original Lucid weights via lucent) + pre-ReLU recorder
acc/data.py       ImageNet val from Hugging Face (parquet or tar) or an ImageFolder
acc/cache.py      resumable, sharded activation cache (pre-ReLU spatial max/min per channel)
acc/circuits.py   exact 1x1 weights and expanded 3x3/5x5 weights between mixed layers
acc/analysis.py   top/bottom examples, class composition, preregistered class-level test
acc/stimuli.py    synthetic V / triangle / U / O / disk / line / square stimuli and tuning curves
acc/stage0.py     everything above for one unit, used by the notebook, the CLI and Modal
notebooks/00_stage0_colab.ipynb   the Colab A100 path
modal_app.py      the Modal H100 path
prereg/4e55.json  predictions (edit + commit before running the analysis)
```

## Quick start: Colab A100

1. Accept the terms on the Hugging Face `ILSVRC/imagenet-1k` dataset page; make a read token.
2. Colab → key icon (Secrets) → add `HF_TOKEN`, allow notebook access.
3. Put this folder in Google Drive at `MyDrive/already-clean-check` (or push to GitHub and set `REPO_URL`).
4. Open `notebooks/00_stage0_colab.ipynb` in Colab, Runtime → A100, run top to bottom.
   **Edit and commit `prereg/4e55.json` + `PREREGISTRATION.md` before section 4.**

The cache lives on Drive in 5k-image shards, so a disconnect only loses the shard in progress.

## When Colab compute runs out: Modal H100

```bash
pip install modal && modal setup
modal secret create huggingface HF_TOKEN=hf_xxx
modal run modal_app.py --step download      # val set -> Modal Volume (CPU only)
modal run modal_app.py --step stage0        # cache + analyses on one H100, commits after every shard
modal volume get already-clean-check runs/stage0 ./runs/stage0
```

## Local smoke test (CPU, synthetic data, checks the plumbing only)

```bash
pip install -e .
python scripts/run_stage0.py --data fake --out /tmp/smoke --limit 40 --shard-size 16 --workers 2
```

## Pilot observations (CPU sandbox, no natural images yet)

* Indexing matches the essay: `mixed4d:493` is the #1 input weight into `mixed4e:55`
  (477 is #8, 201 is #15); `mixed5a:94` is the #1 direct reader and `mixed5a:352` the #1 expanded reader.
* Synthetic shapes: V, triangle, square, line **and U** drive 4e:55 strongly; O and disk barely do.
  So "converging arms / corners vs closed round contour" is a live alternative to "pointed vs round".
  Not yet normalized against natural images.
