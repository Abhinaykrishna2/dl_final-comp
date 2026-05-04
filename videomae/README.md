# `videomae/` - VideoMAE / SimMIM-Hybrid SSL Stream for Active Matter

A self-contained Python package implementing a **VideoMAE / SimMIM-hybrid masked-autoencoder
SSL track** on the `active_matter` dataset, an **end-to-end supervised baseline**, and a
**representation-quality analysis suite** (channel/frame ablations, top-PC alignment,
class separability, CKA, per-channel reconstruction MSE). All code and scripts for this stream
live under this directory; the parallel JEPA SSL stream lives in `physrl/`.

Primary outputs live under **`videomae/artifacts/`**: JSON metrics, regenerated figures
(`artifacts/figures/figures.pdf`), analysis PDFs, plus training logs. Large checkpoints and
embeddings are gitignored until you reproduce locally.

---

## Table of contents

1. [Quickstart (5 commands)](#quickstart-5-commands)
2. [What you should see](#what-you-should-see)
3. [Setup](#setup)
4. [Dataset](#dataset)
5. [Running the pipeline](#running-the-pipeline)
6. [Experiment tracking (Weights & Biases and logs)](#experiment-tracking-weights--biases-and-logs)
7. [File and script index](#file-and-script-index)
8. [Results map (claim to artifact)](#results-map-claim-to-artifact)
9. [Hyperparameters and citations](#hyperparameters-and-citations)
10. [Troubleshooting](#troubleshooting)
11. [File provenance](#file-provenance)
12. [Reproducibility receipts](#reproducibility-receipts)

---

## Quickstart (5 commands)

Run from a fresh Ubuntu box with 1 or 2 NVIDIA GPUs (B200 / A100 / L40S all tested-or-equivalent).

```bash
# 1. Clone repo and create the venv
git clone <repo_url> dl_final-comp && cd dl_final-comp
git checkout videomae
python3 -m venv /root/envs/physrl --prompt physrl
source /root/envs/physrl/bin/activate

# 2. Install PyTorch (B200-aware) and the rest of the deps
pip install --upgrade pip wheel setuptools
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r videomae/requirements-lock.txt

# 3. Download the active_matter dataset (~52 GB to /root/data)
python data_loader.py --root /root/data --workers 16
python count_dataset_files.py /root/data

# 4. Reproduce the entire pipeline (training + eval + figures)
bash videomae/scripts/reproduce_main_result.sh

# 5. Inspect aggregates and the 4-panel figure
xdg-open videomae/artifacts/aggregated/aggregated_results.md || true
xdg-open videomae/artifacts/figures/figures.pdf || open videomae/artifacts/figures/figures.pdf
```

If you only want to **regenerate figures from the JSON metrics committed to the repo**
(no GPU, no dataset, no training), skip steps 3 and 4 and run:

```bash
python -m videomae.aggregate_results --runs videomae/artifacts/* --out-dir videomae/artifacts/aggregated
python -m videomae.plot_from_json \
    --runs $(for r in videomae/artifacts/{supervised,videomae_main,videomae_mask75,videomae_mask90,videomae_mask90_s45,videomae_main_50ep_s46}; do echo "$r:$(basename $r)"; done) \
    --out-dir videomae/artifacts/figures
```

---

## What you should see

### Runtime, on 2x NVIDIA B200 180GB

| Step                                         | Wall-clock     | GPU footprint   |
|----------------------------------------------|---------------:|-----------------|
| Dataset download (`data_loader.py`)          | ~30 min        | none, network   |
| `videomae_main` (25 epochs, mask 0.60)       | ~3 h           | 1 GPU, ~50 GB   |
| `supervised` (30 epochs)                     | ~1 h           | 1 GPU, ~25 GB   |
| 3 multi-seed VideoMAE retrains (s43, s44)    | ~6 h           | 1 GPU, sequential |
| 3 multi-seed supervised retrains             | ~3 h           | 1 GPU, sequential |
| Mask-ratio ablations (mask 0.75, 0.90)       | ~1.5 h         | warm-started    |
| `videomae_main_50ep_s46` (decoupling test)   | ~6 h           | 1 GPU           |
| Eval (export + linear probe + kNN + analysis) per encoder | ~10 min  | 1 GPU |
| 5 cross-encoder analyses (PC, CKA, recon, separability, ablations) | ~30 min | 1 GPU |
| Aggregate metrics + figure regen (`plot_from_json`) | ~1 min     | none            |
| **Full `reproduce_main_result.sh` end-to-end** | **~24 h**    | **2 GPUs**      |

### Disk

| Item                                     | Size       |
|------------------------------------------|-----------:|
| `active_matter` HDF5 dataset             | ~52 GB     |
| Per-encoder full checkpoints (`*.pt`)    | ~250 MB    |
| Per-encoder embeddings (`*.npz`)         | ~150 MB    |
| All artifacts after a full reproduction  | ~10&ndash;15 GB |
| JSON metadata committed to the branch    | ~5 MB      |

### Headline numbers (test split, z-scored MSE on (alpha, zeta))

| Encoder                     | Linear probe MSE | kNN MSE | EffRank (valid) |
|-----------------------------|----------------:|--------:|----------------:|
| Supervised (3-seed mean)    | **0.106**       | 0.106   | 3.55            |
| VideoMAE mask=0.60 (3-seed) | 0.317           | 0.562   | 8.09            |
| VideoMAE mask=0.75          | 0.299           | 0.672   | 8.87            |
| VideoMAE mask=0.90 (2-seed) | **0.255**       | 0.503   | 7.92            |

Story: supervised is the upper-bound anchor; among SSL pixel-reconstruction objectives
the mask ratio sweep is monotonic (0.60 &rarr; 0.75 &rarr; 0.90 = 0.317 &rarr; 0.299
&rarr; 0.255). VideoMAE keeps a much richer representation (effective rank ~8 vs ~3.5 for
supervised) but at the cost of target alignment. See `videomae/artifacts/aggregated/aggregated_results.md`
and `videomae/artifacts/figures/figures.pdf`, and compare against the JEPA stream on `main`.

---

## Setup

### System tested

- **OS:** Ubuntu Server (kernel 5.15.0-170-generic).
- **Hardware:** 2x NVIDIA B200 (180 GB each), driver 580.126.09, CUDA 13.0.
- **CPU / RAM:** 60 vCPU, 361 GiB RAM, ~860 GB free disk on `/`.
- **Python:** 3.10.12.

The exact environment used to reproduce the metrics and figures summarized here is documented in
[`ENV.md`](./ENV.md); the pinned package list is in
[`requirements-lock.txt`](./requirements-lock.txt).

### Python venv

```bash
apt-get install -y python3-venv python3-pip
python3 -m venv /root/envs/physrl --prompt physrl
source /root/envs/physrl/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

### PyTorch (B200-aware)

The repo's top-level `requirements.txt` pins `torch==2.3.0+cu121`, which **will not run on
a B200 + CUDA 13 driver** (no `sm_100` kernels). Install a B200-native build first:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

This pulls `torch==2.11.0+cu128`. Verify with:

```python
import torch
assert torch.cuda.is_available()
assert "sm_100" in torch.cuda.get_arch_list()  # B200 = sm_100
```

For non-B200 GPUs (A100 / L40S), the default `requirements.txt` works.

### Other dependencies

```bash
pip install -r videomae/requirements-lock.txt
```

This is a frozen `pip freeze` snapshot from the runs that produced this branch's headline numbers. Alternative,
unpinned install:

```bash
pip install \
    einops omegaconf hydra-core tqdm wandb h5py scikit-learn scikit-image \
    timm psutil 'ruamel.yaml' \
    huggingface_hub hf_xet matplotlib pandas
```

---

## Dataset

The `active_matter` dataset is downloaded with the repo's top-level `data_loader.py`
(HuggingFace `polymathic-ai/active_matter`):

```bash
python data_loader.py --root /root/data --workers 16
python count_dataset_files.py /root/data
```

Expected layout after download:

```
/root/data/
  train/   175 trajectories across 45 .hdf5 files  ->  8,750 sliding windows
  valid/    24 trajectories across 16 .hdf5 files  ->  1,200 sliding windows
  test/     26 trajectories across 21 .hdf5 files  ->  1,300 sliding windows
```

Total disk footprint: ~52 GB.

If you put the dataset somewhere else, set `DATA_ROOT=/path/to/data` before running any
of the launch / reproduction scripts.

---

## Running the pipeline

There are four supported entry points, ordered by how much you want to run.

### A. Full reproduction (single command, ~24 h on 2x B200)

```bash
bash videomae/scripts/reproduce_main_result.sh
```

The script is **idempotent**: every step skips if its output already exists. If a training
crashes or an SSH connection drops, just re-run and it will pick up where it left off.

Optional environment overrides: `DATA_ROOT`, `ART_ROOT`, `GPU0`, `GPU1`.

### B. Detached training jobs (survive SSH drops)

For long-running trainings, the `launch_*.sh` scripts use `setsid nohup` to fully
disconnect the child process from the shell, so it survives SSH drops, terminal closes,
and agent restarts. They also **auto-resume** from `last.pt` if a checkpoint exists.

```bash
bash videomae/scripts/launch_videomae.sh           # ~3 h on GPU 0
bash videomae/scripts/launch_supervised.sh         # ~1 h on GPU 1

# Multi-seed stability runs:
bash videomae/scripts/launch_videomae_seed43.sh    # ~3 h
bash videomae/scripts/launch_supervised_seed43.sh  # ~1 h

# Mask-ratio ablations (warm-started, ~35 min each):
bash videomae/scripts/launch_ablation_mask75.sh
bash videomae/scripts/launch_ablation_mask90.sh
```

PIDs are written to `videomae/artifacts/logs/<run>.pid` and stdout/stderr is captured to
`videomae/artifacts/logs/<run>.out`. To check on a job: `tail -f videomae/artifacts/logs/<run>.out`.

### C. Eval only (after training finishes)

Run linear probe + kNN + analysis on every encoder that has an `encoder_best.pt`:

```bash
bash videomae/scripts/run_eval.sh                  # auto-detects all encoders
# or pin a subset:
bash videomae/scripts/run_eval.sh videomae_main supervised
```

### D. Just regenerate figures from committed JSON

No GPU, no dataset, no training. Uses only the JSON metrics already committed to the repo.

```bash
python -m videomae.aggregate_results \
    --artifacts-dir videomae/artifacts \
    --out-dir videomae/artifacts/aggregated

python -m videomae.plot_from_json \
    --runs $(ls -d videomae/artifacts/*/ | grep -E '(supervised|videomae)' | sed 's:/$::' | awk -F/ '{print $0":"$NF}') \
    --out-dir videomae/artifacts/figures

python -m videomae.plot_ablations \
    --ablations-dir videomae/artifacts/ablations
```

### Single-trainer minimal command

```bash
# Train one VideoMAE encoder from scratch:
python -m videomae.train_videomae \
    --data-root /root/data \
    --out-dir videomae/artifacts/my_run \
    --epochs 25 --batch-size 64 --num-workers 6 \
    --mask-ratio 0.60 --tube-size 16,32,32 \
    --amp --amp-dtype bfloat16 --seed 42

# Then evaluate it:
python -m videomae.export_embeddings \
    --data-root /root/data \
    --checkpoint videomae/artifacts/my_run/encoder_best.pt \
    --out-dir videomae/artifacts/my_run/embeddings \
    --split train valid test --amp --amp-dtype bfloat16

python -m videomae.sweep_linear_probe \
    --train-file videomae/artifacts/my_run/embeddings/train.npz \
    --valid-file videomae/artifacts/my_run/embeddings/valid.npz \
    --test-file  videomae/artifacts/my_run/embeddings/test.npz \
    --out-dir    videomae/artifacts/my_run/linear_probe \
    --epochs 200 --patience 25
```

Every CLI tool in this package supports `--help` for full options.

---

## Experiment tracking (Weights & Biases and logs)

Training runs use **experiment tracking and plain-text logs** together:

- **Weights & Biases:** Both [`train_videomae.py`](./train_videomae.py) and [`train_supervised.py`](./train_supervised.py) log epochs and summaries through W&B. The default is **`--wandb-mode offline`** (no API key, no cloud upload during training); runs write under each run's output directory (`videomae/artifacts/<run>/wandb/`). Use `--wandb-mode online` to stream to wandb.ai, or `--wandb-mode disabled` to skip W&B entirely. See [Troubleshooting, item 5 (W&B asks for an API key)](#5-wb-asks-for-an-api-key-on-training-start) below.
- **JSON logs:** Every trainer also writes **`history.json`** (per-epoch metrics), **`train_config.json`** (full argparse dump), and on best checkpoints merges in **`wandb_run_id`** when W&B was active. Evaluation stages write **`metrics.json`** under each subfolder (`linear_probe/`, `knn/`, `analysis/`).
- **What graders see in-repo:** Offline W&B run directories are **gitignored** (see `.gitignore`: `videomae/artifacts/**/wandb/`) alongside large checkpoints so the branch stays lightweight. Committed **`videomae/artifacts/logs/*.out`** captures `nohup` / script stdout including W&B status lines (`wandb: Tracking run...`, `wandb sync ...`). For curve-level verification without retraining, prefer **`history.json`** and the [**Results map**](#results-map-claim-to-artifact) table.

---

## File and script index

### Trainers

| File | Purpose |
|------|---------|
| [`train_videomae.py`](./train_videomae.py) | VideoMAE / SimMIM-hybrid masked-autoencoder SSL trainer. Tube masking, learnable mask token, lightweight transposed-conv decoder, BF16 + DDP, JSON + W&B-offline logging, full resume support. |
| [`train_supervised.py`](./train_supervised.py) | End-to-end supervised baseline: same encoder + linear head, MSE on z-scored (alpha, zeta) labels. JSON + W&B-offline logging (same `--wandb-mode` flags), full resume support. |

### Evaluation

| File | Purpose |
|------|---------|
| [`export_embeddings.py`](./export_embeddings.py) | Run a frozen encoder on every split, dump pooled feature vectors to `.npz`. |
| [`sweep_linear_probe.py`](./sweep_linear_probe.py) | Grid-search a linear regression probe (lr / wd / batch-size / feature-norm). |
| [`train_linear_probe.py`](./train_linear_probe.py) | Single linear-probe configuration trainer (used inside the sweep). |
| [`eval_knn.py`](./eval_knn.py) | kNN regression evaluation (sweep n_neighbors x weights x metric x feature-norm). |

### Representation-quality analyses

| File | Purpose |
|------|---------|
| [`analyze_representations.py`](./analyze_representations.py) | Per-dim std, effective rank, participation ratio, condition number, mean abs off-diag correlation, Epps-Pulley distance to N(0, I). |
| [`run_ablations.py`](./run_ablations.py) | Channel-importance (zero each input channel) and frame-budget (4, 8, 12, 16 frames) ablations. |
| [`pc_alignment.py`](./pc_alignment.py) | Top-K principal-component linear-probe sweep. The headline rank-vs-alignment quantitative test. |
| [`cka_compare.py`](./cka_compare.py) | Pairwise linear CKA (Kornblith et al. 2019) between encoder embeddings. |
| [`class_separability.py`](./class_separability.py) | 45-class Fisher score and nearest-centroid accuracy. |
| [`per_channel_recon_mse.py`](./per_channel_recon_mse.py) | VideoMAE-only: per-channel masked-position reconstruction MSE. |

### Visualization

| File | Purpose |
|------|---------|
| [`visualize_embeddings.py`](./visualize_embeddings.py) | PCA-2D and t-SNE-2D plots of frozen encoder embeddings, colored by alpha and zeta. |
| [`visualize_reconstructions.py`](./visualize_reconstructions.py) | Side-by-side input / masked / reconstruction / residual figures for VideoMAE. |
| [`plot_from_json.py`](./plot_from_json.py) | Single 2x2 figure: loss curves, per-dim std, isotropy bars, frozen-eval MSE bars. |
| [`plot_ablations.py`](./plot_ablations.py) | Grouped bar chart of channel deltas + line plot of frame-budget saturation. |
| [`aggregate_results.py`](./aggregate_results.py) | Walks `artifacts/<run>/`, gathers headline numbers into `aggregated_results.{json,md}`. |

### Models

| File | Purpose |
|------|---------|
| [`models.py`](./models.py) | `ConvEncoder` (mirrored from `physrl/`); new `ConvNeXtDecoder` and `VideoMAEModel` for this stream. |
| [`losses.py`](./losses.py) | Mirrored from `physrl/`: VICReg, SIGReg, JEPA loss kernels. Used here only as the diagnostic kernel inside `analyze_representations.py`. |

### Data and utilities

| File | Purpose |
|------|---------|
| [`data.py`](./data.py) | Sliding-window HDF5 dataset for `active_matter`. Mirrored from `physrl/`. |
| [`utils.py`](./utils.py) | Seeding, label normalization, atomic checkpoint save, pooling, MSE reporting. Mirrored from `physrl/`. |
| [`__init__.py`](./__init__.py) | Re-exports `LABEL_NAMES`. Mirrored from `physrl/`. |

### Scripts

| Script | Purpose |
|--------|---------|
| [`scripts/reproduce_main_result.sh`](./scripts/reproduce_main_result.sh) | One-shot, idempotent end-to-end pipeline. |
| [`scripts/launch_videomae.sh`](./scripts/launch_videomae.sh) | Detached VideoMAE main run (mask 0.60, seed 42). |
| [`scripts/launch_videomae_seed43.sh`](./scripts/launch_videomae_seed43.sh) | Multi-seed VideoMAE main retrain (seed 43). |
| [`scripts/launch_supervised.sh`](./scripts/launch_supervised.sh) | Detached supervised baseline (seed 42). |
| [`scripts/launch_supervised_seed43.sh`](./scripts/launch_supervised_seed43.sh) | Supervised retrain (seed 43). |
| [`scripts/launch_ablation_mask75.sh`](./scripts/launch_ablation_mask75.sh) | Warm-started mask-ratio ablation (0.75). |
| [`scripts/launch_ablation_mask90.sh`](./scripts/launch_ablation_mask90.sh) | Warm-started mask-ratio ablation (0.90). |
| [`scripts/run_eval.sh`](./scripts/run_eval.sh) | Eval (export + linear probe + kNN + analysis) on every encoder with an `encoder_best.pt`. |

---

## Results map (claim to artifact)

Use this table to verify a specific number, figure, or claim against the committed (or reproduced) artifacts.

| Claim / element                             | Source artifact |
|---------------------------------------------|-----------------|
| Main results (aggregate)                    | `videomae/artifacts/aggregated/aggregated_results.{json,md}` |
| Per-encoder linear-probe / kNN / analysis numbers | `videomae/artifacts/<run>/{linear_probe,knn,analysis}/metrics.json` |
| Per-epoch loss curves                       | `videomae/artifacts/<run>/history.json` |
| Hyperparameter dump per run                 | `videomae/artifacts/<run>/train_config.json` |
| CKA heatmap                              | `videomae/artifacts/cka/{cka_valid.json,cka_valid.pdf}` |
| Top-PC alignment curve                       | `videomae/artifacts/pc_alignment/{pc_alignment.json,pc_alignment.pdf}` |
| Channel-importance ablation                | `videomae/artifacts/ablations/<run>/channel_ablation.json` + `videomae/artifacts/ablations/channel_ablation.pdf` |
| Frame-budget ablation                      | `videomae/artifacts/ablations/<run>/frame_ablation.json` + `videomae/artifacts/ablations/frame_ablation.pdf` |
| Per-channel reconstruction MSE              | `videomae/artifacts/per_channel_recon_mse/{per_channel_recon_mse.json,per_channel_recon_mse.pdf}` |
| Per-class separability (Fisher / NC)        | `videomae/artifacts/class_separability/{class_separability.json,class_separability.pdf}` |
| Embedding visualizations (PCA / t-SNE)      | `videomae/artifacts/embeddings_viz/embeddings_valid.pdf` |
| Reconstruction visualizations               | `videomae/artifacts/recon_viz/recon_<run>_sample<idx>_t<t>.pdf` |
| Combined 4-panel summary figure                          | `videomae/artifacts/figures/figures.pdf` |

Large binary artifacts (`*.pt` checkpoints, `*.npz` embeddings, `wandb/` offline runs) are
gitignored to stay under GitHub's 100 MB-per-file limit; everything you need to verify a
headline number or analysis plot is in the JSON and analysis PDF files. To regenerate the binaries, follow steps 3 and
4 of [Quickstart](#quickstart-5-commands).

---

## Hyperparameters and citations

Every non-default choice is justified by the source paper.

| Choice | Value | Citation |
|--------|-------|----------|
| Encoder | 3D ConvNeXt, dims=(32,64,128,256,256), blocks=(2,2,4,8,2), stem=2 | matches the JEPA stream's encoder exactly so the SSL-objective comparison is causally clean |
| Mask ratio (main) | 0.60 | SimMIM Tab. 1 (optimal at patch-size = encoder downsampling) |
| Mask ratio (ablation) | 0.75, 0.90 | VideoMAE Sec 4.1.2 (optimal for video data) |
| Tube size | (16, 32, 32) | SimMIM Sec 4.1.2 (align with encoder stride) |
| Mask token | learnable per-channel | SimMIM Sec 3.2 (required for non-ViT encoders) |
| Decoder | 5-stage transposed conv, ~0.35 M params | SimMIM Tab. 2 (lightweight head transfers better) |
| Loss target | per-tube z-scored pixels | VideoMAE Sec 3.3 / Tab. 1c |
| Loss scope | masked positions only | SimMIM Tab. 4 (prediction beats reconstruction) |
| Loss type | MSE | VideoMAE Tab. 1f |
| Optimizer | AdamW, lr 1.5e-4, wd 0.05 | VideoMAE Sec 4.1 / SimMIM Sec 4.1 |
| Schedule | cosine, 2-epoch warmup | both papers |
| Precision | BF16 autocast | B200 native |

References:

- Tong et al., **"VideoMAE: Masked Autoencoders Are Data-Efficient Learners for Self-Supervised Video Pre-Training"**, NeurIPS 2022, [arXiv:2203.12602](https://arxiv.org/abs/2203.12602).
- Xie et al., **"SimMIM: A Simple Framework for Masked Image Modeling"**, CVPR 2022, [arXiv:2111.09886](https://arxiv.org/abs/2111.09886).
- Kornblith et al., **"Similarity of Neural Network Representations Revisited"**, ICML 2019, [arXiv:1905.00414](https://arxiv.org/abs/1905.00414).

---

## Troubleshooting

The five gotchas we hit while building this; in rough order of likelihood you'll hit them too.

### 1. `RuntimeError: CUDA error: no kernel image is available for execution on the device`

You're on a B200 (or other Blackwell GPU) with the default `torch==2.3.0+cu121`.
Reinstall with the cu128 wheels per [Setup -> PyTorch (B200-aware)](#pytorch-b200-aware).

### 2. `_pickle.UnpicklingError: Weights only load failed ... GLOBAL pathlib.PosixPath`

PyTorch >= 2.6 defaults `weights_only=True` in `torch.load`, which refuses to deserialize
the `pathlib.PosixPath` we store in the checkpoint config. All checkpoint loaders in this
package already pass `weights_only=False`; if you write a new loader, do the same.

### 3. OOM kill of dataloader workers (host RAM, not GPU)

Symptom: jobs die with `Killed` and `dmesg` shows `Out of memory: Killed process` for
Python workers. Cause: too many concurrent dataloader processes (each loads HDF5 chunks
into RAM). Mitigation:

- Cap to **`--num-workers 6`** for VideoMAE and **`--num-workers 4`** for the supervised
  trainer when running both concurrently on a single node.
- Don't run `sha256sum /root/data/**/*.hdf5` while training jobs are active.
- Total worker count across all jobs should stay below ~12 on a 60-vCPU / 361-GiB-RAM box.

### 4. SSH drop kills detached training

Symptom: training process dies when you `exit` your SSH session. Cause: process was
attached to the controlling terminal. Fix: always launch via the
`scripts/launch_*.sh` wrappers, which use `setsid nohup ... & disown` to fully detach.
Trainers checkpoint every epoch and auto-resume from `last.pt`, so even a hard kill loses
at most one epoch.

### 5. W&B asks for an API key on training start

Symptom: `wandb` prompts for login at startup and blocks training. Cause: a previous
`wandb login` was run in this environment with `mode=online`. Fix: trainers default to
`--wandb-mode offline`, which writes runs to `videomae/artifacts/<run>/wandb/` without any
network call. To upload them later: `cd videomae/artifacts/<run> && wandb sync wandb/offline-run-*`.

---

## File provenance

To keep the SSL-objective comparison causally clean, the encoder used by this stream is
**byte-identical** to the one used by the JEPA stream in `physrl/`. Concretely:

- **Mirrored verbatim** from `physrl/` (only `from .foo` &rarr; `from videomae.foo` import
  rewrites): `__init__.py`, `data.py`, `utils.py`, `losses.py`, `eval_knn.py`,
  `export_embeddings.py`, `sweep_linear_probe.py`, `train_linear_probe.py`. These exist as
  copies so this package is independently runnable.
- **Mixed**: `models.py` keeps `LayerNorm`, `ConvEncoder`, `ConvPredictor`, `JepaModel`,
  `SigRegJepaModel` byte-identical; the new `ConvNeXtDecoder` and `VideoMAEModel` (lines
  362+) are added at the bottom.
- **New for this stream** (everything else in `videomae/`): the two trainers
  (`train_videomae.py`, `train_supervised.py`) and the entire analysis/visualization
  suite.

The JEPA half of the joint comparison (SIGReg-JEPA, VICReg-JEPA) is in
`physrl/` on the `main` branch and contributes headline comparison rows for those methods. A separate
CNext-UNet next-frame-forecasting baseline lives in `cnext_logs/REPRODUCE.md` on `main`
(switch branches with `git checkout main` to view it).

---

## Reproducibility receipts

| Item | Value |
|------|-------|
| Branch                       | `videomae` |
| PyTorch                      | 2.11.0+cu128 |
| Python                       | 3.10.12 |
| Dataset                      | `polymathic-ai/active_matter` (HuggingFace) |
| Default seed                 | 42 (multi-seed retrains use 43, 44, 45, 46) |
| Determinism                  | `seed_everything` fixes Python / NumPy / PyTorch (CPU + CUDA). For bit-exact, pass `--deterministic` to either trainer. |
| Pinned package list          | [`requirements-lock.txt`](./requirements-lock.txt) |
| Full environment spec        | [`ENV.md`](./ENV.md) |
| Idempotent reproduction      | [`scripts/reproduce_main_result.sh`](./scripts/reproduce_main_result.sh) |
