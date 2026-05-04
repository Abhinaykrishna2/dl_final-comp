# Environment specification for the videomae stream

This is the exact environment used to produce every aggregated result and figure artifact in this branch. Reproducibility is one-shot: a fresh container that runs
`videomae/scripts/reproduce_main_result.sh` after replicating this environment
should land within the documented tolerance of every reported number.

## System

- **OS:** Ubuntu Server (kernel `5.15.0-170-generic`).
- **Hardware:** 2 x NVIDIA B200 (180 GB each), driver `580.126.09`, CUDA 13.0.
- **CPUs / RAM:** 60 vCPU, 361 GiB RAM, ~860 GB free disk on `/`.
- **Python:** 3.10.12 (system Python, used to bootstrap the venv).

## Python venv

We use a dedicated venv at `/root/envs/physrl` (the name is shared with the
colleague's stream so the repo's `requirements.txt` works for both). Recreate
with:

```bash
apt-get install -y python3-venv python3-pip
python3 -m venv /root/envs/physrl --prompt physrl
source /root/envs/physrl/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

## PyTorch (B200-aware install)

The colleague's `requirements.txt` pins `torch==2.3.0+cu121`, which **will not
run on a B200 + CUDA 13 driver** (no `sm_100` kernels). We install a B200-native
build from the cu128 PyPI index instead:

```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

This pulls **`torch==2.11.0+cu128`** (the version we ran with). Verify by:

```python
import torch
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (10, 0)  # B200 = sm_100
assert "sm_100" in torch.cuda.get_arch_list()
```

## Other Python packages

Install the rest of the deps in one shot (these match the colleague's
`requirements.txt`, with `huggingface_hub` and `hf_xet` added for dataset
download and `matplotlib` + `pandas` for figure regeneration):

```bash
pip install \
    einops omegaconf hydra-core tqdm wandb h5py scikit-learn scikit-image \
    timm psutil 'ruamel.yaml' \
    huggingface_hub hf_xet matplotlib pandas
```

Exact versions used in our run (from `pip freeze`, after the install above):

```
torch==2.11.0+cu128
torchvision==0.26.0+cu128
torchaudio==2.11.0+cu128
numpy==2.2.6
einops==0.8.2
omegaconf==2.3.0
hydra-core==1.3.2
tqdm==4.67.3
wandb==0.26.1
h5py==3.16.0
scikit-learn==1.7.2
scikit-image==0.25.2
timm==1.0.26
psutil==7.2.2
ruamel.yaml==0.19.1
huggingface_hub==1.13.0
hf_xet==1.4.3
matplotlib==3.10.9
pandas==2.3.3
Pillow==12.1.1
```

A complete pinned snapshot lives at `videomae/requirements-lock.txt`.

## Determinism

We fix three sources of seeds at every entry point: Python's `random`, NumPy,
and PyTorch (CPU + CUDA). See `videomae/utils.py::seed_everything`. The default
`--seed 42` is used for both the VideoMAE main run and the supervised baseline;
the mask-ratio ablation uses `--seed 43`.

We do **not** set `torch.use_deterministic_algorithms(True)` for our runs
(efficiency vs. strict bit-reproducibility trade-off). Pass `--deterministic` to
either trainer if exact reproducibility is required.

## Experiment tracking

Every trainer writes:

* **JSON logs** -- always written, no extra setup. Files:
  - `videomae/artifacts/<run>/train_config.json` -- frozen hyperparameter dump
  - `videomae/artifacts/<run>/history.json` -- per-epoch metrics
  - `videomae/artifacts/<run>/{linear_probe,knn,analysis}/metrics.json` -- eval results
* **W&B offline-mode logs** -- written by default to `videomae/artifacts/<run>/wandb/`
  as `offline-run-...` directories. No account or API key is required at training
  time. Optionally sync them to wandb.ai later via:
  ```bash
  pip install wandb
  wandb login                        # only if syncing
  cd videomae/artifacts/<run> && wandb sync wandb/offline-run-*
  ```

## Dataset

Downloaded with the colleague's `data_loader.py` (HuggingFace `polymathic-ai/active_matter`):

```bash
python data_loader.py --root /root/data --workers 16
python count_dataset_files.py /root/data
```

Expected layout:

```
/root/data/
  train/   175 trajectories across 45 .hdf5 files  =>  8750 sliding windows
  valid/    24 trajectories across 16 .hdf5 files  =>  1200 sliding windows
  test/     26 trajectories across 21 .hdf5 files  =>  1300 sliding windows
```

Total disk footprint ~52 GB.
