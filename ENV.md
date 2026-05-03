# Environment

This project was developed and evaluated with Python and PyTorch. The final CNext-U-Net run was trained on a rented NVIDIA B200 server using bfloat16 AMP. The exact package pins used by this repository are in `requirements.txt`.

## Core Runtime

- Python: 3.10+ recommended for training servers.
- Local verification environment observed by Codex: Python 3.13.7 on macOS arm64.
- Training framework: PyTorch.
- Data format: HDF5 via `h5py`.
- Downstream evaluation: PyTorch and scikit-learn.
- Logging: Weights & Biases optional; scripts also write JSON metrics and stdout logs.

## Install

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

On Blackwell GPUs such as B200, use a PyTorch build with CUDA support compatible with that hardware. The final server run used a newer CUDA/PyTorch stack than the conservative pins in `requirements.txt`; see `cnext_logs/REPRODUCE.md`.

## Dataset

Download or place the `active_matter` dataset under a root directory containing `train`, `valid`, and `test` splits, then set:

```bash
export DATA_ROOT=/path/to/active_matter
```

If the dataset is not already present, use:

```bash
python data_loader.py --root "$DATA_ROOT" --workers 16
```

## Final Evaluation

The final frozen encoder and embeddings are committed under `artifacts/`. To reproduce downstream metrics from committed artifacts:

```bash
python -m active_matter_ssl.sweep_linear_probe \
    --train-file artifacts/emb_cnext_unet96_avg/train.npz \
    --valid-file artifacts/emb_cnext_unet96_avg/valid.npz \
    --test-file artifacts/emb_cnext_unet96_avg/test.npz \
    --out-dir artifacts/lp_cnext_unet96_avg \
    --epochs 200 \
    --patience 25

python -m active_matter_ssl.eval_knn \
    --train-file artifacts/emb_cnext_unet96_avg/train.npz \
    --valid-file artifacts/emb_cnext_unet96_avg/valid.npz \
    --test-file artifacts/emb_cnext_unet96_avg/test.npz \
    --out-dir artifacts/knn_cnext_unet96_avg \
    --backend torch
```

For the full train/export/evaluate pipeline, use:

```bash
bash experiments/03_cnext_unet_forecasting_final/run_full_pipeline.sh
```

## Build The ICML Report

The final report uses the ICML 2026 style files in `icml2026/`.

```bash
pdflatex -interaction=nonstopmode final_report.tex
bibtex final_report
pdflatex -interaction=nonstopmode final_report.tex
pdflatex -interaction=nonstopmode final_report.tex
```

This produces `final_report.pdf`.
