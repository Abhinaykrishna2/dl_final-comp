# CNext-UNet reproduction recipe

The CNext-UNet forecaster was trained on a private 1x B200 server, with
`DATA_ROOT=/root/data` instead of the HPC `/scratch/$USER/data` path. Below is
the exact sequence of commands a grader can run to reproduce the result. All
commands assume:

- The dataset is downloaded to `$DATA_ROOT` (use `python data_loader.py
  --root $DATA_ROOT --workers 16` if not already present).
- The Python environment has PyTorch 2.7+ with CUDA 12.8+ wheels (Blackwell
  support) and the dependencies in `requirements.txt`.

```bash
cd dl_final-comp
git checkout main
export DATA_ROOT=/root/data        # change to /scratch/$USER/data on HPC
export RUN_ROOT=/root/dl_final-comp/artifacts/cnext_unet96

# 1. Train the CNext-U-Net forecaster (50 epochs, ~30 min on 1x B200).
python -m active_matter_ssl.train_cnext_forecaster \
    --data-root "$DATA_ROOT" --out-dir "$RUN_ROOT" \
    --epochs 50 --batch-size 16 --num-workers 4 --prefetch-factor 4 \
    --resolution 96 --context-frames 4 --target-frames 1 \
    --lr 3e-4 --weight-decay 5e-4 \
    --amp --amp-dtype bfloat16 --save-every 2 --seed 42 \
    --wandb-mode disabled

# 2. Export frozen-encoder embeddings on every split (~3 min).
python -m active_matter_ssl.export_cnext_embeddings \
    --data-root "$DATA_ROOT" \
    --checkpoint "$RUN_ROOT/encoder_best.pt" \
    --out-dir artifacts/emb_cnext_unet96_avg \
    --batch-size 64 --pool avg --clip-frames 16 --window-stride 1 --amp

# 3. Linear-probe sweep (~5 min, 36 trials with patience=25).
python -m active_matter_ssl.sweep_linear_probe \
    --train-file artifacts/emb_cnext_unet96_avg/train.npz \
    --valid-file artifacts/emb_cnext_unet96_avg/valid.npz \
    --test-file  artifacts/emb_cnext_unet96_avg/test.npz \
    --out-dir artifacts/lp_cnext_unet96_avg \
    --epochs 200 --patience 25

# 4. kNN sweep (~2 min, 72 trials).
python -m active_matter_ssl.eval_knn \
    --train-file artifacts/emb_cnext_unet96_avg/train.npz \
    --valid-file artifacts/emb_cnext_unet96_avg/valid.npz \
    --test-file  artifacts/emb_cnext_unet96_avg/test.npz \
    --out-dir artifacts/knn_cnext_unet96_avg --backend torch
```

## Final headline numbers (test split, z-scored MSE, lower is better)

| Method | mean MSE | alpha MSE | zeta MSE | Best config |
|---|---:|---:|---:|---|
| Linear probe | **0.1043** | 0.0371 | 0.1715 | feat_norm=zscore, lr=3e-3, wd=0, bs=4096 |
| kNN          | **0.0989** | 0.0121 | 0.1857 | k=3, weights=uniform, metric=euclidean, feat_norm=zscore |

`metrics.json` files under `artifacts/{lp,knn}_cnext_unet96_avg/` contain the
full sweep results plus the best trial; `train.out` contains the per-epoch
training history.

## What is committed vs. on disk only

Files committed to `main` for grading:
- `artifacts/cnext_unet96/encoder_best.pt`  (~71 MB, frozen encoder used by all eval steps)
- `artifacts/cnext_unet96/encoder_last.pt`  (final-epoch encoder)
- `artifacts/cnext_unet96/history.json`     (per-epoch metrics)
- `artifacts/cnext_unet96/train_config.json`(frozen training-time hyperparameters)
- `artifacts/emb_cnext_unet96_avg/{train,valid,test}.npz`  (frozen embeddings + labels)
- `artifacts/lp_cnext_unet96_avg/{metrics.json,best_linear_probe.pt,*_predictions.npz}`
- `artifacts/knn_cnext_unet96_avg/{metrics.json,best_test_predictions.npz}`
- `artifacts/emb_cnext_unet96_{max,avgmax}/` and matching LP/kNN metrics for the pooling ablation
- `cnext_logs/train.out`                    (full stdout/stderr of training)
- `cnext_logs/eval.out`                     (original avg+max eval stdout retained for audit; final avg-pool metrics are in artifacts)

Not committed (gitignored, exceed GitHub's 100 MB per-file limit):
- `artifacts/cnext_unet96/best.pt`, `last.pt`, `epoch_*.pt`  (full training-state
  snapshots, ~223 MB each; reconstructable from `encoder_best.pt` for inference)
- `artifacts/**/wandb/`                                       (W&B local run dirs)

A grader can fully re-run the eval pipeline starting from `encoder_best.pt`;
they only need the larger checkpoints if they want to *restart training* from a
mid-training state.
