# Active Matter Self-Supervised Representation Learning

This repository contains the final project implementation for learning self-supervised video representations on the `active_matter` dataset from The Well. The learned encoder is frozen and evaluated on the physical parameters `alpha` and `zeta` using only the two downstream evaluators allowed by the project: a single linear layer and kNN regression.

The final selected model is a 96x96 CNext-U-Net future-frame forecaster. The repository also keeps the scored JEPA/VICReg baseline and the failed SigReg JEPA experiment because those runs document the representation-learning study, collapse diagnosis, and final model selection.

## Final Results

All numbers below are normalized MSE values on the held-out test split. Hyperparameters were selected using validation MSE.

| Encoder / pretraining objective | Downstream evaluator | Test mean MSE | Test alpha MSE | Test zeta MSE | Notes |
|---|---|---:|---:|---:|---|
| CNext-U-Net future-frame forecasting | kNN regression | 0.1285 | 0.0121 | 0.2449 | Best final result |
| CNext-U-Net future-frame forecasting | Linear probe | 0.1992 | 0.1157 | 0.2827 | Required frozen linear evaluation |
| JEPA + VICReg baseline | Linear probe | 0.2485 | 0.1183 | 0.3786 | Original 224x224 baseline |
| JEPA + VICReg baseline | kNN regression | 0.2709 | 0.0472 | 0.4946 | Original 224x224 baseline |
| ConvNeXt JEPA + SigReg | Linear probe | 0.5829 | 0.1872 | 0.9787 | Failed run; export used only 175 train samples |
| ConvNeXt JEPA + SigReg | kNN regression | 0.5924 | 0.2058 | 0.9790 | Failed run; export used only 175 train samples |

The best submitted result is kNN regression on frozen CNext-U-Net embeddings, with normalized test mean MSE `0.1285`. The same frozen encoder also satisfies the required linear-probe evaluation, with normalized test mean MSE `0.1992`.

## Compliance With Project Constraints

- Representation learning is self-supervised and trained from scratch.
- No pretrained weights or external datasets are used.
- The final CNext-U-Net has 18.59M total parameters and 7.34M encoder parameters, below the 100M parameter limit.
- Physical labels are not loaded during representation learning. The pretraining scripts construct train/valid datasets with `include_labels=False`.
- The encoder is frozen before downstream evaluation.
- Downstream evaluation uses only a single linear layer and kNN regression.
- Validation split metrics are used for model and hyperparameter selection.
- The test split is used only for final reporting.
- The final model uses 96x96 frames after TA clarification that the patch/frame size may be changed.

## Repository Layout

- `active_matter_ssl/`: dataset, models, pretraining scripts, embedding export, and downstream evaluation.
- `active_matter_ssl/train_jepa.py`: JEPA training with VICReg or SigReg losses.
- `active_matter_ssl/train_cnext_forecaster.py`: final CNext-U-Net future-frame forecaster.
- `active_matter_ssl/export_embeddings.py`: frozen embedding export for JEPA-style encoders.
- `active_matter_ssl/export_cnext_embeddings.py`: frozen embedding export for the CNext-U-Net encoder.
- `active_matter_ssl/sweep_linear_probe.py`: single-linear-layer downstream regression sweep.
- `active_matter_ssl/eval_knn.py`: kNN downstream regression sweep.
- `experiments/`: command wrappers and short notes for the scored approaches.
- `past_analysis/`: detailed run analysis, collapse diagnosis, and historical score reports.
- `artifacts/`: committed final CNext checkpoints, embeddings, and downstream metrics.
- `cnext_logs/`: final-run reproduction recipe and stdout logs.
- `baseline.txt`: compact record of the first JEPA baseline metrics.

## Approach 1: JEPA + VICReg Baseline

The first completed baseline used `active_matter_ssl/train_jepa.py`, with a ConvNeXt-style JEPA encoder and a VICReg objective. The model predicts target feature maps from context feature maps while VICReg keeps the representation non-collapsed through explicit similarity, variance, and covariance terms.

Architecture and training summary:

- Model: `JepaModel` with `ConvEncoder` and `ConvPredictor`.
- Parameters: 3.27M.
- Input: 16 frames, 11 channels, 224x224 resolution.
- Optimizer: AdamW, learning rate `5e-4`, weight decay `5e-4`.
- Best checkpoint: epoch 5 of 12.
- Training behavior: validation loss improved through epoch 5 and then diverged after epoch 6.

Embedding-space diagnostics from the best checkpoint:

- Train embedding shape: 11,550 x 128 using sliding-window export.
- Global embedding std: 0.2237.
- Mean per-dimension std: 0.1007.
- Dead dimensions: 0 / 128.
- Dimensions needed for 95% variance: 43 / 128.

This run produced a healthy, non-collapsed embedding space. The downstream results were:

| Evaluator | Valid mean MSE | Test mean MSE | Test alpha MSE | Test zeta MSE |
|---|---:|---:|---:|---:|
| Linear probe | 0.2158 | 0.2485 | 0.1183 | 0.3786 |
| kNN regression | 0.1960 | 0.2709 | 0.0472 | 0.4946 |

The baseline shows that `alpha` is easier to recover than `zeta`. kNN predicts `alpha` particularly well, while linear probing is stronger on `zeta`, suggesting that `alpha` is more locally clustered in the representation space and `zeta` is more globally linear.

## Approach 2: ConvNeXt JEPA + SigReg

The second completed run scaled the JEPA encoder and replaced VICReg with a SigReg-style distribution regularizer. The intent was to improve kNN behavior by encouraging the projected embedding distribution to become isotropic.

Architecture and training summary:

- Model: `SigRegJepaModel` with ConvEncoder, MLP projector, and MLP predictor.
- Parameters: 9.83M.
- Embedding dimension: 256.
- Objective: `0.95 * pred_loss + 0.05 * sigreg_loss`.
- Training length: 25 epochs.
- `--target-stop-grad` was used in the corrected run to break the symmetric collapse path.

Observed training failure:

- `pred_loss` reached near zero by epoch 2.
- `train_sigreg_loss` stayed near 25.7268 from epoch 2 through epoch 25.
- `valid_sigreg_loss` stayed near 9.625 for the full run.
- The encoder optimized the temporal prediction objective but did not learn the intended isotropic embedding distribution.

Embedding diagnostics showed collapse:

- Exported train embedding shape: 175 x 256 in the failed evaluation export.
- Mean per-dimension std: 0.0176.
- Dead dimensions: 62 / 256.
- Dimensions needed for 95% variance: 29 / 256.

The recorded downstream scores were poor:

| Evaluator | Test mean MSE | Test alpha MSE | Test zeta MSE |
|---|---:|---:|---:|
| Linear probe | 0.5829 | 0.1872 | 0.9787 |
| kNN regression | 0.5924 | 0.2058 | 0.9790 |

These numbers are retained as a failed experiment, not as a fair comparison to the baseline. The SigReg embeddings were exported with `single_clip` mode, giving only 175 training embeddings instead of the 11,550 sliding-window embeddings used by the JEPA baseline. This export mismatch severely weakened both the linear probe and kNN evaluation. Independently, the frozen SigReg loss and dead-dimension statistics show that the intended distribution regularization did not take effect.

This result motivated switching away from a weak global distribution penalty and toward a direct future-frame forecasting objective.

## Final Selected Model: CNext-U-Net Future-Frame Forecaster

The final selected approach uses `active_matter_ssl/train_cnext_forecaster.py`, a scratch CNext-U-Net future-frame forecaster inspired by The Well `UNetConvNext` benchmark architecture. The decoder is used only during self-supervised pretraining; downstream evaluation uses frozen encoder embeddings.

Training setup:

- Input: 4 context frames.
- Target: next 1 frame.
- Resolution: 96x96.
- Objective: future-frame MSE over active-matter physical fields.
- Optimizer: AdamW, learning rate `3e-4`, weight decay `5e-4`.
- Epochs: 50.
- Best checkpoint: `artifacts/cnext_unet96/encoder_best.pt`.
- Parameters: 18.59M total, 7.34M encoder.

The forecasting training curve showed stable learning: validation relative MSE dropped from about `0.1153` at epoch 1 to roughly `0.0025` by the end of training.

Embedding export:

- Encoder frozen before export.
- One deterministic 16-frame clip per simulation.
- Sliding 4-frame windows inside each clip.
- Bottleneck maps pooled with avg+max pooling.
- Embeddings saved under `artifacts/emb_cnext_unet96_avgmax/`.

Downstream results:

| Evaluator | Valid mean MSE | Test mean MSE | Test alpha MSE | Test zeta MSE | Best configuration |
|---|---:|---:|---:|---:|---|
| Linear probe | 0.0886 | 0.1992 | 0.1157 | 0.2827 | zscore features, lr 3e-4, wd 1e-4, full batch |
| kNN regression | 0.0612 | 0.1285 | 0.0121 | 0.2449 | zscore features, k=5, euclidean, distance weights |

This is the final selected encoder. It improves over the JEPA/VICReg baseline in both downstream evaluators and gives the strongest overall score with kNN regression.

## Reproducing The Final Run

Set the dataset path:

```bash
export DATA_ROOT=/path/to/active_matter
```

Train the final encoder:

```bash
python -m active_matter_ssl.train_cnext_forecaster \
    --data-root "$DATA_ROOT" \
    --out-dir artifacts/cnext_unet96 \
    --epochs 50 \
    --batch-size 16 \
    --num-workers 4 \
    --prefetch-factor 4 \
    --resolution 96 \
    --context-frames 4 \
    --target-frames 1 \
    --lr 3e-4 \
    --weight-decay 5e-4 \
    --amp \
    --amp-dtype bfloat16 \
    --save-every 2 \
    --seed 42 \
    --wandb-mode disabled
```

Export frozen embeddings:

```bash
python -m active_matter_ssl.export_cnext_embeddings \
    --data-root "$DATA_ROOT" \
    --checkpoint artifacts/cnext_unet96/encoder_best.pt \
    --out-dir artifacts/emb_cnext_unet96_avgmax \
    --batch-size 64 \
    --pool avgmax \
    --clip-frames 16 \
    --window-stride 1 \
    --amp
```

Run downstream evaluation:

```bash
python -m active_matter_ssl.sweep_linear_probe \
    --train-file artifacts/emb_cnext_unet96_avgmax/train.npz \
    --valid-file artifacts/emb_cnext_unet96_avgmax/valid.npz \
    --test-file artifacts/emb_cnext_unet96_avgmax/test.npz \
    --out-dir artifacts/lp_cnext_unet96_avgmax \
    --epochs 200 \
    --patience 25

python -m active_matter_ssl.eval_knn \
    --train-file artifacts/emb_cnext_unet96_avgmax/train.npz \
    --valid-file artifacts/emb_cnext_unet96_avgmax/valid.npz \
    --test-file artifacts/emb_cnext_unet96_avgmax/test.npz \
    --out-dir artifacts/knn_cnext_unet96_avgmax \
    --backend torch
```

The exact server-side reproduction recipe and committed artifact list are in `cnext_logs/REPRODUCE.md`.

## Leakage Prevention

The training and evaluation code is structured to prevent label leakage:

- `ActiveMatterWindowDataset` supports `include_labels=False`.
- `train_jepa.py` and `train_cnext_forecaster.py` pass `include_labels=False` for representation learning.
- Labels are attached only when exporting frozen embeddings for downstream evaluation.
- Downstream label normalization is fit from train labels.
- Linear and kNN hyperparameters are selected using validation normalized MSE.
- Test metrics are reported only after validation-based selection.
- The committed final checkpoints are trained from scratch and do not load pretrained weights.

## Notes On Comparability

The original JEPA/VICReg baseline used 224x224 frames. The final CNext-U-Net model uses 96x96 frames after TA clarification that this is allowed. Therefore, the final table reports the project trajectory and final model selection, not a controlled resolution-matched ablation.

The SigReg downstream scores are included for completeness, but the run has two known issues: the regularization term did not affect the encoder distribution, and the first downstream export used `single_clip` mode with only 175 training embeddings. This is why it is treated as a failed experiment rather than a competitive final model.
