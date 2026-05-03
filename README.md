# Active Matter Self-Supervised Representation Learning

This repository contains the final project implementation for learning self-supervised video representations on the `active_matter` dataset from The Well. The learned encoder is frozen and evaluated on the physical parameters `alpha` and `zeta` using only the two downstream evaluators allowed by the project: a single linear layer and kNN regression.

The final selected model is a 96x96 CNext-U-Net future-frame forecaster. The repository also keeps the scored JEPA/VICReg baseline and the failed SigReg JEPA experiment because those runs document the representation-learning study, collapse diagnosis, and final model selection.

## Final Results

All numbers below are normalized MSE values on the held-out test split. Hyperparameters were selected using validation MSE.

| Encoder / pretraining objective | Downstream evaluator | Test mean MSE | Test alpha MSE | Test zeta MSE | Notes |
|---|---|---:|---:|---:|---|
| CNext-U-Net future-frame forecasting | kNN regression | 0.0989 | 0.0121 | 0.1857 | Best final result; avg pooling |
| CNext-U-Net future-frame forecasting | Linear probe | 0.1043 | 0.0371 | 0.1715 | Required frozen linear evaluation; avg pooling |
| JEPA + VICReg baseline | Linear probe | 0.2485 | 0.1183 | 0.3786 | Original 224x224 baseline |
| JEPA + VICReg baseline | kNN regression | 0.2709 | 0.0472 | 0.4946 | Original 224x224 baseline |
| ConvNeXt JEPA + SigReg | Linear probe | 0.5829 | 0.1872 | 0.9787 | SigReg term frozen all 25 epochs; export bug (175 vs 11,550 train samples) |
| ConvNeXt JEPA + SigReg | kNN regression | 0.5924 | 0.2058 | 0.9790 | SigReg term frozen all 25 epochs; export bug (175 vs 11,550 train samples) |

The best submitted result is kNN regression on average-pooled frozen CNext-U-Net embeddings, with normalized test mean MSE `0.0989`. The same frozen encoder also satisfies the required linear-probe evaluation, with normalized test mean MSE `0.1043`.

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

- `active_matter_ssl/`: dataset, models, pretraining scripts, embedding export, and downstream evaluation for the final CNext and shared evaluators.
- `baseline_jepa/`: isolated JEPA/VICReg baseline package defaulted to 96x96 for a controlled rerun against the final CNext model.
- `baseline_jepa/train_jepa.py`: 96x96 JEPA/VICReg baseline rerun entry point.
- `active_matter_ssl/train_jepa.py`: historical JEPA training with VICReg or SigReg losses.
- `active_matter_ssl/train_cnext_forecaster.py`: final CNext-U-Net future-frame forecaster.
- `active_matter_ssl/export_embeddings.py`: frozen embedding export for JEPA-style encoders.
- `active_matter_ssl/export_cnext_embeddings.py`: frozen embedding export for the CNext-U-Net encoder.
- `active_matter_ssl/sweep_linear_probe.py`: single-linear-layer downstream regression sweep.
- `active_matter_ssl/eval_knn.py`: kNN downstream regression sweep.
- `experiments/`: command wrappers and short notes for the scored approaches.
- `artifacts/`: committed final CNext checkpoints, embeddings, and downstream metrics.
- `cnext_logs/`: final-run reproduction recipe and stdout logs.
- `past_analysis/`: detailed run analysis, collapse diagnosis, and historical score reports.

## Approach 1: JEPA + VICReg Baseline

The first completed baseline used `active_matter_ssl/train_jepa.py`, with a ConvNeXt-style JEPA encoder and a VICReg objective. For a controlled rerun at the same frame size as the final model, the baseline code is also available as the isolated `baseline_jepa/` package; its defaults and wrapper now use 96x96 frames. The model predicts target feature maps from context feature maps while VICReg keeps the representation non-collapsed through explicit similarity, variance, and covariance terms.

Architecture and training summary:

- Model: `JepaModel` with `ConvEncoder` and `ConvPredictor`.
- Parameters: 3.27M.
- Historical recorded input: 16 frames, 11 channels, 224x224 resolution.
- Comparable rerun input: 16 frames, 11 channels, 96x96 resolution via `baseline_jepa/`.
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

To rerun this baseline at 96x96 for a resolution-matched comparison:

```bash
DATA_ROOT=/path/to/active_matter bash experiments/01_ejepa_vicreg_baseline/run_full_pipeline.sh
```

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

The training finished but the SigReg regularization never worked. `train_sigreg_loss` is frozen at 25.7268 and `valid_sigreg_loss` at 9.625 for all 25 epochs, unchanged from epoch 2 to epoch 25. `--target-stop-grad` fixed the trivial collapse but the distribution term never got traction.

**What happened:** `pred_loss` hit near-zero by epoch 2 (~9.7e-4) and the encoder locked into a local minimum. The SigReg gradient at `lejepa-lambda=0.05` (only 5% weight) was too weak to pull it out. The distribution regularization provided effectively zero gradient signal the entire run.

**What the encoder actually learned:** It is a temporal prediction model — it learned to map context → target representations, but the embedding distribution is not isotropic. The kNN benefit expected from SigReg did not materialize.

The model is still worth examining: it is 3× larger than the JEPA baseline (9.83M vs 3.27M params) and `pred_loss` converged very well (~7.5e-7). However, the downstream scores below cannot be treated as a fair comparison because the embedding export also used `single_clip` mode, yielding only 175 training embeddings instead of the 11,550 sliding-window embeddings used by the JEPA baseline. Both failures — the frozen regularizer and the export mismatch — need to be corrected before a valid comparison is possible.

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

These scores are retained for completeness but are not a fair comparison to the baseline for the two reasons documented above. This result motivated switching away from a weak global distribution penalty and toward a direct future-frame forecasting objective.

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
- Bottleneck maps pooled with average pooling after validation-based pooling selection.
- Embeddings saved under `artifacts/emb_cnext_unet96_avg/`.

Downstream results:

| Evaluator | Valid mean MSE | Test mean MSE | Test alpha MSE | Test zeta MSE | Best configuration |
|---|---:|---:|---:|---:|---|
| Linear probe | 0.0593 | 0.1043 | 0.0371 | 0.1715 | zscore features, lr 3e-3, wd 0, effective full batch |
| kNN regression | 0.0240 | 0.0989 | 0.0121 | 0.1857 | zscore features, k=3, euclidean, uniform weights |

This is the final selected encoder and pooling export. It improves over the JEPA/VICReg baseline in both downstream evaluators and gives the strongest overall score with kNN regression.

Pooling ablation:

| Pooling export | Linear test mean MSE | kNN test mean MSE |
|---|---:|---:|
| Average pooling | 0.1043 | 0.0989 |
| Max pooling | 0.0959 | 0.1878 |
| Average + max pooling | 0.1992 | 0.1285 |

Average pooling had the best validation MSE for both downstream evaluators and is therefore the selected final export. The max-pooling linear test value is lower post hoc, but selecting it based on test performance would violate the validation-only selection protocol.

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
    --out-dir artifacts/emb_cnext_unet96_avg \
    --batch-size 64 \
    --pool avg \
    --clip-frames 16 \
    --window-stride 1 \
    --amp
```

Run downstream evaluation:

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

The original JEPA/VICReg baseline used 224x224 frames. The final CNext-U-Net model uses 96x96 frames after TA clarification that this is allowed. The repository now includes a `baseline_jepa` 96x96 rerun wrapper so the baseline can be recomputed under the same frame size; until those new metrics are available, the final table should be read as the project trajectory and final model selection rather than a controlled resolution-matched ablation.

The SigReg downstream scores are included for completeness, but the run has two independent failures: (1) `train_sigreg_loss` was frozen at 25.7268 from epoch 2 to epoch 25 — the distribution regularization provided effectively zero gradient signal because `pred_loss` hit near-zero first and the 5% SigReg weight was too weak to move the encoder out of that local minimum; (2) the downstream export used `single_clip` mode, giving only 175 training embeddings against the 11,550 used by the JEPA baseline. The reported scores reflect both failures simultaneously and cannot be used as a fair comparison.
