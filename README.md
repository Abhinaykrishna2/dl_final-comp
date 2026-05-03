# Active Matter Self-Supervised Representation Learning

This repository contains the final project code for learning frozen video representations on the `active_matter` dataset from The Well, then evaluating those representations with only a linear probe and kNN regression on the physical parameters `alpha` and `zeta`.

The current best submission path is the 96x96 CNext-U-Net forecaster. The older JEPA, SIGReg, and V-JEPA experiments are kept as part of the study because the project is evaluated on the representation-learning process and collapse analysis, not just the final downstream number.

## Project Rules

The implementation is organized around the constraints in `Final Project.docx`:

- Representation learning is self-supervised and trained from scratch.
- No pretrained weights or external datasets are used.
- The encoder has fewer than 100M parameters. The final CNext-U-Net run has 18.59M total parameters and 7.34M encoder parameters.
- The representation-learning scripts construct train/valid pretraining datasets with `include_labels=False`, so `alpha` and `zeta` are not loaded during encoder training.
- The encoder is frozen before downstream evaluation.
- Downstream evaluation uses only a single linear layer and kNN regression.
- Hyperparameters are selected with the validation split. The test split is used only for final reporting.
- Frames are resized to 96x96 for the final CNext run. This differs from the original 224x224 default, but the TA clarified that using 96x96 is allowed.

## Repository Layout

- `active_matter_ssl/`: reusable dataset, models, training, embedding export, and downstream evaluation code.
- `active_matter_ssl/train_jepa.py`: ConvNeXt-style JEPA training with VICReg or SIGReg losses.
- `active_matter_ssl/train_vjepa.py`: V-JEPA-style masked prediction with an EMA target encoder.
- `active_matter_ssl/train_cnext_forecaster.py`: final CNext-U-Net future-frame forecaster.
- `active_matter_ssl/export_embeddings.py`: frozen embedding export for JEPA/V-JEPA-style encoders.
- `active_matter_ssl/export_cnext_embeddings.py`: frozen embedding export for the CNext-U-Net encoder.
- `active_matter_ssl/sweep_linear_probe.py`: single-linear-layer downstream regression sweep.
- `active_matter_ssl/eval_knn.py`: kNN downstream regression sweep.
- `experiments/`: organized command wrappers and notes for each major approach.
- `artifacts/`: committed final CNext checkpoints, embeddings, and downstream metrics.
- `cnext_logs/`: reproduction commands and stdout logs for the final run.
- `baseline.txt`: recorded first JEPA baseline metrics.

## Approaches

### 1. E-JEPA / VICReg Baseline

The first baseline used the ConvNeXt-style JEPA encoder in `active_matter_ssl/train_jepa.py`. It used 16 context frames and 16 target frames, with a VICReg-style representation objective to avoid complete collapse.

Recorded baseline results from `baseline.txt`:

| Method | Valid normalized mean MSE | Test normalized mean MSE | Test raw mean MSE |
|---|---:|---:|---:|
| Linear probe | 0.2158 | 0.2485 | 5.27 |
| kNN | 0.1960 | 0.2709 | 6.77 |

This baseline was trained with the earlier 224x224 preprocessing. It is still useful as the first milestone, but it should be disclosed as a different-resolution baseline. A strict architecture-only comparison would require retraining it at 96x96.

### 2. ConvNeXt JEPA + SIGReg Attempt

The SIGReg experiment tried to use a JEPA prediction loss plus a distribution regularizer:

- prediction term: context representation predicts the target representation
- SIGReg term: encourages the representation distribution to match a non-collapsed target distribution
- `--target-stop-grad`: added to avoid the most trivial collapse mode

Observed failure mode:

- `train_sigreg_loss` stayed frozen at about 25.7268.
- `valid_sigreg_loss` stayed frozen at about 9.625.
- These values did not meaningfully change from epoch 2 through epoch 25.
- `pred_loss` fell near zero by epoch 2, around `9.7e-4`, and later converged much lower.
- The encoder found a local minimum where prediction was solved, but the exported embedding distribution was not shaped by SIGReg.

The main interpretation is that the SIGReg gradient was too weak at `lejepa-lambda=0.05`, especially once the prediction branch had already become easy. When SIGReg was applied to the projection head, the regularization also did not directly guarantee that the pooled encoder features used for downstream kNN were isotropic.

This run is useful in the final report as a collapse-mitigation study: `--target-stop-grad` fixed the direct target-branch collapse, but it did not make the distribution regularizer strong enough to affect the final frozen embedding space.

### 3. V-JEPA / EMA Masking Attempt

The V-JEPA-style path in `active_matter_ssl/train_vjepa.py` was added as a more conservative collapse-mitigation experiment. It uses spatial masking and an EMA target encoder so the online encoder predicts target features without directly chasing a rapidly moving target branch.

This is a good ablation to discuss because it addresses a concrete problem from the SIGReg run: the prediction task should remain nontrivial longer, and the EMA target should reduce degenerate shortcuts. It is not the final selected result in this repository because the CNext-U-Net forecaster produced stronger downstream numbers under the time limit.

### 4. Final CNext-U-Net Forecaster

The final approach uses `active_matter_ssl/train_cnext_forecaster.py`, a scratch CNext-U-Net future-frame forecaster inspired by The Well `UNetConvNext` benchmark architecture. It is not pretrained. The decoder is used only during self-supervised pretraining.

Training setup:

- input: 4 context frames
- target: next 1 frame
- resolution: 96x96
- objective: future-frame MSE over physical fields
- split used for weight updates: train only
- validation: selected the best forecasting checkpoint
- frozen representation: bottleneck encoder features from `encoder_best.pt`
- downstream embedding export: one deterministic 16-frame clip per simulation, sliding 4-frame windows internally, avg+max pooling

Final CNext downstream results:

| Method | Valid normalized mean MSE | Test normalized mean MSE | Test alpha MSE | Test zeta MSE | Best config |
|---|---:|---:|---:|---:|---|
| Linear probe | 0.0886 | 0.1992 | 0.1157 | 0.2827 | feature zscore, lr 3e-4, wd 1e-4, full batch |
| kNN | 0.0612 | 0.1285 | 0.0121 | 0.2449 | k=5, distance weights, euclidean, feature zscore |

The final kNN result is the strongest recorded number in this repository. The linear probe also improves over the original baseline, but the larger gain is in kNN, which suggests that the CNext future-prediction objective produced a more useful neighborhood geometry than the earlier JEPA baseline.

## Reproducing The Final Run

Set the dataset path first:

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

The exact server-side reproduction recipe and committed artifact list are also in `cnext_logs/REPRODUCE.md`.

## What To Report

Use the CNext-U-Net kNN result as the best final metric:

- test normalized mean MSE: `0.1285`
- test normalized alpha MSE: `0.0121`
- test normalized zeta MSE: `0.2449`

Use the CNext-U-Net linear probe as the required linear downstream result:

- test normalized mean MSE: `0.1992`
- test normalized alpha MSE: `0.1157`
- test normalized zeta MSE: `0.2827`

In the writeup, present the 224x224 E-JEPA/VICReg result as the original baseline and clearly state that the final model uses 96x96 after TA clarification. The fairest wording is that the final CNext model improves the end-to-end downstream result, while the baseline comparison is not a pure controlled resolution ablation.

## Leakage Checks

The code path is designed to avoid label leakage:

- `ActiveMatterWindowDataset` has an `include_labels` flag.
- `train_jepa.py`, `train_vjepa.py`, and `train_cnext_forecaster.py` pass `include_labels=False` for representation learning.
- Labels are exported only after the encoder is frozen.
- The linear probe and kNN scripts fit normalizers and models from train embeddings, select hyperparameters by validation MSE, and report the selected model on test.
- The committed final checkpoints are trained from scratch and do not load pretrained weights.

The final report should avoid claiming that the SIGReg run succeeded. It should frame SIGReg as a negative result that motivated the move to a direct future-frame forecasting objective.
