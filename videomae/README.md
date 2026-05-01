# `videomae/` -- Person B's contribution

This package is Person B's half of the final project. It implements a
**VideoMAE / SimMIM hybrid masked-autoencoder SSL track** on the
**byte-identical 3D ConvNeXt encoder** that Person A's `physrl/` package uses
for SIGReg-JEPA, plus an end-to-end **supervised baseline** and a **collapse /
isotropy analysis suite**. The two streams meet only in the joint report.

## Directory contents

| Path | Owner | Origin |
|------|-------|--------|
| `data.py`, `utils.py`, `losses.py`, `eval_knn.py`, `sweep_linear_probe.py`, `train_linear_probe.py`, `export_embeddings.py` | derivative | Verbatim copies of the matching files in `physrl/` (with the import paths rewritten to `videomae.*`). Used unmodified for evaluation. |
| `models.py` | mixed | Encoder block / `ConvEncoder` / `JepaModel` / `SigRegJepaModel` are byte-identical copies of `physrl/models.py`. **`ConvNeXtDecoder` and `VideoMAEModel` are new** (Person B). |
| `train_videomae.py` | new | The main SSL trainer for this stream. |
| `train_supervised.py` | new | End-to-end supervised baseline (upper-bound anchor for the report). |
| `analyze_representations.py` | new | Collapse / isotropy / rank / Epps-Pulley diagnostics. |
| `plot_from_json.py` | new | Regenerates every report figure from JSON logs. |
| `scripts/` | new | Detached launchers + one-shot reproduction script. |
| `artifacts/` | new | Per-run checkpoints, embeddings, eval results, figures (gitignored except for small JSON metadata). |
| `ENV.md`, `requirements-lock.txt` | new | Exact environment spec for reproducibility. |

## Method summary (one paragraph)

We train a **VideoMAE / SimMIM hybrid** masked autoencoder on the active matter
dataset using a 3D ConvNeXt encoder (~7.7 M params) that is byte-identical to
the one Person A uses for SIGReg-JEPA. Each input clip of shape `(11, 16, 224,
224)` is partitioned into 49 spatiotemporal tubes of shape `(16, 32, 32)` -- one
per output token of the encoder, following the patch-size-equals-encoder-stride
recipe of SimMIM (Xie et al. 2022). With a default mask ratio of 0.60, ~30
random tubes are replaced with a learnable per-channel mask token (the SimMIM
adaptation needed because depthwise-3D ConvNeXts cannot use VideoMAE's
asymmetric encoder-decoder). The encoder produces a `(256, 7, 7)` latent which a
lightweight 5-stage transposed-conv decoder (~0.35 M params) maps back to the
input shape. Training optimizes the VideoMAE per-tube z-scored MSE loss on
**masked positions only** (SimMIM Tab. 4: prediction-only beats full
reconstruction). After SSL pretraining, the encoder is frozen and we evaluate
representations with a single linear probe and kNN regression on the (alpha,
zeta) physical parameters, exactly as the assignment requires.

## Running it

Set up the env per `videomae/ENV.md`, then either:

```bash
# Run the entire pipeline (training + eval + figures) end-to-end:
bash videomae/scripts/reproduce_main_result.sh
```

or launch the long-running training jobs detached so they survive SSH drops:

```bash
bash videomae/scripts/launch_videomae.sh    # GPU 0, ~3 h
bash videomae/scripts/launch_supervised.sh  # GPU 1, ~1 h
```

Then once they complete, run eval + figures:

```bash
bash videomae/scripts/run_eval.sh            # exports + probes + kNN + analysis
python -m videomae.plot_from_json \
    --runs videomae/artifacts/videomae_main:VideoMAE-mask60 \
           videomae/artifacts/videomae_mask75:VideoMAE-mask75 \
           videomae/artifacts/supervised:Supervised \
    --out-dir videomae/artifacts/figures
```

## Hyperparameter choices and citations

| Choice | Value | Citation |
|--------|-------|----------|
| Encoder | 3D ConvNeXt, dims=(32,64,128,256,256), blocks=(2,2,4,8,2), stem=2 | matches Person A's `SigRegJepaModel` exactly |
| Mask ratio (main) | 0.60 | SimMIM Tab. 1: optimal at patch-size = encoder downsampling |
| Mask ratio (ablation) | 0.75 | VideoMAE Sec 4.1.2: optimal for video data |
| Tube size | (16, 32, 32) | SimMIM Sec 4.1.2: align with encoder stride |
| Mask token | learnable per-channel | SimMIM Sec 3.2 (required for non-ViT encoders) |
| Decoder | 5-stage transposed conv, 350k params | SimMIM Tab. 2: lightweight head transfers better |
| Loss target | per-tube z-scored pixels | VideoMAE Sec 3.3 / Tab. 1c |
| Loss scope | masked positions only | SimMIM Tab. 4: prediction beats reconstruction |
| Loss type | MSE | VideoMAE Tab. 1f |
| Optimizer | AdamW, lr 1.5e-4, wd 0.05 | VideoMAE Sec 4.1 / SimMIM Sec 4.1 |
| Schedule | cosine, 2-epoch warmup | both papers |
| Precision | BF16 autocast | B200 native |

## Constraint check (matches the assignment's Section 8 hard rules)

- Train from scratch, no pretrained weights: yes (no checkpoint loading from external sources).
- Total params < 100 M: 8.07 M (encoder + decoder + mask token).
- Active matter dataset only: yes.
- No training on val/test: `LabelNormalizer.fit` runs on train labels only; test split is
  touched only by `eval_knn.py` and the final-eval block of `sweep_linear_probe.py`.
- Linear probe + kNN only for evaluation: yes; the supervised baseline is a separate
  comparison artifact, not the main eval.
