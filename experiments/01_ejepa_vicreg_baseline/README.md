# 01 E-JEPA VICReg Baseline

This is the first frozen-encoder baseline. The historical scored run used the ConvNeXt-style JEPA encoder from `active_matter_ssl/train_jepa.py`, then exported frozen embeddings and evaluated them with a single linear layer and kNN regression.

The wrapper in this folder now uses the renamed `baseline_jepa/` package and trains at 96x96, so it can be rerun as a resolution-matched baseline against the final CNext-U-Net model. New rerun artifacts will be written to `artifacts/baseline_jepa96`, `artifacts/emb_baseline_jepa96_avg`, `artifacts/lp_baseline_jepa96`, and `artifacts/knn_baseline_jepa96`.

Historical recorded results from `past_analysis/report_newJEPA.txt`:

| Method | Valid normalized mean MSE | Test normalized mean MSE | Test alpha MSE | Test zeta MSE | Test raw mean MSE |
|---|---:|---:|---:|---:|---:|
| Linear probe | 0.2158 | 0.2485 | 0.1183 | 0.3786 | 5.27 |
| kNN | 0.1960 | 0.2709 | 0.0472 | 0.4946 | 6.77 |

Embedding diagnostics from the best checkpoint:

- Train embeddings: 11,550 x 128 using sliding-window export.
- Dead dimensions: 0 / 128.
- Dimensions needed for 95% variance: 43 / 128.
- Best checkpoint: epoch 5 of 12; validation loss rose after epoch 6.

Important reporting note: the table above used the earlier 224x224 preprocessing. The current script uses 96x96 for a controlled rerun, but its metrics should replace the historical table only after the rerun finishes and downstream evaluation is complete.
