# 01 E-JEPA VICReg Baseline

This is the first frozen-encoder baseline. It used the ConvNeXt-style JEPA encoder from `active_matter_ssl/train_jepa.py`, then exported frozen embeddings and evaluated them with a single linear layer and kNN regression.

Recorded results from `baseline.txt` and `past_analysis/report_newJEPA.txt`:

| Method | Valid normalized mean MSE | Test normalized mean MSE | Test alpha MSE | Test zeta MSE | Test raw mean MSE |
|---|---:|---:|---:|---:|---:|
| Linear probe | 0.2158 | 0.2485 | 0.1183 | 0.3786 | 5.27 |
| kNN | 0.1960 | 0.2709 | 0.0472 | 0.4946 | 6.77 |

Embedding diagnostics from the best checkpoint:

- Train embeddings: 11,550 x 128 using sliding-window export.
- Dead dimensions: 0 / 128.
- Dimensions needed for 95% variance: 43 / 128.
- Best checkpoint: epoch 5 of 12; validation loss rose after epoch 6.

Important reporting note: this baseline used the earlier 224x224 preprocessing. The final CNext run uses 96x96 after TA clarification. Keep this as a historical baseline, not a controlled resolution ablation.
