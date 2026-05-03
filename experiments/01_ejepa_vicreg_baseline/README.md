# 01 E-JEPA VICReg Baseline

This is the first frozen-encoder baseline. It used the ConvNeXt-style JEPA encoder from `active_matter_ssl/train_jepa.py`, then exported frozen embeddings and evaluated them with a single linear layer and kNN regression.

Recorded result from `baseline.txt`:

| Method | Valid normalized mean MSE | Test normalized mean MSE | Test raw mean MSE |
|---|---:|---:|---:|
| Linear probe | 0.2158 | 0.2485 | 5.27 |
| kNN | 0.1960 | 0.2709 | 6.77 |

Important reporting note: this baseline used the earlier 224x224 preprocessing. The final CNext run uses 96x96 after TA clarification. Keep this as a historical baseline, not a controlled resolution ablation.
