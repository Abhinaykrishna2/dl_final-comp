# 03 V-JEPA EMA Masking

This folder records the V-JEPA-style ablation implemented in `active_matter_ssl/train_vjepa.py`.

The motivation was to address the SIGReg failure with a more stable target branch:

- the online encoder predicts masked target features
- the target encoder is updated by EMA
- spatial masks keep the task nontrivial
- labels are not loaded during representation learning

This approach is useful for the final report as a collapse-mitigation attempt. It is not the selected final run in this repository because the CNext-U-Net forecaster produced the best downstream validation and test results under the project deadline.
