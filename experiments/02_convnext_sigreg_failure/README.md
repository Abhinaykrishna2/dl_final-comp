# 02 ConvNeXt JEPA SIGReg Failure

This experiment tried to improve the JEPA embedding geometry with SIGReg. The goal was to get a distributionally healthier representation for kNN while still keeping the self-supervised context-to-target prediction task.

Observed behavior:

- `--target-stop-grad` fixed the most direct trivial-collapse path.
- The prediction loss became very small by epoch 2.
- `train_sigreg_loss` stayed near 25.7268.
- `valid_sigreg_loss` stayed near 9.625.
- The SIGReg values stayed effectively fixed from epoch 2 through epoch 25.

Interpretation:

The prediction task reached a local minimum quickly, and the SIGReg coefficient was too weak at `lejepa-lambda=0.05` to reshape the encoder. If SIGReg is applied only to the projector, it also does not guarantee that the pooled encoder features exported for downstream kNN become isotropic.

This experiment is retained as a negative result. It shows that the SIGReg configuration used here did not produce the intended embedding distribution, and it motivated the later V-JEPA/EMA masking ablation and the final CNext-U-Net forecasting objective.
