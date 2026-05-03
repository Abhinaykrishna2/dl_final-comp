# 02 ConvNeXt JEPA SIGReg Failure

This experiment tried to improve the JEPA embedding geometry with SIGReg. The goal was to get a distributionally healthier representation for kNN while still keeping the self-supervised context-to-target prediction task.

Observed behavior:

- `--target-stop-grad` fixed the most direct trivial-collapse path.
- The prediction loss became very small by epoch 2.
- `train_sigreg_loss` stayed near 25.7268.
- `valid_sigreg_loss` stayed near 9.625.
- The SIGReg values stayed effectively fixed from epoch 2 through epoch 25.
- Exported train embeddings used only 175 single-clip samples, while the baseline used 11,550 sliding-window samples.

Recorded failed-run downstream scores:

| Method | Test normalized mean MSE | Test alpha MSE | Test zeta MSE |
|---|---:|---:|---:|
| Linear probe | 0.5829 | 0.1872 | 0.9787 |
| kNN | 0.5924 | 0.2058 | 0.9790 |

Embedding diagnostics:

- Train embeddings: 175 x 256 in the failed export.
- Dead dimensions: 62 / 256.
- Dimensions needed for 95% variance: 29 / 256.

Interpretation:

The prediction task reached a local minimum quickly, and the SIGReg coefficient was too weak at `lejepa-lambda=0.05` to reshape the encoder. If SIGReg is applied only to the projector, it also does not guarantee that the pooled encoder features exported for downstream kNN become isotropic. The downstream scores are retained as a failed-run record, but they are not a fair apples-to-apples comparison because of the single-clip export mismatch.

This experiment is retained as a negative result. It shows that the SIGReg configuration used here did not produce the intended embedding distribution, and it motivated switching to the final CNext-U-Net forecasting objective.
