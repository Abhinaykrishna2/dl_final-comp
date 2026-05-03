# 04 CNext-U-Net Forecasting Final

This is the final selected approach.

The model is a scratch CNext-U-Net forecaster in `active_matter_ssl/train_cnext_forecaster.py`, with the encoder implemented by `CNextUNetForecaster` in `active_matter_ssl/models.py`. It is inspired by The Well `UNetConvNext` benchmark architecture and trained from scratch on active matter trajectories.

Training task:

- resize frames to 96x96
- use 4 context frames
- predict the next 1 frame
- train with forecasting MSE
- save `encoder_best.pt` from the best validation forecasting checkpoint

Embedding export:

- freeze the encoder
- use one deterministic 16-frame clip per simulation
- run sliding 4-frame windows inside that clip
- pool bottleneck maps with avg+max pooling

Final downstream result:

| Method | Valid normalized mean MSE | Test normalized mean MSE | Test alpha MSE | Test zeta MSE |
|---|---:|---:|---:|---:|
| Linear probe | 0.0886 | 0.1992 | 0.1157 | 0.2827 |
| kNN | 0.0612 | 0.1285 | 0.0121 | 0.2449 |

Use kNN `0.1285` normalized test MSE as the headline final number, and include the linear probe `0.1992` because the project requires both.
