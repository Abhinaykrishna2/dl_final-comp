# 03 CNext-U-Net Forecasting Final

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
- pool bottleneck maps with average pooling after validation-based pooling selection

Final downstream result:

| Method | Valid normalized mean MSE | Test normalized mean MSE | Test alpha MSE | Test zeta MSE |
|---|---:|---:|---:|---:|
| Linear probe | 0.0593 | 0.1043 | 0.0371 | 0.1715 |
| kNN | 0.0240 | 0.0989 | 0.0121 | 0.1857 |

The best final downstream result is kNN regression with normalized test mean MSE `0.0989`. The frozen encoder also satisfies the required linear-probe evaluation, with normalized test mean MSE `0.1043`.
