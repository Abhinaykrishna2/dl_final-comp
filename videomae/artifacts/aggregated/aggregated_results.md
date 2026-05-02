# Aggregated Results

| Encoder | Params | Linear MSE | Linear alpha | Linear zeta | kNN MSE | kNN alpha | kNN zeta | EffRank (valid) | Epps-Pulley (valid) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| supervised | 7.72M | 0.1055 | 0.0333 | 0.1777 | 0.1230 | 0.0364 | 0.2096 | 3.6048 | 0.0864 |
| supervised_s43 | 7.72M | 0.0984 | 0.0332 | 0.1637 | 0.0776 | 0.0075 | 0.1476 | 3.3242 | 0.0882 |
| videomae_main | 8.07M | 0.3311 | 0.1168 | 0.5455 | 0.6098 | 0.1832 | 1.0365 | 7.6519 | 0.0424 |
| videomae_main_s43 | 8.07M | 0.2824 | 0.1117 | 0.4531 | 0.5271 | 0.1485 | 0.9058 | 8.3506 | 0.0424 |
| videomae_mask75 | 8.07M | 0.2987 | 0.0749 | 0.5224 | 0.6717 | 0.2970 | 1.0463 | 8.8709 | 0.0389 |
| videomae_mask90 | 8.07M | 0.2532 | 0.1113 | 0.3952 | 0.4647 | 0.2001 | 0.7293 | 7.7536 | 0.0420 |

All MSE values are on the test split, computed against z-scored (alpha, zeta) targets.
Effective rank and Epps-Pulley distance are computed on validation embeddings.