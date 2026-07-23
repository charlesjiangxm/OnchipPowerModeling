# Run report — ElasticNetCV

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl2/aq_lsu_lm/aq_lsu_lm_internal_rfe_elasticnet.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/aq_lsu_lm_internal_rfe_elasticnet_20260617_124000`
- Algorithm: **ElasticNetCV**
- Feature selection: **rfe** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 9 |
| after_preprocess | 5461 | 1365 | 34205 | 8 |
| after_feature_selection | 5461 | 1365 | 34205 | 8 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.694 | 0.692 | 0.00000 | 0.00000 | 0.0000 |
| val | 0.591 | 0.591 | 0.00000 | 0.00000 | -0.0218 |
| test | 0.341 | 0.341 | 0.00000 | 0.00000 | -0.0468 |

## Best HPO trial

| key | value |
|---|---|
| `n_alphas` | `20` |
| `l1_ratio` | `[0.9]` |
| `cv` | `5` |

## Figures

### pred_vs_true_train

![pred_vs_true_train](artifacts/pred_vs_true_train.png)

### pred_vs_true_test

![pred_vs_true_test](artifacts/pred_vs_true_test.png)

### top_features

![top_features](artifacts/top_features.png)

### hpo_optimization_history

![hpo_optimization_history](hpo/optimization_history.png)

## Interaction heatmap

Not produced for **ElasticNetCV** (interaction extraction not defined or unavailable in this environment).
