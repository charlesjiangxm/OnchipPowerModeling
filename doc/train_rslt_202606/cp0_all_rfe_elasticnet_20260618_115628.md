# Run report — ElasticNetCV

- Config: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/configs/aq_core_lvl1/cp0/cp0_all_rfe_elasticnet.yaml`
- Output dir: `/scratch/PI/eeweiz/jjiangan/OnchipPowerModeling/output/cp0_all_rfe_elasticnet_20260618_115628`
- Algorithm: **ElasticNetCV**
- Feature selection: **rfe** (top_k=20)
- Seed: 42

## Dataset counts

| Stage | train rows | val rows | test rows | features |
|---|---:|---:|---:|---:|
| loaded | 699124 | 174783 | 4378363 | 394 |
| after_preprocess | 5461 | 1365 | 34205 | 276 |
| after_feature_selection | 5461 | 1365 | 34205 | 20 |

## Metrics (in original y units)

| Split | sMAPE (%) | MAPE (%) | RMSE | MAE | R^2 |
|---|---:|---:|---:|---:|---:|
| train | 0.111 | 0.112 | 0.00001 | 0.00000 | 0.4553 |
| val | 0.098 | 0.099 | 0.00001 | 0.00000 | 0.5075 |
| test | 0.134 | 0.134 | 0.00001 | 0.00001 | -0.2211 |

## Best HPO trial

| key | value |
|---|---|
| `n_alphas` | `10` |
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

### hpo_param_importances

![hpo_param_importances](hpo/param_importances.png)

## Interaction heatmap

Not produced for **ElasticNetCV** (interaction extraction not defined or unavailable in this environment).
